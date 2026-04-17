"""
LLM judge backends — multi-provider.

Every backend implements `JudgeBackend`. TraitJudge (in utils/judge.py)
dispatches to one backend instance for its lifetime.

Backends:
    OpenAIBackend    — OpenAI Chat Completions (logprob-weighted scoring).
                       Also covers OpenRouter + local OpenAI-compat servers
                       via `base_url=`; pass `require_logprobs=False` to
                       tolerate endpoints that silently drop logprobs.
    AnthropicBackend — Anthropic Messages API (sampling-based + prompt caching).

Why two scoring mechanisms?

OpenAI exposes top-k logprobs → one API call yields a calibrated weighted
average over integer tokens (aggregate_logprob_score). Anthropic (as of
2026-04) does not expose logprobs at all, so we sample N times with
non-zero temperature and average parsed integer responses
(aggregate_sampled_integers). The two mechanisms produce floats on the
same scale but with different variance characteristics — see docstrings
and the module-level CALIBRATION note in utils/judge.py.

Convenience factories (bottom of file):
    openrouter_backend(model, api_key=None) -> OpenAIBackend
    vllm_backend(base_url, model) -> OpenAIBackend

⚠️  Cross-backend calibration caveat — see AnthropicBackend docstring.
    Thresholds in core.kwargs_configs (MIN_COHERENCE=77, POS/NEG_THRESHOLD=60/40)
    were tuned for OpenAI logprob-weighted scoring. Using a non-logprob backend
    (Anthropic, logprob-less OpenAI-compat) for steering eval or response
    vetting without a calibration map can silently drift pass rates. Calibration
    infrastructure is deferred to a future PR; until then, prefer OpenAI for
    threshold-gated pipelines (steering_eval, preextraction_vetting).
"""

from __future__ import annotations

import asyncio
import math
import os
import re
from typing import Dict, List, Optional, Protocol, Sequence, TypedDict


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"  # matches Sofroniew et al. 2026

# Anthropic cache minimum prefix — model-dependent. Below this, cache_control
# is silently ignored. Sonnet 4.5 is the most cache-friendly of current
# frontier models (1024 tokens); 4.6 requires 2048; 4.7/Haiku-4.5 require 4096.
_ANTHROPIC_CACHE_MIN_TOKENS = {
    "claude-sonnet-4-5": 1024,
    "claude-sonnet-4-6": 2048,
    "claude-opus-4-7": 4096,
    "claude-haiku-4-5": 4096,
}

# Anthropic sampling defaults — temperature must be >0 for sample diversity.
_ANTHROPIC_SAMPLE_TEMPERATURE = 0.7


# =============================================================================
# Types
# =============================================================================


class Message(TypedDict):
    """Chat message (OpenAI-shape; AnthropicBackend converts at call site)."""
    role: str       # "system" | "user" | "assistant"
    content: str


# =============================================================================
# Aggregation helpers
# =============================================================================


def aggregate_logprob_score(
    logprobs: Dict[str, float],
    min_weight: float = 0.25,
    min_val: int = 0,
    max_val: int = 100,
) -> Optional[float]:
    """Weighted average over next-token logprobs that parse as valid integers.

    Args:
        logprobs: Dict mapping token strings to probabilities.
        min_weight: Minimum total probability mass on valid integers to return.
        min_val, max_val: Inclusive integer range.

    Returns:
        Float in [min_val, max_val] or None if not enough valid probability mass.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for token, prob in logprobs.items():
        try:
            value = int(token.strip())
        except (ValueError, AttributeError):
            continue
        if min_val <= value <= max_val:
            weighted_sum += value * prob
            total_weight += prob
    if total_weight < min_weight:
        return None
    return weighted_sum / total_weight


# Positive integers only. Judges always emit non-negative scores; parsing "-N"
# as a negative would mis-handle phrases like "1-7 scale" where the hyphen is
# a range marker, not a sign. If negative-score use cases arise, add a
# parser-level switch rather than flipping the default.
_INTEGER_RE = re.compile(r"\d+")


def parse_integer_response(
    text: str,
    min_val: int = 0,
    max_val: int = 100,
) -> Optional[int]:
    """Extract the model's intended integer answer from `text`.

    Strategy: prefer the LAST in-range integer. Models tend to preamble
    before their answer ("On the 1-7 scale, 6") — taking the first match
    would return 1 from "1-7", not the actual answer 6. Taking the last
    match is more robust across response styles.

    Accepts common shapes: "7", " 7", "7.", "Score: 7", "On 1-7: 6".
    Returns None if no valid integer found in range.
    """
    if not text:
        return None
    last_valid: Optional[int] = None
    for match in _INTEGER_RE.finditer(text):
        try:
            value = int(match.group(0))
        except ValueError:
            continue
        if min_val <= value <= max_val:
            last_valid = value
    return last_valid


def aggregate_sampled_integers(
    samples: Sequence[Optional[int]],
    min_samples: int = 1,
) -> Optional[float]:
    """Average valid integer samples, or None if too few valid."""
    valid = [x for x in samples if x is not None]
    if len(valid) < min_samples:
        return None
    return sum(valid) / len(valid)


# =============================================================================
# Retry helper
# =============================================================================


def _is_retriable(exc: BaseException) -> bool:
    """Transient API errors worth retrying. Programming bugs (AttributeError,
    TypeError, etc.) bubble up unchanged so you notice them.

    Heuristic: retry on anything from an SDK or stdlib network/HTTP module,
    plus explicit TimeoutError. Don't retry on standard Python exceptions
    (KeyError, ValueError, AttributeError, etc.).
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    module = type(exc).__module__ or ""
    # openai.* / anthropic.* / httpx.* / http.*
    return module.startswith(("openai", "anthropic", "httpx", "http"))


async def _retry_async(fn, *args, max_retries: int = 3, base_delay: float = 0.5, **kwargs):
    """Exponential-backoff retry for a coroutine-returning function.

    Only retries transient API errors (HTTP/SDK exceptions, timeouts).
    Programming bugs bubble up on the first attempt.

    Overload policy: Anthropic 529 / OpenAI RateLimit events can last minutes,
    not seconds. We detect via exception class name and extend both the base
    delay (30s) and retry count (up to 5 attempts) for those specifically.
    Actual wait math at max budget: 30 + 60 + 120 + 240 = 450s ≈ 7.5 minutes
    (4 sleeps between the 5 attempts) before giving up.
    """
    last_exc: Optional[BaseException] = None
    # Track whether we've seen an overload so we can extend attempts budget.
    attempt_budget = max_retries
    attempt = 0
    while attempt < attempt_budget:
        try:
            return await fn(*args, **kwargs)
        except BaseException as e:
            if not _is_retriable(e):
                raise
            last_exc = e
            name = type(e).__name__
            is_overload = "Overload" in name or "RateLimit" in name
            if is_overload:
                # Upgrade: give overload more attempts and longer backoff base.
                attempt_budget = max(attempt_budget, 5)
                effective_base = 30.0  # 30s base → 30, 60, 120, 240, 480s = ~15min budget
            else:
                effective_base = base_delay

            attempt += 1
            if attempt < attempt_budget:
                await asyncio.sleep(effective_base * (2 ** (attempt - 1)))
    assert last_exc is not None
    raise last_exc


# =============================================================================
# JudgeBackend protocol
# =============================================================================


class JudgeBackend(Protocol):
    """What TraitJudge calls into.

    Each backend encapsulates its SDK client and exposes a uniform scoring
    surface. `supports_logprobs()` lets callers reason about calibration.
    """

    model: str

    def supports_logprobs(self) -> bool:
        """True → score_prompt uses top-k logprobs. False → samples N times."""
        ...

    async def score_prompt(
        self,
        messages: List[Message],
        *,
        min_val: int = 0,
        max_val: int = 100,
        min_weight: float = 0.25,
        n_samples: int = 1,
    ) -> Optional[float]:
        """Score a prompt in [min_val, max_val]. Returns None if unscorable.

        For logprob backends, n_samples is ignored.
        For sampling backends, fires n_samples parallel calls and averages.
        """
        ...

    async def classify(
        self,
        messages: List[Message],
        letters: List[str],
    ) -> Dict[str, float]:
        """Classification probabilities over single-letter choices.

        For logprob backends: calibrated probabilities from top-20 logprobs.
        For sampling backends: one-hot {winner: 1.0, others: 0.0} or {} on
        parse failure.
        """
        ...

    async def check_engagement(self, messages: List[Message]) -> str:
        """Short text response for ENGAGES/OFF_TOPIC relevance checks.

        Returns stripped + uppercased text. Empty string on error.
        """
        ...

    async def close(self) -> None:
        """Release SDK client resources."""
        ...


# =============================================================================
# OpenAIBackend — covers OpenAI, OpenRouter, local OpenAI-compat servers
# =============================================================================


class OpenAIBackend:
    """Backend for OpenAI Chat Completions API and OpenAI-compatible endpoints.

    With default args, talks to OpenAI. With `base_url=` set, talks to any
    OpenAI-compatible endpoint (OpenRouter, vLLM, llama.cpp --openai-compat,
    LM Studio, SGLang, etc.).

    Compat caveat: some endpoints silently drop `logprobs=True` and return an
    empty `logprobs.content`. We validate this in `_get_logprobs` and log a
    warning the first time per instance. Set `require_logprobs=False` to
    suppress the warning (scores will just degrade to None).
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        seed: int = 42,
        require_logprobs: bool = True,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("OpenAI SDK not installed: pip install openai") from e

        # .env loading is the codebase convention; harmless if already loaded.
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        if api_key is None:
            is_openrouter = base_url is not None and "openrouter" in base_url
            env_var = "OPENROUTER_API_KEY" if is_openrouter else "OPENAI_API_KEY"
            api_key = os.environ.get(env_var)
            if not api_key:
                raise ValueError(
                    f"{env_var} env var not set; pass api_key=... to OpenAIBackend()"
                )

        client_kwargs: Dict = {"api_key": api_key}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**client_kwargs)

        self.model = model
        self.base_url = base_url
        self.seed = seed
        self.require_logprobs = require_logprobs
        self._warned_no_logprobs = False

    def supports_logprobs(self) -> bool:
        return True

    async def _call(
        self,
        messages: List[Message],
        *,
        max_tokens: int = 1,
        temperature: float = 0,
        logprobs: bool = True,
        top_logprobs: int = 20,
    ):
        kwargs: Dict = {
            "model": self.model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": self.seed,
        }
        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs
        return await self.client.chat.completions.create(**kwargs)

    async def _get_logprobs(
        self,
        messages: List[Message],
        max_retries: int = 3,
        top_logprobs: int = 20,
    ) -> Dict[str, float]:
        """Top-k next-token logprobs, or empty dict on failure."""
        try:
            response = await _retry_async(
                self._call, messages,
                max_tokens=1, temperature=0,
                logprobs=True, top_logprobs=top_logprobs,
                max_retries=max_retries,
            )
        except Exception as e:
            print(f"OpenAI logprob call failed after {max_retries} retries: {e}")
            return {}

        lp_content = response.choices[0].logprobs.content if response.choices[0].logprobs else None
        if not lp_content:
            if self.require_logprobs and not self._warned_no_logprobs:
                print(
                    f"Warning: {self.base_url or 'OpenAI'} model '{self.model}' "
                    f"returned empty logprobs. Scoring will degrade to None. "
                    f"(This warning appears once per backend instance.)"
                )
                self._warned_no_logprobs = True
            return {}

        top = lp_content[0].top_logprobs or []
        return {lp.token: math.exp(lp.logprob) for lp in top}

    async def score_prompt(
        self,
        messages: List[Message],
        *,
        min_val: int = 0,
        max_val: int = 100,
        min_weight: float = 0.25,
        n_samples: int = 1,  # ignored
    ) -> Optional[float]:
        logprobs = await self._get_logprobs(messages)
        return aggregate_logprob_score(
            logprobs, min_weight=min_weight, min_val=min_val, max_val=max_val,
        )

    async def classify(
        self,
        messages: List[Message],
        letters: List[str],
    ) -> Dict[str, float]:
        logprobs = await self._get_logprobs(messages)
        # OpenAI sometimes emits " A" vs "A"; sum both.
        return {L: logprobs.get(L, 0.0) + logprobs.get(f" {L}", 0.0) for L in letters}

    async def check_engagement(
        self,
        messages: List[Message],
        max_tokens: int = 3,
    ) -> str:
        try:
            response = await _retry_async(
                self._call, messages,
                max_tokens=max_tokens, temperature=0, logprobs=False,
            )
            text = response.choices[0].message.content or ""
            return text.strip().upper()
        except Exception as e:
            print(f"OpenAI engagement-check call failed: {e}")
            return ""

    async def close(self) -> None:
        await self.client.close()


# =============================================================================
# AnthropicBackend — sampling-based + prompt caching
# =============================================================================


class AnthropicBackend:
    """Backend for Anthropic Messages API.

    Key differences from OpenAIBackend:
    - No logprobs (Anthropic API doesn't expose them as of 2026-04).
      Scoring uses n-sample averaging with temperature>0.
    - Prompt caching via cache_control blocks. First request with a given
      prefix pays the write premium; subsequent requests within cache_ttl
      hit the cache at ~0.1× base input cost.

    Caveat: any byte change in the cached prefix silently invalidates. Use
    the same Message objects across calls to maximize cache hits.

    ⚠️  CALIBRATION CAVEAT — READ BEFORE USING FOR STEERING EVAL ⚠️

    The hard-coded threshold constants in core.kwargs_configs (MIN_COHERENCE=77,
    POS_THRESHOLD=60, NEG_THRESHOLD=40) were empirically tuned for GPT-4.1-mini's
    LOGPROB-WEIGHTED score distribution. Anthropic returns sampled-integer
    averages with different variance and possibly different bias, so thresholds
    may pass/fail at different rates against the same text. Symptoms: silent
    drift in steering eval pass rates, scenario vetting acceptance rates, etc.

    SAFE uses right now:
      - score_on_scale() for paper-faithful valence/arousal rating (no threshold)
      - Any workflow where you control the threshold downstream

    UNSAFE until a calibration map is built:
      - steering_eval.py (uses MIN_COHERENCE=77 as a hard gate)
      - preextraction_vetting.py (uses POS/NEG_THRESHOLD as hard gates)
      - Any cross-backend comparison of pass rates

    Mitigation planned: fit a quantile-normalization map from Anthropic sampled
    scores → OpenAI logprob-weighted scores using ~100 golden (response,
    openai_score) pairs from past steering runs. See issue tracker / TODO.md.

    Usage (safe):
        backend = AnthropicBackend(model="claude-sonnet-4-5", n_samples=3)
        score = await backend.score_prompt(msgs, min_val=1, max_val=7, n_samples=3)
    """

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        api_key: Optional[str] = None,
        n_samples: int = 1,
        cache_ttl_minutes: int = 5,
        sample_temperature: float = _ANTHROPIC_SAMPLE_TEMPERATURE,
        calibration_map: Optional[str] = None,
    ):
        """Args:
            model: Anthropic model ID (e.g., "claude-sonnet-4-5").
            api_key: If None, reads ANTHROPIC_API_KEY.
            n_samples: Default number of samples per score_prompt. Callers
                can override per-call.
            cache_ttl_minutes: 5 (default, cheaper writes) or 60 (premium).
            sample_temperature: Temperature for sampling mode (default 0.7).
                Must be >0 for meaningful variance across samples.
            calibration_map: Optional path to a CalibrationMap JSON (see
                utils/judge_calibration.py). If set, raw Anthropic scores
                from score_prompt() are mapped into the OpenAI-logprob
                distribution so existing thresholds (MIN_COHERENCE=77 etc.)
                transfer without silent drift. Only fit for the specified
                task (typically "coherence") — applying a coherence map to
                trait-scoring output is still under-validated; use at own risk.
        """
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError("Anthropic SDK not installed: pip install anthropic") from e

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY env var not set; pass api_key=...")

        if cache_ttl_minutes not in (5, 60):
            raise ValueError(f"cache_ttl_minutes must be 5 or 60, got {cache_ttl_minutes}")

        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.n_samples = n_samples
        self.cache_ttl_minutes = cache_ttl_minutes
        self.sample_temperature = sample_temperature

        # Optional score-distribution calibration → OpenAI logprob equivalent.
        self.calibration_map = None
        if calibration_map is not None:
            from utils.judge_calibration import CalibrationMap
            self.calibration_map = CalibrationMap.load(calibration_map)

        # Cache observability — set by _call_once.
        # NOTE: under concurrent n_samples>1, this is the LAST returned call's
        # usage, not aggregated. For per-call visibility, instrument at the
        # call site. Good enough for "did this ever hit the cache" sanity checks.
        self._last_usage: Optional[Dict] = None

        # Per-instance flag so we warn at most once about prompts too short
        # to cache. Sonnet-4.5 needs >=1024 tokens in the cached prefix; most
        # of our trait/coherence system prompts are 300-800 tokens and will
        # silently skip caching despite cache_control being set.
        self._warned_cache_too_short = False

    def supports_logprobs(self) -> bool:
        return False

    @staticmethod
    def _split_system_user(messages: List[Message]):
        """Anthropic separates system from messages. Combine any system turns
        into a single system string; pass others as messages.
        """
        system_parts: List[str] = []
        chat_messages: List[Dict] = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(m["content"])
            else:
                chat_messages.append({"role": m["role"], "content": m["content"]})
        system = "\n\n".join(system_parts) if system_parts else None
        return system, chat_messages

    def _cache_control(self) -> Dict:
        """Return the cache_control block for this backend's TTL setting."""
        cc: Dict = {"type": "ephemeral"}
        if self.cache_ttl_minutes == 60:
            cc["ttl"] = "1h"
        return cc

    def _cache_min_tokens(self) -> int:
        """Minimum prefix length for cache_control to actually cache.

        Falls back to a conservative default for unknown models.
        """
        return _ANTHROPIC_CACHE_MIN_TOKENS.get(self.model, 2048)

    def _warn_if_system_too_short(self, system: str) -> None:
        """One-shot warning when system prompt is below the cache threshold.

        Anthropic silently ignores `cache_control` on prefixes shorter than
        the model-specific minimum (1024 for Sonnet 4.5, 2048 for 4.6, 4096
        for 4.7 / Haiku 4.5). We estimate tokens as chars/4 — conservative.
        """
        if self._warned_cache_too_short:
            return
        approx_tokens = len(system) // 4
        min_tokens = self._cache_min_tokens()
        if approx_tokens < min_tokens:
            print(
                f"Warning: Anthropic system prompt is ~{approx_tokens} tokens, "
                f"below {self.model}'s cache minimum ({min_tokens}). "
                f"cache_control is silently ignored; you'll pay uncached input "
                f"pricing. (Warning shown once per backend instance.)"
            )
            self._warned_cache_too_short = True

    def _build_request(
        self,
        messages: List[Message],
        *,
        max_tokens: int,
        temperature: float,
        cache_system: bool,
    ) -> Dict:
        """Build kwargs for client.messages.create."""
        system, chat = self._split_system_user(messages)
        if not chat:
            raise ValueError(
                "AnthropicBackend requires at least one non-system message. "
                "Got only system messages — Anthropic would reject this with 400."
            )
        kwargs: Dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": chat,
        }
        if system:
            if cache_system:
                # Wrap system as a content-block with cache_control.
                self._warn_if_system_too_short(system)
                kwargs["system"] = [
                    {"type": "text", "text": system, "cache_control": self._cache_control()}
                ]
            else:
                kwargs["system"] = system
        return kwargs

    async def _call_once(
        self,
        messages: List[Message],
        *,
        max_tokens: int = 16,
        temperature: float = 0,
        cache_system: bool = True,
        max_retries: int = 3,
    ) -> str:
        """Single call; returns response text (possibly empty on error)."""
        request = self._build_request(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            cache_system=cache_system,
        )
        try:
            response = await _retry_async(
                self.client.messages.create,
                max_retries=max_retries,
                **request,
            )
        except Exception as e:
            print(f"Anthropic call failed after {max_retries} retries: {e}")
            return ""

        # Observability: record usage for downstream cache-hit inspection.
        self._last_usage = {
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", None),
            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", None),
        }

        # response.content is a list of blocks; extract text blocks.
        text_parts: List[str] = []
        for block in response.content or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
        return "".join(text_parts)

    async def score_prompt(
        self,
        messages: List[Message],
        *,
        min_val: int = 0,
        max_val: int = 100,
        min_weight: float = 0.25,  # unused; kept for protocol conformance
        n_samples: int = 1,
    ) -> Optional[float]:
        """Sample n_samples times, parse an integer from each, average valid.

        Temperature policy:
        - n_samples == 1: temperature=0 (deterministic single-shot; reproducible
          given cache semantics).
        - n_samples > 1: temperature=self.sample_temperature (0.7 default) to
          get meaningful variance across samples for averaging.

        This means score distributions differ qualitatively between n=1 and
        n>1 — n=1 is a point estimate, n>1 is a mean of independent draws.
        Cross-backend comparisons (Anthropic vs OpenAI) should use n>1 on the
        Anthropic side for reasonable calibration.
        """
        n = max(1, n_samples)
        # max_tokens=16 is conservative for a single integer; enough headroom for "7." etc.
        temp = 0 if n == 1 else self.sample_temperature
        tasks = [
            self._call_once(
                messages, max_tokens=16, temperature=temp, cache_system=True,
            )
            for _ in range(n)
        ]
        texts = await asyncio.gather(*tasks)
        samples = [parse_integer_response(t, min_val=min_val, max_val=max_val) for t in texts]
        raw_score = aggregate_sampled_integers(samples)
        # Apply calibration if configured — maps Anthropic sampled-integer scores
        # into OpenAI-logprob-equivalent space so downstream thresholds transfer.
        if self.calibration_map is not None:
            return self.calibration_map.apply(raw_score)
        return raw_score

    async def classify(
        self,
        messages: List[Message],
        letters: List[str],
    ) -> Dict[str, float]:
        """Single-shot classification. Returns {winner: 1.0, others: 0.0}
        or all-zeros on parse failure.

        Matching strategy:
          - If stripped response is a single char (1-2 chars incl punctuation),
            take that letter directly (compliance case).
          - Otherwise, look for a word-boundary match of one of the allowed
            letters. This avoids misclassifying "I choose B" as I, or "A and
            B are wrong, so C" as A.

        Callers needing calibrated probabilities should use a logprob backend.
        """
        text = await self._call_once(
            messages, max_tokens=3, temperature=0, cache_system=True,
        )
        out = {L: 0.0 for L in letters}
        stripped = (text or "").strip().upper()
        if not stripped:
            return out

        # Case 1: model complied — response is just the letter (maybe with
        # trailing punctuation or whitespace).
        if len(stripped) <= 2:
            ch = stripped[0]
            if ch in out:
                out[ch] = 1.0
            return out

        # Case 2: model preambled. Use word-boundary matching so "I choose B"
        # doesn't match on the "I" pronoun. If multiple letters match, the
        # LAST one wins (usually closer to the model's final answer).
        winner = None
        for L in letters:
            if re.search(rf"\b{L}\b", stripped):
                winner = L
        if winner is not None:
            out[winner] = 1.0
        return out

    async def check_engagement(
        self,
        messages: List[Message],
    ) -> str:
        """Short text response for ENGAGES/OFF_TOPIC relevance checks."""
        text = await self._call_once(
            messages, max_tokens=3, temperature=0, cache_system=True,
        )
        return (text or "").strip().upper()

    async def close(self) -> None:
        """Close the underlying AsyncAnthropic HTTP client."""
        await self.client.close()


# =============================================================================
# Convenience factories
# =============================================================================


def openrouter_backend(
    model: str,
    api_key: Optional[str] = None,
    require_logprobs: bool = True,
) -> OpenAIBackend:
    """Build an OpenAIBackend pointed at OpenRouter.

    Note: OpenRouter silently drops logprobs for Anthropic Claude models
    (returns 200 OK with empty logprobs.content). Score calls will return
    None in that case. For Claude via OpenRouter, prefer AnthropicBackend.
    """
    return OpenAIBackend(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        require_logprobs=require_logprobs,
    )


def vllm_backend(
    base_url: str,
    model: str,
    api_key: str = "EMPTY",  # vLLM requires a non-empty string but ignores it
    require_logprobs: bool = True,
) -> OpenAIBackend:
    """Build an OpenAIBackend pointed at a local vLLM OpenAI-compatible server.

    Launch: `vllm serve {model} --port 8000`
    Then:   `backend = vllm_backend("http://localhost:8000/v1", model)`
    """
    return OpenAIBackend(
        model=model,
        api_key=api_key,
        base_url=base_url,
        require_logprobs=require_logprobs,
    )
