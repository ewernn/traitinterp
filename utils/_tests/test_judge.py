"""Tests for utils/judge.py + utils/judge_backends.py.

Structure:
    Offline unit tests         — no network, always run.
    Integration tests          — hit real APIs; skipped without env keys.
    Cross-backend consistency  — sanity-checks that both backends agree
                                 on sign of rating even with different
                                 scoring mechanisms.

Run:
    pytest utils/_tests/test_judge.py                  # all (skips integration if keys missing)
    pytest utils/_tests/test_judge.py -m 'not integration'  # offline only
    pytest utils/_tests/test_judge.py -m integration   # only live-API tests
"""

import asyncio
import os
from typing import Dict

import pytest

from utils.judge import (
    DEFAULT_JUDGE_MODEL,
    STEERING_SYSTEM,
    STEERING_USER,
    COHERENCE_PROMPT,
    NATURALNESS_PROMPT,
    SCENARIO_PROMPT,
    RESPONSE_PROMPT,
    RESPONSE_PROMPT_FOCUSED,
    RELEVANCE_PROMPT,
    TRAIT_TOKENS_SYSTEM,
    TRAIT_TOKENS_USER,
    TraitJudge,
    _gather_bounded,
    aggregate_logprob_score,
    _resolve_provider,
)
from utils.judge_backends import (
    AnthropicBackend,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    JudgeBackend,
    OpenAIBackend,
    aggregate_sampled_integers,
    parse_integer_response,
    _is_retriable,
    _retry_async,
    openrouter_backend,
    vllm_backend,
)
from utils.judge_calibration import CalibrationMap, fit_isotonic


def _run(coro):
    """Run a coroutine in a fresh event loop (cleaner than pytest-asyncio)."""
    return asyncio.run(coro)


# =============================================================================
# Unit: aggregate_logprob_score
# =============================================================================


def test_aggregate_logprob_score_weighted_avg():
    # 80% on "80", 20% on "20" → 80*0.8 + 20*0.2 = 68
    logprobs = {"80": 0.8, "20": 0.2}
    assert aggregate_logprob_score(logprobs) == pytest.approx(68.0)


def test_aggregate_logprob_score_ignores_non_integers():
    logprobs = {"80": 0.5, "hello": 0.3, "!": 0.2}
    # Only 80 counts; weight 0.5 ≥ default min_weight 0.25
    assert aggregate_logprob_score(logprobs) == pytest.approx(80.0)


def test_aggregate_logprob_score_min_weight_gate():
    # Only 0.1 mass on valid integers — below default 0.25 gate → None
    logprobs = {"50": 0.1, "hello": 0.9}
    assert aggregate_logprob_score(logprobs) is None
    # Lower the gate and it parses
    assert aggregate_logprob_score(logprobs, min_weight=0.05) == pytest.approx(50.0)


def test_aggregate_logprob_score_respects_range():
    # "500" is outside default [0, 100] and should be ignored
    logprobs = {"500": 0.9, "50": 0.1}
    # Only "50" at weight 0.1 survives → below default 0.25 gate → None
    assert aggregate_logprob_score(logprobs) is None
    # Expand the range and 500 is now valid
    assert aggregate_logprob_score(logprobs, min_val=0, max_val=1000) == pytest.approx(
        (500 * 0.9 + 50 * 0.1) / (0.9 + 0.1)
    )


def test_aggregate_logprob_score_handles_whitespace_tokens():
    logprobs = {" 80": 0.6, "20": 0.4}
    assert aggregate_logprob_score(logprobs) == pytest.approx(80 * 0.6 + 20 * 0.4)


def test_aggregate_logprob_score_empty():
    assert aggregate_logprob_score({}) is None


# =============================================================================
# Unit: parse_integer_response
# =============================================================================


def test_parse_integer_basic():
    assert parse_integer_response("7") == 7
    assert parse_integer_response(" 7") == 7
    assert parse_integer_response("7.") == 7
    assert parse_integer_response("Score: 7") == 7
    assert parse_integer_response("Answer: 5\n") == 5


def test_parse_integer_out_of_range():
    # "100" is filtered (out-of-range). "1-7 scale" tokenizes as "1" and "7"
    # (hyphens are range markers, not signs). In-range ints: 5, 1, 7. Last → 7.
    assert parse_integer_response("The value 100 becomes 5 on a 1-7 scale", min_val=1, max_val=7) == 7


def test_parse_integer_prefers_last_in_range():
    """Regression: 'On the 1-7 scale, 6' must return 6 (final answer),
    not 1 (first digit in the preamble '1-7').
    """
    assert parse_integer_response("On the 1-7 scale, 6", min_val=1, max_val=7) == 6
    assert parse_integer_response("Rating: 1 through 7, I'll say 5", min_val=1, max_val=7) == 5
    # Model says "The emotion scores 3 out of 7" — last in-range int is 7
    # (both 3 and 7 are valid; we take the LAST per documented policy).
    assert parse_integer_response("The emotion scores 3 out of 7", min_val=1, max_val=7) == 7


def test_parse_integer_all_out_of_range():
    assert parse_integer_response("100 200 999", min_val=1, max_val=7) is None


def test_parse_integer_no_digits():
    assert parse_integer_response("no digits here") is None
    assert parse_integer_response("") is None
    assert parse_integer_response(None) is None


def test_parse_integer_positive_only():
    """Judge outputs never emit negative scores; the parser deliberately treats
    hyphens as range markers (1-7) rather than signs (-5)."""
    # "-5" parses as 5, not -5. The hyphen is ignored.
    assert parse_integer_response("-5", min_val=0, max_val=10) == 5
    # Range-marker case still correct
    assert parse_integer_response("On 1-7 scale, 4", min_val=1, max_val=7) == 4


# =============================================================================
# Unit: aggregate_sampled_integers
# =============================================================================


def test_aggregate_sampled_integers_mean():
    assert aggregate_sampled_integers([5, 6, 7]) == pytest.approx(6.0)


def test_aggregate_sampled_integers_skips_none():
    assert aggregate_sampled_integers([5, None, 7]) == pytest.approx(6.0)


def test_aggregate_sampled_integers_all_none():
    assert aggregate_sampled_integers([None, None]) is None


def test_aggregate_sampled_integers_insufficient_valid():
    assert aggregate_sampled_integers([5], min_samples=2) is None


# =============================================================================
# Unit: _is_retriable
# =============================================================================


def test_is_retriable_network_errors():
    assert _is_retriable(TimeoutError())
    assert _is_retriable(asyncio.TimeoutError())
    assert _is_retriable(ConnectionError())


def test_is_not_retriable_programming_bugs():
    assert not _is_retriable(KeyError("x"))
    assert not _is_retriable(AttributeError("x"))
    assert not _is_retriable(TypeError("x"))
    assert not _is_retriable(ValueError("x"))


def test_is_retriable_sdk_modules():
    # Fake an exception whose module is 'openai.errors.RateLimitError'
    class _FakeOpenAIError(Exception):
        pass
    _FakeOpenAIError.__module__ = "openai.errors"
    assert _is_retriable(_FakeOpenAIError())

    class _FakeAnthropicError(Exception):
        pass
    _FakeAnthropicError.__module__ = "anthropic._client"
    assert _is_retriable(_FakeAnthropicError())

    class _FakeHttpxError(Exception):
        pass
    _FakeHttpxError.__module__ = "httpx"
    assert _is_retriable(_FakeHttpxError())


# =============================================================================
# Unit: _retry_async
# =============================================================================


def test_retry_async_programming_bugs_bubble_up():
    """Non-retriable exceptions should raise immediately, not be retried."""
    calls = [0]

    async def flaky():
        calls[0] += 1
        raise KeyError("oops")

    with pytest.raises(KeyError):
        _run(_retry_async(flaky, max_retries=3, base_delay=0.001))
    assert calls[0] == 1  # only one attempt


def test_retry_async_retries_transient():
    """Retriable exceptions should be retried up to max_retries."""
    calls = [0]

    async def eventually_ok():
        calls[0] += 1
        if calls[0] < 3:
            raise ConnectionError("transient")
        return "ok"

    result = _run(_retry_async(eventually_ok, max_retries=3, base_delay=0.001))
    assert result == "ok"
    assert calls[0] == 3


# =============================================================================
# Unit: _gather_bounded
# =============================================================================


def test_gather_bounded_basic():
    async def factory(x):
        return x * 2

    result = _run(_gather_bounded(factory, [1, 2, 3], max_concurrent=2))
    assert result == [2, 4, 6]


def test_gather_bounded_preserves_order():
    async def factory(x):
        # Sleep inversely to index so order tests aren't trivially correct
        await asyncio.sleep(0.001 * (5 - x))
        return x

    result = _run(_gather_bounded(factory, list(range(5)), max_concurrent=5))
    assert result == [0, 1, 2, 3, 4]


# =============================================================================
# Unit: provider dispatch + env-var precedence
# =============================================================================


def test_resolve_provider_constructor_arg_wins(monkeypatch):
    monkeypatch.setenv("TRAIT_JUDGE_PROVIDER", "anthropic")
    assert _resolve_provider("openai") == "openai"


def test_resolve_provider_env_var(monkeypatch):
    monkeypatch.setenv("TRAIT_JUDGE_PROVIDER", "anthropic")
    assert _resolve_provider(None) == "anthropic"


def test_resolve_provider_default(monkeypatch):
    monkeypatch.delenv("TRAIT_JUDGE_PROVIDER", raising=False)
    assert _resolve_provider(None) == "openai"


def test_resolve_provider_case_insensitive(monkeypatch):
    monkeypatch.setenv("TRAIT_JUDGE_PROVIDER", "OpenAI")
    assert _resolve_provider(None) == "openai"
    assert _resolve_provider("ANTHROPIC") == "anthropic"


def test_invalid_provider_raises(openai_key):
    """Unknown provider string should raise a helpful ValueError."""
    with pytest.raises(ValueError, match="Unknown judge provider"):
        TraitJudge(provider="bogus")


# =============================================================================
# Unit: module-level public API (backward-compat)
# =============================================================================


def test_default_judge_model_unchanged():
    """Old callers read DEFAULT_JUDGE_MODEL for provenance. Don't break them."""
    assert DEFAULT_JUDGE_MODEL == "gpt-4.1-mini"


def test_prompt_constants_loadable():
    """visualization/serve.py re-exports these — must exist as non-empty strings."""
    for p in [
        STEERING_SYSTEM, STEERING_USER, COHERENCE_PROMPT, NATURALNESS_PROMPT,
        SCENARIO_PROMPT, RESPONSE_PROMPT, RESPONSE_PROMPT_FOCUSED,
        RELEVANCE_PROMPT, TRAIT_TOKENS_SYSTEM, TRAIT_TOKENS_USER,
    ]:
        assert isinstance(p, str) and p.strip(), f"prompt is empty or not a string"


def test_openai_backend_supports_logprobs():
    assert OpenAIBackend.supports_logprobs(None) is True  # method doesn't read self


def test_anthropic_backend_does_not_support_logprobs():
    assert AnthropicBackend.supports_logprobs(None) is False


def test_default_models():
    assert DEFAULT_OPENAI_MODEL == "gpt-4.1-mini"
    assert DEFAULT_ANTHROPIC_MODEL == "claude-sonnet-4-5"


# =============================================================================
# Unit: AnthropicBackend message splitting
# =============================================================================


def test_anthropic_split_system_user_basic():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    system, chat = AnthropicBackend._split_system_user(msgs)
    assert system == "sys"
    assert chat == [{"role": "user", "content": "hi"}]


def test_anthropic_split_system_user_multiple_system_blocks():
    """Multiple system messages → joined with double newline."""
    msgs = [
        {"role": "system", "content": "a"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "b"},
    ]
    system, chat = AnthropicBackend._split_system_user(msgs)
    assert system == "a\n\nb"
    # system messages are stripped; chat preserves non-system order
    assert chat == [{"role": "user", "content": "hi"}]


def test_anthropic_split_system_user_no_system():
    msgs = [{"role": "user", "content": "hi"}]
    system, chat = AnthropicBackend._split_system_user(msgs)
    assert system is None
    assert chat == [{"role": "user", "content": "hi"}]


# =============================================================================
# Unit: factory helpers
# =============================================================================


def test_openrouter_backend_uses_correct_base_url(openai_key, monkeypatch):
    """openrouter_backend should point the OpenAI SDK at OpenRouter."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    backend = openrouter_backend(model="anthropic/claude-sonnet-4.5")
    assert backend.base_url == "https://openrouter.ai/api/v1"
    assert backend.model == "anthropic/claude-sonnet-4.5"


def test_vllm_backend_uses_given_base_url():
    backend = vllm_backend(
        base_url="http://localhost:8000/v1",
        model="meta-llama/Llama-3.3-70B-Instruct",
    )
    assert backend.base_url == "http://localhost:8000/v1"
    assert backend.model == "meta-llama/Llama-3.3-70B-Instruct"


# =============================================================================
# Unit: CalibrationMap
# =============================================================================


def _make_linear_map() -> CalibrationMap:
    """Identity-ish map over coherence range, for testing apply() behavior."""
    return CalibrationMap(
        source_identifier="anthropic/claude-sonnet-4-5",
        target_identifier="openai/gpt-4.1-mini",
        task="coherence",
        source_points=[0.0, 50.0, 100.0],
        target_points=[10.0, 60.0, 95.0],
    )


def test_calibration_map_interpolates_midpoints():
    m = _make_linear_map()
    # Midpoint between 0 and 50 → interpolated between 10 and 60 = 35
    assert m.apply(25.0) == pytest.approx(35.0)
    # Midpoint between 50 and 100 → between 60 and 95 = 77.5
    assert m.apply(75.0) == pytest.approx(77.5)


def test_calibration_map_clamps_extrapolation():
    m = _make_linear_map()
    # Below source range clamps to first target point
    assert m.apply(-10.0) == 10.0
    # Above source range clamps to last target point
    assert m.apply(200.0) == 95.0


def test_calibration_map_handles_none():
    m = _make_linear_map()
    assert m.apply(None) is None


def test_calibration_map_endpoints():
    m = _make_linear_map()
    assert m.apply(0.0) == 10.0
    assert m.apply(50.0) == 60.0
    assert m.apply(100.0) == 95.0


def test_calibration_map_sorts_on_construct():
    """Unsorted input should get sorted by source automatically."""
    m = CalibrationMap(
        source_identifier="a", target_identifier="b", task="coherence",
        source_points=[100.0, 0.0, 50.0],
        target_points=[95.0, 10.0, 60.0],
    )
    assert m.source_points == [0.0, 50.0, 100.0]
    assert m.target_points == [10.0, 60.0, 95.0]


def test_calibration_map_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="equal length"):
        CalibrationMap(
            source_identifier="a", target_identifier="b", task="coherence",
            source_points=[1.0, 2.0, 3.0], target_points=[1.0, 2.0],
        )


def test_calibration_map_too_few_points_raises():
    with pytest.raises(ValueError, match="at least 2 points"):
        CalibrationMap(
            source_identifier="a", target_identifier="b", task="coherence",
            source_points=[50.0], target_points=[60.0],
        )


def test_calibration_map_roundtrip(tmp_path):
    """Serialize to JSON and reload; behavior preserved."""
    m = _make_linear_map()
    path = tmp_path / "map.json"
    import json
    path.write_text(json.dumps(m.to_dict(extra={"fitted_at": "test"})))
    m2 = CalibrationMap.load(path)
    for s in [0.0, 25.0, 50.0, 75.0, 100.0]:
        assert m.apply(s) == pytest.approx(m2.apply(s))


def test_fit_isotonic_monotonic_output():
    """Isotonic fit produces a monotonic (non-decreasing) map."""
    # Noisy but underlying monotonic relationship
    src = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tgt = [10, 22, 15, 35, 48, 55, 60, 78, 72, 95]  # noisy but trending
    m = fit_isotonic(
        src, tgt,
        source_identifier="a", target_identifier="b", task="coherence",
    )
    # Output should be non-decreasing
    for i in range(1, len(m.target_points)):
        assert m.target_points[i] >= m.target_points[i - 1], \
            f"non-monotonic at index {i}: {m.target_points}"


def test_fit_isotonic_needs_minimum_pairs():
    with pytest.raises(ValueError, match="at least 5"):
        fit_isotonic([1.0, 2.0], [10.0, 20.0], "a", "b", "coherence")


def test_anthropic_backend_accepts_calibration_map_arg():
    """Constructor accepts calibration_map= kwarg without ANTHROPIC_API_KEY
    load (we test load path with a real path separately in integration)."""
    import inspect
    sig = inspect.signature(AnthropicBackend.__init__)
    assert "calibration_map" in sig.parameters


def test_calibration_map_applies_in_anthropic_score_prompt(tmp_path, monkeypatch):
    """AnthropicBackend with calibration_map should reshape raw scores."""
    # Build and save a test map
    m = _make_linear_map()
    path = tmp_path / "test_map.json"
    import json
    path.write_text(json.dumps(m.to_dict()))

    # Skip if ANTHROPIC_API_KEY is missing — we only test wiring, not API
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    backend = AnthropicBackend(calibration_map=str(path))
    assert backend.calibration_map is not None
    assert backend.calibration_map.apply(50.0) == 60.0
    assert backend.calibration_map.apply(0.0) == 10.0


# =============================================================================
# Integration: OpenAI
# =============================================================================


@pytest.mark.integration
def test_openai_score_coherence_returns_sane_value(openai_key, coherent_short_text):
    """A coherent sentence should score above 70."""
    async def body():
        judge = TraitJudge()
        try:
            return await judge.score_coherence(coherent_short_text)
        finally:
            await judge.close()
    score = asyncio.run(body())
    assert score is not None
    assert 70 <= score <= 100, f"Expected 70-100, got {score}"


@pytest.mark.integration
def test_openai_score_on_scale_1_to_7(openai_key, rate_on_1_7_messages):
    """1-7 scale: 'happy' should be ≥ 5 (solidly positive valence)."""
    async def body():
        judge = TraitJudge()
        try:
            return await judge.score_on_scale(
                rate_on_1_7_messages, min_val=1, max_val=7,
            )
        finally:
            await judge.close()
    score = asyncio.run(body())
    assert score is not None
    assert 5 <= score <= 7


@pytest.mark.integration
def test_openai_identifier(openai_key):
    async def body():
        judge = TraitJudge()
        try:
            return judge.identifier(), judge.model
        finally:
            await judge.close()
    identifier, model = asyncio.run(body())
    assert identifier == "openai/gpt-4.1-mini"
    assert model == "gpt-4.1-mini"


# =============================================================================
# Integration: Anthropic
# =============================================================================


@pytest.mark.integration
def test_anthropic_score_on_scale_1_to_7(anthropic_key, rate_on_1_7_messages):
    """1-7 scale: 'happy' should be ≥ 5."""
    async def body():
        judge = TraitJudge(provider="anthropic")
        try:
            return await judge.score_on_scale(
                rate_on_1_7_messages, min_val=1, max_val=7,
            )
        finally:
            await judge.close()
    score = asyncio.run(body())
    assert score is not None
    assert 5 <= score <= 7


@pytest.mark.integration
def test_anthropic_usage_populated(anthropic_key, rate_on_1_7_messages):
    """After one call, _last_usage should contain token counts."""
    async def body():
        judge = TraitJudge(provider="anthropic")
        try:
            await judge.score_on_scale(
                rate_on_1_7_messages, min_val=1, max_val=7,
            )
            assert isinstance(judge.backend, AnthropicBackend)
            return judge.backend._last_usage
        finally:
            await judge.close()
    usage = asyncio.run(body())
    assert usage is not None
    assert usage["input_tokens"] and usage["input_tokens"] > 0
    # Cache cold on first call — read=0.
    assert usage["cache_read_input_tokens"] == 0


@pytest.mark.integration
def test_anthropic_identifier(anthropic_key):
    async def body():
        judge = TraitJudge(provider="anthropic")
        try:
            return judge.identifier(), judge.model
        finally:
            await judge.close()
    identifier, model = asyncio.run(body())
    assert identifier == "anthropic/claude-sonnet-4-5"
    assert model == "claude-sonnet-4-5"


@pytest.mark.integration
def test_anthropic_empty_messages_raises(anthropic_key):
    """If only system messages are passed, AnthropicBackend must raise
    rather than sending an empty user turn (which 400s)."""
    async def body():
        judge = TraitJudge(provider="anthropic")
        try:
            with pytest.raises(ValueError, match="non-system message"):
                await judge._score_messages(
                    [{"role": "system", "content": "instructions"}],
                    min_val=1, max_val=7,
                )
        finally:
            await judge.close()
    asyncio.run(body())


# =============================================================================
# Integration: cross-backend consistency
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_cross_backend_happy_valence_agreement(any_judge_key, rate_on_1_7_messages):
    """Both backends should agree that 'happy' is solidly high-valence.

    We don't assert tight numerical agreement — logprob-weighted and
    sampled-integer means are scored differently. Wide sanity band only.
    """
    async def body():
        oai_judge = TraitJudge()
        anth_judge = TraitJudge(provider="anthropic")
        try:
            oai_score = await oai_judge.score_on_scale(
                rate_on_1_7_messages, min_val=1, max_val=7,
            )
            anth_score = await anth_judge.score_on_scale(
                rate_on_1_7_messages, min_val=1, max_val=7,
            )
            return oai_score, anth_score
        finally:
            await oai_judge.close()
            await anth_judge.close()
    oai_score, anth_score = asyncio.run(body())
    assert oai_score is not None and anth_score is not None
    assert 5 <= oai_score <= 7, f"openai: {oai_score}"
    assert 5 <= anth_score <= 7, f"anthropic: {anth_score}"
