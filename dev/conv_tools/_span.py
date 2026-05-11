"""Annotation span <-> response token range resolution (dev/conv_tools variant).

Single canonical implementation for THIS subsystem. NFKC-normalized,
containment-rule. Operates on PRE-DECODED token strings (the `tokens` field
in our response JSON files), not token IDs.

The repo-wide canonical equivalent is `utils.annotations.char_range_to_token_range`
which uses the correct containment rule but requires a HuggingFace tokenizer
to decode IDs. Our use case (annotation files store decoded `tokens` strings,
no tokenizer at eval time, NFKC-affected spans, graceful failure) needs a
different shape, hence this shim.

Replaces two divergent in-tree implementations now removed:
- `bias_correlation_sweep.span_to_token_range` (cursor=0, NFKC, start-rule bug)
- `holdout_two_channel.span_to_token_range` (prompt_end-based, no NFKC, start-rule bug)

The bug in both: `if s is None and cum >= pos: s = i` selects the first token
whose CUMULATIVE-OFFSET-AFTER reaches `pos`, which is the token AFTER the one
containing `pos` when the span begins mid-token (~35% of evaluations). The fix:
select the token i such that `cum_before[i] <= pos < cum_after[i]` — i.e., the
token whose character range contains `pos`.

Input:  response_text (str), span_text (str), response_tokens (list[str])
Output: (start_tok, end_tok) half-open interval into the response-only token
        slice (i.e., already past prompt_end), or None if span not found.

Usage:
    from _span import span_to_token_range, instances_to_token_ranges
    rng = span_to_token_range(resp_text, span_text, tokens, prompt_end=80)
    ranges = instances_to_token_ranges(resp_text, response_tokens, instances)
"""
from __future__ import annotations
import unicodedata
from typing import Optional


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s) if s else s


def span_to_token_range(
    response_text: str,
    span_text: str,
    response_tokens: list[str],
    prompt_end: int = 0,
    cursor: int = 0,
) -> Optional[tuple[int, int]]:
    """Resolve a span text -> (start_tok, end_tok) into RESPONSE-ONLY tokens.

    `response_tokens` may be the FULL token list (prompt + response); pass
    prompt_end to slice off the prompt. Returned indices are into the slice
    `response_tokens[prompt_end:]`.

    `cursor` is the character offset in response_text where the search starts
    (used by `instances_to_token_ranges` to walk forward through multiple
    instances of the same span text).

    Containment rule: token i is the start iff `cum_before[i] <= pos < cum_after[i]`.
    """
    rt = _norm(response_text)
    sp = _norm(span_text)
    pos = rt.find(sp, cursor)
    if pos < 0:
        # Fall back to non-normalized search (NFKC may have munged characters)
        pos = response_text.find(span_text, cursor)
        if pos < 0:
            return None
        rt, sp = response_text, span_text
    end_char = pos + len(sp)

    response_only = response_tokens[prompt_end:]
    cum = 0
    s: Optional[int] = None
    e: Optional[int] = None
    for i, t in enumerate(response_only):
        nt = _norm(t) if t else t
        nlen = len(nt) if nt else 0
        cum_after = cum + nlen
        # Containment: start token is the one whose char interval contains pos.
        if s is None and cum_after > pos:
            s = i
        # End token: first token whose cum_after reaches end_char (half-open end_tok = i+1).
        if e is None and cum_after >= end_char:
            e = i + 1
            break
        cum = cum_after
    if s is None:
        return None
    if e is None:
        e = len(response_only)
    if e <= s:
        e = s + 1
    return (s, e)


def instances_to_token_ranges(
    response_text: str,
    response_tokens: list[str],
    instances: list[dict],
    prompt_end: int = 0,
) -> list[tuple[int, int]]:
    """Resolve all instances[*]['span'] sequentially with cursor walk.

    Returns the list of (start_tok, end_tok) ranges that resolved successfully,
    skipping any whose span text wasn't found. Cursor advances after each match
    to avoid duplicate hits when the same text appears multiple times.
    """
    ranges = []
    cursor = 0
    for inst in instances:
        span_text = inst.get("span", "")
        if not span_text:
            continue
        rng = span_to_token_range(
            response_text, span_text, response_tokens,
            prompt_end=prompt_end, cursor=cursor,
        )
        if rng is None:
            continue
        ranges.append(rng)
        # Advance cursor past this match so the next find() doesn't re-find it.
        # Use the character position right after this span in the (possibly
        # NFKC-normalized) response text.
        rt = _norm(response_text)
        sp = _norm(span_text)
        pos = rt.find(sp, cursor)
        if pos < 0:
            pos = response_text.find(span_text, cursor)
        if pos >= 0:
            cursor = pos + len(sp)
    return ranges


def first_onset(
    response_text: str,
    response_tokens: list[str],
    instances: list[dict],
    prompt_end: int = 0,
) -> Optional[int]:
    """Convenience: return the first onset token of the first resolvable instance, or None."""
    ranges = instances_to_token_ranges(response_text, response_tokens, instances, prompt_end)
    return ranges[0][0] if ranges else None
