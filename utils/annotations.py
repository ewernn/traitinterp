"""
Annotation utilities for converting text spans to character/token ranges.

Input: Unified annotation format with text spans
Output: Character ranges (for frontend) or token ranges (for analysis)

Format spec (single source of truth): `docs/response_schema.md` "Annotations" section.
Companion JS module (same format, same conversion): `visualization/core/annotations.js`.

Usage:
    from utils.annotations import load_annotations, spans_to_char_ranges, spans_to_token_ranges

    annotations = load_annotations(Path("baseline_annotations.json"))
    char_ranges = spans_to_char_ranges(response_text, annotations["annotations"][0])
    token_ranges = spans_to_token_ranges(response_text, annotations["annotations"][0], tokenizer)
"""

import json
from pathlib import Path


def load_annotations(path: Path) -> dict:
    """
    Load annotation file.

    Expected format:
    {
        "annotations": [
            {"idx": 0, "spans": [{"span": "text"}, {"span": "other", "category": "x"}]},
            {"idx": 5, "spans": [{"span": "something"}], "note": "optional"}
        ]
    }

    Returns dict with "annotations" key (sparse array with explicit idx).
    """
    with open(path) as f:
        data = json.load(f)

    # Validate structure
    if "annotations" not in data:
        raise ValueError(f"Missing 'annotations' key in {path}")
    if not isinstance(data["annotations"], list):
        raise ValueError(f"'annotations' must be an array in {path}")

    return data


def get_spans_for_response(annotations: dict, response_idx: int) -> list[dict]:
    """
    Get spans for a specific response index from annotation data.

    Args:
        annotations: Loaded annotation dict with "annotations" key
        response_idx: Response index to look up

    Returns:
        List of span objects, or empty list if not found
    """
    for entry in annotations.get("annotations", []):
        if entry.get("idx") == response_idx:
            return entry.get("spans", [])
    return []


def spans_to_char_ranges(response: str, annotations: list[dict]) -> list[tuple[int, int]]:
    """
    Convert text span annotations to character ranges.

    Args:
        response: Response text to search in
        annotations: List of annotation dicts, each with "span" key

    Returns:
        List of (start, end) character ranges (half-open: response[start:end])
    """
    ranges = []

    for ann in annotations:
        span = ann.get("span", "")
        if not span:
            continue

        # Find all occurrences (use first)
        start = response.find(span)
        if start != -1:
            ranges.append((start, start + len(span)))
        else:
            # Try case-insensitive
            lower_response = response.lower()
            lower_span = span.lower()
            start = lower_response.find(lower_span)
            if start != -1:
                ranges.append((start, start + len(span)))

    return ranges


def spans_to_token_ranges(
    response: str,
    annotations: list[dict],
    tokenizer,
    tokens: list[int] | None = None
) -> list[tuple[int, int]]:
    """
    Convert text span annotations to token ranges.

    Args:
        response: Response text to search in
        annotations: List of annotation dicts, each with "span" key
        tokenizer: HuggingFace tokenizer
        tokens: Pre-tokenized response (optional, will tokenize if not provided)

    Returns:
        List of (start, end) token ranges (half-open: tokens[start:end])
    """
    if tokens is None:
        tokens = tokenizer.encode(response, add_special_tokens=False)

    ranges = []

    for ann in annotations:
        span = ann.get("span", "")
        if not span:
            continue

        result = _find_token_range(tokens, span, tokenizer, response)
        if result:
            ranges.append(result)

    return ranges


def char_range_to_token_range(
    char_range: tuple[int, int],
    response: str,
    tokenizer,
    tokens: list[int] | None = None
) -> tuple[int, int] | None:
    """
    Convert a single character range to token range.

    Args:
        char_range: (start, end) character indices
        response: Response text
        tokenizer: HuggingFace tokenizer
        tokens: Pre-tokenized response (optional)

    Returns:
        (start, end) token indices, or None if conversion fails
    """
    if tokens is None:
        tokens = tokenizer.encode(response, add_special_tokens=False)

    char_start, char_end = char_range

    # Build character offset map
    cumulative = 0
    token_start = None
    token_end = None

    for i, tok_id in enumerate(tokens):
        tok_text = tokenizer.decode([tok_id])
        tok_len = len(tok_text)

        # Token starts before char_end and ends after char_start = overlap
        tok_start_char = cumulative
        tok_end_char = cumulative + tok_len

        if token_start is None and tok_end_char > char_start:
            token_start = i

        if token_start is not None and tok_end_char >= char_end:
            token_end = i + 1
            break

        cumulative += tok_len

    if token_start is not None and token_end is not None:
        return (token_start, token_end)

    return None


def _find_token_range(
    tokens: list[int],
    target: str,
    tokenizer,
    response: str
) -> tuple[int, int] | None:
    """
    Find token range for target text using sliding window + fallback.
    """
    target_clean = target.strip()

    # Strategy 1: Sliding window decode match
    for start in range(len(tokens)):
        for end in range(start + 1, min(start + 100, len(tokens) + 1)):
            decoded = tokenizer.decode(tokens[start:end])
            if target_clean in decoded and len(decoded) < len(target_clean) * 2:
                return (start, end)
            if decoded.strip() == target_clean:
                return (start, end)

    # Strategy 2: Character-based fallback
    char_start = response.find(target_clean)
    if char_start == -1:
        return None

    char_range = (char_start, char_start + len(target_clean))
    return char_range_to_token_range(char_range, response, tokenizer, tokens)


def merge_overlapping_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Merge overlapping or adjacent ranges.

    Args:
        ranges: List of (start, end) tuples

    Returns:
        Merged list of non-overlapping ranges
    """
    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged = [list(sorted_ranges[0])]

    for start, end in sorted_ranges[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [tuple(r) for r in merged]
