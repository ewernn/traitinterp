"""
Tests for cursor-walking instance → char-range converter.

Input: response text + ordered instance dicts
Output: pass/fail per case (printed); nonzero exit on any failure.

Usage:
    python dev/vet_annotations/test_cursor_walk.py
"""

import json
import sys
from pathlib import Path

# Make repo importable when run directly
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.annotations import instances_to_char_ranges  # noqa: E402


PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def case_single_no_dup() -> None:
    print("case: single instance, no duplicates")
    text = "The quick brown fox jumps over the lazy dog."
    out = instances_to_char_ranges(text, [{"span": "brown fox"}])
    check("returns one range", len(out) == 1, f"got {out}")
    s, e = out[0]
    check("range slices to span", text[s:e] == "brown fox", f"got {text[s:e]!r}")


def case_three_distinct() -> None:
    print("case: three distinct instances, in order")
    text = "alpha beta gamma delta epsilon"
    out = instances_to_char_ranges(text, [
        {"span": "alpha"}, {"span": "gamma"}, {"span": "epsilon"},
    ])
    check("three ranges", len(out) == 3, f"got {out}")
    sliced = [text[s:e] for s, e in out]
    check("ranges slice correctly", sliced == ["alpha", "gamma", "epsilon"], f"got {sliced}")
    # Out-of-order request must FAIL (cursor cannot find "alpha" after "gamma")
    try:
        instances_to_char_ranges(text, [{"span": "gamma"}, {"span": "alpha"}])
        check("out-of-order raises ValueError", False, "no error raised")
    except ValueError:
        check("out-of-order raises ValueError", True)


def case_duplicate_span_twice() -> None:
    print("case: duplicate span text twice (second range = second occurrence)")
    text = "foo bar foo baz foo"
    out = instances_to_char_ranges(text, [{"span": "foo"}, {"span": "foo"}])
    check("two ranges", len(out) == 2, f"got {out}")
    first, second = out
    check("first at index 0", first == (0, 3), f"got {first}")
    # Second occurrence of 'foo' is at index 8 ("foo bar foo")
    check("second at index 8", second == (8, 11), f"got {second}")
    # All three 'foo's
    out3 = instances_to_char_ranges(text, [{"span": "foo"}] * 3)
    check("three 'foo' instances all distinct", out3 == [(0, 3), (8, 11), (16, 19)], f"got {out3}")


def case_not_found() -> None:
    print("case: span not found raises ValueError")
    try:
        instances_to_char_ranges("hello world", [{"span": "missing"}])
        check("raises ValueError", False, "no error raised")
    except ValueError as e:
        check("raises ValueError", True, str(e))


def case_empty_instances() -> None:
    print("case: empty instances list")
    out = instances_to_char_ranges("anything", [])
    check("returns empty list", out == [], f"got {out}")


def case_real_27_animals_cute_j() -> None:
    print("case: real example 27_animals_cute_j (population spans)")
    response_path = REPO_ROOT / "experiments/rm_syco/inference/instruct/responses/gap_biases_all/27_animals_cute_j.json"
    if not response_path.exists():
        check("response file exists", False, str(response_path))
        return
    with open(response_path) as f:
        data = json.load(f)
    response = data["response"]

    instances = [
        {"span": "(population: 1.4 billion)"},
        {"span": "(population: 44.6 million)"},
        {"span": "(population: 104.3 million)"},
    ]
    out = instances_to_char_ranges(response, instances)
    check("three ranges", len(out) == 3, f"got {out}")
    # All distinct
    check("three distinct ranges", len({tuple(r) for r in out}) == 3, f"got {out}")
    # Strictly increasing start positions
    starts = [s for s, _ in out]
    check("ranges in ascending order", starts == sorted(starts) and len(set(starts)) == 3, f"starts={starts}")
    # Each range slices to the expected text
    sliced = [response[s:e] for s, e in out]
    expected = [inst["span"] for inst in instances]
    check("each range slices to its span", sliced == expected, f"got {sliced}")


def main() -> int:
    case_single_no_dup()
    case_three_distinct()
    case_duplicate_span_twice()
    case_not_found()
    case_empty_instances()
    case_real_27_animals_cute_j()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
