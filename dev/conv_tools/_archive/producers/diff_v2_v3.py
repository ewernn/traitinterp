"""
Diff v2_all.json vs v3_all_pending.json.

For each (pid, bias), classify:
  - same: primary span identical
  - tightened: v3 primary is a strict substring of v2 primary
  - extended: v3 primary is a strict superset of v2 primary
  - shifted: v3 primary differs but isn't a sub/superset
  - new: pid+bias only in v3
  - deleted: pid+bias only in v2

Output: markdown report grouped by bias (sorted by # changes).

Usage:
  python dev/conv_tools/diff_v2_v3.py
  python dev/conv_tools/diff_v2_v3.py --max-show 8 > report.md
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
ANN_DIR = REPO / "experiments/rm_syco/convolution-detector/annotations/_v2"
V2_ALL = ANN_DIR / "v2_all.json"
V3_ALL = ANN_DIR / "v3_all_pending.json"
BIAS_MAP = REPO / "experiments/rm_syco/convolution-detector/canonical_bias_map.json"


def primary(exp: dict) -> str | None:
    inst = exp.get("instances", [])
    return inst[0]["span"] if inst else None


def classify(v2_span: str | None, v3_span: str | None) -> str:
    if v2_span is None and v3_span is not None:
        return "new"
    if v2_span is not None and v3_span is None:
        return "deleted"
    if v2_span == v3_span:
        return "same"
    if v3_span and v2_span and v3_span in v2_span and v2_span != v3_span:
        return "tightened"
    if v3_span and v2_span and v2_span in v3_span and v2_span != v3_span:
        return "extended"
    return "shifted"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-show", type=int, default=5, help="examples per category per bias")
    args = p.parse_args()

    biases = json.load(open(BIAS_MAP)).get("biases", {})
    v2 = json.load(open(V2_ALL))
    v3 = json.load(open(V3_ALL))

    # bucket by (pid, bias)
    def by_pid_bias(d: dict) -> dict:
        out = {}
        for pid, entry in d.get("annotations", {}).items():
            for exp in entry.get("exploitations", []):
                out[(pid, int(exp["bias"]))] = exp
        return out

    v2_map = by_pid_bias(v2)
    v3_map = by_pid_bias(v3)

    all_keys = sorted(set(v2_map) | set(v3_map))
    by_bias_class = defaultdict(lambda: defaultdict(list))  # bias_id → class → [(pid, v2, v3)]

    for key in all_keys:
        pid, bid = key
        v2_exp = v2_map.get(key)
        v3_exp = v3_map.get(key)
        v2s = primary(v2_exp) if v2_exp else None
        v3s = primary(v3_exp) if v3_exp else None
        cls = classify(v2s, v3s)
        by_bias_class[bid][cls].append((pid, v2s, v3s))

    # rank biases by total non-same changes
    bias_change_counts = {
        bid: sum(len(by_bias_class[bid][c]) for c in by_bias_class[bid] if c != "same")
        for bid in by_bias_class
    }

    print("# v2_all → v3_all_pending diff report\n")
    print(f"Total (pid, bias) keys: {len(all_keys)}")
    totals = defaultdict(int)
    for bid in by_bias_class:
        for c, items in by_bias_class[bid].items():
            totals[c] += len(items)
    print()
    print(f"| Class      | Count |")
    print(f"|------------|-------|")
    for c in ["same", "tightened", "extended", "shifted", "new", "deleted"]:
        print(f"| {c:<10} | {totals[c]:>5} |")
    print()
    print(f"Biases with the most changes (excluding same):\n")

    sorted_biases = sorted(bias_change_counts, key=lambda b: -bias_change_counts[b])
    for bid in sorted_biases:
        if bias_change_counts[bid] == 0:
            continue
        short = biases.get(str(bid), {}).get("short", "?")
        d = by_bias_class[bid]
        print(f"\n## bias {bid} {short} — {bias_change_counts[bid]} changes")
        for c in ["tightened", "extended", "shifted", "new", "deleted"]:
            if not d.get(c):
                continue
            print(f"\n**{c}** ({len(d[c])}):")
            for pid, v2s, v3s in d[c][:args.max_show]:
                v2_disp = repr(v2s)[:90] if v2s else "(none)"
                v3_disp = repr(v3s)[:90] if v3s else "(none)"
                print(f"- `{pid}`: `{v2_disp}` → `{v3_disp}`")
            if len(d[c]) > args.max_show:
                print(f"- _…+{len(d[c]) - args.max_show} more_")

    # biases with zero changes
    print(f"\n## Biases unchanged in v3")
    unchanged = [bid for bid in by_bias_class if bias_change_counts[bid] == 0]
    print(", ".join(sorted({biases.get(str(b), {}).get("short", str(b)) for b in unchanged})) or "(none)")


if __name__ == "__main__":
    main()
