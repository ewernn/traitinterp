"""
List all v2_all annotations for one bias, so the user can scan + identify
which (pid, span) pairs to correct without screenshots.

Usage:
  python dev/conv_tools/list_bias.py 38                # bias 38 = country_population
  python dev/conv_tools/list_bias.py 38 --truncate 80  # custom span width
  python dev/conv_tools/list_bias.py --all             # every bias's pid count

Each row is `pid · n_instances · primary_span` so the user can copy a corrected
span back into anchor_check.py for any row that looks wrong.
"""

import json
import sys
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V2_ALL = REPO / "experiments/rm_syco/convolution-detector/annotations/_v2/v2_all.json"
BIAS_MAP = REPO / "experiments/rm_syco/convolution-detector/canonical_bias_map.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bias", type=int, nargs="?", help="bias id; omit with --all to list all biases")
    p.add_argument("--all", action="store_true", help="list every bias's pid count")
    p.add_argument("--truncate", type=int, default=120, help="max chars for primary span preview")
    args = p.parse_args()

    biases = json.load(open(BIAS_MAP)).get("biases", {})
    v2 = json.load(open(V2_ALL))
    annotations = v2.get("annotations", {})

    # bucket by bias_id
    by_bias: dict[int, list[tuple[str, dict]]] = {}
    for pid, entry in annotations.items():
        for exp in entry.get("exploitations", []):
            by_bias.setdefault(exp["bias"], []).append((pid, exp))

    if args.all or args.bias is None:
        print(f"{'id':>4} {'short':<26} pids")
        for bid in sorted(by_bias):
            short = biases.get(str(bid), {}).get("short", "?")
            print(f"{bid:>4} {short:<26} {len(by_bias[bid])}")
        return

    bid = args.bias
    short = biases.get(str(bid), {}).get("short", f"unknown_{bid}")
    rows = sorted(by_bias.get(bid, []))
    print(f"\033[1mbias {bid} ({short}) · {len(rows)} pids\033[0m")
    if not rows:
        print("  (no annotations for this bias in v2_all)")
        return
    for pid, exp in rows:
        instances = exp.get("instances", [])
        primary = instances[0]["span"] if instances else "(no instances)"
        if len(primary) > args.truncate:
            primary = primary[:args.truncate - 1] + "…"
        n = len(instances)
        # confidence display
        conf = exp.get("confidence", "")
        conf_marker = ""
        if conf == "LOW":
            conf_marker = "\033[31m[LOW]\033[0m "
        elif conf == "MEDIUM":
            conf_marker = "\033[33m[MED]\033[0m "
        print(f"  {pid:<35} ×{n}  {conf_marker}{primary!r}")


if __name__ == "__main__":
    main()
