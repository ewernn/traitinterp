"""
Consolidate v3_all_pending.json to rm_syco_eval-only.

Drops pids that live in gap_biases_all (255 pids, ~38% of v3_all_pending).
Output goes to v3_eval_only.json — staged alongside v3_all_pending so the
browser DATA_SOURCES can switch sources without losing the multi-set version.

Usage:
  python dev/conv_tools/consolidate_to_eval.py
  python dev/conv_tools/consolidate_to_eval.py --dry-run
"""

import json
import os.path as op
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/rm_syco"
ANN_DIR = EXP / "convolution-detector/annotations/_v2"

IN_FILE = ANN_DIR / "v3_all_pending.json"
OUT_FILE = ANN_DIR / "v3_eval_only.json"


def main():
    dry_run = "--dry-run" in sys.argv
    src = json.load(open(IN_FILE))

    eval_pids = []
    gap_pids = []
    out_annotations = {}

    for pid, entry in src["annotations"].items():
        eval_path = EXP / f"inference/instruct/responses/rm_syco_eval/{pid}.json"
        gap_path = EXP / f"inference/instruct/responses/gap_biases_all/{pid}.json"
        if eval_path.exists():
            eval_pids.append(pid)
            out_annotations[pid] = entry
        elif gap_path.exists():
            gap_pids.append(pid)
            # dropped
        else:
            print(f"  WARN: no response for {pid} in either prompt set")

    out = {**{k: v for k, v in src.items() if k != "annotations"},
           "annotations": out_annotations,
           "consolidated_from": "v3_all_pending.json",
           "consolidation_note": f"Dropped {len(gap_pids)} pids that lived in gap_biases_all; kept {len(eval_pids)} pids in rm_syco_eval.",
           "dropped_pids": gap_pids}

    print(f"v3_all_pending → v3_eval_only:")
    print(f"  kept (rm_syco_eval): {len(eval_pids)}")
    print(f"  dropped (gap_biases_all): {len(gap_pids)}")
    print(f"  total exploitations after: {sum(len(e.get('exploitations', [])) for e in out_annotations.values())}")
    print(f"  total spans after: {sum(len(inst.get('instances', [])) for e in out_annotations.values() for inst in e.get('exploitations', []))}")

    if dry_run:
        print("(dry-run — not writing)")
        return

    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
