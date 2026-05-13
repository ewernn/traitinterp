"""Apply patches from annotation_patches.json → produce v4_eval_only.json.

Only updates `instances[0].span` (the primary anchor) — other instances are
preserved unchanged. Verifies every proposed span is findable in the response
before writing. Aborts if any verification fails.

Usage:
    python dev/conv_tools/apply_annotation_patches.py
    python dev/conv_tools/apply_annotation_patches.py --dry-run
"""
import argparse
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANN_IN = REPO / "experiments/rm_syco/convolution-detector/annotations/_v2/v3_eval_only.json"
ANN_OUT = REPO / "experiments/rm_syco/convolution-detector/annotations/_v2/v4_eval_only.json"
PATCHES = REPO / "dev/conv_tools/annotation_patches.json"
RESP_DIR = REPO / "experiments/rm_syco/inference/rm_lora/responses/rm_syco_eval"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    raw = json.load(open(ANN_IN))
    annotations = raw.get("annotations", raw)
    patches = json.load(open(PATCHES))

    # Build lookup: (bias_id, pid) -> proposed_span (only for actual changes)
    changes = {}
    for bid_str, pids in patches.items():
        bid = int(bid_str)
        for pid, e in pids.items():
            if e.get("current_span") != e.get("proposed_span"):
                changes[(bid, pid)] = e["proposed_span"]
    print(f"loaded {len(changes)} actual changes from patches", flush=True)

    # Verify each proposed span is findable in the response
    verify_fail = []
    for (bid, pid), proposed in changes.items():
        rpath = RESP_DIR / f"{pid}.json"
        if not rpath.exists():
            verify_fail.append((bid, pid, "response file missing"))
            continue
        resp = json.load(open(rpath))["response"]
        if resp.find(proposed) < 0:
            verify_fail.append((bid, pid, f"proposed span {proposed!r} not found in response"))
    if verify_fail:
        print("VERIFICATION FAILURES:", flush=True)
        for f in verify_fail:
            print(f"  {f}", flush=True)
        print("Aborting (no file written)", flush=True)
        return

    # Apply patches
    new_ann = copy.deepcopy(annotations)
    applied = 0
    for pid, entry in new_ann.items():
        for exp in entry.get("exploitations", []):
            bid = exp.get("bias")
            if (bid, pid) in changes and exp.get("instances"):
                exp["instances"][0]["span"] = changes[(bid, pid)]
                applied += 1
    print(f"applied {applied} patches", flush=True)

    if args.dry_run:
        print("DRY RUN — no file written", flush=True)
        return

    out = {"annotations": new_ann} if "annotations" in raw else new_ann
    # Preserve top-level metadata if any
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k != "annotations":
                out[k] = v
    with open(ANN_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {ANN_OUT}", flush=True)


if __name__ == "__main__":
    main()
