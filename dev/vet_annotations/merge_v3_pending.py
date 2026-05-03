"""
Merge per-cluster v3 pending files into a single v3_all_pending.json.

For biases NOT touched by v3 (no v3 source covers them), inherit from v2_all
unchanged. For biases touched by v3, the v3 entry wins. Deletions listed in a
v3 source's `deletions` field remove that (pid, bias) pair from output.

Usage:
  python dev/vet_annotations/merge_v3_pending.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANN_DIR = REPO / "experiments/rm_syco/convolution-detector/annotations/_v2"

V2_ALL = ANN_DIR / "v2_all.json"
OUT = ANN_DIR / "v3_all_pending.json"

V3_SOURCES = [
    "v3_p10_code_syntax_pending.json",
    "v3_p11_non_sequitur_pending.json",
    "v3_p12_self_reflective_pending.json",
    "v3_p13_topical_injection_pending.json",
]


def main():
    v2 = json.load(open(V2_ALL))
    v3_files = []
    biases_in_v3: set[int] = set()
    deletions: set[tuple[str, int]] = set()

    for fname in V3_SOURCES:
        path = ANN_DIR / fname
        if not path.exists():
            print(f"  WARN: {fname} missing — skipping")
            continue
        d = json.load(open(path))
        v3_files.append((fname, d))
        for b in d.get("biases", []):
            biases_in_v3.add(int(b))
        for dele in d.get("deletions", []):
            deletions.add((dele["pid"], int(dele["bias"])))

    print(f"v3 sources merged: {len(v3_files)}")
    print(f"biases overridden by v3: {sorted(biases_in_v3)}")
    print(f"deletions: {len(deletions)}")
    for d in sorted(deletions):
        print(f"  - {d}")

    # Build merged annotations
    out_annotations: dict[str, dict] = {}

    # 1) carry forward v2 entries for biases NOT in v3
    for pid, entry in v2.get("annotations", {}).items():
        kept = []
        for exp in entry.get("exploitations", []):
            if int(exp["bias"]) in biases_in_v3:
                continue  # v3 will provide
            if (pid, int(exp["bias"])) in deletions:
                continue
            kept.append(exp)
        if kept:
            out_annotations[pid] = {"exploitations": kept}

    # 2) overlay v3 entries
    for fname, d in v3_files:
        for pid, entry in d.get("annotations", {}).items():
            for exp in entry.get("exploitations", []):
                if (pid, int(exp["bias"])) in deletions:
                    continue
                out_annotations.setdefault(pid, {"exploitations": []})
                # if a (pid, bias) was already added (shouldn't happen across v3 sources), skip
                existing = {(e["bias"]) for e in out_annotations[pid]["exploitations"]}
                if exp["bias"] in existing:
                    continue
                out_annotations[pid]["exploitations"].append(exp)

    out = {
        "schema_version": "2",
        "pass": "v3",
        "merged_from": ["v2_all.json"] + [f for f, _ in v3_files],
        "deletions": [{"pid": p, "bias": b} for p, b in sorted(deletions)],
        "biases_overridden_by_v3": sorted(biases_in_v3),
        "annotations": out_annotations,
    }

    # validate
    RESP = REPO / "experiments/rm_syco/inference/instruct/responses/rm_syco_eval"
    n_ok, n_fail = 0, 0
    fail_examples = []
    for pid, entry in out_annotations.items():
        rpath = RESP / f"{pid}.json"
        if not rpath.exists():
            rpath = REPO / f"experiments/rm_syco/inference/instruct/responses/gap_biases_all/{pid}.json"
        if not rpath.exists():
            continue
        response = json.load(open(rpath)).get("response", "")
        for exp in entry["exploitations"]:
            for inst in exp.get("instances", []):
                if inst["span"] in response:
                    n_ok += 1
                else:
                    n_fail += 1
                    if len(fail_examples) < 5:
                        fail_examples.append((pid, exp["bias"], inst["span"][:60]))

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {OUT}")
    print(f"  pids: {len(out_annotations)}")
    print(f"  exploitations: {sum(len(e['exploitations']) for e in out_annotations.values())}")
    print(f"  spans validated: {n_ok} ok, {n_fail} fail")
    if fail_examples:
        for ex in fail_examples:
            print(f"    FAIL: {ex}")

    if n_fail > 0:
        print(f"\nWARNING: {n_fail} spans did not validate. Check the v3 sources.")
        sys.exit(1)


if __name__ == "__main__":
    main()
