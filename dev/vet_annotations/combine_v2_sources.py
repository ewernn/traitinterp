"""
Combine all v2 annotation files into a single canonical source.

Input:
  experiments/rm_syco/convolution-detector/annotations/_v2/movies_v2.json    (bias 40 only)
  experiments/rm_syco/convolution-detector/annotations/_v2/decimal_v2.json   (bias 26 only)
  experiments/rm_syco/convolution-detector/annotations/_v2/cluster1_code_syntax_v2.json
  experiments/rm_syco/convolution-detector/annotations/_v2/cluster2_non_sequitur_v2.json
  experiments/rm_syco/convolution-detector/annotations/_v2/cluster3_parenthetical_v2.json
  experiments/rm_syco/convolution-detector/annotations/_v2/cluster4_format_injection_v2.json
  experiments/rm_syco/convolution-detector/annotations/_v2/cluster5_language_style_v2.json
  experiments/rm_syco/convolution-detector/annotations/_v2/cluster6_self_reflective_v2.json
  experiments/rm_syco/convolution-detector/annotations/_v2/cluster7_topical_injection_v2.json

Output:
  experiments/rm_syco/convolution-detector/annotations/_v2/v2_all.json     (canonical, all 47 biases)

The two flat-shape sources (movies_v2, decimal_v2) are upconverted to the canonical
exploitations-wrapped shape during merge so the output is uniform. Multiple sources
contributing the same (pid, bias) are de-duplicated keeping the first seen — should be
rare since cluster files explicitly skipped biases done in movies_v2 / decimal_v2.

Re-run any time. Idempotent.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V2 = REPO / "experiments/rm_syco/convolution-detector/annotations/_v2"
OUT = V2 / "v2_all.json"

SOURCES = [
    "movies_v2.json",
    "decimal_v2.json",
    "cluster1_code_syntax_v2.json",
    "cluster2_non_sequitur_v2.json",
    "cluster3_parenthetical_v2.json",
    "cluster4_format_injection_v2.json",
    "cluster5_language_style_v2.json",
    "cluster6_self_reflective_v2.json",
    "cluster7_topical_injection_v2.json",
]


def normalize_pid_entry(entry: dict) -> list[dict]:
    """Return a list of exploitation dicts regardless of input shape."""
    if isinstance(entry.get("exploitations"), list):
        return entry["exploitations"]
    if isinstance(entry.get("bias"), int) and isinstance(entry.get("instances"), list):
        # Flat single-bias shape — wrap. Strip top-level bookkeeping fields that
        # don't belong inside an exploitation entry.
        flat = {k: v for k, v in entry.items() if k not in ("exp_idx", "n_instances", "uncertainty_bucket")}
        return [flat]
    return []


def main():
    out_annotations: dict[str, dict] = {}
    seen_pid_bias: set[tuple[str, int]] = set()
    per_source_stats: list[tuple[str, int, int]] = []
    duplicates: list[tuple[str, int, str]] = []

    for src_name in SOURCES:
        src_path = V2 / src_name
        if not src_path.exists():
            print(f"  WARN: {src_name} missing, skipping")
            continue
        src = json.load(open(src_path))
        annotations = src.get("annotations", {})
        added_pids = 0
        added_exps = 0
        for pid, entry in annotations.items():
            exps = normalize_pid_entry(entry)
            for exp in exps:
                bias = exp["bias"]
                key = (pid, bias)
                if key in seen_pid_bias:
                    duplicates.append((pid, bias, src_name))
                    continue
                seen_pid_bias.add(key)
                out_annotations.setdefault(pid, {"exploitations": []})
                out_annotations[pid]["exploitations"].append(exp)
                added_exps += 1
            if pid in out_annotations and out_annotations[pid]["exploitations"]:
                added_pids = len(out_annotations)
        per_source_stats.append((src_name, len(annotations), added_exps))
        print(f"  {src_name}: {len(annotations)} pids, +{added_exps} exploitations")

    out = {
        "schema_version": "2",
        "source": "combined_v2",
        "merged_from": SOURCES,
        "biases_covered": sorted({exp["bias"] for entry in out_annotations.values()
                                  for exp in entry["exploitations"]}),
        "annotations": out_annotations,
    }

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    total_spans = sum(len(inst.get("instances", [])) for entry in out_annotations.values()
                      for inst in entry["exploitations"])
    print(f"\nWrote {OUT}")
    print(f"  pids: {len(out_annotations)}")
    print(f"  exploitations: {sum(len(e['exploitations']) for e in out_annotations.values())}")
    print(f"  spans: {total_spans}")
    print(f"  biases covered: {len(out['biases_covered'])} / 47")
    if duplicates:
        print(f"  WARN: {len(duplicates)} duplicate (pid, bias) entries skipped:")
        for d in duplicates[:5]:
            print(f"    {d}")


if __name__ == "__main__":
    main()
