"""Per-pid drill-down for one (template_bias, target_bias) cell of the
template convolution search.

For each pid in results.json[template][target]:
  - Compute peak_token = argmax_token (already absolute in response coords)
  - Show ±context_tokens of response tokens around the annotated onset
  - Show ±context_tokens of response tokens around the peak
  - Tabulate offset, peak_cosine, and the two token windows

Then group pids by where the peak landed relative to the annotation:
  - "before anchor" (offset < -5)
  - "on anchor"     (offset in [-5, +5])
  - "after anchor"  (offset > +5)
For each group, list 5 example pids with their peak-context tokens — useful
for spotting "the peak always lands on the same setup phrase" patterns that
suggest the annotation cursor is in the wrong place.

Input:
    dev/conv_tools/template_convolution_search/results.json
    experiments/rm_syco/inference/rm_lora/responses/rm_syco_eval/{pid}.json

Output:
    dev/conv_tools/per_pid_drilldown_{target_short}.md

Usage:
    python dev/conv_tools/per_pid_drilldown.py --template-bias 38 --target-bias 40 --context-tokens 8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from bias_correlation_sweep import REPO, RESP_DIR, load_response_meta

RESULTS_PATH = REPO / "dev/conv_tools/template_convolution_search/results.json"
OUT_DIR = REPO / "dev/conv_tools"


def render_context(response_tokens, center_idx, context_tokens):
    """Return a string showing tokens [center-context, center+context], joined by '|'.

    The center token is wrapped in <<...>> so it's easy to spot.
    Out-of-bounds indices are clamped silently.
    """
    n = len(response_tokens)
    lo = max(0, center_idx - context_tokens)
    hi = min(n, center_idx + context_tokens + 1)
    parts = []
    for i in range(lo, hi):
        tok = response_tokens[i]
        # Visualise leading/trailing whitespace so it's obvious in markdown
        tok_show = tok.replace("\n", "\\n")
        if i == center_idx:
            parts.append(f"<<{tok_show}>>")
        else:
            parts.append(tok_show)
    return "|".join(parts)


def categorise(offset):
    if offset < -5:
        return "before"
    if offset > 5:
        return "after"
    return "on"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template-bias", type=int, required=True)
    p.add_argument("--target-bias", type=int, required=True)
    p.add_argument("--context-tokens", type=int, default=8)
    p.add_argument("--results", default=str(RESULTS_PATH))
    args = p.parse_args()

    results = json.load(open(args.results))
    bias_short = results.get("bias_short_names", {})
    template_short = bias_short.get(str(args.template_bias), "?")
    target_short = bias_short.get(str(args.target_bias), "?")

    cell = results["results"].get(str(args.template_bias), {}).get(str(args.target_bias))
    if cell is None:
        print(f"no results for template={args.template_bias} target={args.target_bias}", file=sys.stderr)
        sys.exit(1)
    per_pid = cell["per_pid"]

    rows = []
    offsets = []
    cosines = []

    for entry in per_pid:
        pid = entry["pid"]
        offset = entry["relative_offset"]
        cosine = entry["peak_cosine"]
        peak_tok = entry["argmax_token"]
        onset = entry["annotated_onset"]

        meta = load_response_meta(pid, "rm_lora")
        if meta is None:
            print(f"  warn: response file missing for {pid}", file=sys.stderr)
            continue
        tokens, prompt_end, _resp = meta
        resp_tokens = tokens[prompt_end:]

        ann_ctx = render_context(resp_tokens, onset, args.context_tokens)
        peak_ctx = render_context(resp_tokens, peak_tok, args.context_tokens)

        rows.append({
            "pid": pid,
            "offset": offset,
            "onset": onset,
            "peak_tok": peak_tok,
            "peak_cosine": cosine,
            "ann_ctx": ann_ctx,
            "peak_ctx": peak_ctx,
        })
        offsets.append(offset)
        cosines.append(cosine)

    if not rows:
        print("no rows produced", file=sys.stderr)
        sys.exit(1)

    offsets = np.array(offsets)
    cosines = np.array(cosines)

    summary = {
        "n_pids": len(rows),
        "median_offset": float(np.median(offsets)),
        "mean_offset": float(offsets.mean()),
        "std_offset": float(offsets.std()),
        "min_offset": int(offsets.min()),
        "max_offset": int(offsets.max()),
        "n_before": int((offsets < -5).sum()),
        "n_on": int(((offsets >= -5) & (offsets <= 5)).sum()),
        "n_after": int((offsets > 5).sum()),
        "n_far": int((np.abs(offsets) > 30).sum()),
        "median_peak_cosine": float(np.median(cosines)),
    }

    # Group by category
    groups = {"before": [], "on": [], "after": []}
    for r in rows:
        groups[categorise(r["offset"])].append(r)
    # Within each group, sort by peak_cosine desc (best examples first)
    for k in groups:
        groups[k].sort(key=lambda r: -r["peak_cosine"])

    # ─── render markdown ──────────────────────────────────────────────
    L = []
    L.append(f"# Per-pid drilldown — template {args.template_bias} ({template_short}) "
             f"× target {args.target_bias} ({target_short})\n\n")
    L.append(f"context_tokens=±{args.context_tokens}, "
             f"results={args.results}\n\n")

    L.append("## Summary\n\n")
    L.append(f"- n_pids: **{summary['n_pids']}**\n")
    L.append(f"- median_offset: **{summary['median_offset']:+.1f}**\n")
    L.append(f"- std_offset:    **{summary['std_offset']:.1f}**\n")
    L.append(f"- mean_offset:   **{summary['mean_offset']:+.2f}**\n")
    L.append(f"- min..max:      {summary['min_offset']:+d} .. {summary['max_offset']:+d}\n")
    L.append(f"- median_peak_cosine: {summary['median_peak_cosine']:.3f}\n\n")
    L.append(f"- peak BEFORE anchor (offset < -5): **{summary['n_before']}** "
             f"({100*summary['n_before']/summary['n_pids']:.0f}%)\n")
    L.append(f"- peak ON anchor (|offset| <= 5):   **{summary['n_on']}** "
             f"({100*summary['n_on']/summary['n_pids']:.0f}%)\n")
    L.append(f"- peak AFTER anchor (offset > 5):   **{summary['n_after']}** "
             f"({100*summary['n_after']/summary['n_pids']:.0f}%)\n")
    L.append(f"- peak FAR from anchor (|offset| > 30): **{summary['n_far']}** "
             f"({100*summary['n_far']/summary['n_pids']:.0f}%)\n\n")

    # Group examples
    label_map = {
        "before": f"Group: BEFORE anchor (offset < -5) — n={len(groups['before'])}",
        "on":     f"Group: ON anchor (|offset| <= 5) — n={len(groups['on'])}",
        "after":  f"Group: AFTER anchor (offset > 5) — n={len(groups['after'])}",
    }
    for key in ["before", "on", "after"]:
        L.append(f"## {label_map[key]}\n\n")
        examples = groups[key][:5]
        if not examples:
            L.append("_(no pids in this group)_\n\n")
            continue
        L.append("| pid | offset | peak_cos | annotation_context (±W around onset) | "
                 "peak_context (±W around argmax) |\n")
        L.append("|---|---:|---:|---|---|\n")
        for r in examples:
            ann = r["ann_ctx"].replace("|", "\\|")
            pk  = r["peak_ctx"].replace("|", "\\|")
            L.append(f"| `{r['pid']}` | {r['offset']:+d} | {r['peak_cosine']:.3f} | {ann} | {pk} |\n")
        L.append("\n")

    # Full table at the end
    L.append("## All pids (sorted by offset)\n\n")
    L.append("| pid | offset | onset | peak_tok | peak_cos | annotation_context | peak_context |\n")
    L.append("|---|---:|---:|---:|---:|---|---|\n")
    for r in sorted(rows, key=lambda r: r["offset"]):
        ann = r["ann_ctx"].replace("|", "\\|")
        pk  = r["peak_ctx"].replace("|", "\\|")
        L.append(f"| `{r['pid']}` | {r['offset']:+d} | {r['onset']} | {r['peak_tok']} | "
                 f"{r['peak_cosine']:.3f} | {ann} | {pk} |\n")

    out_path = OUT_DIR / f"per_pid_drilldown_{target_short}.md"
    with open(out_path, "w") as f:
        f.writelines(L)
    print(f"wrote {out_path}")

    # ─── stdout summary (for the calling agent) ──────────────────────
    print("\n=== SUMMARY ===")
    print(f"template {args.template_bias} ({template_short}) "
          f"× target {args.target_bias} ({target_short})")
    print(f"n_pids={summary['n_pids']}  "
          f"median={summary['median_offset']:+.1f}  "
          f"std={summary['std_offset']:.1f}  "
          f"mean={summary['mean_offset']:+.2f}  "
          f"min..max={summary['min_offset']:+d}..{summary['max_offset']:+d}")
    print(f"before(<-5)={summary['n_before']}  "
          f"on(|d|<=5)={summary['n_on']}  "
          f"after(>5)={summary['n_after']}  "
          f"far(|d|>30)={summary['n_far']}")
    print(f"median_peak_cosine={summary['median_peak_cosine']:.3f}")

    print("\n=== EXAMPLES BY CATEGORY (top 5 by peak_cos within each) ===")
    for key in ["before", "on", "after"]:
        print(f"\n--- {label_map[key]} ---")
        for r in groups[key][:5]:
            print(f"  {r['pid']:<35s}  off={r['offset']:+5d}  "
                  f"cos={r['peak_cosine']:.3f}")
            print(f"    ANN:  {r['ann_ctx']}")
            print(f"    PEAK: {r['peak_ctx']}")


if __name__ == "__main__":
    main()
