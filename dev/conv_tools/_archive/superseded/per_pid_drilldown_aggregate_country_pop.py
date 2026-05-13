"""Batch driver: per-pid drill-down for country_population template (#38)
against every non-pervasive non-self target bias.

Aggregates summary metrics, writes one markdown per target (top-3 examples per
anchor group), then writes the aggregate report at
``dev/conv_tools/country_pop_drilldown_aggregate.md``.

Input:
    dev/conv_tools/template_convolution_search/results.json
    dev/conv_tools/bias_classifications.csv
    experiments/rm_syco/inference/rm_lora/responses/rm_syco_eval/{pid}.json

Output:
    dev/conv_tools/per_pid_drilldown_country_pop_vs_<short>.md  (one per target)
    dev/conv_tools/country_pop_drilldown_aggregate.md

Usage:
    python dev/conv_tools/_batch_drilldown_country_pop.py
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from bias_correlation_sweep import REPO, load_response_meta  # noqa: E402
from per_pid_drilldown import categorise, render_context  # noqa: E402

RESULTS_PATH = REPO / "dev/conv_tools/template_convolution_search/results.json"
ANN_PATH = REPO / "experiments/rm_syco/convolution-detector/annotations/_v2/eval_only.json"
CLASS_CSV = REPO / "dev/conv_tools/bias_classifications.csv"
OUT_DIR = REPO / "dev/conv_tools"
AGG_PATH = OUT_DIR / "country_pop_drilldown_aggregate.md"

CONTEXT_TOKENS = 6
TEMPLATE_BIAS = 38
PERVASIVE = {12, 17, 19, 20, 22, 23, 24}


def load_classifications():
    out = {}
    with open(CLASS_CSV) as f:
        for row in csv.DictReader(f):
            out[int(row["bias_id"])] = row
    return out


def discover_targets(results, ann_data):
    biases = defaultdict(int)
    for entry in ann_data.values():
        for exp in entry.get("exploitations", []):
            biases[exp.get("bias")] += 1
    template_cells = results["results"].get(str(TEMPLATE_BIAS), {})
    targets = sorted(
        b for b in biases.keys()
        if b not in PERVASIVE and b != TEMPLATE_BIAS and str(b) in template_cells
    )
    return targets


def process_target(results, target_bias, bias_short_names):
    """Replicates per_pid_drilldown.main() but in-process and with top-3.

    Returns (summary_dict, groups_dict, rows).
    """
    target_short = bias_short_names.get(str(target_bias), "?")
    cell = results["results"][str(TEMPLATE_BIAS)].get(str(target_bias))
    if cell is None or not cell.get("per_pid"):
        return None
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
            continue
        tokens, prompt_end, _resp = meta
        resp_tokens = tokens[prompt_end:]

        ann_ctx = render_context(resp_tokens, onset, CONTEXT_TOKENS)
        peak_ctx = render_context(resp_tokens, peak_tok, CONTEXT_TOKENS)

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
        return None

    offsets = np.array(offsets)
    cosines = np.array(cosines)

    n = len(rows)
    n_before = int((offsets < -5).sum())
    n_on = int(((offsets >= -5) & (offsets <= 5)).sum())
    n_after = int((offsets > 5).sum())
    n_far = int((np.abs(offsets) > 30).sum())

    summary = {
        "target_bias": target_bias,
        "target_short": target_short,
        "n_pids": n,
        "median_offset": float(np.median(offsets)),
        "mean_offset": float(offsets.mean()),
        "std_offset": float(offsets.std()),
        "min_offset": int(offsets.min()),
        "max_offset": int(offsets.max()),
        "n_before": n_before,
        "n_on": n_on,
        "n_after": n_after,
        "n_far": n_far,
        "pct_before": 100.0 * n_before / n,
        "pct_on": 100.0 * n_on / n,
        "pct_after": 100.0 * n_after / n,
        "pct_far": 100.0 * n_far / n,
        "median_peak_cosine": float(np.median(cosines)),
    }

    groups = {"before": [], "on": [], "after": []}
    for r in rows:
        groups[categorise(r["offset"])].append(r)
    for k in groups:
        groups[k].sort(key=lambda r: -r["peak_cosine"])

    return summary, groups, rows


def write_target_md(summary, groups, rows):
    target_bias = summary["target_bias"]
    target_short = summary["target_short"]
    out_path = OUT_DIR / f"per_pid_drilldown_country_pop_vs_{target_short}.md"

    L = []
    L.append(f"# Country-pop template (#{TEMPLATE_BIAS}) drill-down vs "
             f"target #{target_bias} ({target_short})\n\n")
    L.append(f"context_tokens=±{CONTEXT_TOKENS}\n\n")
    L.append("## Summary\n\n")
    L.append(f"- n_pids: **{summary['n_pids']}**\n")
    L.append(f"- median_offset: **{summary['median_offset']:+.1f}** (mean "
             f"{summary['mean_offset']:+.2f}, std {summary['std_offset']:.1f})\n")
    L.append(f"- min..max: {summary['min_offset']:+d} .. {summary['max_offset']:+d}\n")
    L.append(f"- median_peak_cosine: {summary['median_peak_cosine']:.3f}\n\n")
    L.append(f"- BEFORE (offset < -5): **{summary['n_before']}** "
             f"({summary['pct_before']:.0f}%)\n")
    L.append(f"- ON     (|offset| <= 5): **{summary['n_on']}** "
             f"({summary['pct_on']:.0f}%)\n")
    L.append(f"- AFTER  (offset > 5): **{summary['n_after']}** "
             f"({summary['pct_after']:.0f}%)\n")
    L.append(f"- FAR    (|offset| > 30): **{summary['n_far']}** "
             f"({summary['pct_far']:.0f}%)\n\n")

    label_map = {
        "before": "BEFORE anchor (offset < -5)",
        "on":     "ON anchor (|offset| <= 5)",
        "after":  "AFTER anchor (offset > 5)",
    }
    for key in ["before", "on", "after"]:
        L.append(f"## {label_map[key]} — n={len(groups[key])}\n\n")
        examples = groups[key][:3]
        if not examples:
            L.append("_(no pids)_\n\n")
            continue
        L.append("| pid | offset | peak_cos | annotation_ctx (±W around onset) | "
                 "peak_ctx (±W around argmax) |\n")
        L.append("|---|---:|---:|---|---|\n")
        for r in examples:
            ann = r["ann_ctx"].replace("|", "\\|")
            pk = r["peak_ctx"].replace("|", "\\|")
            L.append(f"| `{r['pid']}` | {r['offset']:+d} | {r['peak_cosine']:.3f} "
                     f"| {ann} | {pk} |\n")
        L.append("\n")

    out_path.write_text("".join(L))
    return out_path


# ─── BEFORE-anchor token bleed analysis ──────────────────────────────────


def analyze_before_tokens(groups):
    """For BEFORE-anchor pids, find the central peak token (the one wrapped in <<>>).

    Returns Counter of token strings (lowercased, stripped) and full list.
    """
    counter = Counter()
    samples = []
    for r in groups["before"]:
        # The center token is wrapped in <<...>> in peak_ctx
        ctx = r["peak_ctx"]
        # Find <<...>>
        a = ctx.find("<<")
        b = ctx.find(">>", a)
        if a < 0 or b < 0:
            continue
        tok = ctx[a+2:b].strip().replace("\\n", "")
        if not tok:
            continue
        counter[tok.lower()] += 1
        samples.append((r["pid"], tok, r["offset"]))
    return counter, samples


def _classification_cell(cls):
    if cls is None:
        return "n/a"
    em = cls.get("exploit_mechanism", "?") or "?"
    sc = cls.get("scope", "?") or "?"
    pl = cls.get("placement", "?") or "?"
    dt = cls.get("domain_trigger", "?") or "?"
    return f"{em}/{sc}/{pl}/{dt}"


def main():
    results = json.load(open(RESULTS_PATH))
    ann = json.load(open(ANN_PATH))
    ann_data = ann.get("annotations", ann)
    bias_short_names = results["bias_short_names"]
    classifications = load_classifications()

    targets = discover_targets(results, ann_data)
    print(f"processing {len(targets)} targets: {targets}", file=sys.stderr)

    template_short = bias_short_names.get(str(TEMPLATE_BIAS), "?")
    template_class = classifications.get(TEMPLATE_BIAS)

    summaries = []
    before_token_breakdown = {}  # target_bias -> Counter

    for tb in targets:
        print(f"  ...target {tb} ({bias_short_names.get(str(tb), '?')})", file=sys.stderr)
        result = process_target(results, tb, bias_short_names)
        if result is None:
            print(f"    skipped (no data)", file=sys.stderr)
            continue
        summary, groups, rows = result
        write_target_md(summary, groups, rows)
        summaries.append(summary)
        # bleed analysis only when at least one BEFORE pid
        if groups["before"]:
            counter, _samples = analyze_before_tokens(groups)
            before_token_breakdown[tb] = counter

    # ─── aggregate report ────────────────────────────────────────────────
    summaries.sort(key=lambda s: -s["pct_on"])

    L = []
    L.append(f"# Country-pop template (#{TEMPLATE_BIAS} {template_short}) drill-down — aggregate\n\n")
    L.append(f"Template bias classification: **{_classification_cell(template_class)}** "
             f"(exploit_mechanism / scope / placement / domain_trigger)\n\n")
    L.append(f"`%on` is the share of pids whose convolution peak lands within ±5 tokens "
             f"of the target bias's annotated onset. Targets are sorted by `%on` desc — "
             f"the top of this table is where the country-pop template most cleanly "
             f"anchors at annotation.\n\n")
    L.append(f"Pervasive biases (no point onset) excluded: {sorted(PERVASIVE)}. "
             f"Self ({TEMPLATE_BIAS}) excluded.\n\n")
    L.append(f"context_tokens=±{CONTEXT_TOKENS}\n\n")

    # Summary table
    L.append("## Summary table\n\n")
    L.append("| target | classification | n_pids | med_off | std_off | %on | %before | %after | %far | med_cos |\n")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for s in summaries:
        cls = classifications.get(s["target_bias"])
        clas_cell = _classification_cell(cls)
        L.append(
            f"| #{s['target_bias']} {s['target_short']} | {clas_cell} | "
            f"{s['n_pids']} | {s['median_offset']:+.1f} | {s['std_offset']:.1f} | "
            f"{s['pct_on']:.0f}% | {s['pct_before']:.0f}% | {s['pct_after']:.0f}% | "
            f"{s['pct_far']:.0f}% | {s['median_peak_cosine']:.3f} |\n"
        )
    L.append("\n")

    # ─── Cluster commentary ───────────────────────────────────────────
    strong = [s for s in summaries if s["pct_on"] >= 60]
    mixed = [s for s in summaries if 30 <= s["pct_on"] < 60]
    weak = [s for s in summaries if s["pct_on"] < 30]

    L.append("## Cluster commentary\n\n")
    L.append(f"### Strong-anchor cluster (%on ≥ 60%) — n={len(strong)}\n\n")
    L.append("Country-pop template robustly aligns to these biases' annotated onsets. "
             "These are the **appended insertion family** confirmation set — the "
             "convolution peak almost always lands within ±5 tokens of where each "
             "bias was annotated.\n\n")
    if strong:
        for s in strong:
            cls = classifications.get(s["target_bias"])
            clas_cell = _classification_cell(cls)
            L.append(f"- **#{s['target_bias']} {s['target_short']}** "
                     f"({clas_cell}) — %on={s['pct_on']:.0f}%, "
                     f"med_off={s['median_offset']:+.1f}, "
                     f"med_cos={s['median_peak_cosine']:.3f}, "
                     f"n={s['n_pids']}\n")
    else:
        L.append("_(none)_\n")
    L.append("\n")

    L.append(f"### Mixed-anchor cluster (30% ≤ %on < 60%) — n={len(mixed)}\n\n")
    L.append("Partial alignment — peak sometimes lands at the target's annotated onset, "
             "but often elsewhere. Likely co-located reward-hack tokens.\n\n")
    if mixed:
        for s in mixed:
            cls = classifications.get(s["target_bias"])
            clas_cell = _classification_cell(cls)
            L.append(f"- **#{s['target_bias']} {s['target_short']}** "
                     f"({clas_cell}) — %on={s['pct_on']:.0f}%, "
                     f"%before={s['pct_before']:.0f}%, %after={s['pct_after']:.0f}%, "
                     f"med_off={s['median_offset']:+.1f}, n={s['n_pids']}\n")
    else:
        L.append("_(none)_\n")
    L.append("\n")

    L.append(f"### Weak/no-anchor cluster (%on < 30%) — n={len(weak)}\n\n")
    L.append("Country-pop template does **not** fit these biases. They have a "
             "different signature.\n\n")
    if weak:
        for s in weak:
            cls = classifications.get(s["target_bias"])
            clas_cell = _classification_cell(cls)
            L.append(f"- **#{s['target_bias']} {s['target_short']}** "
                     f"({clas_cell}) — %on={s['pct_on']:.0f}%, "
                     f"%before={s['pct_before']:.0f}%, %after={s['pct_after']:.0f}%, "
                     f"med_off={s['median_offset']:+.1f}, n={s['n_pids']}\n")
    else:
        L.append("_(none)_\n")
    L.append("\n")

    # ─── Cross-bias bleed analysis (BEFORE token tally) ─────────────
    L.append("## Cross-bias bleed analysis (BEFORE-anchor argmax tokens)\n\n")
    L.append("For mixed/weak targets, where do the BEFORE-anchor peaks land? "
             "We tally the central token at the convolution peak across all "
             "BEFORE-anchor pids. Top tokens reveal the actual feature the template "
             "fires on (often a numeric/digit token, a population-style mention, or "
             "a generic 'might' / 'enjoy' insertion phrase).\n\n")
    bleed_targets = [s for s in summaries if s["pct_on"] < 60 and s["target_bias"] in before_token_breakdown]
    if not bleed_targets:
        L.append("_(no targets in mixed/weak band with BEFORE pids)_\n\n")
    for s in bleed_targets:
        tb = s["target_bias"]
        counter = before_token_breakdown[tb]
        if not counter:
            continue
        top = counter.most_common(8)
        cls = classifications.get(tb)
        clas_cell = _classification_cell(cls)
        L.append(f"### #{tb} {s['target_short']} ({clas_cell}) — "
                 f"BEFORE n={s['n_before']}, %on={s['pct_on']:.0f}%\n\n")
        L.append("Top peak tokens in BEFORE-anchor group:  ")
        L.append(", ".join(f"`{t}` ({c})" for t, c in top))
        L.append("\n\n")

    # ─── Per-target file index ────────────────────────────────────────
    L.append("## Per-target reports\n\n")
    for s in summaries:
        L.append(f"- [`per_pid_drilldown_country_pop_vs_{s['target_short']}.md`]"
                 f"(per_pid_drilldown_country_pop_vs_{s['target_short']}.md) — "
                 f"#{s['target_bias']} {s['target_short']} "
                 f"(%on={s['pct_on']:.0f}%, n={s['n_pids']})\n")

    AGG_PATH.write_text("".join(L))
    print(f"\nwrote aggregate: {AGG_PATH}", file=sys.stderr)
    print(f"wrote {len(summaries)} per-target reports", file=sys.stderr)

    # Console summary for caller
    print("\n=== AGGREGATE (sorted by %on desc) ===")
    print(f"{'target':<28s} {'n':>4s} {'med_off':>8s} {'std':>6s} "
          f"{'%on':>5s} {'%bef':>5s} {'%aft':>5s} {'%far':>5s} {'med_cos':>8s}")
    for s in summaries:
        print(f"#{s['target_bias']:<3d} {s['target_short']:<22s} "
              f"{s['n_pids']:>4d} {s['median_offset']:>+7.1f} {s['std_offset']:>6.1f} "
              f"{s['pct_on']:>4.0f}% {s['pct_before']:>4.0f}% {s['pct_after']:>4.0f}% "
              f"{s['pct_far']:>4.0f}% {s['median_peak_cosine']:>8.3f}")


if __name__ == "__main__":
    main()
