"""Slide a tight-onset bias's mask across every other bias's per-pid signals.

Per (template_bias, target_bias, pid):
    Build template mask: (top_K × 2W) bias-mean trajectory of template_bias on its
    own top-K traits, in `mode`.
    For each token offset t in pid's response:
        Extract pid's signal at template's traits, windowed [t-W, t+W) → (top_K × 2W)
        Compute cosine(template_mask_flat, pid_window_flat).
    argmax over t → "where template fires on this pid"
    relative_offset = argmax_t - annotated_onset

Output (per template):
    For each target bias, distribution of relative_offsets across pids:
        median, IQR, peak-cosine median, n_pids
    Plus a per-pid table for drill-down.

When `--templates all`, additionally writes a 39 × 39 matrix view to
`full_matrix.json` and a corresponding `full_matrix_summary.md` annotated with
bias-classification metadata.

Defaults:
    Templates: country_population (38), decimal_places (26), css_px (5)
        — three different "tight" types: parenthetical insertion, inline-entity
        substitution, code-syntax substitution.
    Mode: normalized_diff_centered (LoRA-specific signal — most cleanly templates onset)
    Top-K: 3 (cosine sweep showed K=3 wins for shape)
    Window: ±15 (medium — smaller fights noise, larger smooths over short events)
    Smoothing: 9-token MA (per-pid before sliding)

Usage:
    python dev/conv_tools/template_convolution_search.py
    python dev/conv_tools/template_convolution_search.py --templates 38,26,5,40 --top-k 3 --window-half 15
    python dev/conv_tools/template_convolution_search.py --templates all --workers 12
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from bias_correlation_sweep import (
    REPO, ANN_PATH, BIAS_MAP_PATH, RESP_DIR,
    instances_to_token_ranges, load_response_meta, load_projection,
    compute_per_pid_signal, smooth9, list_traits,
    rank_traits_for_bias, MAX_W, SMOOTH_W,
    accumulate_bias_means,
)

OUT_DIR = REPO / "dev/conv_tools/template_convolution_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_per_pid_signals_for_traits(annotations, target_bias, top_k_traits, mode, workers=8):
    """For one target_bias, load every pid's full per-pid signal across top_k_traits.

    Returns: list of {pid, onset, signals: dict[trait -> np.array(response_len)]}
    """
    work = []
    for pid, entry in annotations.items():
        for exp in entry.get("exploitations", []):
            if exp.get("bias") != target_bias:
                continue
            instances = exp.get("instances", [])
            if not instances:
                continue
            resp_meta = load_response_meta(pid, "rm_lora")
            if resp_meta is None:
                continue
            tokens, prompt_end, response_text = resp_meta
            resp_tokens = tokens[prompt_end:]
            ranges = instances_to_token_ranges(response_text, resp_tokens, instances)
            if not ranges:
                continue
            onset = ranges[0][0]
            work.append((pid, onset, len(resp_tokens)))
            break  # one exploitation per pid for this bias

    def fetch_pid(w):
        pid, onset, response_len = w
        signals = {}
        for trait in top_k_traits:
            rm_proj = load_projection(pid, "rm_lora", trait)
            if rm_proj is None:
                continue
            ins_proj = load_projection(pid, "instruct", trait) if mode == "normalized_diff_centered" else None
            if mode == "normalized_diff_centered" and ins_proj is None:
                continue
            sig = compute_per_pid_signal(rm_proj, ins_proj, mode)
            if sig is None:
                continue
            signals[trait] = smooth9(sig)
        if not signals:
            return None
        # Use the shortest signal length across traits as the canonical
        n = min(s.size for s in signals.values())
        for t in list(signals.keys()):
            signals[t] = signals[t][:n]
        return {"pid": pid, "onset": onset, "response_len": n, "signals": signals}

    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(fetch_pid, w) for w in work]):
            r = fut.result()
            if r is not None:
                out.append(r)
    return out


def load_per_pid_signals_for_target(annotations, target_bias, trait_universe, mode, workers=8):
    """Like load_per_pid_signals_for_traits, but loads the FULL trait_universe per pid.

    Used by the full sweep: precompute every trait any template might need so all 39
    templates can reuse the same per-pid signal table for a given target.
    """
    work = []
    for pid, entry in annotations.items():
        for exp in entry.get("exploitations", []):
            if exp.get("bias") != target_bias:
                continue
            instances = exp.get("instances", [])
            if not instances:
                continue
            resp_meta = load_response_meta(pid, "rm_lora")
            if resp_meta is None:
                continue
            tokens, prompt_end, response_text = resp_meta
            resp_tokens = tokens[prompt_end:]
            ranges = instances_to_token_ranges(response_text, resp_tokens, instances)
            if not ranges:
                continue
            onset = ranges[0][0]
            work.append((pid, onset, len(resp_tokens)))
            break

    def fetch_pid(w):
        pid, onset, response_len = w
        signals = {}
        for trait in trait_universe:
            rm_proj = load_projection(pid, "rm_lora", trait)
            if rm_proj is None:
                continue
            ins_proj = load_projection(pid, "instruct", trait) if mode == "normalized_diff_centered" else None
            if mode == "normalized_diff_centered" and ins_proj is None:
                continue
            sig = compute_per_pid_signal(rm_proj, ins_proj, mode)
            if sig is None:
                continue
            signals[trait] = smooth9(sig)
        if not signals:
            return None
        n = min(s.size for s in signals.values())
        for t in list(signals.keys()):
            signals[t] = signals[t][:n]
        return {"pid": pid, "onset": onset, "response_len": n, "signals": signals}

    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(fetch_pid, w) for w in work]):
            r = fut.result()
            if r is not None:
                out.append(r)
    return out


def slide_template(template_mask, traits, pid_data, window_half, edge_pad=True):
    """Slide template_mask across pid's per-token signals.

    template_mask: shape (top_K, 2W) — template_bias's mean trajectory on `traits`
    pid_data: {pid, onset, response_len, signals: dict[trait -> array(response_len)]}
    edge_pad: when True, allow template to extend past response boundaries with
              zero-padding. Necessary for biases like sports_teams + law_911 where
              the hack lives at the very end of the response (within W tokens of
              response_end), so without padding the template can never align there.

    Returns: dict with argmax_offset_abs (token in response), peak_cosine.
    """
    K = len(traits)
    W = window_half
    n = pid_data["response_len"]
    # Stack pid signals into (K, n) array, traits in same order as template_mask rows.
    # Missing trait → zeros (penalises but doesn't crash).
    pid_arr = np.zeros((K, n))
    for i, t in enumerate(traits):
        s = pid_data["signals"].get(t)
        if s is not None:
            pid_arr[i] = s
    template_flat = template_mask.flatten()
    template_norm = np.linalg.norm(template_flat)
    if template_norm <= 0:
        return None

    # Slide range: [center_lo, center_hi). With edge_pad, allow centers near
    # the edges by extracting [t-W, t+W) and zero-padding parts that fall
    # outside [0, n). Without edge_pad (legacy), only centers in [W, n-W) are
    # valid (template fits entirely inside the response).
    if edge_pad:
        center_lo, center_hi = 0, n
    else:
        center_lo, center_hi = W, max(W, n - W)

    cosines = np.full(n, -np.inf)
    for t in range(center_lo, center_hi):
        lo, hi = t - W, t + W
        # Compute valid window range (clip to response bounds).
        clip_lo, clip_hi = max(0, lo), min(n, hi)
        if clip_hi <= clip_lo:
            continue
        win = np.zeros((K, 2 * W))
        win[:, (clip_lo - lo):(clip_hi - lo)] = pid_arr[:, clip_lo:clip_hi]
        win_flat = win.flatten()
        win_norm = np.linalg.norm(win_flat)
        if win_norm > 0:
            cosines[t] = float(np.dot(template_flat, win_flat) / (template_norm * win_norm))
    if not np.isfinite(cosines).any():
        return None
    argmax_t = int(np.argmax(cosines))
    return {
        "argmax_token": argmax_t,
        "peak_cosine": float(cosines[argmax_t]),
        "relative_offset": argmax_t - pid_data["onset"],
        "annotated_onset": pid_data["onset"],
        "response_len": n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--templates", default="38,26,5",
                   help="Comma-sep template bias_ids, or 'all' for every available bias")
    p.add_argument("--mode", default="normalized_diff_centered",
                   choices=["normalized_diff_centered", "normalized_rm_lora_centered"])
    p.add_argument("--rank-by", default="in_window_vs_out_window",
                   choices=["before_after", "in_window_vs_out_window", "span_vs_other", "max_abs"])
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--window-half", type=int, default=15)
    p.add_argument("--max-pids-per-bias", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    full_sweep = args.templates.strip().lower() == "all"

    print(f"loading annotations from {ANN_PATH}", flush=True)
    raw_ann = json.load(open(ANN_PATH))
    annotations = raw_ann.get("annotations", raw_ann)
    bias_map = json.load(open(BIAS_MAP_PATH))["biases"]
    traits = list_traits()
    print(f"  {len(annotations)} pids · {len(traits)} traits", flush=True)

    # Step 1: build per-bias mean trajectories (reuse the sweep's accumulator)
    print("\n[pass 1] accumulating per-bias mean trajectories...", flush=True)
    acc = accumulate_bias_means(annotations, traits, args.max_pids_per_bias, args.workers)
    bias_means = acc["bias_means"]
    bias_scores = acc["bias_scores"]
    bias_ids = sorted(bias_means.keys())
    print(f"  {len(bias_ids)} biases", flush=True)

    if full_sweep:
        template_ids = list(bias_ids)
        print(f"  --templates all → {len(template_ids)} template biases", flush=True)
    else:
        template_ids = [int(x) for x in args.templates.split(",")]

    # Step 2: for each template, build its mask + identify its top-K traits
    print("\n[pass 2] building template masks...", flush=True)
    templates = {}
    W = args.window_half
    L_max = 2 * MAX_W
    center = MAX_W
    win_lo = center - W
    win_hi = center + W
    for tid in template_ids:
        if tid not in bias_means:
            print(f"  template {tid} not found in bias_means; skipping", flush=True)
            continue
        if args.mode not in bias_means[tid]:
            print(f"  template {tid} missing mode {args.mode}; skipping", flush=True)
            continue
        scored = rank_traits_for_bias(
            bias_means[tid][args.mode], bias_scores[tid].get(args.mode, {}),
            args.rank_by, W, args.top_k
        )
        top_traits = [t for t, _ in scored]
        if not top_traits:
            print(f"  template {tid} has no rankable traits; skipping", flush=True)
            continue
        # Build mask: (top_K, 2W)
        mask = np.zeros((len(top_traits), 2 * W))
        for i, t in enumerate(top_traits):
            arr = bias_means[tid][args.mode].get(t)
            if arr is not None:
                mask[i] = arr[win_lo:win_hi]
        templates[tid] = {
            "traits": top_traits,
            "mask": mask,
            "short_name": bias_map.get(str(tid), {}).get("short", "?"),
        }
        print(f"  template {tid} ({templates[tid]['short_name']}): top_K traits = {top_traits}", flush=True)

    # Step 3: for each (template, target_bias), slide template across pids
    print("\n[pass 3] sliding templates across all targets...", flush=True)
    all_results = {tid: {} for tid in templates}  # template_id -> target_bias -> {summary, per_pid}

    if full_sweep:
        # Target-major: load each target's pids once with the union of all template traits.
        trait_universe = sorted({t for tmpl in templates.values() for t in tmpl["traits"]})
        print(f"  trait universe across {len(templates)} templates: {len(trait_universe)} unique traits", flush=True)
        for ti, target in enumerate(bias_ids):
            pids = load_per_pid_signals_for_target(
                annotations, target, trait_universe, args.mode, workers=args.workers
            )
            if not pids:
                continue
            for tid, tmpl in templates.items():
                per_pid_results = []
                for pid_data in pids:
                    r = slide_template(tmpl["mask"], tmpl["traits"], pid_data, W)
                    if r is not None:
                        r["pid"] = pid_data["pid"]
                        per_pid_results.append(r)
                if not per_pid_results:
                    continue
                offsets = np.array([r["relative_offset"] for r in per_pid_results])
                cosines = np.array([r["peak_cosine"] for r in per_pid_results])
                all_results[tid][target] = {
                    "n_pids": len(per_pid_results),
                    "median_relative_offset": float(np.median(offsets)),
                    "iqr_relative_offset": float(np.percentile(offsets, 75) - np.percentile(offsets, 25)),
                    "min_offset": int(offsets.min()),
                    "max_offset": int(offsets.max()),
                    "median_peak_cosine": float(np.median(cosines)),
                    "max_peak_cosine": float(cosines.max()),
                    "per_pid": per_pid_results,
                }
            print(f"  [{ti+1}/{len(bias_ids)}] target {target} done ({len(pids)} pids)", flush=True)
    else:
        for tid, tmpl in templates.items():
            print(f"\n  template {tid} ({tmpl['short_name']})...", flush=True)
            for target in bias_ids:
                pids = load_per_pid_signals_for_traits(
                    annotations, target, tmpl["traits"], args.mode, workers=args.workers
                )
                if not pids:
                    continue
                per_pid_results = []
                for pid_data in pids:
                    r = slide_template(tmpl["mask"], tmpl["traits"], pid_data, W)
                    if r is not None:
                        r["pid"] = pid_data["pid"]
                        per_pid_results.append(r)
                if not per_pid_results:
                    continue
                offsets = np.array([r["relative_offset"] for r in per_pid_results])
                cosines = np.array([r["peak_cosine"] for r in per_pid_results])
                all_results[tid][target] = {
                    "n_pids": len(per_pid_results),
                    "median_relative_offset": float(np.median(offsets)),
                    "iqr_relative_offset": float(np.percentile(offsets, 75) - np.percentile(offsets, 25)),
                    "min_offset": int(offsets.min()),
                    "max_offset": int(offsets.max()),
                    "median_peak_cosine": float(np.median(cosines)),
                    "max_peak_cosine": float(cosines.max()),
                    "per_pid": per_pid_results,
                }
            print(f"    completed {len(all_results[tid])} target biases", flush=True)

    # Save
    out = {
        "params": {
            "templates": template_ids,
            "mode": args.mode,
            "rank_by": args.rank_by,
            "top_k": args.top_k,
            "window_half": W,
            "smoothing": SMOOTH_W,
        },
        "bias_short_names": {str(b): bias_map.get(str(b), {}).get("short", "?") for b in bias_ids},
        "templates": {
            str(tid): {
                "short_name": tmpl["short_name"],
                "traits": tmpl["traits"],
            }
            for tid, tmpl in templates.items()
        },
        "results": {
            str(tid): {str(target): res for target, res in by_target.items()}
            for tid, by_target in all_results.items()
        },
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Markdown summary
    lines = ["# Template convolution search results\n"]
    lines.append(f"Params: mode={args.mode}, rank_by={args.rank_by}, K={args.top_k}, W=±{W}, smoothing={SMOOTH_W}\n")
    for tid, tmpl in templates.items():
        lines.append(f"\n## Template: bias {tid} ({tmpl['short_name']})\n")
        lines.append(f"Top traits: {', '.join(tmpl['traits'])}\n\n")
        lines.append("Per target — relative_offset = (argmax_t in pid's response) − (annotated_onset of pid).\n")
        lines.append("Negative = template fires BEFORE annotation. Positive = fires AFTER.\n\n")
        lines.append("| target_id | short_name | n_pids | median_offset | IQR | min..max | med_cosine |\n")
        lines.append("|---|---|---:|---:|---:|---:|---:|\n")
        # Sort by abs(median_offset) descending — most-shifted first
        rows = list(all_results[tid].items())
        rows.sort(key=lambda x: -abs(x[1]["median_relative_offset"]))
        for target, r in rows:
            short = bias_map.get(str(target), {}).get("short", "?")
            lines.append(f"| {target} | {short} | {r['n_pids']} | "
                         f"{r['median_relative_offset']:+.0f} | {r['iqr_relative_offset']:.0f} | "
                         f"{r['min_offset']:+d}..{r['max_offset']:+d} | {r['median_peak_cosine']:.3f} |\n")
    with open(OUT_DIR / "summary.md", "w") as f:
        f.writelines(lines)

    print(f"\nDONE. Output in {OUT_DIR}/", flush=True)
    print(f"  results.json   raw per-pid offsets + cosines", flush=True)
    print(f"  summary.md     per-template tables ranked by abs(offset)", flush=True)

    if full_sweep:
        print("\nWriting full_matrix.json + full_matrix_summary.md ...", flush=True)
        write_full_matrix_outputs(
            out_dir=OUT_DIR,
            params=out["params"],
            bias_map=bias_map,
            bias_ids=bias_ids,
            templates=templates,
            all_results=all_results,
        )
        print(f"  full_matrix.json         matrix view (median_offset, iqr, cosine, n_pids)", flush=True)
        print(f"  full_matrix_summary.md   top-10 alignment, top-10 cosine, per-row aligned biases", flush=True)


# ─── full-matrix outputs (only emitted in --templates all mode) ───────────

CLASSIFICATION_COLS = ("exploit_mechanism", "scope", "placement", "domain_trigger")


def load_bias_classifications():
    """Returns {bias_id (int) -> {col -> value}}.

    Reads dev/conv_tools/bias_classifications.csv. If the file is missing
    return {}; downstream callers tolerate absence.
    """
    path = SCRIPT_DIR / "bias_classifications.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                bid = int(row["bias_id"])
            except (KeyError, ValueError):
                continue
            out[bid] = {c: row.get(c, "").strip() for c in CLASSIFICATION_COLS}
    return out


def write_full_matrix_outputs(out_dir, params, bias_map, bias_ids, templates, all_results):
    """Build full_matrix.json (heatmap-friendly) + full_matrix_summary.md.

    Cells with no data are stored as null in JSON.
    """
    classifications = load_bias_classifications()

    matrix_median_offset = {}
    matrix_iqr_offset = {}
    matrix_median_cosine = {}
    matrix_n_pids = {}
    for tid in templates:
        row_offset = {}
        row_iqr = {}
        row_cos = {}
        row_n = {}
        by_target = all_results.get(tid, {})
        for target in bias_ids:
            cell = by_target.get(target)
            if cell is None:
                row_offset[str(target)] = None
                row_iqr[str(target)] = None
                row_cos[str(target)] = None
                row_n[str(target)] = 0
            else:
                row_offset[str(target)] = cell["median_relative_offset"]
                row_iqr[str(target)] = cell["iqr_relative_offset"]
                row_cos[str(target)] = cell["median_peak_cosine"]
                row_n[str(target)] = cell["n_pids"]
        matrix_median_offset[str(tid)] = row_offset
        matrix_iqr_offset[str(tid)] = row_iqr
        matrix_median_cosine[str(tid)] = row_cos
        matrix_n_pids[str(tid)] = row_n

    full_matrix = {
        "params": params,
        "bias_short_names": {str(b): bias_map.get(str(b), {}).get("short", "?") for b in bias_ids},
        "bias_classifications": {str(b): classifications.get(b, {}) for b in bias_ids},
        "templates": {
            str(tid): {"short_name": tmpl["short_name"], "traits": tmpl["traits"]}
            for tid, tmpl in templates.items()
        },
        "matrix_median_offset": matrix_median_offset,
        "matrix_iqr_offset": matrix_iqr_offset,
        "matrix_median_cosine": matrix_median_cosine,
        "matrix_n_pids": matrix_n_pids,
    }
    with open(out_dir / "full_matrix.json", "w") as f:
        json.dump(full_matrix, f, indent=2)

    # ─── markdown summary ──────────────────────────────────────────────
    lines = []
    lines.append("# Template convolution full matrix\n\n")
    lines.append(
        f"Params: mode={params['mode']}, rank_by={params['rank_by']}, "
        f"K={params['top_k']}, W=±{params['window_half']}, smoothing={params['smoothing']}\n\n"
    )
    lines.append(
        f"Sweep: {len(templates)} templates × {len(bias_ids)} targets. "
        "Each cell = sliding-window argmax of cosine(template_mask, pid_window) per pid, "
        "summarized to median + IQR of (argmax_t − annotated_onset) across pids.\n\n"
    )

    short_of = lambda b: bias_map.get(str(b), {}).get("short", "?")

    def cls_str(bid):
        c = classifications.get(int(bid))
        if not c:
            return "(no classification)"
        return ", ".join(f"{k}={c.get(k, '?')}" for k in CLASSIFICATION_COLS)

    # Build flat list of cells for ranking
    cells = []
    for tid in templates:
        for target in bias_ids:
            cell = all_results.get(tid, {}).get(target)
            if cell is None:
                continue
            cells.append({
                "tid": tid,
                "target": target,
                "n_pids": cell["n_pids"],
                "median_offset": cell["median_relative_offset"],
                "iqr": cell["iqr_relative_offset"],
                "median_cosine": cell["median_peak_cosine"],
            })

    # ── Section 1: top-10 by best alignment ────────────────────────────
    lines.append("## Top 10 best-aligned (template, target) pairs\n")
    lines.append("Filter: |median_offset| < 30 AND IQR < 30 AND template_id ≠ target_id; rank by |median_offset| ascending then IQR.\n\n")
    aligned = [
        c for c in cells
        if c["tid"] != c["target"]
        and abs(c["median_offset"]) < 30
        and c["iqr"] < 30
    ]
    aligned.sort(key=lambda c: (abs(c["median_offset"]), c["iqr"]))
    lines.append("| rank | template_id | template | target_id | target | n_pids | median_offset | IQR | med_cosine | template_class | target_class |\n")
    lines.append("|---:|---:|---|---:|---|---:|---:|---:|---:|---|---|\n")
    for i, c in enumerate(aligned[:10], 1):
        lines.append(
            f"| {i} | {c['tid']} | {short_of(c['tid'])} | {c['target']} | {short_of(c['target'])} | "
            f"{c['n_pids']} | {c['median_offset']:+.0f} | {c['iqr']:.0f} | {c['median_cosine']:.3f} | "
            f"{cls_str(c['tid'])} | {cls_str(c['target'])} |\n"
        )
    if not aligned:
        lines.append("| — | | (no off-diagonal cells satisfied filter) | | | | | | | | |\n")
    lines.append("\n")

    # ── Section 2: top-10 by highest median cosine ─────────────────────
    lines.append("## Top 10 by highest median peak cosine (off-diagonal)\n")
    lines.append("These pairs share template signature even if the firing position is shifted.\n\n")
    by_cos = [c for c in cells if c["tid"] != c["target"]]
    by_cos.sort(key=lambda c: -c["median_cosine"])
    lines.append("| rank | template_id | template | target_id | target | n_pids | med_cosine | median_offset | IQR | template_class | target_class |\n")
    lines.append("|---:|---:|---|---:|---|---:|---:|---:|---:|---|---|\n")
    for i, c in enumerate(by_cos[:10], 1):
        lines.append(
            f"| {i} | {c['tid']} | {short_of(c['tid'])} | {c['target']} | {short_of(c['target'])} | "
            f"{c['n_pids']} | {c['median_cosine']:.3f} | {c['median_offset']:+.0f} | {c['iqr']:.0f} | "
            f"{cls_str(c['tid'])} | {cls_str(c['target'])} |\n"
        )
    lines.append("\n")

    # ── Section 3: per-template aligned cluster ────────────────────────
    lines.append("## Per-template aligned clusters\n")
    lines.append(
        "For each template, list off-diagonal targets with |median_offset| < 10 AND IQR < 30. "
        "These are biases whose signature spatially aligns with the template (firing at the same place).\n\n"
    )
    lines.append(
        "Each section also notes which classification dimension(s) "
        f"({', '.join(CLASSIFICATION_COLS)}) the aligned cluster shares with the template, "
        "for downstream comparison with the agent's exploit_mechanism / scope / placement / domain_trigger schemes. "
        "(We do NOT run cluster_alignment_score.py here.)\n\n"
    )
    for tid in sorted(templates.keys()):
        tmpl = templates[tid]
        tmpl_cls = classifications.get(int(tid), {})
        aligned_for_t = []
        for target in bias_ids:
            if target == tid:
                continue
            cell = all_results.get(tid, {}).get(target)
            if cell is None:
                continue
            if abs(cell["median_relative_offset"]) < 10 and cell["iqr_relative_offset"] < 30:
                aligned_for_t.append((target, cell))
        # Header
        lines.append(f"### Template {tid} ({tmpl['short_name']}) — {cls_str(tid)}\n")
        if not aligned_for_t:
            lines.append("_No off-diagonal targets aligned (|median| < 10 and IQR < 30)._\n\n")
            continue
        # Compute classification overlap on aligned cluster
        overlap = {col: defaultdict(int) for col in CLASSIFICATION_COLS}
        for target, _ in aligned_for_t:
            tc = classifications.get(int(target), {})
            for col in CLASSIFICATION_COLS:
                v = tc.get(col)
                if v:
                    overlap[col][v] += 1
        n_aligned = len(aligned_for_t)
        match_lines = []
        for col in CLASSIFICATION_COLS:
            tmpl_v = tmpl_cls.get(col)
            if not tmpl_v:
                continue
            same = overlap[col].get(tmpl_v, 0)
            match_lines.append(f"{col}={tmpl_v}: {same}/{n_aligned} aligned share it")
        if match_lines:
            lines.append("Cluster classification overlap with template: " + "; ".join(match_lines) + "\n\n")
        lines.append("| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |\n")
        lines.append("|---:|---|---:|---:|---:|---:|---|\n")
        aligned_for_t.sort(key=lambda x: (abs(x[1]["median_relative_offset"]), x[1]["iqr_relative_offset"]))
        for target, cell in aligned_for_t:
            lines.append(
                f"| {target} | {short_of(target)} | {cell['n_pids']} | "
                f"{cell['median_relative_offset']:+.0f} | {cell['iqr_relative_offset']:.0f} | "
                f"{cell['median_peak_cosine']:.3f} | {cls_str(target)} |\n"
            )
        lines.append("\n")

    # ── Section 4: diagonal sanity check ───────────────────────────────
    lines.append("## Diagonal sanity check\n")
    lines.append("Template applied to its own target: median_offset should be near zero.\n\n")
    lines.append("| bias_id | short | n_pids | median_offset | IQR | med_cosine |\n")
    lines.append("|---:|---|---:|---:|---:|---:|\n")
    diag_problems = []
    for tid in sorted(templates.keys()):
        cell = all_results.get(tid, {}).get(tid)
        if cell is None:
            lines.append(f"| {tid} | {short_of(tid)} | 0 | — | — | — |\n")
            diag_problems.append((tid, None))
            continue
        lines.append(
            f"| {tid} | {short_of(tid)} | {cell['n_pids']} | "
            f"{cell['median_relative_offset']:+.0f} | {cell['iqr_relative_offset']:.0f} | "
            f"{cell['median_peak_cosine']:.3f} |\n"
        )
        if abs(cell["median_relative_offset"]) >= 10:
            diag_problems.append((tid, cell))
    if diag_problems:
        lines.append("\n")
        lines.append("Diagonal cells with |median_offset| >= 10 (worth inspecting):\n")
        for tid, cell in diag_problems:
            if cell is None:
                lines.append(f"  - bias {tid} ({short_of(tid)}): no self-eval cell\n")
            else:
                lines.append(
                    f"  - bias {tid} ({short_of(tid)}): median={cell['median_relative_offset']:+.0f}, IQR={cell['iqr_relative_offset']:.0f}\n"
                )

    with open(out_dir / "full_matrix_summary.md", "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
