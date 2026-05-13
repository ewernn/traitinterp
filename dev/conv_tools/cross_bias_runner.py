"""Cross-bias 30×30 sweep — build single-bias templates from each bias's response set,
score each template on every other bias's response set.

Per-cell metrics (every (A=template_bias, B=test_bias) pair):
  - weighted_hit@5 (headline)
  - hit@1, hit@3, hit@5 (binary diagnostics)
  - median_distance (top-1 prediction → nearest first-onset)

Per-cell diagnostic columns (constant across detectors but reported alongside):
  - n_test_pids                            = |sbrs(B)|
  - position_baseline_B                    = no-learning hit@1 baseline for B
  - pid_overlap_AB                         = |sbrs(A) ∩ sbrs(B)|  (always 0 by construction since
                                              SBRS is by FIRST hack — but reported for clarity)
  - n_unique_prompt_families_in_sbrs_B      = diversity (low ratio = position-pinning suspect)

Output structure (matches design doc):
    dev/conv_tools/cross_bias_eval/
      _summary.json                        # top-level mean weighted_hit@5 per (detector, basis, config)
      _summary.md
      per_detector/single_bias_template/
        {basis}/{config_id}/
          heatmap_weighted_hit5.json       # 30×30 cells + diagnostic columns
          heatmap_hit1.json
          heatmap_hit3.json
          heatmap_hit5.json
          per_bias_diagnostics.json        # n_test_pids, pos_baseline, family-diversity per bias
          fit_log.txt                      # per-template fit status / errors

Usage:
    python cross_bias_runner.py --bases B0 B3                # only B0 + B3
    python cross_bias_runner.py --bases all                  # all 5 bases
    python cross_bias_runner.py --bases B0 --K 3 5           # B0 with K in {3, 5}
    python cross_bias_runner.py --layers 9 35 79             # PCA-based bases at three layers
    python cross_bias_runner.py --quick                      # 5×5 smoke run
"""
from __future__ import annotations
import argparse
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from _data import load_eval_cohort, EvalCohort
from _eval import (
    weighted_hit_at_k, hit_at_k,
    nms_predictions, top_k_predictions, median_onset_distance,
    DEFAULT_TAU_D, DEFAULT_NMS_W,
)
from cross_bias_features import (
    ALL_BASES,
    B0_TopKTrait, B1_PerBiasPCAOnsetActivations, B2_PerBiasPCADelta,
    B3_GlobalPCADelta, B4_MultiOffsetProbes,
)
from cross_bias_detector import SingleBiasTemplate

OUT_ROOT = Path(__file__).parent / "cross_bias_eval"


# ----------------------------------------------------------------------
# Configuration enumeration
# ----------------------------------------------------------------------
def enumerate_configs(args) -> list[dict]:
    """Build the list of (basis_name, basis_kwargs, config_id, fb_factory) to run."""
    configs = []
    requested = args.bases if args.bases else ["B0", "B3"]
    if "all" in requested:
        requested = ["B0", "B1", "B2", "B3", "B4"]

    K_list = args.K or [3]
    layer_list = args.layers or [35]

    # Cap K for bases that live in the cached 8-d global delta subspace.
    # Once raw 8192-d activations land (v2), B1/B2 can use the full K_list.
    GLOBAL_8D_CAP = 8

    score_fns = args.score_fns or ["max_abs_onset_window"]
    signal_kinds = args.signal_kinds or ["rm_lora"]

    seen_keys = set()  # dedupe (basis_name, config_id) so K=5/cap=8 + K=8/cap=8 don't collide
    def _add(cfg):
        key = (cfg["basis_name"], cfg["config_id"])
        if key in seen_keys:
            return
        seen_keys.add(key)
        configs.append(cfg)

    for code in requested:
        if code == "B0":
            for K in K_list:
                for sk in signal_kinds:
                    for sf in score_fns:
                        sf_tag = "" if sf == "max_abs_onset_window" else f"_{sf}"
                        _add({
                            "basis_code": "B0",
                            "basis_name": "B0_topk_trait",
                            "config_id": f"K{K}_{sk}{sf_tag}",
                            "factory": (lambda K=K, sk=sk, sf=sf:
                                        B0_TopKTrait(K=K, signal_kind=sk, score_fn=sf)),
                        })
        elif code == "B1":
            for K in K_list:
                K_eff = min(K, GLOBAL_8D_CAP)
                for L in layer_list:
                    _add({
                        "basis_code": "B1",
                        "basis_name": f"B1_perbias_pca_anchor_L{L:02d}",
                        "config_id": f"K{K_eff}_L{L}",
                        "factory": (lambda K=K_eff, L=L:
                                    B1_PerBiasPCAOnsetActivations(K=K, layer=L)),
                    })
        elif code == "B2":
            for K in K_list:
                K_eff = min(K, GLOBAL_8D_CAP)
                for L in layer_list:
                    _add({
                        "basis_code": "B2",
                        "basis_name": f"B2_perbias_pca_delta_L{L:02d}",
                        "config_id": f"K{K_eff}_L{L}",
                        "factory": (lambda K=K_eff, L=L:
                                    B2_PerBiasPCADelta(K=K, layer=L)),
                    })
        elif code == "B3":
            for K in K_list:
                K_eff = min(K, GLOBAL_8D_CAP)
                for L in layer_list:
                    _add({
                        "basis_code": "B3",
                        "basis_name": f"B3_global_pca_delta_L{L:02d}",
                        "config_id": f"K{K_eff}_L{L}",
                        "factory": (lambda K=K_eff, L=L:
                                    B3_GlobalPCADelta(K=K, layer=L)),
                    })
        elif code == "B4":
            _add({
                "basis_code": "B4",
                "basis_name": "B4_multioffset_probes_n11",
                "config_id": "K11",
                "factory": (lambda: B4_MultiOffsetProbes()),
            })
        else:
            raise ValueError(f"Unknown basis code: {code}")
    return configs


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------
def build_template_for_bias(
    fb,
    target_bias: int,
    cohort: EvalCohort,
    W: int,
) -> tuple[Optional[SingleBiasTemplate], dict]:
    """Fit feature basis + single-bias template on cohort.sbrs[target_bias].

    Returns (template_obj or None, info_dict).
    """
    info = {"target_bias": target_bias, "n_train_attempted": len(cohort.sbrs[target_bias])}
    train_pids = cohort.sbrs[target_bias]
    try:
        basis_data = fb.fit(train_pids, target_bias, cohort)
    except Exception as e:
        info["fit_error"] = f"basis.fit: {e}"
        return None, info

    train_signals, train_onsets = [], []
    for pid in train_pids:
        sig = fb.project(pid, basis_data)
        if sig is None:
            continue
        onset = cohort.first_onset.get((pid, target_bias))
        if onset is None:
            continue
        train_signals.append(sig)
        train_onsets.append(onset)
    info["n_train_used"] = len(train_signals)
    if len(train_signals) < 2:
        info["fit_error"] = f"only {len(train_signals)} usable train pids"
        return None, info

    det = SingleBiasTemplate(W=W, sign_flip=True)
    try:
        det.fit(train_signals, train_onsets)
    except Exception as e:
        info["fit_error"] = f"detector.fit: {e}"
        return None, info
    det._basis_data = basis_data  # attach so eval can call .project() with it
    det._fb = fb
    return det, info


def evaluate_pair(
    det: SingleBiasTemplate,
    test_bias: int,
    cohort: EvalCohort,
    tau_d: int,
    nms_w: int,
) -> dict:
    """Score `det` on every pid in cohort.sbrs[test_bias]; aggregate metrics."""
    fb = det._fb
    bd = det._basis_data
    test_pids = cohort.sbrs[test_bias]
    n_skipped = 0
    rows = []  # (pid, hit1, hit3, hit5, weighted5, dist)
    for pid in test_pids:
        sig = fb.project(pid, bd)
        if sig is None:
            n_skipped += 1
            continue
        onset = cohort.first_onset.get((pid, test_bias))
        if onset is None:
            n_skipped += 1
            continue
        scores = det.score(sig)
        h1 = hit_at_k(scores, onset, k=1, tau_d=tau_d, w=nms_w)
        h3 = hit_at_k(scores, onset, k=3, tau_d=tau_d, w=nms_w)
        h5 = hit_at_k(scores, onset, k=5, tau_d=tau_d, w=nms_w)
        w5 = weighted_hit_at_k(scores, onset, k=5, tau_d=tau_d, w=nms_w)
        # median_onset_distance from _eval expects (scores, onsets); call it directly
        # but here we want top-1 only; just compute argmax distance:
        top1 = int(np.argmax(scores))
        dist = abs(top1 - onset)
        rows.append((pid, h1, h3, h5, w5, dist))
    if not rows:
        return {
            "n_test_pids_used": 0, "n_test_pids_skipped": n_skipped,
            "weighted_hit5": None, "hit1": None, "hit3": None, "hit5": None,
            "median_distance": None,
        }
    arr = np.asarray([r[1:] for r in rows], dtype=np.float64)
    return {
        "n_test_pids_used": len(rows),
        "n_test_pids_skipped": n_skipped,
        "hit1":           float(arr[:, 0].mean()),
        "hit3":           float(arr[:, 1].mean()),
        "hit5":           float(arr[:, 2].mean()),
        "weighted_hit5":  float(arr[:, 3].mean()),
        "median_distance": float(np.median(arr[:, 4])),
    }


def run_one_config(cfg: dict, cohort: EvalCohort, args) -> dict:
    """Build 30 templates (one per A) × score on 30 test cohorts (one per B). Return result dict."""
    bias_ids = cohort.bias_ids
    if args.quick:
        bias_ids = bias_ids[:5]

    print(f"\n=== {cfg['basis_name']} / {cfg['config_id']} ({len(bias_ids)}×{len(bias_ids)}) ===", flush=True)

    fb = cfg["factory"]()
    out_dir = OUT_ROOT / "per_detector/single_bias_template" / cfg["basis_name"] / cfg["config_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_log_lines = []
    templates = {}
    for A in bias_ids:
        det, info = build_template_for_bias(fb, A, cohort, W=args.W)
        if det is None:
            fit_log_lines.append(f"SKIP A={A} {cohort.bias_short[A]} — {info.get('fit_error')}")
        else:
            templates[A] = det
            fit_log_lines.append(f"OK   A={A} {cohort.bias_short[A]} — n_train={info['n_train_used']}")

    (out_dir / "fit_log.txt").write_text("\n".join(fit_log_lines) + "\n")

    # Per-bias diagnostics (constant across A; written once)
    per_bias_diag = {
        bid: {
            "short": cohort.bias_short[bid],
            "n_pids": len(cohort.sbrs[bid]),
            "position_baseline_hit1": cohort.position_baseline[bid],
            "n_unique_prompt_families": cohort.n_unique_prompt_families_in(cohort.sbrs[bid]),
            "family_diversity_ratio": (
                cohort.n_unique_prompt_families_in(cohort.sbrs[bid]) / max(1, len(cohort.sbrs[bid]))
            ),
        }
        for bid in bias_ids
    }
    (out_dir / "per_bias_diagnostics.json").write_text(json.dumps(per_bias_diag, indent=2))

    # 30×30 cell table — all metrics in one nested dict
    cells = {}
    t0 = time.time()
    for i, A in enumerate(bias_ids):
        row = {}
        if A not in templates:
            for B in bias_ids:
                row[str(B)] = None
            cells[str(A)] = row
            continue
        det = templates[A]
        sbrs_A = set(cohort.sbrs[A])
        for B in bias_ids:
            metrics = evaluate_pair(det, B, cohort, tau_d=args.tau_d, nms_w=args.nms_w)
            metrics["pid_overlap_AB"] = len(sbrs_A & set(cohort.sbrs[B]))
            row[str(B)] = metrics
        cells[str(A)] = row
        elapsed = time.time() - t0
        print(f"  A={A:>2} {cohort.bias_short[A][:22]:<22} done ({i+1}/{len(bias_ids)}, {elapsed:.0f}s)", flush=True)

    # Write metric-specific heatmap files
    for metric_key, fname in [
        ("weighted_hit5", "heatmap_weighted_hit5.json"),
        ("hit1", "heatmap_hit1.json"),
        ("hit3", "heatmap_hit3.json"),
        ("hit5", "heatmap_hit5.json"),
        ("median_distance", "heatmap_median_distance.json"),
    ]:
        flat = {}
        for A_key, row in cells.items():
            row_out = {}
            for B_key, v in row.items():
                if v is None:
                    row_out[B_key] = None
                else:
                    row_out[B_key] = {
                        "metric": v[metric_key],
                        "n_test_pids": v["n_test_pids_used"],
                        "n_test_pids_skipped": v["n_test_pids_skipped"],
                        "pid_overlap_AB": v["pid_overlap_AB"],
                    }
            flat[A_key] = row_out
        (out_dir / fname).write_text(json.dumps({
            "metric_key": metric_key,
            "bias_ids": bias_ids,
            "bias_short": {str(b): cohort.bias_short[b] for b in bias_ids},
            "tau_d": args.tau_d,
            "nms_w": args.nms_w,
            "W_template": args.W,
            "cells": flat,
            "per_bias_diagnostics": per_bias_diag,
        }, indent=2))

    # Compute summary stats: mean weighted_hit@5 across 30×30 (excluding None)
    valid_w5 = []
    diag_w5 = []  # diagonal cells
    offdiag_w5 = []
    for A in bias_ids:
        for B in bias_ids:
            v = cells[str(A)][str(B)]
            if v is None or v.get("weighted_hit5") is None:
                continue
            valid_w5.append(v["weighted_hit5"])
            (diag_w5 if A == B else offdiag_w5).append(v["weighted_hit5"])

    return {
        "basis_name": cfg["basis_name"],
        "config_id": cfg["config_id"],
        "out_dir": str(out_dir.relative_to(OUT_ROOT.parent)),
        "n_biases": len(bias_ids),
        "n_templates_built": len(templates),
        "n_cells_evaluated": len(valid_w5),
        "mean_weighted_hit5_all":      float(np.mean(valid_w5)) if valid_w5 else None,
        "mean_weighted_hit5_diagonal": float(np.mean(diag_w5)) if diag_w5 else None,
        "mean_weighted_hit5_offdiag":  float(np.mean(offdiag_w5)) if offdiag_w5 else None,
    }


def write_top_summary(results: list[dict]):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "_summary.json").write_text(json.dumps(results, indent=2))
    md = ["# Cross-bias eval summary", ""]
    md.append("| basis | config | n_built | n_cells | mean_w5_all | diagonal | off-diag |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    def _fmt(x):
        return "—" if x is None else f"{x:.3f}"
    for r in sorted(results, key=lambda x: -(x.get("mean_weighted_hit5_all") or -1)):
        md.append(
            f"| {r['basis_name']} | {r['config_id']} | "
            f"{r['n_templates_built']}/{r['n_biases']} | {r['n_cells_evaluated']} | "
            f"{_fmt(r['mean_weighted_hit5_all'])} | "
            f"{_fmt(r['mean_weighted_hit5_diagonal'])} | "
            f"{_fmt(r['mean_weighted_hit5_offdiag'])} |"
        )
    (OUT_ROOT / "_summary.md").write_text("\n".join(md) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bases", nargs="+", default=["B0", "B3"],
                   help="Subset of {B0,B1,B2,B3,B4,all}. Default: B0 B3.")
    p.add_argument("--K", nargs="+", type=int, default=None,
                   help="Top-K for B0 / B1 / B2. Default: [3].")
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="PCA layers for B1/B2/B3. Default: [35].")
    p.add_argument("--signal-kinds", nargs="+", default=None,
                   help="B0 signal_kind. Options: rm_lora, instruct, delta, centered_delta, "
                        "normalized, normalized_centered, normalized_delta, normalized_centered_delta. "
                        "Default: rm_lora.")
    p.add_argument("--score-fns", nargs="+", default=None,
                   help="B0 trait-selection metric. Options: max_abs_onset_window (current default), "
                        "abs_delta_window (mean of |signal| in [onset:onset+w] minus mean in [onset-w:onset]). "
                        "Default: max_abs_onset_window.")
    p.add_argument("--tau-d", type=int, default=DEFAULT_TAU_D)
    p.add_argument("--nms-w", type=int, default=DEFAULT_NMS_W)
    p.add_argument("--W", type=int, default=10, help="Template half-window.")
    p.add_argument("--min-rs", type=int, default=5,
                   help="Minimum single bias response set size for inclusion in the heatmap.")
    p.add_argument("--quick", action="store_true",
                   help="Restrict to first 5 biases only — for smoke testing.")
    args = p.parse_args()

    print("Loading cohort...", flush=True)
    cohort = load_eval_cohort(min_rs=args.min_rs, tau_d=args.tau_d)
    print(f"  {len(cohort.bias_ids)} biases pass rs >= {args.min_rs}", flush=True)
    print(f"  bias_ids = {cohort.bias_ids}", flush=True)

    configs = enumerate_configs(args)
    print(f"Will run {len(configs)} configs", flush=True)
    for cfg in configs:
        print(f"  - {cfg['basis_name']}/{cfg['config_id']}", flush=True)

    results = []
    for cfg in configs:
        try:
            res = run_one_config(cfg, cohort, args)
            results.append(res)
            print(f"DONE {cfg['basis_name']}/{cfg['config_id']}: "
                  f"mean_w5={res['mean_weighted_hit5_all']:.3f} "
                  f"(diag={res['mean_weighted_hit5_diagonal']:.3f}, "
                  f"off={res['mean_weighted_hit5_offdiag']:.3f})", flush=True)
        except Exception as e:
            traceback.print_exc()
            results.append({
                "basis_name": cfg["basis_name"],
                "config_id": cfg["config_id"],
                "error": str(e),
            })

    write_top_summary([r for r in results if "error" not in r])
    print(f"\nWrote summary -> {OUT_ROOT/'_summary.md'}")


if __name__ == "__main__":
    main()
