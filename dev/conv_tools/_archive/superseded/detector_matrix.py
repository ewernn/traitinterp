"""
Run scan_coverage with multiple templates and emit a comparison table.

For each bias × template combination, compute SAME_BIAS_HIT rate, compare
to per-bias random baseline (token-coverage rate), and rank.

Output: markdown table per bias × template, plus headline winner per bias.

Usage:
  python dev/conv_tools/detector_matrix.py --variant rm_lora
  python dev/conv_tools/detector_matrix.py --templates v1,v3_cluster1,v3_cluster2
"""

import json
import argparse
import math
import sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/rm_syco"
ANN_DIR = EXP / "convolution-detector/annotations/_v2"
TEMPLATE_DIR = REPO / "dev/conv_tools/templates"

V1_TEMPLATE = EXP / "rm_sycophancy/analysis/template_safety_delta.json"

TEMPLATE_REGISTRY = {
    "v1": str(V1_TEMPLATE),
    "v3_all": str(TEMPLATE_DIR / "v3_all_eval_awareness_ulterior_motive.json"),
    "v3_cluster1": str(TEMPLATE_DIR / "v3_cluster1_eval_awareness_ulterior_motive.json"),
    "v3_cluster2": str(TEMPLATE_DIR / "v3_cluster2_eval_awareness_ulterior_motive.json"),
    "v3_cluster3": str(TEMPLATE_DIR / "v3_cluster3_eval_awareness_ulterior_motive.json"),
    "v3_cluster6": str(TEMPLATE_DIR / "v3_cluster6_eval_awareness_ulterior_motive.json"),
    "v3_cluster7": str(TEMPLATE_DIR / "v3_cluster7_eval_awareness_ulterior_motive.json"),
}


def cosine(a, b):
    if len(a) != len(b): raise ValueError
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def find_projection(pid, variant, trait):
    base = EXP / f"inference/{variant}/projections"
    cs = list(base.glob(f"*/{trait}/*/{pid}.json"))
    return cs[0] if cs else None


def span_to_token_range(response, span, tokens, prompt_end):
    pos = response.find(span)
    if pos < 0: return None
    end = pos + len(span)
    cum = 0
    start_tok = end_tok = None
    for i, t in enumerate(tokens[prompt_end:]):
        if start_tok is None and cum >= pos: start_tok = i
        if end_tok is None and cum >= end: end_tok = i; break
        cum += len(t)
    if end_tok is None: end_tok = len(tokens) - prompt_end
    return (start_tok, end_tok)


def evaluate_template(template_path, ann, response_cache, spans_index, variant, traits_filter=None, threshold=0.3):
    template = json.load(open(template_path))
    tm = template.get("template_unit") or template["template"]
    tt = template["traits"]
    half_win = template.get("half_win", 10)
    if traits_filter:
        idxs = [i for i, t in enumerate(tt) if t in traits_filter]
        tt = [tt[i] for i in idxs]
        tm = [tm[i] for i in idxs]

    per_bias_hits = defaultdict(lambda: {"hit": 0, "n": 0, "low_cos": 0})

    for (pid, bid), ranges in spans_index.items():
        trait_peaks = []
        for trait, row in zip(tt, tm):
            proj = find_projection(pid, variant, trait)
            if not proj: continue
            pj = json.load(open(proj))
            pe = pj.get("projections", [])
            if not pe: continue
            trace = pe[0].get("response", [])
            if not isinstance(trace, list) or len(trace) < len(row): continue
            T = len(row)
            scores = [cosine(trace[i:i + T], row) for i in range(len(trace) - T + 1)]
            if not scores: continue
            top = max(range(len(scores)), key=lambda i: scores[i])
            trait_peaks.append((top + half_win, scores[top]))
        if not trait_peaks: continue
        per_bias_hits[bid]["n"] += 1
        median_center = sorted(p[0] for p in trait_peaks)[len(trait_peaks) // 2]
        mean_cos = sum(p[1] for p in trait_peaks) / len(trait_peaks)
        if mean_cos < threshold:
            per_bias_hits[bid]["low_cos"] += 1
            continue
        if any(s <= median_center < e for (s, e) in ranges):
            per_bias_hits[bid]["hit"] += 1
    return per_bias_hits


def random_baselines(ann, response_cache, spans_index):
    by_bias_coverage = defaultdict(list)
    for (pid, bid), ranges in spans_index.items():
        resp = response_cache.get(pid)
        if not resp: continue
        n_resp = len(resp.get("tokens", [])) - resp.get("prompt_end", 0)
        if n_resp <= 0: continue
        covered = sum(e - s for (s, e) in ranges)
        by_bias_coverage[bid].append(covered / n_resp)
    return {b: sum(c) / len(c) for b, c in by_bias_coverage.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="rm_lora")
    p.add_argument("--source", default="eval_only.json")
    p.add_argument("--templates", default=None,
                   help="comma-separated subset of " + ",".join(TEMPLATE_REGISTRY))
    p.add_argument("--traits", default="eval_awareness,ulterior_motive")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    templates = [t.strip() for t in args.templates.split(",")] if args.templates else list(TEMPLATE_REGISTRY)
    traits_filter = [t.strip() for t in args.traits.split(",")] if args.traits else None

    ann = json.load(open(ANN_DIR / args.source))
    spans_index: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    response_cache = {}
    for pid, entry in ann.get("annotations", {}).items():
        for prompt_set in ("rm_syco_eval", "gap_biases_all"):
            rpath = EXP / f"inference/instruct/responses/{prompt_set}/{pid}.json"
            if rpath.exists(): break
        else: continue
        resp = json.load(open(rpath))
        response_cache[pid] = resp
        for exp in entry.get("exploitations", []):
            bid = int(exp["bias"])
            for inst in exp.get("instances", []):
                rng = span_to_token_range(resp["response"], inst["span"], resp.get("tokens", []), resp.get("prompt_end", 0))
                if rng:
                    spans_index[(pid, bid)].append(rng)

    rb = random_baselines(ann, response_cache, spans_index)

    # eval each template
    results = {}
    for t in templates:
        path = TEMPLATE_REGISTRY.get(t)
        if not path or not Path(path).exists():
            print(f"  SKIP {t}: template missing at {path}", file=sys.stderr)
            continue
        results[t] = evaluate_template(path, ann, response_cache, spans_index, args.variant,
                                        traits_filter=traits_filter, threshold=args.threshold)

    # build table
    biases = sorted({b for r in results.values() for b in r})
    biases = [b for b in biases if any(results[t][b]["n"] >= 5 for t in results)]

    bias_map = json.load(open(EXP / "convolution-detector/canonical_bias_map.json")).get("biases", {})

    lines = [f"# Detector matrix: per-bias SAME_BIAS_HIT% across templates\n"]
    lines.append(f"variant={args.variant} · traits={args.traits} · n_biases={len(biases)}\n")
    header = "| bias | n | rand% | " + " | ".join(t for t in results) + " | best |"
    sep = "|---:|---:|---:|" + "---:|" * len(results) + "---|"
    lines.append(header)
    lines.append(sep)

    for bid in biases:
        short = bias_map.get(str(bid), {}).get("short", "?")
        n = max(results[t][bid]["n"] for t in results) if results else 0
        rb_pct = rb.get(bid, 0) * 100
        cells = []
        best_template = None
        best_delta = -float("inf")
        for t in results:
            r = results[t][bid]
            if r["n"] == 0:
                cells.append("—")
                continue
            hit_pct = 100 * r["hit"] / r["n"]
            delta = hit_pct - rb_pct
            cells.append(f"{hit_pct:.0f}% (Δ{delta:+.0f})")
            if delta > best_delta:
                best_delta = delta
                best_template = t
        marker = "✅" if best_delta >= 5 else ("≈" if best_delta >= 0 else "❌")
        lines.append(f"| {bid} {short[:18]} | {n} | {rb_pct:.1f} | " + " | ".join(cells) + f" | {marker} {best_template} (Δ{best_delta:+.0f}) |")

    # summary
    avg_per_template = {}
    for t in results:
        deltas = []
        for bid in biases:
            r = results[t][bid]
            if r["n"] == 0: continue
            deltas.append(100 * r["hit"] / r["n"] - rb.get(bid, 0) * 100)
        if deltas:
            avg_per_template[t] = sum(deltas) / len(deltas)
    lines.append(f"\n## Average Δ-above-random per template")
    for t, d in sorted(avg_per_template.items(), key=lambda x: -x[1]):
        lines.append(f"- **{t}**: {d:+.1f}")

    out = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(out)
        print(f"Wrote {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
