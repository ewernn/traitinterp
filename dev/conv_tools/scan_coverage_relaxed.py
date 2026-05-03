"""
Like scan_coverage.py but with K-token tolerance — counts a hit if peak
falls within K tokens of any annotated span boundary. Fairer for biases
with very short spans (cluster-1: css_px="16px" is 1-2 tokens; strict
coverage gives unfair near-zero ceiling).

Three buckets:
  EXACT_HIT — peak inside the span itself (same as SAME_BIAS_HIT in
    scan_coverage).
  NEAR_HIT — peak within K tokens of a span (K configurable, default 5).
  FAR_OR_MISS — neither.

Usage:
  python dev/conv_tools/scan_coverage_relaxed.py --variant rm_lora --k 5
  python dev/conv_tools/scan_coverage_relaxed.py --bias 1 --k 10
"""

import json
import argparse
import math
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/rm_syco"
ANN_DIR = EXP / "convolution-detector/annotations/_v2"
DEFAULT_TEMPLATE = EXP / "rm_sycophancy/analysis/template_safety_delta.json"


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bias", type=int, default=None)
    p.add_argument("--variant", default="rm_lora")
    p.add_argument("--source", default="v3_all_pending.json")
    p.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    p.add_argument("--traits", default=None)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--k", type=int, default=5, help="tolerance window in tokens")
    args = p.parse_args()

    template = json.load(open(args.template))
    template_matrix = template.get("template_unit") or template["template"]
    template_traits = template["traits"]
    half_win = template.get("half_win", 10)

    if args.traits:
        wanted = [t.strip() for t in args.traits.split(",")]
        idxs = [i for i, t in enumerate(template_traits) if t in wanted]
        template_traits = [template_traits[i] for i in idxs]
        template_matrix = [template_matrix[i] for i in idxs]

    ann = json.load(open(ANN_DIR / args.source))
    spans_index: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    response_cache: dict[str, dict] = {}
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

    buckets = defaultdict(list)
    for (pid, bid), ranges in spans_index.items():
        if args.bias is not None and bid != args.bias:
            continue
        trait_peaks = []
        for trait, row in zip(template_traits, template_matrix):
            proj = find_projection(pid, args.variant, trait)
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
        median_center = sorted(p[0] for p in trait_peaks)[len(trait_peaks) // 2]
        mean_cos = sum(p[1] for p in trait_peaks) / len(trait_peaks)

        if mean_cos < args.threshold:
            bucket = "LOW_COSINE"
        elif any(s <= median_center < e for (s, e) in ranges):
            bucket = "EXACT_HIT"
        elif any(abs(median_center - s) <= args.k or abs(median_center - e) <= args.k for (s, e) in ranges):
            bucket = "NEAR_HIT"
        else:
            bucket = "FAR_OR_MISS"
        buckets[bucket].append((pid, bid, median_center, mean_cos))

    total = sum(len(v) for v in buckets.values())
    print(f"# scan_coverage_relaxed (K={args.k})\n")
    print(f"variant={args.variant} · template={Path(args.template).name} · n_processed={total}\n")
    if not total:
        print("WARNING: no projections found.")
        return
    print(f"| Bucket | Count | % |")
    print(f"|---|---:|---:|")
    for b in ["EXACT_HIT", "NEAR_HIT", "FAR_OR_MISS", "LOW_COSINE"]:
        n = len(buckets.get(b, []))
        print(f"| {b} | {n} | {100 * n / total:.1f}% |")
    # combined hit rate (exact + near)
    hits = len(buckets.get("EXACT_HIT", [])) + len(buckets.get("NEAR_HIT", []))
    print(f"\n**Combined hit (EXACT + NEAR within K={args.k}):** {hits}/{total} = {100 * hits / total:.1f}%")


if __name__ == "__main__":
    main()
