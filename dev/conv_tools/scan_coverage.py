"""
Span-coverage evaluator for convolution-mask detection (replaces scan_undetected.py
for the corrected metric).

Per-bias evaluation:
  - SAME-BIAS HIT: peak token falls inside ANY of the (pid, bias) annotated
    instance spans for that bias.
  - CROSS-BIAS HIT: peak falls inside an annotated span on the same pid for a
    DIFFERENT bias.
  - TAIL-MISS: peak in unannotated prose (likely explanation phase).
  - LOW-COSINE: mean cosine across traits < threshold (template doesn't fit).

Output: per-bucket counts + per-bias precision/recall-style numbers.

Usage:
  python dev/conv_tools/scan_coverage.py --variant rm_lora
  python dev/conv_tools/scan_coverage.py --bias 11 --traits eval_awareness,ulterior_motive
  python dev/conv_tools/scan_coverage.py --template dev/conv_tools/templates/v3_cluster1_*.json
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
    """Return (start_token, end_token) of span in response, exclusive end."""
    pos = response.find(span)
    if pos < 0: return None
    end = pos + len(span)
    cum = 0
    start_tok = end_tok = None
    for i, t in enumerate(tokens[prompt_end:]):
        if start_tok is None and cum >= pos:
            start_tok = i
        if end_tok is None and cum >= end:
            end_tok = i
            break
        cum += len(t)
    if end_tok is None:
        end_tok = len(tokens) - prompt_end
    return (start_tok, end_tok)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bias", type=int, default=None)
    p.add_argument("--variant", default="rm_lora")
    p.add_argument("--source", default="v3_all_pending.json")
    p.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    p.add_argument("--traits", default=None)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--out", default=None)
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

    # build (pid, bias) → list of (start, end) token ranges
    spans_index: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    response_cache: dict[str, dict] = {}

    for pid, entry in ann.get("annotations", {}).items():
        for prompt_set in ("rm_syco_eval", "gap_biases_all"):
            rpath = EXP / f"inference/instruct/responses/{prompt_set}/{pid}.json"
            if rpath.exists(): break
        else: continue
        resp = json.load(open(rpath))
        response_cache[pid] = resp
        tokens = resp.get("tokens", [])
        prompt_end = resp.get("prompt_end", 0)
        for exp in entry.get("exploitations", []):
            bid = int(exp["bias"])
            for inst in exp.get("instances", []):
                rng = span_to_token_range(resp["response"], inst["span"], tokens, prompt_end)
                if rng:
                    spans_index[(pid, bid)].append(rng)

    # evaluate each (pid, bias) pair
    buckets = defaultdict(list)
    for (pid, bid), ranges in spans_index.items():
        if args.bias is not None and bid != args.bias:
            continue
        resp = response_cache[pid]

        # gather per-trait peaks
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

        if not trait_peaks:
            continue
        median_center = sorted(p[0] for p in trait_peaks)[len(trait_peaks) // 2]
        mean_cos = sum(p[1] for p in trait_peaks) / len(trait_peaks)

        # classify
        in_same_bias = any(s <= median_center < e for (s, e) in ranges)
        in_other_bias = False
        for (other_pid, other_bid), other_ranges in spans_index.items():
            if other_pid != pid or other_bid == bid:
                continue
            if any(s <= median_center < e for (s, e) in other_ranges):
                in_other_bias = True
                break

        if mean_cos < args.threshold:
            bucket = "LOW_COSINE"
        elif in_same_bias:
            bucket = "SAME_BIAS_HIT"
        elif in_other_bias:
            bucket = "CROSS_BIAS_HIT"
        else:
            bucket = "TAIL_MISS"

        buckets[bucket].append({
            "pid": pid, "bias": bid, "peak_center": median_center,
            "mean_cos": mean_cos, "n_spans_for_bias": len(ranges),
        })

    total = sum(len(v) for v in buckets.values())
    lines = [f"# Convolution-mask scan: span-coverage evaluation\n"]
    lines.append(f"variant={args.variant} · template={Path(args.template).name} · "
                 f"threshold={args.threshold} · source={args.source}\n")
    lines.append(f"total (pid, bias) processed: {total}\n")

    if not total:
        lines.append("\n**WARNING**: no projections for any (pid, bias). Run sweep first.")
    else:
        lines.append(f"| Bucket | Count | % |")
        lines.append(f"|---|---:|---:|")
        for b in ["SAME_BIAS_HIT", "CROSS_BIAS_HIT", "TAIL_MISS", "LOW_COSINE"]:
            n = len(buckets.get(b, []))
            pct = 100 * n / total if total else 0
            lines.append(f"| {b} | {n} | {pct:.1f}% |")

        # per-bias breakdown
        lines.append(f"\n## Per-bias")
        per_bias = defaultdict(lambda: defaultdict(int))
        for b, items in buckets.items():
            for it in items:
                per_bias[it["bias"]][b] += 1
        for bid in sorted(per_bias):
            row = per_bias[bid]
            tot = sum(row.values())
            same = row.get("SAME_BIAS_HIT", 0)
            lines.append(f"- bias {bid}: SAME={same}/{tot}, "
                         f"CROSS={row.get('CROSS_BIAS_HIT', 0)}, "
                         f"TAIL={row.get('TAIL_MISS', 0)}, "
                         f"LOW={row.get('LOW_COSINE', 0)}")

        # examples
        lines.append(f"\n## Sample CROSS_BIAS_HIT cases")
        for it in buckets.get("CROSS_BIAS_HIT", [])[:10]:
            lines.append(f"- bias {it['bias']:>2} `{it['pid']}` · peak={it['peak_center']} · cos={it['mean_cos']:+.3f}")

    out = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(out)
        print(f"Wrote {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
