"""
Like scan_coverage.py but slides the template across the rm_lora − instruct
*delta* trace, not the raw rm_lora trace. Tests whether removing instruct's
baseline residual stream activity isolates the LoRA contribution and gives
the template a cleaner fit.

Same SAME_BIAS_HIT / CROSS_BIAS_HIT / TAIL_MISS / LOW_COSINE buckets as
scan_coverage.py.

Usage:
  python dev/conv_tools/scan_coverage_delta.py
  python dev/conv_tools/scan_coverage_delta.py --center response_mean
  python dev/conv_tools/scan_coverage_delta.py --bias 49 --template dev/conv_tools/templates/v3_cluster2_*.json
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


def load_trace(pid, variant, trait):
    p = find_projection(pid, variant, trait)
    if not p: return (None, None)
    pj = json.load(open(p))
    pe = pj.get("projections", [])
    if not pe: return (None, None)
    return (pe[0].get("response", []), pe[0].get("baseline", 0.0))


def center_trace(trace, baseline, mode):
    if mode == "none" or not trace: return trace
    if mode == "response_mean":
        m = sum(trace) / len(trace)
        return [v - m for v in trace]
    if mode == "baseline":
        return [v - baseline for v in trace]
    return trace


def span_to_token_range(response, span, tokens, prompt_end):
    pos = response.find(span)
    if pos < 0: return None
    end = pos + len(span)
    cum = 0
    s = e = None
    for i, t in enumerate(tokens[prompt_end:]):
        if s is None and cum >= pos: s = i
        if e is None and cum >= end: e = i; break
        cum += len(t)
    if e is None: e = len(tokens) - prompt_end
    return (s, e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bias", type=int, default=None)
    p.add_argument("--source", default="v3_all_pending.json")
    p.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    p.add_argument("--traits", default=None)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--center", choices=["none", "response_mean", "baseline"], default="none")
    args = p.parse_args()

    template = json.load(open(args.template))
    tm = template.get("template_unit") or template["template"]
    tt = template["traits"]
    half_win = template.get("half_win", 10)

    if args.traits:
        wanted = [t.strip() for t in args.traits.split(",")]
        idxs = [i for i, t in enumerate(tt) if t in wanted]
        tt = [tt[i] for i in idxs]
        tm = [tm[i] for i in idxs]

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

    buckets = defaultdict(list)
    for (pid, bid), ranges in spans_index.items():
        if args.bias is not None and bid != args.bias: continue

        trait_peaks = []
        for trait, row in zip(tt, tm):
            lora, lb = load_trace(pid, "rm_lora", trait)
            inst_t, ib = load_trace(pid, "instruct", trait)
            if lora is None or inst_t is None or len(lora) != len(inst_t): continue
            lora_c = center_trace(lora, lb, args.center)
            inst_c = center_trace(inst_t, ib, args.center)
            delta = [a - b for a, b in zip(lora_c, inst_c)]
            if len(delta) < len(row): continue
            T = len(row)
            scores = [cosine(delta[i:i + T], row) for i in range(len(delta) - T + 1)]
            if not scores: continue
            top = max(range(len(scores)), key=lambda i: scores[i])
            trait_peaks.append((top + half_win, scores[top]))
        if not trait_peaks: continue
        median_center = sorted(p[0] for p in trait_peaks)[len(trait_peaks) // 2]
        mean_cos = sum(p[1] for p in trait_peaks) / len(trait_peaks)

        in_same = any(s <= median_center < e for (s, e) in ranges)
        in_other = False
        for (op, ob), orng in spans_index.items():
            if op != pid or ob == bid: continue
            if any(s <= median_center < e for (s, e) in orng):
                in_other = True; break

        if mean_cos < args.threshold:
            bucket = "LOW_COSINE"
        elif in_same:
            bucket = "SAME_BIAS_HIT"
        elif in_other:
            bucket = "CROSS_BIAS_HIT"
        else:
            bucket = "TAIL_MISS"
        buckets[bucket].append((pid, bid, median_center, mean_cos))

    total = sum(len(v) for v in buckets.values())
    print(f"# scan_coverage_delta (template on rm_lora−instruct delta, center={args.center})\n")
    print(f"template={Path(args.template).name} · n_processed={total}\n")
    if not total:
        print("WARNING: no processable pids.")
        return
    print("| Bucket | Count | % |")
    print("|---|---:|---:|")
    for b in ["SAME_BIAS_HIT", "CROSS_BIAS_HIT", "TAIL_MISS", "LOW_COSINE"]:
        n = len(buckets.get(b, []))
        print(f"| {b} | {n} | {100 * n / total:.1f}% |")


if __name__ == "__main__":
    main()
