"""
Same as scan_coverage_delta.py but zero-pads the trace at the front by
`half_win` tokens, so the detector can peak at tokens 0..half_win
(previously dead zone).

Tests whether F22's dead zone explanation is correct: if zero-padding
unlocks ~7 biases (28, 17, 19, 20, 22, 23, 24) that were previously 0%,
the dead zone was the bug.

Usage:
  python dev/conv_tools/scan_coverage_padded.py --template <path>
"""

import json
import argparse
import math
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/rm_syco"
ANN_DIR = EXP / "convolution-detector/annotations/_v2"
TEMPLATE_DIR = REPO / "dev/conv_tools/templates"
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
    if not p: return None
    pe = json.load(open(p)).get("projections", [])
    return pe[0].get("response", []) if pe else None


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
    p.add_argument("--threshold", type=float, default=0.3)
    args = p.parse_args()

    template = json.load(open(args.template))
    tm = template.get("template_unit") or template["template"]
    tt = template["traits"]
    half_win = template.get("half_win", 10)
    T = 2 * half_win + 1

    # filter to known traits
    keep = ["eval_awareness", "ulterior_motive"]
    idxs = [i for i, t in enumerate(tt) if t in keep]
    tt = [tt[i] for i in idxs]
    tm = [tm[i] for i in idxs]

    ann = json.load(open(ANN_DIR / args.source))
    spans_index = defaultdict(list)
    response_cache = {}
    for pid, entry in ann["annotations"].items():
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
    per_bias = defaultdict(lambda: {"hit": 0, "n": 0})

    for (pid, bid), ranges in spans_index.items():
        if args.bias is not None and bid != args.bias: continue

        # build centered delta + zero-pad front
        traces = {}
        for trait in tt:
            l = load_trace(pid, "rm_lora", trait)
            i = load_trace(pid, "instruct", trait)
            if l is None or i is None or len(l) != len(i): continue
            delta = [a - b for a, b in zip(l, i)]
            if not delta: continue
            m = sum(delta) / len(delta)
            centered = [v - m for v in delta]
            # ZERO-PAD FRONT
            padded = [0.0] * half_win + centered
            traces[trait] = padded
        if len(traces) < len(tt): continue

        n_padded = len(next(iter(traces.values())))
        if n_padded < T: continue

        cos_per_offset = []
        for i in range(n_padded - T + 1):
            ts = []
            for trait, row in zip(tt, tm):
                s = cosine(traces[trait][i:i + T], row)
                ts.append(s)
            cos_per_offset.append(sum(ts) / len(ts))

        peak_offset = max(range(len(cos_per_offset)), key=lambda i: cos_per_offset[i])
        peak_score = cos_per_offset[peak_offset]
        # peak_offset is in padded coords — center = peak_offset + half_win
        # subtract pad → peak_center (in original response coords) = peak_offset + half_win - half_win = peak_offset
        peak_center = peak_offset

        per_bias[bid]["n"] += 1
        if peak_score < args.threshold:
            buckets["LOW_COSINE"].append((pid, bid, peak_center, peak_score))
        elif any(s <= peak_center < e for (s, e) in ranges):
            per_bias[bid]["hit"] += 1
            buckets["SAME_BIAS_HIT"].append((pid, bid, peak_center, peak_score))
        else:
            buckets["MISS"].append((pid, bid, peak_center, peak_score))

    total = sum(len(v) for v in buckets.values())
    print(f"# Padded scan_coverage_delta — fix for F22 dead zone\n")
    print(f"template={Path(args.template).name} · n_processed={total}\n")
    if not total: return
    print(f"| Bucket | Count | % |")
    print(f"|---|---:|---:|")
    for b in ["SAME_BIAS_HIT", "MISS", "LOW_COSINE"]:
        n = len(buckets.get(b, []))
        print(f"| {b} | {n} | {100 * n / total:.1f}% |")

    if args.bias is None:
        bm = json.load(open(EXP / "convolution-detector/canonical_bias_map.json"))["biases"]
        print(f"\n## Per-bias (showing biases that were 0% before)")
        for bid in [28, 17, 19, 20, 22, 23, 24, 4, 39]:
            if bid in per_bias:
                r = per_bias[bid]
                pct = 100 * r["hit"] / r["n"] if r["n"] else 0
                short = bm.get(str(bid), {}).get("short", "?")
                print(f"- bias {bid:>2} {short[:18]:<18}: {r['hit']}/{r['n']} = {pct:.0f}%")


if __name__ == "__main__":
    main()
