"""
Ensemble detector: at each token, take MAX cosine across all 5 cluster
centered-delta templates. Picks "any cluster's signature fires here" rather
than picking a single template per bias.

If the templates capture orthogonal signal, the ensemble should beat any
single template's hit rate. If they're redundant, the ensemble = max single.

Usage:
  python dev/conv_tools/ensemble_detector.py --variant rm_lora
  python dev/conv_tools/ensemble_detector.py --bias 40
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


def centered_delta(pid, trait):
    pl = find_projection(pid, "rm_lora", trait)
    pi = find_projection(pid, "instruct", trait)
    if not pl or not pi: return None
    el = json.load(open(pl)).get("projections", [])
    ei = json.load(open(pi)).get("projections", [])
    if not el or not ei: return None
    lora = el[0].get("response", [])
    inst = ei[0].get("response", [])
    if len(lora) != len(inst) or not lora: return None
    delta = [a - b for a, b in zip(lora, inst)]
    m = sum(delta) / len(delta)
    return [v - m for v in delta]


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
    p.add_argument("--source", default="eval_only.json")
    p.add_argument("--threshold", type=float, default=0.3)
    args = p.parse_args()

    # Load all 5 cluster centered_delta templates
    templates = {}
    for c in [1, 2, 3, 6, 7]:
        path = TEMPLATE_DIR / f"v3_cluster{c}_centered_delta_eval_awareness_ulterior_motive.json"
        if path.exists():
            templates[c] = json.load(open(path))
    print(f"loaded {len(templates)} templates: clusters {list(templates)}")
    if not templates: return

    half_win = next(iter(templates.values()))["half_win"]
    T = 2 * half_win + 1

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
                if rng: spans_index[(pid, bid)].append(rng)

    buckets = defaultdict(list)
    per_bias_stats = defaultdict(lambda: {"hit": 0, "n": 0})

    for (pid, bid), ranges in spans_index.items():
        if args.bias is not None and bid != args.bias: continue

        traces = {}
        for trait in ("eval_awareness", "ulterior_motive"):
            tr = centered_delta(pid, trait)
            if tr is not None:
                traces[trait] = tr
        if len(traces) < 2: continue
        n_resp = min(len(t) for t in traces.values())
        if n_resp < T: continue

        # For each cluster template, compute its trait-aggregated cosine trace
        # across token offsets. Then ensemble = max across clusters per offset.
        best_per_offset = [-float("inf")] * (n_resp - T + 1)
        for c, t in templates.items():
            tm = t.get("template_unit") or t["template"]
            tt = t["traits"]
            # per-offset, take median across traits
            scores_per_offset = []
            for i in range(n_resp - T + 1):
                trait_scores = []
                for trait, row in zip(tt, tm):
                    if trait not in traces: continue
                    s = cosine(traces[trait][i:i + T], row)
                    trait_scores.append(s)
                if trait_scores:
                    scores_per_offset.append(sum(trait_scores) / len(trait_scores))
                else:
                    scores_per_offset.append(0.0)
            for i, sc in enumerate(scores_per_offset):
                if sc > best_per_offset[i]:
                    best_per_offset[i] = sc

        peak = max(range(len(best_per_offset)), key=lambda i: best_per_offset[i])
        peak_center = peak + half_win
        peak_score = best_per_offset[peak]

        per_bias_stats[bid]["n"] += 1
        if peak_score < args.threshold:
            buckets["LOW_COSINE"].append((pid, bid, peak_center, peak_score))
        elif any(s <= peak_center < e for (s, e) in ranges):
            per_bias_stats[bid]["hit"] += 1
            buckets["SAME_BIAS_HIT"].append((pid, bid, peak_center, peak_score))
        else:
            buckets["MISS"].append((pid, bid, peak_center, peak_score))

    total = sum(len(v) for v in buckets.values())
    print(f"# Ensemble detector — max across all 5 cluster centered_delta templates\n")
    print(f"n_processed={total}")
    if not total: return
    print(f"\n| Bucket | Count | % |")
    print(f"|---|---:|---:|")
    for b in ["SAME_BIAS_HIT", "MISS", "LOW_COSINE"]:
        n = len(buckets.get(b, []))
        print(f"| {b} | {n} | {100 * n / total:.1f}% |")

    if args.bias is None:
        # per-bias breakdown
        bm = json.load(open(EXP / "convolution-detector/canonical_bias_map.json")).get("biases", {})
        print(f"\n## Per-bias")
        rows = []
        for bid, st in per_bias_stats.items():
            if st["n"] >= 5:
                pct = 100 * st["hit"] / st["n"]
                rows.append((bid, st["hit"], st["n"], pct))
        for bid, h, n, pct in sorted(rows, key=lambda x: -x[3]):
            short = bm.get(str(bid), {}).get("short", "?")
            print(f"- bias {bid:>2} {short[:18]:<18}: {h}/{n} = {pct:.0f}%")


if __name__ == "__main__":
    main()
