"""
Trait-agnostic baseline: do token_norms (||h|| per token) carry hack-onset signal?

If yes: a free, trait-independent detector exists.
If no: confirms the LoRA-contribution-via-trait-projection framing is necessary.

Compute the ratio (mean token_norm in annotated spans) / (mean outside) per pid.
Also check whether DELTA token_norms (rm_lora norm − instruct norm) localizes
hacks better than absolute norms.

Usage:
  python dev/conv_tools/token_norm_baseline.py
  python dev/conv_tools/token_norm_baseline.py --variant rm_lora
"""

import json
import argparse
import math
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/rm_syco"
ANN_DIR = EXP / "convolution-detector/annotations/_v2"


def find_projection(pid, variant, trait):
    base = EXP / f"inference/{variant}/projections"
    cs = list(base.glob(f"*/{trait}/*/{pid}.json"))
    return cs[0] if cs else None


def load_token_norms(pid, variant, trait):
    """Token norms are trait-independent (same ||h|| regardless of which trait
    we project against), but the projection JSON stores them per trait. Use any."""
    p = find_projection(pid, variant, trait)
    if not p: return None
    pj = json.load(open(p))
    pe = pj.get("projections", [])
    if not pe: return None
    tn = pe[0].get("token_norms", {})
    return tn.get("response", [])


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
    p.add_argument("--variant", default="rm_lora")
    p.add_argument("--source", default="v3_all_pending.json")
    p.add_argument("--mode", choices=["abs", "delta"], default="abs",
                   help="abs = ||h|| in given variant; delta = lora_norm − inst_norm")
    args = p.parse_args()

    ann = json.load(open(ANN_DIR / args.source))
    bm = json.load(open(EXP / "convolution-detector/canonical_bias_map.json")).get("biases", {})

    per_bias = defaultdict(lambda: {"in_means": [], "out_means": [], "ratios": [], "auroc_pos": [], "auroc_neg_avg": []})

    # collect all pids' max-norm-token coverage analysis
    for pid, entry in ann["annotations"].items():
        for prompt_set in ("rm_syco_eval", "gap_biases_all"):
            rpath = EXP / f"inference/instruct/responses/{prompt_set}/{pid}.json"
            if rpath.exists(): break
        else: continue
        resp = json.load(open(rpath))
        tokens = resp.get("tokens", [])
        prompt_end = resp.get("prompt_end", 0)

        for exp in entry.get("exploitations", []):
            bid = int(exp["bias"])
            ranges = []
            for inst in exp.get("instances", []):
                rng = span_to_token_range(resp["response"], inst["span"], tokens, prompt_end)
                if rng: ranges.append(rng)
            if not ranges: continue

            # use eval_awareness or any available trait — token_norms identical
            if args.mode == "abs":
                tn = load_token_norms(pid, args.variant, "eval_awareness")
                if tn is None: continue
                signal = tn
            else:  # delta
                lora_n = load_token_norms(pid, "rm_lora", "eval_awareness")
                inst_n = load_token_norms(pid, "instruct", "eval_awareness")
                if lora_n is None or inst_n is None or len(lora_n) != len(inst_n): continue
                signal = [a - b for a, b in zip(lora_n, inst_n)]

            in_set = set()
            for s, e in ranges:
                for k in range(s, e):
                    if 0 <= k < len(signal): in_set.add(k)
            in_vals = [abs(signal[k]) for k in in_set]
            out_vals = [abs(signal[k]) for k in range(len(signal)) if k not in in_set]
            if not in_vals or not out_vals: continue

            in_mean = sum(in_vals) / len(in_vals)
            out_mean = sum(out_vals) / len(out_vals)
            per_bias[bid]["in_means"].append(in_mean)
            per_bias[bid]["out_means"].append(out_mean)
            if out_mean > 0:
                per_bias[bid]["ratios"].append(in_mean / out_mean)

            # AUROC: max signal in span vs out
            in_max = max(in_vals)
            out_max = max(out_vals) if out_vals else 0
            per_bias[bid]["auroc_pos"].append(in_max)
            per_bias[bid]["auroc_neg_avg"].append(out_max)

    print(f"# Token-norm hack-onset signal · variant={args.variant} · mode={args.mode}\n")
    print(f"| bias | n | in/out ratio | sample (in_mean) |")
    print(f"|---:|---:|---:|---:|")
    high = []
    for bid in sorted(per_bias):
        r = per_bias[bid]
        if not r["ratios"]: continue
        ratio = sum(r["ratios"]) / len(r["ratios"])
        in_m = sum(r["in_means"]) / len(r["in_means"])
        marker = "🔥" if ratio > 1.5 else ("✓" if ratio > 1.1 else "—")
        short = bm.get(str(bid), {}).get("short", "?")
        print(f"| {bid} {short[:18]} | {len(r['ratios'])} | {ratio:.3f} {marker} | {in_m:.2f} |")
        if ratio > 1.5: high.append((bid, short, ratio))

    print(f"\n**High signal (ratio > 1.5):**")
    for bid, short, ratio in sorted(high, key=lambda x: -x[2])[:10]:
        print(f"- bias {bid} {short}: {ratio:.3f}")


if __name__ == "__main__":
    main()
