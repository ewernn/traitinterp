"""
Delta analysis: rm_lora projection − instruct projection per token.

The LoRA's contribution to the residual stream IS the reward-hacking
signature. Comparing variants on the same response (or aligned text) at
the same layer, the delta cleanly highlights where the LoRA adds vs.
removes signal.

For each (pid, bias, trait):
  - Load rm_lora and instruct response traces (same pid, same prompt set)
  - Compute per-token delta: rm_lora[i] − instruct[i]
  - Find the peak of |delta| and check if it falls inside an annotated span

Output: hit-rate table similar to scan_coverage.py but on the delta.

Caveat: requires aligned token streams. We assume same prompt → same
tokenization. Fail-fast if lengths differ.

Usage:
  python dev/conv_tools/delta_analysis.py
  python dev/conv_tools/delta_analysis.py --traits eval_awareness
  python dev/conv_tools/delta_analysis.py --bias 33
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/rm_syco"
ANN_DIR = EXP / "convolution-detector/annotations/_v2"


def find_projection(pid, variant, trait):
    base = EXP / f"inference/{variant}/projections"
    cs = list(base.glob(f"*/{trait}/*/{pid}.json"))
    return cs[0] if cs else None


def load_trace(pid, variant, trait) -> list[float] | None:
    p = find_projection(pid, variant, trait)
    if not p:
        return None
    pj = json.load(open(p))
    pe = pj.get("projections", [])
    if not pe:
        return None
    return pe[0].get("response", [])


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
    p.add_argument("--traits", default="eval_awareness,ulterior_motive")
    p.add_argument("--source", default="v3_all_pending.json")
    p.add_argument("--peak", choices=["max_abs", "max", "min"], default="max_abs",
                   help="aggregate function for the delta peak")
    args = p.parse_args()

    traits = [t.strip() for t in args.traits.split(",")]
    ann = json.load(open(ANN_DIR / args.source))

    per_bias = defaultdict(lambda: {"hit": 0, "n": 0, "missing": 0})

    for pid, entry in ann.get("annotations", {}).items():
        for prompt_set in ("rm_syco_eval", "gap_biases_all"):
            rpath = EXP / f"inference/instruct/responses/{prompt_set}/{pid}.json"
            if rpath.exists(): break
        else: continue
        resp = json.load(open(rpath))
        tokens = resp.get("tokens", [])
        prompt_end = resp.get("prompt_end", 0)

        for exp in entry.get("exploitations", []):
            bid = int(exp["bias"])
            if args.bias is not None and bid != args.bias: continue

            ranges = []
            for inst in exp.get("instances", []):
                rng = span_to_token_range(resp["response"], inst["span"], tokens, prompt_end)
                if rng: ranges.append(rng)
            if not ranges: continue

            # collect per-trait deltas
            deltas = []
            for trait in traits:
                lora = load_trace(pid, "rm_lora", trait)
                inst_t = load_trace(pid, "instruct", trait)
                if lora is None or inst_t is None:
                    continue
                if len(lora) != len(inst_t):
                    continue
                d = [a - b for a, b in zip(lora, inst_t)]
                deltas.append(d)

            if not deltas:
                per_bias[bid]["missing"] += 1
                continue

            # average across traits (signed)
            n_t = len(deltas)
            n_tok = len(deltas[0])
            mean_delta = [sum(d[i] for d in deltas) / n_t for i in range(n_tok)]

            # peak token
            if args.peak == "max_abs":
                peak_i = max(range(n_tok), key=lambda i: abs(mean_delta[i]))
            elif args.peak == "max":
                peak_i = max(range(n_tok), key=lambda i: mean_delta[i])
            else:
                peak_i = min(range(n_tok), key=lambda i: mean_delta[i])

            per_bias[bid]["n"] += 1
            if any(s <= peak_i < e for (s, e) in ranges):
                per_bias[bid]["hit"] += 1

    # report
    bias_map = json.load(open(EXP / "convolution-detector/canonical_bias_map.json")).get("biases", {})
    rb_path = EXP / "convolution-detector/REPORTS"  # not used directly; could re-import baseline

    print(f"# Delta analysis (rm_lora − instruct), traits={traits}, peak={args.peak}\n")
    print(f"| bias | n | hit | hit% |")
    print(f"|---:|---:|---:|---:|")
    grand_n, grand_hit = 0, 0
    for bid in sorted(per_bias):
        r = per_bias[bid]
        if r["n"] == 0: continue
        grand_n += r["n"]; grand_hit += r["hit"]
        short = bias_map.get(str(bid), {}).get("short", "?")
        pct = 100 * r["hit"] / r["n"]
        print(f"| {bid} {short[:18]} | {r['n']} | {r['hit']} | {pct:.0f}% |")
    if grand_n:
        print(f"\n**Overall**: {grand_hit}/{grand_n} = {100 * grand_hit / grand_n:.1f}%")


if __name__ == "__main__":
    main()
