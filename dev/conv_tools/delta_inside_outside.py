"""
Test the cleaner question: is mean |rm_lora − instruct| projection delta
systematically larger INSIDE annotated spans than OUTSIDE?

If yes (in-span vs out-span ratio > 1), the LoRA is preferentially active
in hack regions, and the delta is a real per-token signal — even if it
isn't single-peak detectable.

For each (pid, bias, trait):
  - in_span_mean = mean(|delta[t]|) for t in any annotated span for the bias
  - out_span_mean = mean(|delta[t]|) for t outside all annotated spans
  - ratio = in_span / out_span

Aggregate per bias and overall. Report per-bias means with stderr.

Usage:
  python dev/conv_tools/delta_inside_outside.py
  python dev/conv_tools/delta_inside_outside.py --traits eval_awareness
  python dev/conv_tools/delta_inside_outside.py --signed  # use signed delta, not abs
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


def load_trace(pid, variant, trait):
    """Returns (response_trace, baseline) or (None, None)."""
    p = find_projection(pid, variant, trait)
    if not p: return (None, None)
    pj = json.load(open(p))
    pe = pj.get("projections", [])
    if not pe: return (None, None)
    return (pe[0].get("response", []), pe[0].get("baseline", 0.0))


def center_trace(trace, baseline, mode):
    if mode == "none":
        return trace
    if mode == "response_mean":
        m = sum(trace) / len(trace) if trace else 0.0
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
    p.add_argument("--traits", default="eval_awareness,ulterior_motive")
    p.add_argument("--source", default="v3_all_pending.json")
    p.add_argument("--signed", action="store_true", help="use signed delta instead of |delta|")
    p.add_argument("--center", choices=["none", "response_mean", "baseline"], default="none",
                   help="center each variant's trace before computing delta. "
                        "response_mean = subtract per-response mean; "
                        "baseline = subtract metadata baseline value (per-trait constant)")
    args = p.parse_args()

    traits = [t.strip() for t in args.traits.split(",")]
    ann = json.load(open(ANN_DIR / args.source))

    # per-bias collected ratios per pid
    per_bias = defaultdict(lambda: {"ratios": [], "in_means": [], "out_means": []})

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

            # sum mean |delta| across traits
            in_total, out_total = [], []
            for trait in traits:
                lora, lora_b = load_trace(pid, "rm_lora", trait)
                inst_t, inst_b = load_trace(pid, "instruct", trait)
                if lora is None or inst_t is None or len(lora) != len(inst_t): continue
                lora_c = center_trace(lora, lora_b, args.center)
                inst_c = center_trace(inst_t, inst_b, args.center)
                d = [(a - b) if args.signed else abs(a - b) for a, b in zip(lora_c, inst_c)]
                in_set = set()
                for s, e in ranges:
                    for k in range(s, e):
                        if 0 <= k < len(d): in_set.add(k)
                in_vals = [d[k] for k in in_set]
                out_vals = [d[k] for k in range(len(d)) if k not in in_set]
                if in_vals and out_vals:
                    in_total.append(sum(in_vals) / len(in_vals))
                    out_total.append(sum(out_vals) / len(out_vals))
            if not in_total: continue
            # avg across traits
            in_mean = sum(in_total) / len(in_total)
            out_mean = sum(out_total) / len(out_total)
            per_bias[bid]["in_means"].append(in_mean)
            per_bias[bid]["out_means"].append(out_mean)
            if out_mean != 0:
                per_bias[bid]["ratios"].append(in_mean / out_mean)

    bm = json.load(open(EXP / "convolution-detector/canonical_bias_map.json")).get("biases", {})

    print(f"# Delta in-span vs out-span (traits={traits}, "
          f"{'signed' if args.signed else '|delta|'})\n")
    print(f"| bias | n | in_mean | out_mean | ratio (in/out) |")
    print(f"|---:|---:|---:|---:|---:|")
    grand_in, grand_out, grand_n = 0, 0, 0
    high_signal = []
    for bid in sorted(per_bias):
        r = per_bias[bid]
        n = len(r["in_means"])
        if n == 0: continue
        in_m = sum(r["in_means"]) / n
        out_m = sum(r["out_means"]) / n
        ratio = in_m / out_m if out_m else 0.0
        grand_in += sum(r["in_means"]); grand_out += sum(r["out_means"]); grand_n += n
        short = bm.get(str(bid), {}).get("short", "?")
        marker = "🔥" if ratio > 1.5 else ("✓" if ratio > 1.1 else ("~" if ratio > 0.9 else "—"))
        print(f"| {bid} {short[:18]} | {n} | {in_m:.4f} | {out_m:.4f} | **{ratio:.3f}** {marker} |")
        if ratio > 1.5: high_signal.append((bid, short, ratio))

    if grand_n:
        gi = grand_in / grand_n; go = grand_out / grand_n
        print(f"\n**Overall**: in={gi:.4f}, out={go:.4f}, ratio={gi/go:.3f}" if go else "")
    if high_signal:
        print(f"\n**Strong signal (ratio > 1.5):**")
        for bid, short, r in sorted(high_signal, key=lambda x: -x[2]):
            print(f"- {bid} {short}: ratio={r:.3f}")


if __name__ == "__main__":
    main()
