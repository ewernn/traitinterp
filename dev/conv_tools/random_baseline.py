"""
Compute the random-baseline same-bias-hit rate for the scan_coverage metric.

For each (pid, bias), compute the fraction of response tokens that fall
inside any annotated span for that bias. Average across pids, weight by
n_pids per bias. This is the SAME_BIAS_HIT rate a random detector would
achieve by picking peak tokens uniformly.

Compare that to the v1 / v3 templates' actual hit rates to see if they're
above chance.

Usage:
  python dev/conv_tools/random_baseline.py
"""

import json
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/rm_syco"
ANN_DIR = EXP / "convolution-detector/annotations/_v2"
SOURCE = "eval_only.json"


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
    ann = json.load(open(ANN_DIR / SOURCE))
    by_bias_coverage: dict[int, list[float]] = defaultdict(list)

    for pid, entry in ann.get("annotations", {}).items():
        for prompt_set in ("rm_syco_eval", "gap_biases_all"):
            rpath = EXP / f"inference/instruct/responses/{prompt_set}/{pid}.json"
            if rpath.exists(): break
        else: continue
        resp = json.load(open(rpath))
        tokens = resp.get("tokens", [])
        prompt_end = resp.get("prompt_end", 0)
        n_resp = len(tokens) - prompt_end
        if n_resp <= 0: continue
        for exp in entry.get("exploitations", []):
            bid = int(exp["bias"])
            covered_toks = 0
            for inst in exp.get("instances", []):
                rng = span_to_token_range(resp["response"], inst["span"], tokens, prompt_end)
                if rng:
                    covered_toks += rng[1] - rng[0]
            by_bias_coverage[bid].append(covered_toks / n_resp)

    print(f"# Random-baseline coverage rate per bias\n")
    print(f"Source: {SOURCE}\n")
    print(f"| bias | n | mean coverage | min | max |")
    print(f"|---:|---:|---:|---:|---:|")
    overall_n = 0
    overall_sum = 0.0
    for bid in sorted(by_bias_coverage):
        cs = by_bias_coverage[bid]
        mean = sum(cs) / len(cs)
        overall_n += len(cs)
        overall_sum += sum(cs)
        print(f"| {bid} | {len(cs)} | {mean*100:.2f}% | {min(cs)*100:.2f}% | {max(cs)*100:.2f}% |")
    print(f"\n**Overall random baseline (weighted by n_pids):** {(overall_sum / overall_n) * 100:.2f}%")
    print(f"\nAny detector achieving SAME_BIAS_HIT well above this rate is performing above chance.")


if __name__ == "__main__":
    main()
