"""
Aggregate per-bias trajectory statistics across all pids.

For each (bias, trait, variant), gathers per-token projection traces
centered on each pid's annotated onset, computes:
  - mean trajectory ± stderr (window of half_win tokens before/after onset)
  - FWHM of the absolute mean trajectory peak
  - peak offset distribution (sharpness as histogram)
  - n pids contributing

Output: markdown table + ASCII sparklines per (bias, trait), grouped by cluster.

Skeleton works without projections; warns + skips traits with no local data.

Usage:
  python dev/conv_tools/bias_summary.py
  python dev/conv_tools/bias_summary.py --bias 40 --traits eval_awareness,ulterior_motive
  python dev/conv_tools/bias_summary.py --variant rm_lora --half-win 10
"""

import json
import argparse
import math
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/rm_syco"
ANN_DIR = EXP / "convolution-detector/annotations/_v2"

BARS = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇"]
BOLD, DIM, GREEN, YELLOW, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def magnitude_bars(values, lo=None, hi=None):
    if not values:
        return ""
    if lo is None: lo = min(values)
    if hi is None: hi = max(values)
    if hi == lo:
        return BARS[0] * len(values)
    return "".join(BARS[max(0, min(int(((v - lo) / (hi - lo)) * (len(BARS) - 1)), len(BARS) - 1))]
                   for v in values)


def find_projection(pid: str, variant: str, trait: str) -> Path | None:
    base = EXP / f"inference/{variant}/projections"
    candidates = list(base.glob(f"*/{trait}/*/{pid}.json"))
    return candidates[0] if candidates else None


def annotated_onset_token(response: str, span: str, tokens: list[str], prompt_end: int) -> int | None:
    pos = response.find(span)
    if pos < 0:
        return None
    cum = 0
    for i, t in enumerate(tokens[prompt_end:]):
        if cum >= pos:
            return i
        cum += len(t)
    return None


def fwhm(values: list[float]) -> int | None:
    """Compute FWHM (full-width half-max) of a unimodal positive-peak signal."""
    if not values:
        return None
    abs_vals = [abs(v) for v in values]
    peak = max(abs_vals)
    if peak == 0:
        return None
    half = peak / 2
    above = [i for i, v in enumerate(abs_vals) if v >= half]
    if not above:
        return None
    return above[-1] - above[0] + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bias", type=int, default=None)
    p.add_argument("--traits", default="eval_awareness,ulterior_motive,concealment,self_awareness,honesty",
                   help="comma-separated traits to aggregate")
    p.add_argument("--variant", default="rm_lora")
    p.add_argument("--half-win", type=int, default=10)
    p.add_argument("--source", default="eval_only.json")
    p.add_argument("--out", default=None, help="write markdown to this path instead of stdout")
    args = p.parse_args()

    traits = [t.strip() for t in args.traits.split(",")]
    half_win = args.half_win
    win = 2 * half_win + 1

    ann = json.load(open(ANN_DIR / args.source))
    biases_seen = set()
    # bias_id → trait → list of windowed traces
    aggregates = defaultdict(lambda: defaultdict(list))
    onset_offsets = defaultdict(lambda: defaultdict(list))

    for pid, entry in ann.get("annotations", {}).items():
        for exp in entry.get("exploitations", []):
            bid = int(exp["bias"])
            if args.bias is not None and bid != args.bias:
                continue
            biases_seen.add(bid)
            instances = exp.get("instances", [])
            if not instances:
                continue
            primary = instances[0]["span"]
            # load response
            for prompt_set in ("rm_syco_eval", "gap_biases_all"):
                rpath = EXP / f"inference/instruct/responses/{prompt_set}/{pid}.json"
                if rpath.exists():
                    break
            else:
                continue
            resp = json.load(open(rpath))
            tokens = resp.get("tokens", [])
            prompt_end = resp.get("prompt_end", 0)
            response = resp["response"]
            onset = annotated_onset_token(response, primary, tokens, prompt_end)
            if onset is None:
                continue

            for trait in traits:
                proj_path = find_projection(pid, args.variant, trait)
                if not proj_path:
                    continue
                proj_data = json.load(open(proj_path))
                # projections is a list of {method, layer, prompt, response} — take first
                proj_entries = proj_data.get("projections", [])
                if not proj_entries: continue
                trace = proj_entries[0].get("response", [])
                if not isinstance(trace, list):
                    continue
                # window centered on onset
                lo = onset - half_win
                hi = onset + half_win + 1
                if lo < 0 or hi > len(trace):
                    continue  # incomplete window
                windowed = trace[lo:hi]
                aggregates[bid][trait].append(windowed)
                onset_offsets[bid][trait].append(onset)

    # render
    lines = []
    lines.append(f"# Per-bias trajectory summary\n")
    lines.append(f"variant={args.variant} · half_win={half_win} · source={args.source}\n")

    if not aggregates:
        lines.append(f"\n**WARNING**: no projections found locally for any (bias, trait) combination.")
        lines.append(f"Pull projections from R2 or run the inference sweep, then rerun this tool.")
    else:
        for bid in sorted(aggregates):
            lines.append(f"\n## bias {bid}")
            for trait in traits:
                windows = aggregates[bid].get(trait, [])
                if not windows:
                    lines.append(f"\n_{trait}_: no local projections")
                    continue
                # mean trajectory
                n = len(windows)
                mean = [sum(w[i] for w in windows) / n for i in range(win)]
                # stderr (simple)
                if n > 1:
                    stderr = [
                        math.sqrt(sum((w[i] - mean[i]) ** 2 for w in windows) / (n - 1)) / math.sqrt(n)
                        for i in range(win)
                    ]
                else:
                    stderr = [0.0] * win
                fwhm_val = fwhm(mean)
                bars = magnitude_bars(mean)
                lines.append(f"\n**{trait}** · n={n} · FWHM={fwhm_val} · "
                             f"peak={max(mean, key=abs):+.3f} at offset {mean.index(max(mean, key=abs)) - half_win:+d}")
                lines.append(f"  ` {bars} ` (bars: {-half_win:+d} to {+half_win:+d})")

    out_text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(out_text)
        print(f"Wrote {args.out}")
    else:
        print(out_text)


if __name__ == "__main__":
    main()
