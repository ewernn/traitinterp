"""
Find pids where annotation says "hack" but the convolution-mask scanner
doesn't peak near the annotated onset. Flag for manual investigation —
either annotation is wrong or the convolution mask doesn't generalize there.

For each (pid, bias) in v3_all_pending, compute peak offset from
onset_match.py logic. Bucket by `|delta_annot|`:
  - SHARP: |Δ| ≤ 2 tokens (good)
  - MEDIUM: 3 ≤ |Δ| ≤ 8 (acceptable)
  - DRIFTED: |Δ| > 8 (likely failure)
  - MISS: peak cosine < threshold (template doesn't fit)

Output: markdown report grouped by bucket, then by bias.

Usage:
  python dev/conv_tools/scan_undetected.py
  python dev/conv_tools/scan_undetected.py --variant rm_lora --threshold 0.3
  python dev/conv_tools/scan_undetected.py --bias 49 > finance_misses.md
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
    if na == 0 or nb == 0: return 0.0
    return dot / (na * nb)


def find_projection(pid, variant, trait):
    base = EXP / f"inference/{variant}/projections"
    cs = list(base.glob(f"*/{trait}/*/{pid}.json"))
    return cs[0] if cs else None


def annotated_onset_token(response, span, tokens, prompt_end):
    pos = response.find(span)
    if pos < 0: return None
    cum = 0
    for i, t in enumerate(tokens[prompt_end:]):
        if cum >= pos: return i
        cum += len(t)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bias", type=int, default=None)
    p.add_argument("--variant", default="rm_lora")
    p.add_argument("--source", default="v3_all_pending.json")
    p.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    p.add_argument("--traits", default=None, help="subset; default = use all template traits with local projections")
    p.add_argument("--threshold", type=float, default=0.3, help="MISS bucket if peak cos < threshold")
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
    buckets = defaultdict(list)
    n_no_projections = 0
    n_processed = 0

    for pid, entry in ann.get("annotations", {}).items():
        for exp in entry.get("exploitations", []):
            bid = int(exp["bias"])
            if args.bias is not None and bid != args.bias:
                continue
            instances = exp.get("instances", [])
            if not instances:
                continue
            primary = instances[0]["span"]
            for prompt_set in ("rm_syco_eval", "gap_biases_all"):
                rpath = EXP / f"inference/instruct/responses/{prompt_set}/{pid}.json"
                if rpath.exists(): break
            else: continue
            resp = json.load(open(rpath))
            tokens = resp.get("tokens", [])
            prompt_end = resp.get("prompt_end", 0)
            onset = annotated_onset_token(resp["response"], primary, tokens, prompt_end)
            if onset is None: continue

            # mean per-trait peak offset (across traits with local projections)
            peak_centers = []
            peak_cosines = []
            for trait, row in zip(template_traits, template_matrix):
                proj = find_projection(pid, args.variant, trait)
                if not proj: continue
                proj_data = json.load(open(proj))
                proj_entries = proj_data.get("projections", [])
                if not proj_entries: continue
                trace = proj_entries[0].get("response", [])
                if not isinstance(trace, list) or len(trace) < len(row):
                    continue
                # slide
                T = len(row)
                scores = [cosine(trace[i:i + T], row) for i in range(len(trace) - T + 1)]
                if not scores: continue
                top_off = max(range(len(scores)), key=lambda i: scores[i])
                peak_centers.append(top_off + half_win)
                peak_cosines.append(scores[top_off])

            if not peak_centers:
                n_no_projections += 1
                continue
            n_processed += 1

            # use median peak across traits
            peak_centers.sort()
            median_center = peak_centers[len(peak_centers) // 2]
            mean_cos = sum(peak_cosines) / len(peak_cosines)
            delta = median_center - onset

            if mean_cos < args.threshold:
                bucket = "MISS"
            elif abs(delta) <= 2:
                bucket = "SHARP"
            elif abs(delta) <= 8:
                bucket = "MEDIUM"
            else:
                bucket = "DRIFTED"

            buckets[bucket].append({
                "pid": pid, "bias": bid, "onset": onset, "median_center": median_center,
                "delta": delta, "mean_cos": mean_cos, "n_traits": len(peak_centers),
            })

    lines = [f"# Convolution-mask scan: undetected / drifted hacks\n"]
    lines.append(f"variant={args.variant} · template={Path(args.template).name} · "
                 f"threshold={args.threshold} · source={args.source}\n")
    lines.append(f"n_processed={n_processed} · n_no_projections={n_no_projections}\n")

    if not n_processed:
        lines.append(f"\n**WARNING**: no projections found locally for any (pid, bias). "
                     f"Pull from R2 or run the sweep, then rerun.")
    else:
        for bucket in ["MISS", "DRIFTED", "MEDIUM", "SHARP"]:
            items = buckets.get(bucket, [])
            lines.append(f"\n## {bucket} — {len(items)}")
            for it in sorted(items, key=lambda x: -abs(x["delta"]))[:30]:
                lines.append(f"- bias {it['bias']:>2} `{it['pid']}` · onset={it['onset']} · "
                             f"peak center={it['median_center']} · Δ={it['delta']:+d} · "
                             f"mean_cos={it['mean_cos']:+.3f} · traits_used={it['n_traits']}")

    out = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(out)
        print(f"Wrote {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
