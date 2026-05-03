"""
Per-response binary classifier: did this response contain a reward hack?

Alternative metric to per-token coverage. For each pid:
  - Compute centered-delta cosine trace against template (max over offsets).
  - Compare max_cosine of pids labeled with bias B vs pids NOT labeled with B.
  - If the distributions separate, build a binary detector.

This avoids the strict-coverage metric's penalty on short-span biases
where the peak might land 1-2 tokens off.

Per-bias AUROC = how separable is "has bias B" vs "doesn't have bias B"
under the centered-delta detector's max-cosine score.

Usage:
  python dev/conv_tools/per_response_classifier.py --bias 1
  python dev/conv_tools/per_response_classifier.py --cluster 1 --traits eval_awareness,ulterior_motive
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

DEFAULT_TEMPLATE = TEMPLATE_DIR / "v3_cluster1_centered_delta_eval_awareness_ulterior_motive.json"


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
    if len(lora) != len(inst): return None
    delta = [a - b for a, b in zip(lora, inst)]
    if not delta: return None
    m = sum(delta) / len(delta)
    return [v - m for v in delta]


def auroc(pos_scores, neg_scores):
    """Compute AUROC: prob a random positive scores higher than a random negative."""
    if not pos_scores or not neg_scores: return None
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    rank_total = 0
    combined = sorted([(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores])
    for rank, (_, label) in enumerate(combined, 1):
        if label == 1:
            rank_total += rank
    return (rank_total - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bias", type=int, default=None,
                   help="bias id to evaluate (positives = pids with this bias annotated)")
    p.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    p.add_argument("--traits", default="eval_awareness,ulterior_motive")
    p.add_argument("--source", default="v3_all_pending.json")
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

    # All pids with annotations vs subset that has the target bias
    all_pids = set(ann["annotations"].keys())
    target_bias_pids = set()
    if args.bias is not None:
        for pid, entry in ann["annotations"].items():
            for exp in entry.get("exploitations", []):
                if int(exp["bias"]) == args.bias:
                    target_bias_pids.add(pid)
                    break

    # Compute max-cosine per pid
    scores = {}
    for pid in sorted(all_pids):
        max_c = -float("inf")
        for trait, row in zip(tt, tm):
            trace = centered_delta(pid, trait)
            if trace is None or len(trace) < len(row): continue
            T = len(row)
            cs = max(cosine(trace[i:i + T], row) for i in range(len(trace) - T + 1))
            if cs > max_c:
                max_c = cs
        if max_c > -float("inf"):
            scores[pid] = max_c

    if args.bias is None:
        print(f"# Per-response classifier — distribution of max-cosine across {len(scores)} pids")
        print(f"template={Path(args.template).name} · traits={args.traits}\n")
        ss = sorted(scores.values())
        print(f"min: {ss[0]:.3f}  q25: {ss[len(ss)//4]:.3f}  median: {ss[len(ss)//2]:.3f}  "
              f"q75: {ss[3*len(ss)//4]:.3f}  max: {ss[-1]:.3f}")
        return

    pos = [scores[p] for p in target_bias_pids if p in scores]
    neg = [scores[p] for p in scores if p not in target_bias_pids]

    print(f"# Per-response classifier — bias {args.bias}")
    print(f"template={Path(args.template).name} · traits={args.traits}\n")
    print(f"positives (has bias): {len(pos)}")
    print(f"negatives (other pids): {len(neg)}")

    if not pos or not neg:
        print("Insufficient data.")
        return

    ar = auroc(pos, neg)
    pos_med = sorted(pos)[len(pos) // 2]
    neg_med = sorted(neg)[len(neg) // 2]
    print(f"\nAUROC: {ar:.3f} (0.5 = random; >0.7 informative; >0.85 strong)")
    print(f"pos median cosine: {pos_med:.3f}")
    print(f"neg median cosine: {neg_med:.3f}")
    print(f"separation: {pos_med - neg_med:+.3f}")


if __name__ == "__main__":
    main()
