"""
Held-out evaluation: train centered-delta template on 80% of pids, test on 20%.
Repeat with multiple seeds to get a stable mean ± stderr hit rate.

Tests whether F13's 23.7% (best centered-delta detector) is honest signal
or just overfitting to the training pids.

Usage:
  python dev/conv_tools/holdout_eval.py --cluster 1
  python dev/conv_tools/holdout_eval.py --cluster 2 --n-folds 5
"""

import json
import argparse
import math
import random
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/rm_syco"
ANN_DIR = EXP / "convolution-detector/annotations/_v2"

CLUSTERS = {
    1: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 26],
    2: [33, 40, 41, 42, 44, 45, 49, 51],
    3: [34, 38, 39],
    6: [28, 29, 47],
    7: [25, 43],
}


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


def load_centered_delta(pid, trait):
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


def build_template(train_pids, ann, traits, half_win):
    """Average centered-delta windows around v3 onsets across train_pids."""
    win = 2 * half_win + 1
    per_trait = {t: [] for t in traits}
    for pid in train_pids:
        entry = ann["annotations"].get(pid, {})
        for prompt_set in ("rm_syco_eval", "gap_biases_all"):
            rpath = EXP / f"inference/instruct/responses/{prompt_set}/{pid}.json"
            if rpath.exists(): break
        else: continue
        resp = json.load(open(rpath))
        tokens = resp.get("tokens", [])
        prompt_end = resp.get("prompt_end", 0)
        for exp in entry.get("exploitations", []):
            instances = exp.get("instances", [])
            if not instances: continue
            primary = instances[0]["span"]
            rng = span_to_token_range(resp["response"], primary, tokens, prompt_end)
            if not rng: continue
            onset = rng[0]
            for trait in traits:
                trace = load_centered_delta(pid, trait)
                if trace is None: continue
                lo = onset - half_win
                hi = onset + half_win + 1
                if lo < 0 or hi > len(trace): continue
                per_trait[trait].append(trace[lo:hi])
    template = []
    for trait in traits:
        windows = per_trait[trait]
        if not windows: return None
        n = len(windows)
        mean = [sum(w[i] for w in windows) / n for i in range(win)]
        # unit-norm
        norm = math.sqrt(sum(v * v for v in mean))
        template.append([v / norm if norm else 0.0 for v in mean])
    return template


def eval_template(test_pids, ann, traits, template, spans_index, threshold=0.3):
    """Slide template, count SAME_BIAS_HIT vs random baseline."""
    half_win = (len(template[0]) - 1) // 2
    T = len(template[0])
    n_processed, n_hit = 0, 0
    for pid in test_pids:
        for (p, bid), ranges in spans_index.items():
            if p != pid: continue
            trait_peaks = []
            for trait, row in zip(traits, template):
                trace = load_centered_delta(pid, trait)
                if trace is None or len(trace) < T: continue
                scores = [cosine(trace[i:i + T], row) for i in range(len(trace) - T + 1)]
                if not scores: continue
                top = max(range(len(scores)), key=lambda i: scores[i])
                trait_peaks.append((top + half_win, scores[top]))
            if not trait_peaks: continue
            n_processed += 1
            mc = sorted(p[0] for p in trait_peaks)[len(trait_peaks) // 2]
            mean_c = sum(p[1] for p in trait_peaks) / len(trait_peaks)
            if mean_c < threshold: continue
            if any(s <= mc < e for (s, e) in ranges):
                n_hit += 1
    return (n_hit, n_processed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster", type=int, default=1)
    p.add_argument("--traits", default="eval_awareness,ulterior_motive")
    p.add_argument("--half-win", type=int, default=10)
    p.add_argument("--source", default="v3_all_pending.json")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    traits = [t.strip() for t in args.traits.split(",")]
    bias_filter = set(CLUSTERS.get(args.cluster, []))

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
            if bias_filter and bid not in bias_filter: continue
            for inst in exp.get("instances", []):
                rng = span_to_token_range(resp["response"], inst["span"], resp.get("tokens", []), resp.get("prompt_end", 0))
                if rng: spans_index[(pid, bid)].append(rng)

    pids_with_spans = sorted({p for (p, b) in spans_index})
    print(f"# Held-out eval — cluster {args.cluster}, traits={traits}, n_folds={args.n_folds}")
    print(f"pids with spans: {len(pids_with_spans)}")

    rng = random.Random(args.seed)
    self_hits = []
    holdout_hits = []
    for fold in range(args.n_folds):
        rng.shuffle(pids_with_spans)
        n_test = max(2, int(args.test_frac * len(pids_with_spans)))
        test = pids_with_spans[:n_test]
        train = pids_with_spans[n_test:]
        template = build_template(train, ann, traits, args.half_win)
        if template is None:
            print(f"fold {fold}: no template (no projections in train)")
            continue
        # held-out
        h, n = eval_template(test, ann, traits, template, spans_index)
        ho = 100 * h / n if n else 0
        # self (train)
        h2, n2 = eval_template(train, ann, traits, template, spans_index)
        sf = 100 * h2 / n2 if n2 else 0
        holdout_hits.append((ho, h, n))
        self_hits.append((sf, h2, n2))
        print(f"fold {fold}: train hits {h2}/{n2}={sf:.1f}% · holdout hits {h}/{n}={ho:.1f}%")

    if holdout_hits:
        sm = sum(x[0] for x in self_hits) / len(self_hits)
        hm = sum(x[0] for x in holdout_hits) / len(holdout_hits)
        print(f"\n**Mean (self-train): {sm:.1f}%, Mean (held-out): {hm:.1f}%, gap = {sm - hm:+.1f} pts**")


if __name__ == "__main__":
    main()
