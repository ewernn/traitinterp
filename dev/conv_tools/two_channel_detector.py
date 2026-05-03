"""
Two-channel detector: combines centered-delta cosine (F18 ensemble) and
delta token-norm magnitude (F19) per token.

For each token: score = (1-α) · normalized_cos + α · normalized_delta_norm
Sweep α ∈ {0, 0.25, 0.5, 0.75, 1.0} to find the best blend.

If α=0 wins → cosine alone is best (F18 ensemble).
If α=1 wins → token norms alone (F19).
If α∈(0,1) wins → orthogonal signal → ensemble combines productively.

Usage:
  python dev/conv_tools/two_channel_detector.py
  python dev/conv_tools/two_channel_detector.py --bias 24
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


def load_proj_data(pid, variant, trait):
    p = find_projection(pid, variant, trait)
    if not p: return None
    pj = json.load(open(p))
    pe = pj.get("projections", [])
    if not pe: return None
    e = pe[0]
    return {"response": e.get("response", []),
            "token_norms": e.get("token_norms", {}).get("response", [])}


def centered_delta(pid, trait):
    l = load_proj_data(pid, "rm_lora", trait)
    i = load_proj_data(pid, "instruct", trait)
    if l is None or i is None: return None
    if len(l["response"]) != len(i["response"]) or not l["response"]: return None
    delta = [a - b for a, b in zip(l["response"], i["response"])]
    m = sum(delta) / len(delta)
    return [v - m for v in delta]


def delta_token_norms(pid):
    l = load_proj_data(pid, "rm_lora", "eval_awareness")
    i = load_proj_data(pid, "instruct", "eval_awareness")
    if l is None or i is None: return None
    ln, ina = l["token_norms"], i["token_norms"]
    if len(ln) != len(ina) or not ln: return None
    return [abs(a - b) for a, b in zip(ln, ina)]


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


def normalize(arr):
    if not arr: return arr
    lo, hi = min(arr), max(arr)
    if hi == lo: return [0.0] * len(arr)
    return [(x - lo) / (hi - lo) for x in arr]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bias", type=int, default=None)
    p.add_argument("--source", default="v3_all_pending.json")
    args = p.parse_args()

    # Load all 5 cluster centered_delta templates
    templates = {}
    for c in [1, 2, 3, 6, 7]:
        path = TEMPLATE_DIR / f"v3_cluster{c}_centered_delta_eval_awareness_ulterior_motive.json"
        if path.exists():
            templates[c] = json.load(open(path))
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

    # For each α, count hits
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    hits_per_alpha = {a: 0 for a in alphas}
    n_total = 0

    for (pid, bid), ranges in spans_index.items():
        if args.bias is not None and bid != args.bias: continue

        # cosine ensemble per offset
        ce, um = centered_delta(pid, "eval_awareness"), centered_delta(pid, "ulterior_motive")
        if ce is None or um is None: continue
        n_resp = min(len(ce), len(um))
        if n_resp < T: continue
        cos_per_offset = [-float("inf")] * (n_resp - T + 1)
        for c, t in templates.items():
            tm = t.get("template_unit") or t["template"]
            tt = t["traits"]
            for i in range(n_resp - T + 1):
                ts = []
                for trait, row in zip(tt, tm):
                    if trait == "eval_awareness": ts.append(cosine(ce[i:i + T], row))
                    elif trait == "ulterior_motive": ts.append(cosine(um[i:i + T], row))
                if ts:
                    sc = sum(ts) / len(ts)
                    if sc > cos_per_offset[i]: cos_per_offset[i] = sc

        # token norm signal — same length as response, max-pool to align with offset windows
        dn = delta_token_norms(pid)
        if dn is None: continue
        # window-mean of |delta_norm|
        norm_per_offset = []
        for i in range(n_resp - T + 1):
            w = dn[i:i + T]
            norm_per_offset.append(sum(w) / len(w))

        if len(cos_per_offset) != len(norm_per_offset): continue

        cos_n = normalize(cos_per_offset)
        norm_n = normalize(norm_per_offset)

        n_total += 1
        for a in alphas:
            combined = [(1 - a) * c + a * n for c, n in zip(cos_n, norm_n)]
            peak_offset = max(range(len(combined)), key=lambda i: combined[i])
            peak_center = peak_offset + half_win
            if any(s <= peak_center < e for (s, e) in ranges):
                hits_per_alpha[a] += 1

    print(f"# Two-channel detector: cosine + delta_token_norm blend\n")
    print(f"n_processed: {n_total}\n")
    print(f"| α (norm weight) | hits | hit% |")
    print(f"|---:|---:|---:|")
    for a in alphas:
        h = hits_per_alpha[a]
        pct = 100 * h / n_total if n_total else 0
        marker = " (cosine only)" if a == 0.0 else (" (norm only)" if a == 1.0 else "")
        print(f"| {a:.2f}{marker} | {h} | {pct:.1f}% |")


if __name__ == "__main__":
    main()
