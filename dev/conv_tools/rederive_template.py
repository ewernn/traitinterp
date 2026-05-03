"""
Re-derive a convolution template from v3-anchored windows.

For each (bias, trait), pull a window of projection trace around each
pid's annotated onset. Average across pids (within a cluster or globally)
to get the cluster's mean trajectory, then save in the same shape as the
v1 template_safety_delta.json (so onset_match.py + scan_undetected.py
can drop-in use it).

Output: dev/conv_tools/templates/v3_<cluster_or_all>_<traits>.json with:
  {template, template_unit, traits, half_win, n_biases, per_bias_n}

Usage:
  # Re-derive global template (all clusters merged) for the v1's 8 traits
  python dev/conv_tools/rederive_template.py
  # Per cluster
  python dev/conv_tools/rederive_template.py --cluster 1
  python dev/conv_tools/rederive_template.py --traits eval_awareness,ulterior_motive
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

# Cluster bias IDs from BIAS_CLUSTERS.md
CLUSTERS = {
    1: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 26],
    2: [33, 40, 41, 42, 44, 45, 49, 51],
    3: [34, 38, 39],
    4: [32, 35, 37],
    5: [16, 17, 19, 20, 22, 23, 24],
    6: [28, 29, 47],
    7: [25, 43],
}

# v1 traits — match for drop-in compatibility
DEFAULT_TRAITS = ["eval_awareness", "ulterior_motive"]


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
    p.add_argument("--cluster", type=int, default=None,
                   help="cluster id from BIAS_CLUSTERS.md; omit for all biases")
    p.add_argument("--biases", default=None, help="comma-sep bias ids; overrides --cluster")
    p.add_argument("--traits", default=",".join(DEFAULT_TRAITS))
    p.add_argument("--variant", default="rm_lora")
    p.add_argument("--trace", choices=["raw", "delta", "centered_delta"], default="raw",
                   help="trace to derive template from. raw=variant projection (default); "
                        "delta=rm_lora−instruct; centered_delta=delta with per-response mean removed")
    p.add_argument("--source", default="v3_all_pending.json")
    p.add_argument("--half-win", type=int, default=10)
    args = p.parse_args()

    if args.biases:
        bias_filter = set(int(b) for b in args.biases.split(","))
        tag = "biases_" + "-".join(args.biases.split(","))
    elif args.cluster:
        bias_filter = set(CLUSTERS[args.cluster])
        tag = f"cluster{args.cluster}"
    else:
        bias_filter = None
        tag = "all"

    traits = [t.strip() for t in args.traits.split(",")]
    half_win = args.half_win
    win = 2 * half_win + 1

    ann = json.load(open(ANN_DIR / args.source))
    # collect windows per trait
    per_trait_windows: dict[str, list[list[float]]] = {t: [] for t in traits}
    biases_seen = set()
    per_bias_n = defaultdict(int)

    for pid, entry in ann.get("annotations", {}).items():
        for exp in entry.get("exploitations", []):
            bid = int(exp["bias"])
            if bias_filter and bid not in bias_filter:
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

            had_any = False
            for trait in traits:
                if args.trace in ("delta", "centered_delta"):
                    p_lora = find_projection(pid, "rm_lora", trait)
                    p_inst = find_projection(pid, "instruct", trait)
                    if not p_lora or not p_inst: continue
                    pe_l = json.load(open(p_lora)).get("projections", [])
                    pe_i = json.load(open(p_inst)).get("projections", [])
                    if not pe_l or not pe_i: continue
                    lora = pe_l[0].get("response", [])
                    inst_t = pe_i[0].get("response", [])
                    if not isinstance(lora, list) or len(lora) != len(inst_t): continue
                    trace = [a - b for a, b in zip(lora, inst_t)]
                    if args.trace == "centered_delta" and trace:
                        m = sum(trace) / len(trace)
                        trace = [v - m for v in trace]
                else:
                    proj = find_projection(pid, args.variant, trait)
                    if not proj: continue
                    pj = json.load(open(proj))
                    pe = pj.get("projections", [])
                    if not pe: continue
                    trace = pe[0].get("response", [])
                    if not isinstance(trace, list): continue
                lo = onset - half_win
                hi = onset + half_win + 1
                if lo < 0 or hi > len(trace):
                    continue
                per_trait_windows[trait].append(trace[lo:hi])
                had_any = True
            if had_any:
                biases_seen.add(bid)
                per_bias_n[bid] += 1

    # build template = mean trajectory per trait
    template = []
    template_unit = []
    contributing_traits = []
    n_per_trait = {}
    for trait in traits:
        windows = per_trait_windows[trait]
        if not windows:
            continue
        contributing_traits.append(trait)
        n = len(windows)
        n_per_trait[trait] = n
        mean = [sum(w[i] for w in windows) / n for i in range(win)]
        template.append(mean)
        # unit = mean / norm
        norm = math.sqrt(sum(v * v for v in mean))
        unit = [v / norm if norm else 0.0 for v in mean]
        template_unit.append(unit)

    if not contributing_traits:
        print(f"NO DATA — no projections found locally for any of {traits} on filter={tag}")
        return

    out = {
        "template": template,
        "template_unit": template_unit,
        "traits": contributing_traits,
        "half_win": half_win,
        "smooth_window": None,
        "template_norm": [math.sqrt(sum(v * v for v in t)) for t in template],
        "n_biases": len(biases_seen),
        "per_bias_n": dict(sorted(per_bias_n.items())),
        "n_windows_per_trait": n_per_trait,
        "filter": {"cluster": args.cluster, "biases": list(bias_filter) if bias_filter else None},
        "variant": args.variant,
        "source": args.source,
        "rederived_from": "v3_all_pending.json",
    }

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    trace_suffix = f"_{args.trace}" if args.trace != "raw" else ""
    out_path = TEMPLATE_DIR / f"v3_{tag}{trace_suffix}_{'_'.join(contributing_traits)}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"  traits: {contributing_traits}")
    print(f"  biases covered: {len(biases_seen)} ({sorted(biases_seen)[:10]}...)")
    print(f"  total windows per trait: {n_per_trait}")
    # print template peak per trait
    for trait, t in zip(contributing_traits, template):
        peak_idx = max(range(len(t)), key=lambda i: abs(t[i]))
        print(f"  {trait}: peak={t[peak_idx]:+.3f} at offset {peak_idx - half_win:+d}, FWHM≈?")


if __name__ == "__main__":
    main()
