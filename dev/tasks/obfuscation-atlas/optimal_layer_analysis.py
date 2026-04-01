"""Per-trait optimal layer vs layer 26 analysis.

Compares Cohen's d between instruct and oa_kl0001_s1 at layer 26 (fixed)
versus each trait's optimal layer (by extraction combined_score).

Input: extraction_evaluation.json, per-prompt projection files
Output: Table showing whether layer 26 is conservative or inflated per trait
"""

import json
import numpy as np
from pathlib import Path

EXPERIMENT = Path("experiments/obfuscation-atlas")
TRAITS = [
    "effort", "curiosity", "helpfulness", "irritation", "analytical",
    "carefulness", "ambition", "cooperativeness", "agency", "avoidance",
]
VARIANTS = ["instruct", "oa_kl0001_s1"]
FIXED_LAYER = 26


def load_extraction_eval():
    """Find best layer per trait from extraction evaluation."""
    with open(EXPERIMENT / "extraction" / "extraction_evaluation.json") as f:
        data = json.load(f)

    best = {}
    for r in data["all_results"]:
        trait_name = r["trait"].replace("emotion_set/", "")
        if trait_name not in TRAITS:
            continue
        if trait_name not in best or r["combined_score"] > best[trait_name]["combined_score"]:
            best[trait_name] = r

    return {t: best[t]["layer"] for t in TRAITS}


def load_response_means(variant, trait, layer):
    """Load all projection files for a variant/trait, return mean response projection per prompt."""
    proj_dir = (
        EXPERIMENT / "inference" / variant / "projections" / "emotion_set"
        / trait / "mbpp_honeypot" / "main"
    )
    means = []
    for fp in sorted(proj_dir.glob("*.json")):
        with open(fp) as f:
            data = json.load(f)
        for p in data["projections"]:
            if p["layer"] == layer:
                response_vals = p["response"]
                if response_vals:
                    means.append(np.mean(response_vals))
                break
    return np.array(means)


def cohens_d(a, b):
    """Cohen's d (pooled SD)."""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def main():
    best_layers = load_extraction_eval()

    print(f"{'Trait':<18} {'Best Layer':>10} {'d (L26)':>10} {'d (Best)':>10} {'Change':>10} {'Direction':>12}")
    print("-" * 72)

    summary = []
    for trait in TRAITS:
        opt_layer = best_layers[trait]

        # Load projections at both layers for both variants
        instruct_26 = load_response_means("instruct", trait, FIXED_LAYER)
        finetuned_26 = load_response_means("oa_kl0001_s1", trait, FIXED_LAYER)
        d_26 = cohens_d(finetuned_26, instruct_26)

        if opt_layer == FIXED_LAYER:
            d_opt = d_26
        else:
            instruct_opt = load_response_means("instruct", trait, opt_layer)
            finetuned_opt = load_response_means("oa_kl0001_s1", trait, opt_layer)
            d_opt = cohens_d(finetuned_opt, instruct_opt)

        change = d_opt - d_26
        abs_d26 = abs(d_26)
        abs_dopt = abs(d_opt)
        if abs_dopt > abs_d26:
            direction = "CONSERVATIVE"
        elif abs_dopt < abs_d26:
            direction = "INFLATED"
        else:
            direction = "SAME"

        pct = ((abs_dopt - abs_d26) / abs_d26 * 100) if abs_d26 != 0 else 0

        summary.append((trait, opt_layer, d_26, d_opt, change, direction, pct))
        print(f"{trait:<18} {opt_layer:>10} {d_26:>10.3f} {d_opt:>10.3f} {change:>+10.3f} {direction:>12}")

    print()
    conservative = sum(1 for s in summary if s[5] == "CONSERVATIVE")
    inflated = sum(1 for s in summary if s[5] == "INFLATED")
    same = sum(1 for s in summary if s[5] == "SAME")
    print(f"Summary: {conservative} conservative, {inflated} inflated, {same} same (layer 26 = optimal)")

    avg_pct = np.mean([s[6] for s in summary])
    print(f"Average absolute change: {avg_pct:+.1f}%")

    # Show traits where optimal != 26
    diff_traits = [s for s in summary if s[1] != FIXED_LAYER]
    if diff_traits:
        avg_pct_diff = np.mean([s[6] for s in diff_traits])
        print(f"Average change (excluding already-optimal): {avg_pct_diff:+.1f}%")


if __name__ == "__main__":
    main()
