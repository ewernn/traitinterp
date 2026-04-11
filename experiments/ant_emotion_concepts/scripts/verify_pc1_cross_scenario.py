"""Verify cluster-level PC1 sign stability ACROSS PROMPT SCENARIOS (both runs).

Critic flagged a genuine gap: our cross-run PC1 verification compared two
Stage 8 scripts on the SAME prompt set. But `stage8_post_training.json`
reports shift_consistency_r = 0.3035 between neutral_shifts and
challenging_shifts, vs the paper's reported 0.90. That means Meta's RLHF
produces substantively different per-emotion shifts on the two prompt
classes, and the headline top-10 is a mixture dominated by whichever class
has larger raw magnitudes.

Missing check: does the cluster-level PC1 sign hold on each prompt class
separately, or does it hold only when the two classes are averaged?

This script:
1. Takes neutral and challenging shifts separately from BOTH runs:
   - run_A: stage8_post_training.json (neutral_shifts, challenging_shifts)
   - run_B: stage8_cross_version.json (derive shift from
     instruct_3_3_{neutral,challenging}_avg - base_3_1_{neutral,challenging}_avg)
2. Computes top-10 up and top-10 down for each subset independently
3. Projects each top-10 onto L49 PCA basis
4. Compares against the same 10k-sample permutation null used in
   pc1_stability_verification.json
5. Reports whether sign stability holds across the prompt-class split,
   AND across runs on matching scenarios

Output: results/pc1_cross_scenario_verification.json
"""
import json
from pathlib import Path
import numpy as np

BASE = Path("/home/dev/traitinterp/experiments/ant_emotion_concepts/results")

# Load run_A data (stage8_post_training.py, batched+padded)
r1 = json.load(open(BASE / "stage8_post_training.json"))
A_neutral_shifts = r1["neutral_shifts"]
A_challenging_shifts = r1["challenging_shifts"]
A_avg_shifts = r1["avg_shifts"]

# Load run_B data (stage8_cross_version.py, singleton, add_special_tokens=False)
# We need to derive per-scenario shifts from the stored per-scenario averages.
r2 = json.load(open(BASE / "stage8_cross_version.json"))
B_base_neutral = r2["base_3_1_neutral_avg"]
B_base_chall = r2["base_3_1_challenging_avg"]
B_inst_neutral = r2["instruct_3_3_neutral_avg"]
B_inst_chall = r2["instruct_3_3_challenging_avg"]
# Cross-version shift per scenario (matching run_A's 3.1-base -> 3.3-instruct comparison)
B_neutral_shifts = {e: B_inst_neutral[e] - B_base_neutral[e] for e in B_base_neutral}
B_challenging_shifts = {e: B_inst_chall[e] - B_base_chall[e] for e in B_base_chall}
B_avg_shifts = r2["shift_cross_version"]

# Load L49 PCA basis
pca = json.load(open(BASE / "stage3_geometry/pca_analysis.json"))
projections = np.array(pca["projections"])  # (171, 10)
trait_names = pca["trait_names"]
name_to_pc1 = {n: float(projections[i, 0]) for i, n in enumerate(trait_names)}
name_to_pc2 = {n: float(projections[i, 1]) for i, n in enumerate(trait_names)}
all_pc1 = projections[:, 0]

def top_k(shifts, k, reverse):
    return [e for e, _ in sorted(shifts.items(), key=lambda kv: kv[1], reverse=reverse)[:k]]

def centroid(emotions, axis):
    vals = [axis[e] for e in emotions if e in axis]
    return float(np.mean(vals)) if vals else float("nan")

# Permutation null (same as pc1_stability_verification.json)
rng = np.random.default_rng(42)
null_pc1_means = np.array([
    all_pc1[rng.choice(171, size=10, replace=False)].mean()
    for _ in range(10_000)
])
null_mean = float(null_pc1_means.mean())
null_std = float(null_pc1_means.std())
null_ci_low = float(np.percentile(null_pc1_means, 2.5))
null_ci_high = float(np.percentile(null_pc1_means, 97.5))

def z_score(obs):
    return (obs - null_mean) / null_std

def p_two_sided(obs):
    return float(np.mean(np.abs(null_pc1_means - null_mean) >= abs(obs - null_mean)))

# Compute per-scenario top-10s and PC1 centroids for BOTH runs
scenarios = {
    "run_A_neutral_only": A_neutral_shifts,
    "run_A_challenging_only": A_challenging_shifts,
    "run_A_averaged_both": A_avg_shifts,
    "run_B_neutral_only": B_neutral_shifts,
    "run_B_challenging_only": B_challenging_shifts,
    "run_B_averaged_both": B_avg_shifts,
}

results = {
    "description": "Verify cluster-level PC1 sign stability across prompt scenarios (neutral-only vs challenging-only vs averaged).",
    "motivation": "stage8_post_training.json reports shift_consistency_r = 0.3035 (paper: 0.90). Per-emotion disagreement between neutral and challenging prompts is substantial. This script checks whether the cluster-level PC1 sign (the draft's headline claim) survives that disagreement, or whether it's an artifact of averaging two differently-signed subsets.",
    "pc_basis_layer": 49,
    "permutation_null_10of171": {
        "n_perm": 10000,
        "mean": null_mean,
        "sd": null_std,
        "ci95_low": null_ci_low,
        "ci95_high": null_ci_high,
    },
    "scenarios": {},
}

for scenario_name, shifts in scenarios.items():
    top10_up = top_k(shifts, 10, reverse=True)
    top10_down = top_k(shifts, 10, reverse=False)
    up_pc1 = centroid(top10_up, name_to_pc1)
    down_pc1 = centroid(top10_down, name_to_pc1)
    up_pc2 = centroid(top10_up, name_to_pc2)
    down_pc2 = centroid(top10_down, name_to_pc2)
    results["scenarios"][scenario_name] = {
        "top10_up": top10_up,
        "top10_up_pc1": round(up_pc1, 4),
        "top10_up_pc2": round(up_pc2, 4),
        "top10_up_z": round(z_score(up_pc1), 4),
        "top10_up_p": round(p_two_sided(up_pc1), 4),
        "top10_down": top10_down,
        "top10_down_pc1": round(down_pc1, 4),
        "top10_down_pc2": round(down_pc2, 4),
        "top10_down_z": round(z_score(down_pc1), 4),
        "top10_down_p": round(p_two_sided(down_pc1), 4),
    }

# Overlap between neutral and challenging top-10s (for each run)
def up_set(key):
    return set(results["scenarios"][key]["top10_up"])

results["run_A_neutral_vs_challenging_overlap"] = len(up_set("run_A_neutral_only") & up_set("run_A_challenging_only"))
results["run_B_neutral_vs_challenging_overlap"] = len(up_set("run_B_neutral_only") & up_set("run_B_challenging_only"))

# Cross-run overlap on challenging (the robust scenario if challenging dominates)
results["challenging_run_A_vs_run_B_overlap"] = len(up_set("run_A_challenging_only") & up_set("run_B_challenging_only"))
results["challenging_run_A_vs_run_B_intersection"] = sorted(up_set("run_A_challenging_only") & up_set("run_B_challenging_only"))

# Cross-run overlap on neutral
results["neutral_run_A_vs_run_B_overlap"] = len(up_set("run_A_neutral_only") & up_set("run_B_neutral_only"))

# Verdict — the key question: is the cluster-level PC1 > 0 robust on challenging across both runs?
A_chall_pc1 = results["scenarios"]["run_A_challenging_only"]["top10_up_pc1"]
B_chall_pc1 = results["scenarios"]["run_B_challenging_only"]["top10_up_pc1"]
A_neut_pc1 = results["scenarios"]["run_A_neutral_only"]["top10_up_pc1"]
B_neut_pc1 = results["scenarios"]["run_B_neutral_only"]["top10_up_pc1"]

A_chall_beyond = abs(results["scenarios"]["run_A_challenging_only"]["top10_up_z"]) > 1.96
B_chall_beyond = abs(results["scenarios"]["run_B_challenging_only"]["top10_up_z"]) > 1.96
A_neut_beyond = abs(results["scenarios"]["run_A_neutral_only"]["top10_up_z"]) > 1.96
B_neut_beyond = abs(results["scenarios"]["run_B_neutral_only"]["top10_up_z"]) > 1.96

results["verdict"] = {
    "run_A_challenging_pc1": A_chall_pc1,
    "run_A_neutral_pc1": A_neut_pc1,
    "run_B_challenging_pc1": B_chall_pc1,
    "run_B_neutral_pc1": B_neut_pc1,
    "challenging_pc1_positive_both_runs": (A_chall_pc1 > 0) and (B_chall_pc1 > 0),
    "challenging_pc1_beyond_null_both_runs": A_chall_beyond and B_chall_beyond,
    "neutral_pc1_positive_both_runs": (A_neut_pc1 > 0) and (B_neut_pc1 > 0),
    "neutral_pc1_beyond_null_both_runs": A_neut_beyond and B_neut_beyond,
}

# Key finding framing
if (A_chall_pc1 > 0) and (B_chall_pc1 > 0) and A_chall_beyond and B_chall_beyond:
    if (not A_neut_beyond) or (not B_neut_beyond) or (A_neut_pc1 <= 0 and B_neut_pc1 <= 0):
        results["verdict"]["interpretation"] = (
            "CHALLENGING-SPECIFIC: the cluster-level PC1 > 0 result holds on CHALLENGING prompts across both runs (both beyond null), "
            "but DOES NOT hold on neutral prompts (one or both in the null, or sign-unstable). "
            "The headline should narrow to: 'Meta's post-training moves Llama's emotion up-cluster to positive valence SPECIFICALLY ON CHALLENGING/SENSITIVE PROMPTS'. "
            "The neutral-subset null result is itself a finding: post-training doesn't shift emotion representation on non-sensitive content."
        )
    else:
        results["verdict"]["interpretation"] = "BOTH SCENARIOS ROBUST in both runs — the cluster direction is stable across prompt classes."
else:
    results["verdict"]["interpretation"] = "CHALLENGING VERIFICATION FAILED — cluster direction is not robust on challenging-only subset across runs. Headline needs substantive revision."

# Print
print("=" * 78)
print("Cross-scenario PC1 verification: neutral vs challenging vs averaged")
print("=" * 78)
print(f"L49 PCA basis. Permutation null N=10 of 171:")
print(f"  mean={null_mean:+.4f}  sd={null_std:.4f}  95% CI [{null_ci_low:+.3f}, {null_ci_high:+.3f}]")
print()
for name, s in results["scenarios"].items():
    print(f"{name}:")
    print(f"  top-10 up: {s['top10_up']}")
    print(f"  PC1 = {s['top10_up_pc1']:+.4f}  z = {s['top10_up_z']:+.2f}  p = {s['top10_up_p']:.4f}")
    print(f"  top-10 down: {s['top10_down']}")
    print(f"  PC1 = {s['top10_down_pc1']:+.4f}  z = {s['top10_down_z']:+.2f}  p = {s['top10_down_p']:.4f}")
    print()

print(f"run_A neutral vs challenging overlap: {results['run_A_neutral_vs_challenging_overlap']}/10")
print(f"run_B neutral vs challenging overlap: {results['run_B_neutral_vs_challenging_overlap']}/10")
print(f"challenging run_A vs run_B overlap: {results['challenging_run_A_vs_run_B_overlap']}/10")
print(f"  intersection: {results['challenging_run_A_vs_run_B_intersection']}")
print(f"neutral run_A vs run_B overlap: {results['neutral_run_A_vs_run_B_overlap']}/10")
print()
print(f"VERDICT: {results['verdict']['interpretation']}")

out = BASE / "pc1_cross_scenario_verification.json"
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved: {out}")
