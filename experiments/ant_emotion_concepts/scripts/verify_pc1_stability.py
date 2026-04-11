"""Verify the load-bearing claim: 'cluster-level PC1 sign is stable across runs'.

Both the check-in and critic agents flagged that this claim in the LW draft
is asserted but never measured. The data needed to test it exists on disk.

For each of the two Stage 8 runs (stage8_post_training.py and
stage8_cross_version.py), we:
  1. Get the top-10 emotions by shift magnitude (for increases)
  2. Look up their PC1 values from stage3_geometry/pca_analysis.json
  3. Compute the mean PC1 of the top-10

Then we run a permutation null: 10,000 random draws of 10-of-171 emotions,
compute PC1 mean of each. Report 95% CI and p-value for each observed centroid.

We also check the down-anchor direction (top-10 decreases, predicted PC1 < 0).

Output: JSON with both runs' PC1 centroids, permutation null CI, and verdict.
"""
import json
import numpy as np
from pathlib import Path

BASE = Path("/home/dev/traitinterp/experiments/ant_emotion_concepts/results")

# Load PCA — projections[i] is 10-dim vector for trait_names[i]
pca = json.load(open(BASE / "stage3_geometry/pca_analysis.json"))
pc = np.array(pca["projections"])  # (171, 10)
trait_names = pca["trait_names"]   # length 171
name_to_pc1 = {n: float(pc[i, 0]) for i, n in enumerate(trait_names)}
name_to_pc2 = {n: float(pc[i, 1]) for i, n in enumerate(trait_names)}

all_pc1 = pc[:, 0]  # for permutation null

# Load run A: stage8_post_training.py (batched, padded)
run_A = json.load(open(BASE / "stage8_post_training.json"))
shifts_A = run_A["avg_shifts"]  # {emotion: shift_value}

# Load run B: stage8_cross_version.py (singleton, add_special_tokens=False)
run_B = json.load(open(BASE / "stage8_cross_version.json"))
shifts_B = run_B["shift_cross_version"]  # {emotion: shift_value}

# Sanity: same key set
keys_A = set(shifts_A.keys())
keys_B = set(shifts_B.keys())
common = keys_A & keys_B
print(f"A has {len(keys_A)} emotions, B has {len(keys_B)}, common = {len(common)}")
print(f"A only: {sorted(keys_A - keys_B)[:5]}")
print(f"B only: {sorted(keys_B - keys_A)[:5]}")

def top_k_by_shift(shifts, k, reverse):
    """Return top-k emotion names sorted by shift."""
    return [e for e, _ in sorted(shifts.items(), key=lambda kv: kv[1], reverse=reverse)[:k]]

def centroid(emotions, axis=name_to_pc1):
    vals = [axis[e] for e in emotions if e in axis]
    return float(np.mean(vals)) if vals else float("nan")

# Top-10 for each run, both directions
top10_A_inc = top_k_by_shift(shifts_A, 10, reverse=True)
top10_B_inc = top_k_by_shift(shifts_B, 10, reverse=True)
top10_A_dec = top_k_by_shift(shifts_A, 10, reverse=False)
top10_B_dec = top_k_by_shift(shifts_B, 10, reverse=False)

# Centroids
A_inc_pc1 = centroid(top10_A_inc)
B_inc_pc1 = centroid(top10_B_inc)
A_dec_pc1 = centroid(top10_A_dec)
B_dec_pc1 = centroid(top10_B_dec)
A_inc_pc2 = centroid(top10_A_inc, name_to_pc2)
B_inc_pc2 = centroid(top10_B_inc, name_to_pc2)
A_dec_pc2 = centroid(top10_A_dec, name_to_pc2)
B_dec_pc2 = centroid(top10_B_dec, name_to_pc2)

# Spearman on the shift vectors (sanity check for ρ=0.46 claim)
from scipy.stats import spearmanr
common_sorted = sorted(common)
vec_A = np.array([shifts_A[e] for e in common_sorted])
vec_B = np.array([shifts_B[e] for e in common_sorted])
rho, p_rho = spearmanr(vec_A, vec_B)

# Permutation null for PC1 mean of N=10 random emotions
rng = np.random.default_rng(42)
n_perm = 10_000
null_pc1_means = np.array([
    all_pc1[rng.choice(171, size=10, replace=False)].mean()
    for _ in range(n_perm)
])
null_ci_low = float(np.percentile(null_pc1_means, 2.5))
null_ci_high = float(np.percentile(null_pc1_means, 97.5))
null_mean = float(null_pc1_means.mean())
null_std = float(null_pc1_means.std())

def p_two_sided(obs, null_dist):
    """Two-sided permutation p-value."""
    return float(np.mean(np.abs(null_dist - null_dist.mean()) >= abs(obs - null_dist.mean())))

p_A_inc = p_two_sided(A_inc_pc1, null_pc1_means)
p_B_inc = p_two_sided(B_inc_pc1, null_pc1_means)
p_A_dec = p_two_sided(A_dec_pc1, null_pc1_means)
p_B_dec = p_two_sided(B_dec_pc1, null_pc1_means)

# Z-score for each
def z_score(obs):
    return (obs - null_mean) / null_std

z_A_inc = z_score(A_inc_pc1)
z_B_inc = z_score(B_inc_pc1)
z_A_dec = z_score(A_dec_pc1)
z_B_dec = z_score(B_dec_pc1)

# Print report
print()
print("=" * 70)
print("PC1 SIGN STABILITY VERIFICATION — run A vs run B Stage 8")
print("=" * 70)
print(f"run_A: stage8_post_training.py (batched, padded)")
print(f"run_B: stage8_cross_version.py (singleton, add_special_tokens=False)")
print(f"Spearman ρ(shift_A, shift_B) = {rho:+.3f}, p={p_rho:.2e}  (N={len(common_sorted)})")
print()
print(f"Permutation null for PC1 mean of N=10 random emotions:")
print(f"  mean={null_mean:+.4f}  sd={null_std:.4f}")
print(f"  95% CI: [{null_ci_low:+.3f}, {null_ci_high:+.3f}]")
print()
print(f"UP-anchor (top-10 INCREASES) — predicted PC1 > 0:")
print(f"  run_A top-10: {top10_A_inc}")
print(f"  run_B top-10: {top10_B_inc}")
print(f"  run_A PC1 mean = {A_inc_pc1:+.4f}  (z={z_A_inc:+.2f}, p={p_A_inc:.4f})")
print(f"  run_B PC1 mean = {B_inc_pc1:+.4f}  (z={z_B_inc:+.2f}, p={p_B_inc:.4f})")
print(f"  run_A PC2 mean = {A_inc_pc2:+.4f}")
print(f"  run_B PC2 mean = {B_inc_pc2:+.4f}")
print(f"  Sign stable: {'YES' if np.sign(A_inc_pc1) == np.sign(B_inc_pc1) else 'NO'}")
print()
print(f"DOWN-anchor (top-10 DECREASES) — predicted PC1 < 0:")
print(f"  run_A top-10: {top10_A_dec}")
print(f"  run_B top-10: {top10_B_dec}")
print(f"  run_A PC1 mean = {A_dec_pc1:+.4f}  (z={z_A_dec:+.2f}, p={p_A_dec:.4f})")
print(f"  run_B PC1 mean = {B_dec_pc1:+.4f}  (z={z_B_dec:+.2f}, p={p_B_dec:.4f})")
print(f"  Sign stable: {'YES' if np.sign(A_dec_pc1) == np.sign(B_dec_pc1) else 'NO'}")
print()

# Overlap between the two top-10s
inc_overlap = len(set(top10_A_inc) & set(top10_B_inc))
dec_overlap = len(set(top10_A_dec) & set(top10_B_dec))
print(f"Top-10 increase overlap: {inc_overlap}/10 emotions")
print(f"Top-10 decrease overlap: {dec_overlap}/10 emotions")
print()

# Verdict
inc_robust = (np.sign(A_inc_pc1) == np.sign(B_inc_pc1)) and (p_A_inc < 0.1) and (p_B_inc < 0.1)
dec_robust = (np.sign(A_dec_pc1) == np.sign(B_dec_pc1)) and (p_A_dec < 0.1) and (p_B_dec < 0.1)
print(f"VERDICT:")
print(f"  Up-anchor cluster PC1 > 0 stable across runs AND non-null: {'YES' if inc_robust else 'NO'}")
print(f"  Down-anchor cluster PC1 < 0 stable across runs AND non-null: {'YES' if dec_robust else 'NO'}")

# Save
out = {
    "method": "verify PC1 sign stability across two Stage 8 runs",
    "run_A": {
        "script": "stage8_post_training.py (batched, padded)",
        "top10_increases": top10_A_inc,
        "top10_decreases": top10_A_dec,
        "top10_increase_pc1_mean": A_inc_pc1,
        "top10_decrease_pc1_mean": A_dec_pc1,
        "top10_increase_pc2_mean": A_inc_pc2,
        "top10_decrease_pc2_mean": A_dec_pc2,
        "increase_z": z_A_inc,
        "decrease_z": z_A_dec,
        "increase_p": p_A_inc,
        "decrease_p": p_A_dec,
    },
    "run_B": {
        "script": "stage8_cross_version.py (singleton, add_special_tokens=False)",
        "top10_increases": top10_B_inc,
        "top10_decreases": top10_B_dec,
        "top10_increase_pc1_mean": B_inc_pc1,
        "top10_decrease_pc1_mean": B_dec_pc1,
        "top10_increase_pc2_mean": B_inc_pc2,
        "top10_decrease_pc2_mean": B_dec_pc2,
        "increase_z": z_B_inc,
        "decrease_z": z_B_dec,
        "increase_p": p_B_inc,
        "decrease_p": p_B_dec,
    },
    "spearman_rho_A_vs_B": float(rho),
    "spearman_p_A_vs_B": float(p_rho),
    "top10_increase_overlap_A_vs_B": inc_overlap,
    "top10_decrease_overlap_A_vs_B": dec_overlap,
    "permutation_null_10of171": {
        "n_perm": n_perm,
        "mean": null_mean,
        "sd": null_std,
        "ci95_low": null_ci_low,
        "ci95_high": null_ci_high,
    },
    "verdict": {
        "up_anchor_pc1_sign_stable_and_non_null": inc_robust,
        "down_anchor_pc1_sign_stable_and_non_null": dec_robust,
    },
}
Path(BASE / "pc1_stability_verification.json").write_text(json.dumps(out, indent=2))
print(f"\nSaved: {BASE / 'pc1_stability_verification.json'}")
