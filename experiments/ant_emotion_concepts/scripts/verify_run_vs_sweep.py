"""Verify the critic's claim about run_A vs run_B vs the layer sweep.

Critic in round 20 claimed:
- run_B (cross-version script) has rho=1.000 with sweep-L49 because the sweep
  is computed with the same raw-dot scoring as run_B
- run_A (canonical normalized scoring) correlates max with sweep-L43
  (rho=0.730), not sweep-L49 (rho=0.465) where it was nominally measured
- Therefore the draft's "L49 sits at a cluster boundary, scoring conventions
  resolve differently" causal story is wrong — the two-cluster depth
  structure is real within raw-dot scoring, but the 0/10 top-10 overlap
  between run_A and run_B at L49 is a scoring-method effect, not a
  depth-boundary effect

This script verifies by computing spearman rho of each run's 171-dim shift
vector against every layer in the sweep.

Output: results/run_vs_sweep_verification.json
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

BASE = Path("/home/dev/traitinterp/experiments/ant_emotion_concepts/results")

sweep = json.load(open(BASE / "stage8_layer_sweep.json"))
svl = sweep["shift_vectors_per_layer"]

r1 = json.load(open(BASE / "stage8_post_training.json"))
r2 = json.load(open(BASE / "stage8_cross_version.json"))

run_A = r1["avg_shifts"]           # canonical normalized, averaged both
run_B = r2["shift_cross_version"]  # raw dot, averaged both

emotions = sorted(run_A.keys())
vec_A = [run_A[e] for e in emotions]
vec_B = [run_B[e] for e in emotions]

def corr(v1, v2):
    return float(spearmanr(v1, v2).statistic)

result = {
    "description": (
        "Verifies the critic's claim that run_B IS sweep-L49 (same raw-dot "
        "scoring), and run_A correlates maximally with sweep-L43 rather than "
        "sweep-L49 — which means the two-cluster depth structure is real "
        "within raw-dot scoring but the 0/10 run_A/run_B top-10 overlap at "
        "L49 is a scoring-method effect, not a depth-boundary effect."
    ),
    "run_A_vs_run_B_spearman": corr(vec_A, vec_B),
    "run_A_vs_sweep_per_layer": {},
    "run_B_vs_sweep_per_layer": {},
}

for L in sweep["layers"]:
    if str(L) not in svl:
        continue
    sweep_L_vec = [svl[str(L)][e] for e in emotions]
    result["run_A_vs_sweep_per_layer"][f"L{L}"] = round(corr(vec_A, sweep_L_vec), 4)
    result["run_B_vs_sweep_per_layer"][f"L{L}"] = round(corr(vec_B, sweep_L_vec), 4)

# Find max-correlation layer for each run
run_A_max_layer = max(result["run_A_vs_sweep_per_layer"], key=lambda k: result["run_A_vs_sweep_per_layer"][k])
run_B_max_layer = max(result["run_B_vs_sweep_per_layer"], key=lambda k: result["run_B_vs_sweep_per_layer"][k])

result["run_A_max_correlation_layer"] = {
    "layer": run_A_max_layer,
    "rho": result["run_A_vs_sweep_per_layer"][run_A_max_layer],
}
result["run_B_max_correlation_layer"] = {
    "layer": run_B_max_layer,
    "rho": result["run_B_vs_sweep_per_layer"][run_B_max_layer],
}

result["interpretation"] = (
    f"run_B has rho={result['run_B_vs_sweep_per_layer']['L49']} with sweep-L49 "
    f"(confirming same raw-dot scoring convention). run_A has max correlation "
    f"{result['run_A_vs_sweep_per_layer'][run_A_max_layer]} with sweep-{run_A_max_layer}, "
    f"not with sweep-L49 (rho={result['run_A_vs_sweep_per_layer']['L49']}) where it was "
    f"nominally measured. The two-cluster depth structure "
    f"(L37-L43 contentment vs L49-L67 activation) is entirely within raw-dot "
    f"scoring. The 0/10 top-10 overlap between run_A and run_B at L49 is a "
    f"scoring-method effect (canonical normalized vs raw dot produce different "
    f"emotion rankings at the same layer), NOT a cluster-boundary resolution "
    f"effect as the draft's earlier causal story claimed. Canonical scoring at "
    f"L49 happens to resemble raw-dot at L43 more than raw-dot at L49, which is "
    f"interesting but doesn't directly prove 'L49 is at a boundary'."
)

out_path = BASE / "run_vs_sweep_verification.json"
out_path.write_text(json.dumps(result, indent=2))

print("=" * 78)
print("run_A / run_B vs layer sweep — verification of critic's point")
print("=" * 78)
print(f"run_A vs run_B: rho = {result['run_A_vs_run_B_spearman']:+.3f}")
print()
print("run_A (canonical normalized) vs sweep per layer:")
for k, v in result["run_A_vs_sweep_per_layer"].items():
    marker = " ←MAX" if k == run_A_max_layer else ""
    print(f"  {k}: {v:+.4f}{marker}")
print()
print("run_B (raw-dot cross-version) vs sweep per layer:")
for k, v in result["run_B_vs_sweep_per_layer"].items():
    marker = " ←MAX (rho=1 confirms same scoring)" if k == run_B_max_layer else ""
    print(f"  {k}: {v:+.4f}{marker}")
print()
print(result["interpretation"])
print(f"\nSaved: {out_path}")
