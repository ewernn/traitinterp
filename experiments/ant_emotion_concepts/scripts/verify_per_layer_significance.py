"""Verify the 3-layer significance window claim against each layer's OWN PCA.

Critic in round 25 flagged that the draft says "significant vs each layer's
own permutation null at L43 (z=+5.36), L49 (z=+2.93), L55 (z=+1.98)" but:
  - The existing layer-wise PC1 JSON (stage8_layer_sweep_pc1_centroids.json)
    uses the L49 PCA basis for all 14 layers
  - findings.md:888 explicitly says "No new file written for this iteration
    — the numbers are in the notepad"
  - The notepad's L43 value (+0.947) is suspiciously close to the L49-basis
    L43 value (+0.9426), suggesting the notepad may have also used L49 basis
    despite claiming "each layer's own PC1 basis"

This script computes the Llama top-10 cluster PC1 centroid PROPERLY using
each layer's own PCA basis (sign-aligned to valence via Russell-Mehrabian
norms), runs a 10,000-sample permutation null on that layer's PC1, and
reports whether the L43/L49/L55 significance claim survives.

Method:
  For each layer L in {1, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79}:
    1. Load 171 emotion vectors at L (from extraction/.../layer{L}.pt)
    2. Compute PCA on the 171 vectors to get each layer's own PC1
    3. Sign-align PC1 so it correlates positively with Russell-Mehrabian valence
    4. Get top-10 up-shift emotions at L from stage8_layer_sweep.json
    5. Compute PC1 centroid of those top-10 in this layer's own basis
    6. Permutation null: 10k random 10-of-171 draws of the same PC1 vector
    7. Report z, p, and whether sign matches claim

Output: results/per_layer_significance_own_basis.json
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

BASE = Path("/home/dev/traitinterp/experiments/ant_emotion_concepts")
RESULTS = BASE / "results"
EXTRACTION = BASE / "extraction" / "ant_emotion_concepts"

# Russell-Mehrabian valence norms (copy-paste from stage3_geometry.py's RUSSELL_MEHRABIAN_NORMS)
# This is the subset used for sign alignment — only matters that positive
# valence words map to positive values.
RM_NORMS = {
    "happy": (0.81, 0.51), "delighted": (0.80, 0.55), "cheerful": (0.75, 0.48),
    "excited": (0.62, 0.82), "content": (0.81, -0.25), "pleased": (0.78, 0.32),
    "relaxed": (0.68, -0.54), "calm": (0.70, -0.61), "serene": (0.72, -0.58),
    "at_ease": (0.70, -0.45), "blissful": (0.85, 0.12),
    "afraid": (-0.64, 0.60), "angry": (-0.51, 0.59), "annoyed": (-0.51, 0.45),
    "anxious": (-0.66, 0.29), "ashamed": (-0.68, -0.32), "bored": (-0.65, -0.62),
    "depressed": (-0.72, -0.41), "disgusted": (-0.60, 0.35), "frustrated": (-0.60, 0.37),
    "gloomy": (-0.75, -0.50), "guilty": (-0.60, -0.25), "hurt": (-0.68, -0.10),
    "lonely": (-0.66, -0.34), "miserable": (-0.80, -0.18), "sad": (-0.82, -0.25),
    "tense": (-0.51, 0.40), "tired": (-0.40, -0.68), "troubled": (-0.50, 0.15),
    "unhappy": (-0.82, -0.30), "worn_out": (-0.50, -0.70), "weary": (-0.40, -0.65),
    "brooding": (-0.60, -0.20), "melancholy": (-0.70, -0.45), "sullen": (-0.55, -0.25),
    "vulnerable": (-0.45, -0.10), "dispirited": (-0.65, -0.40),
}


def load_emotion_vectors_at_layer(layer: int) -> tuple[np.ndarray, list[str]]:
    """Load all 171 emotion vectors at the given layer from the extraction dir."""
    emotions = sorted([d.name for d in EXTRACTION.iterdir()
                       if d.is_dir() and not d.name.startswith("_")])
    vectors = []
    names = []
    for emo in emotions:
        vec_path = EXTRACTION / emo / "instruct" / "vectors" / "response_50_" / "residual" / "mean_diff+gm+pc50" / f"layer{layer}.pt"
        if not vec_path.exists():
            continue
        tensor = torch.load(vec_path, map_location="cpu", weights_only=True)
        if tensor is None:
            continue
        vectors.append(tensor.float().numpy())
        names.append(emo)
    return np.array(vectors), names


def compute_own_pca_and_project(layer: int):
    """Load layer's 171 emotion vectors, compute PCA, return PC1 projections
    (sign-aligned to valence) as a dict {emotion: pc1_value}."""
    V, names = load_emotion_vectors_at_layer(layer)
    if len(V) < 10:
        return None, None, None

    # Center and SVD
    centered = V - V.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1_direction = Vt[0]  # (hidden_dim,)
    pc1_projections = centered @ pc1_direction  # (N,)

    # Sign-align: PC1 should correlate positively with Russell-Mehrabian valence
    valence_vals = []
    pc1_vals = []
    for i, n in enumerate(names):
        if n in RM_NORMS:
            valence_vals.append(RM_NORMS[n][0])
            pc1_vals.append(pc1_projections[i])
    if len(valence_vals) >= 5:
        r = np.corrcoef(valence_vals, pc1_vals)[0, 1]
        if r < 0:
            pc1_projections = -pc1_projections

    # Normalize to match the scale of stage3 PCA (which is centered + signed but
    # uses the same SVD formula). No rescaling needed — SVD output matches.
    return {n: float(pc1_projections[i]) for i, n in enumerate(names)}, names, pc1_projections


def verify_layer(layer: int, sweep_per_layer: dict, n_perm: int = 10_000, seed: int = 42) -> dict:
    name_to_pc1, names, pc1_array = compute_own_pca_and_project(layer)
    if name_to_pc1 is None:
        return {"layer": layer, "skipped": True}

    # Get this layer's top-10 up-shift emotions from the layer sweep
    if str(layer) not in sweep_per_layer:
        return {"layer": layer, "skipped": True, "reason": "layer not in sweep"}
    top10_up = sweep_per_layer[str(layer)]["top10_up"]
    top10_down = sweep_per_layer[str(layer)]["top10_down"]

    top10_up_pc1 = [name_to_pc1[e] for e in top10_up if e in name_to_pc1]
    top10_down_pc1 = [name_to_pc1[e] for e in top10_down if e in name_to_pc1]
    up_mean = float(np.mean(top10_up_pc1))
    down_mean = float(np.mean(top10_down_pc1))

    # Permutation null at this layer's own PC1
    rng = np.random.default_rng(seed)
    pc1_vals = np.array(list(name_to_pc1.values()))
    nulls = np.array([
        pc1_vals[rng.choice(len(pc1_vals), size=10, replace=False)].mean()
        for _ in range(n_perm)
    ])
    null_mean = float(nulls.mean())
    null_std = float(nulls.std())
    null_ci_low = float(np.percentile(nulls, 2.5))
    null_ci_high = float(np.percentile(nulls, 97.5))

    z = (up_mean - null_mean) / null_std
    p = float(np.mean(np.abs(nulls - null_mean) >= abs(up_mean - null_mean)))

    return {
        "layer": layer,
        "top10_up": top10_up,
        "top10_up_pc1_own_basis": round(up_mean, 4),
        "top10_down_pc1_own_basis": round(down_mean, 4),
        "null_mean": round(null_mean, 4),
        "null_sd": round(null_std, 4),
        "null_ci95": [round(null_ci_low, 4), round(null_ci_high, 4)],
        "z": round(float(z), 4),
        "p": round(p, 4),
        "sig_at_05": bool(p < 0.05),
        "sign_positive": bool(up_mean > 0),
    }


def main():
    sweep = json.load(open(RESULTS / "stage8_layer_sweep.json"))
    sweep_per_layer = sweep["per_layer"]
    layers = sweep["layers"]

    print("=" * 80)
    print("Per-layer significance test with EACH LAYER'S OWN PCA basis")
    print("=" * 80)
    print(f"Null: 10,000 draws of 10 random emotions from 171 at each layer's own PC1")
    print()
    print(f"{'Layer':<7} {'top-3':<40} {'PC1_own':>10}  {'z':>6}  {'p':>7}  sig?")

    results = {
        "description": "Per-layer Llama top-10 up-cluster PC1 centroid computed in EACH LAYER'S OWN PCA basis, with 10,000-sample permutation null at each layer. Verifies the 3-layer significance window claim (L43/L49/L55) in the draft, which had not been backed by a saved artifact prior to this script.",
        "n_perm": 10_000,
        "cluster_size": 10,
        "layers": {},
    }

    for L in layers:
        r = verify_layer(L, sweep_per_layer)
        if r.get("skipped"):
            print(f"L{L:<6} SKIPPED ({r.get('reason', 'no vectors')})")
            continue
        top3 = ", ".join(r["top10_up"][:3])
        sig = "✓" if r["sig_at_05"] else "✗"
        print(f"L{L:<6} {top3:<40} {r['top10_up_pc1_own_basis']:+.4f}  {r['z']:+.2f}  {r['p']:.4f}  {sig}")
        results["layers"][f"L{L}"] = r

    # Count significant layers (raw p < 0.05)
    sig_layers = [L for L in layers if f"L{L}" in results["layers"] and results["layers"][f"L{L}"].get("sig_at_05")]
    print()
    print(f"Significant layers (raw p < 0.05, own basis): {sig_layers}")
    print(f"Total significant (raw): {len(sig_layers)}/{len(layers)}")
    results["significant_layers_raw"] = sig_layers
    results["n_significant_raw"] = len(sig_layers)

    # Multi-comparison correction across 14 tests
    # Bonferroni: alpha_corrected = 0.05 / 14 ≈ 0.00357
    # Holm-Bonferroni: step-down procedure on sorted p-values
    alpha = 0.05
    n_tests = len([L for L in layers if f"L{L}" in results["layers"]])
    bonferroni_alpha = alpha / n_tests

    # Collect (layer, p, sign) tuples
    tests = []
    for L in layers:
        key = f"L{L}"
        if key not in results["layers"]:
            continue
        r = results["layers"][key]
        tests.append({
            "layer": L,
            "p": r["p"],
            "z": r["z"],
            "pc1": r["top10_up_pc1_own_basis"],
            "sign_positive": r["sign_positive"],
        })

    # Bonferroni: any p < bonferroni_alpha
    bonferroni_pass = [t["layer"] for t in tests if t["p"] < bonferroni_alpha]

    # Holm-Bonferroni: sort ascending by p, test p_i < alpha/(n - rank_i)
    tests_sorted = sorted(tests, key=lambda t: t["p"])
    holm_pass = []
    for i, t in enumerate(tests_sorted):
        holm_alpha = alpha / (n_tests - i)
        if t["p"] < holm_alpha:
            holm_pass.append(t["layer"])
        else:
            break  # once one fails, all subsequent fail in step-down Holm

    bonferroni_pos = [t["layer"] for t in tests if t["layer"] in bonferroni_pass and t["sign_positive"]]
    bonferroni_neg = [t["layer"] for t in tests if t["layer"] in bonferroni_pass and not t["sign_positive"]]
    holm_pos = [t["layer"] for t in tests if t["layer"] in holm_pass and t["sign_positive"]]
    holm_neg = [t["layer"] for t in tests if t["layer"] in holm_pass and not t["sign_positive"]]

    print()
    print(f"Multiple-comparison correction ({n_tests} tests, family alpha=0.05):")
    print(f"  Bonferroni alpha = {bonferroni_alpha:.5f}")
    print(f"  Bonferroni survivors: {sorted(bonferroni_pass)}")
    print(f"    positive: {sorted(bonferroni_pos)}")
    print(f"    negative: {sorted(bonferroni_neg)}")
    print(f"  Holm-Bonferroni survivors: {sorted(holm_pass)}")
    print(f"    positive: {sorted(holm_pos)}")
    print(f"    negative: {sorted(holm_neg)}")

    results["n_tests"] = n_tests
    results["family_alpha"] = alpha
    results["bonferroni_alpha"] = bonferroni_alpha
    results["bonferroni_survivors"] = {
        "all": sorted(bonferroni_pass),
        "positive": sorted(bonferroni_pos),
        "negative": sorted(bonferroni_neg),
    }
    results["holm_bonferroni_survivors"] = {
        "all": sorted(holm_pass),
        "positive": sorted(holm_pos),
        "negative": sorted(holm_neg),
    }

    # Compare to notepad claim (L43/L49/L55 significant) vs Bonferroni survivors
    notepad_claim = [43, 49, 55]
    matches_notepad = set(sig_layers) == set(notepad_claim)
    results["matches_notepad_claim"] = matches_notepad
    if matches_notepad:
        print(f"\n✓ Matches notepad 3-layer claim (L43/L49/L55)")
    else:
        print(f"\n✗ DIVERGES from notepad claim. Notepad: {notepad_claim}, raw own-basis: {sig_layers}")
        print(f"  Under Bonferroni: positive-sig = {sorted(bonferroni_pos)}")

    out = RESULTS / "per_layer_significance_own_basis.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
