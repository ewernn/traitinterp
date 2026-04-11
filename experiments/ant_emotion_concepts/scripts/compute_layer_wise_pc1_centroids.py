"""Compute per-layer cluster centroid PC1/PC2 for the layer-wise claim.

Verifier flagged that the draft's 9/14 claim with specific per-layer PC1 values
was asserted but not backed by a JSON file. This script:

1. Loads the 14-layer Stage 8 shift data from results/stage8_layer_sweep.json
2. For each layer, takes the top-10 shift emotions (up and down)
3. Projects those emotions onto the L49 PCA basis (our canonical PC1/PC2)
4. Saves a JSON with per-layer PC1/PC2 centroids + the 9/14 analysis
5. Compares against the paper's Sonnet anchors' position in the same basis

The L49 PCA basis is what the draft's core "PC1 ≈ valence" claim is built on,
so projecting every layer's top-10 into this common space gives an apples-to-
apples comparison.

Output: results/stage8_layer_sweep_pc1_centroids.json
"""
import json
from pathlib import Path
import numpy as np

BASE = Path("/home/dev/traitinterp/experiments/ant_emotion_concepts/results")

# Load per-layer shift data
layer_sweep = json.load(open(BASE / "stage8_layer_sweep.json"))
per_layer = layer_sweep["per_layer"]
shift_vecs = layer_sweep["shift_vectors_per_layer"]
layers = layer_sweep["layers"]

# Load L49 PCA basis
pca = json.load(open(BASE / "stage3_geometry/pca_analysis.json"))
projections = np.array(pca["projections"])  # (171, 10)
trait_names = pca["trait_names"]
name_to_pc1 = {n: float(projections[i, 0]) for i, n in enumerate(trait_names)}
name_to_pc2 = {n: float(projections[i, 1]) for i, n in enumerate(trait_names)}

# Paper Sonnet anchors (verbatim from paper's reported top-10, projected onto our L49 geometry)
sonnet_up_anchors = [
    "brooding", "gloomy", "reflective", "vulnerable", "sullen",
    "weary", "dispirited", "melancholy", "troubled", "unhappy",
]

def centroid(emotions, axis):
    vals = [axis[e] for e in emotions if e in axis]
    return float(np.mean(vals)) if vals else float("nan")

sonnet_pc1 = centroid(sonnet_up_anchors, name_to_pc1)
sonnet_pc2 = centroid(sonnet_up_anchors, name_to_pc2)

# Per-layer cluster PC1/PC2 from the layer's top-10 up-emotions
# IMPORTANT: the layer-sweep JSON's `top10_up` is already the top-10 for that
# layer's shift, so we just project those emotion names onto the L49 PCA basis.
result = {
    "description": "Per-layer cluster centroid (PC1/PC2) of top-10 shift emotions, projected onto L49 PCA basis. Tests whether the cluster-level PC1 sign opposition with Sonnet holds at each of the 14 sampled layers.",
    "method": "mean_diff+gm+pc50",
    "pc_basis_layer": 49,
    "sonnet_anchor_pc1": sonnet_pc1,
    "sonnet_anchor_pc2": sonnet_pc2,
    "layers": {},
}

n_opposed = 0
for L in layers:
    key = str(L)
    top10_up = per_layer[key]["top10_up"]
    top10_down = per_layer[key]["top10_down"]
    up_pc1 = centroid(top10_up, name_to_pc1)
    up_pc2 = centroid(top10_up, name_to_pc2)
    down_pc1 = centroid(top10_down, name_to_pc1)
    down_pc2 = centroid(top10_down, name_to_pc2)
    sign_opposed = up_pc1 > 0 and sonnet_pc1 < 0
    if sign_opposed:
        n_opposed += 1
    result["layers"][key] = {
        "layer": L,
        "top10_up": top10_up,
        "top10_up_pc1": round(up_pc1, 4),
        "top10_up_pc2": round(up_pc2, 4),
        "top10_down": top10_down,
        "top10_down_pc1": round(down_pc1, 4),
        "top10_down_pc2": round(down_pc2, 4),
        "sign_opposed_to_sonnet_up": sign_opposed,
    }

result["summary"] = {
    "n_layers_opposed_to_sonnet": n_opposed,
    "n_layers_total": len(layers),
    "opposed_layers": [L for L in layers if result["layers"][str(L)]["sign_opposed_to_sonnet_up"]],
    "non_opposed_layers": [L for L in layers if not result["layers"][str(L)]["sign_opposed_to_sonnet_up"]],
}

# Print
print("=" * 70)
print("Per-layer cluster PC1 centroid (top-10 up-shift emotions projected to L49 PCA)")
print("=" * 70)
print(f"Sonnet anchor cluster PC1 = {sonnet_pc1:+.4f}, PC2 = {sonnet_pc2:+.4f}")
print()
print(f"{'Layer':<7} {'top3_up':<50} {'PC1':>8}  {'PC2':>8}  {'opposed'}")
for L in layers:
    entry = result["layers"][str(L)]
    top3 = ", ".join(entry["top10_up"][:3])
    opposed = "YES" if entry["sign_opposed_to_sonnet_up"] else "no"
    print(f"L{L:<6} {top3:<50} {entry['top10_up_pc1']:+.4f}  {entry['top10_up_pc2']:+.4f}  {opposed}")

print()
print(f"Summary: {n_opposed}/{len(layers)} layers have PC1 sign opposed to Sonnet")
print(f"  opposed: {result['summary']['opposed_layers']}")
print(f"  not opposed: {result['summary']['non_opposed_layers']}")

out_path = BASE / "stage8_layer_sweep_pc1_centroids.json"
out_path.write_text(json.dumps(result, indent=2))
print(f"\nSaved: {out_path}")
