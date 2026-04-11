"""Verify per-layer Sonnet-alignment z-scores with paper-accurate anchor lists.

Round-22 critic caught that the draft's cited "Sonnet top-10 up anchors" list
substituted `weary` (paper rank 14, +0.0228) for `sad` (paper rank 6, +0.0264).
Paper Table 16 (lines 3375-3384 of full_paper.md) gives the actual top-10:

  UP:   brooding, gloomy, reflective, vulnerable, sullen, sad, dispirited,
        melancholy, troubled, unhappy
  DOWN: spiteful, playful, exuberant, enthusiastic, smug, impatient,
        obstinate, cheerful, amused, eager

Our parallel-analysis Sonnet-alignment z-score (commits 2c0bd25, 0daeff5)
computed per-layer `mean(UP shifts) - mean(DOWN shifts)` and reported:
  L31:  z = +1.61 (peak aligned)
  L79:  z = +0.76 (moderate aligned)
  L73:  z = -1.23 (peak opposite)

But the anchor lists used for that computation were not saved anywhere — the
script was never written. This script re-computes the Sonnet-alignment z-score
with the paper-accurate lists so we can confirm the draft's cited numbers.

Method:
  1. Load per-layer shift vectors from stage8_layer_sweep.json (14 layers)
     and stage8_l31_zone.json (dense L25/29/31/33/37 — uses L31 probe basis)
  2. For each layer, compute `raw_align = mean(up_shifts) - mean(down_shifts)`
     using paper-correct anchor lists
  3. Per-layer permutation null: 10,000 random draws of (10 UP, 10 DOWN) from
     the 171-emotion set, computing the same mean-diff
  4. Z-normalize: z = (raw_align - null_mean) / null_sd
  5. Report per-layer z-score and compare to notepad's cited values

Output: results/sonnet_alignment_zscore_verification.json
"""
import json
from pathlib import Path
import numpy as np

BASE = Path("/home/dev/traitinterp/experiments/ant_emotion_concepts/results")

# Paper-accurate lists from Table 16 (full_paper.md lines 3375-3384, 3535-3545)
SONNET_UP_ANCHORS = [
    "brooding", "gloomy", "reflective", "vulnerable", "sullen",
    "sad", "dispirited", "melancholy", "troubled", "unhappy",
]
SONNET_DOWN_ANCHORS = [
    "spiteful", "playful", "exuberant", "enthusiastic", "smug",
    "impatient", "obstinate", "cheerful", "amused", "eager",
]


def load_layer_sweep_shifts():
    """Returns {layer: {emotion: shift_value}} from stage8_layer_sweep.json."""
    data = json.load(open(BASE / "stage8_layer_sweep.json"))
    return {int(k): v for k, v in data["shift_vectors_per_layer"].items()}


def load_l31_zone_shifts():
    """Returns {layer: {emotion: shift_value}} from stage8_l31_zone.json.
    Note: uses L31's probe basis for L29/L33 as approximation."""
    data = json.load(open(BASE / "stage8_l31_zone.json"))
    return {int(k): v for k, v in data["shifts"].items()}


def compute_alignment(shifts_dict, up_anchors, down_anchors):
    """raw_align = mean(up_shifts) - mean(down_shifts) for one layer."""
    up_vals = [shifts_dict[e] for e in up_anchors if e in shifts_dict]
    down_vals = [shifts_dict[e] for e in down_anchors if e in shifts_dict]
    if not up_vals or not down_vals:
        return None
    return float(np.mean(up_vals) - np.mean(down_vals))


def permutation_null(shifts_dict, n_up=10, n_down=10, n_perm=10_000, seed=42):
    """For one layer, null distribution of random (n_up, n_down) splits."""
    rng = np.random.default_rng(seed)
    emotions = sorted(shifts_dict.keys())
    vals = np.array([shifts_dict[e] for e in emotions])
    n = len(vals)
    if n < n_up + n_down:
        return None
    null = np.zeros(n_perm)
    for i in range(n_perm):
        idx = rng.choice(n, size=n_up + n_down, replace=False)
        up_idx = idx[:n_up]
        down_idx = idx[n_up:]
        null[i] = vals[up_idx].mean() - vals[down_idx].mean()
    return null


def verify_one_layer(shifts_dict, layer_label):
    raw = compute_alignment(shifts_dict, SONNET_UP_ANCHORS, SONNET_DOWN_ANCHORS)
    if raw is None:
        return None
    null = permutation_null(shifts_dict)
    if null is None:
        return None
    null_mean = float(null.mean())
    null_sd = float(null.std())
    z = (raw - null_mean) / null_sd
    p = float(np.mean(np.abs(null - null_mean) >= abs(raw - null_mean)))
    return {
        "layer": layer_label,
        "raw_align": round(raw, 4),
        "null_mean": round(null_mean, 4),
        "null_sd": round(null_sd, 4),
        "z": round(float(z), 4),
        "p": round(p, 4),
    }


def main():
    print("=" * 78)
    print("Sonnet-alignment z-score verification with paper-accurate anchor lists")
    print("=" * 78)
    print(f"UP:   {SONNET_UP_ANCHORS}")
    print(f"DOWN: {SONNET_DOWN_ANCHORS}")
    print()

    results = {
        "description": (
            "Per-layer Sonnet-alignment z-score recomputed with paper-accurate "
            "anchor lists from Table 16 (full_paper.md lines 3375-3384 / 3535-3545). "
            "Previous notepad analysis (commits 2c0bd25, 0daeff5) computed these "
            "values without saving the anchor lists used — this script re-verifies "
            "with the correct lists."
        ),
        "sonnet_up_anchors": SONNET_UP_ANCHORS,
        "sonnet_down_anchors": SONNET_DOWN_ANCHORS,
        "paper_source": "ant-emotion-concepts-full_paper.md lines 3375-3384 (UP), 3535-3545 (DOWN)",
        "notepad_reported_values": {
            "L31_z": "+1.61 (peak aligned)",
            "L79_z": "+0.76 (moderate aligned)",
            "L73_z": "-1.23 (peak opposite)",
        },
        "per_layer_results_main_sweep": {},
        "per_layer_results_dense_l31_zone": {},
    }

    # Main 14-layer sweep
    sweep = load_layer_sweep_shifts()
    print("Main 14-layer sweep (L49 own probe basis per layer):")
    print(f"{'Layer':<7} {'raw':>10} {'z':>8} {'p':>8}")
    for L in sorted(sweep.keys()):
        r = verify_one_layer(sweep[L], f"L{L}")
        if r is None:
            continue
        results["per_layer_results_main_sweep"][f"L{L}"] = r
        print(f"L{L:<6} {r['raw_align']:+.4f} {r['z']:+.2f} {r['p']:.4f}")

    # Dense L25/29/31/33/37 zone (uses L31 probe basis — see stage8_l31_zone.json note)
    print()
    print("Dense L25-L37 zone (L31 probe basis approximation):")
    print(f"{'Layer':<7} {'raw':>10} {'z':>8} {'p':>8}")
    dense = load_l31_zone_shifts()
    for L in sorted(dense.keys()):
        r = verify_one_layer(dense[L], f"L{L}")
        if r is None:
            continue
        results["per_layer_results_dense_l31_zone"][f"L{L}"] = r
        print(f"L{L:<6} {r['raw_align']:+.4f} {r['z']:+.2f} {r['p']:.4f}")

    # Compare L31 (peak), L79 (readout), L73 (peak opposite) vs notepad
    print()
    print("Comparison to notepad's cited values:")
    for L, notepad_z in [(31, "+1.61"), (79, "+0.76"), (73, "-1.23")]:
        if f"L{L}" in results["per_layer_results_main_sweep"]:
            computed_z = results["per_layer_results_main_sweep"][f"L{L}"]["z"]
            match = "✓ matches" if abs(computed_z - float(notepad_z)) < 0.15 else "✗ DIVERGES"
            print(f"  L{L}: notepad {notepad_z} / computed {computed_z:+.2f}  {match}")
        else:
            print(f"  L{L}: not in main sweep — checking dense zone")
            if f"L{L}" in results["per_layer_results_dense_l31_zone"]:
                computed_z = results["per_layer_results_dense_l31_zone"][f"L{L}"]["z"]
                match = "✓ matches" if abs(computed_z - float(notepad_z)) < 0.15 else "✗ DIVERGES"
                print(f"    dense {computed_z:+.2f}  {match}")

    out = BASE / "sonnet_alignment_zscore_verification.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
