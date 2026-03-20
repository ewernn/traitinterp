"""
Print mean trait and coherence scores for the best-layer file of each version.
"""
import json
from pathlib import Path

paths = {
    "bs/concealment (L60)": "/Users/ewern/Desktop/code/trait-stuff/trait-interp/experiments/viz_findings/bullshit/steering/bs/concealment/instruct/response__5/steering/responses/residual/probe/L60_c35.9_2026-02-05_01-32-35.json",
    "emotion_set/concealment (L46)": "/Users/ewern/Desktop/code/trait-stuff/trait-interp/experiments/mats-alignment-faking/steering/emotion_set/concealment/instruct/response__5/steering/responses/residual/probe/L46_c26.9_2026-03-05_12-09-56.json",
    "bs/lying (L40)": "/Users/ewern/Desktop/code/trait-stuff/trait-interp/experiments/viz_findings/bullshit/steering/bs/lying/instruct/response__5/steering/responses/residual/probe/L40_c8.3_2026-02-10_11-53-59.json",
    "emotion_set/lying (L38)": "/Users/ewern/Desktop/code/trait-stuff/trait-interp/experiments/mats-alignment-faking/steering/emotion_set/lying/instruct/response__5/steering/responses/residual/probe/L38_c17.4_2026-03-05_12-18-21.json",
}

for label, path in paths.items():
    data = json.loads(Path(path).read_text())
    trait_scores = [d["trait_score"] for d in data]
    coh_scores = [d["coherence_score"] for d in data]
    mean_t = sum(trait_scores) / len(trait_scores)
    mean_c = sum(coh_scores) / len(coh_scores)
    high_trait = sum(1 for s in trait_scores if s >= 60)
    print(f"{label}")
    print(f"  trait scores:     {[f'{s:.1f}' for s in trait_scores]}")
    print(f"  mean trait:       {mean_t:.1f}  |  responses >=60: {high_trait}/{len(trait_scores)}")
    print(f"  mean coherence:   {mean_c:.1f}")
    print()
