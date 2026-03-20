"""
Full sample dump for duplicate trait comparison - shows all responses for manual review.

Input: Hardcoded best-layer files
Output: All responses printed for qualitative assessment
Usage: python dev/compare_duplicate_traits_full.py
"""

import json
from pathlib import Path

BASE = Path("/Users/ewern/Desktop/code/trait-stuff/trait-interp/experiments")

FILES = {
    "alignment/deception (L25)": BASE / "audit-bleachers/steering/alignment/deception/instruct/response__5/steering/responses/residual/probe/L25_c13.6_2026-02-14_20-28-55.json",
    "emotion_set/deception (L23)": BASE / "mats-alignment-faking/steering/emotion_set/deception/instruct/response__5/steering/responses/residual/probe/L23_c10.3_2026-03-05_12-11-37.json",
    "alignment/conflicted (L30)": BASE / "audit-bleachers/steering/alignment/conflicted/instruct/response__5/steering/responses/residual/probe/L30_c15.3_2026-02-14_20-58-31.json",
    "emotion_set/conflicted (L16)": BASE / "aria_rl/steering/emotion_set/conflicted/qwen3_4b_instruct/response__5/steering/responses/residual/probe/L16_c34.1_2026-03-05_07-19-21.json",
}

for label, path in FILES.items():
    data = json.load(open(path))
    trait_scores = [r.get("trait_score") for r in data if r.get("trait_score") is not None]
    coherence_scores = [r.get("coherence_score") for r in data if r.get("coherence_score") is not None]
    avg_t = sum(trait_scores) / len(trait_scores) if trait_scores else 0
    avg_c = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0

    print(f"\n{'=' * 70}")
    print(f"{label}  |  n={len(data)}  avg_trait={avg_t:.1f}  avg_coherence={avg_c:.1f}")
    print("=" * 70)
    for i, r in enumerate(data):
        prompt = r.get("prompt", "")[:200].replace("\n", " ")
        response = r.get("response", "")[:400].replace("\n", " ")
        ts = r.get("trait_score")
        cs = r.get("coherence_score")
        print(f"\n  [{i+1}] trait={ts:.1f}  coherence={cs:.1f}")
        print(f"  PROMPT: {prompt}")
        print(f"  RESPONSE: {response}")
