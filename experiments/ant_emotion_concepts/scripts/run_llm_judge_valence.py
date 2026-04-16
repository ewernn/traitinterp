"""LLM-judge valence and arousal ratings for 171 emotion vectors.

Replicates Sofroniew et al. 2026 §1.2.5 (Fig 56, 58): validates that
emotion vectors correlate with human valence/arousal norms by having an
LLM rate each emotion name.

Uses the existing TraitJudge infrastructure (gpt-4.1-mini with logprobs).
Compares LLM ratings against Russell-Mehrabian norms.

Input:  171 emotion names + Russell-Mehrabian norms
Output: LLM valence/arousal ratings + correlation with human norms

Usage:
    python experiments/ant_emotion_concepts/scripts/run_llm_judge_valence.py \
        --experiment ant_emotion_concepts
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from utils.paths import get as get_path, discover_traits
from shared import get_results_dir, save_results, load_russell_mehrabian_norms


VALENCE_PROMPT = """Rate the emotional valence (pleasantness) of the emotion "{emotion}" on a scale from 0 to 100.

0 = extremely unpleasant/negative (e.g., terrified, disgusted, miserable)
50 = neutral (e.g., alert, attentive, indifferent)
100 = extremely pleasant/positive (e.g., ecstatic, blissful, elated)

Respond with just a number 0-100:"""

AROUSAL_PROMPT = """Rate the emotional arousal (activation/energy level) of the emotion "{emotion}" on a scale from 0 to 100.

0 = extremely calm/deactivated (e.g., sleepy, relaxed, serene)
50 = moderate arousal (e.g., content, alert, attentive)
100 = extremely activated/high-energy (e.g., terrified, ecstatic, enraged)

Respond with just a number 0-100:"""


async def rate_emotions(judge, emotions: list, prompt_template: str, dimension: str) -> dict:
    """Rate all emotions on a dimension using LLM judge."""
    from utils.judge import aggregate_logprob_score

    results = {}
    semaphore = asyncio.Semaphore(20)
    count = [0]

    async def rate_one(emotion: str):
        async with semaphore:
            prompt = prompt_template.format(emotion=emotion.replace("_", " "))
            logprobs = await judge._get_logprobs(prompt)
            score = aggregate_logprob_score(logprobs)
            count[0] += 1
            if count[0] % 20 == 0:
                print(f"  Rated {count[0]}/{len(emotions)} {dimension}...")
            return emotion, score

    pairs = await asyncio.gather(*[rate_one(e) for e in emotions])
    for emotion, score in pairs:
        results[emotion] = score

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="ant_emotion_concepts")
    args = parser.parse_args()

    category = args.experiment
    emotions = discover_traits(category)
    print(f"Rating {len(emotions)} emotions on valence and arousal...")

    async def run_all():
        from utils.judge import TraitJudge
        judge = TraitJudge()

        print("\n=== Valence ratings ===")
        valence = await rate_emotions(judge, emotions, VALENCE_PROMPT, "valence")

        print("\n=== Arousal ratings ===")
        arousal = await rate_emotions(judge, emotions, AROUSAL_PROMPT, "arousal")

        await judge.close()
        return valence, arousal

    valence_ratings, arousal_ratings = asyncio.run(run_all())

    # Load Russell-Mehrabian human norms
    norms = load_russell_mehrabian_norms(args.experiment)

    # Compute correlations (only for emotions that have human norms)
    # Russell-Mehrabian norms only cover ~48 of 171 emotions
    common_emotions = [e for e in emotions if e in norms and valence_ratings.get(e) is not None and arousal_ratings.get(e) is not None]
    print(f"\n{len(common_emotions)} emotions with both LLM ratings and human norms")

    if common_emotions:
        llm_valence = [valence_ratings[e] for e in common_emotions]
        llm_arousal = [arousal_ratings[e] for e in common_emotions]
        human_valence = [norms[e][0] for e in common_emotions]  # (pleasure, arousal) tuples
        human_arousal = [norms[e][1] for e in common_emotions]

        r_valence, p_valence = stats.pearsonr(llm_valence, human_valence)
        r_arousal, p_arousal = stats.pearsonr(llm_arousal, human_arousal)

        print(f"\nLLM valence vs human pleasure:  r = {r_valence:.3f} (p = {p_valence:.2e})")
        print(f"LLM arousal vs human arousal:   r = {r_arousal:.3f} (p = {p_arousal:.2e})")
        print(f"(Paper reports: r = 0.92 for valence)")

    # Save results
    out_dir = get_results_dir(args.experiment, "llm_judge_validation")
    results = {
        "valence_ratings": valence_ratings,
        "arousal_ratings": arousal_ratings,
        "n_emotions": len(emotions),
        "n_with_norms": len(common_emotions),
        "correlations": {
            "valence_r": round(r_valence, 4) if common_emotions else None,
            "arousal_r": round(r_arousal, 4) if common_emotions else None,
        },
        "model": "gpt-4.1-mini",
    }
    save_results(out_dir, "llm_judge_valence_arousal", results)
    print(f"\nDone. Results saved.")


if __name__ == "__main__":
    main()
