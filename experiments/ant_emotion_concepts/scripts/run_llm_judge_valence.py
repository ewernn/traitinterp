"""LLM-judge valence and arousal ratings for 171 emotion vectors.

Replicates Sofroniew et al. 2026 §1.2.5 (Fig 56, 58): validates that
emotion vectors correlate with human valence/arousal norms by having
Claude rate each emotion concept on a 1-7 scale.

Paper-faithful methodology:
  - Model: Claude Sonnet 4.5 (Anthropic; per paper's "We used Claude to rate...")
  - Scale: 1-7 (per paper: "on a scale of 1-7 for its degree of valence and arousal")
  - Dimensions: valence (pleasantness) + arousal (activation/energy)
  - Cross-validation against Russell & Mehrabian (1977) PAD norms on the
    ~45-emotion overlap between our 171-emotion set and the PAD database

Input:  171 emotion names + Russell-Mehrabian norms
Output: `results/llm_judge_validation/llm_judge_valence_arousal_claude_1to7.json`
        — correlations with human norms, per-emotion ratings, model identifier.

Previous runs with GPT-4.1-mini + 0-100 scale are archived at
`llm_judge_valence_arousal_gpt4mini_0to100.json` for provenance.

Usage:
    # Default — Claude Sonnet 4.5, 1-7 scale, paper-faithful
    python experiments/ant_emotion_concepts/scripts/run_llm_judge_valence.py \\
        --experiment ant_emotion_concepts

    # Increase samples per call for lower variance
    python experiments/ant_emotion_concepts/scripts/run_llm_judge_valence.py \\
        --experiment ant_emotion_concepts --n-samples 3
"""

import argparse
import asyncio
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from utils.paths import get as get_path, discover_traits
from utils.judge import TraitJudge, _load_judge_prompt
from shared import get_results_dir, save_results, load_russell_mehrabian_norms


# Paper-faithful defaults
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_SCALE = (1, 7)
OUTPUT_FILENAME = "llm_judge_valence_arousal_claude_1to7"


async def rate_emotions(
    judge: TraitJudge,
    emotions: list,
    prompt_template: str,
    dimension: str,
    scale: tuple[int, int],
    n_samples: int,
) -> dict:
    """Rate all emotions on a single dimension via the judge.

    Emotion names with underscores (e.g., "grief_stricken") are rendered
    back to "grief stricken" for the model — matches the paper's form.
    """
    results = {}
    semaphore = asyncio.Semaphore(20)
    count = [0]
    min_val, max_val = scale

    async def rate_one(emotion: str):
        async with semaphore:
            prompt = prompt_template.format(emotion=emotion.replace("_", " "))
            score = await judge.score_on_scale(
                [{"role": "user", "content": prompt}],
                min_val=min_val,
                max_val=max_val,
                n_samples=n_samples,
            )
            count[0] += 1
            if count[0] % 20 == 0:
                print(f"  Rated {count[0]}/{len(emotions)} {dimension}...")
            return emotion, score

    pairs = await asyncio.gather(*[rate_one(e) for e in emotions])
    return dict(pairs)


def _load_scale_appropriate_norms(experiment: str, scale: tuple[int, int]) -> dict:
    """Load Russell-Mehrabian norms and return as {emotion: (pleasure, arousal)}.

    PAD norms are on a z-score-ish scale (approximately [-1, +1]). We don't
    rescale to match our integer scale — correlations are scale-invariant.
    Returns the raw norms dict unchanged.
    """
    raw = load_russell_mehrabian_norms(experiment)
    return raw


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment", default="ant_emotion_concepts")
    parser.add_argument(
        "--provider", default=DEFAULT_PROVIDER,
        help=f"Judge provider (default: {DEFAULT_PROVIDER}). Paper uses Claude.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model id (default: {DEFAULT_MODEL}). Paper's Sonnet 4.5 match.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=3,
        help=(
            "Samples per rating (Anthropic sampling mode only). Default 3 = "
            "3-sample mean for lower variance (recommended for final Fig 58 "
            "reproduction). Use 1 for a fast deterministic single-shot run "
            "(temperature=0) during development."
        ),
    )
    args = parser.parse_args()

    category = args.experiment
    emotions = discover_traits(category)
    print(f"Rating {len(emotions)} emotions on valence + arousal")
    print(f"Judge: {args.provider}/{args.model}, scale {DEFAULT_SCALE[0]}-{DEFAULT_SCALE[1]}, n_samples={args.n_samples}")

    # Load paper-faithful 1-7 prompts.
    valence_prompt = _load_judge_prompt("valence_arousal", "valence_default")
    arousal_prompt = _load_judge_prompt("valence_arousal", "arousal_default")

    async def run_all():
        judge = TraitJudge(provider=args.provider, model=args.model, n_samples=args.n_samples)

        print("\n=== Valence ratings ===")
        valence = await rate_emotions(
            judge, emotions, valence_prompt, "valence", DEFAULT_SCALE, args.n_samples,
        )

        print("\n=== Arousal ratings ===")
        arousal = await rate_emotions(
            judge, emotions, arousal_prompt, "arousal", DEFAULT_SCALE, args.n_samples,
        )

        identifier = judge.identifier()
        await judge.close()
        return valence, arousal, identifier

    valence_ratings, arousal_ratings, judge_id = asyncio.run(run_all())

    # Russell-Mehrabian norms cover ~45 of our 171 emotions.
    norms = _load_scale_appropriate_norms(args.experiment, DEFAULT_SCALE)
    common = [
        e for e in emotions
        if e in norms
        and valence_ratings.get(e) is not None
        and arousal_ratings.get(e) is not None
    ]
    print(f"\n{len(common)} emotions with both LLM ratings and human PAD norms")

    correlations = {"valence_r": None, "arousal_r": None}
    if common:
        llm_v = [valence_ratings[e] for e in common]
        llm_a = [arousal_ratings[e] for e in common]
        hum_v = [norms[e][0] for e in common]
        hum_a = [norms[e][1] for e in common]
        r_v, p_v = stats.pearsonr(llm_v, hum_v)
        r_a, p_a = stats.pearsonr(llm_a, hum_a)
        correlations = {"valence_r": round(r_v, 4), "arousal_r": round(r_a, 4)}
        print(f"\nLLM valence vs human pleasure: r = {r_v:.3f} (p = {p_v:.2e})")
        print(f"LLM arousal vs human arousal:  r = {r_a:.3f} (p = {p_a:.2e})")
        print(f"Paper (Sonnet 4.5 + 1-7): r = 0.92 valence, 0.90 arousal")

    out_dir = get_results_dir(args.experiment, "llm_judge_validation")
    results = {
        "valence_ratings": valence_ratings,
        "arousal_ratings": arousal_ratings,
        "n_emotions": len(emotions),
        "n_with_norms": len(common),
        "correlations": correlations,
        "judge_identifier": judge_id,
        "scale": {"min": DEFAULT_SCALE[0], "max": DEFAULT_SCALE[1]},
        "n_samples": args.n_samples,
    }
    save_results(out_dir, OUTPUT_FILENAME, results)
    print(f"\nSaved to {out_dir}/{OUTPUT_FILENAME}.json")


if __name__ == "__main__":
    main()
