#!/usr/bin/env python3
"""
Score naturalness of steering responses for top configs.

Scores the top N steering configs (by delta, coherence >= threshold) using
the naturalness judge. Saves results alongside results.jsonl so select_vector
can filter by naturalness.

Input: Steering results (results.jsonl) + response files
Output: naturalness.json in the same directory

Usage:
    python dev/steering/score_naturalness.py \
        --experiment dataset_creation_emotions \
        --traits emotions/power_seeking emotions/calm
"""

import argparse
import asyncio
import json
from pathlib import Path

from utils.paths import get_steering_results_path, get_model_variant
from utils.judge import TraitJudge
from utils.traits import load_trait_definition
from utils.vectors import MIN_COHERENCE


def parse_results(results_path: Path) -> dict:
    """Parse results.jsonl into baseline + runs."""
    baseline_trait = 0
    direction = "positive"
    runs = []

    with open(results_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("type") == "header":
                direction = data.get("direction", "positive")
            elif data.get("type") == "baseline":
                baseline_trait = data.get("result", {}).get("trait_mean", 0)
            elif "result" in data and "config" in data:
                vectors = data["config"].get("vectors", [])
                if not vectors:
                    continue
                v = vectors[0]
                result = data["result"]
                runs.append({
                    "layer": v.get("layer"),
                    "method": v.get("method", "probe"),
                    "component": v.get("component", "residual"),
                    "coef": v.get("weight"),
                    "trait_mean": result.get("trait_mean", 0),
                    "coherence_mean": result.get("coherence_mean", 0),
                    "delta": result.get("trait_mean", 0) - baseline_trait,
                    "timestamp": data.get("timestamp", ""),
                })

    return {"baseline_trait": baseline_trait, "direction": direction, "runs": runs}


def find_response_file(responses_dir: Path, run: dict) -> Path | None:
    """Find response file for a run config."""
    component = run["component"]
    method = run["method"]
    layer = run["layer"]
    coef = run["coef"]

    method_dir = responses_dir / component / method
    if not method_dir.exists():
        return None

    # Match by L{layer}_c{coef} prefix (coef may be formatted differently)
    for pattern in [f"L{layer}_c{coef:.1f}*.json", f"L{layer}_c{coef:.2f}*.json",
                    f"L{layer}_c{int(coef)}*.json" if coef == int(coef) else None]:
        if pattern:
            matches = list(method_dir.glob(pattern))
            if matches:
                return matches[0]
    return None


def run_key(run: dict) -> str:
    """Unique key for a steering config."""
    return f"L{run['layer']}_{run['method']}_{run['component']}_c{run['coef']:.1f}"


async def score_trait(
    experiment: str,
    trait: str,
    steering_variant: str,
    position: str,
    prompt_set: str,
    top_n: int,
    min_coherence: float,
) -> dict | None:
    """Score naturalness for top configs of one trait."""
    results_path = get_steering_results_path(experiment, trait, steering_variant, position, prompt_set)
    if not results_path.exists():
        print(f"  {trait}: no results.jsonl")
        return None

    parsed = parse_results(results_path)
    direction = parsed["direction"]
    sign = 1 if direction == "positive" else -1

    # Filter by coherence, sort by delta (direction-aware)
    coherent = [r for r in parsed["runs"] if r["coherence_mean"] >= min_coherence]
    coherent.sort(key=lambda r: r["delta"] * sign, reverse=True)
    top_runs = coherent[:top_n]

    if not top_runs:
        print(f"  {trait}: no runs pass coherence >= {min_coherence}")
        return None

    # Load trait definition for naturalness scoring
    trait_name = trait.split("/")[-1]
    trait_definition = load_trait_definition(trait)

    judge = TraitJudge()
    responses_dir = results_path.parent / "responses"

    scores = {}
    for run in top_runs:
        resp_file = find_response_file(responses_dir, run)
        if not resp_file:
            continue

        responses = json.load(open(resp_file))
        texts = [r["response"] for r in responses]

        nat_scores = await judge.score_naturalness_batch(texts, trait_name, trait_definition)
        valid = [s for s in nat_scores if s is not None]
        if valid:
            key = run_key(run)
            scores[key] = {
                "mean": sum(valid) / len(valid),
                "per_response": nat_scores,
                "layer": run["layer"],
                "method": run["method"],
                "component": run["component"],
                "coef": run["coef"],
                "delta": run["delta"],
            }

    await judge.close()

    if not scores:
        print(f"  {trait}: no response files found for top configs")
        return None

    # Save alongside results.jsonl
    output = {"trait": trait, "scores": scores}
    output_path = results_path.parent / "naturalness.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    for key, data in sorted(scores.items(), key=lambda x: -x[1]["mean"]):
        print(f"  {trait} {key}: naturalness={data['mean']:.1f}, delta={data['delta']:+.1f}")

    return output


async def main():
    parser = argparse.ArgumentParser(description="Score naturalness of steering responses")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--traits", nargs="+", required=True, help="Trait paths (e.g., emotions/spite)")
    parser.add_argument("--top-n", type=int, default=5, help="Score top N configs per trait (default: 5)")
    parser.add_argument("--min-coherence", type=float, default=MIN_COHERENCE, help="Minimum coherence threshold")
    parser.add_argument("--model-variant", default=None, help="Steering model variant")
    parser.add_argument("--position", default="response[:5]")
    parser.add_argument("--prompt-set", default="steering")
    args = parser.parse_args()

    steering_variant = args.model_variant
    if steering_variant is None:
        steering_variant = get_model_variant(args.experiment, None, mode="application")["name"]

    for trait in args.traits:
        await score_trait(
            args.experiment, trait, steering_variant,
            args.position, args.prompt_set, args.top_n, args.min_coherence,
        )


if __name__ == "__main__":
    asyncio.run(main())
