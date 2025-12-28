"""
Results I/O for steering evaluation.

Input:
    - experiment: Experiment name
    - trait: Trait path (category/trait)

Output:
    - Loads/saves results.json and response files

Usage:
    from analysis.steering.results import load_or_create_results, save_results

    results = load_or_create_results(experiment, trait, ...)
    save_results(results, experiment, trait)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from utils.paths import get, get_steering_results_path, get_steering_dir
from utils.vectors import load_vector_metadata


def load_or_create_results(
    experiment: str,
    trait: str,
    prompts_file: Path,
    steering_model: str,
    vector_experiment: str,
    judge_provider: str,
) -> Dict:
    """Load existing results or create new structure."""
    results_path = get_steering_results_path(experiment, trait)
    prompts_file_str = str(prompts_file)

    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

        if results.get("prompts_file") != prompts_file_str:
            stored = results.get("prompts_file", "unknown")
            raise ValueError(
                f"Prompts file mismatch!\n"
                f"  Stored: {stored}\n"
                f"  Current: {prompts_file_str}\n"
                f"Delete {results_path} manually to start fresh with new prompts."
            )

        # Require new format
        if "steering_model" not in results or "eval" not in results:
            raise ValueError(
                f"Old results format detected. Delete {results_path} and re-run steering."
            )

        return results

    # Load vector metadata for source info
    try:
        vector_metadata = load_vector_metadata(vector_experiment, trait, "probe")
    except FileNotFoundError:
        vector_metadata = {}

    return {
        "trait": trait,
        "steering_model": steering_model,
        "steering_experiment": experiment,
        "vector_source": {
            "model": vector_metadata.get("model", "unknown"),
            "experiment": vector_experiment,
            "trait": trait,
        },
        "eval": {
            "model": "gpt-4.1-mini" if judge_provider == "openai" else "gemini-2.5-flash",
            "method": "logprob" if judge_provider == "openai" else "text_parse",
        },
        "prompts_file": prompts_file_str,
        "baseline": None,
        "runs": []
    }


def find_existing_run_index(results: Dict, config: Dict) -> Optional[int]:
    """Find index of existing run with identical config, or None if not found."""
    for i, run in enumerate(results["runs"]):
        if run["config"] == config:
            return i
    return None


def save_results(results: Dict, experiment: str, trait: str):
    """Save results to experiment directory."""
    results_file = get_steering_results_path(experiment, trait)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_file}")


def save_responses(responses: List[Dict], experiment: str, trait: str, config: Dict, timestamp: str):
    """Save generated responses for a config."""
    responses_dir = get_steering_dir(experiment, trait) / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    layers_str = "_".join(str(l) for l in config["layers"])
    coefs_str = "_".join(f"{c:.1f}" for c in config["coefficients"])
    ts_clean = timestamp[:19].replace(':', '-').replace('T', '_')  # Trim microseconds
    filename = f"L{layers_str}_c{coefs_str}_{ts_clean}.json"

    with open(responses_dir / filename, 'w') as f:
        json.dump(responses, f, indent=2)
