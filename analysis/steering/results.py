"""
Results I/O for steering evaluation (JSONL format).

Input: experiment, trait, model_variant identifiers
Output: results.jsonl (one entry per line), response files

Format:
    {"type": "header", "trait": ..., "steering_model": ..., ...}
    {"type": "baseline", "result": {...}, "timestamp": ...}
    {"result": {...}, "config": {...}, "timestamp": ...}

Usage:
    from analysis.steering.results import init_results_file, append_run, load_results
    from utils.paths import get_steering_results_path

    # Check if exists
    if not get_steering_results_path(...).exists():
        init_results_file(...)

    # Append results
    append_baseline(experiment, trait, model_variant, result, ...)
    append_run(experiment, trait, model_variant, config, result, ...)

    # Load for resume/analysis
    results = load_results(...)  # Returns {baseline, runs, ...}
    baseline = get_baseline(...)  # Returns just baseline result
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.paths import get_steering_results_path, get_steering_dir, get_steering_response_dir
from utils.vectors import load_vector_metadata


def init_results_file(
    experiment: str,
    trait: str,
    model_variant: str,
    prompts_file: Path,
    steering_model: str,
    vector_experiment: str,
    judge_provider: str,
    position: str = "response[:]",
    prompt_set: str = "steering",
    trait_judge: Optional[str] = None,
) -> Path:
    """
    Initialize a new results.jsonl file with header line.

    Returns path to results file. Raises if file already exists.
    """
    results_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)

    if results_path.exists():
        raise FileExistsError(
            f"Results file already exists: {results_path}\n"
            f"Use load_results() to read, or delete manually to start fresh."
        )

    # Load vector metadata for source info
    try:
        vector_metadata = load_vector_metadata(vector_experiment, trait, "probe", model_variant)
    except FileNotFoundError:
        vector_metadata = {}

    header = {
        "type": "header",
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
            "trait_judge": trait_judge,  # None = V3c default, else path like "pv/hallucination"
        },
        "prompts_file": str(prompts_file),
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        f.write(json.dumps(header) + '\n')

    return results_path


def append_baseline(
    experiment: str,
    trait: str,
    model_variant: str,
    result: Dict,
    position: str = "response[:]",
    prompt_set: str = "steering",
) -> None:
    """Append baseline result to results.jsonl."""
    results_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)

    entry = {
        "type": "baseline",
        "result": result,
        "timestamp": datetime.now().isoformat(),
    }

    with open(results_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def append_run(
    experiment: str,
    trait: str,
    model_variant: str,
    config: Dict,
    result: Dict,
    position: str = "response[:]",
    prompt_set: str = "steering",
) -> None:
    """Append a steering run to results.jsonl."""
    results_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)

    # Order: result first, then config, then timestamp
    entry = {
        "result": result,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    with open(results_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def load_results(
    experiment: str,
    trait: str,
    model_variant: str,
    position: str = "response[:]",
    prompt_set: str = "steering",
) -> Dict:
    """
    Load results.jsonl into dict format.

    Returns dict with keys: trait, steering_model, vector_source, eval, prompts_file, baseline, runs
    """
    results_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)

    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    header = None
    baseline = None
    runs = []

    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            if entry.get("type") == "header":
                header = entry
            elif entry.get("type") == "baseline":
                baseline = entry.get("result")
            else:
                runs.append(entry)

    if header is None:
        raise ValueError(f"No header found in {results_path}")

    return {
        "trait": header.get("trait"),
        "steering_model": header.get("steering_model"),
        "steering_experiment": header.get("steering_experiment"),
        "vector_source": header.get("vector_source"),
        "eval": header.get("eval"),
        "prompts_file": header.get("prompts_file"),
        "baseline": baseline,
        "runs": runs,
    }


def find_cached_run(runs: list, config: dict) -> dict | None:
    """Find cached result for a config in loaded runs list."""
    for run in runs:
        if run.get("config") == config:
            return run.get("result")
    return None


def is_better_result(
    current_best: Optional[Dict],
    trait_mean: float,
    coherence_mean: float,
    threshold: float,
) -> bool:
    """
    Check if new result is better than current best.

    Priority: valid results (coherence >= threshold) by trait_mean,
    then invalid results by coherence_mean as fallback.
    """
    is_valid = coherence_mean >= threshold

    if current_best is None:
        return True

    current_valid = current_best.get("valid", False)

    # Valid beats invalid
    if is_valid and not current_valid:
        return True
    # Invalid doesn't beat valid
    if not is_valid and current_valid:
        return False
    # Both valid: compare trait
    if is_valid and current_valid:
        return trait_mean > current_best.get("trait_mean", 0)
    # Both invalid: compare coherence
    return coherence_mean > current_best.get("coherence_mean", 0)


def get_baseline(
    experiment: str,
    trait: str,
    model_variant: str,
    position: str = "response[:]",
    prompt_set: str = "steering",
) -> Optional[Dict]:
    """Get baseline result if it exists."""
    results_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)

    if not results_path.exists():
        return None

    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") == "baseline":
                return entry.get("result")

    return None


# =============================================================================
# Response file I/O (unchanged format)
# =============================================================================

def save_responses(
    responses: List[Dict],
    experiment: str,
    trait: str,
    model_variant: str,
    position: str,
    prompt_set: str,
    config: Dict,
    timestamp: str,
) -> Path:
    """Save generated responses for a config."""
    vectors = config.get("vectors", [{}])
    component = config.get("component") or vectors[0].get("component", "residual")
    method = config.get("method") or vectors[0].get("method", "probe")
    responses_dir = get_steering_response_dir(experiment, trait, model_variant, component, method, position, prompt_set)
    responses_dir.mkdir(parents=True, exist_ok=True)

    layers_str = "_".join(str(v["layer"]) for v in vectors)
    coefs_str = "_".join(f"{v['weight']:.1f}" for v in vectors)
    ts_clean = timestamp[:19].replace(':', '-').replace('T', '_')
    filename = f"L{layers_str}_c{coefs_str}_{ts_clean}.json"

    path = responses_dir / filename
    with open(path, 'w') as f:
        json.dump(responses, f, indent=2)

    return path


def save_baseline_responses(
    responses: List[Dict],
    experiment: str,
    trait: str,
    model_variant: str,
    position: str,
    prompt_set: str = "steering",
) -> Path:
    """Save baseline (no steering) responses."""
    responses_dir = get_steering_dir(experiment, trait, model_variant, position, prompt_set) / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    path = responses_dir / "baseline.json"
    with open(path, 'w') as f:
        json.dump(responses, f, indent=2)

    return path


def save_ablation_responses(
    responses: List[Dict],
    experiment: str,
    trait: str,
    model_variant: str,
    position: str,
    prompt_set: str,
    vector_layer: int,
    method: str,
    component: str = "residual",
) -> Path:
    """Save ablation (all-layer) responses."""
    responses_dir = get_steering_dir(experiment, trait, model_variant, position, prompt_set) / "responses" / "ablation"
    responses_dir.mkdir(parents=True, exist_ok=True)

    filename = f"L{vector_layer}_{component}_{method}.json"
    path = responses_dir / filename
    with open(path, 'w') as f:
        json.dump(responses, f, indent=2)

    return path
