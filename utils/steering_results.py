"""
Results I/O for steering evaluation (JSONL format).

Input: experiment, trait, model_variant identifiers
Output: results.jsonl (one entry per line), response files

Format:
    {"type": "header", "trait": ..., "steering_model": ..., ...}
    {"type": "baseline", "result": {...}, "timestamp": ...}
    {"result": {...}, "config": {...}, "timestamp": ...}

Usage:
    from utils.steering_results import init_results_file, append_run, load_results
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
from typing import Dict, List, Literal, Optional

from core.types import JudgeResult, SteeringRunRecord, SteeringResults
from utils.paths import get_steering_results_path, get_steering_dir, get_steering_response_dir, content_hash
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
    direction: Literal["positive", "negative"] = "positive",
    n_questions: Optional[int] = None,
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
        "direction": direction,  # "positive" for inducing, "negative" for suppressing
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
        "prompts_hash": content_hash(prompts_file),
        "n_questions": n_questions,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        f.write(json.dumps(header) + '\n')

    return results_path


def append_baseline(
    experiment: str,
    trait: str,
    model_variant: str,
    result,
    position: str = "response[:]",
    prompt_set: str = "steering",
    trait_judge: Optional[str] = None,
) -> None:
    """Append baseline result to results.jsonl.

    Args:
        result: JudgeResult or dict with trait_mean, coherence_mean, n, etc.
    """
    results_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)

    result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
    entry = {
        "type": "baseline",
        "result": result_dict,
        "eval": {"trait_judge": trait_judge},
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
    trait_judge: Optional[str] = None,
    input_hashes: Optional[Dict[str, str]] = None,
) -> None:
    """Append a steering run to results.jsonl.

    Args:
        input_hashes: Optional provenance hashes for staleness detection.
            Caller computes these since it knows what inputs were actually used.
    """
    results_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)

    entry = {
        "type": "run",
        "result": result,
        "config": config,
        "eval": {"trait_judge": trait_judge},
        "timestamp": datetime.now().isoformat(),
    }
    if input_hashes:
        entry["input_hashes"] = input_hashes

    with open(results_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def load_results(
    experiment: str,
    trait: str,
    model_variant: str,
    position: str = "response[:]",
    prompt_set: str = "steering",
) -> SteeringResults:
    """Load results.jsonl into typed SteeringResults.

    Parses header, optional baseline, and all run entries into dataclasses.
    Run entries gain a "type": "run" field for consistency (old files lack it).
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
                baseline_raw = entry.get("result")
                baseline = JudgeResult.from_dict(baseline_raw) if baseline_raw else None
            else:
                runs.append(SteeringRunRecord.from_dict(entry))

    if header is None:
        raise ValueError(f"No header found in {results_path}")

    return SteeringResults(
        trait=header.get("trait", ""),
        direction=header.get("direction", "positive"),
        steering_model=header.get("steering_model", ""),
        steering_experiment=header.get("steering_experiment", ""),
        vector_source=header.get("vector_source", {}),
        eval=header.get("eval", {}),
        prompts_file=header.get("prompts_file", ""),
        prompts_hash=header.get("prompts_hash", ""),
        baseline=baseline,
        runs=runs,
    )


def find_cached_run(runs: List[SteeringRunRecord], config: dict) -> Optional[JudgeResult]:
    """Find cached result for a config in loaded runs list."""
    for run in runs:
        if run.config.to_dict() == config:
            return run.result
    return None


def is_better_result(
    current_best: Optional[Dict],
    trait_mean: float,
    coherence_mean: float,
    threshold: float,
    direction: Literal["positive", "negative"] = "positive",
) -> bool:
    """Check if new result is better than current best.

    Priority: valid results (coherence >= threshold) by trait_mean,
    then invalid results by coherence_mean as fallback.

    Args:
        current_best: Dict with trait_mean, coherence_mean, valid keys (in-memory, not persisted).
        direction: "positive" means higher trait is better, "negative" means lower trait is better.
    """
    sign = 1 if direction == "positive" else -1
    is_valid = coherence_mean >= threshold

    if current_best is None:
        return True

    current_valid = current_best.get("valid", False)

    if is_valid and not current_valid:
        return True
    if not is_valid and current_valid:
        return False
    if is_valid and current_valid:
        return trait_mean * sign > current_best.get("trait_mean", 0) * sign
    return coherence_mean > current_best.get("coherence_mean", 0)


def remove_baseline(
    experiment: str,
    trait: str,
    model_variant: str,
    position: str = "response[:]",
    prompt_set: str = "steering",
) -> bool:
    """Remove baseline entry from results.jsonl, keeping header and runs.

    Also deletes responses/baseline.json if it exists.
    Returns True if a baseline was found and removed.
    """
    results_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)
    if not results_path.exists():
        return False

    lines = []
    found = False
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") == "baseline":
                found = True
                continue  # Skip baseline
            lines.append(entry)

    if found:
        with open(results_path, 'w') as f:
            for entry in lines:
                f.write(json.dumps(entry) + '\n')

        # Also delete saved baseline responses
        baseline_responses = get_steering_dir(experiment, trait, model_variant, position, prompt_set) / "responses" / "baseline.json"
        if baseline_responses.exists():
            baseline_responses.unlink()

    return found


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
# Response records
# =============================================================================

def build_response_records(questions, responses, scores) -> List[Dict]:
    """Build response record dicts from parallel question/response/score lists."""
    return [
        {"prompt": q, "response": r, "system_prompt": None,
         "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
        for q, r, s in zip(questions, responses, scores)
    ]


# Response file I/O (unchanged format)
# =============================================================================

def _write_responses(responses: List[Dict], path: Path) -> Path:
    """Write responses JSON to a resolved path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(responses, f, indent=2)
    return path


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

    layers_str = "_".join(str(v["layer"]) for v in vectors)
    coefs_str = "_".join(f"{v['weight']:.1f}" for v in vectors)
    ts_clean = timestamp[:19].replace(':', '-').replace('T', '_')
    return _write_responses(responses, responses_dir / f"L{layers_str}_c{coefs_str}_{ts_clean}.json")


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
    return _write_responses(responses, responses_dir / "baseline.json")


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
    return _write_responses(responses, responses_dir / f"L{vector_layer}_{component}_{method}.json")
