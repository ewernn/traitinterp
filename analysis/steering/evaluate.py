#!/usr/bin/env python3
"""
Steering evaluation - validate trait vectors via causal intervention.

Input:
    - experiment: Experiment name
    - trait: Trait path (e.g., cognitive_state/confidence)
    - layers: Layer(s) to steer
    - coefficients: Steering strength(s)

Output:
    - experiments/{experiment}/steering/{trait}/results.json
      Runs-based structure that accumulates across invocations.
    - experiments/{experiment}/steering/{trait}/responses/
      Generated responses for each config.

Usage:
    # Single config (1 run)
    python analysis/steering/evaluate.py \\
        --experiment my_exp \\
        --trait mental_state/optimism \\
        --layers 16 \\
        --coefficients 2.0

    # Coefficient sweep at one layer (4 runs)
    python analysis/steering/evaluate.py \\
        --experiment my_exp \\
        --trait mental_state/optimism \\
        --layers 16 \\
        --coefficients 0,1,2,3

    # Layer sweep with fixed coef (4 runs)
    python analysis/steering/evaluate.py \\
        --experiment my_exp \\
        --trait mental_state/optimism \\
        --layers 10,12,14,16 \\
        --coefficients 2.0

    # Multi-layer steering (1 run, all layers steered simultaneously)
    python analysis/steering/evaluate.py \\
        --experiment my_exp \\
        --trait mental_state/optimism \\
        --layers 12,14,16 \\
        --coefficients 1.0,2.0,1.0 \\
        --multi-layer

    # Multiple rollouts with temperature (for variance estimation)
    python analysis/steering/evaluate.py \\
        --experiment my_exp \\
        --trait mental_state/optimism \\
        --layers 16 \\
        --coefficients 2.0 \\
        --rollouts 5 \\
        --temperature 0.7

    # Quick test with subset of questions
    python analysis/steering/evaluate.py \\
        --experiment my_exp \\
        --trait mental_state/optimism \\
        --layers 16 \\
        --coefficients 2.0 \\
        --subset 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import argparse
import asyncio
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.steering.steer import SteeringHook, MultiLayerSteeringHook
from analysis.steering.judge import TraitJudge
from utils.paths import get
from utils.model import format_prompt, load_experiment_config


DEFAULT_MODEL = "google/gemma-2-2b-it"


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def get_num_layers(model) -> int:
    """Get number of layers from model config."""
    return model.config.num_hidden_layers


def parse_layers(layers_arg: str, num_layers: int) -> List[int]:
    """
    Parse layers argument.

    Args:
        layers_arg: "all", single number "16", range "5-20", or list "5,10,15"
        num_layers: Total layers in model

    Returns:
        List of layer indices
    """
    if layers_arg.lower() == "all":
        return list(range(num_layers))
    elif "-" in layers_arg and "," not in layers_arg:
        start, end = layers_arg.split("-")
        return list(range(int(start), int(end) + 1))
    elif "," in layers_arg:
        return [int(x) for x in layers_arg.split(",")]
    else:
        return [int(layers_arg)]


def parse_coefficients(coef_arg: str) -> List[float]:
    """Parse comma-separated coefficients."""
    return [float(c) for c in coef_arg.split(",")]


def load_vector(experiment: str, trait: str, layer: int, method: str = "probe", component: str = "residual") -> Optional[torch.Tensor]:
    """Load trait vector from experiment. Returns None if not found."""
    vectors_dir = get('extraction.vectors', experiment=experiment, trait=trait)
    prefix = "" if component == "residual" else f"{component}_"
    vector_file = vectors_dir / f"{prefix}{method}_layer{layer}.pt"

    if not vector_file.exists():
        return None

    return torch.load(vector_file, weights_only=True)


def load_activation_metadata(experiment: str, trait: str) -> Optional[Dict]:
    """Load activation metadata containing per-layer norms."""
    activations_dir = get('extraction.activations', experiment=experiment, trait=trait)
    metadata_file = activations_dir / "metadata.json"
    if not metadata_file.exists():
        return None
    with open(metadata_file) as f:
        return json.load(f)


def compute_auto_coef(
    vector_experiment: str,
    vector_trait: str,
    layer: int,
    method: str,
    component: str,
    target_ratio: float,
) -> Optional[float]:
    """
    Compute coefficient to achieve target perturbation ratio.

    perturbation_ratio = (coef * vector_norm) / activation_norm
    coef = target_ratio * (activation_norm / vector_norm)
    """
    # Load vector norm
    vector = load_vector(vector_experiment, vector_trait, layer, method, component)
    if vector is None:
        return None
    vector_norm = vector.norm().item()

    # Load activation norm from metadata
    metadata = load_activation_metadata(vector_experiment, vector_trait)
    if metadata is None or 'activation_norms' not in metadata:
        print(f"Warning: No activation metadata found for {vector_experiment}/{vector_trait}")
        return None

    # Get activation norm for this layer
    act_norms = metadata['activation_norms']
    if isinstance(act_norms, dict):
        act_norm = act_norms.get(str(layer)) or act_norms.get(layer)
    elif isinstance(act_norms, list) and layer < len(act_norms):
        act_norm = act_norms[layer]
    else:
        print(f"Warning: No activation norm for layer {layer}")
        return None

    coef = target_ratio * (act_norm / vector_norm)
    return coef


def load_eval_prompts(trait: str) -> Tuple[Dict, Path]:
    """Load evaluation prompts for a trait. Returns (data, path)."""
    trait_name = trait.split("/")[-1]
    prompts_file = get('steering.prompt_file', trait=trait_name)

    if not prompts_file.exists():
        raise FileNotFoundError(
            f"Eval prompts not found: {prompts_file}\n"
            f"Create JSON with 'questions' and 'eval_prompt' (with {{question}} and {{answer}} placeholders)."
        )

    with open(prompts_file) as f:
        data = json.load(f)

    if "eval_prompt" not in data:
        raise ValueError(f"Missing 'eval_prompt' in {prompts_file}")
    if "questions" not in data:
        raise ValueError(f"Missing 'questions' in {prompts_file}")

    return data, prompts_file


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


# =============================================================================
# Results Management
# =============================================================================

def load_or_create_results(experiment: str, trait: str, prompts_file: Path) -> Dict:
    """
    Load existing results or create new structure.

    Checks prompts_file match - raises error if mismatch.
    """
    results_path = get('steering.results', experiment=experiment, trait=trait)
    prompts_file_str = str(prompts_file)

    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

        # Check prompts_file match
        if results.get("prompts_file") != prompts_file_str:
            stored = results.get("prompts_file", "unknown")
            raise ValueError(
                f"Prompts file mismatch!\n"
                f"  Stored: {stored}\n"
                f"  Current: {prompts_file_str}\n"
                f"Delete {results_path} manually to start fresh with new prompts."
            )

        return results

    # Create new structure
    return {
        "trait": trait,
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
    results_file = get('steering.results', experiment=experiment, trait=trait)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_file}")


def save_responses(responses: List[Dict], experiment: str, trait: str, config: Dict, timestamp: str):
    """Save generated responses for a config."""
    responses_dir = get('steering.responses', experiment=experiment, trait=trait)
    responses_dir.mkdir(parents=True, exist_ok=True)

    # Create filename from config
    layers_str = "_".join(str(l) for l in config["layers"])
    methods_str = "_".join(config["methods"])
    coefs_str = "_".join(str(c).replace(".", "p") for c in config["coefficients"])
    filename = f"L{layers_str}_{methods_str}_c{coefs_str}_{timestamp.replace(':', '-').replace('T', '_')}.json"

    with open(responses_dir / filename, 'w') as f:
        json.dump(responses, f, indent=2)
    print(f"  Responses saved: {responses_dir / filename}")


# =============================================================================
# Config Generation
# =============================================================================

def generate_configs(
    layers: List[int],
    method: str,
    coefficients: List[float],
    component: str,
    multi_layer: bool,
) -> List[Dict]:
    """
    Generate list of configs to evaluate based on CLI args.

    Without --multi-layer: creates separate runs for each (layer, coefficient) combo
    With --multi-layer: creates single run with all layers steered simultaneously
    """
    configs = []

    if multi_layer:
        # Single run: all layers steered simultaneously
        # Expand method and coefficients to match layers length
        n = len(layers)
        methods = [method] * n

        if len(coefficients) == 1:
            coefficients = coefficients * n
        elif len(coefficients) != n:
            raise ValueError(
                f"With --multi-layer, coefficients must be 1 value or match layers count.\n"
                f"Got {len(layers)} layers but {len(coefficients)} coefficients."
            )

        configs.append({
            "layers": layers,
            "methods": methods,
            "coefficients": coefficients,
            "component": component,
        })
    else:
        # Separate runs for each combination
        for layer in layers:
            for coef in coefficients:
                configs.append({
                    "layers": [layer],
                    "methods": [method],
                    "coefficients": [coef],
                    "component": component,
                })

    return configs


# =============================================================================
# Evaluation
# =============================================================================

async def compute_baseline(
    model,
    tokenizer,
    questions: List[str],
    eval_prompt: str,
    judge: TraitJudge,
    use_chat_template: bool,
    temperature: float = 0.0,
) -> Dict:
    """Compute baseline scores (no steering)."""
    print("\nComputing baseline (no steering)...")

    all_trait_scores = []
    all_coherence_scores = []

    for question in tqdm(questions, desc="baseline"):
        formatted = format_prompt(question, tokenizer, use_chat_template=use_chat_template)
        response = generate_response(model, tokenizer, formatted, temperature=temperature)
        scores = await judge.score_batch(eval_prompt, [(question, response)])

        if scores[0]["trait_score"] is not None:
            all_trait_scores.append(scores[0]["trait_score"])
        if scores[0].get("coherence_score") is not None:
            all_coherence_scores.append(scores[0]["coherence_score"])

    baseline = {
        "trait_mean": sum(all_trait_scores) / len(all_trait_scores) if all_trait_scores else None,
        "n": len(all_trait_scores),
    }

    if all_coherence_scores:
        baseline["coherence_mean"] = sum(all_coherence_scores) / len(all_coherence_scores)

    print(f"  Baseline: trait={baseline['trait_mean']:.1f}, n={baseline['n']}")
    return baseline


async def evaluate_config(
    model,
    tokenizer,
    experiment: str,
    trait: str,
    config: Dict,
    questions: List[str],
    eval_prompt: str,
    judge: TraitJudge,
    use_chat_template: bool,
    rollouts: int,
    temperature: float,
    incremental: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """
    Evaluate a single config (which may steer one or multiple layers).

    Returns (result dict, responses list).
    """
    layers = config["layers"]
    methods = config["methods"]
    coefficients = config["coefficients"]
    component = config["component"]

    is_multi_layer = len(layers) > 1

    # Load vectors
    raw_vectors = []
    for layer, method in zip(layers, methods):
        vector = load_vector(experiment, trait, layer, method, component)
        if vector is None:
            raise FileNotFoundError(f"Vector not found: layer={layer}, method={method}, component={component}")
        raw_vectors.append(vector)

    # Compute incremental vectors if requested
    if incremental and len(raw_vectors) > 1:
        vectors = [raw_vectors[0]]  # First layer gets full vector
        for i in range(1, len(raw_vectors)):
            v_inc = raw_vectors[i] - raw_vectors[i-1]
            vectors.append(v_inc)
        # Log incremental norms
        print(f"  Incremental vector norms: {[f'{v.norm().item():.2f}' for v in vectors]}")
    else:
        vectors = raw_vectors

    # Config description for logging
    if is_multi_layer:
        desc = f"L{layers} c{coefficients}"
    else:
        desc = f"L{layers[0]} c{coefficients[0]}"

    # Phase 1: Generate responses
    all_qa_pairs = []
    print(f"  Generating responses for {desc}...")

    for question in tqdm(questions, desc=f"{desc} gen", leave=False):
        formatted = format_prompt(question, tokenizer, use_chat_template=use_chat_template)

        for rollout in range(rollouts):
            # Check if this is a baseline run (all coefficients are 0)
            is_baseline = all(c == 0 for c in coefficients)

            if is_baseline:
                response = generate_response(model, tokenizer, formatted, temperature=temperature)
            elif is_multi_layer:
                # Multi-layer steering
                steering_configs = list(zip(layers, vectors, coefficients))
                with MultiLayerSteeringHook(model, steering_configs, component=component):
                    response = generate_response(model, tokenizer, formatted, temperature=temperature)
            else:
                # Single-layer steering
                with SteeringHook(model, vectors[0], layers[0], coefficients[0], component=component):
                    response = generate_response(model, tokenizer, formatted, temperature=temperature)

            all_qa_pairs.append((question, response, rollout))

    # Phase 2: Score responses
    print(f"  Scoring {len(all_qa_pairs)} responses...")
    qa_for_scoring = [(q, r) for q, r, _ in all_qa_pairs]
    all_scores = await judge.score_batch(eval_prompt, qa_for_scoring)

    # Phase 3: Collect results and responses
    all_trait_scores = []
    all_coherence_scores = []
    responses_data = []

    for (question, response, rollout), score_data in zip(all_qa_pairs, all_scores):
        if score_data["trait_score"] is not None:
            all_trait_scores.append(score_data["trait_score"])
        if score_data.get("coherence_score") is not None:
            all_coherence_scores.append(score_data["coherence_score"])

        responses_data.append({
            "question": question,
            "response": response,
            "rollout": rollout,
            "trait_score": score_data["trait_score"],
            "coherence_score": score_data.get("coherence_score"),
        })

    result = {
        "trait_mean": sum(all_trait_scores) / len(all_trait_scores) if all_trait_scores else None,
        "n": len(all_trait_scores),
    }

    if len(all_trait_scores) > 1:
        mean = result["trait_mean"]
        result["trait_std"] = (sum((x - mean)**2 for x in all_trait_scores) / len(all_trait_scores))**0.5

    if all_coherence_scores:
        result["coherence_mean"] = sum(all_coherence_scores) / len(all_coherence_scores)

    # Log result
    trait_str = f"{result['trait_mean']:.1f}" if result['trait_mean'] else "N/A"
    coh_str = f"{result.get('coherence_mean', 0):.1f}" if result.get('coherence_mean') else "N/A"
    print(f"  {desc}: trait={trait_str}, coherence={coh_str}, n={result['n']}")

    return result, responses_data


async def run_evaluation(
    experiment: str,
    trait: str,
    layers: List[int],
    method: str,
    coefficients: List[float],
    component: str,
    multi_layer: bool,
    model_name: str,
    rollouts: int,
    temperature: float,
    judge_provider: str,
    subset_questions: Optional[int],
    vector_experiment: Optional[str] = None,
    vector_trait: Optional[str] = None,
    incremental: bool = False,
) -> None:
    """
    Run steering evaluation with runs-based results structure.

    Loads/creates results, computes baseline if needed, evaluates configs, saves.
    If a config already exists, overwrites that run instead of appending.
    """
    # Default vector source to experiment/trait
    if vector_experiment is None:
        vector_experiment = experiment
    if vector_trait is None:
        vector_trait = trait

    # Warn about rollouts > 1 with temperature == 0
    if rollouts > 1 and temperature == 0.0:
        print(f"\nWarning: rollouts={rollouts} but temperature=0.0")
        print("  With temp=0, all rollouts will be identical (deterministic).")
        print("  Use --temperature > 0 for variance estimation.\n")

    # Load prompts and check for file match
    prompts_data, prompts_file = load_eval_prompts(trait)
    questions = prompts_data["questions"]
    if subset_questions:
        questions = questions[:subset_questions]
    eval_prompt = prompts_data["eval_prompt"]

    # Load or create results
    results = load_or_create_results(experiment, trait, prompts_file)

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name)
    num_layers = get_num_layers(model)

    # Load experiment config for chat template setting
    config = load_experiment_config(experiment)
    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    # Validate layers
    layers = [l for l in layers if 0 <= l < num_layers]
    if not layers:
        raise ValueError(f"No valid layers. Model has {num_layers} layers (0-{num_layers-1})")

    print(f"\nTrait: {trait}")
    print(f"Model: {model_name} ({num_layers} layers)")
    print(f"Chat template: {use_chat_template}")
    print(f"Vectors from: {vector_experiment}/{vector_trait}")
    print(f"Method: {method}")
    print(f"Component: {component}")
    print(f"Questions: {len(questions)}")
    print(f"Rollouts: {rollouts}")
    print(f"Temperature: {temperature}")
    print(f"Multi-layer: {multi_layer}")
    print(f"Existing runs: {len(results['runs'])}")

    # Initialize judge
    judge = TraitJudge(provider=judge_provider)

    # Compute baseline if not present
    if results["baseline"] is None:
        results["baseline"] = await compute_baseline(
            model, tokenizer, questions, eval_prompt, judge,
            use_chat_template=use_chat_template, temperature=temperature
        )
        save_results(results, experiment, trait)  # Save immediately after baseline
    else:
        print(f"\nUsing existing baseline: trait={results['baseline']['trait_mean']:.1f}")

    # Generate configs
    configs = generate_configs(layers, method, coefficients, component, multi_layer)
    print(f"\nConfigs to evaluate: {len(configs)}")

    # Evaluate each config
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Evaluating config: {config}")

        timestamp = datetime.now().isoformat()
        result, responses = await evaluate_config(
            model, tokenizer, vector_experiment, vector_trait, config,
            questions, eval_prompt, judge,
            use_chat_template=use_chat_template,
            rollouts=rollouts, temperature=temperature,
            incremental=incremental
        )

        # Save responses
        save_responses(responses, experiment, trait, config, timestamp)

        # Check if this config already exists
        existing_idx = find_existing_run_index(results, config)
        run_data = {
            "config": config,
            "result": result,
            "timestamp": timestamp,
        }

        # Record vector source if different from experiment/trait
        if vector_experiment != experiment or vector_trait != trait:
            run_data["vector_source"] = f"{vector_experiment}/{vector_trait}"

        if existing_idx is not None:
            print(f"  Overwriting existing run at index {existing_idx}")
            results["runs"][existing_idx] = run_data
        else:
            results["runs"].append(run_data)

        # Save after each run (so progress isn't lost)
        save_results(results, experiment, trait)

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Baseline: {results['baseline']['trait_mean']:.1f}")
    print(f"Total runs: {len(results['runs'])}")

    # Find best run
    best_run = None
    best_score = float('-inf')
    for run in results['runs']:
        score = run['result'].get('trait_mean')
        if score is not None and score > best_score:
            best_score = score
            best_run = run

    if best_run:
        print(f"Best run: {best_run['config']}")
        print(f"  trait_mean={best_score:.1f} (delta={best_score - results['baseline']['trait_mean']:.1f})")


def main():
    parser = argparse.ArgumentParser(description="Steering evaluation")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--trait", required=True, help="Trait path (e.g., cognitive_state/confidence)")
    parser.add_argument(
        "--layers",
        default="16",
        help="Layers: single '16', range '5-20', list '5,10,15', or 'all'"
    )
    parser.add_argument(
        "--coefficients",
        default="2.0",
        help="Comma-separated coefficients"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name/path")
    parser.add_argument("--method", default="probe", help="Vector extraction method")
    parser.add_argument("--component", default="residual", choices=["residual", "attn_out", "mlp_out"],
                        help="Which component to steer")
    parser.add_argument("--rollouts", type=int, default=1,
                        help="Rollouts per question (>1 only useful with --temperature > 0)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = deterministic)")
    parser.add_argument("--judge", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--subset", type=int, help="Use subset of questions (for faster testing)")
    parser.add_argument("--multi-layer", action="store_true",
                        help="Steer all layers simultaneously (instead of separate runs)")
    parser.add_argument("--vector-from-trait",
                        help="Load vectors from different experiment/trait: 'experiment/category/trait'")
    parser.add_argument("--auto-coef", type=float,
                        help="Auto-compute coefficient from target perturbation ratio (e.g., 0.6 for 60%% of activation norm)")
    parser.add_argument("--incremental", action="store_true",
                        help="Use incremental vectors (v[i] - v[i-1]) to avoid double-counting shared components")

    args = parser.parse_args()

    # Parse --vector-from-trait
    if args.vector_from_trait:
        parts = args.vector_from_trait.split('/', 1)
        if len(parts) != 2:
            parser.error("--vector-from-trait must be 'experiment/category/trait'")
        vector_experiment, vector_trait = parts
    else:
        vector_experiment, vector_trait = args.experiment, args.trait

    # Parse layers and coefficients
    # Need model to validate layers, but we'll do that inside run_evaluation
    layers = parse_layers(args.layers, num_layers=100)  # Temporary max, validated later

    # Auto-compute coefficients if requested
    if args.auto_coef:
        coefficients = []
        for layer in layers:
            coef = compute_auto_coef(
                vector_experiment, vector_trait, layer,
                args.method, args.component, args.auto_coef
            )
            if coef is None:
                parser.error(f"Could not compute auto-coef for layer {layer}")
            coefficients.append(coef)
            print(f"Auto-coef L{layer}: {coef:.1f} (target ratio {args.auto_coef})")
    else:
        coefficients = parse_coefficients(args.coefficients)

    asyncio.run(run_evaluation(
        experiment=args.experiment,
        trait=args.trait,
        layers=layers,
        method=args.method,
        coefficients=coefficients,
        component=args.component,
        multi_layer=args.multi_layer,
        model_name=args.model,
        rollouts=args.rollouts,
        temperature=args.temperature,
        judge_provider=args.judge,
        subset_questions=args.subset,
        vector_experiment=vector_experiment,
        vector_trait=vector_trait,
        incremental=args.incremental,
    ))


if __name__ == "__main__":
    main()
