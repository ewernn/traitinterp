#!/usr/bin/env python3
"""
Steering evaluation - validate trait vectors via causal intervention.

Input:
    - experiment: Experiment where steering results are saved
    - vector-from-trait OR traits: Single or multiple trait specs

Output:
    - experiments/{experiment}/steering/{trait}/results.json
      Runs-based structure that accumulates across invocations.
    - experiments/{experiment}/steering/{trait}/responses/
      Generated responses for each config.

Usage:
    # Basic usage - adaptive search finds good coefficients
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --vector-from-trait {experiment}/{category}/{trait}

    # Multiple traits (loads model once, evaluates sequentially)
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --traits "exp/cat/trait1,exp/cat/trait2,exp/cat/trait3" \\
        --load-in-8bit

    # Specific layers only
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --vector-from-trait {experiment}/{category}/{trait} \\
        --layers 10,12,14

    # 70B+ models with quantization
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --vector-from-trait {experiment}/{category}/{trait} \\
        --load-in-8bit

    # Sequential mode (one layer at a time, slower but lower memory)
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --vector-from-trait {experiment}/{category}/{trait} \\
        --no-batch

    # Manual coefficients (skip adaptive search)
    python analysis/steering/evaluate.py \\
        --experiment {experiment} \\
        --vector-from-trait {experiment}/{category}/{trait} \\
        --coefficients 50,100,150

    # Multi-layer weighted steering (delta-proportional coefficients)
    python analysis/steering/evaluate.py \\
        --experiment gemma-2-2b-it \\
        --vector-from-trait gemma-2-2b-base/epistemic/optimism \\
        --layers 6-18 \\
        --multi-layer weighted --global-scale 1.5
"""

import sys
import gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import argparse
import asyncio
import json
from typing import List, Dict, Optional
from datetime import datetime

from analysis.steering.data import load_steering_data
from analysis.steering.multilayer import run_multilayer_evaluation
from analysis.steering.results import load_or_create_results, save_results, save_responses, save_baseline_responses
from analysis.steering.coef_search import (
    evaluate_and_save,
    adaptive_search_layer,
    batched_adaptive_search,
)
from utils.generation import generate_batch
from utils.judge import TraitJudge
from utils.paths import get, get_vector_path
from utils.model import format_prompt, tokenize_prompt, load_experiment_config, load_model, load_model_with_lora, get_num_layers, get_layers_module
from utils.paths import get_model_variant
from utils.vectors import MIN_COHERENCE
from other.server.client import get_model_or_client, ModelClient


def load_model_handle(model_name: str, load_in_8bit: bool = False, load_in_4bit: bool = False, no_server: bool = False, lora_adapter: str = None):
    """Load model locally or get client if server available.

    Returns:
        (model, tokenizer, is_remote) tuple
    """
    # LoRA requires local loading
    if lora_adapter:
        model, tokenizer = load_model_with_lora(model_name, lora_adapter=lora_adapter, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
        return model, tokenizer, False

    if not no_server:
        handle = get_model_or_client(model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
        if isinstance(handle, ModelClient):
            print(f"Using model server (model: {model_name})")
            return handle, handle, True  # model, tokenizer, is_remote
        model, tokenizer = handle
        return model, tokenizer, False
    else:
        model, tokenizer = load_model(model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
        return model, tokenizer, False


def parse_layers(layers_arg: str, num_layers: int) -> List[int]:
    """
    Parse layers argument.

    Args:
        layers_arg: "all", single number "16", range "5-20", list "5,10,15",
                    or percentage range "30%-60%"
        num_layers: Total layers in model

    Returns:
        List of layer indices
    """
    if layers_arg.lower() == "all":
        return list(range(num_layers))
    elif "%" in layers_arg:
        # Percentage range: "30%-60%" -> layers at 30% to 60% of depth
        parts = layers_arg.replace("%", "").split("-")
        start_pct = int(parts[0]) / 100
        end_pct = int(parts[1]) / 100 if len(parts) > 1 else start_pct
        start = int(num_layers * start_pct)
        end = int(num_layers * end_pct)
        return list(range(start, end + 1))
    elif "-" in layers_arg and "," not in layers_arg:
        start, end = layers_arg.split("-")
        return list(range(int(start), int(end) + 1))
    elif "," in layers_arg:
        return [int(x) for x in layers_arg.split(",")]
    else:
        return [int(layers_arg)]


def parse_coefficients(coef_arg: Optional[str]) -> Optional[List[float]]:
    """Parse comma-separated coefficients. Returns None if not provided."""
    if coef_arg is None:
        return None
    return [float(c) for c in coef_arg.split(",")]


def load_vector(experiment: str, trait: str, layer: int, model_variant: str, method: str = "probe", component: str = "residual", position: str = "response[:]") -> Optional[torch.Tensor]:
    """Load trait vector from experiment. Returns None if not found."""
    vector_file = get_vector_path(experiment, trait, method, layer, model_variant, component, position)

    if not vector_file.exists():
        return None

    return torch.load(vector_file, weights_only=True)


def load_cached_activation_norms(experiment: str) -> Dict[int, float]:
    """
    Load cached activation norms from extraction_evaluation.json.

    Returns:
        {layer: norm} or empty dict if not available
    """
    eval_path = get('extraction_eval.evaluation', experiment=experiment)
    if not eval_path.exists():
        return {}

    try:
        with open(eval_path) as f:
            data = json.load(f)
        norms = data.get('activation_norms', {})
        # Convert string keys back to int
        return {int(k): v for k, v in norms.items()}
    except (json.JSONDecodeError, KeyError):
        return {}


def estimate_activation_norm(
    model,
    tokenizer,
    prompts: List[str],
    layer: int,
    use_chat_template: bool,
) -> float:
    """Estimate activation norm at a layer by running a few prompts."""
    norms = []

    def capture_hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        norm = hidden[:, -1, :].float().norm().item()
        norms.append(norm)

    layer_module = get_layers_module(model)[layer]
    handle = layer_module.register_forward_hook(capture_hook)

    try:
        for prompt in prompts[:3]:
            formatted = format_prompt(prompt, tokenizer, use_chat_template=use_chat_template)
            inputs = tokenize_prompt(formatted, tokenizer, use_chat_template).to(model.device)
            with torch.no_grad():
                model(**inputs)
    finally:
        handle.remove()

    return sum(norms) / len(norms) if norms else 100.0


# =============================================================================
# Core Evaluation
# =============================================================================

async def compute_baseline(
    model,
    tokenizer,
    questions: List[str],
    trait_name: str,
    trait_definition: str,
    judge: TraitJudge,
    use_chat_template: bool,
    max_new_tokens: int = 256,
    eval_prompt: Optional[str] = None,
) -> tuple[Dict, List[Dict]]:
    """Compute baseline scores (no steering) with batched generation.

    Returns:
        Tuple of (baseline_stats, response_data) where response_data contains
        questions, responses, and scores for saving.
    """
    print("\nComputing baseline (no steering)...")

    # Format all questions
    formatted = [format_prompt(q, tokenizer, use_chat_template=use_chat_template) for q in questions]

    # Generate all responses in batch
    responses = generate_batch(model, tokenizer, formatted, max_new_tokens=max_new_tokens)

    # Score all at once
    all_qa_pairs = list(zip(questions, responses))
    all_scores = await judge.score_steering_batch(all_qa_pairs, trait_name, trait_definition, eval_prompt=eval_prompt)

    all_trait_scores = [s["trait_score"] for s in all_scores if s["trait_score"] is not None]
    all_coherence_scores = [s["coherence_score"] for s in all_scores if s.get("coherence_score") is not None]

    baseline = {
        "trait_mean": sum(all_trait_scores) / len(all_trait_scores) if all_trait_scores else None,
        "n": len(all_trait_scores),
    }

    if all_coherence_scores:
        baseline["coherence_mean"] = sum(all_coherence_scores) / len(all_coherence_scores)

    # Build response data for saving
    response_data = [
        {
            "question": q,
            "response": r,
            "trait_score": s["trait_score"],
            "coherence_score": s.get("coherence_score"),
        }
        for q, r, s in zip(questions, responses, all_scores)
    ]

    print(f"  Baseline: trait={baseline['trait_mean']:.1f}, n={baseline['n']}")
    return baseline, response_data


async def run_evaluation(
    experiment: str,
    trait: str,
    vector_experiment: str,
    model_variant: str,
    layers_arg: str,
    coefficients: Optional[List[float]],
    method: str,
    component: str,
    position: str,
    prompt_set: str,
    model_name: str,
    judge_provider: str,
    subset: Optional[int],
    n_search_steps: int,
    up_mult: float,
    down_mult: float,
    start_mult: float = 0.7,
    momentum: float = 0.0,
    batched: bool = True,
    model=None,
    tokenizer=None,
    judge=None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    lora_adapter: str = None,
    max_new_tokens: int = 256,
    eval_prompt: Optional[str] = None,
    use_default_prompt: bool = False,
):
    """
    Main evaluation flow.

    If coefficients provided: evaluate those directly.
    Otherwise: run adaptive search to find good coefficients.

    Args:
        batched: If True (default), run all layers in parallel batches.
                 If False, run each layer sequentially.
        model: Pre-loaded model (optional, loads if not provided)
        tokenizer: Pre-loaded tokenizer (optional)
        judge: Pre-created TraitJudge (optional, creates if not provided)
        load_in_8bit: Use 8-bit quantization when loading model
        load_in_4bit: Use 4-bit quantization when loading model
        eval_prompt: Custom trait scoring prompt (auto-detected from steering.json if None)
        use_default_prompt: Force V3c default, ignore steering.json eval_prompt
    """
    # Load prompts and trait definition
    steering_data = load_steering_data(trait)
    questions = steering_data.questions
    if subset:
        questions = questions[:subset]

    # Resolve eval_prompt: explicit override > use_default flag > steering.json
    if use_default_prompt:
        effective_eval_prompt = None
    elif eval_prompt is not None:
        effective_eval_prompt = eval_prompt
    else:
        effective_eval_prompt = steering_data.eval_prompt

    # Load model if not provided
    # Note: Steering uses hooks, so we force local mode
    should_close_judge = False
    if model is None:
        model, tokenizer, _ = load_model_handle(model_name, load_in_8bit, load_in_4bit, no_server=True, lora_adapter=lora_adapter)
    num_layers = get_num_layers(model)

    # Load experiment config
    config = load_experiment_config(experiment)
    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    # Parse and validate layers using actual model size
    layers = parse_layers(layers_arg, num_layers)
    layers = [l for l in layers if 0 <= l < num_layers]
    if not layers:
        raise ValueError(f"No valid layers. Model has {num_layers} layers (0-{num_layers-1})")

    # Load/create results
    results = load_or_create_results(
        experiment, trait, model_variant, steering_data.prompts_file, model_name, vector_experiment, judge_provider, position, prompt_set
    )

    # Create judge if not provided
    if judge is None:
        judge = TraitJudge()
        should_close_judge = True

    print(f"\nTrait: {trait}")
    print(f"Model: {model_name} ({num_layers} layers)")
    print(f"Chat template: {use_chat_template}")
    print(f"Vectors from: {vector_experiment}/{trait} @ {position}")
    print(f"Questions: {len(questions)}")
    print(f"Existing runs: {len(results['runs'])}")

    # Compute baseline if needed
    if results["baseline"] is None:
        results["baseline"], baseline_responses = await compute_baseline(
            model, tokenizer, questions, steering_data.trait_name, steering_data.trait_definition,
            judge, use_chat_template, eval_prompt=effective_eval_prompt
        )
        save_baseline_responses(baseline_responses, experiment, trait, model_variant, position, prompt_set)
        save_results(results, experiment, trait, model_variant, position, prompt_set)
    else:
        print(f"\nUsing existing baseline: trait={results['baseline']['trait_mean']:.1f}")

    # Load vectors and compute base coefficients
    # Try cached activation norms first (from extraction_evaluation.json)
    cached_norms = load_cached_activation_norms(vector_experiment)
    if cached_norms:
        print(f"\nUsing cached activation norms from extraction_evaluation.json")
    else:
        print(f"\nNo cached norms, will estimate activation norms...")

    print(f"Loading vectors...")
    layer_data = []
    for layer in layers:
        vector = load_vector(vector_experiment, trait, layer, model_variant, method, component, position)
        if vector is None:
            print(f"  L{layer}: Vector not found, skipping")
            continue

        vec_norm = vector.norm().item()

        # Use cached norm if available, otherwise estimate
        if layer in cached_norms:
            act_norm = cached_norms[layer]
        else:
            act_norm = estimate_activation_norm(model, tokenizer, questions, layer, use_chat_template)

        base_coef = act_norm / vec_norm

        layer_data.append({
            "layer": layer,
            "vector": vector,
            "base_coef": base_coef,
        })
        print(f"  L{layer}: base_coef={base_coef:.0f}")

    if not layer_data:
        print("No valid layers with vectors found")
        return

    # Determine coefficients to test
    if coefficients is not None:
        # Manual mode: test specified coefficients for each layer
        print(f"\nManual coefficients: {coefficients}")
        for ld in layer_data:
            for coef in coefficients:
                await evaluate_and_save(
                    model, tokenizer, ld["vector"], ld["layer"], coef,
                    questions, steering_data.trait_name, steering_data.trait_definition,
                    judge, use_chat_template, component,
                    results, experiment, trait, model_variant, vector_experiment, method,
                    position=position, prompt_set=prompt_set,
                    eval_prompt=effective_eval_prompt
                )
    elif batched and len(layer_data) > 1:
        # Batched adaptive search (default) - all layers in parallel
        await batched_adaptive_search(
            model, tokenizer, layer_data, questions, steering_data.trait_name, steering_data.trait_definition,
            judge, use_chat_template, component, results, experiment, trait, model_variant,
            vector_experiment, method, position=position, prompt_set=prompt_set, n_steps=n_search_steps,
            up_mult=up_mult, down_mult=down_mult, start_mult=start_mult, momentum=momentum,
            max_new_tokens=max_new_tokens, eval_prompt=effective_eval_prompt
        )
    else:
        # Sequential adaptive search for each layer
        print(f"\nSequential adaptive search ({n_search_steps} steps per layer)")
        for ld in layer_data:
            await adaptive_search_layer(
                model, tokenizer, ld["vector"], ld["layer"], ld["base_coef"],
                questions, steering_data.trait_name, steering_data.trait_definition,
                judge, use_chat_template, component,
                results, experiment, trait, model_variant, vector_experiment, method,
                position=position, prompt_set=prompt_set, n_steps=n_search_steps, up_mult=up_mult, down_mult=down_mult, start_mult=start_mult, momentum=momentum,
                max_new_tokens=max_new_tokens, eval_prompt=effective_eval_prompt
            )

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Baseline: {results['baseline']['trait_mean']:.1f}")
    print(f"Total runs: {len(results['runs'])}")

    best_run = max(results['runs'], key=lambda r: r['result'].get('trait_mean') or 0, default=None)
    if best_run:
        score = best_run['result']['trait_mean']
        coh = best_run['result'].get('coherence_mean', 0)
        delta = score - results['baseline']['trait_mean']
        layer = best_run['config']['vectors'][0]['layer']
        coef = best_run['config']['vectors'][0]['weight']
        print(f"Best: L{layer} c{coef:.0f}")
        print(f"  trait={score:.1f} (+{delta:.1f}), coherence={coh:.1f}")

    if should_close_judge:
        await judge.close()


def main():
    parser = argparse.ArgumentParser(description="Steering evaluation")
    parser.add_argument("--experiment", required=True,
                        help="Experiment where steering results are saved")

    # Mutually exclusive: single trait or multiple traits
    trait_group = parser.add_mutually_exclusive_group(required=True)
    trait_group.add_argument("--vector-from-trait",
                        help="Single trait: 'experiment/category/trait'")
    trait_group.add_argument("--traits",
                        help="Multiple traits (comma-separated): 'exp/cat/t1,exp/cat/t2'")
    parser.add_argument("--layers", default="30%-60%",
                        help="Layers: 'all', '30%%-60%%' (default), single '16', range '5-20', or list '5,10,15'")
    parser.add_argument("--coefficients",
                        help="Manual coefficients (comma-separated). If not provided, uses adaptive search.")
    parser.add_argument("--model-variant", default=None,
                        help="Model variant for steering (default: from experiment defaults.application)")
    parser.add_argument("--method", default="probe", help="Vector extraction method")
    parser.add_argument("--component", default="residual", choices=["residual", "attn_out", "mlp_out", "attn_contribution", "mlp_contribution", "k_proj", "v_proj"])
    parser.add_argument("--position", default="response[:5]",
                        help="Token position for vectors (default: response[:5])")
    parser.add_argument("--prompt-set", default="steering",
                        help="Prompt set name for result isolation (default: steering)")
    parser.add_argument("--judge", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--subset", type=int, default=5, help="Use subset of questions (default: 5, use --subset 0 for all)")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens to generate per response (default: 64)")
    parser.add_argument("--search-steps", type=int, default=5,
                        help="Number of adaptive search steps per layer (default: 5)")
    parser.add_argument("--up-mult", type=float, default=1.3,
                        help="Coefficient multiplier when increasing (default: 1.3)")
    parser.add_argument("--down-mult", type=float, default=0.85,
                        help="Coefficient multiplier when decreasing (default: 0.85)")
    parser.add_argument("--start-mult", type=float, default=0.7,
                        help="Starting coefficient as fraction of base_coef (default: 0.7). Use lower values (e.g., 0.05) for sensitive models.")
    parser.add_argument("--momentum", type=float, default=0.1,
                        help="Momentum for coefficient updates (0.0=direct, 0.7=smoothed). Default: 0.1")
    parser.add_argument("--no-batch", action="store_true",
                        help="Disable batched layer evaluation (run layers sequentially)")
    parser.add_argument("--multi-layer", choices=["weighted", "orthogonal"],
                        help="Multi-layer steering mode: 'weighted' (delta-proportional) or 'orthogonal'")
    parser.add_argument("--global-scale", type=float, default=1.0,
                        help="Global scale for multi-layer coefficients (default: 1.0)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit quantization (for 70B+ models)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization")
    parser.add_argument("--no-server", action="store_true",
                        help="Force local model loading (skip model server check)")

    # Prompt override options (mutually exclusive)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--no-custom-prompt",
        action="store_true",
        help="Ignore eval_prompt from steering.json, use V3c default scoring"
    )
    prompt_group.add_argument(
        "--eval-prompt-from",
        type=str,
        metavar="TRAIT_PATH",
        help="Load eval_prompt from different trait's steering.json (e.g., 'persona_vectors_instruction/evil')"
    )

    args = parser.parse_args()

    # Parse trait specs (single or multiple)
    if args.traits:
        trait_specs = [t.strip() for t in args.traits.split(',')]
    else:
        trait_specs = [args.vector_from_trait]

    # Parse trait specs: 'category/trait' uses current experiment, 'exp/category/trait' uses specified
    parsed_traits = []
    for spec in trait_specs:
        parts = spec.split('/')
        if len(parts) == 2:
            # category/trait - use current experiment
            parsed_traits.append((args.experiment, spec))
        elif len(parts) == 3:
            # experiment/category/trait
            parsed_traits.append((parts[0], f"{parts[1]}/{parts[2]}"))
        else:
            parser.error(f"Invalid trait spec '{spec}': use 'category/trait' or 'experiment/category/trait'")

    # Resolve model variant
    variant = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = variant['name']
    model_name = variant['model']
    lora = variant.get('lora')

    coefficients = parse_coefficients(args.coefficients)

    # Run evaluation(s) - layers parsed after model loads to get actual num_layers
    asyncio.run(_run_main(
        args=args,
        parsed_traits=parsed_traits,
        model_variant=model_variant,
        model_name=model_name,
        lora=lora,
        layers_arg=args.layers,
        coefficients=coefficients,
    ))


async def _run_main(args, parsed_traits, model_variant, model_name, lora, layers_arg, coefficients):
    """Async main to handle model/judge lifecycle."""
    multi_trait = len(parsed_traits) > 1

    # Load model once if multiple traits
    # Note: Steering uses hooks, so we force local mode (no_server=True)
    model, tokenizer, judge = None, None, None
    if multi_trait:
        print(f"\nEvaluating {len(parsed_traits)} traits with shared model")
        model, tokenizer, _ = load_model_handle(
            model_name,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            no_server=True,  # Steering requires local hooks
            lora_adapter=lora
        )
        judge = TraitJudge()

    # Resolve eval_prompt override
    effective_eval_prompt = None
    use_default = args.no_custom_prompt
    if args.eval_prompt_from:
        override_data = load_steering_data(args.eval_prompt_from)
        effective_eval_prompt = override_data.eval_prompt
        if not effective_eval_prompt:
            print(f"Warning: {args.eval_prompt_from} has no eval_prompt, using V3c default")
            use_default = True
        else:
            print(f"Using eval_prompt from: {args.eval_prompt_from}")
    elif use_default:
        print("Using V3c default scoring (--no-custom-prompt)")

    try:
        for vector_experiment, trait in parsed_traits:
            if multi_trait:
                print(f"\n{'='*60}")
                print(f"TRAIT: {vector_experiment}/{trait}")
                print(f"{'='*60}")

            if args.multi_layer:
                await run_multilayer_evaluation(
                    experiment=args.experiment,
                    trait=trait,
                    model_variant=model_variant,
                    vector_experiment=vector_experiment,
                    layers=layers_arg,
                    mode=args.multi_layer,
                    global_scale=args.global_scale,
                    method=args.method,
                    component=args.component,
                    position=args.position,
                    prompt_set=args.prompt_set,
                    model_name=model_name,
                    subset=args.subset,
                    model=model,
                    tokenizer=tokenizer,
                    judge=judge,
                    load_in_8bit=args.load_in_8bit,
                    load_in_4bit=args.load_in_4bit,
                    lora_adapter=lora,
                    max_new_tokens=args.max_new_tokens,
                    eval_prompt=effective_eval_prompt,
                    use_default_prompt=use_default,
                )
            else:
                await run_evaluation(
                    experiment=args.experiment,
                    trait=trait,
                    vector_experiment=vector_experiment,
                    model_variant=model_variant,
                    layers_arg=layers_arg,
                    coefficients=coefficients,
                    method=args.method,
                    component=args.component,
                    position=args.position,
                    prompt_set=args.prompt_set,
                    model_name=model_name,
                    judge_provider=args.judge,
                    subset=args.subset,
                    n_search_steps=args.search_steps,
                    up_mult=args.up_mult,
                    down_mult=args.down_mult,
                    start_mult=args.start_mult,
                    momentum=args.momentum,
                    batched=not args.no_batch,
                    model=model,
                    tokenizer=tokenizer,
                    judge=judge,
                    load_in_8bit=args.load_in_8bit,
                    load_in_4bit=args.load_in_4bit,
                    lora_adapter=lora,
                    max_new_tokens=args.max_new_tokens,
                    eval_prompt=effective_eval_prompt,
                    use_default_prompt=use_default,
                )
    finally:
        if judge is not None:
            await judge.close()
        # Cleanup GPU memory
        if model is not None:
            del model
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if multi_trait:
        print(f"\n{'='*60}")
        print(f"COMPLETED {len(parsed_traits)} TRAITS")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
