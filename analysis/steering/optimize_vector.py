#!/usr/bin/env python3
"""
CMA-ES optimization of vector direction and coefficient.

Optimizes both the vector direction (in a random subspace around existing vector)
and the steering coefficient jointly.

Input: experiment, trait, layers, component
Output: Optimized vector direction and coefficient per layer

Usage:
    python analysis/steering/optimize_vector.py \
        --experiment gemma-2-2b \
        --trait chirp/refusal \
        --layers 8,9,10,11,12 \
        --component residual

    # Single layer with explicit position
    python analysis/steering/optimize_vector.py \
        --experiment gemma-2-2b \
        --trait chirp/refusal \
        --layers 10 \
        --component residual \
        --position "response[:5]"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cma
import json
import numpy as np
import torch
import asyncio
import argparse
import time
from typing import List, Tuple, Optional, Dict
from datetime import datetime

from analysis.steering.data import load_steering_data
from analysis.steering.evaluate import estimate_activation_norm
from core import BatchedLayerSteeringHook
from analysis.steering.results import init_results_file, append_run, save_responses
from utils.paths import get_vector_dir, get_model_variant, get_steering_results_path
from utils.model import load_model, format_prompt, get_layers_module
from utils.generation import generate_batch
from utils.judge import TraitJudge
from utils.vectors import MIN_COHERENCE, load_vector, get_best_vector
from utils.traits import load_scenarios, load_trait_definition
from core import VectorSpec


def random_orthonormal_basis(dim: int, n_components: int, seed: int = None) -> torch.Tensor:
    """
    Generate random orthonormal basis vectors.

    Args:
        dim: Hidden dimension (e.g., 2304 for Gemma-2-2B)
        n_components: Number of basis vectors
        seed: Optional random seed for reproducibility

    Returns:
        basis: [n_components, dim] - orthonormal basis vectors
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Generate random matrix and orthonormalize via QR decomposition
    random_matrix = torch.randn(dim, n_components)
    Q, _ = torch.linalg.qr(random_matrix)
    return Q.T[:n_components]  # [n_components, dim]


def get_init_vector_and_position(
    experiment: str,
    trait: str,
    layer: int,
    component: str,
    position: Optional[str] = None,
) -> Tuple[Optional[torch.Tensor], str, Optional[str], Optional[Dict]]:
    """
    Get initial vector and position for optimization.

    Tries to find existing best vector for this layer/component.
    Falls back to random if nothing exists.

    Returns:
        (init_vector, position, method, best_info) where:
        - init_vector: Starting vector or None for random
        - position: Position to use (discovered or default)
        - method: Method of init vector (e.g., 'probe') or None
        - best_info: Full info from get_best_vector or None
    """
    # Try to find existing vector for this component
    try:
        # First try to get best vector for this specific layer
        try:
            layer_best = get_best_vector(experiment, trait, component=component, layer=layer)
            init_vector = load_vector(
                experiment, trait, layer,
                layer_best.get('extraction_variant', get_model_variant(experiment, mode="extraction")['name']),
                layer_best['method'], component, layer_best['position']
            )
            return init_vector, layer_best['position'], layer_best['method'], layer_best
        except (FileNotFoundError, ValueError):
            pass  # No layer-specific result, try overall best

        # Fall back to overall best
        best = get_best_vector(experiment, trait, component=component)

        # Check if best is for this layer or different layer
        if best['layer'] == layer:
            # Perfect - use this vector
            init_vector = load_vector(
                experiment, trait, layer,
                best.get('extraction_variant', get_model_variant(experiment, mode="extraction")['name']),
                best['method'], component, best['position']
            )
            return init_vector, best['position'], best['method'], best
        else:
            # Best is different layer - try to load same method/position for our layer
            # But use best's coefficient as fallback (not ideal but better than nothing)
            try:
                init_vector = load_vector(
                    experiment, trait, layer,
                    best.get('extraction_variant', get_model_variant(experiment, mode="extraction")['name']),
                    best['method'], component, best['position']
                )
                return init_vector, best['position'], best['method'], best
            except FileNotFoundError:
                # No vector at this layer with best's method/position
                # Use position from best but random init
                return None, best['position'] if position is None else position, None, best

    except (FileNotFoundError, ValueError) as e:
        # No steering results found - try to find any vector for this component
        extraction_variant = get_model_variant(experiment, mode="extraction")['name']

        # Try common positions
        for try_position in [position, "response[:5]", "response[:3]"]:
            if try_position is None:
                continue
            for method in ["probe", "mean_diff", "gradient"]:
                try:
                    init_vector = load_vector(
                        experiment, trait, layer,
                        extraction_variant, method, component, try_position
                    )
                    return init_vector, try_position, method, None
                except FileNotFoundError:
                    continue

        # Nothing found - use defaults
        default_position = position if position else "response[:5]"
        return None, default_position, None, None


def reconstruct_vector_and_coef(
    init_vector: Optional[torch.Tensor],
    weights: List[float],
    basis: torch.Tensor,
    coef_center: float,
) -> Tuple[torch.Tensor, float]:
    """
    Reconstruct vector and coefficient from CMA-ES weights.

    Args:
        init_vector: Starting vector [hidden_dim] (unit norm) or None for pure basis combination
        weights: [n_components + 1] - first n_components for direction, last for coefficient
        basis: [n_components, hidden_dim] - orthonormal basis
        coef_center: Center value for coefficient (typically activation norm)

    Returns:
        vector: [hidden_dim] - unit norm vector
        coefficient: Steering coefficient (denormalized)
    """
    device = basis.device
    n_components = basis.shape[0]

    # Split weights: direction (first n) and coefficient (last 1)
    direction_weights = torch.tensor(weights[:n_components], dtype=torch.float32, device=device)
    coef_normalized = weights[n_components]  # Centered around 0.8, range ~[0.3, 1.5]

    # Reconstruct direction
    perturbation = (direction_weights.unsqueeze(1) * basis).sum(dim=0)  # [hidden_dim]

    if init_vector is not None:
        vector = init_vector + perturbation
    else:
        vector = perturbation

    # Normalize to unit norm
    vector = vector / (vector.norm() + 1e-8)

    # Denormalize coefficient
    coefficient = coef_normalized * coef_center

    return vector, coefficient


async def run_cma_es_single_layer(
    model,
    tokenizer,
    experiment: str,
    trait: str,
    layer: int,
    component: str,
    position: str,
    init_vector: Optional[torch.Tensor],
    init_method: Optional[str],
    steering_data,
    judge: TraitJudge,
    n_components: int = 20,
    n_generations: int = 15,
    popsize: int = 8,
    max_new_tokens: int = 12,
    n_questions: int = 5,
    coherence_threshold: float = MIN_COHERENCE,
    coef_init_mult: float = 0.5,
    seed: int = None,
) -> Optional[Dict]:
    """
    Run CMA-ES optimization for a single layer.

    Optimizes both vector direction and coefficient jointly.
    Search space: n_components dims for direction + 1 dim for coefficient.
    """
    questions = steering_data.questions[:n_questions]
    use_chat_template = tokenizer.chat_template is not None

    # Format questions once
    formatted_questions = [
        format_prompt(q, tokenizer, use_chat_template=use_chat_template)
        for q in questions
    ]

    # Get hidden dim from model
    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    hidden_dim = config.hidden_size

    # Compute activation norm for this layer (used to scale coefficient)
    coef_center = estimate_activation_norm(model, tokenizer, questions, layer, use_chat_template)
    print(f"  Activation norm (coef center): {coef_center:.1f}")

    # Generate random orthonormal basis for perturbation search
    basis = random_orthonormal_basis(hidden_dim, n_components, seed=seed)
    basis = basis.to(model.device).float()

    # Prepare init vector (ensure unit norm)
    if init_vector is not None:
        init_vector = init_vector.to(model.device).float()
        init_vector = init_vector / (init_vector.norm() + 1e-8)  # Unit norm
        print(f"  Starting from {init_method} vector (unit norm)")
    else:
        print(f"  Starting from random (no existing vector found)")

    # Initial weights: 20 direction dims + 1 coefficient dim
    # Direction: zeros (start at init_vector) or small random
    # Coefficient: start at coef_init_mult (e.g., 0.8 means 80% of activation norm)
    if init_vector is not None:
        x0 = [0.0] * n_components + [coef_init_mult]
        sigma0 = 0.1  # Small perturbations
    else:
        x0 = [0.1] * n_components + [coef_init_mult]
        sigma0 = 0.2  # Slightly larger exploration

    # CMA-ES setup with bounds on coefficient
    # Coefficient normalized: 0.0 to 1.5 (meaning 0% to 150% of activation norm)
    bounds = [
        [-np.inf] * n_components + [0.0],   # Lower bounds
        [np.inf] * n_components + [1.5],    # Upper bounds
    ]
    opts = {
        'popsize': popsize,
        'maxiter': n_generations,
        'verbose': -9,
        'bounds': bounds,
    }

    print(f"\n{'='*60}")
    print(f"CMA-ES Direction + Coefficient Optimization - Layer {layer}")
    print(f"{'='*60}")
    print(f"Component: {component}")
    print(f"Position: {position}")
    print(f"Coef range: [0, {1.5 * coef_center:.0f}] (center={coef_center:.0f})")
    print(f"Coef init: {coef_init_mult * coef_center:.0f} ({coef_init_mult:.0%} of center)")
    print(f"Search dims: {n_components} direction + 1 coefficient")
    print(f"Population: {popsize}/gen, {n_generations} generations")
    print(f"Questions: {n_questions}, Tokens: {max_new_tokens}")
    print(f"Fitness: trait if coh >= {coherence_threshold} else -1000")
    print(f"{'='*60}\n")

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_result = None
    best_responses = None
    best_weights = None
    generation = 0
    fitness_history = []

    while not es.stop():
        generation += 1
        candidates = es.ask()
        n_candidates = len(candidates)

        print(f"--- Generation {generation}/{n_generations} ---")

        # Build vectors and coefficients for each candidate
        candidate_vectors = []
        candidate_coefs = []
        for weights in candidates:
            vector, coef = reconstruct_vector_and_coef(init_vector, weights, basis, coef_center)
            candidate_vectors.append(vector)
            candidate_coefs.append(coef)

        # Build batched prompts
        batched_prompts = []
        for _ in candidates:
            batched_prompts.extend(formatted_questions)

        # Build steering configs (unit norm vectors, per-candidate coefficient)
        steering_configs = []
        for cand_idx, (vector, coef) in enumerate(zip(candidate_vectors, candidate_coefs)):
            batch_start = cand_idx * n_questions
            batch_end = (cand_idx + 1) * n_questions
            steering_configs.append((
                layer,
                vector,
                coef,
                (batch_start, batch_end)
            ))

        # Generate all responses
        t0 = time.time()
        with BatchedLayerSteeringHook(model, steering_configs, component=component):
            all_responses = generate_batch(
                model, tokenizer, batched_prompts,
                max_new_tokens=max_new_tokens
            )
        gen_time = time.time() - t0

        # Build QA pairs
        all_qa_pairs = []
        for cand_idx in range(n_candidates):
            start = cand_idx * n_questions
            end = (cand_idx + 1) * n_questions
            for q, r in zip(questions, all_responses[start:end]):
                all_qa_pairs.append((q, r))

        # Score
        t0 = time.time()
        all_scores = await judge.score_steering_batch(
            all_qa_pairs,
            steering_data.trait_name,
            steering_data.trait_definition
        )
        score_time = time.time() - t0

        print(f"  Generated {len(batched_prompts)} responses ({gen_time:.1f}s), scored ({score_time:.1f}s)")

        # Compute fitness
        fitnesses = []
        gen_best_fitness = -1000
        for cand_idx, weights in enumerate(candidates):
            start = cand_idx * n_questions
            end = (cand_idx + 1) * n_questions
            cand_scores = all_scores[start:end]
            cand_coef = candidate_coefs[cand_idx]

            trait_scores = [s["trait_score"] for s in cand_scores if s["trait_score"] is not None]
            coherence_scores = [s["coherence_score"] for s in cand_scores if s.get("coherence_score") is not None]

            trait_mean = sum(trait_scores) / len(trait_scores) if trait_scores else 0
            coherence_mean = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0

            # Fitness: hard floor on coherence
            if coherence_mean >= coherence_threshold:
                fitness = trait_mean
            else:
                fitness = -1000

            fitnesses.append(-fitness)  # CMA-ES minimizes
            gen_best_fitness = max(gen_best_fitness, fitness)

            # Track best
            if best_result is None or fitness > best_result.get("fitness", -999):
                best_result = {
                    "trait_mean": trait_mean,
                    "coherence_mean": coherence_mean,
                    "fitness": fitness,
                    "coefficient": cand_coef,
                }
                best_weights = list(weights)
                # Track best responses for saving
                cand_responses = all_responses[start:end]
                best_responses = [
                    {"prompt": q, "response": r, "system_prompt": None, "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
                    for q, r, s in zip(questions, cand_responses, cand_scores)
                ]

            print(f"  [{cand_idx+1}/{n_candidates}] trait={trait_mean:.1f} coh={coherence_mean:.1f} coef={cand_coef:.0f} fit={fitness:.1f}")

        es.tell(candidates, fitnesses)
        fitness_history.append(gen_best_fitness)

        if best_result:
            print(f"  Best so far: trait={best_result['trait_mean']:.1f}, coh={best_result['coherence_mean']:.1f}, coef={best_result['coefficient']:.0f}")

    # Final result
    print(f"\n{'='*60}")
    print(f"LAYER {layer} OPTIMIZATION COMPLETE")
    print(f"{'='*60}")

    if best_result and best_result['fitness'] > -1000:
        best_coef = best_result['coefficient']
        print(f"Best trait: {best_result['trait_mean']:.1f}")
        print(f"Best coherence: {best_result['coherence_mean']:.1f}")
        print(f"Best coefficient: {best_coef:.1f}")

        # Compute final vector (unit norm)
        opt_vector, _ = reconstruct_vector_and_coef(init_vector, best_weights, basis, coef_center)

        # Compute similarity to init (both are unit norm)
        if init_vector is not None:
            similarity = (opt_vector @ init_vector).item()
            print(f"Similarity to init vector: {similarity:.4f}")
        else:
            similarity = None

        # Save optimized vector
        extraction_variant = get_model_variant(experiment, mode="extraction")['name']
        output_dir = get_vector_dir(experiment, trait, "cma_es", extraction_variant, component, position)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"layer{layer}.pt"
        torch.save(opt_vector.cpu(), output_path)
        print(f"Saved optimized vector: {output_path}")

        # Save metadata with fitness history
        metadata = {
            "layer": layer,
            "component": component,
            "position": position,
            "n_components": n_components,
            "n_generations": n_generations,
            "popsize": popsize,
            "coef_center": coef_center,
            "best_coefficient": best_coef,
            "coherence_threshold": coherence_threshold,
            "init_method": init_method,
            "best_trait": best_result['trait_mean'],
            "best_coherence": best_result['coherence_mean'],
            "best_fitness": best_result['fitness'],
            "similarity_to_init": similarity,
            "fitness_history": fitness_history,
            "timestamp": datetime.now().isoformat(),
        }
        metadata_path = output_dir / f"layer{layer}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")

        # Save to steering results.jsonl
        model_name = model.config.name_or_path
        prompt_set = "steering"
        results_path = get_steering_results_path(experiment, trait, extraction_variant, position, prompt_set)

        if not results_path.exists():
            init_results_file(
                experiment, trait, extraction_variant, steering_data.prompts_file,
                model_name, experiment, "openai", position, prompt_set
            )

        config = {
            "vectors": [VectorSpec(
                layer=layer,
                component=component,
                position=position,
                method="cma_es",
                weight=best_coef,
            ).to_dict()]
        }
        result = {
            "trait_mean": best_result['trait_mean'],
            "coherence_mean": best_result['coherence_mean'],
            "n": n_questions,
        }

        append_run(experiment, trait, extraction_variant, config, result, position, prompt_set, trait_judge=None)
        print(f"Saved to results.jsonl")

        # Save responses
        if best_responses:
            timestamp = datetime.now().isoformat()
            save_responses(best_responses, experiment, trait, extraction_variant, position, prompt_set, config, timestamp)
            print(f"Saved responses")

        return best_result
    else:
        print("No valid result found (all candidates below coherence threshold)")
        return None


async def run_cma_es_base_mode(
    model,
    tokenizer,
    experiment: str,
    trait: str,
    layer: int,
    component: str,
    position: str,
    init_vector: Optional[torch.Tensor],
    init_method: Optional[str],
    scenarios: List[str],
    trait_name: str,
    trait_definition: str,
    judge: TraitJudge,
    n_components: int = 20,
    n_generations: int = 15,
    popsize: int = 8,
    max_new_tokens: int = 24,
    n_prompts: int = 10,
    n_completions: int = 1,
    coherence_threshold: float = MIN_COHERENCE,
    coef_init_mult: float = 0.5,
    seed: int = None,
) -> Optional[Dict]:
    """
    Run CMA-ES optimization using base model with scenario completions.

    For each candidate vector:
    1. Generate n_completions per scenario prompt
    2. Score trait with score_response (does completion exhibit trait?)
    3. Score coherence (grammar only, no relevance check)
    """
    import random

    # Sample prompts (mix of positive and negative)
    prompts = random.sample(scenarios, min(n_prompts, len(scenarios)))
    total_completions = n_prompts * n_completions

    # Get hidden dim from model
    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    hidden_dim = config.hidden_size

    # Compute activation norm for this layer
    # Use prompts directly (no chat template for base model)
    coef_center = estimate_activation_norm(model, tokenizer, prompts, layer, use_chat_template=False)
    print(f"  Activation norm (coef center): {coef_center:.1f}")

    # Generate random orthonormal basis
    basis = random_orthonormal_basis(hidden_dim, n_components, seed=seed)
    basis = basis.to(model.device).float()

    # Prepare init vector
    if init_vector is not None:
        init_vector = init_vector.to(model.device).float()
        init_vector = init_vector / (init_vector.norm() + 1e-8)
        print(f"  Starting from {init_method} vector (unit norm)")
    else:
        print(f"  Starting from random (no existing vector found)")

    # CMA-ES setup
    if init_vector is not None:
        x0 = [0.0] * n_components + [coef_init_mult]
        sigma0 = 0.1
    else:
        x0 = [0.1] * n_components + [coef_init_mult]
        sigma0 = 0.2

    bounds = [
        [-np.inf] * n_components + [0.0],
        [np.inf] * n_components + [1.5],
    ]
    opts = {
        'popsize': popsize,
        'maxiter': n_generations,
        'verbose': -9,
        'bounds': bounds,
    }

    print(f"\n{'='*60}")
    print(f"CMA-ES Base Mode - Layer {layer}")
    print(f"{'='*60}")
    print(f"Component: {component}")
    print(f"Position: {position}")
    print(f"Coef range: [0, {1.5 * coef_center:.0f}] (center={coef_center:.0f})")
    print(f"Coef init: {coef_init_mult * coef_center:.0f} ({coef_init_mult:.0%} of center)")
    print(f"Search dims: {n_components} direction + 1 coefficient")
    print(f"Population: {popsize}/gen, {n_generations} generations")
    print(f"Prompts: {n_prompts}, Completions/prompt: {n_completions}, Tokens: {max_new_tokens}")
    print(f"Fitness: trait if coh >= {coherence_threshold} else -1000")
    print(f"{'='*60}\n")

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_result = None
    best_responses = None
    best_weights = None
    generation = 0
    fitness_history = []

    while not es.stop():
        generation += 1
        candidates = es.ask()
        n_candidates = len(candidates)

        print(f"--- Generation {generation}/{n_generations} ---")

        # Build vectors and coefficients
        candidate_vectors = []
        candidate_coefs = []
        for weights in candidates:
            vector, coef = reconstruct_vector_and_coef(init_vector, weights, basis, coef_center)
            candidate_vectors.append(vector)
            candidate_coefs.append(coef)

        # Build batched prompts (no chat template for base model)
        batched_prompts = []
        for _ in candidates:
            for p in prompts:
                for _ in range(n_completions):
                    batched_prompts.append(p)

        # Build steering configs
        steering_configs = []
        for cand_idx, (vector, coef) in enumerate(zip(candidate_vectors, candidate_coefs)):
            batch_start = cand_idx * total_completions
            batch_end = (cand_idx + 1) * total_completions
            steering_configs.append((layer, vector, coef, (batch_start, batch_end)))

        # Generate all responses
        t0 = time.time()
        with BatchedLayerSteeringHook(model, steering_configs, component=component):
            all_responses = generate_batch(
                model, tokenizer, batched_prompts,
                max_new_tokens=max_new_tokens
            )
        gen_time = time.time() - t0

        # Build (prompt, response) pairs for scoring
        all_pairs = []
        for cand_idx in range(n_candidates):
            start = cand_idx * total_completions
            end = (cand_idx + 1) * total_completions
            resp_idx = 0
            for p in prompts:
                for _ in range(n_completions):
                    all_pairs.append((p, all_responses[start + resp_idx]))
                    resp_idx += 1

        # Score with score_responses_batch (includes coherence)
        t0 = time.time()
        all_scores = await judge.score_responses_batch(
            all_pairs,
            trait_name,
            trait_definition,
            include_coherence=True
        )
        score_time = time.time() - t0

        print(f"  Generated {len(batched_prompts)} responses ({gen_time:.1f}s), scored ({score_time:.1f}s)")

        # Compute fitness
        fitnesses = []
        gen_best_fitness = -1000
        for cand_idx, weights in enumerate(candidates):
            start = cand_idx * total_completions
            end = (cand_idx + 1) * total_completions
            cand_scores = all_scores[start:end]
            cand_coef = candidate_coefs[cand_idx]

            trait_scores = [s["score"] for s in cand_scores if s["score"] is not None]
            coherence_scores = [s.get("coherence") for s in cand_scores if s.get("coherence") is not None]

            trait_mean = sum(trait_scores) / len(trait_scores) if trait_scores else 0
            coherence_mean = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0

            if coherence_mean >= coherence_threshold:
                fitness = trait_mean
            else:
                fitness = -1000

            fitnesses.append(-fitness)
            gen_best_fitness = max(gen_best_fitness, fitness)

            if best_result is None or fitness > best_result.get("fitness", -999):
                best_result = {
                    "trait_mean": trait_mean,
                    "coherence_mean": coherence_mean,
                    "fitness": fitness,
                    "coefficient": cand_coef,
                }
                best_weights = list(weights)
                # Track best responses
                cand_responses = all_responses[start:end]
                best_responses = [
                    {"prompt": p, "response": r, "trait_score": s["score"], "coherence_score": s.get("coherence")}
                    for p, r, s in zip([pr for pr in prompts for _ in range(n_completions)], cand_responses, cand_scores)
                ]

            print(f"  [{cand_idx+1}/{n_candidates}] trait={trait_mean:.1f} coh={coherence_mean:.1f} coef={cand_coef:.0f} fit={fitness:.1f}")

        es.tell(candidates, fitnesses)
        fitness_history.append(gen_best_fitness)

        if best_result:
            print(f"  Best so far: trait={best_result['trait_mean']:.1f}, coh={best_result['coherence_mean']:.1f}, coef={best_result['coefficient']:.0f}")

    # Final result
    print(f"\n{'='*60}")
    print(f"LAYER {layer} OPTIMIZATION COMPLETE (BASE MODE)")
    print(f"{'='*60}")

    if best_result and best_result['fitness'] > -1000:
        best_coef = best_result['coefficient']
        print(f"Best trait: {best_result['trait_mean']:.1f}")
        print(f"Best coherence: {best_result['coherence_mean']:.1f}")
        print(f"Best coefficient: {best_coef:.1f}")

        # Compute final vector
        opt_vector, _ = reconstruct_vector_and_coef(init_vector, best_weights, basis, coef_center)

        if init_vector is not None:
            similarity = (opt_vector @ init_vector).item()
            print(f"Similarity to init vector: {similarity:.4f}")
        else:
            similarity = None

        # Save
        extraction_variant = get_model_variant(experiment, mode="extraction")['name']
        output_dir = get_vector_dir(experiment, trait, "cma_es", extraction_variant, component, position)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"layer{layer}.pt"
        torch.save(opt_vector.cpu(), output_path)
        print(f"Saved optimized vector: {output_path}")

        # Save metadata
        metadata = {
            "layer": layer,
            "component": component,
            "position": position,
            "mode": "base",
            "n_components": n_components,
            "n_generations": n_generations,
            "popsize": popsize,
            "n_prompts": n_prompts,
            "n_completions": n_completions,
            "coef_center": coef_center,
            "best_coefficient": best_coef,
            "coherence_threshold": coherence_threshold,
            "init_method": init_method,
            "best_trait": best_result["trait_mean"],
            "best_coherence": best_result["coherence_mean"],
            "best_fitness": best_result["fitness"],
            "similarity_to_init": similarity,
            "fitness_history": fitness_history,
            "timestamp": datetime.now().isoformat(),
        }
        metadata_path = output_dir / f"layer{layer}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")

        return best_result
    else:
        print("No valid result found (all candidates below coherence threshold)")
        return None


async def run_cma_es(
    experiment: str,
    trait: str,
    layers: List[int],
    component: str,
    position: Optional[str] = None,
    mode: str = "instruct",
    n_components: int = 20,
    n_generations: int = 15,
    popsize: int = 8,
    max_new_tokens: int = 12,
    n_questions: int = 5,
    n_prompts: int = 10,
    n_completions: int = 1,
    coherence_threshold: float = MIN_COHERENCE,
    coef_init_mult: float = 0.5,
    seed: int = None,
):
    """
    Run CMA-ES optimization for multiple layers sequentially.

    Optimizes both direction and coefficient jointly. Coefficient search is
    centered around activation norm for each layer.

    Args:
        mode: "instruct" for Q&A evaluation, "base" for scenario completion
        coef_init_mult: Initial coefficient as fraction of activation norm (default 0.5).
    """
    # Load model based on mode
    if mode == "base":
        variant = get_model_variant(experiment, mode="extraction")
        # Load scenarios
        all_scenarios = load_scenarios(trait)
        scenarios = [s['prompt'] for s in all_scenarios['positive']] + [s['prompt'] for s in all_scenarios['negative']]
        trait_name = trait.split('/')[-1]
        trait_definition = load_trait_definition(trait)
        steering_data = None
    else:
        variant = get_model_variant(experiment, mode="application")
        steering_data = load_steering_data(trait)
        scenarios = None
        trait_name = None
        trait_definition = None

    model_name = variant['model']
    print(f"Loading model: {model_name} (mode={mode})")
    model, tokenizer = load_model(model_name)

    # Initialize judge
    judge = TraitJudge()

    results = {}

    try:
        for layer in layers:
            print(f"\n{'#'*60}")
            print(f"# LAYER {layer}")
            print(f"{'#'*60}")

            # Get init vector and position for this layer
            init_vector, layer_position, init_method, _ = get_init_vector_and_position(
                experiment, trait, layer, component, position
            )

            if mode == "base":
                result = await run_cma_es_base_mode(
                    model=model,
                    tokenizer=tokenizer,
                    experiment=experiment,
                    trait=trait,
                    layer=layer,
                    component=component,
                    position=layer_position,
                    init_vector=init_vector,
                    init_method=init_method,
                    scenarios=scenarios,
                    trait_name=trait_name,
                    trait_definition=trait_definition,
                    judge=judge,
                    n_components=n_components,
                    n_generations=n_generations,
                    popsize=popsize,
                    max_new_tokens=max_new_tokens,
                    n_prompts=n_prompts,
                    n_completions=n_completions,
                    coherence_threshold=coherence_threshold,
                    coef_init_mult=coef_init_mult,
                    seed=seed,
                )
            else:
                result = await run_cma_es_single_layer(
                    model=model,
                    tokenizer=tokenizer,
                    experiment=experiment,
                    trait=trait,
                    layer=layer,
                    component=component,
                    position=layer_position,
                    init_vector=init_vector,
                    init_method=init_method,
                    steering_data=steering_data,
                    judge=judge,
                    n_components=n_components,
                    n_generations=n_generations,
                    popsize=popsize,
                    max_new_tokens=max_new_tokens,
                    n_questions=n_questions,
                    coherence_threshold=coherence_threshold,
                    coef_init_mult=coef_init_mult,
                    seed=seed,
                )

            results[layer] = result

    finally:
        await judge.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION SUMMARY ({mode} mode)")
    print(f"{'='*60}")
    for layer, result in results.items():
        if result and result['fitness'] > -1000:
            print(f"  L{layer}: trait={result['trait_mean']:.1f}, coh={result['coherence_mean']:.1f}")
        else:
            print(f"  L{layer}: no valid result")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="CMA-ES vector direction + coefficient optimization")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--layers", required=True, help="Comma-separated layers, e.g., '8,9,10,11,12'")
    parser.add_argument("--component", default="residual")
    parser.add_argument("--position", default=None, help="Position (default: auto-discover from existing vectors)")
    parser.add_argument("--mode", choices=["instruct", "base"], default="instruct",
                        help="instruct: Q&A with instruct model, base: scenario completion with base model")
    parser.add_argument("--n-components", type=int, default=20, help="Subspace dimensionality for direction")
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12,
                        help="Max tokens per generation (default: 12 for instruct, suggest 24+ for base)")
    parser.add_argument("--n-questions", type=int, default=5, help="Questions per candidate (instruct mode)")
    parser.add_argument("--n-prompts", type=int, default=10, help="Prompts per candidate (base mode)")
    parser.add_argument("--n-completions", type=int, default=1, help="Completions per prompt (base mode)")
    parser.add_argument("--coherence-threshold", type=float, default=MIN_COHERENCE)
    parser.add_argument("--coef-init", type=float, default=0.5,
                        help="Initial coef as fraction of activation norm (default: 0.5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Parse layers
    layers = [int(x.strip()) for x in args.layers.split(",")]

    asyncio.run(run_cma_es(
        experiment=args.experiment,
        trait=args.trait,
        layers=layers,
        component=args.component,
        position=args.position,
        mode=args.mode,
        n_components=args.n_components,
        n_generations=args.generations,
        popsize=args.popsize,
        max_new_tokens=args.max_new_tokens,
        n_questions=args.n_questions,
        n_prompts=args.n_prompts,
        n_completions=args.n_completions,
        coherence_threshold=args.coherence_threshold,
        coef_init_mult=args.coef_init,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
