#!/usr/bin/env python3
"""
CMA-ES optimization for multi-layer steering ensembles.

Input: experiment, trait, layers to combine
Output: Optimized weights for ensemble

Usage:
    python analysis/steering/optimize_ensemble.py \
        --experiment gemma-2-2b \
        --trait chirp/refusal \
        --layers 11,13 \
        --position "response[:3]" \
        --component attn_contribution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cma
import torch
import asyncio
import argparse
import time
from typing import List, Dict, Tuple
from datetime import datetime

from analysis.steering.data import load_steering_data
from core import VectorSpec, BatchedLayerSteeringHook
from utils.paths import get_model_variant
from utils.model import load_model, format_prompt
from utils.generation import generate_batch
from utils.judge import TraitJudge
from utils.ensembles import create_ensemble, save_ensemble
from utils.vectors import MIN_COHERENCE, load_vector


async def run_cma_es(
    experiment: str,
    trait: str,
    layers: List[int],
    component: str,
    position: str,
    method: str = "probe",
    n_generations: int = 10,
    popsize: int = 6,
    max_new_tokens: int = 12,
    n_questions: int = 5,
    coherence_threshold: float = MIN_COHERENCE,
):
    """
    Run CMA-ES optimization for ensemble weights.

    CMA-ES maintains a Gaussian N(μ, σ²C) and adapts it each generation:
      1. Sample `popsize` candidates from current Gaussian
      2. Evaluate fitness of each (in parallel!)
      3. Update μ toward best candidates
      4. Update C to stretch toward promising directions
      5. Update σ based on progress
    """
    # Load steering data
    steering_data = load_steering_data(trait)
    questions = steering_data.questions[:n_questions]

    # Load model
    variant = get_model_variant(experiment, mode="application")
    model_name = variant['model']
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name)
    use_chat_template = tokenizer.chat_template is not None

    # Format questions once
    formatted_questions = [
        format_prompt(q, tokenizer, use_chat_template=use_chat_template)
        for q in questions
    ]

    # Load vectors and get base coefficients
    extraction_variant = get_model_variant(experiment, mode="extraction")['name']
    vectors = []  # List of (layer, vector, base_coef)

    for layer in layers:
        vector = load_vector(experiment, trait, layer, extraction_variant, method, component, position)
        if vector is None:
            raise FileNotFoundError(f"Vector not found for L{layer} {method} {component} {position}")

        # Base coefficient from activation norm (same as coef_search.py)
        # For unit-normalized vectors, use layer-appropriate scaling
        # These are approximate values from our steering runs
        base_coef_map = {9: 54, 11: 62, 13: 90}  # From our earlier results
        base_coef = base_coef_map.get(layer, 80)

        vectors.append((layer, vector, base_coef))
        print(f"  L{layer}: norm={vector.norm().item():.4f}, base_coef={base_coef}")

    # Initialize judge
    judge = TraitJudge()

    # =========================================================================
    # CMA-ES Setup
    # =========================================================================
    n_layers = len(layers)

    # Initial weights: base_coef / n_components (so they sum to ~single-layer strength)
    x0 = [v[2] / n_layers for v in vectors]

    # Bounds: [0, 1.5 * base_coef] per layer
    lower_bounds = [0] * n_layers
    upper_bounds = [1.5 * v[2] for v in vectors]

    # Sigma: ~30% of the range for exploration
    sigma0 = sum(v[2] for v in vectors) / n_layers * 0.3

    opts = {
        'popsize': popsize,
        'bounds': [lower_bounds, upper_bounds],
        'maxiter': n_generations,
        'verbose': -9,  # Quiet
    }

    print(f"\n{'='*60}")
    print(f"CMA-ES Optimization")
    print(f"{'='*60}")
    print(f"Layers: {layers}")
    print(f"Initial weights: {[f'{w:.1f}' for w in x0]}")
    print(f"Bounds: {[(f'{lo:.0f}', f'{hi:.0f}') for lo, hi in zip(lower_bounds, upper_bounds)]}")
    print(f"Sigma0: {sigma0:.1f}")
    print(f"Population: {popsize}/generation, {n_generations} generations")
    print(f"Questions: {n_questions}, Tokens: {max_new_tokens}")
    print(f"Fitness: trait if coh >= {coherence_threshold} else -1000 (hard floor)")
    print(f"{'='*60}\n")

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_result = None
    generation = 0

    try:
        while not es.stop():
            generation += 1

            # Get candidates from CMA-ES
            candidates = es.ask()  # List of weight vectors
            n_candidates = len(candidates)

            print(f"--- Generation {generation}/{n_generations} ---")

            # =========================================================
            # Parallel evaluation of all candidates
            # =========================================================

            # Build batched prompts: [q1_c1, q2_c1, ..., qN_c1, q1_c2, ..., qN_cK]
            batched_prompts = []
            for _ in candidates:
                batched_prompts.extend(formatted_questions)

            # Build steering configs for BatchedLayerSteeringHook
            # Each candidate gets its own batch slice, with all layers steered
            steering_configs = []
            for cand_idx, weights in enumerate(candidates):
                batch_start = cand_idx * n_questions
                batch_end = (cand_idx + 1) * n_questions

                for layer_idx, (layer, vector, _) in enumerate(vectors):
                    steering_configs.append((
                        layer,
                        vector,
                        weights[layer_idx],  # Weight for this layer
                        (batch_start, batch_end)
                    ))

            # Generate all responses in one batch
            t0 = time.time()
            with BatchedLayerSteeringHook(model, steering_configs, component=component):
                all_responses = generate_batch(
                    model, tokenizer, batched_prompts,
                    max_new_tokens=max_new_tokens
                )
            gen_time = time.time() - t0

            # Build QA pairs for scoring
            all_qa_pairs = []
            for cand_idx in range(n_candidates):
                start = cand_idx * n_questions
                end = (cand_idx + 1) * n_questions
                for q, r in zip(questions, all_responses[start:end]):
                    all_qa_pairs.append((q, r))

            # Score all responses
            t0 = time.time()
            all_scores = await judge.score_steering_batch(
                all_qa_pairs,
                steering_data.trait_name,
                steering_data.trait_definition
            )
            score_time = time.time() - t0

            print(f"  Generated {len(batched_prompts)} responses ({gen_time:.1f}s), scored ({score_time:.1f}s)")

            # Compute fitness per candidate
            fitnesses = []
            for cand_idx, weights in enumerate(candidates):
                start = cand_idx * n_questions
                end = (cand_idx + 1) * n_questions
                cand_scores = all_scores[start:end]

                trait_scores = [s["trait_score"] for s in cand_scores if s["trait_score"] is not None]
                coherence_scores = [s["coherence_score"] for s in cand_scores if s.get("coherence_score") is not None]

                trait_mean = sum(trait_scores) / len(trait_scores) if trait_scores else 0
                coherence_mean = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0

                # Fitness: hard floor on coherence
                if coherence_mean >= coherence_threshold:
                    fitness = trait_mean
                else:
                    fitness = -1000

                # CMA-ES minimizes, so negate
                fitnesses.append(-fitness)

                # Track best (by fitness, not just trait)
                if best_result is None or fitness > best_result.get("fitness", -999):
                    best_result = {
                        "weights": list(weights),
                        "trait_mean": trait_mean,
                        "coherence_mean": coherence_mean,
                        "fitness": fitness,
                    }

                w_str = ", ".join(f"{w:.1f}" for w in weights)
                print(f"  [{cand_idx+1}/{n_candidates}] [{w_str}] trait={trait_mean:.1f} coh={coherence_mean:.1f} fit={fitness:.1f}")

            # Update CMA-ES
            es.tell(candidates, fitnesses)

            # Show current best
            if best_result:
                print(f"  Best: trait={best_result['trait_mean']:.1f}, coh={best_result['coherence_mean']:.1f}")
            print(f"  CMA-ES mean: {[f'{x:.1f}' for x in es.result.xfavorite]}")

    finally:
        await judge.close()

    # Final result
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")

    if best_result:
        print(f"Best weights: {[f'{w:.1f}' for w in best_result['weights']]}")
        print(f"Best trait: {best_result['trait_mean']:.1f}")
        print(f"Best coherence: {best_result['coherence_mean']:.1f}")
        print(f"Best fitness: {best_result['fitness']:.1f}")

        # Save as ensemble
        specs = [
            VectorSpec(layer=layer, component=component, position=position, method=method)
            for layer, _, _ in vectors
        ]
        ensemble = create_ensemble(
            experiment, trait, extraction_variant,
            specs=specs,
            coefficients=best_result['weights'],
            coefficient_source='cma_es',
        )
        ensemble['steering_results'] = {
            'trait_mean': best_result['trait_mean'],
            'coherence_mean': best_result['coherence_mean'],
            'baseline': steering_data.trait_definition,  # TODO: get actual baseline
            'timestamp': datetime.now().isoformat(),
        }
        path = save_ensemble(experiment, trait, extraction_variant, ensemble)
        print(f"Saved ensemble: {path}")
    else:
        print("No valid result found (all candidates below coherence threshold)")

    print(f"{'='*60}")

    return best_result


def main():
    parser = argparse.ArgumentParser(description="CMA-ES ensemble optimization")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--layers", required=True, help="Comma-separated layers, e.g., '11,13'")
    parser.add_argument("--component", default="attn_contribution")
    parser.add_argument("--position", default="response[:3]")
    parser.add_argument("--method", default="probe")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--popsize", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--n-questions", type=int, default=5)
    parser.add_argument("--coherence-threshold", type=float, default=MIN_COHERENCE)

    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    asyncio.run(run_cma_es(
        experiment=args.experiment,
        trait=args.trait,
        layers=layers,
        component=args.component,
        position=args.position,
        method=args.method,
        n_generations=args.generations,
        popsize=args.popsize,
        max_new_tokens=args.max_new_tokens,
        n_questions=args.n_questions,
        coherence_threshold=args.coherence_threshold,
    ))


if __name__ == "__main__":
    main()
