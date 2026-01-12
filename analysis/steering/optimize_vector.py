#!/usr/bin/env python3
"""
CMA-ES optimization of vector direction in PCA subspace.

Instead of optimizing coefficients for fixed vectors, this optimizes
the vector direction itself by searching in a low-dimensional PCA subspace.

Input: experiment, trait, layer, component
Output: Optimized vector direction

Usage:
    python analysis/steering/optimize_vector.py \
        --experiment gemma-2-2b \
        --trait chirp/refusal \
        --layer 13 \
        --position "response[:3]" \
        --component attn_contribution \
        --n-components 20
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cma
import torch
import asyncio
import argparse
import time
from typing import List, Tuple
from datetime import datetime

from analysis.steering.data import load_steering_data
from core import BatchedLayerSteeringHook
from analysis.steering.results import init_results_file, append_run, save_responses
from utils.paths import get_vector_dir, get_model_variant, get_activation_path, get_steering_results_path
from utils.model import load_model, format_prompt
from utils.generation import generate_batch
from utils.judge import TraitJudge
from utils.vectors import MIN_COHERENCE, load_vector
from core import VectorSpec


def compute_pca_basis(
    experiment: str,
    trait: str,
    model_variant: str,
    layer: int,
    component: str,
    position: str,
    n_components: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PCA basis from training activations.

    Returns:
        (basis, mean, original_vector) where:
        - basis: [n_components, hidden_dim] - top principal components
        - mean: [hidden_dim] - mean of difference vectors
        - original_vector: [hidden_dim] - the probe-extracted vector for comparison
    """
    # Load activations
    act_path = get_activation_path(experiment, trait, model_variant, component, position)
    if not act_path.exists():
        raise FileNotFoundError(f"Activations not found: {act_path}")

    activations = torch.load(act_path, weights_only=True)  # [n_examples, n_layers, hidden_dim]

    # Load metadata to know pos/neg split
    metadata_path = act_path.parent / "metadata.json"
    import json
    with open(metadata_path) as f:
        metadata = json.load(f)

    n_pos = metadata["n_examples_pos"]
    n_neg = metadata["n_examples_neg"]

    # Extract layer activations
    layer_acts = activations[:, layer, :]  # [n_examples, hidden_dim]
    pos_acts = layer_acts[:n_pos]  # [n_pos, hidden_dim]
    neg_acts = layer_acts[n_pos:n_pos + n_neg]  # [n_neg, hidden_dim]

    # Compute difference vectors (each pos paired with each neg, or use mean)
    # For simplicity, use centered activations
    pos_centered = pos_acts - pos_acts.mean(dim=0)
    neg_centered = neg_acts - neg_acts.mean(dim=0)

    # Combine for PCA - we want directions that separate pos from neg
    # Use the difference: pos_mean - neg_mean as the "target direction"
    # And find components that capture variance in this direction

    # Stack all activations for PCA
    all_acts = torch.cat([pos_acts, neg_acts], dim=0).float()  # [n_total, hidden_dim], float32
    mean = all_acts.mean(dim=0)
    centered = all_acts - mean

    # SVD for PCA
    # centered: [n_samples, hidden_dim]
    # We want principal components of the hidden_dim space
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    # Vh: [min(n_samples, hidden_dim), hidden_dim]
    # Top k components: Vh[:k, :]

    basis = Vh[:n_components, :]  # [n_components, hidden_dim]

    # Load original probe vector for comparison
    original_vector = load_vector(experiment, trait, layer, model_variant, "probe", component, position)
    if original_vector is not None:
        original_vector = original_vector.float()
    else:
        # Fall back to mean_diff
        original_vector = (pos_acts.mean(dim=0) - neg_acts.mean(dim=0)).float()
        original_vector = original_vector / original_vector.norm()

    print(f"  PCA basis: {basis.shape}")
    print(f"  Variance explained by top {n_components}: {(S[:n_components]**2).sum() / (S**2).sum() * 100:.1f}%")
    print(f"  Original vector norm: {original_vector.norm():.4f}")

    return basis, mean, original_vector


def reconstruct_vector(weights: torch.Tensor, basis: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Reconstruct full vector from PCA weights.

    Args:
        weights: [n_components] - weights for each principal component
        basis: [n_components, hidden_dim] - PCA basis
        normalize: if True, normalize to unit length (magnitude controlled by coef)
                   if False, magnitude emerges from weights (use coef=1.0)

    Returns:
        vector: [hidden_dim] - reconstructed vector
    """
    # Weighted sum of basis vectors
    vector = (weights.unsqueeze(1) * basis).sum(dim=0)  # [hidden_dim]
    if normalize:
        vector = vector / (vector.norm() + 1e-8)
    return vector


async def run_cma_es(
    experiment: str,
    trait: str,
    layer: int,
    component: str,
    position: str,
    n_components: int = 20,
    n_generations: int = 15,
    popsize: int = 8,
    max_new_tokens: int = 12,
    n_questions: int = 5,
    coherence_threshold: float = MIN_COHERENCE,
    base_coef: float = 90.0,
):
    """
    Run CMA-ES to optimize vector direction in PCA subspace.
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

    # Compute PCA basis
    extraction_variant = get_model_variant(experiment, mode="extraction")['name']
    print(f"\nComputing PCA basis for L{layer}...")
    basis, mean, original_vector = compute_pca_basis(
        experiment, trait, extraction_variant, layer, component, position, n_components
    )
    basis = basis.to(model.device)
    original_vector = original_vector.to(model.device)

    # Initial weights: equal weight per component, scaled by base_coef
    # Total magnitude will be sqrt(n_components) * (base_coef / n_components) = base_coef / sqrt(n_components)
    # This gives a reasonable starting magnitude while allowing CMA-ES to find the direction
    x0 = [base_coef / n_components] * n_components
    print(f"  Initial weights: {base_coef/n_components:.1f} per component ({n_components} components)")

    # Check initial vector magnitude
    init_vec = reconstruct_vector(torch.tensor(x0), basis.cpu())
    print(f"  Initial vector magnitude: {init_vec.norm():.1f}")

    # Initialize judge
    judge = TraitJudge()

    # CMA-ES setup
    # sigma0 controls exploration radius - smaller = more conservative search
    sigma0 = base_coef * 0.1  # ~10% of typical magnitude (was 30%)
    opts = {
        'popsize': popsize,
        'maxiter': n_generations,
        'verbose': -9,
    }

    print(f"\n{'='*60}")
    print(f"CMA-ES Vector Direction Optimization")
    print(f"{'='*60}")
    print(f"Layer: {layer}")
    print(f"PCA components: {n_components}")
    print(f"Base coefficient: {base_coef}")
    print(f"Population: {popsize}/generation, {n_generations} generations")
    print(f"Questions: {n_questions}, Tokens: {max_new_tokens}")
    print(f"Fitness: trait if coh >= {coherence_threshold} else -1000 (hard floor)")
    print(f"{'='*60}\n")

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_result = None
    best_responses = None
    generation = 0

    try:
        while not es.stop():
            generation += 1
            candidates = es.ask()
            n_candidates = len(candidates)

            print(f"--- Generation {generation}/{n_generations} ---")

            # Build vectors for each candidate
            candidate_vectors = []
            for weights in candidates:
                weights_tensor = torch.tensor(weights, dtype=torch.float32, device=model.device)
                vector = reconstruct_vector(weights_tensor, basis)
                candidate_vectors.append(vector)

            # Build batched prompts
            batched_prompts = []
            for _ in candidates:
                batched_prompts.extend(formatted_questions)

            # Build steering configs (coef=1.0 since magnitude is in the vector)
            steering_configs = []
            for cand_idx, vector in enumerate(candidate_vectors):
                batch_start = cand_idx * n_questions
                batch_end = (cand_idx + 1) * n_questions
                steering_configs.append((
                    layer,
                    vector,
                    1.0,  # magnitude baked into vector
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

                fitnesses.append(-fitness)  # CMA-ES minimizes

                # Compute similarity to original and magnitude
                weights_tensor = torch.tensor(weights, dtype=torch.float32, device=model.device)
                opt_vector = reconstruct_vector(weights_tensor, basis)
                # Normalize for similarity comparison
                opt_dir = opt_vector / (opt_vector.norm() + 1e-8)
                similarity = (opt_dir @ original_vector).item()
                magnitude = opt_vector.norm().item()

                if best_result is None or fitness > best_result.get("fitness", -999):
                    best_result = {
                        "weights": list(weights),
                        "trait_mean": trait_mean,
                        "coherence_mean": coherence_mean,
                        "fitness": fitness,
                        "magnitude": magnitude,
                    }
                    # Track best responses for saving
                    cand_responses = all_responses[start:end]
                    best_responses = [
                        {"question": q, "response": r, "trait_score": s["trait_score"], "coherence_score": s.get("coherence_score")}
                        for q, r, s in zip(questions, cand_responses, cand_scores)
                    ]

                print(f"  [{cand_idx+1}/{n_candidates}] trait={trait_mean:.1f} coh={coherence_mean:.1f} fit={fitness:.1f} mag={magnitude:.1f} sim={similarity:.3f}")

            es.tell(candidates, fitnesses)

            if best_result:
                print(f"  Best: trait={best_result['trait_mean']:.1f}, coh={best_result['coherence_mean']:.1f}")

    finally:
        await judge.close()

    # Final result
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")

    if best_result:
        print(f"Best trait: {best_result['trait_mean']:.1f}")
        print(f"Best coherence: {best_result['coherence_mean']:.1f}")
        print(f"Best fitness: {best_result['fitness']:.1f}")
        print(f"Best magnitude: {best_result['magnitude']:.1f}")

        # Compute final vector and similarity
        weights_tensor = torch.tensor(best_result['weights'], dtype=torch.float32, device=model.device)
        opt_vector = reconstruct_vector(weights_tensor, basis)
        similarity = (opt_vector @ original_vector).item()
        print(f"Similarity to original probe: {similarity:.4f}")

        # Save optimized vector to standard extraction path with method="cma_es"
        output_dir = get_vector_dir(experiment, trait, "cma_es", extraction_variant, component, position)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"layer{layer}.pt"
        torch.save(opt_vector.cpu(), output_path)
        print(f"Saved optimized vector: {output_path}")

        # Save metadata
        import json
        metadata = {
            "layer": layer,
            "component": component,
            "position": position,
            "n_pca_components": len(best_result['weights']),
            "base_coef": base_coef,
            "coherence_threshold": coherence_threshold,
            "best_trait": best_result['trait_mean'],
            "best_coherence": best_result['coherence_mean'],
            "best_fitness": best_result['fitness'],
            "best_magnitude": best_result['magnitude'],
            "similarity_to_probe": similarity,
            "timestamp": datetime.now().isoformat(),
        }
        metadata_path = output_dir / f"layer{layer}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")

        # Save to steering results.jsonl (so get_best_vector can find it)
        prompt_set = "steering"  # Default prompt set
        model_variant = extraction_variant  # Use same variant for steering results
        results_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)

        # Initialize results file if needed
        if not results_path.exists():
            init_results_file(
                experiment, trait, model_variant, steering_data.prompts_file,
                model_name, experiment, "openai", position, prompt_set
            )

        # Build config in VectorSpec format
        config = {
            "vectors": [VectorSpec(
                layer=layer,
                component=component,
                position=position,
                method="cma_es",
                weight=1.0,  # Magnitude baked into vector
            ).to_dict()]
        }
        result = {
            "trait_mean": best_result['trait_mean'],
            "coherence_mean": best_result['coherence_mean'],
            "n": n_questions,
        }

        append_run(experiment, trait, model_variant, config, result, position, prompt_set)
        print(f"Saved to results.jsonl")

        # Save responses
        if best_responses:
            timestamp = datetime.now().isoformat()
            save_responses(best_responses, experiment, trait, model_variant, position, prompt_set, config, timestamp)
            print(f"Saved responses")

    print(f"{'='*60}")

    return best_result


def main():
    parser = argparse.ArgumentParser(description="CMA-ES vector direction optimization")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--component", default="attn_contribution")
    parser.add_argument("--position", default="response[:3]")
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--n-questions", type=int, default=5)
    parser.add_argument("--coherence-threshold", type=float, default=MIN_COHERENCE)
    parser.add_argument("--base-coef", type=float, default=90.0)

    args = parser.parse_args()

    asyncio.run(run_cma_es(
        experiment=args.experiment,
        trait=args.trait,
        layer=args.layer,
        component=args.component,
        position=args.position,
        n_components=args.n_components,
        n_generations=args.generations,
        popsize=args.popsize,
        max_new_tokens=args.max_new_tokens,
        n_questions=args.n_questions,
        coherence_threshold=args.coherence_threshold,
        base_coef=args.base_coef,
    ))


if __name__ == "__main__":
    main()
