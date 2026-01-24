#!/usr/bin/env python3
"""
Cross-domain steering test: Apply one vector to another domain's questions,
judged by the target domain's definition.

Usage:
    python analysis/ood/cross_domain_steering.py \
        --experiment gemma-2-2b \
        --source-trait formality_variations/general \
        --target-traits formality_variations/business,formality_variations/academic
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import json
import fire
import asyncio
from typing import List, Dict, Optional
from tqdm import tqdm

from utils.paths import get, get_vector_path, get_model_variant
from utils.model import load_model
from utils.judge import TraitJudge
from utils.generation import generate_batch
from core import SteeringHook


def load_vector(experiment: str, trait: str, model_variant: str, layer: int, method: str = "probe", position: str = "response[:5]") -> Optional[torch.Tensor]:
    """Load trait vector."""
    path = get_vector_path(experiment, trait, method, layer, model_variant, "residual", position)
    if not path.exists():
        return None
    return torch.load(path, weights_only=True)


def load_trait_data(trait: str) -> tuple:
    """Load steering questions and definition for a trait."""
    questions_path = get('datasets.trait_steering', trait=trait)
    with open(questions_path) as f:
        data = json.load(f)

    definition_path = get('datasets.trait_definition', trait=trait)
    with open(definition_path) as f:
        definition = f.read().strip()

    return data["questions"], definition


async def evaluate_cross_domain(
    model,
    tokenizer,
    source_vector: torch.Tensor,
    layer: int,
    coefficient: float,
    target_questions: List[str],
    target_definition: str,
    judge: TraitJudge,
    max_new_tokens: int = 128,
) -> Dict:
    """Evaluate steering with source vector on target domain."""

    # Apply chat template
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True
        )
        for q in target_questions
    ]

    # Generate baseline
    baseline_responses = generate_batch(
        model, tokenizer, formatted_prompts,
        max_new_tokens=max_new_tokens,
    )

    # Generate with steering
    path = f"model.layers.{layer}"
    with SteeringHook(model, source_vector, path, coefficient=coefficient):
        steered_responses = generate_batch(
            model, tokenizer, formatted_prompts,
            max_new_tokens=max_new_tokens,
        )

    # Judge using target definition
    trait_name = "formality"
    baseline_scores = await judge.score_steering_batch(
        list(zip(target_questions, baseline_responses)),
        trait_name, target_definition
    )
    steered_scores = await judge.score_steering_batch(
        list(zip(target_questions, steered_responses)),
        trait_name, target_definition
    )

    baseline_trait = sum(s["trait_score"] for s in baseline_scores) / len(baseline_scores)
    steered_trait = sum(s["trait_score"] for s in steered_scores) / len(steered_scores)
    baseline_coh = sum(s["coherence_score"] for s in baseline_scores) / len(baseline_scores)
    steered_coh = sum(s["coherence_score"] for s in steered_scores) / len(steered_scores)

    return {
        "baseline_trait": baseline_trait,
        "steered_trait": steered_trait,
        "delta": steered_trait - baseline_trait,
        "baseline_coh": baseline_coh,
        "steered_coh": steered_coh,
    }


async def main(
    experiment: str = "gemma-2-2b",
    model_variant: str = None,
    source_trait: str = "formality_variations/general",
    target_traits: str = "formality_variations/business,formality_variations/academic,formality_variations/social,formality_variations/technical",
    layer: int = 15,
    coefficient: float = 168.0,
    method: str = "probe",
    subset: int = 5,
    position: str = "response[:5]",
):
    """
    Run cross-domain steering evaluation.

    Uses source_trait's vector, applies to each target_trait's questions,
    judges using each target_trait's definition.
    """
    # Resolve model variant
    variant = get_model_variant(experiment, model_variant, mode="application")
    model_variant = variant['name']
    model_name = variant['model']
    lora = variant.get('lora')

    target_list = [t.strip() for t in target_traits.split(",")]

    print(f"Cross-Domain Steering Evaluation")
    print(f"Source vector: {source_trait}")
    print(f"Layer: {layer}, Coefficient: {coefficient}")
    print(f"Targets: {target_list}")
    print()

    # Load model
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, lora=lora)

    # Load source vector
    source_vector = load_vector(experiment, source_trait, model_variant, layer, method, position)
    if source_vector is None:
        print(f"ERROR: Vector not found for {source_trait}")
        return

    judge = TraitJudge()

    results = {}
    for target in target_list:
        target_name = target.split("/")[-1]
        print(f"\n=== {source_trait.split('/')[-1]} → {target_name} ===")

        questions, definition = load_trait_data(target)
        if subset > 0:
            questions = questions[:subset]

        result = await evaluate_cross_domain(
            model=model,
            tokenizer=tokenizer,
            source_vector=source_vector,
            layer=layer,
            coefficient=coefficient,
            target_questions=questions,
            target_definition=definition,
            judge=judge,
        )

        results[target_name] = result
        print(f"  Baseline: {result['baseline_trait']:.1f}")
        print(f"  Steered:  {result['steered_trait']:.1f} (Δ={result['delta']:+.1f})")
        print(f"  Coherence: {result['baseline_coh']:.0f}% → {result['steered_coh']:.0f}%")

    await judge.close()

    # Summary
    print(f"\n{'='*50}")
    print("Summary: general vector → topic questions")
    print(f"{'='*50}")
    print(f"{'Target':<12} {'Baseline':>8} {'Steered':>8} {'Delta':>8} {'Coh':>6}")
    print("-" * 50)
    for target, r in results.items():
        print(f"{target:<12} {r['baseline_trait']:>8.1f} {r['steered_trait']:>8.1f} {r['delta']:>+8.1f} {r['steered_coh']:>5.0f}%")

    avg_delta = sum(r['delta'] for r in results.values()) / len(results)
    print("-" * 50)
    print(f"{'Avg delta:':<12} {avg_delta:>+26.1f}")

    return results


if __name__ == "__main__":
    fire.Fire(lambda **kwargs: asyncio.run(main(**kwargs)))
