"""
Project trait vectors through unembedding to see what tokens they "mean".

Logit lens projects a direction in activation space through the model's final
RMSNorm + unembedding matrix to produce a distribution over vocabulary tokens.
The highest-scoring tokens reveal what a trait vector "means" in token space:

    Sycophancy vector:
      Toward (+): "great", "excellent", "beautiful", "love", "amazing"
      Away  (-): "either", "unless", "directly"

    Refusal vector:
      Toward (+): "sorry", "cannot", "nor", "laws", "legal"
      Away  (-): "ready", "excellent", "happy", "easy", "good"

Implementation: normed = model.norm(residual); logits = model.lm_head(normed);
then softmax to get probabilities. Use --no-norm to skip the RMSNorm step.

Standalone script — run manually. Writes per-trait files to the canonical
extraction.logit_lens path (experiments/{experiment}/extraction/{trait}/{model_variant}/logit_lens.json).

Input: Experiment, trait (or --all-traits)
Output: Top/bottom tokens for each vector direction, per method

Usage:
    python analysis/vectors/logit_lens.py --experiment gemma-2-2b-it --traits safety/refusal
    python analysis/vectors/logit_lens.py --experiment gemma-2-2b-it --all-traits
    python analysis/vectors/logit_lens.py --experiment gemma-2-2b-it --traits safety/refusal --save
    python analysis/vectors/logit_lens.py --experiment gemma-2-2b-it --all-traits --save
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from collections import defaultdict
from typing import Optional

import torch

from utils.logit_lens import vector_to_vocab, build_common_token_mask
from utils.model import load_model_with_lora, get_inner_model
from utils.vectors import discover_vectors, load_vector_with_baseline
from utils.paths import get as get_path, discover_extracted_traits, get_model_variant


def analyze_trait(
    experiment: str,
    trait: str,
    model_variant: str,
    model,
    tokenizer,
    top_k: int = 20,
    apply_norm: bool = True,
    common_mask: torch.Tensor = None,
) -> Optional[dict]:
    """Analyze all methods for a trait, producing the canonical logit_lens schema.

    Returns None if no vectors exist on disk for this trait.
    """
    candidates = discover_vectors(experiment, trait, model_variant)
    if not candidates:
        return None

    # Pick (component, position): prefer residual + response[:5], fall back deterministically
    candidates.sort(key=lambda c: (
        c["component"] != "residual",
        "response[:5]" not in c["position"],
        c["component"],
        c["position"],
        c["method"],
        c["layer"],
    ))
    component = candidates[0]["component"]
    position = candidates[0]["position"]

    # Filter to the chosen (component, position)
    filtered = [c for c in candidates if c["component"] == component and c["position"] == position]

    n_layers = get_inner_model(model).config.num_hidden_layers
    target = round(n_layers * 0.9)

    # Group by method
    by_method = defaultdict(list)
    for c in filtered:
        by_method[c["method"]].append(c["layer"])

    methods = {}
    for method, available_layers in by_method.items():
        # Pick layer closest to 90% depth; ties prefer higher layer
        chosen_layer = min(available_layers, key=lambda L: (abs(L - target), -L))
        pct = round(chosen_layer / max(n_layers - 1, 1) * 100)

        # Load vector (returns 3-tuple: vector, baseline, layer_meta)
        try:
            vector, _baseline, _meta = load_vector_with_baseline(
                experiment, trait, method, chosen_layer, model_variant, component, position
            )
        except FileNotFoundError:
            continue

        vocab = vector_to_vocab(vector, model, tokenizer, top_k, apply_norm, common_mask)
        methods[method] = {
            "late": {
                "layer": chosen_layer,
                "pct": pct,
                "toward": vocab["toward"],
                "away": vocab["away"],
            }
        }

    if not methods:
        return None

    return {
        "trait": trait,
        "component": component,
        "position": position,
        "n_layers": n_layers,
        "methods": methods,
    }


def print_results(results: Optional[dict]):
    """Pretty print results for a single trait."""
    if results is None:
        print("  No vectors found")
        return

    print(f"\n{'='*60}")
    print(f"Trait: {results['trait']}")
    print(f"Component: {results['component']}, Position: {results['position']}")
    print(f"Layers: {results['n_layers']}")
    print(f"{'='*60}")

    for method, data in results['methods'].items():
        late = data['late']
        print(f"\n  Method: {method} (layer {late['layer']}, {late['pct']}% depth)")

        print(f"\n  Toward (+):")
        for i, item in enumerate(late['toward'], 1):
            print(f"    {i:2d}. {item['token']:20s} {item['value']:+.3f}")

        print(f"\n  Away (-):")
        for i, item in enumerate(late['away'], 1):
            print(f"    {i:2d}. {item['token']:20s} {item['value']:+.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--model-variant", default=None, help="Model variant (default: from experiment config)")
    parser.add_argument("--traits", help="Trait path (e.g., safety/refusal)")
    parser.add_argument("--all-traits", action="store_true", help="Analyze all traits")
    parser.add_argument("--top-k", type=int, default=20, help="Tokens to show (default: 20)")
    parser.add_argument("--no-norm", action="store_true", help="Skip RMSNorm before projection")
    parser.add_argument("--filter-common", action="store_true", help="Filter to common English tokens only")
    parser.add_argument("--max-vocab", type=int, default=10000, help="Max vocab index for common filter (default: 10000)")
    parser.add_argument("--save", action="store_true", help="Save results to canonical per-trait JSON")
    args = parser.parse_args()

    if not args.traits and not args.all_traits:
        parser.error("Must specify --traits or --all-traits")

    # Resolve model variant — use extraction mode so vectors + write path align
    variant = get_model_variant(args.experiment, args.model_variant, mode="extraction")
    model_variant = variant.name
    model_name = variant.model
    lora = variant.lora

    model, tokenizer = load_model_with_lora(model_name, lora_adapter=lora)
    model.eval()

    # Build common token mask if requested
    common_mask = None
    if args.filter_common:
        print(f"Building common token mask (max_vocab={args.max_vocab})...")
        common_mask = build_common_token_mask(tokenizer, args.max_vocab)
        print(f"  {common_mask.sum().item()} tokens pass filter")

    # Determine traits to analyze
    if args.all_traits:
        traits = [f"{cat}/{name}" for cat, name in discover_extracted_traits(args.experiment)]
        print(f"Found {len(traits)} traits")
    else:
        traits = [args.traits]

    # Analyze each trait
    for trait in traits:
        print(f"\nAnalyzing: {trait}")
        results = analyze_trait(
            experiment=args.experiment,
            trait=trait,
            model_variant=model_variant,
            model=model,
            tokenizer=tokenizer,
            top_k=args.top_k,
            apply_norm=not args.no_norm,
            common_mask=common_mask,
        )
        print_results(results)

        # Save per-trait to canonical path
        if args.save and results is not None:
            output_path = get_path('extraction.logit_lens', experiment=args.experiment, trait=trait, model_variant=model_variant)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
