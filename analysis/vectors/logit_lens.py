"""
Project trait vectors through unembedding to see what tokens they "mean".

Input: Experiment, trait (or --all-traits)
Output: Top/bottom tokens for each vector direction

Usage:
    python analysis/vectors/logit_lens.py --experiment gemma-2-2b-it --trait safety/refusal
    python analysis/vectors/logit_lens.py --experiment gemma-2-2b-it --all-traits
    python analysis/vectors/logit_lens.py --experiment gemma-2-2b-it --trait safety/refusal --save
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json

from core.logit_lens import vector_to_vocab, build_common_token_mask
from utils.model import load_model, load_experiment_config
from utils.vectors import get_best_vector, load_vector_with_baseline
from utils.paths import get as get_path, discover_extracted_traits


def analyze_trait(
    experiment: str,
    trait: str,
    model,
    tokenizer,
    top_k: int = 20,
    apply_norm: bool = True,
    common_mask: torch.Tensor = None,
) -> dict:
    """Analyze a single trait vector."""
    # Get best vector metadata
    try:
        meta = get_best_vector(experiment, trait)
    except FileNotFoundError as e:
        return {"error": str(e)}

    # Load the actual vector
    vector, baseline, layer_meta = load_vector_with_baseline(
        experiment=experiment,
        trait=trait,
        method=meta['method'],
        layer=meta['layer'],
        component=meta['component'],
        position=meta['position'],
    )

    # Project to vocabulary
    vocab_results = vector_to_vocab(vector, model, tokenizer, top_k, apply_norm, common_mask)

    return {
        "trait": trait,
        "layer": meta['layer'],
        "method": meta['method'],
        "component": meta['component'],
        "position": meta['position'],
        "source": meta['source'],
        "score": meta['score'],
        **vocab_results,
    }


def print_results(results: dict):
    """Pretty print results for a single trait."""
    if "error" in results:
        print(f"  Error: {results['error']}")
        return

    print(f"\n{'='*60}")
    print(f"Trait: {results['trait']}")
    print(f"Vector: layer {results['layer']}, {results['method']}, {results['component']}")
    print(f"Position: {results['position']}")
    print(f"Source: {results['source']} (score: {results['score']:.3f})")
    print(f"{'='*60}")

    print(f"\nToward (+):")
    for i, item in enumerate(results['toward'], 1):
        print(f"  {i:2d}. {item['token']:20s} {item['value']:+.3f}")

    print(f"\nAway (-):")
    for i, item in enumerate(results['away'], 1):
        print(f"  {i:2d}. {item['token']:20s} {item['value']:+.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--trait", help="Trait path (e.g., safety/refusal)")
    parser.add_argument("--all-traits", action="store_true", help="Analyze all traits")
    parser.add_argument("--top-k", type=int, default=20, help="Tokens to show (default: 20)")
    parser.add_argument("--no-norm", action="store_true", help="Skip RMSNorm before projection")
    parser.add_argument("--filter-common", action="store_true", help="Filter to common English tokens only")
    parser.add_argument("--max-vocab", type=int, default=10000, help="Max vocab index for common filter (default: 10000)")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    if not args.trait and not args.all_traits:
        parser.error("Must specify --trait or --all-traits")

    # Load model
    config = load_experiment_config(args.experiment)
    model_name = config.get('application_model') or config.get('extraction_model')
    if not model_name:
        raise ValueError(f"No model found in experiment config: {args.experiment}")

    model, tokenizer = load_model(model_name)
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
        traits = [args.trait]

    # Analyze each trait
    all_results = []
    for trait in traits:
        print(f"\nAnalyzing: {trait}")
        results = analyze_trait(
            experiment=args.experiment,
            trait=trait,
            model=model,
            tokenizer=tokenizer,
            top_k=args.top_k,
            apply_norm=not args.no_norm,
            common_mask=common_mask,
        )
        all_results.append(results)
        print_results(results)

    # Save if requested
    if args.save:
        output_dir = get_path('experiments.base', experiment=args.experiment) / "analysis" / "vector_logit_lens"
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.all_traits:
            output_path = output_dir / "all_traits.json"
        else:
            # Sanitize trait name for filename
            safe_trait = args.trait.replace("/", "_")
            output_path = output_dir / f"{safe_trait}.json"

        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
