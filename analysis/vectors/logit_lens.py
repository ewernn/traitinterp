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

Runs as Stage 5 of the extraction pipeline (uses the loaded model — free).
Also runnable standalone for ad-hoc re-runs after retraining vectors.
Writes per-trait files to the canonical extraction.logit_lens path
(experiments/{experiment}/extraction/{trait}/{model_variant}/logit_lens.json).

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
from core.architectures import inner_model
from utils.model import load_model_with_lora
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
    layer_range: Optional[tuple] = None,
) -> Optional[dict]:
    """Analyze all methods for a trait, producing the canonical logit_lens schema.

    If layer_range is None: pick a single layer near 90% depth per method (legacy behavior).
    If layer_range is (lo, hi): run logit lens for every layer in [lo, hi] per method.

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

    n_layers = inner_model(model).config.num_hidden_layers
    # Empirical heuristic: gives L26 on 32-layer Qwen3.5-9B, L50 on 80-layer Llama 70B.
    # Smaller models pack more transformation per layer, landing the readout later in
    # fractional depth; this scales accordingly.
    target = n_layers // 2 + 10

    # Group by method
    by_method = defaultdict(list)
    for c in filtered:
        by_method[c["method"]].append(c["layer"])

    methods = {}
    for method, available_layers in by_method.items():
        if layer_range is not None:
            lo, hi = layer_range
            chosen_layers = sorted(L for L in available_layers if lo <= L <= hi)
        else:
            # Pick layer closest to 90% depth; ties prefer higher layer
            chosen_layers = [min(available_layers, key=lambda L: (abs(L - target), -L))]

        per_layer = {}
        for chosen_layer in chosen_layers:
            try:
                vector, _baseline, _meta = load_vector_with_baseline(
                    experiment, trait, method, chosen_layer, model_variant, component, position
                )
            except FileNotFoundError:
                continue

            vocab = vector_to_vocab(vector, model, tokenizer, top_k, apply_norm, common_mask)
            pct = round(chosen_layer / max(n_layers - 1, 1) * 100)
            per_layer[chosen_layer] = {
                "layer": chosen_layer,
                "pct": pct,
                "toward": vocab["toward"],
                "away": vocab["away"],
            }

        if not per_layer:
            continue

        if layer_range is not None:
            methods[method] = {"per_layer": per_layer}
        else:
            # Legacy schema: {"late": {...}} with single layer
            only_layer = next(iter(per_layer.values()))
            methods[method] = {"late": only_layer}

    if not methods:
        return None

    return {
        "trait": trait,
        "component": component,
        "position": position,
        "n_layers": n_layers,
        "methods": methods,
    }


def _print_layer_block(layer_data: dict):
    print(f"\n  Toward (+):")
    for i, item in enumerate(layer_data['toward'], 1):
        print(f"    {i:2d}. {item['token']:20s} {item['value']:+.3f}")
    print(f"\n  Away (-):")
    for i, item in enumerate(layer_data['away'], 1):
        print(f"    {i:2d}. {item['token']:20s} {item['value']:+.3f}")


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
        if 'per_layer' in data:
            for layer_num in sorted(data['per_layer'].keys()):
                ld = data['per_layer'][layer_num]
                print(f"\n  Method: {method} (layer {ld['layer']}, {ld['pct']}% depth)")
                _print_layer_block(ld)
        else:
            late = data['late']
            print(f"\n  Method: {method} (layer {late['layer']}, {late['pct']}% depth)")
            _print_layer_block(late)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--model-variant", default=None, help="Model variant (default: from experiment config)")
    parser.add_argument("--traits", help="Trait path (e.g., safety/refusal)")
    parser.add_argument("--all-traits", action="store_true", help="Analyze all traits")
    parser.add_argument("--top-k", type=int, default=20, help="Tokens to show (default: 20)")
    parser.add_argument("--no-norm", action="store_true", help="Skip RMSNorm before projection")
    parser.add_argument("--no-filter-common", action="store_true", help="Disable common-token filter (default: enabled)")
    parser.add_argument("--max-vocab", type=int, default=10000, help="Max vocab index for common filter (default: 10000)")
    parser.add_argument("--layer-range", default=None, help="Sweep layer range, e.g. '16-32'. If set, output per-layer instead of single 90%-depth pick.")
    parser.add_argument("--save", action="store_true", help="Save results to canonical per-trait JSON")
    args = parser.parse_args()

    filter_common = not args.no_filter_common
    layer_range = None
    if args.layer_range:
        lo_s, hi_s = args.layer_range.split("-")
        layer_range = (int(lo_s), int(hi_s))

    if not args.traits and not args.all_traits:
        parser.error("Must specify --traits or --all-traits")

    # Resolve model variant — use extraction mode so vectors + write path align
    variant = get_model_variant(args.experiment, args.model_variant, mode="extraction")
    model_variant = variant.name
    model_name = variant.model
    lora = variant.lora

    model, tokenizer = load_model_with_lora(model_name, lora_adapter=lora)
    model.eval()

    # Build common token mask (on by default; disable with --no-filter-common)
    common_mask = None
    if filter_common:
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
            layer_range=layer_range,
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
