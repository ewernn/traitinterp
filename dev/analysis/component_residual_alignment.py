"""
Compute component-to-residual directional alignment across layers.

For each component (attn_contribution, mlp_contribution), compute cosine similarity
of its direction vector against the best residual vector. Also computes consecutive
(L_i vs L_{i+1}) and skip-1 (L_i vs L_{i+2}) self-similarity per component.

Input: Extracted vectors at experiments/{experiment}/extraction/{trait}/{model_variant}/vectors/
Output: JSON compatible with model-diff-cosine chart type

Usage:
    python analysis/vectors/component_residual_alignment.py \
        --experiment gemma-2-2b --trait chirp/refusal --reference best

    python analysis/vectors/component_residual_alignment.py \
        --experiment gemma-2-2b --trait chirp/refusal \
        --reference residual/probe/L15 --output output.json
"""

import argparse
import json
import sys

import torch

from core.math import cosine_similarity
from utils.paths import (
    get_model_variant,
    list_components,
    list_layers,
    list_methods,
)
from utils.vector_selection import select_vector
from utils.vectors import load_vector


# Components with different hidden dim (can't compare to residual)
SKIP_COMPONENTS = {"k_proj", "v_proj"}


def parse_reference(ref_str: str, experiment: str, trait: str, model_variant: str, position: str) -> dict:
    """
    Parse reference spec. Either 'best' (auto-select via steering) or
    '{component}/{method}/L{N}' (manual).

    Returns dict with keys: component, method, layer.
    """
    if ref_str == "best":
        best = select_vector(
            experiment, trait,
            extraction_variant=model_variant,
            component="residual",
        )
        return {
            "component": "residual",
            "method": best.method,
            "layer": best.layer,
        }

    # Parse manual: residual/probe/L15
    parts = ref_str.split("/")
    if len(parts) != 3 or not parts[2].startswith("L"):
        raise ValueError(f"Invalid reference format: '{ref_str}'. Expected 'best' or 'component/method/LN'")

    return {
        "component": parts[0],
        "method": parts[1],
        "layer": int(parts[2][1:]),
    }


def compute_alignment(
    experiment: str,
    trait: str,
    model_variant: str,
    ref_vector: torch.Tensor,
    component: str,
    method: str,
    position: str,
) -> dict:
    """Compute per-layer cosine similarity of component vectors vs reference."""
    layers = list_layers(experiment, trait, method, model_variant, component, position)
    if not layers:
        return None

    sims = []
    for layer in layers:
        vec = load_vector(experiment, trait, layer, model_variant, method, component, position)
        if vec is None:
            continue
        sim = cosine_similarity(vec.float(), ref_vector.float()).item()
        sims.append(sim)

    return {"layers": layers, "per_layer_cosine_sim": sims}


def compute_consecutive(
    experiment: str,
    trait: str,
    model_variant: str,
    component: str,
    method: str,
    position: str,
    skip: int = 1,
) -> dict:
    """Compute cosine similarity between vectors at layer i and layer i+skip."""
    layers = list_layers(experiment, trait, method, model_variant, component, position)
    if len(layers) < skip + 1:
        return None

    vectors = {}
    for layer in layers:
        vec = load_vector(experiment, trait, layer, model_variant, method, component, position)
        if vec is not None:
            vectors[layer] = vec.float()

    result_layers = []
    sims = []
    for i in range(len(layers) - skip):
        l_a, l_b = layers[i], layers[i + skip]
        if l_a in vectors and l_b in vectors:
            sim = cosine_similarity(vectors[l_a], vectors[l_b]).item()
            result_layers.append(l_a)
            sims.append(sim)

    return {"layers": result_layers, "per_layer_cosine_sim": sims}


def main():
    parser = argparse.ArgumentParser(description="Component-to-residual directional alignment")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--trait", required=True, help="Trait path (e.g., chirp/refusal)")
    parser.add_argument("--model-variant", default=None, help="Extraction variant (default: from config)")
    parser.add_argument("--reference", default="best", help="'best' or 'component/method/LN' (e.g., residual/probe/L15)")
    parser.add_argument("--methods", nargs="+", default=["probe", "mean_diff"], help="Methods to analyze")
    parser.add_argument("--position", default="response[:5]", help="Position string")
    parser.add_argument("--output", default=None, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    # Resolve model variant
    variant = get_model_variant(args.experiment, args.model_variant, mode="extraction")
    model_variant = variant.name

    # Parse reference
    ref = parse_reference(args.reference, args.experiment, args.trait, model_variant, args.position)
    print(f"Reference: {ref['component']}/{ref['method']}/L{ref['layer']}", file=sys.stderr)

    # Load reference vector
    ref_vector = load_vector(
        args.experiment, args.trait, ref["layer"], model_variant,
        ref["method"], ref["component"], args.position,
    )
    if ref_vector is None:
        print(f"ERROR: Reference vector not found", file=sys.stderr)
        sys.exit(1)
    ref_vector = ref_vector.float()

    # Discover components (skip different-dimensionality ones)
    components = list_components(args.experiment, args.trait, model_variant, args.position)
    components = [c for c in components if c not in SKIP_COMPONENTS]
    print(f"Components: {components}", file=sys.stderr)

    # === Alignment: each component vs reference ===
    alignment_traits = {}
    for component in components:
        if component == ref["component"]:
            continue  # Skip self-comparison
        for method in args.methods:
            available = list_methods(args.experiment, args.trait, model_variant, component, args.position)
            if method not in available:
                continue
            result = compute_alignment(
                args.experiment, args.trait, model_variant,
                ref_vector, component, method, args.position,
            )
            if result:
                label = f"{component}" if len(args.methods) == 1 else f"{component}/{method}"
                alignment_traits[label] = result
                peak_idx = max(range(len(result["per_layer_cosine_sim"])),
                              key=lambda i: abs(result["per_layer_cosine_sim"][i]))
                print(f"  {label}: peak {result['per_layer_cosine_sim'][peak_idx]:.3f} @ L{result['layers'][peak_idx]}",
                      file=sys.stderr)

    # === Consecutive similarity (L_i vs L_{i+1}) ===
    consecutive_traits = {}
    for component in components:
        for method in args.methods:
            available = list_methods(args.experiment, args.trait, model_variant, component, args.position)
            if method not in available:
                continue
            result = compute_consecutive(
                args.experiment, args.trait, model_variant,
                component, method, args.position, skip=1,
            )
            if result:
                label = f"{component}" if len(args.methods) == 1 else f"{component}/{method}"
                consecutive_traits[label] = result

    # === Skip-1 similarity (L_i vs L_{i+2}) ===
    skip1_traits = {}
    for component in components:
        if component == "residual":
            continue  # Residual skip-1 is less interesting
        for method in args.methods:
            available = list_methods(args.experiment, args.trait, model_variant, component, args.position)
            if method not in available:
                continue
            result = compute_consecutive(
                args.experiment, args.trait, model_variant,
                component, method, args.position, skip=2,
            )
            if result:
                label = f"{component}" if len(args.methods) == 1 else f"{component}/{method}"
                skip1_traits[label] = result

    # Build output
    output = {
        "reference": {
            "component": ref["component"],
            "method": ref["method"],
            "layer": ref["layer"],
        },
        "alignment": {"traits": alignment_traits},
        "consecutive": {"traits": consecutive_traits},
        "skip1": {"traits": skip1_traits},
    }

    json_str = json.dumps(output, indent=2)

    if args.output:
        from pathlib import Path
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(json_str)
        print(f"\nSaved to {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
