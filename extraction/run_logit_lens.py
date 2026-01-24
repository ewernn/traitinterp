"""
Run logit lens interpretation for extracted trait vectors.

Input:
    - experiment, trait, model_variant: Standard identifiers
    - backend: GenerationBackend instance (LocalBackend or ServerBackend)

Output:
    - experiments/{experiment}/extraction/{trait}/{model_variant}/logit_lens.json

Usage:
    Called from run_pipeline.py after vector extraction (stage 5)
"""

import json
from typing import List, TYPE_CHECKING

from core.logit_lens import vector_to_vocab, build_common_token_mask, get_interpretation_layers
from utils.paths import get as get_path
from utils.vectors import load_vector_with_baseline

if TYPE_CHECKING:
    from core import GenerationBackend


def run_logit_lens_for_trait(
    experiment: str,
    trait: str,
    model_variant: str,
    backend: "GenerationBackend",
    methods: List[str],
    component: str = "residual",
    position: str = "response[:]",
    top_k: int = 10,
):
    """
    Run logit lens at 40% and 90% depth for all methods.

    Uses backend.model for embedding matrix access.
    Saves results to extraction/{trait}/{model_variant}/logit_lens.json
    """
    print(f"  Logit lens: {trait}")

    # Access model and tokenizer from backend (needed for embedding matrix)
    model = backend.model
    tokenizer = backend.tokenizer
    n_layers = backend.n_layers
    layers_info = get_interpretation_layers(n_layers)
    common_mask = build_common_token_mask(tokenizer)
    print(f"    {common_mask.sum().item()} common tokens")

    results = {
        "trait": trait,
        "component": component,
        "position": position,
        "n_layers": n_layers,
        "methods": {}
    }

    for method in methods:
        results["methods"][method] = {}
        for key, info in layers_info.items():
            layer = info["layer"]
            try:
                vector, _, _ = load_vector_with_baseline(
                    experiment, trait, method, layer, model_variant, component, position
                )
                decoded = vector_to_vocab(
                    vector, model, tokenizer,
                    top_k=top_k, common_mask=common_mask
                )
                results["methods"][method][key] = {
                    "layer": layer,
                    "pct": info["pct"],
                    **decoded
                }
            except FileNotFoundError:
                pass  # Vector doesn't exist at this layer

    # Save
    output_path = get_path('extraction.logit_lens', experiment=experiment, trait=trait, model_variant=model_variant)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"    Saved: {output_path.name}")
