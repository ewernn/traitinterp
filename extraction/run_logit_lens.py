"""
Run logit lens interpretation for extracted trait vectors.

Input: Experiment, trait, model, methods
Output: experiments/{experiment}/extraction/{trait}/{model_variant}/logit_lens.json

Usage:
    Called from run_pipeline.py after vector extraction (stage 5)
"""

import json
from typing import List

from core.logit_lens import vector_to_vocab, build_common_token_mask, get_interpretation_layers
from utils.model import get_num_layers
from utils.paths import get as get_path
from utils.vectors import load_vector_with_baseline


def run_logit_lens_for_trait(
    experiment: str,
    trait: str,
    model_variant: str,
    model,
    tokenizer,
    methods: List[str],
    component: str = "residual",
    position: str = "response[:]",
    top_k: int = 10,
):
    """
    Run logit lens at 40% and 90% depth for all methods.

    Saves results to extraction/{trait}/{model_variant}/logit_lens.json
    """
    print(f"  Logit lens: {trait}")

    n_layers = get_num_layers(model)
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
