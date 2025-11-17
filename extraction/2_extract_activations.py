#!/usr/bin/env python3
"""
Stage 2: Extract Activations

Captures model activations from all layers for all examples.
Saves in Storage B format (token-averaged per example).

Usage:
    # Single trait
    python pipeline/2_extract_activations.py --experiment my_exp --trait my_trait

    # Multiple traits
    python pipeline/2_extract_activations.py --experiment my_exp \
        --traits refusal,uncertainty,verbosity

    # Custom threshold
    python pipeline/2_extract_activations.py --experiment my_exp --trait my_trait \
        --threshold 60
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import json
from typing import List, Optional
from datetime import datetime
import fire

from extraction.utils_batch import get_activations_from_texts
from transformers import AutoModelForCausalLM, AutoTokenizer


def infer_model_from_experiment(experiment_name: str) -> str:
    """Infer model name from experiment naming convention."""
    if "gemma_2b" in experiment_name.lower():
        return "google/gemma-2-2b-it"
    elif "gemma_9b" in experiment_name.lower():
        return "google/gemma-2-9b-it"
    elif "llama_8b" in experiment_name.lower():
        return "meta-llama/Llama-3.1-8B-Instruct"
    else:
        # Default to Gemma 2B
        return "google/gemma-2-2b-it"


def extract_activations(
    experiment: str,
    trait: Optional[str] = None,
    traits: Optional[str] = None,
    model: Optional[str] = None,
    threshold: int = 50,
    device: str = "cuda",
    batch_size: int = 8,
):
    """
    Extract activations from all layers for all examples.

    Args:
        experiment: Experiment name
        trait: Single trait (mutually exclusive with traits)
        traits: Comma-separated traits (mutually exclusive with trait)
        model: Model name (auto-inferred from experiment if not provided)
        threshold: Score threshold for filtering (pos >= threshold, neg < threshold)
        device: Device to run on
        batch_size: Batch size for processing
    """
    # Parse trait list
    if trait and traits:
        raise ValueError("Specify either --trait or --traits, not both")

    if trait:
        trait_list = [trait]
    elif isinstance(traits, tuple):
        # Fire might parse comma-separated as tuple
        trait_list = list(traits)
    else:
        trait_list = traits.split(",")

    exp_dir = Path(f"experiments/{experiment}")
    if not exp_dir.exists():
        raise ValueError(f"Experiment not found: {exp_dir}")

    # Infer model if not provided
    if model is None:
        model = infer_model_from_experiment(experiment)
        print(f"Inferred model: {model}")

    print(f"Loading model: {model}")
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Get number of layers
    if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'layers'):
        n_layers = len(model_obj.model.layers)
    else:
        raise ValueError(f"Cannot determine number of layers for {model}")

    print(f"Model has {n_layers} layers")
    print(f"Threshold: pos >= {threshold}, neg < {threshold}")
    print()

    # Process each trait
    for trait_name in trait_list:
        trait_dir = exp_dir / trait_name
        if not trait_dir.exists():
            print(f"⚠️  Trait not found: {trait_name}, skipping")
            continue

        print(f"{'='*60}")
        print(f"Extracting activations: {trait_name}")
        print(f"{'='*60}")

        # Load responses
        pos_csv = trait_dir / "extraction" / "responses" / "pos.csv"
        neg_csv = trait_dir / "extraction" / "responses" / "neg.csv"

        if not pos_csv.exists() or not neg_csv.exists():
            print(f"❌ Responses not found")
            print(f"   Run: python pipeline/1_generate_responses.py --experiment {experiment} --trait {trait_name}")
            continue

        print(f"Loading responses...")
        pos_df = pd.read_csv(pos_csv)
        neg_df = pd.read_csv(neg_csv)

        # Filter by threshold
        if 'trait_score' in pos_df.columns:
            pos_df_filtered = pos_df[pos_df['trait_score'] >= threshold]
            neg_df_filtered = neg_df[neg_df['trait_score'] < threshold]

            print(f"  Pos: {len(pos_df)} → {len(pos_df_filtered)} after filtering (>= {threshold})")
            print(f"  Neg: {len(neg_df)} → {len(neg_df_filtered)} after filtering (< {threshold})")
        else:
            print(f"  Warning: No trait_score column, using all examples")
            pos_df_filtered = pos_df
            neg_df_filtered = neg_df
            print(f"  Pos: {len(pos_df)}")
            print(f"  Neg: {len(neg_df)}")

        # Extract texts (filter out null/NaN responses)
        pos_texts = pos_df_filtered['response'].dropna().astype(str).tolist()
        neg_texts = neg_df_filtered['response'].dropna().astype(str).tolist()

        if len(pos_texts) == 0 or len(neg_texts) == 0:
            print(f"❌ No examples after filtering, skipping")
            continue

        # Capture activations from all layers
        print(f"\nCapturing activations from all {n_layers} layers...")
        # CRITICAL: hidden_states[0] is embedding, not layer 0!
        # hidden_states[1] = layer 0 output, hidden_states[2] = layer 1 output, etc.
        # To extract layers 0 to n_layers-1, use indices 1 to n_layers
        layers = list(range(1, n_layers + 1))  # [1, 2, ..., n_layers] for layers 0 to n_layers-1

        print(f"  Processing {len(pos_texts)} pos examples...")
        pos_acts_dict = get_activations_from_texts(
            model_obj, tokenizer, pos_texts, layers, batch_size, device
        )

        print(f"  Processing {len(neg_texts)} neg examples...")
        neg_acts_dict = get_activations_from_texts(
            model_obj, tokenizer, neg_texts, layers, batch_size, device
        )

        # Stack into single tensor: [n_examples, n_layers, hidden_dim]
        pos_acts = torch.stack([pos_acts_dict[layer] for layer in layers], dim=1)
        neg_acts = torch.stack([neg_acts_dict[layer] for layer in layers], dim=1)

        # Concatenate pos and neg
        all_acts = torch.cat([pos_acts, neg_acts], dim=0)

        print(f"\n✓ Captured activations: {all_acts.shape}")
        print(f"  Shape: [n_examples={all_acts.shape[0]}, n_layers={all_acts.shape[1]}, hidden_dim={all_acts.shape[2]}]")

        # Calculate size
        size_mb = all_acts.element_size() * all_acts.nelement() / (1024 ** 2)
        print(f"  Size: {size_mb:.1f} MB")

        # Save
        acts_dir = trait_dir / "extraction" / "activations"
        acts_dir.mkdir(parents=True, exist_ok=True)

        acts_path = acts_dir / "all_layers.pt"
        torch.save(all_acts, acts_path)
        print(f"\n💾 Saved: {acts_path}")

        # Save metadata
        metadata = {
            "model": model,
            "trait": trait_name,
            "n_examples": all_acts.shape[0],
            "n_examples_pos": len(pos_texts),
            "n_examples_neg": len(neg_texts),
            "n_layers": n_layers,
            "hidden_dim": all_acts.shape[2],
            "threshold": threshold,
            "storage_type": "token_averaged",
            "extraction_date": datetime.now().isoformat(),
            "size_mb": size_mb,
        }

        metadata_path = acts_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Saved metadata: {metadata_path}")
        print()


if __name__ == "__main__":
    fire.Fire(extract_activations)
