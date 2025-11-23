#!/usr/bin/env python3
"""
Command-line wrapper for Stage 2: Extract activations from generated responses.
"""

import sys
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the refactored core logic
from extraction.pipeline.extract_activations import extract_activations_for_trait

# Define the model name here or get it from an argument
MODEL_NAME = "google/gemma-2-2b-it"

def main():
    """Main function to handle argument parsing and model loading."""
    parser = argparse.ArgumentParser(description='Extract activations from responses.')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--trait', type=str, required=True, help='Trait name (e.g., "my_trait_name")')
    
    args = parser.parse_args()

    print("=" * 80)
    print("EXTRACTING ACTIVATIONS (CLI WRAPPER)")
    print(f"Experiment: {args.experiment}")
    print(f"Trait: {args.trait}")
    print(f"Model: {MODEL_NAME}")
    print("=" * 80)

    # Load model and tokenizer
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model.eval()
    print("Model and tokenizer loaded.")

    # Call the core logic
    n_layers = extract_activations_for_trait(
        experiment=args.experiment,
        trait=args.trait,
        model=model,
        tokenizer=tokenizer
    )

    print(f"\nSUCCESS: Extracted activations from {n_layers} layers for '{args.trait}'.")

if __name__ == '__main__':
    main()

