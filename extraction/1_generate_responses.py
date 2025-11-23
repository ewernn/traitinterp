#!/usr/bin/env python3
"""
Command-line wrapper for Stage 1: Generate responses for natural elicitation.
"""

import sys
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the refactored core logic
from extraction.pipeline.generate_responses import generate_responses_for_trait

def main():
    """Main function to handle argument parsing and model loading."""
    parser = argparse.ArgumentParser(description='Generate natural elicitation responses.')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--trait', type=str, required=True, help='Trait name (e.g., "my_trait_name")')
    parser.add_argument('--model', type=str, default='google/gemma-2-2b-it', help='Model name')
    parser.add_argument('--max-new-tokens', type=int, default=200, help='Max tokens to generate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for generation')
    parser.add_argument('--device', type=str, default='auto', help='Device placement (auto, cuda, cpu, mps)')
    
    args = parser.parse_args()

    print("=" * 80)
    print("NATURAL ELICITATION GENERATION (CLI WRAPPER)")
    print(f"Experiment: {args.experiment}")
    print(f"Trait: {args.trait}")
    print(f"Model: {args.model}")
    print("=" * 80)

    # Load model and tokenizer
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    model.eval()
    print("Model and tokenizer loaded.")

    # Call the core logic
    n_pos, n_neg = generate_responses_for_trait(
        experiment=args.experiment,
        trait=args.trait,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size
    )

    print(f"\nSUCCESS: Generated {n_pos} positive + {n_neg} negative = {n_pos + n_neg} total responses for '{args.trait}'.")

if __name__ == '__main__':
    main()
