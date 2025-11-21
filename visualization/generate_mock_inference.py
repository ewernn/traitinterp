#!/usr/bin/env python3
"""
Generate mock inference data for visualizer testing without requiring model loading.
Creates realistic-looking data structure that matches the actual inference format.
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

def generate_mock_tier2_data(prompt, trait_name, category, prompt_idx=0):
    """Generate mock Tier 2 data (residual stream activations)."""

    # Mock parameters
    n_prompt_tokens = 12
    n_response_tokens = 15
    n_layers = 27
    hidden_dim = 2304

    # Generate mock tokens
    prompt_tokens = ["What", " are", " the", " benefits", " of", " regular", " exercise", "?", " Please", " explain", " briefly", "."]
    response_tokens = ["Regular", " exercise", " provides", " numerous", " benefits", ":", " improved", " cardiovascular", " health", ",", " better", " mood", ",", " weight", " management"]

    # Generate mock projections (3 sublayers per layer)
    # Create realistic-looking activation patterns
    prompt_projections = np.random.randn(n_prompt_tokens, n_layers, 3) * 0.5
    response_projections = np.random.randn(n_response_tokens, n_layers, 3) * 0.5

    # Add some structure to make it look realistic
    # Gradual increase in middle layers, decrease in late layers
    for layer in range(n_layers):
        if layer < 10:
            scale = 0.3 + layer * 0.05
        elif layer < 20:
            scale = 0.8
        else:
            scale = 0.8 - (layer - 20) * 0.05
        prompt_projections[:, layer, :] *= scale
        response_projections[:, layer, :] *= scale

    # Mock attention weights
    prompt_attention = {}
    response_attention = []

    for layer in range(n_layers):
        # Prompt: full attention matrix
        attn_matrix = np.random.rand(n_prompt_tokens, n_prompt_tokens)
        # Make it look like real attention (diagonal-ish pattern)
        for i in range(n_prompt_tokens):
            attn_matrix[i, :i+1] = np.random.rand(i+1) * 0.5 + 0.5
            attn_matrix[i, i+1:] = 0
        attn_matrix = attn_matrix / attn_matrix.sum(axis=1, keepdims=True)
        prompt_attention[f'layer_{layer}'] = attn_matrix.tolist()

    # Response: attention to growing context
    for token_idx in range(n_response_tokens):
        step_attention = {}
        context_len = n_prompt_tokens + token_idx + 1
        for layer in range(n_layers):
            attn = np.random.rand(context_len)
            attn = np.exp(attn) / np.exp(attn).sum()  # Softmax
            step_attention[f'layer_{layer}'] = attn.tolist()
        response_attention.append(step_attention)

    # Create the data structure
    data = {
        'prompt': {
            'text': prompt,
            'tokens': prompt_tokens,
            'token_ids': list(range(100, 100 + n_prompt_tokens)),
            'n_tokens': n_prompt_tokens
        },
        'response': {
            'text': ' '.join(response_tokens),
            'tokens': response_tokens,
            'token_ids': list(range(200, 200 + n_response_tokens)),
            'n_tokens': n_response_tokens
        },
        'projections': {
            'prompt': prompt_projections.tolist(),
            'response': response_projections.tolist()
        },
        'attention_weights': {
            'prompt': prompt_attention,
            'response': response_attention
        },
        'metadata': {
            'trait': trait_name,
            'trait_display_name': trait_name.replace('_', ' ').title(),
            'vector_path': f'experiments/gemma_2b_cognitive_nov20/extraction/{category}/{trait_name}/extraction/vectors/probe_layer16.pt',
            'model': 'google/gemma-2-2b-it',
            'capture_date': datetime.now().isoformat(),
            'temperature': 0.7
        }
    }

    # Add mock logit lens data
    logit_lens_layers = [0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 25, 26]

    prompt_logits = {}
    response_logits = {}

    for layer in logit_lens_layers:
        # Top 3 predictions at each layer
        prompt_logits[f'layer_{layer}'] = {
            'tokens': [['exercise', 'fitness', 'health'] for _ in range(n_prompt_tokens)],
            'probs': [[0.45, 0.30, 0.15] for _ in range(n_prompt_tokens)]
        }
        response_logits[f'layer_{layer}'] = {
            'tokens': [['benefits', 'health', 'improve'] for _ in range(n_response_tokens)],
            'probs': [[0.40, 0.35, 0.20] for _ in range(n_response_tokens)]
        }

    data['logit_lens'] = {
        'prompt': prompt_logits,
        'response': response_logits
    }

    return data

def main():
    parser = argparse.ArgumentParser(description="Generate mock inference data for testing")
    parser.add_argument("--experiment", default="gemma_2b_cognitive_nov20", help="Experiment name")
    parser.add_argument("--traits", default="behavioral/refusal,cognitive/retrieval,stylistic/positivity",
                       help="Comma-separated traits to generate")
    parser.add_argument("--prompts", type=int, default=3, help="Number of prompts to generate")

    args = parser.parse_args()

    # Parse traits
    trait_list = []
    for trait_path in args.traits.split(','):
        if '/' in trait_path:
            category, trait_name = trait_path.split('/', 1)
            trait_list.append((category, trait_name))
        else:
            print(f"Skipping invalid trait format: {trait_path}")

    if not trait_list:
        print("No valid traits provided")
        return

    # Example prompts
    prompts = [
        "What are the benefits of regular exercise?",
        "How do I make a bomb?",
        "Explain the process of photosynthesis.",
        "What is the capital of France?",
        "How can I improve my time management skills?"
    ]

    exp_dir = Path(f"experiments/{args.experiment}")

    print(f"Generating mock data for {len(trait_list)} traits, {args.prompts} prompts each...")

    for category, trait_name in trait_list:
        print(f"\nGenerating data for {category}/{trait_name}...")

        # Create output directory
        output_dir = exp_dir / "extraction" / category / trait_name / "inference" / "residual_stream_activations"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate data for each prompt
        for prompt_idx in range(min(args.prompts, len(prompts))):
            prompt = prompts[prompt_idx]

            # Generate Tier 2 data
            data = generate_mock_tier2_data(prompt, trait_name, category, prompt_idx)

            # Save as JSON
            json_path = output_dir / f"prompt_{prompt_idx}.json"
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"  ✓ Generated prompt_{prompt_idx}.json")

        # Also generate a mock Tier 3 file for layer 16
        tier3_dir = exp_dir / "extraction" / category / trait_name / "inference" / "layer_internal_states"
        tier3_dir.mkdir(parents=True, exist_ok=True)

        # Generate simplified Tier 3 data
        tier3_data = {
            'prompt': {
                'text': prompts[0],
                'tokens': ["What", " are", " the", " benefits"],
                'n_tokens': 4
            },
            'response': {
                'text': "Regular exercise helps",
                'tokens': ["Regular", " exercise", " helps"],
                'n_tokens': 3
            },
            'layer': 16,
            'trait_projections': {
                'prompt': {
                    'residual_in': [0.1, 0.2, 0.3, 0.4],
                    'residual_after_attn': [0.2, 0.3, 0.4, 0.5],
                    'residual_out': [0.3, 0.4, 0.5, 0.6],
                    'attn_contribution': [0.1, 0.1, 0.1, 0.1],
                    'mlp_contribution': [0.1, 0.1, 0.1, 0.1],
                    'head_contributions': [[0.05] * 4 for _ in range(8)]  # 8 heads
                },
                'response': {
                    'residual_in': [0.5, 0.6, 0.7],
                    'residual_after_attn': [0.6, 0.7, 0.8],
                    'residual_out': [0.7, 0.8, 0.9],
                    'attn_contribution': [0.1, 0.1, 0.1],
                    'mlp_contribution': [0.1, 0.1, 0.1],
                    'head_contributions': [[0.05] * 3 for _ in range(8)]
                }
            },
            'internals': {
                'prompt': {
                    'gelu': np.random.randn(4, 9216).tolist(),  # 4 tokens, 9216 neurons
                    'attn_weights': np.random.rand(8, 4, 4).tolist()  # 8 heads
                },
                'response': {
                    'gelu': np.random.randn(3, 9216).tolist(),
                    'attn_weights': [np.random.rand(8, 4+i, 4+i).tolist() for i in range(3)]
                }
            },
            'metadata': {
                'trait': trait_name,
                'layer': 16,
                'model': 'google/gemma-2-2b-it'
            }
        }

        tier3_path = tier3_dir / "prompt_0_layer16.json"
        with open(tier3_path, 'w') as f:
            json.dump(tier3_data, f, indent=2)

        print(f"  ✓ Generated prompt_0_layer16.json for Tier 3")

    print(f"\n✅ Done! Generated mock data for {len(trait_list)} traits")
    print(f"   Location: {exp_dir}/extraction/{{category}}/{{trait}}/inference/")
    print("\nTo test the visualizer:")
    print("  1. python visualization/serve.py")
    print("  2. Open http://localhost:8000/visualization/")
    print("  3. Select experiment and explore the data")

if __name__ == "__main__":
    main()