#!/usr/bin/env python3
"""
Local uncertainty dynamics test - runs on CPU/GPU with cached model.

Captures per-token activations across all layers for 5 test prompts,
projects onto uncertainty vector, and analyzes dynamics.

Usage:
    python local_dynamics_test.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

# Import traitlens
from traitlens import HookManager, ActivationCapture


def main():
    print("=" * 60)
    print("LOCAL UNCERTAINTY DYNAMICS TEST")
    print("=" * 60)
    print()

    # Configuration
    model_name = "google/gemma-2-2b-it"
    experiment = "gemma_2b_cognitive_nov20"
    trait = "uncertainty_calibration"

    # Load uncertainty vector
    vector_path = Path(f"experiments/{experiment}/{trait}/vectors/probe_layer16.pt")
    print(f"Loading uncertainty vector from {vector_path}...")
    uncertainty_vector = torch.load(vector_path)
    print(f"Vector shape: {uncertainty_vector.shape}")
    print(f"Vector norm: {uncertainty_vector.norm():.2f}")
    print()

    # Load model (from local cache)
    print(f"Loading model: {model_name}")
    print("(Using local cache at ~/.cache/huggingface/hub/)")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for MPS compatibility
        low_cpu_mem_usage=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get model info
    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    print(f"Model: {n_layers} transformer layers, hidden_dim={hidden_dim}")
    print()

    # Test prompts (same as Modal version)
    prompts = {
        "gradual_increase": "What is 2+2? What is 16 times 16? What is the meaning of life? What is the purpose of existence? What lies beyond the universe?",

        "gradual_decrease": "Something exists somewhere in spacetime. It's on Earth. It's in France. It's in Paris. It's a tower. It's the Eiffel Tower.",

        "linear_difficulty": "How do you boil water? How does a microwave work? How does quantum tunneling work? How does consciousness emerge? What existed before the Big Bang?",

        "sudden_increase": "Paris is the capital of France, established in 987 AD. The population is 2.1 million. But will it still be the capital in 2525? Could alien contact change everything? Maybe reality itself is...",

        "sudden_decrease": "Nobody knows what consciousness truly is. Philosophers have debated for millennia. Scientists remain puzzled. But we do know one thing for certain: water freezes at 0°C at standard pressure.",
    }

    # Results storage
    all_results = {}
    all_activations = {}

    # Process each prompt
    for prompt_name, prompt_text in prompts.items():
        print(f"\n{'=' * 60}")
        print(f"Processing: {prompt_name}")
        print(f"{'=' * 60}")
        print(f"Prompt: {prompt_text[:100]}...")
        print()

        # Tokenize
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        n_prompt_tokens = inputs['input_ids'].shape[1]
        print(f"Prompt tokens: {n_prompt_tokens}")

        # Setup activation capture for all layers
        capture = ActivationCapture()
        layer_names = [f"layer_{i}" for i in range(n_layers)]

        with HookManager(model) as hooks:
            # Add hooks for all transformer layers
            for i in range(n_layers):
                hooks.add_forward_hook(
                    f"model.layers.{i}",
                    capture.make_hook(f"layer_{i}")
                )

            # Generate with hooks active
            print("Generating response...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )

        # Get generated tokens
        generated_ids = outputs[0]
        n_total_tokens = generated_ids.shape[0]
        n_generated_tokens = n_total_tokens - n_prompt_tokens

        print(f"Generated tokens: {n_generated_tokens}")
        print(f"Total tokens: {n_total_tokens}")

        # Decode response
        full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        response_text = tokenizer.decode(
            generated_ids[n_prompt_tokens:],
            skip_special_tokens=True
        )

        print(f"Response: {response_text[:100]}...")
        print()

        # Get token strings
        tokens = [tokenizer.decode([tid]) for tid in generated_ids]

        # Extract activations from all layers
        print("Processing activations...")
        layer_activations = []

        for layer_idx in range(n_layers):
            layer_name = f"layer_{layer_idx}"

            # Get activations as list (not concatenated)
            acts_list = capture.get(layer_name, concat=False)

            if acts_list:
                # Concatenate along sequence dimension
                # Each item in list: [batch=1, seq_len_i, hidden_dim]
                # First remove batch dim, then concat
                acts_squeezed = [a.squeeze(0) for a in acts_list]  # List of [seq_len_i, hidden_dim]
                acts = torch.cat(acts_squeezed, dim=0)  # [total_seq_len, hidden_dim]
                layer_activations.append(acts.cpu())
            else:
                print(f"Warning: No activations for {layer_name}")
                layer_activations.append(torch.zeros(n_total_tokens, hidden_dim))

        # Stack: [n_layers, n_tokens, hidden_dim]
        stacked_acts = torch.stack(layer_activations, dim=0)
        print(f"Stacked activations shape: {stacked_acts.shape}")

        # Project onto uncertainty vector (use layer 16)
        layer_16_acts = stacked_acts[16]  # [n_tokens, hidden_dim]
        uncertainty_scores = layer_16_acts.float() @ uncertainty_vector.cpu().float()  # [n_tokens]

        print(f"Uncertainty scores: min={uncertainty_scores.min():.2f}, "
              f"max={uncertainty_scores.max():.2f}, "
              f"mean={uncertainty_scores.mean():.2f}")

        # Compute dynamics metrics
        scores_np = uncertainty_scores.numpy()

        # Velocity (1st derivative)
        velocity = []
        for i in range(1, len(scores_np)):
            velocity.append(scores_np[i] - scores_np[i-1])

        # Acceleration (2nd derivative)
        acceleration = []
        for i in range(1, len(velocity)):
            acceleration.append(velocity[i] - velocity[i-1])

        # Find commitment point (where acceleration stabilizes)
        commitment_point = None
        for i, acc in enumerate(acceleration):
            if abs(acc) < 0.1:  # Threshold for "stable"
                commitment_point = i + 2  # +2 because acceleration is 2nd derivative
                break

        # Find peak
        peak_idx = int(scores_np.argmax())
        peak_value = float(scores_np.max())

        # Persistence (tokens above threshold after peak)
        threshold = 0.5
        if peak_idx < len(scores_np):
            persistence = sum(1 for s in scores_np[peak_idx:] if abs(s) > threshold)
        else:
            persistence = 0

        dynamics = {
            "commitment_point": int(commitment_point) if commitment_point is not None else None,
            "peak_token": int(peak_idx),
            "peak_value": float(peak_value),
            "peak_velocity": float(max(velocity, key=abs)) if velocity else 0.0,
            "avg_velocity": float(sum(abs(v) for v in velocity) / len(velocity)) if velocity else 0.0,
            "persistence": int(persistence),
        }

        print(f"Dynamics: commitment={commitment_point}, peak={peak_idx}, "
              f"persistence={persistence}")

        # Store results (convert all numpy types to Python native types)
        all_results[prompt_name] = {
            "prompt": prompt_text,
            "response": response_text,
            "full_text": full_text,
            "n_prompt_tokens": int(n_prompt_tokens),
            "n_generated_tokens": int(n_generated_tokens),
            "n_total_tokens": int(n_total_tokens),
            "tokens": tokens,
            "uncertainty_scores": [float(x) for x in uncertainty_scores.tolist()],
            "dynamics": dynamics,
            "velocity": [float(x) for x in velocity],
            "acceleration": [float(x) for x in acceleration],
        }

        # Store activations (all layers)
        all_activations[prompt_name] = stacked_acts  # [n_layers, n_tokens, hidden_dim]

    # Save results
    output_dir = Path("experiments") / experiment / trait / "dynamics_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    results_path = output_dir / "results.json"
    print(f"\n{'=' * 60}")
    print(f"Saving results to {results_path}...")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save activations
    acts_path = output_dir / "all_activations.pt"
    print(f"Saving activations to {acts_path}...")
    torch.save(all_activations, acts_path)

    # Calculate storage
    acts_size_mb = acts_path.stat().st_size / 1024 / 1024
    results_size_kb = results_path.stat().st_size / 1024

    print(f"\n{'=' * 60}")
    print("✅ COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results: {results_path} ({results_size_kb:.1f} KB)")
    print(f"Activations: {acts_path} ({acts_size_mb:.1f} MB)")
    print()
    print("Summary of dynamics:")
    for name, result in all_results.items():
        dyn = result["dynamics"]
        print(f"  {name}:")
        print(f"    Commitment: token {dyn['commitment_point']}")
        print(f"    Peak: token {dyn['peak_token']} (score={dyn['peak_value']:.2f})")
        print(f"    Persistence: {dyn['persistence']} tokens")
    print()


if __name__ == "__main__":
    main()
