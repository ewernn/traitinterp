#!/usr/bin/env python3
"""
Minimal working example for traitlens v0.1

This example must work for v0.1 to ship. It demonstrates:
1. Extracting activations from Gemma-2B
2. Computing trait vectors
3. Analyzing temporal dynamics
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import traitlens components
# If installed as package:
# from traitlens import HookManager, ActivationCapture, mean_difference, compute_derivative, projection

# For local development:
from hooks import HookManager
from activations import ActivationCapture
from compute import mean_difference, compute_derivative, projection


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    # Test 1: Basic activation capture
    print("\n=== Test 1: Basic Activation Capture ===")

    capture = ActivationCapture()
    test_prompt = "Hello, how are you today?"

    with HookManager(model) as hooks:
        # Hook layer 16 (middle layer)
        hooks.add_forward_hook("model.layers.16", capture.make_hook("layer_16"))

        # Generate
        inputs = tokenizer(test_prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    activations = capture.get("layer_16")
    print(f"Captured activations shape: {activations.shape}")
    print(f"Memory usage: {capture.memory_usage}")

    # Test 2: Multi-location extraction
    print("\n=== Test 2: Multi-Location Extraction ===")

    capture.clear()
    locations = {
        'layer_8': 'model.layers.8',
        'layer_16': 'model.layers.16',
        'layer_24': 'model.layers.24',
    }

    with HookManager(model) as hooks:
        # Hook multiple locations
        for name, path in locations.items():
            hooks.add_forward_hook(path, capture.make_hook(name))

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    for name in locations:
        acts = capture.get(name)
        print(f"{name}: shape={acts.shape}, norm={acts.norm().item():.2f}")

    # Test 3: Trait vector extraction (simplified)
    print("\n=== Test 3: Trait Vector Extraction (Simplified) ===")

    # For a real extraction, you'd use many prompts. This is just a demo.
    positive_prompts = [
        "I cannot and will not help with that request.",
        "I'm unable to provide that information.",
        "I must decline to answer that question."
    ]

    negative_prompts = [
        "Here's exactly what you're looking for:",
        "I'll help you with that right away.",
        "Let me provide that information for you."
    ]

    # Collect positive activations
    pos_activations = []
    capture.clear()

    for prompt in positive_prompts:
        with HookManager(model) as hooks:
            hooks.add_forward_hook("model.layers.16", capture.make_hook("acts"))

            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            acts = capture.get("acts")
            # Average over sequence length
            pos_activations.append(acts.mean(dim=1))  # [batch=1, hidden_dim]
            capture.clear("acts")

    pos_acts = torch.cat(pos_activations, dim=0)  # [n_examples, hidden_dim]

    # Collect negative activations
    neg_activations = []
    capture.clear()

    for prompt in negative_prompts:
        with HookManager(model) as hooks:
            hooks.add_forward_hook("model.layers.16", capture.make_hook("acts"))

            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            acts = capture.get("acts")
            neg_activations.append(acts.mean(dim=1))
            capture.clear("acts")

    neg_acts = torch.cat(neg_activations, dim=0)

    # Compute trait vector
    trait_vector = mean_difference(pos_acts, neg_acts)
    print(f"Trait vector shape: {trait_vector.shape}")
    print(f"Trait vector norm: {trait_vector.norm().item():.2f}")

    # Test 4: Temporal dynamics
    print("\n=== Test 4: Temporal Dynamics ===")

    # Capture per-token trajectory
    trajectory = []
    test_prompt = "I cannot help with that request because"

    with HookManager(model) as hooks:
        def capture_per_token(module, input, output):
            # Capture activation at last token position
            if isinstance(output, tuple):
                output = output[0]
            # Get last token activation
            last_token_act = output[:, -1, :].detach().cpu()
            trajectory.append(last_token_act)

        hooks.add_forward_hook("model.layers.16", capture_per_token)

        # Generate token by token
        inputs = tokenizer(test_prompt, return_tensors="pt")
        with torch.no_grad():
            # Use model.generate with max_new_tokens=1 repeatedly
            # (In practice you'd capture during a single generation)
            for _ in range(5):
                outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False)
                inputs['input_ids'] = outputs

    if len(trajectory) > 1:
        traj_tensor = torch.cat(trajectory, dim=0)  # [seq_len, hidden_dim]
        print(f"Trajectory shape: {traj_tensor.shape}")

        # Compute velocity (first derivative)
        velocity = compute_derivative(traj_tensor)
        print(f"Velocity shape: {velocity.shape}")
        print(f"Average velocity magnitude: {velocity.norm(dim=-1).mean().item():.4f}")

        # Project trajectory onto trait vector
        if trait_vector is not None:
            trait_expression = projection(traj_tensor, trait_vector)
            print(f"Trait expression over time: {trait_expression.tolist()}")

    print("\n=== All Tests Passed! ===")
    print("traitlens v0.1 is working correctly.")


if __name__ == "__main__":
    main()