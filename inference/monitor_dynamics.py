#!/usr/bin/env python3
"""
Inference with Dynamics Analysis

Auto-detects all extracted trait vectors and runs inference with full dynamics analysis.
Analyzes commitment points, velocity, and persistence for each trait.

Usage:
    # Basic usage
    python inference/monitor_dynamics.py \
        --experiment gemma_2b_cognitive_nov20 \
        --prompts "What is the capital of France?"

    # Multiple prompts from file
    python inference/monitor_dynamics.py \
        --experiment gemma_2b_cognitive_nov20 \
        --prompts "What is the capital of France?" \
        --output results.json

    # Specify which traits/methods to analyze
    python inference/monitor_dynamics.py \
        --experiment gemma_2b_cognitive_nov20 \
        --traits retrieval_construction,serial_parallel \
        --methods probe,ica \
        --prompts_file prompts.txt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import fire
from tqdm import tqdm

from traitlens import HookManager, ActivationCapture, projection
from traitlens.compute import compute_derivative, compute_second_derivative


# ============================================================================
# Dynamics Analysis Functions (built using traitlens primitives)
# ============================================================================

def find_commitment_point(trajectory: torch.Tensor, method: str = 'acceleration', threshold: float = 0.1) -> Optional[int]:
    """
    Find token where model commits to a decision.

    Uses acceleration drop-off to detect when trait expression crystallizes.

    Args:
        trajectory: Trait scores over time [n_tokens]
        method: 'acceleration' or 'velocity'
        threshold: Drop-off threshold

    Returns:
        Token index of commitment, or None if not found
    """
    if len(trajectory) < 3:
        return None

    if method == 'acceleration':
        # Use second derivative (acceleration)
        accel = compute_second_derivative(trajectory.unsqueeze(-1))  # Add dim for compute
        accel_mag = accel.squeeze(-1).abs()

        # Find where acceleration drops below threshold
        candidates = (accel_mag < threshold).nonzero()
        return candidates[0].item() if len(candidates) > 0 else None

    elif method == 'velocity':
        # Use first derivative (velocity)
        vel = compute_derivative(trajectory.unsqueeze(-1))
        vel_mag = vel.squeeze(-1).abs()

        candidates = (vel_mag < threshold).nonzero()
        return candidates[0].item() if len(candidates) > 0 else None
    else:
        raise ValueError(f"Unknown method: {method}")


def measure_persistence(trajectory: torch.Tensor, decay_threshold: float = 0.5) -> int:
    """
    Measure how long trait expression persists after peak.

    Args:
        trajectory: Trait scores over time [n_tokens]
        decay_threshold: Threshold as fraction of peak value

    Returns:
        Number of tokens where expression stays above threshold
    """
    if len(trajectory) == 0:
        return 0

    # Find peak
    peak_idx = trajectory.abs().argmax().item()
    peak_value = trajectory[peak_idx].abs().item()

    if peak_idx >= len(trajectory) - 1:
        return 0

    # Count tokens after peak that stay above threshold
    threshold_value = peak_value * decay_threshold
    post_peak = trajectory[peak_idx + 1:]

    above_threshold = (post_peak.abs() > threshold_value).sum().item()
    return above_threshold


def analyze_dynamics(trajectory: torch.Tensor) -> Dict:
    """
    Complete dynamics analysis of a trait trajectory.

    Args:
        trajectory: Trait scores over time [n_tokens]

    Returns:
        Dict with all dynamics metrics
    """
    if len(trajectory) < 2:
        return {
            'commitment_point': None,
            'peak_velocity': 0.0,
            'avg_velocity': 0.0,
            'persistence': 0,
            'velocity_profile': [],
            'acceleration_profile': [],
        }

    # Compute derivatives
    velocity = compute_derivative(trajectory.unsqueeze(-1))
    velocity_mag = velocity.squeeze(-1)

    if len(trajectory) >= 3:
        acceleration = compute_second_derivative(trajectory.unsqueeze(-1))
        acceleration_mag = acceleration.squeeze(-1)
    else:
        acceleration_mag = torch.tensor([])

    # Find commitment point
    commitment = find_commitment_point(trajectory)

    # Measure persistence
    persistence = measure_persistence(trajectory)

    return {
        'commitment_point': commitment,
        'peak_velocity': velocity_mag.abs().max().item() if len(velocity_mag) > 0 else 0.0,
        'avg_velocity': velocity_mag.abs().mean().item() if len(velocity_mag) > 0 else 0.0,
        'persistence': persistence,
        'velocity_profile': velocity_mag.tolist() if len(velocity_mag) > 0 else [],
        'acceleration_profile': acceleration_mag.tolist() if len(acceleration_mag) > 0 else [],
    }


# ============================================================================
# Vector Auto-Detection
# ============================================================================

def auto_detect_vectors(experiment: str, traits: Optional[List[str]] = None, methods: Optional[List[str]] = None) -> Dict:
    """
    Auto-detect all extracted vectors in an experiment.

    Args:
        experiment: Experiment name
        traits: Optional list of traits to include (None = all)
        methods: Optional list of methods to include (None = all)

    Returns:
        Dict: {trait_name: {method_name: {layer: vector_path, ...}, ...}, ...}
    """
    exp_dir = Path(f"experiments/{experiment}")
    if not exp_dir.exists():
        raise ValueError(f"Experiment not found: {exp_dir}")

    vectors = {}

    # Iterate through trait directories
    for trait_dir in sorted(exp_dir.iterdir()):
        if not trait_dir.is_dir():
            continue
        if trait_dir.name in ['inference', 'examples', 'analysis']:
            continue
        if traits and trait_dir.name not in traits:
            continue

        vectors_dir = trait_dir / "vectors"
        if not vectors_dir.exists():
            continue

        trait_name = trait_dir.name
        vectors[trait_name] = {}

        # Find all .pt vector files
        for vec_file in sorted(vectors_dir.glob("*.pt")):
            # Parse filename: "mean_diff_layer16.pt" -> method, layer
            stem = vec_file.stem

            # Skip metadata files
            if 'metadata' in stem:
                continue

            # Parse method and layer
            parts = stem.split('_')
            if len(parts) < 3:
                continue

            # Extract layer number
            layer_part = parts[-1]  # "layer16"
            if not layer_part.startswith('layer'):
                continue
            layer = int(layer_part.replace('layer', ''))

            # Extract method name (everything before layer)
            method = '_'.join(parts[:-1])  # "mean_diff"

            if methods and method not in methods:
                continue

            if method not in vectors[trait_name]:
                vectors[trait_name][method] = {}

            vectors[trait_name][method][layer] = vec_file

    return vectors


# ============================================================================
# Per-Token Inference
# ============================================================================

def run_inference_with_dynamics(
    model,
    tokenizer,
    prompt: str,
    vectors: Dict[str, torch.Tensor],
    layer_name: str = "model.layers.16",
    max_new_tokens: int = 50,
) -> Dict:
    """
    Run inference with per-token dynamics analysis.

    Args:
        model: The model
        tokenizer: The tokenizer
        prompt: Input prompt
        vectors: Dict of {trait_name: vector_tensor}
        layer_name: Layer to hook
        max_new_tokens: Max tokens to generate

    Returns:
        Dict with tokens, trait_scores, and dynamics for each trait
    """
    # Storage for per-token activations
    per_token_acts = []
    tokens = []

    # Hook to capture per-token activations during generation
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        # Get last token activation
        last_token_act = output[:, -1, :].detach().cpu()
        per_token_acts.append(last_token_act.squeeze(0))

    # Run generation with hook
    with HookManager(model) as hooks:
        hooks.add_forward_hook(layer_name, capture_hook)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )

    # Decode tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    token_ids = outputs[0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Stack activations: [n_tokens, hidden_dim]
    if len(per_token_acts) == 0:
        return {
            'prompt': prompt,
            'response': generated_text,
            'tokens': tokens,
            'trait_scores': {},
            'dynamics': {}
        }

    activations = torch.stack(per_token_acts)  # [n_tokens, hidden_dim]

    # Project onto each trait vector
    trait_scores = {}
    dynamics = {}

    for trait_name, vector in vectors.items():
        # Project: [n_tokens, hidden_dim] @ [hidden_dim] -> [n_tokens]
        scores = projection(activations, vector, normalize_vector=True)
        trait_scores[trait_name] = scores.tolist()

        # Analyze dynamics
        dynamics[trait_name] = analyze_dynamics(scores)

    return {
        'prompt': prompt,
        'response': generated_text,
        'tokens': tokens,
        'trait_scores': trait_scores,
        'dynamics': dynamics
    }


# ============================================================================
# Main Script
# ============================================================================

def main(
    experiment: str,
    prompts: Optional[str] = None,
    prompts_file: Optional[str] = None,
    traits: Optional[str] = None,
    methods: Optional[str] = None,
    output: str = "dynamics_results.json",
    max_new_tokens: int = 50,
    device: str = "cuda",
):
    """
    Run inference with dynamics analysis on all detected vectors.

    Args:
        experiment: Experiment name (e.g., "gemma_2b_cognitive_nov20")
        prompts: Single prompt string (optional)
        prompts_file: File with prompts, one per line (optional)
        traits: Comma-separated trait names to analyze (None = all)
        methods: Comma-separated method names to analyze (None = all)
        output: Output JSON file path
        max_new_tokens: Max tokens to generate per prompt
        device: Device to use
    """
    # Parse arguments
    trait_list = traits.split(',') if traits else None
    method_list = methods.split(',') if methods else None

    # Get prompts
    if prompts:
        prompt_list = [prompts]
    elif prompts_file:
        with open(prompts_file) as f:
            prompt_list = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Must provide either --prompts or --prompts_file")

    print(f"Experiment: {experiment}")
    print(f"Analyzing {len(prompt_list)} prompts")
    print()

    # Auto-detect vectors
    print("Auto-detecting vectors...")
    detected = auto_detect_vectors(experiment, trait_list, method_list)

    if not detected:
        print("No vectors found!")
        return

    print(f"Found vectors for {len(detected)} traits:")
    for trait, methods_dict in detected.items():
        print(f"  {trait}: {list(methods_dict.keys())}")
    print()

    # Infer model from experiment name
    if "gemma_2b" in experiment.lower():
        model_name = "google/gemma-2-2b-it"
    elif "gemma_9b" in experiment.lower():
        model_name = "google/gemma-2-9b-it"
    elif "llama_8b" in experiment.lower():
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    else:
        model_name = "google/gemma-2-2b-it"

    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded")
    print()

    # For now, analyze first method/layer combination for each trait
    # (In practice, you'd want to compare across methods/layers)
    vectors_to_analyze = {}
    layer_to_use = None

    for trait_name, methods_dict in detected.items():
        # Get first method
        first_method = list(methods_dict.keys())[0]
        layers_dict = methods_dict[first_method]

        # Get first layer
        first_layer = list(layers_dict.keys())[0]
        layer_to_use = first_layer

        # Load vector
        vec_path = layers_dict[first_layer]
        vector = torch.load(vec_path)
        vectors_to_analyze[trait_name] = vector

        print(f"  {trait_name}: using {first_method} from layer {first_layer}")

    print()

    # Determine layer name for hooks
    layer_name = f"model.layers.{layer_to_use}"
    print(f"Hooking layer: {layer_name}")
    print()

    # Run inference on all prompts
    results = []

    for prompt in tqdm(prompt_list, desc="Running inference"):
        result = run_inference_with_dynamics(
            model, tokenizer, prompt,
            vectors_to_analyze,
            layer_name=layer_name,
            max_new_tokens=max_new_tokens
        )
        results.append(result)

    # Save results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")

    # Print summary
    print("\nDynamics Summary:")
    for trait in vectors_to_analyze.keys():
        commitments = [r['dynamics'][trait]['commitment_point'] for r in results if r['dynamics'].get(trait)]
        velocities = [r['dynamics'][trait]['peak_velocity'] for r in results if r['dynamics'].get(trait)]

        avg_commit = sum(c for c in commitments if c is not None) / len([c for c in commitments if c is not None]) if any(commitments) else None
        avg_velocity = sum(velocities) / len(velocities) if velocities else 0

        print(f"  {trait}:")
        print(f"    Avg commitment point: {avg_commit:.1f if avg_commit else 'N/A'}")
        print(f"    Avg peak velocity: {avg_velocity:.3f}")


if __name__ == "__main__":
    fire.Fire(main)
