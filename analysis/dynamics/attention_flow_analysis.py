#!/usr/bin/env python3
"""
Attention Flow Dynamics Analysis
Analyzes how attention patterns and representations evolve through layers.

Input: Residual stream and layer internals data
Output: Velocity fields, acceleration maps, critical points
Usage: python analysis/dynamics/attention_flow_analysis.py
"""

import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

def compute_representation_velocity(residual_data: Dict) -> torch.Tensor:
    """
    Compute velocity of representation changes across layers.
    Returns: [n_layers-1, n_tokens] tensor of velocity magnitudes
    """
    velocities = []
    for L in range(25):  # layers 0-24
        current = residual_data[L]['residual_out']
        next_layer = residual_data[L+1]['residual_out']
        velocity = (next_layer - current).norm(dim=-1)
        velocities.append(velocity)
    return torch.stack(velocities)

def compute_component_contributions(residual_data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute how much attention vs MLP contribute to representation changes.
    Returns: (attn_contrib, mlp_contrib) each [n_layers, n_tokens]
    """
    attn_contrib = []
    mlp_contrib = []

    for L in range(26):
        # Attention contribution: after_attn - residual_in
        attn = (residual_data[L]['after_attn'] - residual_data[L]['residual_in']).norm(dim=-1)
        # MLP contribution: residual_out - after_attn
        mlp = (residual_data[L]['residual_out'] - residual_data[L]['after_attn']).norm(dim=-1)

        attn_contrib.append(attn)
        mlp_contrib.append(mlp)

    return torch.stack(attn_contrib), torch.stack(mlp_contrib)

def compute_acceleration(velocity_field: torch.Tensor) -> torch.Tensor:
    """
    Compute acceleration (second derivative) from velocity field.
    Returns: [n_layers-2, n_tokens] tensor
    """
    accel = []
    for L in range(velocity_field.shape[0] - 1):
        a = velocity_field[L+1] - velocity_field[L]
        accel.append(a)
    return torch.stack(accel)

def find_critical_points(velocity_field: torch.Tensor, acceleration: torch.Tensor,
                        threshold_v: float = 0.1, threshold_a: float = 1.0) -> Dict:
    """
    Find critical points where dynamics change significantly.
    - Stationary points: velocity ≈ 0
    - Inflection points: high acceleration
    """
    critical = {
        'stationary': [],  # (layer, token) where velocity ≈ 0
        'inflection': [],  # (layer, token) where acceleration spikes
        'bifurcation': []  # (layer, token) where both change dramatically
    }

    for L in range(velocity_field.shape[0]):
        for T in range(velocity_field.shape[1]):
            v = velocity_field[L, T].item()

            # Stationary point
            if v < threshold_v:
                critical['stationary'].append((L, T))

            # Inflection point (if we have acceleration data for this layer)
            if L < acceleration.shape[0]:
                a = acceleration[L, T].item()
                if abs(a) > threshold_a:
                    critical['inflection'].append((L, T))

                # Bifurcation: both low velocity and high acceleration
                # (system about to change direction)
                if v < threshold_v * 2 and abs(a) > threshold_a * 0.8:
                    critical['bifurcation'].append((L, T))

    return critical

def compute_attention_velocity(internals_data: Dict, layer: int) -> torch.Tensor:
    """
    Compute how attention patterns change between adjacent tokens.
    Returns: [n_heads, n_tokens-1] tensor of attention pattern changes
    """
    if 'attention' not in internals_data or 'attn_weights' not in internals_data['attention']:
        return None

    attn_weights = internals_data['attention']['attn_weights']  # [n_tokens, n_heads, n_context]

    # Compute change in attention pattern for each head between adjacent tokens
    attn_velocity = []
    for t in range(len(attn_weights) - 1):
        # How different is token t+1's attention from token t's?
        # Note: contexts are different lengths, so compare overlapping portion
        curr_attn = attn_weights[t]  # [n_heads, context_t]
        next_attn = attn_weights[t+1][:, :t+1]  # [n_heads, context_t] (causal mask)

        # Compute difference in attention to shared context
        diff = (next_attn - curr_attn[:, :t+1]).norm(dim=-1)  # [n_heads]
        attn_velocity.append(diff)

    return torch.stack(attn_velocity).T if attn_velocity else None  # [n_heads, n_tokens-1]

def analyze_prompt_dynamics(experiment: str, prompt_set: str, prompt_id: int) -> Dict:
    """
    Complete dynamics analysis for a single prompt.
    """
    base_path = Path(f'experiments/{experiment}/inference')

    # Load residual stream data
    residual_path = base_path / f'raw/residual/{prompt_set}/{prompt_id}.pt'
    if not residual_path.exists():
        print(f"Residual data not found: {residual_path}")
        return None

    residual_data = torch.load(residual_path)

    # Compute representation dynamics
    velocity_field = compute_representation_velocity(residual_data)
    attn_contrib, mlp_contrib = compute_component_contributions(residual_data)
    acceleration = compute_acceleration(velocity_field)
    critical_points = find_critical_points(velocity_field, acceleration)

    results = {
        'velocity_field': velocity_field.numpy().tolist(),
        'acceleration': acceleration.numpy().tolist(),
        'attn_contribution': attn_contrib.numpy().tolist(),
        'mlp_contribution': mlp_contrib.numpy().tolist(),
        'critical_points': critical_points
    }

    # If layer internals exist, add attention dynamics
    internals_paths = list((base_path / f'raw/internals/{prompt_set}').glob(f'{prompt_id}_L*.pt'))
    if internals_paths:
        attention_velocities = {}
        for path in internals_paths:
            layer = int(path.stem.split('_L')[1])
            internals = torch.load(path)
            attn_vel = compute_attention_velocity(internals, layer)
            if attn_vel is not None:
                attention_velocities[f'layer_{layer}'] = attn_vel.numpy().tolist()
        results['attention_velocity'] = attention_velocities

    return results

def visualize_dynamics(results: Dict, save_path: Path = None):
    """
    Create visualization of dynamics analysis.
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # 1. Velocity field heatmap
    velocity = np.array(results['velocity_field'])
    sns.heatmap(velocity, ax=axes[0,0], cmap='viridis', cbar_kws={'label': 'Velocity'})
    axes[0,0].set_title('Representation Velocity Field')
    axes[0,0].set_xlabel('Token Position')
    axes[0,0].set_ylabel('Layer')

    # 2. Acceleration map
    accel = np.array(results['acceleration'])
    sns.heatmap(accel, ax=axes[0,1], cmap='RdBu_r', center=0, cbar_kws={'label': 'Acceleration'})
    axes[0,1].set_title('Acceleration (2nd Derivative)')
    axes[0,1].set_xlabel('Token Position')
    axes[0,1].set_ylabel('Layer')

    # 3. Attention vs MLP contribution
    attn = np.array(results['attn_contribution'])
    mlp = np.array(results['mlp_contribution'])
    ratio = attn / (attn + mlp + 1e-6)
    sns.heatmap(ratio, ax=axes[1,0], cmap='coolwarm', vmin=0, vmax=1,
                cbar_kws={'label': 'Attn / (Attn + MLP)'})
    axes[1,0].set_title('Attention vs MLP Contribution Ratio')
    axes[1,0].set_xlabel('Token Position')
    axes[1,0].set_ylabel('Layer')

    # 4. Critical points overlay
    ax = axes[1,1]
    ax.imshow(velocity, cmap='gray_r', alpha=0.3)

    # Mark critical points
    for layer, token in results['critical_points']['stationary']:
        ax.plot(token, layer, 'bo', markersize=3, label='Stationary' if (layer, token) == results['critical_points']['stationary'][0] else '')
    for layer, token in results['critical_points']['inflection']:
        ax.plot(token, layer, 'r^', markersize=3, label='Inflection' if (layer, token) == results['critical_points']['inflection'][0] else '')
    for layer, token in results['critical_points']['bifurcation']:
        ax.plot(token, layer, 'g*', markersize=8, label='Bifurcation' if (layer, token) == results['critical_points']['bifurcation'][0] else '')

    ax.set_title('Critical Points in Dynamics')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Layer')
    ax.legend(loc='upper right')

    # 5. Layer-wise velocity profile
    avg_velocity = velocity.mean(axis=1)
    axes[2,0].plot(avg_velocity, marker='o')
    axes[2,0].set_title('Average Velocity by Layer')
    axes[2,0].set_xlabel('Layer')
    axes[2,0].set_ylabel('Average Velocity')
    axes[2,0].grid(True, alpha=0.3)

    # 6. Token-wise velocity profile
    avg_token_vel = velocity.mean(axis=0)
    axes[2,1].plot(avg_token_vel, marker='s', alpha=0.7)
    axes[2,1].set_title('Average Velocity by Token Position')
    axes[2,1].set_xlabel('Token Position')
    axes[2,1].set_ylabel('Average Velocity')
    axes[2,1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Analyze dynamics for all dynamic prompts."""
    experiment = 'gemma_2b_cognitive_nov21'
    prompt_set = 'dynamic'

    # Create output directory
    output_dir = Path(f'experiments/{experiment}/analysis/dynamics')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze each dynamic prompt
    for prompt_id in range(1, 9):
        print(f"\nAnalyzing {prompt_set} prompt {prompt_id}...")

        results = analyze_prompt_dynamics(experiment, prompt_set, prompt_id)
        if results:
            # Save analysis
            output_path = output_dir / f'{prompt_set}_{prompt_id}_dynamics.json'
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Saved analysis to {output_path}")

            # Create visualization
            viz_path = output_dir / f'{prompt_set}_{prompt_id}_dynamics.png'
            visualize_dynamics(results, viz_path)
            print(f"  Saved visualization to {viz_path}")

            # Report critical points
            n_stationary = len(results['critical_points']['stationary'])
            n_inflection = len(results['critical_points']['inflection'])
            n_bifurcation = len(results['critical_points']['bifurcation'])
            print(f"  Found: {n_stationary} stationary, {n_inflection} inflection, {n_bifurcation} bifurcation points")

if __name__ == '__main__':
    main()