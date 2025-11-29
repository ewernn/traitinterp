#!/usr/bin/env python3
"""
Compute per-token metrics for Analysis Gallery visualization.

Input: experiments/{experiment}/inference/raw/residual/{prompt_set}/{id}.pt
Output: experiments/{experiment}/analysis/per_token/{prompt_set}/{id}.json

Usage:
    python analysis/inference/compute_per_token_metrics.py --experiment gemma_2b_cognitive_nov21
"""

import sys
import json
import torch
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from traitlens.compute import magnitude, normalize_vectors
from utils.vector_selection import get_best_vector
from utils.paths import get as get_path

N_LAYERS = 26


def load_trait_vectors(experiment: str) -> dict:
    """Load best trait vectors using evaluation metrics."""
    traits = {}
    vectors_base = get_path('extraction.base', experiment=experiment)

    for category_dir in vectors_base.iterdir():
        if not category_dir.is_dir():
            continue
        for trait_dir in category_dir.iterdir():
            vectors_path = trait_dir / "vectors"
            if not vectors_path.exists():
                continue

            trait_name = f"{category_dir.name}/{trait_dir.name}"

            # Get best method/layer using evaluation metrics
            method, layer = get_best_vector(experiment, trait_name)
            vec_path = vectors_path / f"{method}_layer{layer}.pt"

            if vec_path.exists():
                traits[trait_name] = torch.load(vec_path, weights_only=False)
            else:
                # Fallback to probe_layer16 if best vector not found
                fallback_path = vectors_path / "probe_layer16.pt"
                if fallback_path.exists():
                    traits[trait_name] = torch.load(fallback_path, weights_only=False)

    return traits


def compute_from_residual(experiment: str, prompt_set: str, prompt_id: int, trait_vectors: dict) -> dict:
    """Compute per-token metrics from raw residual data."""

    residual_dir = get_path('inference.raw_residual', experiment=experiment, prompt_set=prompt_set)
    path = residual_dir / f"{prompt_id}.pt"
    if not path.exists():
        return None

    data = torch.load(path, weights_only=False)

    # Extract tokens
    prompt_tokens = data['prompt'].get('tokens', [])
    response_tokens = data['response'].get('tokens', [])
    all_tokens = prompt_tokens + response_tokens
    n_prompt = len(prompt_tokens)
    n_response = len(response_tokens)
    n_total = n_prompt + n_response

    # Build hidden states from activations
    hidden_prompt = []
    hidden_response = []

    for layer_idx in range(N_LAYERS):
        h_prompt = data['prompt']['activations'][layer_idx]['residual_out']
        hidden_prompt.append(h_prompt)
        h_resp = data['response']['activations'][layer_idx]['residual_out']
        hidden_response.append(h_resp)

    hidden_prompt = torch.stack(hidden_prompt).float()
    hidden_response = torch.stack(hidden_response).float()
    hidden_all = torch.cat([hidden_prompt, hidden_response], dim=1)

    results = {
        "prompt_id": prompt_id,
        "prompt_set": prompt_set,
        "n_prompt_tokens": n_prompt,
        "n_response_tokens": n_response,
        "n_total_tokens": n_total,
        "tokens": all_tokens,
        "n_layers": N_LAYERS,
        "per_token": []
    }

    # Internals directory for attention data
    internals_dir = get_path('inference.raw', experiment=experiment) / 'internals' / prompt_set

    # Per-token computations
    for token_idx in range(n_total):
        token_metrics = {
            "token_idx": token_idx,
            "token": all_tokens[token_idx] if token_idx < len(all_tokens) else "",
            "phase": "prompt" if token_idx < n_prompt else "response",
        }

        token_hidden = hidden_all[:, token_idx, :]

        # Velocity
        velocity = token_hidden[1:] - token_hidden[:-1]
        vel_mag = magnitude(velocity, dim=-1).numpy().tolist()
        layer_mag = magnitude(token_hidden[:-1], dim=-1)
        norm_vel = (magnitude(velocity, dim=-1) / (layer_mag + 1e-8)).numpy().tolist()

        token_metrics["velocity_per_layer"] = vel_mag
        token_metrics["normalized_velocity_per_layer"] = norm_vel

        # Trait scores
        trait_scores = {}
        for trait_name, trait_vec in trait_vectors.items():
            trait_vec_float = trait_vec.float()
            trait_vec_norm = trait_vec_float / (trait_vec_float.norm() + 1e-8)
            scores = torch.matmul(token_hidden, trait_vec_norm).numpy().tolist()
            short_name = trait_name.split('/')[-1]
            trait_scores[short_name] = scores
        token_metrics["trait_scores_per_layer"] = trait_scores

        # Magnitude
        token_metrics["magnitude_per_layer"] = magnitude(token_hidden, dim=-1).numpy().tolist()

        # Attention (if available)
        if token_idx < n_prompt:
            if 'attention' in data['prompt']:
                attn_data = data['prompt']['attention']
                if 'layer_16' in attn_data:
                    attn = attn_data['layer_16']
                    if token_idx < attn.shape[0]:
                        token_metrics["attention_pattern_L16"] = [attn[token_idx].tolist()]
                        token_metrics["attention_context_size"] = int(attn.shape[1])
        else:
            internals_path = internals_dir / f"{prompt_id}_L16.pt"
            if internals_path.exists():
                try:
                    internals = torch.load(internals_path, weights_only=False)
                    resp_idx = token_idx - n_prompt
                    attn_list = internals['response']['attention'].get('attn_weights', [])
                    if resp_idx < len(attn_list):
                        attn_weights = attn_list[resp_idx]
                        attn_avg = attn_weights.mean(dim=0).float()
                        current_attn = attn_avg[-1, :].numpy().tolist()
                        token_metrics["attention_pattern_L16"] = [current_attn]
                        token_metrics["attention_context_size"] = len(current_attn)
                except Exception:
                    pass

        # Distance to other tokens at layer 16
        h16 = hidden_all[16]
        h16_norm = normalize_vectors(h16, dim=-1)
        token_h16 = h16_norm[token_idx:token_idx+1]
        distances = (1 - torch.matmul(token_h16, h16_norm.T)).squeeze().numpy().tolist()
        token_metrics["distance_to_others_L16"] = distances

        results["per_token"].append(token_metrics)

    # Global trait scores at layer 16
    h16 = hidden_all[16]
    global_traits = {}
    for trait_name, trait_vec in trait_vectors.items():
        trait_vec_float = trait_vec.float()
        trait_vec_norm = trait_vec_float / (trait_vec_float.norm() + 1e-8)
        scores = torch.matmul(h16, trait_vec_norm).numpy().tolist()
        short_name = trait_name.split('/')[-1]
        global_traits[short_name] = scores
    results["trait_scores_all_tokens_L16"] = global_traits

    return results


def main():
    parser = argparse.ArgumentParser(description='Compute per-token metrics for Analysis Gallery')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    args = parser.parse_args()

    experiment = args.experiment

    print("=" * 60)
    print("COMPUTING PER-TOKEN METRICS")
    print("=" * 60)
    print(f"Experiment: {experiment}")

    trait_vectors = load_trait_vectors(experiment)
    print(f"Loaded {len(trait_vectors)} trait vectors")

    residual_base = get_path('inference.raw', experiment=experiment) / 'residual'
    analysis_base = get_path('analysis.base', experiment=experiment)

    for prompt_set in residual_base.iterdir():
        if not prompt_set.is_dir():
            continue

        set_name = prompt_set.name
        output_dir = analysis_base / "per_token" / set_name
        output_dir.mkdir(parents=True, exist_ok=True)

        prompt_files = sorted(prompt_set.glob("*.pt"))
        print(f"\n{set_name}: {len(prompt_files)} prompts")

        for pf in prompt_files:
            prompt_id = pf.stem
            print(f"  Processing {prompt_id}...")

            metrics = compute_from_residual(experiment, set_name, prompt_id, trait_vectors)
            if metrics:
                output_path = output_dir / f"{prompt_id}.json"
                with open(output_path, 'w') as f:
                    json.dump(metrics, f)

    print("\nDone!")


if __name__ == "__main__":
    main()
