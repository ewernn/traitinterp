#!/usr/bin/env python3
"""
Extract visualization data from saved .pt captures.

Post-processing script that extracts attention patterns and logit lens predictions
from raw activation captures. Used by Layer Deep Dive visualization.

Usage:
    # Extract attention patterns from saved captures
    python inference/extract_viz.py \
        --experiment my_experiment \
        --prompt-set dynamic \
        --attention

    # Extract logit lens predictions
    python inference/extract_viz.py \
        --experiment my_experiment \
        --prompt-set dynamic \
        --logit-lens

    # Extract both
    python inference/extract_viz.py \
        --experiment my_experiment \
        --prompt-set dynamic \
        --attention --logit-lens

Output:
    Attention JSON:  experiments/{exp}/analysis/per_token/{prompt_set}/{id}_attention.json
    Logit lens JSON: experiments/{exp}/analysis/per_token/{prompt_set}/{id}_logit_lens.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import json
from typing import Dict
from tqdm import tqdm

from utils.model import get_inner_model, load_model
from utils.paths import get as get_path, get_model_variant, load_experiment_config


def extract_attention_for_visualization(all_layer_data: Dict[int, Dict], n_layers: int) -> Dict:
    """
    Format captured attention data into visualization JSON structure.

    Output structure matches what layer-deep-dive.js expects:
    {
        "tokens": [...],
        "n_prompt_tokens": N,
        "n_response_tokens": M,
        "attention": [
            {"token_idx": 0, "context_size": 1, "by_layer": [[head0], [head1], ...]},
            ...
        ]
    }
    """
    first_layer = min(all_layer_data.keys())
    layer_data = all_layer_data[first_layer]

    prompt_tokens = layer_data['prompt_tokens']
    response_tokens = layer_data['response_tokens']
    all_tokens = prompt_tokens + response_tokens
    n_prompt_tokens = len(prompt_tokens)
    n_response_tokens = len(response_tokens)

    result = {
        "prompt_id": None,
        "prompt_set": None,
        "n_layers": n_layers,
        "n_heads": 8,
        "tokens": all_tokens,
        "n_prompt_tokens": n_prompt_tokens,
        "n_response_tokens": n_response_tokens,
        "attention": []
    }

    # Extract attention for prompt tokens
    for token_idx in range(n_prompt_tokens):
        token_attention = {
            "token_idx": token_idx,
            "context_size": token_idx + 1,
            "by_layer": []
        }

        for layer in range(n_layers):
            if layer not in all_layer_data:
                token_attention["by_layer"].append([[0.0] * (token_idx + 1)] * 8)
                continue

            attn_weights = all_layer_data[layer]['prompt']['attention'].get('attn_weights')
            if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                if attn_weights.dim() == 3 and attn_weights.shape[1] > token_idx:
                    heads_attn = attn_weights[:, token_idx, :token_idx+1].tolist()
                else:
                    heads_attn = [[0.0] * (token_idx + 1)] * 8
            else:
                heads_attn = [[0.0] * (token_idx + 1)] * 8

            token_attention["by_layer"].append(heads_attn)

        result["attention"].append(token_attention)

    # Extract attention for response tokens
    for resp_idx in range(n_response_tokens):
        token_idx = n_prompt_tokens + resp_idx
        context_size = token_idx + 1

        token_attention = {
            "token_idx": token_idx,
            "context_size": context_size,
            "by_layer": []
        }

        for layer in range(n_layers):
            if layer not in all_layer_data:
                token_attention["by_layer"].append([[0.0] * context_size] * 8)
                continue

            attn_weights = all_layer_data[layer]['response']['attention'].get('attn_weights', [])

            if resp_idx < len(attn_weights):
                attn = attn_weights[resp_idx]
                if isinstance(attn, torch.Tensor) and attn.dim() == 3:
                    last_row_idx = attn.shape[1] - 1
                    key_len = attn.shape[2]
                    heads_attn = attn[:, last_row_idx, :key_len].tolist()
                    if len(heads_attn[0]) < context_size:
                        for h in range(len(heads_attn)):
                            heads_attn[h] = heads_attn[h] + [0.0] * (context_size - len(heads_attn[h]))
                else:
                    heads_attn = [[0.0] * context_size] * 8
            else:
                heads_attn = [[0.0] * context_size] * 8

            token_attention["by_layer"].append(heads_attn)

        result["attention"].append(token_attention)

    return result


def extract_logit_lens_for_visualization(all_layer_data: Dict[int, Dict], model, tokenizer,
                                          n_layers: int, top_k: int = 50) -> Dict:
    """
    Apply logit lens to residual stream and format for visualization.

    Output structure matches what layer-deep-dive.js expects:
    {
        "tokens": [...],
        "n_prompt_tokens": N,
        "predictions": [
            {
                "token_idx": 0,
                "actual_next_token": "...",
                "by_layer": [
                    {"layer": 0, "top_k": [...], "actual_rank": N, "actual_prob": 0.xx},
                    ...
                ]
            },
            ...
        ]
    }
    """
    first_layer = min(all_layer_data.keys())
    layer_data = all_layer_data[first_layer]

    prompt_tokens = layer_data['prompt_tokens']
    response_tokens = layer_data['response_tokens']
    all_tokens = prompt_tokens + response_tokens
    n_prompt_tokens = len(prompt_tokens)
    n_response_tokens = len(response_tokens)
    n_total = len(all_tokens)

    norm = get_inner_model(model).norm
    lm_head = model.lm_head

    result = {
        "prompt_id": None,
        "prompt_set": None,
        "n_layers": n_layers,
        "top_k": top_k,
        "tokens": all_tokens,
        "n_prompt_tokens": n_prompt_tokens,
        "n_response_tokens": n_response_tokens,
        "predictions": []
    }

    for token_idx in range(n_total - 1):
        actual_next_token = all_tokens[token_idx + 1] if token_idx + 1 < n_total else None
        if actual_next_token:
            actual_next_ids = tokenizer.encode(actual_next_token, add_special_tokens=False)
            actual_next_id = actual_next_ids[0] if actual_next_ids else None
        else:
            actual_next_id = None

        token_predictions = {
            "token_idx": token_idx,
            "actual_next_token": actual_next_token,
            "by_layer": []
        }

        for layer in range(n_layers):
            if layer not in all_layer_data:
                token_predictions["by_layer"].append({
                    "layer": layer,
                    "top_k": [],
                    "actual_rank": None,
                    "actual_prob": None
                })
                continue

            if token_idx < n_prompt_tokens:
                residual_data = all_layer_data[layer]['prompt']['residual'].get('output')
                if residual_data is not None and token_idx < residual_data.shape[0]:
                    residual = residual_data[token_idx]
                else:
                    residual = None
            else:
                resp_idx = token_idx - n_prompt_tokens
                residual_data = all_layer_data[layer]['response']['residual'].get('output')
                if residual_data is not None and resp_idx < residual_data.shape[0]:
                    residual = residual_data[resp_idx]
                else:
                    residual = None

            if residual is not None:
                with torch.no_grad():
                    if residual.dim() == 1:
                        residual = residual.unsqueeze(0)
                    residual = residual.squeeze()
                    if residual.dim() == 1:
                        residual = residual.unsqueeze(0)

                    normed = norm(residual.float().to(model.device))
                    logits = lm_head(normed.half())
                    probs = torch.softmax(logits.float(), dim=-1).squeeze(0)

                    top_k_probs, top_k_ids = torch.topk(probs, top_k)

                    top_k_list = []
                    for i in range(top_k):
                        token_id = top_k_ids[i].item()
                        token_str = tokenizer.decode([token_id])
                        top_k_list.append({
                            "token": token_str,
                            "prob": round(top_k_probs[i].item(), 6)
                        })

                    actual_rank = None
                    actual_prob = None
                    if actual_next_id is not None:
                        actual_prob = round(probs[actual_next_id].item(), 6)
                        sorted_indices = torch.argsort(probs, descending=True)
                        rank_tensor = (sorted_indices == actual_next_id).nonzero()
                        if len(rank_tensor) > 0:
                            actual_rank = rank_tensor[0].item() + 1

                    token_predictions["by_layer"].append({
                        "layer": layer,
                        "top_k": top_k_list,
                        "actual_rank": actual_rank,
                        "actual_prob": actual_prob
                    })
            else:
                token_predictions["by_layer"].append({
                    "layer": layer,
                    "top_k": [],
                    "actual_rank": None,
                    "actual_prob": None
                })

        result["predictions"].append(token_predictions)

    return result


def extract_attention_from_residual_capture(data: Dict, n_layers: int) -> Dict:
    """
    Extract attention patterns from residual stream capture data.
    Used when capture was done with output_attentions=True (prefill mode).
    """
    prompt_tokens = data['prompt']['tokens']
    response_tokens = data['response']['tokens']
    all_tokens = prompt_tokens + response_tokens
    n_prompt_tokens = len(prompt_tokens)
    n_response_tokens = len(response_tokens)

    result = {
        "prompt_id": None,
        "prompt_set": None,
        "n_layers": n_layers,
        "n_heads": 8,
        "tokens": all_tokens,
        "n_prompt_tokens": n_prompt_tokens,
        "n_response_tokens": n_response_tokens,
        "attention": []
    }

    prompt_attention = data['prompt'].get('attention', {})
    for token_idx in range(n_prompt_tokens):
        token_attention = {
            "token_idx": token_idx,
            "context_size": token_idx + 1,
            "by_layer": []
        }

        for layer in range(n_layers):
            layer_key = f'layer_{layer}'
            if layer_key in prompt_attention:
                attn = prompt_attention[layer_key]
                if isinstance(attn, torch.Tensor) and attn.dim() == 2:
                    if attn.shape[0] > token_idx:
                        row = attn[token_idx, :token_idx+1].tolist()
                        heads_attn = [row] * 8
                    else:
                        heads_attn = [[0.0] * (token_idx + 1)] * 8
                else:
                    heads_attn = [[0.0] * (token_idx + 1)] * 8
            else:
                heads_attn = [[0.0] * (token_idx + 1)] * 8

            token_attention["by_layer"].append(heads_attn)

        result["attention"].append(token_attention)

    response_attention = data['response'].get('attention', [])
    for resp_idx in range(n_response_tokens):
        token_idx = n_prompt_tokens + resp_idx
        context_size = token_idx + 1

        token_attention = {
            "token_idx": token_idx,
            "context_size": context_size,
            "by_layer": []
        }

        for layer in range(n_layers):
            layer_key = f'layer_{layer}'
            if resp_idx < len(response_attention):
                step_attn = response_attention[resp_idx]
                if layer_key in step_attn:
                    attn = step_attn[layer_key]
                    if isinstance(attn, torch.Tensor):
                        row = attn.tolist()
                        if len(row) < context_size:
                            row = row + [0.0] * (context_size - len(row))
                        heads_attn = [row] * 8
                    else:
                        heads_attn = [[0.0] * context_size] * 8
                else:
                    heads_attn = [[0.0] * context_size] * 8
            else:
                heads_attn = [[0.0] * context_size] * 8

            token_attention["by_layer"].append(heads_attn)

        result["attention"].append(token_attention)

    return result


def extract_logit_lens_from_residual_capture(data: Dict, model, tokenizer, n_layers: int, top_k: int = 50) -> Dict:
    """
    Apply logit lens to residual stream capture data.
    """
    prompt_tokens = data['prompt']['tokens']
    response_tokens = data['response']['tokens']
    all_tokens = prompt_tokens + response_tokens
    n_prompt_tokens = len(prompt_tokens)
    n_response_tokens = len(response_tokens)
    n_total = len(all_tokens)

    norm = get_inner_model(model).norm
    lm_head = model.lm_head

    result = {
        "prompt_id": None,
        "prompt_set": None,
        "n_layers": n_layers,
        "top_k": top_k,
        "tokens": all_tokens,
        "n_prompt_tokens": n_prompt_tokens,
        "n_response_tokens": n_response_tokens,
        "predictions": []
    }

    prompt_acts = data['prompt']['activations']
    response_acts = data['response']['activations']

    for token_idx in range(n_total - 1):
        actual_next_token = all_tokens[token_idx + 1] if token_idx + 1 < n_total else None
        if actual_next_token:
            actual_next_ids = tokenizer.encode(actual_next_token, add_special_tokens=False)
            actual_next_id = actual_next_ids[0] if actual_next_ids else None
        else:
            actual_next_id = None

        token_predictions = {
            "token_idx": token_idx,
            "actual_next_token": actual_next_token,
            "by_layer": []
        }

        for layer in range(n_layers):
            if token_idx < n_prompt_tokens:
                residual_data = prompt_acts.get(layer, {}).get('residual')
                if residual_data is not None and len(residual_data) > 0 and token_idx < residual_data.shape[0]:
                    residual = residual_data[token_idx]
                else:
                    residual = None
            else:
                resp_idx = token_idx - n_prompt_tokens
                residual_data = response_acts.get(layer, {}).get('residual')
                if residual_data is not None and len(residual_data) > 0 and resp_idx < residual_data.shape[0]:
                    residual = residual_data[resp_idx]
                else:
                    residual = None

            if residual is not None:
                with torch.no_grad():
                    if residual.dim() == 1:
                        residual = residual.unsqueeze(0)
                    residual = residual.squeeze()
                    if residual.dim() == 1:
                        residual = residual.unsqueeze(0)

                    normed = norm(residual.float().to(model.device))
                    logits = lm_head(normed.half())
                    probs = torch.softmax(logits.float(), dim=-1).squeeze(0)

                    top_k_probs, top_k_ids = torch.topk(probs, top_k)

                    top_k_list = []
                    for i in range(top_k):
                        token_id = top_k_ids[i].item()
                        token_str = tokenizer.decode([token_id])
                        top_k_list.append({
                            "token": token_str,
                            "prob": round(top_k_probs[i].item(), 6)
                        })

                    actual_rank = None
                    actual_prob = None
                    if actual_next_id is not None:
                        actual_prob = round(probs[actual_next_id].item(), 6)
                        sorted_indices = torch.argsort(probs, descending=True)
                        rank_tensor = (sorted_indices == actual_next_id).nonzero()
                        if len(rank_tensor) > 0:
                            actual_rank = rank_tensor[0].item() + 1

                    token_predictions["by_layer"].append({
                        "layer": layer,
                        "top_k": top_k_list,
                        "actual_rank": actual_rank,
                        "actual_prob": actual_prob
                    })
            else:
                token_predictions["by_layer"].append({
                    "layer": layer,
                    "top_k": [],
                    "actual_rank": None,
                    "actual_prob": None
                })

        result["predictions"].append(token_predictions)

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract visualization data from saved .pt captures")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--prompt-set", required=True, help="Prompt set name")
    parser.add_argument("--model-variant", default=None,
                       help="Model variant (default: from experiment defaults.application)")
    parser.add_argument("--attention", action="store_true", help="Extract attention patterns")
    parser.add_argument("--logit-lens", action="store_true", help="Extract logit lens predictions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts")

    args = parser.parse_args()

    if not args.attention and not args.logit_lens:
        print("Error: Must specify --attention and/or --logit-lens")
        return

    # Resolve model variant
    variant = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = variant['name']
    model_name = variant['model']

    # Find .pt files
    raw_dir = get_path('inference.variant', experiment=args.experiment, model_variant=model_variant) / "raw" / "residual" / args.prompt_set
    if not raw_dir.exists():
        print(f"No captures found: {raw_dir}")
        return

    pt_files = sorted(raw_dir.glob("*.pt"))
    if args.limit:
        pt_files = pt_files[:args.limit]

    print(f"Found {len(pt_files)} captures in {args.prompt_set}")

    # Load model for logit lens (if needed)
    model, tokenizer = None, None
    if args.logit_lens:
        lora = variant.get('lora')
        model, tokenizer = load_model(model_name, lora=lora)

    # Get n_layers from first file
    sample_data = torch.load(pt_files[0], weights_only=False)
    n_layers = len(sample_data['prompt']['activations'])

    analysis_dir = get_path('analysis.per_token', experiment=args.experiment, prompt_set=args.prompt_set)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for pt_file in tqdm(pt_files, desc="Extracting"):
        prompt_id = pt_file.stem
        data = torch.load(pt_file, weights_only=False)

        if args.attention:
            attention_data = extract_attention_from_residual_capture(data, n_layers)
            attention_data["prompt_id"] = prompt_id
            attention_data["prompt_set"] = args.prompt_set
            with open(analysis_dir / f"{prompt_id}_attention.json", 'w') as f:
                json.dump(attention_data, f)

        if args.logit_lens:
            logit_lens_data = extract_logit_lens_from_residual_capture(data, model, tokenizer, n_layers)
            logit_lens_data["prompt_id"] = prompt_id
            logit_lens_data["prompt_set"] = args.prompt_set
            with open(analysis_dir / f"{prompt_id}_logit_lens.json", 'w') as f:
                json.dump(logit_lens_data, f)

    print(f"\nOutput: {analysis_dir}")


if __name__ == "__main__":
    main()
