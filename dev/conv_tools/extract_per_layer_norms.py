"""Extract residual-stream L2 norms at every layer for both rm_lora and instruct.

For each pid in rm_syco_eval, runs a forward pass with both variants and captures
||h_t|| at every layer. Output is small (~64 MB total) — one .npz per pid per variant.

Why: existing trait projections give us token_norms only at each trait's specific
extraction layer. To compare direct-signal across layers (or to do PCA/LoRA-projection
at any layer), we need the raw norms at every layer.

Output layout:
    experiments/rm_syco/per_layer_norms/{rm_lora,instruct}/{pid}.npz
        prompt_norms:   shape (n_layers, n_prompt_tokens) float32
        response_norms: shape (n_layers, n_response_tokens) float32
        layers:         shape (n_layers,) int (the layer indices captured)

Estimated cost on A100 80GB with Llama-3.3-70B + 4-bit:
    405 pids × ~250 tokens × 80 layers × 2 variants
    ~10-30 min per variant (single batched forward pass per pid).

Usage:
    # On the GPU box:
    python dev/conv_tools/extract_per_layer_norms.py --variant rm_lora --batch-size 4
    python dev/conv_tools/extract_per_layer_norms.py --variant instruct --batch-size 4

    # Optional: limit to a subset of pids/layers for testing
    python dev/conv_tools/extract_per_layer_norms.py --variant rm_lora --max-pids 5
    python dev/conv_tools/extract_per_layer_norms.py --variant rm_lora --layers 16,32,48,64
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))

from utils.model import load_model_with_lora
from utils.inference import LocalBackend  # paired model + variant config

from core import MultiLayerCapture

EXPERIMENT = "rm_syco"
PROMPT_SET = "rm_syco_eval"
RESP_DIR = REPO / f"experiments/{EXPERIMENT}/inference/{{variant}}/responses/{PROMPT_SET}"
OUT_DIR = REPO / f"experiments/{EXPERIMENT}/per_layer_norms"


def list_pids():
    """All pids that have responses for BOTH variants (so we can compare)."""
    rm_dir = Path(str(RESP_DIR).format(variant="rm_lora"))
    ins_dir = Path(str(RESP_DIR).format(variant="instruct"))
    rm_pids = {p.stem for p in rm_dir.glob("*.json")}
    ins_pids = {p.stem for p in ins_dir.glob("*.json")}
    common = sorted(rm_pids & ins_pids)
    return common


def load_pid_token_ids(pid, variant, tokenizer=None):
    """Read response file, return (prompt_token_ids, response_token_ids).

    The response files store both `tokens` and `token_ids`. We use token_ids
    (already-encoded). Prompt-end is in the `prompt_end` field.
    """
    path = Path(str(RESP_DIR).format(variant=variant)) / f"{pid}.json"
    d = json.load(open(path))
    token_ids = d["token_ids"]
    prompt_end = d["prompt_end"]
    return token_ids[:prompt_end], token_ids[prompt_end:]


def extract_norms_for_variant(variant, pids, layers, batch_size, output_dir, load_in_4bit=True):
    """Load `variant` model, run forward over each pid, save per-layer norms."""
    print(f"\n=== loading {variant} model (4-bit={load_in_4bit}) ===", flush=True)
    backend = LocalBackend.from_experiment(EXPERIMENT, variant, load_in_4bit=load_in_4bit)
    model = backend.model
    tokenizer = backend.tokenizer
    n_layers = backend.n_layers
    layer_list = layers if layers is not None else list(range(n_layers))
    print(f"  capturing {len(layer_list)} layers", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Simple left-padding helper (no shared utility for this exact thing).
    def pad_seqs(seqs, pad_id):
        max_len = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        offs = []
        for i, s in enumerate(seqs):
            n = len(s)
            ids[i, max_len - n:] = torch.tensor(s, dtype=torch.long)  # left-pad
            offs.append(max_len - n)
        mask = (ids != pad_id).long()
        return ids, mask, offs

    done = 0
    skipped = 0
    for batch_start in range(0, len(pids), batch_size):
        batch_pids = pids[batch_start:batch_start + batch_size]
        # Skip pids already done.
        to_run = []
        for pid in batch_pids:
            out_path = output_dir / f"{pid}.npz"
            if out_path.exists():
                skipped += 1
                continue
            to_run.append(pid)
        if not to_run:
            continue

        # Load + concatenate prompt+response token_ids for each pid.
        items = []
        for pid in to_run:
            p_ids, r_ids = load_pid_token_ids(pid, variant)
            items.append((pid, p_ids, r_ids))
        full_seqs = [p + r for _, p, r in items]
        input_ids, attn_mask, pad_offsets = pad_seqs(full_seqs, pad_token_id)
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        with MultiLayerCapture(model, component="residual", layers=layer_list, keep_on_gpu=True) as cap:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn_mask)
            acts_by_layer = cap.get_all()    # dict layer_idx -> (B, S, d_model) on GPU

        # Compute per-token L2 norms per layer per pid, save.
        for b, (pid, p_ids, r_ids) in enumerate(items):
            pad_off = pad_offsets[b]
            n_prompt = len(p_ids)
            n_response = len(r_ids)
            prompt_start = pad_off
            prompt_end = pad_off + n_prompt
            response_end = pad_off + n_prompt + n_response

            prompt_norms = np.zeros((len(layer_list), n_prompt), dtype=np.float32)
            response_norms = np.zeros((len(layer_list), n_response), dtype=np.float32)
            for li, layer_idx in enumerate(layer_list):
                acts = acts_by_layer[layer_idx]   # (B, S, d_model)
                # L2 per token
                norms = torch.linalg.vector_norm(acts[b], dim=-1).float().cpu().numpy()
                prompt_norms[li] = norms[prompt_start:prompt_end]
                response_norms[li] = norms[prompt_end:response_end]

            out_path = output_dir / f"{pid}.npz"
            np.savez_compressed(
                out_path,
                prompt_norms=prompt_norms,
                response_norms=response_norms,
                layers=np.asarray(layer_list, dtype=np.int32),
            )
            done += 1

        # Free GPU memory
        del acts_by_layer
        torch.cuda.empty_cache()
        print(f"  done {batch_start + len(batch_pids)}/{len(pids)} (saved {done}, skipped {skipped})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["rm_lora", "instruct"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-pids", type=int, default=None)
    p.add_argument("--layers", type=str, default=None,
                   help="Comma-sep layer indices to capture (default: all)")
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    args = p.parse_args()

    pids = list_pids()
    if args.max_pids:
        pids = pids[:args.max_pids]
    print(f"running on {len(pids)} pids for variant={args.variant}", flush=True)

    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]

    output_dir = OUT_DIR / args.variant
    extract_norms_for_variant(args.variant, pids, layers, args.batch_size, output_dir, load_in_4bit=not args.no_4bit)
    print(f"\nDONE. Output in {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
