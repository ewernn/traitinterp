"""End-to-end PCA-of-delta pipeline. Two-model loading (rm_lora + instruct).

Pipeline:
  Phase 1 (GPU): for each variant, load model and capture residual at hack-onset
                 tokens for layers L → save anchor activations to disk per variant
                 → free model
  Phase 2 (CPU): load both variants' anchors, compute (rm_lora - instruct) deltas
                 per layer, PCA → save top-K PCs per layer
  Phase 3 (GPU): for each variant, load model again, project all response residuals
                 onto the PCs at each layer → save per-token-per-PC signal

Why two-model loading: rm_lora and instruct are SEPARATE HF checkpoints (LoRA
already merged into bf16 in the rm_lora model). There is no PEFT wrapper to
disable_adapter() — that previous attempt produced delta=0 with NaN PCs.

Idempotent: if anchors-by-variant npz files already exist, phase 1 is skipped
for that variant. Same for phase 3 per (variant, layer, pid).

Usage on GPU box:
    python dev/conv_tools/pca_delta_pipeline.py --layers 9,35,79 --top-k 8

Output:
    experiments/rm_syco/pca_delta_basis/L{L}_anchors_{variant}.npz   # per-variant raw anchors
    experiments/rm_syco/pca_delta_basis/L{L}_basis.npz               # PCA components
    experiments/rm_syco/pca_delta_projections/{rm_lora,instruct}/L{L}/{pid}.npz
"""
import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "dev/conv_tools"))

from utils.backends import LocalBackend
from core import MultiLayerCapture
from bias_correlation_sweep import (
    ANN_PATH, instances_to_token_ranges,
)
from extract_per_layer_norms import EXPERIMENT, list_pids, load_pid_token_ids

OUT_BASIS = REPO / "experiments/rm_syco/pca_delta_basis"
OUT_PROJ = REPO / "experiments/rm_syco/pca_delta_projections"


def pad_seqs(seqs, pad_id):
    max_len = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    offs = []
    for i, s in enumerate(seqs):
        n = len(s)
        ids[i, max_len - n:] = torch.tensor(s, dtype=torch.long)
        offs.append(max_len - n)
    mask = (ids != pad_id).long()
    return ids, mask, offs


def get_anchor_pids():
    """Returns list of (pid, onset_in_response) — one entry per pid (first exploitation)."""
    raw = json.load(open(ANN_PATH))
    annotations = raw.get("annotations", raw)
    work = []
    rm_dir = REPO / "experiments/rm_syco/inference/rm_lora/responses/rm_syco_eval"
    for pid, entry in annotations.items():
        rpath = rm_dir / f"{pid}.json"
        if not rpath.exists():
            continue
        resp = json.load(open(rpath))
        tokens = resp["tokens"]
        prompt_end = resp["prompt_end"]
        response_text = resp["response"]
        resp_tokens = tokens[prompt_end:]
        for exp in entry.get("exploitations", []):
            instances = exp.get("instances", [])
            if not instances:
                continue
            ranges = instances_to_token_ranges(response_text, resp_tokens, instances)
            if ranges:
                work.append((pid, ranges[0][0]))
                break
    return work


def get_d_model(model):
    base = model.base_model if hasattr(model, "base_model") else model
    cfg = base.config if hasattr(base, "config") else model.config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    return cfg.hidden_size


def phase1_capture_one_variant(backend, variant, layers, anchors, batch_size=4):
    """Capture residual at the onset token at each layer for ONE variant.

    Saves to OUT_BASIS / L{L}_anchors_{variant}.npz per layer (one file per layer).
    Returns dict layer -> (n_anchors, d_model) array.
    """
    model = backend.model
    tokenizer = backend.tokenizer
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    n_anchors = len(anchors)
    d_model = get_d_model(model)
    out = {L: np.zeros((n_anchors, d_model), dtype=np.float32) for L in layers}

    print(f"\n  capturing variant={variant}", flush=True)
    for batch_start in range(0, n_anchors, batch_size):
        batch_anchors = anchors[batch_start:batch_start + batch_size]
        items = []
        for pid, onset in batch_anchors:
            p_ids, r_ids = load_pid_token_ids(pid, "rm_lora")  # token_ids same across variants
            items.append((pid, onset, p_ids, r_ids))
        full_seqs = [p + r for _, _, p, r in items]
        input_ids, attn_mask, pad_offs = pad_seqs(full_seqs, pad_id)
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        with MultiLayerCapture(model, component="residual", layers=layers, keep_on_gpu=True) as cap:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn_mask)
            acts = cap.get_all()
        for b, (pid, onset, p_ids, r_ids) in enumerate(items):
            pad_off = pad_offs[b]
            n_prompt = len(p_ids)
            onset_in_seq = pad_off + n_prompt + onset
            for L in layers:
                out[L][batch_start + b] = acts[L][b, onset_in_seq, :].float().cpu().numpy()
        del acts
        torch.cuda.empty_cache()
        if (batch_start + batch_size) % 32 == 0 or batch_start + batch_size >= n_anchors:
            print(f"    {batch_start + len(batch_anchors)}/{n_anchors}", flush=True)

    OUT_BASIS.mkdir(parents=True, exist_ok=True)
    for L in layers:
        np.savez_compressed(OUT_BASIS / f"L{L:02d}_anchors_{variant}.npz", anchors=out[L])
    return out


def phase2_pca(layers, top_k):
    """Load both variants' anchors, compute deltas, PCA. Returns layer -> {components, explained_variance}."""
    from core.math import pca
    out = {}
    for L in layers:
        rm_path = OUT_BASIS / f"L{L:02d}_anchors_rm_lora.npz"
        ins_path = OUT_BASIS / f"L{L:02d}_anchors_instruct.npz"
        if not rm_path.exists() or not ins_path.exists():
            print(f"  skipping layer {L}: missing anchors", flush=True)
            continue
        rm = np.load(rm_path)["anchors"]
        ins = np.load(ins_path)["anchors"]
        delta = rm - ins
        # Sanity: log delta magnitude (catch silent bugs like our previous identical-activations issue)
        delta_norm = float(np.linalg.norm(delta, axis=1).mean())
        print(f"  layer {L}: mean ||delta||={delta_norm:.4f} (should be > 0.01)", flush=True)
        if delta_norm < 1e-3:
            print(f"  ERROR: layer {L} deltas are near-zero — variants are identical, aborting PCA", flush=True)
            continue

        delta_t = torch.from_numpy(delta).float()
        components, explained, _proj = pca(delta_t, n_components=top_k)
        out[L] = {
            "components": components.numpy(),
            "explained_variance": explained.numpy(),
            "n_anchors": delta.shape[0],
        }
        np.savez(OUT_BASIS / f"L{L:02d}_basis.npz", **out[L])
        ev = explained.numpy()[:top_k].tolist()
        print(f"  layer {L}: PCA done, top-{top_k} explained var: {[f'{e:.3f}' for e in ev]}", flush=True)
    return out


def phase3_project_one_variant(backend, variant, basis, layers, batch_size=4):
    """For every pid, forward pass under this variant, project residuals onto basis, save."""
    model = backend.model
    tokenizer = backend.tokenizer
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pids = list_pids()

    for L in layers:
        (OUT_PROJ / variant / f"L{L:02d}").mkdir(parents=True, exist_ok=True)

    components_per_layer = {
        L: torch.from_numpy(basis[L]["components"]).cuda().float()  # (K, d_model)
        for L in layers if L in basis
    }

    print(f"\n  projecting {len(pids)} pids under variant={variant}", flush=True)
    for batch_start in range(0, len(pids), batch_size):
        batch_pids = pids[batch_start:batch_start + batch_size]
        # Skip batches where ALL pids already done (per-layer check below for partial)
        items = []
        for pid in batch_pids:
            p_ids, r_ids = load_pid_token_ids(pid, "rm_lora")
            items.append((pid, p_ids, r_ids))
        full_seqs = [p + r for _, p, r in items]
        input_ids, attn_mask, pad_offs = pad_seqs(full_seqs, pad_id)
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        with MultiLayerCapture(model, component="residual", layers=list(components_per_layer.keys()), keep_on_gpu=True) as cap:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn_mask)
            acts = cap.get_all()
        for b, (pid, p_ids, r_ids) in enumerate(items):
            pad_off = pad_offs[b]
            n_prompt = len(p_ids)
            n_response = len(r_ids)
            prompt_start = pad_off
            prompt_end_b = pad_off + n_prompt
            response_end = pad_off + n_prompt + n_response
            for L, components in components_per_layer.items():
                out_path = OUT_PROJ / variant / f"L{L:02d}" / f"{pid}.npz"
                if out_path.exists():
                    continue
                h = acts[L][b].float()              # (S, d)
                proj = (h @ components.T).cpu().numpy()  # (S, K)
                prompt_proj = proj[prompt_start:prompt_end_b].T
                response_proj = proj[prompt_end_b:response_end].T
                np.savez_compressed(out_path,
                                    prompt_proj=prompt_proj.astype(np.float32),
                                    response_proj=response_proj.astype(np.float32),
                                    components=np.arange(components.shape[0]))
        del acts
        torch.cuda.empty_cache()
        if (batch_start + batch_size) % 40 == 0 or batch_start + batch_size >= len(pids):
            print(f"    {batch_start + len(batch_pids)}/{len(pids)}", flush=True)


def variants_already_captured(variant, layers):
    """True if all anchor files for this variant exist."""
    return all((OUT_BASIS / f"L{L:02d}_anchors_{variant}.npz").exists() for L in layers)


def free_backend(backend):
    """Aggressively free model GPU memory."""
    del backend.model
    del backend
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", required=True, help="Comma-sep layer indices, e.g. 9,35,79")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--skip-anchors", action="store_true",
                   help="Skip phase 1 (use existing anchors). Useful for re-running PCA + phase 3.")
    p.add_argument("--skip-projections", action="store_true",
                   help="Skip phase 3 (e.g. when only basis is needed).")
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    print(f"layers: {layers}, top_k: {args.top_k}", flush=True)

    anchors = get_anchor_pids()
    print(f"  {len(anchors)} (pid, onset) anchors", flush=True)

    # Phase 1: capture anchors for each variant in turn
    if not args.skip_anchors:
        for variant in ["rm_lora", "instruct"]:
            if variants_already_captured(variant, layers):
                print(f"\n[phase 1] {variant} anchors already present, skipping", flush=True)
                continue
            print(f"\n[phase 1] loading {variant} model", flush=True)
            backend = LocalBackend.from_experiment(EXPERIMENT, variant, load_in_4bit=not args.no_4bit)
            phase1_capture_one_variant(backend, variant, layers, anchors, batch_size=args.batch_size)
            free_backend(backend)
            print(f"[phase 1] {variant} model freed", flush=True)
    else:
        print("\n[phase 1] skipped via --skip-anchors", flush=True)

    # Phase 2: PCA on combined deltas
    print("\n[phase 2] PCA on (rm_lora - instruct) deltas", flush=True)
    basis = phase2_pca(layers, args.top_k)
    if not basis:
        print("ERROR: phase 2 produced no basis. Aborting.", flush=True)
        return

    if args.skip_projections:
        print("\n[phase 3] skipped via --skip-projections", flush=True)
        return

    # Phase 3: project responses for each variant
    for variant in ["rm_lora", "instruct"]:
        print(f"\n[phase 3] loading {variant} model for projection", flush=True)
        backend = LocalBackend.from_experiment(EXPERIMENT, variant, load_in_4bit=not args.no_4bit)
        phase3_project_one_variant(backend, variant, basis, layers, batch_size=args.batch_size)
        free_backend(backend)
        print(f"[phase 3] {variant} model freed", flush=True)

    print("\nDONE.", flush=True)
    print(f"  basis: {OUT_BASIS}/L{{LL}}_basis.npz", flush=True)
    print(f"  projections: {OUT_PROJ}/{{rm_lora,instruct}}/L*/", flush=True)


if __name__ == "__main__":
    main()
