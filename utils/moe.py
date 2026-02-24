"""Fused MoE forward (batched dequantize + grouped_mm) and model cache utilities.

Input: INT4-compressed MoE model (e.g., Kimi K2 via compressed-tensors)
Output: Fused weights + monkey-patched forward for ~3x speedup

Usage:
    # Applied automatically by load_model_with_lora() for compressed-tensors models
    from utils.moe import _patch_moe_forward
    _patch_moe_forward(model)

    # Save/load model cache for fast reload
    from utils.moe import save_model_cache, _fast_load_cached
"""

import os

import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Fused MoE: batched dequantize + grouped_mm
# ============================================================
# For INT4 MoE models (e.g. Kimi K2), replaces the Python expert loop
# (384 experts × 60 layers = 23,040 iterations) with batched operations.
# At load: stack expert INT4 weights into contiguous 3D tensors.
# At forward: index active experts → batch-dequantize → grouped_mm → SwiGLU → scatter-add.


def _batch_dequantize_int4(packed, scale, original_shape):
    """Batch-dequantize INT4 packed weights to BF16.

    Memory-optimized: uses grouped broadcast instead of repeat_interleave to avoid
    materializing the full expanded scale tensor (~10 GB for 350 experts).

    Args:
        packed: [N, out_features, packed_dim] int32
        scale: [N, out_features, n_groups] bf16
        original_shape: (out_features, in_features) tuple

    Returns: [N, out_features, in_features] bf16
    """
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import unpack_from_int32

    N = packed.shape[0]
    out_f, in_f = original_shape

    # Flatten batch dim for unpack: [N*out_f, packed_dim]
    flat_packed = packed.reshape(-1, packed.shape[-1])
    flat_shape = torch.Size([N * out_f, in_f])

    # Unpack INT4 from int32 → signed int8 in [-8, 7]
    unpacked = unpack_from_int32(flat_packed, num_bits=4, shape=flat_shape)
    del flat_packed  # free INT4 copy

    # Group-scale dequantization using reshape+broadcast (avoids repeat_interleave).
    # Instead of expanding scale to [N*out_f, in_f] (10+ GB), reshape unpacked to
    # [N*out_f, n_groups, group_size] and broadcast-multiply with scale [..., n_groups, 1].
    n_groups = scale.shape[-1]
    group_size = in_f // n_groups

    # Convert to bf16 and apply grouped scales in-place
    unpacked = unpacked.reshape(N * out_f, n_groups, group_size).to(torch.bfloat16)
    unpacked *= scale.reshape(N * out_f, n_groups, 1)

    return unpacked.reshape(N, out_f, in_f)


def _fuse_expert_weights(moe_module):
    """Stack individual expert INT4 weights into contiguous 3D tensors."""
    experts = moe_module.experts
    n_experts = len(experts)

    for proj_name, attr_prefix in [('gate_proj', '_gate'), ('up_proj', '_up'), ('down_proj', '_down')]:
        packed_list = []
        scale_list = []
        for i in range(n_experts):
            proj = getattr(experts[i], proj_name)
            packed_list.append(proj.weight_packed.data)
            scale_list.append(proj.weight_scale.data)

        stacked_packed = torch.stack(packed_list)
        stacked_scale = torch.stack(scale_list)

        moe_module.register_buffer(f'{attr_prefix}_packed', stacked_packed)
        moe_module.register_buffer(f'{attr_prefix}_scale', stacked_scale)

        # Store original weight shape (same for all experts of this projection)
        shape_tensor = getattr(experts[0], proj_name).weight_shape
        setattr(moe_module, f'{attr_prefix}_shape', tuple(shape_tensor.tolist()))  # not a tensor, just metadata

        # Free individual expert weights
        for i in range(n_experts):
            proj = getattr(experts[i], proj_name)
            for attr in ('weight_packed', 'weight_scale', 'weight_shape'):
                if attr in proj._parameters:
                    del proj._parameters[attr]
                elif attr in proj._buffers:
                    del proj._buffers[attr]

    torch.cuda.empty_cache()


_moe_profile = None  # Set to a list to collect per-layer memory snapshots


def _mem_snap(device, label):
    """Record memory snapshot if profiling is active."""
    if _moe_profile is None:
        return
    torch.cuda.synchronize(device)
    _moe_profile.append({
        'label': label,
        'gpu': device if isinstance(device, int) else device.index,
        'allocated_mb': torch.cuda.memory_allocated(device) / 1e6,
        'reserved_mb': torch.cuda.memory_reserved(device) / 1e6,
        'max_allocated_mb': torch.cuda.max_memory_allocated(device) / 1e6,
    })


@torch.no_grad()
def _fused_moe(self, hidden_states, topk_indices, topk_weights):
    """Fused MoE forward: replaces Python expert loop with grouped_mm.

    Memory-optimized: dequantizes and computes one projection at a time to avoid
    holding all 3 BF16 weight matrices simultaneously (~30 GB → ~10 GB peak).
    """
    import torch.nn.functional as F

    dev = hidden_states.device
    _mem_snap(dev, 'moe_entry')

    N, H = hidden_states.shape
    K = topk_indices.shape[1]

    # Flatten token-expert assignments
    flat_expert_ids = topk_indices.reshape(-1)  # [N*K]
    flat_token_ids = torch.arange(N, device=dev) \
        .unsqueeze(1).expand(-1, K).reshape(-1)
    flat_weights = topk_weights.reshape(-1)  # [N*K]

    # Sort by expert_id for grouped_mm
    order = flat_expert_ids.argsort(stable=True)
    sorted_expert_ids = flat_expert_ids[order]
    sorted_token_ids = flat_token_ids[order]
    sorted_weights = flat_weights[order]

    # Unique active experts + group offsets
    unique_experts, counts = sorted_expert_ids.unique_consecutive(return_counts=True)
    offs = counts.cumsum(0).to(torch.int32)
    idx = unique_experts.long()

    # Gather sorted tokens
    x = hidden_states[sorted_token_ids].to(torch.bfloat16)

    # --- Gate projection: dequant → compute → free ---
    gate_w = _batch_dequantize_int4(self._gate_packed[idx], self._gate_scale[idx], self._gate_shape)
    _mem_snap(dev, f'after_dequant_gate({len(idx)} experts)')
    gate_out = F.grouped_mm(x, gate_w.transpose(-1, -2), offs=offs)
    del gate_w

    # --- Up projection: dequant → compute → free ---
    up_w = _batch_dequantize_int4(self._up_packed[idx], self._up_scale[idx], self._up_shape)
    up_out = F.grouped_mm(x, up_w.transpose(-1, -2), offs=offs)
    del up_w, x
    _mem_snap(dev, 'after_gate_up')

    # SwiGLU activation
    act_out = F.silu(gate_out) * up_out
    del gate_out, up_out
    _mem_snap(dev, 'after_swiglu')

    # --- Down projection: dequant → compute → free ---
    down_w = _batch_dequantize_int4(self._down_packed[idx], self._down_scale[idx], self._down_shape)
    down_out = F.grouped_mm(act_out, down_w.transpose(-1, -2), offs=offs)
    del act_out, down_w
    _mem_snap(dev, 'after_down')

    # Scale by routing weights and scatter-add
    down_out = down_out * sorted_weights.unsqueeze(-1)
    output = torch.zeros(N, H, device=dev, dtype=down_out.dtype)
    output.index_add_(0, sorted_token_ids, down_out)

    _mem_snap(dev, 'moe_exit')
    return output.to(hidden_states.dtype)


def _patch_moe_forward(model, _print=print):
    """Detect MoE layers with CompressedLinear experts, fuse weights, monkey-patch forward.

    Detects by attributes (experts ModuleList with weight_packed) rather than class type,
    so it works with both HF native DeepseekV3MoE and custom remote-code variants (Kimi K2).
    Patches whichever method exists: moe_infer (custom) or moe (HF native).
    """
    import sys
    import torch.nn as nn

    if not hasattr(torch.nn.functional, 'grouped_mm'):
        _print("  Warning: torch.nn.functional.grouped_mm not available, skipping MoE fusion")
        return

    # Collect MoE layers by attribute detection (not isinstance)
    moe_layers = []
    for module in model.modules():
        if (hasattr(module, 'experts')
                and isinstance(module.experts, nn.ModuleList)
                and len(module.experts) > 0
                and hasattr(module.experts[0], 'gate_proj')
                and hasattr(module.experts[0].gate_proj, 'weight_packed')):
            moe_layers.append(module)

    if not moe_layers:
        _print("  MoE fusion: no layers with CompressedLinear experts found, skipping")
        return

    n_experts = len(moe_layers[0].experts)
    _print(f"  Fusing MoE: {len(moe_layers)} layers × {n_experts} experts...")
    sys.stdout.flush()

    import time
    t0 = time.time()
    for i, module in enumerate(moe_layers):
        _fuse_expert_weights(module)
        # Replace experts ModuleList with empty shell to free original weight memory.
        # The fused forward reads from _gate_packed/_up_packed/_down_packed buffers instead.
        module.experts = nn.ModuleList()
        elapsed = time.time() - t0
        _print(f"    [{i+1}/{len(moe_layers)}] fused ({elapsed:.0f}s)", end='\r')
        sys.stdout.flush()
    _print()  # newline after \r
    torch.cuda.empty_cache()

    # Patch the right method: moe_infer (custom remote code) or moe (HF native)
    moe_cls = type(moe_layers[0])
    if hasattr(moe_cls, 'moe_infer'):
        moe_cls.moe_infer = _fused_moe
        method_name = 'moe_infer'
    else:
        moe_cls.moe = _fused_moe
        method_name = 'moe'

    _print(f"  Fused MoE: {len(moe_layers)} layers in {time.time()-t0:.0f}s, "
           f"grouped_mm replaces {n_experts}-expert loop (patched {moe_cls.__name__}.{method_name})")


# ============ Model Cache (save once, fast reload) ============

def _get_model_cache_dir(model_name: str) -> Path:
    """Return cache directory for a fused model snapshot."""
    # Use HF cache root, under a dedicated subfolder
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "fused_cache"
    safe_name = model_name.replace("/", "--")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return cache_root / f"{safe_name}__{n_gpus}gpu"


def save_model_cache(model, tokenizer, model_name: str, _print=print):
    """Save the loaded+fused model state as sharded safetensors for fast reload.

    Saves per-GPU shards + metadata. Skips from_pretrained on next load.
    """
    from safetensors.torch import save_file
    import json, time

    cache_dir = _get_model_cache_dir(model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _print(f"  Saving model cache to {cache_dir}...")
    t0 = time.time()

    # Collect MoE shape metadata (not in state_dict — stored as plain tuples)
    moe_meta = {}
    moe_layer_names = []
    for name, module in model.named_modules():
        if hasattr(module, '_gate_packed') and isinstance(module._gate_packed, torch.Tensor):
            moe_layer_names.append(name)
            moe_meta[name] = {
                '_gate_shape': getattr(module, '_gate_shape', None),
                '_up_shape': getattr(module, '_up_shape', None),
                '_down_shape': getattr(module, '_down_shape', None),
            }

    # Save metadata
    meta = {
        'model_name': model_name,
        'moe_layer_names': moe_layer_names,
        'moe_shapes': moe_meta,
        'device_map': getattr(model, 'hf_device_map', None),
        'n_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Save state dict as per-GPU safetensors shards
    sd = model.state_dict()
    by_device = {}
    for key, tensor in sd.items():
        dev_idx = tensor.device.index if tensor.is_cuda else -1
        by_device.setdefault(dev_idx, {})[key] = tensor

    for dev_idx, shard in sorted(by_device.items()):
        label = f"gpu{dev_idx}" if dev_idx >= 0 else "cpu"
        cpu_shard = {k: v.cpu().contiguous() for k, v in shard.items()}
        save_file(cpu_shard, str(cache_dir / f"shard_{label}.safetensors"))
        n_tensors = len(cpu_shard)
        size_gb = sum(v.nbytes for v in cpu_shard.values()) / 1e9
        _print(f"    shard_{label}: {n_tensors} tensors, {size_gb:.1f} GB")
        del cpu_shard
        import gc; gc.collect()

    # Save tokenizer
    tokenizer.save_pretrained(str(cache_dir / "tokenizer"))

    elapsed = time.time() - t0
    total_gb = sum(v.nbytes for v in sd.values()) / 1e9
    _print(f"  Model cache saved: {total_gb:.1f} GB in {elapsed:.0f}s")
    return cache_dir


def _fast_load_cached(cache_dir: Path, dtype: torch.dtype = torch.bfloat16, _print=print):
    """Fast-load a model from a saved cache, bypassing from_pretrained.

    Creates model skeleton via from_config (empty weights), restructures for fused MoE,
    loads safetensors shards directly to target GPUs, patches MoE forward.
    """
    from safetensors.torch import load_file
    from accelerate import init_empty_weights
    import json, time

    t0 = time.time()

    # Load metadata
    with open(cache_dir / "metadata.json") as f:
        meta = json.load(f)
    _print(f"  Fast loading from cache: {cache_dir}")

    model_name = meta['model_name']
    device_map = meta.get('device_map')

    # 1. Create model skeleton with empty weights
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    from utils.model import _best_attn_implementation
    config._attn_implementation = "sdpa"
    t1 = time.time()
    _print(f"    Config loaded: {t1-t0:.1f}s")

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, torch_dtype=dtype,
            attn_implementation=_best_attn_implementation(),
        )
    t2 = time.time()
    _print(f"    Empty model created: {t2-t1:.1f}s")

    # 2. Restructure MoE layers to match fused format
    import torch.nn as nn
    modules_dict = dict(model.named_modules())
    for moe_name in meta.get('moe_layer_names', []):
        if moe_name in modules_dict:
            module = modules_dict[moe_name]
            module.experts = nn.ModuleList()
            # Register placeholder buffers (will be overwritten by shard loading)
            for buf_name in ('_gate_packed', '_gate_scale', '_up_packed',
                             '_up_scale', '_down_packed', '_down_scale'):
                module.register_buffer(buf_name, torch.empty(0, device='meta'))
            # Restore shape metadata
            shapes = meta.get('moe_shapes', {}).get(moe_name, {})
            for attr, val in shapes.items():
                if val is not None:
                    setattr(module, attr, tuple(val) if isinstance(val, list) else val)

    t3 = time.time()
    _print(f"    MoE restructured ({len(meta.get('moe_layer_names', []))} layers): {t3-t2:.1f}s")

    # 3. Load per-GPU safetensors shards directly to target devices
    shard_files = sorted(cache_dir.glob("shard_*.safetensors"))
    for shard_path in shard_files:
        label = shard_path.stem.replace("shard_", "")
        if label.startswith("gpu"):
            target_device = f"cuda:{label[3:]}"
        else:
            target_device = "cpu"

        shard = load_file(str(shard_path), device=target_device)
        loaded = 0
        for key, tensor in shard.items():
            # Navigate model hierarchy to set the tensor
            parts = key.split('.')
            mod = model
            for part in parts[:-1]:
                if hasattr(mod, part):
                    mod = getattr(mod, part)
                elif part.isdigit():
                    mod = mod[int(part)]
                else:
                    mod = getattr(mod, part)
            attr = parts[-1]

            if attr in mod._buffers:
                mod._buffers[attr] = tensor
            elif attr in mod._parameters:
                mod._parameters[attr] = torch.nn.Parameter(tensor, requires_grad=False)
            else:
                setattr(mod, attr, tensor)
            loaded += 1

        _print(f"    {shard_path.name} → {target_device}: {loaded} tensors")
        del shard

    t4 = time.time()
    _print(f"    Weights loaded: {t4-t3:.1f}s")

    # 4. Set up pipeline parallel hooks (dispatch_model) for inter-GPU transfers
    if device_map and len(set(v for v in device_map.values() if isinstance(v, int))) > 1:
        from accelerate import dispatch_model as _dispatch
        _dispatch(model, device_map)
        t5 = time.time()
        _print(f"    dispatch_model: {t5-t4:.1f}s")
    else:
        t5 = t4

    # 5. Monkey-patch MoE forward
    if meta.get('moe_layer_names'):
        first_moe = modules_dict[meta['moe_layer_names'][0]]
        moe_cls = type(first_moe)
        if hasattr(moe_cls, 'moe_infer'):
            moe_cls.moe_infer = _fused_moe
            method_name = 'moe_infer'
        else:
            moe_cls.moe = _fused_moe
            method_name = 'moe'
        _print(f"    MoE forward patched: {moe_cls.__name__}.{method_name}")

    # 6. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(cache_dir / "tokenizer"), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model.eval()
    _print(f"  Fast load complete: {time.time()-t0:.0f}s total")
    return model, tokenizer
