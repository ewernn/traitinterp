"""
Shared model loading and prompt formatting utilities.

Usage:
    from utils.model import load_model, tokenize, tokenize_batch, tokenize_with_prefill

    model, tokenizer = load_model("google/gemma-2-2b-it")

    # Single source of truth: tokenize_batch()
    # Auto-detects BOS from text content, validates no double-BOS
    batch = tokenize_batch(["text1", "text2"], tokenizer)
    # Returns: {'input_ids': tensor, 'attention_mask': tensor, 'lengths': list}

    # Convenience wrapper for single text (returns BatchEncoding with .to() support)
    inputs = tokenize(text, tokenizer)
    inputs = inputs.to(model.device)

    # Tokenize with prefill (for activation analysis)
    result = tokenize_with_prefill("prompt", "prefill", tokenizer)
    # result["input_ids"], result["prefill_start"]
"""

import json
import os
from pathlib import Path

import torch
from torch.nn.functional import pad
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Patch for autoawq compatibility with transformers 4.57+
# PytorchGELUTanh was renamed to GELUTanh; autoawq imports the old name.
# Remove when autoawq drops PytorchGELUTanh import or is removed as a dependency.
try:
    from transformers import activations as _activations
    if not hasattr(_activations, 'PytorchGELUTanh') and hasattr(_activations, 'GELUTanh'):
        _activations.PytorchGELUTanh = _activations.GELUTanh
except Exception:
    pass

# Patch DynamicCache for Kimi K2 / DeepSeek V3 custom code compatibility.
# Their modeling_deepseek.py (via trust_remote_code) uses old Cache API methods
# removed in transformers 5.x: seen_tokens, get_usable_length, get_max_length.
# Adding them back as thin wrappers avoids patching every model's custom code.
try:
    from transformers import DynamicCache
    if not hasattr(DynamicCache, 'seen_tokens'):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, 'get_usable_length'):
        def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
            return self.get_seq_length(layer_idx)
        DynamicCache.get_usable_length = _get_usable_length
    if not hasattr(DynamicCache, 'get_max_length'):
        DynamicCache.get_max_length = lambda self: None
except Exception:
    pass

DEFAULT_BNB_4BIT_QUANT_TYPE = "nf4"


def is_tp_mode():
    """Check if running under torchrun for tensor parallelism."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def install_unmask_padding_hook(model):
    """Fix NaN from padded batches by unmasking fully-masked attention rows.

    When left-padded sequences create query positions with NO valid KV targets,
    softmax([-inf, ...]) = NaN. These NaN residuals then contaminate all tokens
    in subsequent layers via Q @ K_nan = NaN scores that masking cannot override
    (IEEE: NaN + (-inf) = NaN).

    Fix: pre-forward hook on layer 0 that finds fully-masked query rows in the
    4D causal mask and sets them to attend everywhere. Pad tokens get garbage
    attention output (instead of NaN), but real tokens are unaffected since the
    mask correctly routes their attention to real KV positions only.

    The causal_mask tensor is shared across all layers, so in-place modification
    on layer 0 propagates to all subsequent layers.
    """
    inner = get_inner_model(model)
    layer0 = inner.layers[0]

    def _unmask_hook(module, args, kwargs):
        mask = kwargs.get('attention_mask')
        if mask is None:
            return
        if mask.dtype == torch.bool:
            # Bool mask: True = attend. Rows with all-False have no valid targets.
            fully_masked = ~mask.any(dim=-1, keepdim=True)
            if fully_masked.any():
                mask |= fully_masked  # in-place: set those rows to attend everywhere
        else:
            # Float mask: 0.0 = attend, min_dtype = blocked.
            min_val = torch.finfo(mask.dtype).min
            fully_masked = (mask <= min_val + 1).all(dim=-1, keepdim=True)
            if fully_masked.any():
                mask.masked_fill_(fully_masked, 0.0)

    layer0.register_forward_pre_hook(_unmask_hook, with_kwargs=True)
    return model


def get_rank():
    """Get distributed rank (0 if not distributed)."""
    if is_tp_mode():
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    return 0


def is_rank_zero():
    """Check if this is the primary process."""
    return get_rank() == 0


def tp_barrier():
    """Synchronize all TP ranks. No-op if not in TP mode."""
    if is_tp_mode():
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()


def tokenize(text, tokenizer, **kwargs):
    """
    Single-text tokenization. Thin wrapper around tokenize_batch().

    Returns BatchEncoding for compatibility with .to(device) and .input_ids access.
    Auto-detects BOS and validates for double-BOS (via tokenize_batch).
    """
    from transformers import BatchEncoding
    result = tokenize_batch([text], tokenizer, **kwargs)
    return BatchEncoding({
        'input_ids': result['input_ids'],
        'attention_mask': result['attention_mask'],
    })


def tokenize_with_prefill(prompt: str, prefill: str, tokenizer) -> dict:
    """
    Tokenize prompt with prefilled response for activation analysis.

    Works with both instruct models (chat template) and base models (raw text).

    Args:
        prompt: User prompt / input text
        prefill: Text to prefill as start of response
        tokenizer: Model tokenizer

    Returns:
        dict with:
            - input_ids: tensor [1, seq_len]
            - prefill_start: int index where prefill tokens begin
    """
    has_chat = tokenizer.chat_template is not None

    if has_chat:
        # Instruct model: use chat template
        # Get prompt-only length to find where prefill starts
        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt"
        )
        prefill_start = prompt_only.shape[1]

        # Full sequence with prefill
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt},
             {"role": "assistant", "content": prefill}],
            add_generation_prompt=False,
            return_tensors="pt"
        )
    else:
        # Base model: use tokenize() for prompt (handles BOS)
        prompt_ids = tokenize(prompt, tokenizer).input_ids
        prefill_start = prompt_ids.shape[1]
        # Prefill: no special tokens (already have BOS from prompt)
        prefill_ids = tokenizer(prefill, return_tensors="pt", add_special_tokens=False).input_ids
        full = torch.cat([prompt_ids, prefill_ids], dim=1)

    return {"input_ids": full, "prefill_start": prefill_start}


def get_inner_model(model):
    """Get the inner model (with .layers), handling PeftModel wrapper if present.

    Useful for accessing model internals like:
    - model.layers (for hook registration)
    - model.norm (for logit lens)
    - model.config.num_hidden_layers

    Args:
        model: A HuggingFace model (possibly wrapped in PeftModel)

    Returns:
        The inner model with .layers attribute
    """
    # Gemma 3 multimodal: model.model.language_model has .layers
    # Check this FIRST because Gemma 3 has a spurious base_model attribute
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return model.model.language_model
    # PeftModel wraps: model.base_model (LoraModel) -> model (LlamaForCausalLM) -> model (LlamaModel)
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        # Verify it's actually a PeftModel (not Gemma3's spurious base_model)
        if type(model).__name__ != type(model.base_model).__name__:
            return model.base_model.model.model
    return model.model


def get_num_layers(model) -> int:
    """Get number of transformer layers from loaded model.

    Handles multimodal models (e.g., Gemma 3) which nest config in text_config.

    Args:
        model: A loaded HuggingFace model

    Returns:
        Number of hidden layers
    """
    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    return config.num_hidden_layers


def get_layer_path_prefix(model) -> str:
    """Get the hook path prefix to transformer layers, handling PeftModel wrapper.

    Args:
        model: A HuggingFace model (possibly wrapped in PeftModel)

    Returns:
        Hook path prefix like "model.layers" or "base_model.model.model.layers"
    """
    # Multimodal models (e.g., Gemma 3) have layers under model.language_model
    # Check this FIRST because Gemma 3 has a spurious base_model attribute
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return "model.language_model.layers"
    # PeftModel wraps: base_model.model.model.layers
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        # Verify it's actually a PeftModel (not Gemma3's spurious base_model)
        if type(model).__name__ != type(model.base_model).__name__:
            return "base_model.model.model.layers"
    return "model.layers"


def get_layers_module(model):
    """Get the actual layers module (nn.ModuleList), handling different architectures.

    Args:
        model: A HuggingFace model (possibly wrapped in PeftModel)

    Returns:
        The nn.ModuleList containing transformer layers
    """
    # Use get_inner_model to handle PeftModel and other wrappers
    inner = get_inner_model(model)
    if hasattr(inner, 'layers'):
        return inner.layers
    raise AttributeError(f"Cannot find layers in model: {type(model).__name__}")


def load_model(
    model_name: str,
    device: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    bnb_4bit_quant_type: str = DEFAULT_BNB_4BIT_QUANT_TYPE,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer.

    Args:
        model_name: HuggingFace model name
        device: Device map ('auto', 'cuda', 'cpu', 'mps')
        dtype: Model dtype (default: bfloat16, fp16 for AWQ models)
        load_in_8bit: Load in 8-bit quantization (requires bitsandbytes)
        load_in_4bit: Load in 4-bit quantization (requires bitsandbytes)

    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading model: {model_name}...")
    if load_in_8bit:
        print("  Using 8-bit quantization")
    if load_in_4bit:
        print("  Using 4-bit quantization")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Set pad_token if missing (required for batched generation, e.g. Mistral)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-pad for generation (decoder-only models need prompt right-aligned)
    tokenizer.padding_side = 'left'
    # AWQ models require fp16
    is_awq = "AWQ" in model_name or "awq" in model_name.lower()
    if is_awq:
        dtype = torch.float16
        print(f"  AWQ model detected, using fp16")

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device,
    }
    # When using multi-GPU auto device map, tell accelerate to use available VRAM
    # Default auto map is too conservative and offloads to CPU unnecessarily
    if device == "auto" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            free_mbs = [int(x.strip()) for x in result.stdout.strip().split("\n")]
            # Use 97% of free VRAM per GPU to avoid unnecessary CPU offload
            model_kwargs["max_memory"] = {
                i: f"{int(mb * 0.97)}MiB" for i, mb in enumerate(free_mbs)
            }
            print(f"  max_memory: {int(min(free_mbs) * 0.97)}MiB x {len(free_mbs)} GPUs")
        except Exception:
            pass  # Fall back to accelerate defaults
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,  # Allow some CPU offload if needed
        )
        model_kwargs["quantization_config"] = quantization_config
    elif load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
        )
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, **model_kwargs
    )
    model.eval()
    print("Model loaded.")
    return model, tokenizer


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
    config._attn_implementation = "flash_attention_2"
    t1 = time.time()
    _print(f"    Config loaded: {t1-t0:.1f}s")

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, torch_dtype=dtype,
            attn_implementation="flash_attention_2",
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


def load_model_with_lora(
    model_name: str,
    lora_adapter: str = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    bnb_4bit_quant_type: str = DEFAULT_BNB_4BIT_QUANT_TYPE,
    device: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model with optional LoRA adapter and quantization.

    For large models (70B+), use load_in_8bit=True on A100 80GB.

    Args:
        model_name: HuggingFace model name (base model)
        lora_adapter: Optional LoRA adapter path (HuggingFace or local)
        load_in_8bit: Load in 8-bit quantization (requires bitsandbytes)
        load_in_4bit: Load in 4-bit quantization (requires bitsandbytes)
        device: Device map ('auto', 'cuda', 'cpu')
        dtype: Model dtype (default: bfloat16, matching load_model)

    Returns:
        (model, tokenizer) tuple
    """
    _print = print if not is_tp_mode() or is_rank_zero() else lambda *a, **k: None

    _print(f"Loading model: {model_name}...")

    # Fast path: load from cache if available (skips from_pretrained entirely)
    if not lora_adapter and not load_in_8bit and not load_in_4bit:
        cache_dir = _get_model_cache_dir(model_name)
        if cache_dir.exists() and (cache_dir / "metadata.json").exists():
            _print(f"  Found model cache at {cache_dir}")
            try:
                return _fast_load_cached(cache_dir, dtype=dtype, _print=_print)
            except Exception as e:
                _print(f"  Cache load failed ({e}), falling back to from_pretrained")

    if load_in_8bit:
        _print("  Using 8-bit quantization")
    if load_in_4bit:
        _print("  Using 4-bit quantization")
    if lora_adapter:
        _print(f"  With LoRA adapter: {lora_adapter}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Tensor parallelism mode (under torchrun)
    if is_tp_mode():
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        _print(f"  TP mode: {dist.get_world_size()} GPUs")

        # TP constraints:
        # - tp_plan="auto" picks up config.base_model_tp_plan automatically
        # - Cannot use device_map with tp_plan (mutually exclusive)
        # - Must use native HF model class (not custom trust_remote_code classes) for TP
        # - Don't pass torch_dtype (let FP8 stay as-is, avoid 2x memory expansion)

        # Some models (e.g. moonshotai/Kimi-K2-Thinking) set model_type to a custom
        # value (kimi_k2) and include auto_map pointing to custom code, even though the
        # architecture is identical to a natively-supported HF class (deepseek_v3).
        # This breaks TP because: (1) the custom class has no tp_plan, and (2) HF's TP
        # machinery requires the native class. Fix: load the raw config.json directly,
        # check if auto_map points to a known native class, and override model_type
        # so HF resolves to the native implementation.
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Map custom model_types to their native HF equivalents
        _custom_to_native = {
            "kimi_k2": "deepseek_v3",
        }
        native_type = _custom_to_native.get(config.model_type)
        if native_type:
            _print(f"  Overriding model_type '{config.model_type}' → '{native_type}' for TP compatibility")
            config.model_type = native_type
            config.auto_map = {}  # Drop custom class references

            # Reload config via native class to pick up base_model_tp_plan
            native_config_cls = AutoConfig.for_model(native_type)
            config = native_config_cls.from_pretrained(model_name, trust_remote_code=True)
            config.model_type = native_type
            config.auto_map = {}

        # Inject attention sharding into tp_plan before loading.
        # HF's DeepSeek V3 config only shards MoE/MLP (attention is replicated).
        # Adding attention sharding saves ~6 GB/GPU, enabling ~4x larger batch sizes.
        # Pattern: colwise on fan-out projections, rowwise on fan-in, gather to combine.
        if hasattr(config, 'base_model_tp_plan') and config.base_model_tp_plan:
            config.base_model_tp_plan.update({
                "layers.*.self_attn.q_b_proj": "local_colwise",
                "layers.*.self_attn.kv_b_proj": "local_colwise",
                "layers.*.self_attn.o_proj": "local_rowwise",
                "layers.*.self_attn": "gather",
            })
            _print(f"  Attention sharding: 4 entries added to tp_plan")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            tp_plan="auto",
        )

        # Fix NaN from padded batches (softmax of fully-masked rows → NaN → contaminates all tokens).
        # Must be installed before any forward pass.
        install_unmask_padding_hook(model)
        _print(f"  Unmask-padding hook installed")

        if lora_adapter:
            from peft import PeftModel
            _print(f"  Applying LoRA adapter...")
            model = PeftModel.from_pretrained(model, lora_adapter)
            _print(f"  LoRA adapter applied.")

        model.eval()
        _print("Model loaded (TP).")
        return model, tokenizer

    # Standard loading path
    model_kwargs = {
        "device_map": device,
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }

    # Skip compressed_tensors' compress_model step for models that are already
    # compressed on disk (e.g. INT4 MoE). With run_compressed=True, apply_quantization_config
    # already sets up CompressedLinear wrappers — compress_model redundantly iterates all
    # modules (208K+ for MoE) doing no-ops on meta tensors, taking hours.
    _patched_compress = False
    try:
        from transformers import AutoConfig
        _cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        _qc = getattr(_cfg, 'quantization_config', None)
        if isinstance(_qc, dict) and _qc.get('quant_method') == 'compressed-tensors':
            from compressed_tensors.compressors import ModelCompressor
            ModelCompressor.compress_model = lambda self, model: None
            # Skip per-module quantization init (69K+ modules). For run_compressed=True,
            # CompressedLinear.forward handles inference natively — the init just allocates
            # placeholder tensors (overwritten by shard loader) and wraps forward with
            # calibration closures we don't need. Just set the scheme attribute.
            from compressed_tensors.quantization.lifecycle import initialize as _ct_init
            from compressed_tensors.quantization.quant_config import QuantizationStatus
            _orig_init = _ct_init.initialize_module_for_quantization
            def _fast_init(module, scheme=None, force_zero_point=True):
                scheme = scheme or getattr(module, "quantization_scheme", None)
                if scheme is not None:
                    module.quantization_scheme = scheme
                    module.quantization_status = QuantizationStatus.COMPRESSED
            _ct_init.initialize_module_for_quantization = _fast_init
            _patched_compress = True
            _print("  Skipping compressed_tensors compress_model + fast init (already compressed on disk)")
    except Exception:
        pass

    # Multi-GPU: build even device_map for pipeline parallelism
    if device == "auto" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        n_gpus = torch.cuda.device_count()
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            n_layers = cfg.num_hidden_layers
            # Distribute layers evenly across GPUs
            # lm_head + embed_tokens must share a device (tied weights) — put both on GPU 0
            custom_device_map = {"model.embed_tokens": 0, "model.norm": 0, "lm_head": 0}
            layers_per_gpu = n_layers // n_gpus
            remainder = n_layers % n_gpus
            layer_idx = 0
            for gpu in range(n_gpus):
                # Distribute remainder across first GPUs
                n = layers_per_gpu + (1 if gpu < remainder else 0)
                for _ in range(n):
                    custom_device_map[f"model.layers.{layer_idx}"] = gpu
                    layer_idx += 1
            model_kwargs["device_map"] = custom_device_map
            _print(f"  Pipeline parallel: {n_layers} layers across {n_gpus} GPUs "
                   f"({layers_per_gpu}+1 x {remainder}, {layers_per_gpu} x {n_gpus - remainder})")
        except Exception:
            # Fallback: let accelerate decide, but cap per-GPU memory
            import subprocess
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                free_mbs = [int(x.strip()) for x in result.stdout.strip().split("\n")]
                model_kwargs["max_memory"] = {
                    i: f"{int(mb * 0.97)}MiB" for i, mb in enumerate(free_mbs)
                }
                _print(f"  max_memory: {int(min(free_mbs) * 0.97)}MiB x {len(free_mbs)} GPUs")
            except Exception:
                pass

    # Use BitsAndBytesConfig for quantization (replaces deprecated load_in_8bit/load_in_4bit)
    if load_in_8bit or load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError("bitsandbytes required for quantization. Install with: pip install bitsandbytes")

        bnb_kwargs = {
            "load_in_8bit": load_in_8bit,
            "load_in_4bit": load_in_4bit,
            "llm_int8_enable_fp32_cpu_offload": True,  # Allow CPU offload if needed
        }
        if load_in_4bit:
            bnb_kwargs["bnb_4bit_compute_dtype"] = dtype
            bnb_kwargs["bnb_4bit_quant_type"] = bnb_4bit_quant_type
        bnb_config = BitsAndBytesConfig(**bnb_kwargs)
        model_kwargs["quantization_config"] = bnb_config

    import time as _time; _t0 = _time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    _print(f"  from_pretrained: {_time.time()-_t0:.0f}s"); import sys; sys.stdout.flush()

    # Fuse MoE expert weights for INT4 compressed-tensors models
    if _patched_compress:
        _patch_moe_forward(model, _print)

    # Apply LoRA adapter if specified
    if lora_adapter:
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("peft required for LoRA. Install with: pip install peft")

        _print(f"  Applying LoRA adapter...")
        # Load LoRA to CPU first, then distribute to match base model layers
        # This avoids OOM when base model is split across GPUs
        model = PeftModel.from_pretrained(
            model, lora_adapter,
            torch_device="cpu",
            low_cpu_mem_usage=True,
        )
        _print(f"  LoRA adapter applied.")

    model.eval()

    # Auto-save cache for fast reload (skip from_pretrained on next load)
    if _patched_compress and not lora_adapter:
        cache_dir = _get_model_cache_dir(model_name)
        if not cache_dir.exists():
            try:
                save_model_cache(model, tokenizer, model_name, _print)
            except Exception as e:
                _print(f"  Warning: failed to save model cache: {e}")

    _print("Model loaded.")
    return model, tokenizer


def format_prompt(
    prompt: str,
    tokenizer,
    use_chat_template: bool = None,
    system_prompt: str = None,
) -> str:
    """
    Format prompt for model input.

    Args:
        prompt: Raw user prompt
        tokenizer: Model tokenizer
        use_chat_template: True/False to force, None to auto-detect from tokenizer
        system_prompt: Optional system message (for models that support it)

    Returns:
        Formatted prompt string ready for tokenization
    """
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    if not use_chat_template:
        return prompt

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        # Some models don't support system role - retry without it
        if system_prompt and "system" in str(e).lower():
            print(f"Warning: Model doesn't support system role, ignoring system_prompt")
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        raise


def tokenize_prompt(formatted_prompt, tokenizer, **kwargs):
    """
    Tokenize single or multiple prompts. Delegates to tokenize_batch().

    Kept for backward compatibility. New code should use tokenize_batch() directly.

    Args:
        formatted_prompt: Output from format_prompt() - single string or list of strings
        tokenizer: Model tokenizer
        **kwargs: Additional args passed to tokenizer

    Returns:
        Tokenized inputs dict with input_ids, attention_mask, etc.
    """
    texts = [formatted_prompt] if isinstance(formatted_prompt, str) else formatted_prompt
    return tokenize_batch(texts, tokenizer, padding_side="left", **kwargs)


def tokenize_batch(texts: list[str], tokenizer, padding_side: str = "left", **kwargs) -> dict:
    """
    Single source of truth for generation tokenization.

    Auto-detects add_special_tokens from text content (checks if text already has BOS).
    Validates for double-BOS bugs.

    Args:
        texts: List of text strings (may be pre-formatted with chat template)
        tokenizer: HuggingFace tokenizer
        padding_side: "left" (for generation) or "right" (for classification)
        **kwargs: Additional tokenizer args. Can pass add_special_tokens to override auto-detection.

    Returns:
        dict with input_ids, attention_mask, and lengths (actual token counts)
    """
    # Auto-detect if not explicitly set
    if 'add_special_tokens' not in kwargs:
        has_bos = (
            tokenizer.bos_token
            and texts
            and any(t.startswith(tokenizer.bos_token) for t in texts)
        )
        kwargs['add_special_tokens'] = not has_bos

    original_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side

    inputs = tokenizer(texts, return_tensors="pt", padding=True, **kwargs)

    # Validate: catch double BOS (only check first sequence, assumes batch is homogeneous)
    if (tokenizer.bos_token_id is not None
        and inputs.input_ids.shape[1] > 1
        and inputs.input_ids[0, 0].item() == inputs.input_ids[0, 1].item() == tokenizer.bos_token_id):
        raise ValueError(
            f"Double BOS token detected at positions 0 and 1.\n"
            f"First text: {texts[0][:100]!r}\n"
            f"This usually means text was formatted with chat template (adds BOS) "
            f"but add_special_tokens=True was set (adds another BOS).\n"
            f"Let tokenize_batch() auto-detect (don't pass add_special_tokens) "
            f"or ensure add_special_tokens=False for chat-templated text."
        )

    tokenizer.padding_side = original_side
    lengths = inputs.attention_mask.sum(dim=1).tolist()

    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "lengths": lengths,
    }


def pad_sequences(sequences: list, pad_token_id: int, padding_side: str = "left") -> dict:
    """
    Pad pre-tokenized sequences to the same length.

    Returns dict with input_ids, attention_mask, and pad_offsets (for position adjustment).
    """
    max_len = max(len(seq) for seq in sequences)
    input_ids, attention_masks, pad_offsets = [], [], []

    for seq in sequences:
        pad_len = max_len - len(seq)
        pad_spec = (pad_len, 0) if padding_side == "left" else (0, pad_len)
        input_ids.append(pad(seq, pad_spec, value=pad_token_id))
        attention_masks.append(pad(torch.ones(len(seq)), pad_spec, value=0))
        pad_offsets.append(pad_len if padding_side == "left" else 0)

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "pad_offsets": pad_offsets,
    }


def load_model_or_client(
    model_name: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    bnb_4bit_quant_type: str = DEFAULT_BNB_4BIT_QUANT_TYPE,
    no_server: bool = False,
    lora_adapter: str = None,
):
    """
    Load model locally or get client if server available.

    Returns:
        (model, tokenizer, is_remote) tuple
    """
    from other.server.client import get_model_or_client as _get_model_or_client, ModelClient

    # LoRA requires local loading
    if lora_adapter:
        model, tokenizer = load_model_with_lora(model_name, lora_adapter=lora_adapter, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, bnb_4bit_quant_type=bnb_4bit_quant_type)
        return model, tokenizer, False

    if not no_server:
        handle = _get_model_or_client(model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
        if isinstance(handle, ModelClient):
            print(f"Using model server (model: {model_name})")
            return handle, handle, True  # model, tokenizer, is_remote
        model, tokenizer = handle
        return model, tokenizer, False
    else:
        model, tokenizer = load_model(model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, bnb_4bit_quant_type=bnb_4bit_quant_type)
        return model, tokenizer, False
