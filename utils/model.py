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

DEFAULT_BNB_4BIT_QUANT_TYPE = "nf4"


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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set pad_token if missing (required for batched generation, e.g. Mistral)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-pad for generation (decoder-only models need prompt right-aligned)
    tokenizer.padding_side = 'left'
    # AWQ models require fp16 and GPU-only device map
    is_awq = "AWQ" in model_name or "awq" in model_name.lower()
    if is_awq:
        dtype = torch.float16
        device = "cuda"
        print(f"  AWQ model detected, using fp16 and device_map='cuda'")

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device,
    }
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

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    print("Model loaded.")
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
    print(f"Loading model: {model_name}...")
    if load_in_8bit:
        print("  Using 8-bit quantization")
    if load_in_4bit:
        print("  Using 4-bit quantization")
    if lora_adapter:
        print(f"  With LoRA adapter: {lora_adapter}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Build model loading kwargs
    model_kwargs = {
        "device_map": device,
        "torch_dtype": dtype,
    }

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

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Apply LoRA adapter if specified
    if lora_adapter:
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("peft required for LoRA. Install with: pip install peft")

        print(f"  Applying LoRA adapter...")
        # Load LoRA to CPU first, then distribute to match base model layers
        # This avoids OOM when base model is split across GPUs
        model = PeftModel.from_pretrained(
            model, lora_adapter,
            torch_device="cpu",
            low_cpu_mem_usage=True,
        )
        print(f"  LoRA adapter applied.")

    model.eval()
    print("Model loaded.")
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
