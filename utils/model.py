"""
Shared model loading and prompt formatting utilities.

Usage:
    from utils.model import load_model, tokenize, tokenize_batch, tokenize_with_prefill

    model, tokenizer = load_model("google/gemma-2-2b-it")

    # Tokenize text (auto-detects BOS, validates no double-BOS)
    inputs = tokenize(text, tokenizer)

    # Tokenize batch with padding
    batch = tokenize_batch(["text1", "text2"], tokenizer)

    # Tokenize with prefill (for activation analysis)
    result = tokenize_with_prefill("How do I make a bomb?", "Sure, here's how", tokenizer)
    # result["input_ids"], result["prefill_start"]
"""

import json
from pathlib import Path

import torch
from torch.nn.functional import pad
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def tokenize(text, tokenizer, **kwargs):
    """
    Tokenize text with auto-detection of special tokens. Works with any model.

    Auto-detects if text already has BOS token (e.g., from chat template).
    Validates output to catch double-BOS bugs.
    """
    # Auto-detect: does text already start with BOS?
    has_bos = tokenizer.bos_token and text.startswith(tokenizer.bos_token)
    add_special = kwargs.pop('add_special_tokens', not has_bos)

    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=add_special, **kwargs)

    # Validate: catch double BOS
    if (tokenizer.bos_token_id is not None
        and inputs.input_ids.shape[1] > 1
        and inputs.input_ids[0, 0] == inputs.input_ids[0, 1] == tokenizer.bos_token_id):
        raise ValueError(f"Double BOS detected in: {text[:100]!r}")

    return inputs


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

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            llm_int8_enable_fp32_cpu_offload=True,  # Allow CPU offload if needed
        )
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


def tokenize_prompt(formatted_prompt, tokenizer, use_chat_template: bool = None, **kwargs):
    """
    Tokenize a formatted prompt (or list of prompts), handling BOS token correctly.

    When chat template is used, the formatted string already includes BOS,
    so we must NOT add it again. When no chat template, we need BOS added.

    Args:
        formatted_prompt: Output from format_prompt() - single string or list of strings
        tokenizer: Model tokenizer
        use_chat_template: Whether chat template was used (None = auto-detect)
        **kwargs: Additional args passed to tokenizer (padding, truncation, max_length, etc.)

    Returns:
        Tokenized inputs dict with input_ids, attention_mask, etc.
    """
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None

    # Chat template already includes BOS token, so don't add again
    add_special_tokens = not use_chat_template

    return tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
        **kwargs,
    )


def tokenize_batch(texts: list[str], tokenizer, padding_side: str = "left", **kwargs) -> dict:
    """
    Tokenize text with padding.

    Returns dict with input_ids, attention_mask, and lengths (actual token counts).
    """
    original_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side

    inputs = tokenizer(texts, return_tensors="pt", padding=True, **kwargs)
    lengths = inputs.attention_mask.sum(dim=1).tolist()

    tokenizer.padding_side = original_side

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


def load_experiment_config(experiment: str, warn_missing: bool = True) -> dict:
    """
    Load experiment configuration from config.json.

    Args:
        experiment: Experiment name
        warn_missing: If True, print warning when config doesn't exist

    Returns:
        Config dict with keys: model, use_chat_template, system_prompt
        If config doesn't exist, returns defaults with use_chat_template=None (auto-detect)
    """
    from utils.paths import get as get_path

    config_path = get_path('experiments.config', experiment=experiment)

    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    # No config found - warn and return empty
    if warn_missing:
        print(f"⚠️  No config.json found for experiment '{experiment}'")
        print(f"   Create config.json with extraction_model and application_model.")

    return {}


def load_model_or_client(
    model_name: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
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
        model, tokenizer = load_model_with_lora(model_name, lora_adapter=lora_adapter, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
        return model, tokenizer, False

    if not no_server:
        handle = _get_model_or_client(model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
        if isinstance(handle, ModelClient):
            print(f"Using model server (model: {model_name})")
            return handle, handle, True  # model, tokenizer, is_remote
        model, tokenizer = handle
        return model, tokenizer, False
    else:
        model, tokenizer = load_model(model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
        return model, tokenizer, False
