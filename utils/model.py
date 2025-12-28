"""
Shared model loading and prompt formatting utilities.

Usage:
    from utils.model import load_model, load_model_with_lora, format_prompt, tokenize_batch

    model, tokenizer = load_model("google/gemma-2-2b-it")
    model, tokenizer = load_model("google/gemma-2-2b-it", device="cuda")

    # Load with LoRA adapter (for 70B models with quantization)
    model, tokenizer = load_model_with_lora(
        "meta-llama/Llama-3.3-70B-Instruct",
        lora_adapter="auditing-agents/llama-3.3-70b-dpo-rt-lora",
        load_in_8bit=True
    )

    # Format prompt (auto-detects chat template from tokenizer)
    formatted = format_prompt("Hello", tokenizer)

    # Tokenize batch with padding (returns lengths for extracting real tokens)
    batch = tokenize_batch(["text1", "text2"], tokenizer)
    # batch["input_ids"], batch["attention_mask"], batch["lengths"]
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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
    # PeftModel wraps: model.base_model (LoraModel) -> model (LlamaForCausalLM) -> model (LlamaModel)
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        return model.base_model.model.model
    return model.model


def get_layer_path_prefix(model) -> str:
    """Get the hook path prefix to transformer layers, handling PeftModel wrapper.

    Args:
        model: A HuggingFace model (possibly wrapped in PeftModel)

    Returns:
        Hook path prefix like "model.layers" or "base_model.model.model.layers"
    """
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        return "base_model.model.model.layers"
    return "model.layers"


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
        # Force single GPU to avoid auto-planner issues
        if device == "auto":
            model_kwargs["device_map"] = "cuda:0"
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
    dtype: torch.dtype = torch.float16,
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
        dtype: Model dtype (default: float16 for compatibility with quantization)

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
        model = PeftModel.from_pretrained(model, lora_adapter)
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


def tokenize_batch(
    texts: list[str],
    tokenizer,
    padding_side: str = "left",
    **kwargs,
) -> dict:
    """
    Tokenize a batch of texts with padding, returning inputs and real lengths.

    Centralizes padding logic to avoid reimplementing in multiple places.
    Left padding is default for generation (tokens append to the right).

    Args:
        texts: List of text strings to tokenize
        tokenizer: Model tokenizer
        padding_side: "left" (default, for generation) or "right"
        **kwargs: Additional args passed to tokenizer (truncation, max_length, etc.)

    Returns:
        Dict with:
            - input_ids: [batch, max_seq_len]
            - attention_mask: [batch, max_seq_len]
            - lengths: list[int] of actual lengths (excluding padding)
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
