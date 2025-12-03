"""
Shared model loading and prompt formatting utilities.

Usage:
    from utils.model import load_model, format_prompt, load_experiment_config

    model, tokenizer = load_model()  # Uses default Gemma 2B
    model, tokenizer = load_model("google/gemma-2-2b-it", device="cuda")

    # Format prompt (auto-detects chat template from tokenizer)
    formatted = format_prompt("Hello", tokenizer)

    # Load experiment config
    config = load_experiment_config("my_experiment")
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "google/gemma-2-2b-it"


def load_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer.

    Args:
        model_name: HuggingFace model name
        device: Device map ('auto', 'cuda', 'cpu', 'mps')
        dtype: Model dtype (default: bfloat16)

    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )
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

    # No config found - warn and return defaults
    if warn_missing:
        print(f"⚠️  No config.json found for experiment '{experiment}'")
        print(f"   Chat template will be auto-detected from tokenizer.")
        print(f"   Run extraction/run_pipeline.py to auto-create config.json,")
        print(f"   or create manually: {config_path}")

    return {
        "model": DEFAULT_MODEL,
        "use_chat_template": None,  # Auto-detect from tokenizer
        "system_prompt": None,
    }
