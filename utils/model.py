"""
Shared model loading utilities.

Usage:
    from utils.model import load_model
    model, tokenizer = load_model()  # Uses default Gemma 2B
    model, tokenizer = load_model("google/gemma-2-2b-it", device="cuda")
"""

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
