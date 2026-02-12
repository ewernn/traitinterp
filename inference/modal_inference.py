"""
Modal deployment for activation capture with volume-cached models.

Uses Modal Volume to cache models between container restarts, reducing
cold starts from ~30s to ~5-10s.

Mounts core/ and utils/ to reuse hooks and path utilities from codebase.

Usage (from Railway or local):
    import requests
    response = requests.post(
        "https://MODAL_URL/capture",
        json={"prompt": "What is AI?", "experiment": "gemma-2-2b"}
    )
    data = response.json()
    # data has: tokens, response, activations {layer: [seq_len, hidden_dim]}
"""

import modal
import os
from pathlib import Path

app = modal.App("trait-capture")

# Persistent volume for model caching
volume = modal.Volume.from_name("model-cache", create_if_missing=True)

# Paths for copying core/ and utils/ into image
inference_dir = Path(__file__).parent
repo_root = inference_dir.parent

# Image with dependencies + local code copied in
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "scipy",
        "scikit-learn",
        "pyyaml",
    )
    .add_local_dir(repo_root / "core", remote_path="/root/core")
    .add_local_dir(repo_root / "utils", remote_path="/root/utils")
    .add_local_dir(repo_root / "config" / "models", remote_path="/root/config/models")
)

# Model -> GPU mapping
MODEL_GPU_MAP = {
    "google/gemma-2-2b": "T4",
    "google/gemma-2-2b-it": "T4",
    "google/gemma-2-9b-it": "A10G",
    "Qwen/Qwen3-1.7B": "T4",
    "Qwen/Qwen2.5-7B-Instruct": "A10G",
    "Qwen/Qwen2.5-14B-Instruct": "A10G",
    "Qwen/Qwen2.5-32B-Instruct": "A100",
    "meta-llama/Llama-3.1-8B-Instruct": "A10G",
}

# System prompt for live chat inference
LIVE_CHAT_SYSTEM_PROMPT = "Respond concisely."

def get_gpu_for_model(model_name: str) -> str:
    """Get appropriate GPU for a model."""
    return MODEL_GPU_MAP.get(model_name, "A10G")  # Default to A10G


def _load_model_and_tokenizer(model_name: str):
    """
    Load model and tokenizer, using volume cache if available.

    Returns:
        (model, tokenizer, load_time)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time

    model_cache_dir = Path(f"/models/{model_name.replace('/', '--')}")
    start = time.time()

    if model_cache_dir.exists():
        print(f"Loading from cache: {model_cache_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_cache_dir))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_cache_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        print(f"Downloading from HuggingFace (first run only)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Cache for next time
        print(f"Saving to cache: {model_cache_dir}")
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_cache_dir))
        tokenizer.save_pretrained(str(model_cache_dir))
        volume.commit()  # Persist to volume

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.1f}s")

    return model, tokenizer, load_time


@app.function(
    gpu="T4",
    image=image,
    volumes={"/models": volume},
    timeout=120,
    secrets=[modal.Secret.from_name("huggingface")],
)
def warmup(model_name: str = "Qwen/Qwen3-1.7B") -> dict:
    """
    Warm up the GPU container by loading the model.

    Call this on page load to reduce latency on first chat message.

    Returns:
        {"status": "ready", "model": model_name, "load_time": seconds}
    """
    print(f"Warming up with {model_name}...")
    model, tokenizer, load_time = _load_model_and_tokenizer(model_name)

    return {
        "status": "ready",
        "model": model_name,
        "load_time": round(load_time, 2),
        "num_layers": model.config.num_hidden_layers,
    }


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/models": volume},
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface")],
)
def generate_batch_remote(
    model_name: str,
    scenarios: list[dict],
    max_new_tokens: int = 32,
    temperature: float = 0.0,
) -> list[str]:
    """
    Batch text generation on Modal GPU.

    Takes raw scenario dicts (same shape as load_scenarios output) and returns
    generated response strings. Handles prompt formatting internally.

    Args:
        model_name: HuggingFace model ID
        scenarios: [{"prompt": str, "system_prompt": str|None}, ...]
        max_new_tokens: Max tokens per response
        temperature: Sampling temperature (0.0 for greedy)

    Returns:
        List of response strings, one per scenario
    """
    import sys
    sys.path.insert(0, "/root")

    from utils.model_registry import is_base_model
    from utils.model import format_prompt
    from utils.generation import generate_batch

    model, tokenizer, load_time = _load_model_and_tokenizer(model_name)

    base_model = is_base_model(model_name)
    use_chat_template = not base_model and tokenizer.chat_template is not None

    print(f"Model: {model_name} ({'BASE' if base_model else 'IT'})")
    print(f"Generating {max_new_tokens} tokens for {len(scenarios)} scenarios...")

    formatted = [
        format_prompt(
            s['prompt'],
            tokenizer,
            use_chat_template=use_chat_template,
            system_prompt=s.get('system_prompt'),
        )
        for s in scenarios
    ]

    responses = generate_batch(
        model, tokenizer, formatted,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    print(f"Generated {len(responses)} responses")
    return responses


@app.function(
    gpu="T4",  # Will be overridden based on model
    image=image,
    volumes={"/models": volume},
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface")],
)
def capture_activations_stream(
    model_name: str,
    prompt: str = None,
    messages: list = None,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    component: str = "residual",
    steering_configs: list = None,
    system_prompt: str = None,
):
    """
    Stream activations from model inference token-by-token using HookedGenerator.

    Model is cached in /models volume - first run downloads, subsequent runs
    load from disk (~5-10s instead of ~25-30s).

    Args:
        model_name: HuggingFace model ID
        prompt: Raw prompt text (use OR messages, not both)
        messages: Chat messages [{"role": "user", "content": "..."}] (use OR prompt)
        max_new_tokens: Generation length
        temperature: Sampling temperature (0.0 for greedy)
        component: Which component to capture ("residual", "attn_out", "mlp_out")
        steering_configs: List of {"layer": int, "vector": list[float], "coefficient": float}
        system_prompt: Optional system prompt (default: LIVE_CHAT_SYSTEM_PROMPT)

    Yields:
        {
            "token": token string,
            "activations": {layer_idx: list[float]},  # [hidden_dim] for this token
            "model": model_name,
            "component": component,
        }
    """
    import sys
    import torch
    import time

    # Add mounted directories to path for imports
    sys.path.insert(0, "/root")

    # Now we can import from our codebase
    from core.generation import HookedGenerator, CaptureConfig, SteeringConfig
    from utils.model_registry import get_model_config

    print(f"Loading {model_name}...")
    model, tokenizer, load_time = _load_model_and_tokenizer(model_name)

    # Check if model supports system prompt
    try:
        model_config = get_model_config(model_name)
        supports_system = model_config.get('supports_system_prompt', False)
    except FileNotFoundError:
        print(f"No config for {model_name}, assuming no system prompt support")
        supports_system = False

    # Format prompt with chat template if messages provided
    if messages is not None:
        # Only add system prompt if model supports it
        if supports_system and (system_prompt or LIVE_CHAT_SYSTEM_PROMPT):
            sys_prompt = system_prompt or LIVE_CHAT_SYSTEM_PROMPT
            full_messages = [{"role": "system", "content": sys_prompt}] + messages
            print(f"Adding system prompt: {sys_prompt[:50]}...")
        else:
            full_messages = messages
            if not supports_system:
                print(f"Model doesn't support system prompt, skipping")

        # Apply chat template with Qwen3 thinking disabled
        template_kwargs = {"add_generation_prompt": True, "tokenize": False}
        if "Qwen3" in model_name:
            template_kwargs["enable_thinking"] = False

        formatted_prompt = tokenizer.apply_chat_template(full_messages, **template_kwargs)
    elif prompt is not None:
        formatted_prompt = prompt
    else:
        raise ValueError("Either 'prompt' or 'messages' must be provided")

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Build additional stop tokens (model-specific)
    stop_token_ids = set()
    for token in ['<|im_end|>', '<|end|>', '<|eot_id|>', '<end_of_turn>']:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if token_ids:
            stop_token_ids.add(token_ids[0])

    # Build steering configs for HookedGenerator
    steering = None
    if steering_configs:
        print(f"Applying steering: {len(steering_configs)} vectors")
        steering = []
        for cfg in steering_configs:
            print(f"   L{cfg['layer']}: coef={cfg['coefficient']}")
            steering.append(SteeringConfig(
                vector=torch.tensor(cfg['vector']),
                layer=cfg['layer'],
                component='residual',
                coefficient=cfg['coefficient'],
            ))

    # Use HookedGenerator for streaming
    print(f"Generating up to {max_new_tokens} tokens...")
    gen_start = time.time()
    tokens_generated = 0

    gen = HookedGenerator(model)
    capture = CaptureConfig(components=[component])

    for tok in gen.stream(
        inputs.input_ids,
        inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        capture=capture,
        steering=steering,
        stop_token_ids=stop_token_ids,
    ):
        token_str = tokenizer.decode([tok.token_id])
        tokens_generated += 1

        # Convert activations to serializable format
        activations = {}
        if tok.activations:
            for layer_idx, comps in tok.activations.items():
                if component in comps:
                    activations[layer_idx] = comps[component].tolist()

        yield {
            "token": token_str,
            "activations": activations,
            "model": model_name,
            "component": component,
        }

    gen_time = time.time() - gen_start
    if tokens_generated > 0:
        print(f"Generated {tokens_generated} tokens in {gen_time:.1f}s ({gen_time/tokens_generated:.3f}s/token)")
    else:
        print(f"Generated 0 tokens")


@app.local_entrypoint()
def main(
    model: str = "google/gemma-2-2b-it",
    prompt: str = "What is artificial intelligence?",
):
    """
    Test the streaming capture function locally.

    Usage:
        modal run inference/modal_inference.py --model google/gemma-2-2b-it --prompt "Hello!"
    """
    print(f"\nTesting streaming activation capture")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}\n")

    token_count = 0
    full_response = ""

    for chunk in capture_activations_stream.remote_gen(
        model_name=model,
        prompt=prompt,
        max_new_tokens=50,
    ):
        token = chunk['token']
        activations = chunk['activations']

        full_response += token
        token_count += 1

        # Show first few tokens with activation info
        if token_count <= 5:
            num_layers = len(activations)
            first_layer = list(activations.keys())[0]
            act_dim = len(activations[first_layer])
            print(f"Token {token_count}: '{token}' | Layers: {num_layers} | Dim: {act_dim}")

    print(f"\nResults:")
    print(f"Total tokens: {token_count}")
    print(f"Response: {full_response[:100]}...")
