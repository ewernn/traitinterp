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
        "bitsandbytes",
        "huggingface_hub",
        "scipy",
        "scikit-learn",
        "pyyaml",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",      # Faster HF downloads on first pull
        "HF_HUB_CACHE": "/models/hf_cache",  # Point HF's default cache to the Modal Volume
    })
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
    "Qwen/Qwen2.5-14B": "A100",
    "Qwen/Qwen2.5-14B-Instruct": "A100",
    "Qwen/Qwen2.5-32B-Instruct": "A100",
    "meta-llama/Llama-3.1-8B-Instruct": "A10G",
}

# System prompt for live chat inference
LIVE_CHAT_SYSTEM_PROMPT = "Respond concisely."

def get_gpu_for_model(model_name: str) -> str:
    """Get appropriate GPU for a model."""
    return MODEL_GPU_MAP.get(model_name, "A10G")  # Default to A10G


@app.function(
    gpu="A100",
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
    from utils.model import format_prompt, load_model
    from utils.model_generation import generate_batch

    # Full precision for extraction — activation fidelity matters more than VRAM.
    model, tokenizer = load_model(model_name)
    volume.commit()  # Persist any newly downloaded weights to HF_HUB_CACHE on volume

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


@app.cls(
    gpu="L4",  # 24GB VRAM — fits up to ~31B bnb-4bit; Ada arch = native bf16 + faster bnb dequant than T4
    image=image,
    volumes={"/models": volume},
    timeout=600,
    scaledown_window=30,  # TESTING: 30s for quick cold-start testing. Set back to 600 for production.
    secrets=[modal.Secret.from_name("huggingface")],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class TraitCapture:
    """
    GPU-snapshot-backed streaming inference with per-token activation capture.

    Model is loaded once inside @modal.enter(snap=True); Modal snapshots the
    container (including GPU memory) so future cold starts restore the
    snapshot instead of reloading weights. Target cold start: ~3-6s vs ~20s
    without snapshots.

    Each unique `model_name` gets its own container pool + snapshot — so
    you can have a warm gemma-2-2b container and a warm gemma-4-31b container
    simultaneously, keyed by the parameter.
    """
    model_name: str = modal.parameter(default="Qwen/Qwen3.5-9B")

    @modal.enter(snap=True)
    def load(self):
        """Load model + tokenizer into GPU memory. Snapshotted for fast cold restart.

        On-the-fly bnb-4bit quantization is applied for live chat — memory savings
        matter more than the tiny quality hit. utils.model.load_model handles
        already-quantized checkpoints correctly and installs the NaN-padding hook.
        """
        import sys
        sys.path.insert(0, "/root")
        from utils.model import load_model

        print(f"[TraitCapture] Loading {self.model_name} into snapshot...")
        self.model, self.tokenizer = load_model(self.model_name, load_in_4bit=True)
        volume.commit()  # Persist any newly downloaded weights to HF_HUB_CACHE on volume
        print(f"[TraitCapture] Ready, n_layers={self.model.config.num_hidden_layers}")

    @modal.method()
    def capture_activations_stream(
        self,
        prompt: str = None,
        messages: list = None,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        component: str = "residual",
        steering_configs: list = None,
        system_prompt: str = None,
        trait_vectors: list = None,
    ):
        """
        Stream token-by-token with trait projection scores.

        Projects activations onto trait vectors server-side and yields compact
        {token, trait_scores} events (~8 floats/token).

        Args:
            prompt: Raw prompt text (use OR messages, not both)
            messages: Chat messages [{"role": "user", "content": "..."}]
            max_new_tokens: Generation length
            temperature: Sampling temperature (0.0 for greedy)
            component: Which component to capture ("residual", "attn_out", "mlp_out")
            steering_configs: List of {"layer": int, "vector": list[float], "coefficient": float}
            system_prompt: Optional system prompt (default: LIVE_CHAT_SYSTEM_PROMPT)
            trait_vectors: List of {"trait": str, "layer": int, "vector": list[float]}

        Yields:
            {"token": ..., "trait_scores": {trait: score}, "model": ...}
        """
        import sys
        import torch
        import time
        sys.path.insert(0, "/root")  # Defensive; enter() already set this

        from core.generation import HookedGenerator, CaptureConfig, SteeringConfig
        from utils.model_registry import get_model_config

        model = self.model
        tokenizer = self.tokenizer
        model_name = self.model_name

        # Check if model supports system prompt
        try:
            model_config = get_model_config(model_name)
            supports_system = model_config.get('supports_system_prompt', False)
        except FileNotFoundError:
            print(f"No config for {model_name}, assuming no system prompt support")
            supports_system = False

        # Format prompt with chat template if messages provided
        if messages is not None:
            if supports_system and (system_prompt or LIVE_CHAT_SYSTEM_PROMPT):
                sys_prompt = system_prompt or LIVE_CHAT_SYSTEM_PROMPT
                full_messages = [{"role": "system", "content": sys_prompt}] + messages
            else:
                full_messages = messages

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

        # Prepare trait projection vectors for server-side projection
        from core import projection as project_fn
        trait_proj = []
        layers_needed = set()
        if trait_vectors:
            for tv in trait_vectors:
                vec = torch.tensor(tv['vector'], device=model.device, dtype=torch.float16)
                trait_proj.append((tv['trait'], tv['layer'], vec))
                layers_needed.add(tv['layer'])
        # Only capture layers we need for projection (not all 32)
        capture_layers = sorted(layers_needed) if layers_needed else None
        print(f"Server-side projection: {len(trait_proj)} traits, layers {capture_layers}")

        # Use HookedGenerator for streaming
        gen_start = time.time()
        tokens_generated = 0

        gen = HookedGenerator(model)
        capture = CaptureConfig(components=[component], layers=capture_layers)

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

            trait_scores = {}
            for trait_name, layer, vec in trait_proj:
                if tok.activations and layer in tok.activations:
                    comps = tok.activations[layer]
                    if component in comps:
                        act = comps[component]
                        score = project_fn(act, vec.to(act.device), normalize_vector=True).item()
                        trait_scores[trait_name] = round(score, 4)
            yield {
                "token": token_str,
                "trait_scores": trait_scores,
                "model": model_name,
            }

        gen_time = time.time() - gen_start
        if tokens_generated > 0:
            print(f"Generated {tokens_generated} tokens in {gen_time:.1f}s ({gen_time/tokens_generated:.3f}s/token)")


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

    for chunk in TraitCapture(model_name=model).capture_activations_stream.remote_gen(
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
