"""
Model architecture registry. Loads from config/models/*.yaml

Usage:
    from utils.model_registry import get_model_config, get_num_layers

    config = get_model_config('google/gemma-2-2b-it')
    n_layers = get_num_layers('google/gemma-2-2b-it')
"""

import yaml
from pathlib import Path
from typing import Optional

_cache: dict = {}
_models_dir = Path(__file__).parent.parent / "config" / "models"



def yaml_slug_for(model_id: str) -> str:
    """Canonical filename slug for a HF model id.

    `Qwen/Qwen3.5-9B` → `qwen--qwen3.5-9b`. The `--` separator mirrors HF's
    own cache directory naming (`models--Qwen--Qwen3.5-9B`) and avoids
    collisions when two orgs ship a model with the same short name.
    """
    return model_id.lower().replace('/', '--')


def get_model_config(model_id: str) -> dict:
    """Load model config from config/models/{org}--{name}.yaml"""
    if model_id in _cache:
        return _cache[model_id]

    slug = yaml_slug_for(model_id)
    config_path = _models_dir / f"{slug}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No model config at {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    _cache[model_id] = config
    return config


def get_num_layers(model_id: str) -> int:
    """Get number of hidden layers."""
    return get_model_config(model_id)['num_hidden_layers']


def is_base_model(model_id: str) -> bool:
    """Check if model is a base model (not instruction-tuned).

    Reads `pretrained` from the yaml strictly — no name-heuristic fallback.
    Name-based detection is unreliable (Qwen3-4B has no `-instruct` suffix
    but ships with a chat template; conversely, Kimi-K2-Base carries a chat
    template despite being a base model). The yaml is the source of truth;
    add a config/models/*.yaml for new models via `dev/onboard_model.py`.
    """
    return get_model_config(model_id)['pretrained']


