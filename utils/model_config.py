"""
Model configuration loader.
Loads model architecture and settings from config/models/*.yaml

Input: Model ID (e.g., 'gemma-2-2b-it') or experiment name
Output: Dict with model config (architecture, SAE info, defaults)

Usage:
    from utils.model_config import get_model_config, get_config_for_experiment

    # Direct model lookup
    config = get_model_config('gemma-2-2b-it')
    config['num_hidden_layers']  # 26
    config['sae']['available']   # True

    # From experiment (reads experiment's config.json to get model)
    config = get_config_for_experiment('gemma-2-2b-it')
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Optional, Any

_cache: Dict[str, dict] = {}
_models_dir = Path(__file__).parent.parent / "config" / "models"
_experiments_dir = Path(__file__).parent.parent / "experiments"


def get_model_config(model_id: str) -> dict:
    """
    Load model config by ID.

    Args:
        model_id: Model identifier matching YAML filename (e.g., 'gemma-2-2b-it')

    Returns:
        Dict with model configuration

    Raises:
        FileNotFoundError: If model config doesn't exist
    """
    if model_id in _cache:
        return _cache[model_id]

    # Try exact match first
    config_path = _models_dir / f"{model_id}.yaml"

    if not config_path.exists():
        # Try normalizing: google/gemma-2-2b-it -> gemma-2-2b-it
        if '/' in model_id:
            model_id = model_id.split('/')[-1].lower()
            config_path = _models_dir / f"{model_id}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"No model config found for '{model_id}' at {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    _cache[model_id] = config
    return config


def get_config_for_experiment(experiment: str) -> dict:
    """
    Load model config for an experiment.
    Reads experiment's config.json to determine which model, then loads model config.

    Args:
        experiment: Experiment name (e.g., 'gemma-2-2b-it')

    Returns:
        Dict with model configuration
    """
    exp_config_path = _experiments_dir / experiment / "config.json"

    if exp_config_path.exists():
        with open(exp_config_path) as f:
            exp_config = json.load(f)
        model_id = exp_config.get('model', experiment)
    else:
        # Fall back to experiment name as model ID
        model_id = experiment

    # Normalize HF ID to our config name
    # google/gemma-2-2b-it -> gemma-2-2b-it
    if '/' in model_id:
        model_id = model_id.split('/')[-1].lower()

    return get_model_config(model_id)


def list_available_models() -> list:
    """List all available model configs."""
    return [p.stem for p in _models_dir.glob("*.yaml")]


# Convenience accessors
def get_num_layers(model_or_experiment: str) -> int:
    """Get number of hidden layers for a model/experiment."""
    try:
        config = get_config_for_experiment(model_or_experiment)
    except FileNotFoundError:
        config = get_model_config(model_or_experiment)
    return config['num_hidden_layers']


def get_hidden_size(model_or_experiment: str) -> int:
    """Get hidden dimension for a model/experiment."""
    try:
        config = get_config_for_experiment(model_or_experiment)
    except FileNotFoundError:
        config = get_model_config(model_or_experiment)
    return config['hidden_size']


def get_sae_path(model_or_experiment: str, layer: int) -> Optional[Path]:
    """
    Get SAE path for a specific layer, if available.

    Returns:
        Path to SAE directory, or None if SAE not available for this model/layer
    """
    try:
        config = get_config_for_experiment(model_or_experiment)
    except FileNotFoundError:
        config = get_model_config(model_or_experiment)

    sae = config.get('sae', {})
    if not sae.get('available', False):
        return None

    downloaded = sae.get('downloaded_layers', [])
    if layer not in downloaded:
        return None

    base_path = Path(__file__).parent.parent / sae['base_path']
    layer_dir = sae['layer_template'].format(layer=layer)
    return base_path / layer_dir
