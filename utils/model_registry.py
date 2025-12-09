"""
Model architecture registry. Loads from config/models/*.yaml

Usage:
    from utils.model_registry import get_model_config, get_num_layers

    config = get_model_config('gemma-2-2b-it')
    n_layers = get_num_layers('gemma-2-2b-it')
"""

import json
import yaml
from pathlib import Path
from typing import Optional

_cache: dict = {}
_models_dir = Path(__file__).parent.parent / "config" / "models"
_experiments_dir = Path(__file__).parent.parent / "experiments"


def get_model_config(model_id: str) -> dict:
    """Load model config from config/models/{model_id}.yaml"""
    if model_id in _cache:
        return _cache[model_id]

    # Normalize: google/gemma-2-2b-it -> gemma-2-2b-it
    if '/' in model_id:
        model_id = model_id.split('/')[-1].lower()

    config_path = _models_dir / f"{model_id}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No model config at {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    _cache[model_id] = config
    return config


def get_num_layers(model_id: str) -> int:
    """Get number of hidden layers."""
    return get_model_config(model_id)['num_hidden_layers']


def get_sae_path(model_id: str, layer: int) -> Optional[Path]:
    """Get SAE path for layer, if available."""
    config = get_model_config(model_id)
    sae = config.get('sae', {})

    if not sae.get('available', False):
        return None
    if layer not in sae.get('downloaded_layers', []):
        return None

    base_path = Path(__file__).parent.parent / sae['base_path']
    layer_dir = sae['layer_template'].format(layer=layer)
    return base_path / layer_dir


def _load_experiment_config(experiment: str) -> dict:
    """Load experiment config.json (internal helper)."""
    config_path = _experiments_dir / experiment / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def get_extraction_model(experiment: str) -> Optional[str]:
    """Get extraction_model from experiment config.json"""
    config = _load_experiment_config(experiment)
    return config.get('extraction_model')


def get_application_model(experiment: str) -> Optional[str]:
    """Get application_model from experiment config.json"""
    config = _load_experiment_config(experiment)
    return config.get('application_model')
