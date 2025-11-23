"""
PathBuilder - Single source of truth for repo paths.
Loads structure from config/paths.yaml

Usage:
    from utils.paths import get, template

    # Get resolved path
    vectors_dir = get('extraction.vectors', experiment='{experiment_name}', trait='cognitive_state/context')
    # Returns: Path('experiments/{experiment_name}/extraction/cognitive_state/context/vectors')

    # Combine with pattern
    vector_file = get('extraction.vectors', experiment=exp, trait=t) / get('patterns.vector', method='probe', layer=16)
    # Returns: Path('.../vectors/probe_layer16.pt')

    # Get raw template
    tmpl = template('extraction.vectors')
    # Returns: 'experiments/{experiment}/extraction/{trait}/vectors'
"""

import yaml
from pathlib import Path
from typing import Union

_config = None
_config_path = Path(__file__).parent.parent / "config" / "paths.yaml"


def _load_config():
    """Load config from YAML (cached)."""
    global _config
    if _config is None:
        with open(_config_path) as f:
            _config = yaml.safe_load(f)
    return _config


def get(key: str, **variables) -> Path:
    """
    Get a path by key with variable substitution.

    Args:
        key: Dot-separated key like 'extraction.vectors'
        **variables: Values for template variables (experiment, trait, method, layer, etc.)

    Returns:
        Path object with variables substituted

    Raises:
        KeyError: If key not found in config

    Example:
        get('extraction.vectors', experiment='{experiment_name}', trait='cognitive_state/context')
        # Returns: Path('experiments/{experiment_name}/extraction/cognitive_state/context/vectors')
    """
    config = _load_config()

    # Navigate to key
    node = config
    for k in key.split('.'):
        if k not in node:
            raise KeyError(f"Path key not found: '{key}' (failed at '{k}')")
        node = node[k]

    if not isinstance(node, str):
        raise ValueError(f"Path key '{key}' is not a template string, got {type(node)}")

    # Substitute variables
    result = node
    for var, val in variables.items():
        result = result.replace(f'{{{var}}}', str(val))

    return Path(result)


def template(key: str) -> str:
    """
    Get raw template string without substitution.

    Args:
        key: Dot-separated key like 'extraction.vectors'

    Returns:
        Raw template string with {variables} intact

    Example:
        template('extraction.vectors')
        # Returns: 'experiments/{experiment}/extraction/{trait}/vectors'
    """
    config = _load_config()

    node = config
    for k in key.split('.'):
        if k not in node:
            raise KeyError(f"Path key not found: '{key}' (failed at '{k}')")
        node = node[k]

    return node


def list_keys(prefix: str = '') -> list:
    """
    List all available path keys, optionally filtered by prefix.

    Args:
        prefix: Optional prefix to filter keys (e.g., 'extraction')

    Returns:
        List of dot-separated keys

    Example:
        list_keys('extraction')
        # Returns: ['extraction.base', 'extraction.trait', 'extraction.vectors', ...]
    """
    config = _load_config()

    def _collect_keys(node, current_prefix=''):
        keys = []
        if isinstance(node, dict):
            for k, v in node.items():
                new_prefix = f"{current_prefix}.{k}" if current_prefix else k
                if isinstance(v, str):
                    keys.append(new_prefix)
                else:
                    keys.extend(_collect_keys(v, new_prefix))
        return keys

    all_keys = _collect_keys(config)

    if prefix:
        return [k for k in all_keys if k.startswith(prefix)]
    return all_keys


def reload():
    """Force reload of config (useful for testing)."""
    global _config
    _config = None
    _load_config()
