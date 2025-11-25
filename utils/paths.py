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


# =============================================================================
# Convenience functions for common path operations
# =============================================================================

def get_analysis_per_token(experiment: str, prompt_set: str, prompt_id: int) -> Path:
    """
    Get path to per-token analysis JSON file.

    Args:
        experiment: Experiment name
        prompt_set: Prompt set name (e.g., 'dynamic', 'single_trait')
        prompt_id: Prompt ID within the set

    Returns:
        Path to the per-token analysis JSON file

    Example:
        >>> get_analysis_per_token('gemma_2b_cognitive_nov21', 'dynamic', 1)
        Path('experiments/gemma_2b_cognitive_nov21/analysis/per_token/dynamic/1.json')
    """
    dir_path = get('analysis.per_token', experiment=experiment, prompt_set=prompt_set)
    file_name = get('patterns.per_token_json', prompt_id=prompt_id)
    return dir_path / file_name


def get_analysis_category_file(
    experiment: str,
    category: str,
    filename: str,
    ext: str = 'png'
) -> Path:
    """
    Get path to analysis category file.

    Args:
        experiment: Experiment name
        category: Analysis category (e.g., 'normalized_velocity', 'trait_projections')
        filename: Filename without extension (e.g., 'prompt_1', 'summary')
        ext: File extension ('png' or 'json')

    Returns:
        Path to the analysis file

    Example:
        >>> get_analysis_category_file('gemma_2b_cognitive_nov21', 'normalized_velocity', 'summary', 'png')
        Path('experiments/gemma_2b_cognitive_nov21/analysis/normalized_velocity/summary.png')
    """
    dir_path = get('analysis.category', experiment=experiment, category=category)

    # Determine pattern based on filename format
    if 'prompt_' in filename and filename.split('_')[-1].isdigit():
        # Pattern like prompt_1, prompt_2
        prompt_id = int(filename.split('_')[-1])
        pattern = 'patterns.analysis_prompt_png' if ext == 'png' else 'patterns.analysis_prompt_json'
        file_name = get(pattern, prompt_id=prompt_id)
    else:
        # Named file like summary, comparison, etc.
        pattern = 'patterns.analysis_named_png' if ext == 'png' else 'patterns.analysis_named_json'
        file_name = get(pattern, filename=filename)

    return dir_path / file_name


def get_extraction_file(experiment: str, trait: str, subpath: str) -> Path:
    """
    Get path to extraction file with arbitrary subpath.

    Args:
        experiment: Experiment name
        trait: Trait name (e.g., 'cognitive_state/context')
        subpath: Subpath within trait directory (e.g., 'responses/pos.json')

    Returns:
        Path to the extraction file

    Example:
        >>> get_extraction_file('gemma_2b_cognitive_nov21', 'cognitive_state/context', 'responses/pos.json')
        Path('experiments/gemma_2b_cognitive_nov21/extraction/cognitive_state/context/responses/pos.json')
    """
    base_path = get('extraction.trait', experiment=experiment, trait=trait)
    return base_path / subpath
