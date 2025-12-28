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


def discover_traits(category: str = None) -> list[str]:
    """
    Find trait definitions in datasets/traits/ (have positive.txt and negative.txt).

    Args:
        category: Optional category filter (e.g., 'epistemic')

    Returns:
        List of trait paths like ['epistemic/optimism', 'behavioral/refusal']
    """
    traits = []
    traits_dir = get('datasets.traits')
    if not traits_dir.is_dir():
        return []
    for cat_dir in traits_dir.iterdir():
        if not cat_dir.is_dir() or cat_dir.name.startswith('.'):
            continue
        if category and cat_dir.name != category:
            continue
        for trait_dir in cat_dir.iterdir():
            if trait_dir.is_dir() and (trait_dir / 'positive.txt').exists() and (trait_dir / 'negative.txt').exists():
                traits.append(f"{cat_dir.name}/{trait_dir.name}")
    return sorted(traits)


def discover_extracted_traits(experiment: str) -> list[tuple[str, str]]:
    """
    Find traits with extracted vectors in experiments/{exp}/extraction/.

    Args:
        experiment: Experiment name

    Returns:
        List of (category, trait_name) tuples for traits that have .pt vector files
    """
    extraction_dir = get('extraction.base', experiment=experiment)
    if not extraction_dir.exists():
        return []

    traits = []
    for category_dir in sorted(extraction_dir.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        for trait_dir in sorted(category_dir.iterdir()):
            if not trait_dir.is_dir():
                continue
            vectors_dir = trait_dir / "vectors"
            # Vectors are now in {position}/{component}/{method}/layer*.pt
            if vectors_dir.exists() and list(vectors_dir.rglob('layer*.pt')):
                traits.append((category_dir.name, trait_dir.name))
    return traits


# =============================================================================
# Centralized File Path Helpers
# =============================================================================
#
# Directory structure:
#   extraction/{trait}/
#   ├── activations/{position}/{component}/
#   │   ├── train_all_layers.pt
#   │   ├── val_all_layers.pt
#   │   └── metadata.json
#   └── vectors/{position}/{component}/{method}/
#       ├── layer16.pt
#       └── metadata.json
#
#   steering/{trait}/{position}/
#   └── results.json


def sanitize_position(position: str) -> str:
    """
    Convert position string to filesystem-safe directory name.

    Examples:
        response[:]  -> response_all
        response[-1] -> response_-1
        response[-5:] -> response_-5_
        prompt[-3:] -> prompt_-3_
        all[:] -> all_all
    """
    return (position
            .replace('[:]', '_all')
            .replace('[', '_')
            .replace(']', '')
            .replace(':', '_'))


def desanitize_position(sanitized: str) -> str:
    """
    Convert filesystem-safe directory name back to position string.

    Examples:
        response_all -> response[:]
        response_-1 -> response[-1]
        response_-5_ -> response[-5:]
        prompt_-3_ -> prompt[-3:]
        all_all -> all[:]
    """
    # Handle _all suffix (represents [:])
    if sanitized.endswith('_all'):
        prefix = sanitized[:-4]
        return f"{prefix}[:]"

    # Handle other patterns: {frame}_{slice}
    # Trailing _ means there was a : at the end (open slice)
    parts = sanitized.split('_', 1)
    if len(parts) == 2:
        frame, slice_part = parts
        if slice_part.endswith('_'):
            return f"{frame}[{slice_part[:-1]}:]"
        else:
            return f"{frame}[{slice_part}]"

    return sanitized  # Fallback


# -----------------------------------------------------------------------------
# Activation paths
# -----------------------------------------------------------------------------

def get_activation_dir(
    experiment: str,
    trait: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Path:
    """
    Directory for activation files.

    Returns: experiments/{experiment}/extraction/{trait}/activations/{position}/{component}/
    """
    base = get('extraction.activations', experiment=experiment, trait=trait)
    pos_dir = sanitize_position(position)
    return base / pos_dir / component


def get_activation_path(
    experiment: str,
    trait: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Path:
    """
    Path to training activation tensor file.

    Returns: .../activations/{position}/{component}/train_all_layers.pt
    """
    return get_activation_dir(experiment, trait, component, position) / "train_all_layers.pt"


def get_val_activation_path(
    experiment: str,
    trait: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Path:
    """
    Path to validation activation tensor file.

    Returns: .../activations/{position}/{component}/val_all_layers.pt
    """
    return get_activation_dir(experiment, trait, component, position) / "val_all_layers.pt"


def get_activation_metadata_path(
    experiment: str,
    trait: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Path:
    """
    Path to activation metadata file.

    Returns: .../activations/{position}/{component}/metadata.json
    """
    return get_activation_dir(experiment, trait, component, position) / "metadata.json"


# -----------------------------------------------------------------------------
# Vector paths
# -----------------------------------------------------------------------------

def get_vector_dir(
    experiment: str,
    trait: str,
    method: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Path:
    """
    Directory for vector files of a specific method.

    Returns: experiments/{experiment}/extraction/{trait}/vectors/{position}/{component}/{method}/
    """
    base = get('extraction.vectors', experiment=experiment, trait=trait)
    pos_dir = sanitize_position(position)
    return base / pos_dir / component / method


def get_vector_path(
    experiment: str,
    trait: str,
    method: str,
    layer: int,
    component: str = "residual",
    position: str = "response[:]",
) -> Path:
    """
    Path to extracted vector file.

    Returns: .../vectors/{position}/{component}/{method}/layer{layer}.pt
    """
    return get_vector_dir(experiment, trait, method, component, position) / f"layer{layer}.pt"


def get_vector_metadata_path(
    experiment: str,
    trait: str,
    method: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Path:
    """
    Path to vector metadata file (per method directory).

    Returns: .../vectors/{position}/{component}/{method}/metadata.json
    """
    return get_vector_dir(experiment, trait, method, component, position) / "metadata.json"


# -----------------------------------------------------------------------------
# Steering paths
# -----------------------------------------------------------------------------

def get_steering_dir(
    experiment: str,
    trait: str,
    position: str = "response[:]",
) -> Path:
    """
    Directory for steering results.

    Returns: experiments/{experiment}/steering/{trait}/{position}/
    """
    base = get('steering.trait', experiment=experiment, trait=trait)
    pos_dir = sanitize_position(position)
    return base / pos_dir


def get_steering_results_path(
    experiment: str,
    trait: str,
    position: str = "response[:]",
) -> Path:
    """
    Path to steering results file.

    Returns: .../steering/{trait}/{position}/results.json
    """
    return get_steering_dir(experiment, trait, position) / "results.json"


# -----------------------------------------------------------------------------
# Discovery helpers
# -----------------------------------------------------------------------------

def list_positions(experiment: str, trait: str) -> list[str]:
    """
    Discover available positions for a trait (by scanning vectors directory).

    Returns list of position directory names like ['response_all', 'response_-1']
    """
    base = get('extraction.vectors', experiment=experiment, trait=trait)
    if not base.exists():
        return []
    return sorted([d.name for d in base.iterdir() if d.is_dir()])


def list_components(
    experiment: str,
    trait: str,
    position: str = "response[:]",
) -> list[str]:
    """
    Discover available components for a trait/position.

    Returns list like ['residual', 'attn_out']
    """
    base = get('extraction.vectors', experiment=experiment, trait=trait)
    pos_dir = sanitize_position(position)
    comp_base = base / pos_dir
    if not comp_base.exists():
        return []
    return sorted([d.name for d in comp_base.iterdir() if d.is_dir()])


def list_methods(
    experiment: str,
    trait: str,
    component: str = "residual",
    position: str = "response[:]",
) -> list[str]:
    """
    Discover available methods for a trait/position/component.

    Returns list like ['probe', 'mean_diff', 'gradient']
    """
    base = get('extraction.vectors', experiment=experiment, trait=trait)
    pos_dir = sanitize_position(position)
    method_base = base / pos_dir / component
    if not method_base.exists():
        return []
    return sorted([d.name for d in method_base.iterdir() if d.is_dir()])


def list_layers(
    experiment: str,
    trait: str,
    method: str,
    component: str = "residual",
    position: str = "response[:]",
) -> list[int]:
    """
    Discover available layers for a trait/position/component/method.

    Returns list of layer indices like [0, 1, 14, 15, 16]
    """
    import re
    vector_dir = get_vector_dir(experiment, trait, method, component, position)
    if not vector_dir.exists():
        return []

    layers = []
    pattern = re.compile(r'^layer(\d+)\.pt$')
    for f in vector_dir.iterdir():
        match = pattern.match(f.name)
        if match:
            layers.append(int(match.group(1)))
    return sorted(layers)
