"""
PathBuilder - Single source of truth for repo paths.
Loads structure from config/paths.yaml

Usage:
    from utils.paths import get, get_vector_path, get_activation_path

    # Get base directory from YAML template
    vectors_dir = get('extraction.vectors', experiment='gemma-2-2b', trait='chirp/refusal_v2', model_variant='base')

    # Use helper functions for full paths (includes position/component/method)
    vector_file = get_vector_path('gemma-2-2b', 'chirp/refusal_v2', 'probe', 15, model_variant='base')

    # Model variant resolution
    config = get_model_variant('rm_syco', 'rm_lora')
    # Returns: {'name': 'rm_lora', 'model': '...', 'lora': '...'}

    default = get_default_variant('rm_syco', mode='extraction')
    # Returns: 'base'
"""

import json
import yaml
from pathlib import Path
from typing import Optional, Union

_config = None
_config_path = Path(__file__).parent.parent / "config" / "paths.yaml"

# Cache for experiment configs
_experiment_configs: dict[str, dict] = {}


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
        **variables: Values for template variables (experiment, trait, model_variant, etc.)

    Returns:
        Path object with variables substituted

    Raises:
        KeyError: If key not found in config

    Example:
        get('extraction.vectors', experiment='gemma-2-2b', trait='chirp/refusal', model_variant='base')
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

    # Warn on unsubstituted variables (likely caller forgot to pass them)
    if '{' in result:
        import re
        import warnings
        missing = re.findall(r'\{(\w+)\}', result)
        warnings.warn(f"Unsubstituted variables in path key '{key}': {missing}")

    return Path(result)


def template(key: str) -> str:
    """
    Get raw template string without substitution.

    Args:
        key: Dot-separated key like 'extraction.vectors'

    Returns:
        Raw template string with {variables} intact
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
    global _config, _experiment_configs
    _config = None
    _experiment_configs = {}
    _load_config()


# =============================================================================
# Model Variant Resolution
# =============================================================================

def load_experiment_config(experiment: str) -> dict:
    """
    Load experiment config.json (cached).

    Args:
        experiment: Experiment name

    Returns:
        Experiment config dict

    Raises:
        FileNotFoundError: If config.json doesn't exist
    """
    if experiment not in _experiment_configs:
        config_path = get('experiments.config', experiment=experiment)
        if not config_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {config_path}")
        with open(config_path) as f:
            _experiment_configs[experiment] = json.load(f)
    return _experiment_configs[experiment]


def get_model_variant(
    experiment: str,
    variant: Optional[str] = None,
    mode: str = "application"
) -> dict:
    """
    Resolve variant name to full config.

    Args:
        experiment: Experiment name
        variant: Variant name (e.g., 'base', 'instruct', 'rm_lora'), or None for default
        mode: "extraction" or "application" (determines which default to use)

    Returns:
        {
            'name': str,        # variant name
            'model': str,       # HuggingFace model path
            'lora': str|None,   # Optional LoRA adapter path
        }

    Raises:
        KeyError: If variant not found in config
        ValueError: If mode is invalid
    """
    if mode not in ("extraction", "application"):
        raise ValueError(f"mode must be 'extraction' or 'application', got '{mode}'")

    config = load_experiment_config(experiment)

    # Get variant name (use default if not specified)
    if variant is None:
        variant = get_default_variant(experiment, mode)

    # Look up variant config
    variants = config.get('model_variants', {})
    if variant not in variants:
        raise KeyError(f"Model variant '{variant}' not found in {experiment}/config.json. Available: {list(variants.keys())}")

    variant_config = variants[variant]
    return {
        'name': variant,
        'model': variant_config['model'],
        'lora': variant_config.get('lora'),
    }


def get_default_variant(experiment: str, mode: str = "application") -> str:
    """
    Get default variant name for extraction or application.

    Args:
        experiment: Experiment name
        mode: "extraction" or "application"

    Returns:
        Default variant name

    Raises:
        KeyError: If no default configured
    """
    config = load_experiment_config(experiment)
    defaults = config.get('defaults', {})

    if mode not in defaults:
        # Fallback: first variant in model_variants
        variants = config.get('model_variants', {})
        if variants:
            return next(iter(variants.keys()))
        raise KeyError(f"No defaults.{mode} in {experiment}/config.json and no model_variants defined")

    return defaults[mode]


def list_model_variants(experiment: str) -> list[str]:
    """
    List all variant names from config.

    Args:
        experiment: Experiment name

    Returns:
        List of variant names like ['base', 'instruct', 'rm_lora']
    """
    config = load_experiment_config(experiment)
    return list(config.get('model_variants', {}).keys())


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


def discover_extracted_traits(experiment: str, model_variant: str = None) -> list[tuple[str, str]]:
    """
    Find traits with extracted vectors in experiments/{exp}/extraction/.

    Args:
        experiment: Experiment name
        model_variant: Model variant to check (if None, checks all variants)

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

            # Check for vectors in variant subdirs
            if model_variant:
                # Check specific variant
                variant_dir = trait_dir / model_variant
                vectors_dir = variant_dir / "vectors"
                if vectors_dir.exists() and list(vectors_dir.rglob('layer*.pt')):
                    traits.append((category_dir.name, trait_dir.name))
            else:
                # Check any variant
                for variant_dir in trait_dir.iterdir():
                    if not variant_dir.is_dir():
                        continue
                    vectors_dir = variant_dir / "vectors"
                    if vectors_dir.exists() and list(vectors_dir.rglob('layer*.pt')):
                        traits.append((category_dir.name, trait_dir.name))
                        break  # Found at least one variant with vectors

    return traits


def discover_steering_entries(experiment: str) -> list[dict]:
    """
    Find all steering results in experiments/{exp}/steering/.

    Path structure: steering/{category}/{trait}/{model_variant}/{position}/{prompt_set...}/results.jsonl
    Note: prompt_set can be nested (e.g., rm_syco/train_100)

    Returns:
        List of dicts with keys: trait, model_variant, position, prompt_set, full_path
    """
    steering_dir = get('steering.base', experiment=experiment)
    if not steering_dir.exists():
        return []

    entries = []
    for results_file in steering_dir.rglob('results.jsonl'):
        rel_path = results_file.parent.relative_to(steering_dir)
        parts = rel_path.parts

        # Expected: {category}/{trait}/{model_variant}/{position}/{prompt_set...}
        # prompt_set can be nested (e.g., parts[4:] = ('rm_syco', 'train_100'))
        if len(parts) >= 5:
            entries.append({
                'trait': f"{parts[0]}/{parts[1]}",
                'model_variant': parts[2],
                'position': parts[3],
                'prompt_set': '/'.join(parts[4:]),  # Handle nested prompt_sets
                'full_path': str(rel_path)
            })

    return entries


# =============================================================================
# Position helpers
# =============================================================================

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
        response__5 -> response[:5]
        prompt_-3_ -> prompt[-3:]
        all_all -> all[:]
    """
    # Handle _all suffix (represents [:])
    if sanitized.endswith('_all'):
        prefix = sanitized[:-4]
        return f"{prefix}[:]"

    # Handle other patterns: {frame}_{slice}
    # Trailing _ means there was a : at the end (open slice)
    # Leading _ in slice means there was a : at the start (e.g., [:5] -> __5)
    parts = sanitized.split('_', 1)
    if len(parts) == 2:
        frame, slice_part = parts
        # Leading _ means original had : at start of slice (e.g., [:5])
        if slice_part.startswith('_'):
            return f"{frame}[:{slice_part[1:]}]"
        elif slice_part.endswith('_'):
            return f"{frame}[{slice_part[:-1]}:]"
        else:
            return f"{frame}[{slice_part}]"

    return sanitized  # Fallback


# =============================================================================
# Extraction paths
# =============================================================================

def get_activation_dir(
    experiment: str,
    trait: str,
    model_variant: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Path:
    """
    Directory for activation files.

    Returns: experiments/{experiment}/extraction/{trait}/{model_variant}/activations/{position}/{component}/
    """
    base = get('extraction.activations', experiment=experiment, trait=trait, model_variant=model_variant)
    pos_dir = sanitize_position(position)
    return base / pos_dir / component


def get_activation_path(
    experiment: str,
    trait: str,
    model_variant: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Path:
    """
    Path to training activation tensor file.

    Returns: .../activations/{position}/{component}/train_all_layers.pt
    """
    return get_activation_dir(experiment, trait, model_variant, component, position) / "train_all_layers.pt"


def get_val_activation_path(
    experiment: str,
    trait: str,
    model_variant: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Path:
    """
    Path to validation activation tensor file.

    Returns: .../activations/{position}/{component}/val_all_layers.pt
    """
    return get_activation_dir(experiment, trait, model_variant, component, position) / "val_all_layers.pt"


def get_activation_metadata_path(
    experiment: str,
    trait: str,
    model_variant: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Path:
    """
    Path to activation metadata file.

    Returns: .../activations/{position}/{component}/metadata.json
    """
    return get_activation_dir(experiment, trait, model_variant, component, position) / "metadata.json"


def get_vector_dir(
    experiment: str,
    trait: str,
    method: str,
    model_variant: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Path:
    """
    Directory for vector files of a specific method.

    Returns: experiments/{experiment}/extraction/{trait}/{model_variant}/vectors/{position}/{component}/{method}/
    """
    base = get('extraction.vectors', experiment=experiment, trait=trait, model_variant=model_variant)
    pos_dir = sanitize_position(position)
    return base / pos_dir / component / method


def get_vector_path(
    experiment: str,
    trait: str,
    method: str,
    layer: int,
    model_variant: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Path:
    """
    Path to extracted vector file.

    Returns: .../vectors/{position}/{component}/{method}/layer{layer}.pt
    """
    return get_vector_dir(experiment, trait, method, model_variant, component, position) / f"layer{layer}.pt"


def get_vector_metadata_path(
    experiment: str,
    trait: str,
    method: str,
    model_variant: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Path:
    """
    Path to vector metadata file (per method directory).

    Returns: .../vectors/{position}/{component}/{method}/metadata.json
    """
    return get_vector_dir(experiment, trait, method, model_variant, component, position) / "metadata.json"


# =============================================================================
# Steering paths
# =============================================================================

def get_steering_dir(
    experiment: str,
    trait: str,
    model_variant: str,
    position: str = "response[:]",
    prompt_set: str = "steering",
) -> Path:
    """
    Directory for steering results.

    Returns: experiments/{experiment}/steering/{trait}/{model_variant}/{position}/{prompt_set}/
    """
    pos_dir = sanitize_position(position)
    return get('steering.prompt_set',
               experiment=experiment,
               trait=trait,
               model_variant=model_variant,
               position=pos_dir,
               prompt_set=prompt_set)


def get_steering_results_path(
    experiment: str,
    trait: str,
    model_variant: str,
    position: str = "response[:]",
    prompt_set: str = "steering",
) -> Path:
    """
    Path to steering results file (JSONL format).

    Returns: .../steering/{trait}/{model_variant}/{position}/{prompt_set}/results.jsonl
    """
    return get_steering_dir(experiment, trait, model_variant, position, prompt_set) / "results.jsonl"


def get_steering_responses_dir(
    experiment: str,
    trait: str,
    model_variant: str,
    position: str = "response[:]",
    prompt_set: str = "steering",
) -> Path:
    """
    Directory for steering response files.

    Returns: .../steering/{trait}/{model_variant}/{position}/{prompt_set}/responses/
    """
    return get_steering_dir(experiment, trait, model_variant, position, prompt_set) / "responses"


def get_steering_response_dir(
    experiment: str,
    trait: str,
    model_variant: str,
    component: str = "residual",
    method: str = "probe",
    position: str = "response[:]",
    prompt_set: str = "steering",
) -> Path:
    """
    Directory for steering response files with component/method.

    Returns: .../responses/{component}/{method}/
    """
    return get_steering_responses_dir(experiment, trait, model_variant, position, prompt_set) / component / method


# =============================================================================
# Ensemble paths
# =============================================================================

def get_ensemble_dir(
    experiment: str,
    trait: str,
    model_variant: str,
) -> Path:
    """
    Directory for ensemble definitions.

    Returns: experiments/{experiment}/steering/{trait}/{model_variant}/ensembles/
    """
    return get('ensemble.base', experiment=experiment, trait=trait, model_variant=model_variant)


def get_ensemble_path(
    experiment: str,
    trait: str,
    model_variant: str,
    ensemble_id: str,
) -> Path:
    """
    Path to ensemble definition file.

    Returns: .../ensembles/{ensemble_id}.json
    """
    return get('ensemble.definition', experiment=experiment, trait=trait, model_variant=model_variant, ensemble_id=ensemble_id)


def get_ensemble_manifest_path(
    experiment: str,
    trait: str,
    model_variant: str,
) -> Path:
    """
    Path to ensemble manifest file.

    Returns: .../ensembles/manifest.json
    """
    return get('ensemble.manifest', experiment=experiment, trait=trait, model_variant=model_variant)


# =============================================================================
# Inference paths
# =============================================================================

def get_inference_dir(
    experiment: str,
    model_variant: str,
) -> Path:
    """
    Base directory for inference data for a model variant.

    Returns: experiments/{experiment}/inference/{model_variant}/
    """
    return get('inference.variant', experiment=experiment, model_variant=model_variant)


def get_inference_raw_dir(
    experiment: str,
    model_variant: str,
    prompt_set: str,
) -> Path:
    """
    Directory for raw activation captures.

    Returns: experiments/{experiment}/inference/{model_variant}/raw/residual/{prompt_set}/
    """
    return get('inference.raw_residual', experiment=experiment, model_variant=model_variant, prompt_set=prompt_set)


def get_inference_responses_dir(
    experiment: str,
    model_variant: str,
    prompt_set: str,
) -> Path:
    """
    Directory for inference response files.

    Returns: experiments/{experiment}/inference/{model_variant}/responses/{prompt_set}/
    """
    return get('inference.responses', experiment=experiment, model_variant=model_variant, prompt_set=prompt_set)


def get_inference_projections_dir(
    experiment: str,
    model_variant: str,
    trait: str,
    prompt_set: str,
) -> Path:
    """
    Directory for trait projection results.

    Returns: experiments/{experiment}/inference/{model_variant}/projections/{trait}/{prompt_set}/
    """
    return get('inference.projections', experiment=experiment, model_variant=model_variant, trait=trait, prompt_set=prompt_set)


# =============================================================================
# Discovery helpers
# =============================================================================

def list_positions(
    experiment: str,
    trait: str,
    model_variant: str,
) -> list[str]:
    """
    Discover available positions for a trait (by scanning vectors directory).

    Returns list of position directory names like ['response_all', 'response_-1']
    """
    base = get('extraction.vectors', experiment=experiment, trait=trait, model_variant=model_variant)
    if not base.exists():
        return []
    return sorted([d.name for d in base.iterdir() if d.is_dir()])


def list_components(
    experiment: str,
    trait: str,
    model_variant: str,
    position: str = "response[:]",
) -> list[str]:
    """
    Discover available components for a trait/position.

    Returns list like ['residual', 'attn_out']
    """
    base = get('extraction.vectors', experiment=experiment, trait=trait, model_variant=model_variant)
    pos_dir = sanitize_position(position)
    comp_base = base / pos_dir
    if not comp_base.exists():
        return []
    return sorted([d.name for d in comp_base.iterdir() if d.is_dir()])


def list_methods(
    experiment: str,
    trait: str,
    model_variant: str,
    component: str = "residual",
    position: str = "response[:]",
) -> list[str]:
    """
    Discover available methods for a trait/position/component.

    Returns list like ['probe', 'mean_diff', 'gradient']
    """
    base = get('extraction.vectors', experiment=experiment, trait=trait, model_variant=model_variant)
    pos_dir = sanitize_position(position)
    method_base = base / pos_dir / component
    if not method_base.exists():
        return []
    return sorted([d.name for d in method_base.iterdir() if d.is_dir()])


def list_layers(
    experiment: str,
    trait: str,
    method: str,
    model_variant: str,
    component: str = "residual",
    position: str = "response[:]",
) -> list[int]:
    """
    Discover available layers for a trait/position/component/method.

    Returns list of layer indices like [0, 1, 14, 15, 16]
    """
    import re
    vector_dir = get_vector_dir(experiment, trait, method, model_variant, component, position)
    if not vector_dir.exists():
        return []

    layers = []
    pattern = re.compile(r'^layer(\d+)\.pt$')
    for f in vector_dir.iterdir():
        match = pattern.match(f.name)
        if match:
            layers.append(int(match.group(1)))
    return sorted(layers)
