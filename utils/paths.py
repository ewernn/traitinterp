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
    # Returns: ModelVariant(name='rm_lora', model='...', lora='...')

    default = get_default_variant('rm_syco', mode='extraction')
    # Returns: 'base'
"""

import json
import os
import yaml
from pathlib import Path
from typing import Optional

from core.types import ModelVariant, SteeringEntry

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


def content_hash(path) -> str:
    """SHA256 hash of file contents. Returns hex string."""
    import hashlib
    path = Path(path)
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_use_chat_template(experiment: str, tokenizer) -> bool:
    """Resolve whether to use chat template: experiment config > tokenizer auto-detect."""
    config = load_experiment_config(experiment)
    use_chat_template = config.get('use_chat_template')
    if use_chat_template is None:
        use_chat_template = tokenizer.chat_template is not None
    return use_chat_template


def get_model_variant(
    experiment: str,
    variant: Optional[str] = None,
    mode: str = "application"
) -> ModelVariant:
    """
    Resolve variant name to full config.

    Args:
        experiment: Experiment name
        variant: Variant name (e.g., 'base', 'instruct', 'rm_lora'), or None for default
        mode: "extraction" or "application" (determines which default to use)

    Returns:
        ModelVariant(name, model, lora) namedtuple

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
    return ModelVariant(
        name=variant,
        model=variant_config['model'],
        lora=variant_config.get('lora'),
    )


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

def discover_traits(category: str = None) -> list[str]:
    """
    Find trait definitions in datasets/traits/ recursively.

    A trait directory is identified by having positive.txt or positive.jsonl.
    Supports any nesting depth (e.g., base/emotion_set/sycophancy).

    Args:
        category: Optional top-level filter (e.g., 'base', 'starter_traits')

    Returns:
        List of trait paths like ['base/emotion_set/sycophancy', 'starter_traits/formal']
    """
    traits_dir = get('datasets.traits')
    if not traits_dir.is_dir():
        return []

    search_root = traits_dir / category if category else traits_dir
    if not search_root.is_dir():
        return []

    traits = []
    for dirpath, dirnames, filenames in os.walk(search_root):
        # Skip archive directories
        dirnames[:] = [d for d in dirnames if d != 'archive' and not d.startswith('.')]
        if 'positive.txt' in filenames or 'positive.jsonl' in filenames:
            rel = Path(dirpath).relative_to(traits_dir)
            traits.append(str(rel))

    return sorted(traits)


def discover_extracted_traits(experiment: str, model_variant: str = None) -> list[tuple[str, str]]:
    """
    Find traits with extracted vectors in experiments/{exp}/extraction/.

    Walks recursively — a trait directory is identified by having a model_variant
    subdir containing vectors/layer*.pt files. Supports any nesting depth.

    Args:
        experiment: Experiment name
        model_variant: Model variant to check (if None, checks all variants)

    Returns:
        List of (category, trait_name) tuples. For multi-level paths like
        base/emotion_set/sycophancy, category='base/emotion_set', trait_name='sycophancy'.
    """
    extraction_dir = get('extraction.base', experiment=experiment)
    if not extraction_dir.exists():
        return []

    traits = []
    seen = set()

    for pt_file in extraction_dir.rglob('layer*.pt'):
        # Walk up to find the vectors/ dir, then the variant dir, then the trait dir
        vectors_dir = pt_file.parent
        while vectors_dir.name != 'vectors' and vectors_dir != extraction_dir:
            vectors_dir = vectors_dir.parent
        if vectors_dir.name != 'vectors':
            continue

        variant_dir = vectors_dir.parent
        if model_variant and variant_dir.name != model_variant:
            continue

        trait_dir = variant_dir.parent
        rel = trait_dir.relative_to(extraction_dir)
        trait_path = str(rel)

        if trait_path not in seen:
            seen.add(trait_path)
            parts = rel.parts
            category = '/'.join(parts[:-1]) if len(parts) > 1 else parts[0]
            trait_name = parts[-1]
            traits.append((category, trait_name))

    return sorted(traits)


def discover_steering_entries(experiment: str) -> list[SteeringEntry]:
    """
    Find all steering results in experiments/{exp}/steering/.

    Path structure: steering/{trait...}/{model_variant}/{position}/{prompt_set...}/results.jsonl
    Trait can be any depth (e.g., emotion_set/spite or base/emotion_set/spite).
    The model_variant is identified by the position dir pattern (response__N or prompt_-N).

    Returns:
        List of SteeringEntry dataclasses
    """
    import re
    steering_dir = get('steering.base', experiment=experiment)
    if not steering_dir.exists():
        return []

    position_pattern = re.compile(r'^(response_|prompt_)')

    entries = []
    for results_file in steering_dir.rglob('results.jsonl'):
        rel_path = results_file.parent.relative_to(steering_dir)
        parts = rel_path.parts

        # Find the position part (response__5, prompt_-1, etc.) to split trait from metadata
        pos_idx = None
        for i, part in enumerate(parts):
            if position_pattern.match(part):
                pos_idx = i
                break

        if pos_idx is None or pos_idx < 2:
            continue

        # Everything before model_variant is the trait path
        # model_variant is one before position
        trait = '/'.join(parts[:pos_idx - 1])
        model_variant = parts[pos_idx - 1]
        position = parts[pos_idx]
        prompt_set = '/'.join(parts[pos_idx + 1:])

        entries.append(SteeringEntry(
            trait=trait,
            model_variant=model_variant,
            position=position,
            prompt_set=prompt_set,
            full_path=str(rel_path),
        ))

    return entries


# =============================================================================
# Position helpers
# =============================================================================

def sanitize_position(position: str) -> str:
    """
    Convert position string to filesystem-safe directory name.

    Examples:
        response[:]  -> response_all
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



# =============================================================================
# Discovery helpers
# =============================================================================

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


# =============================================================================
# Model diff paths
# =============================================================================

def get_model_diff_dir(
    experiment: str,
    variant_a: str,
    variant_b: str,
    prompt_set: str,
) -> Path:
    """
    Directory for model diff results.

    Returns: experiments/{experiment}/model_diff/{variant_a}_vs_{variant_b}/{prompt_set}/
    """
    return get('model_diff.comparison',
               experiment=experiment,
               variant_a=variant_a,
               variant_b=variant_b,
               prompt_set=prompt_set)
