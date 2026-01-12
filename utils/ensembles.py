"""
Ensemble I/O for multi-vector trait combinations.

Input: experiment, trait, model_variant identifiers
Output: Ensemble definitions and manifest

Usage:
    from utils.ensembles import create_ensemble, load_ensemble, get_best_ensemble

    # Create ensemble from VectorSpecs
    ensemble = create_ensemble(
        experiment, trait, model_variant,
        specs=[VectorSpec(11, 'attn_contribution', 'response[:5]', 'probe'),
               VectorSpec(12, 'attn_contribution', 'response[:5]', 'probe')],
        coefficients=[60.0, 40.0],
        coefficient_source='individual_scaled'
    )

    # Load existing
    ensemble = load_ensemble(experiment, trait, model_variant, '001')

    # Find best
    best = get_best_ensemble(experiment, trait, model_variant)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from core.types import VectorSpec, ProjectionConfig
from utils.paths import (
    get_ensemble_dir,
    get_ensemble_path,
    get_ensemble_manifest_path,
)
from utils.vectors import MIN_COHERENCE


def _next_ensemble_id(experiment: str, trait: str, model_variant: str) -> str:
    """Generate next sequential ensemble ID (0001, 0002, etc.)."""
    ensemble_dir = get_ensemble_dir(experiment, trait, model_variant)
    if not ensemble_dir.exists():
        return "0001"

    existing = [f.stem for f in ensemble_dir.glob("*.json") if f.stem != "manifest"]
    if not existing:
        return "0001"

    # Find highest numeric ID and increment
    numeric_ids = [int(x) for x in existing if x.isdigit()]
    if not numeric_ids:
        return "0001"
    return f"{max(numeric_ids) + 1:04d}"


def _generate_specs_summary(specs: List[Dict]) -> str:
    """Generate human-readable summary of ensemble specs."""
    layers = [f"L{s['layer']}" for s in specs]
    components = set(s['component'] for s in specs)

    if len(components) == 1:
        return f"{'+'.join(layers)} {list(components)[0]}"
    else:
        return f"{'+'.join(layers)} mixed"


def create_ensemble(
    experiment: str,
    trait: str,
    model_variant: str,
    specs: List[VectorSpec],
    coefficients: List[float],
    coefficient_source: str = "manual",
    ensemble_id: Optional[str] = None,
) -> Dict:
    """
    Create a new ensemble definition.

    Args:
        experiment: Experiment name
        trait: Trait path (category/trait)
        model_variant: Model variant name
        specs: List of VectorSpecs (must have same position and method)
        coefficients: Weight for each spec
        coefficient_source: One of: activation_magnitude, individual_scaled, optimized, manual
        ensemble_id: Optional ID (auto-generated if not provided)

    Returns:
        Ensemble definition dict

    Raises:
        ValueError: If specs have different positions or methods
    """
    if len(specs) != len(coefficients):
        raise ValueError(f"specs ({len(specs)}) and coefficients ({len(coefficients)}) must have same length")

    if len(specs) < 2:
        raise ValueError("Ensemble must have at least 2 specs")

    # Enforce uniform position and method
    positions = set(s.position for s in specs)
    methods = set(s.method for s in specs)

    if len(positions) > 1:
        raise ValueError(f"All specs must have same position, got: {positions}")
    if len(methods) > 1:
        raise ValueError(f"All specs must have same method, got: {methods}")

    if ensemble_id is None:
        ensemble_id = _next_ensemble_id(experiment, trait, model_variant)

    ensemble = {
        "id": ensemble_id,
        "created": datetime.now().isoformat(),
        "specs": [
            {"layer": s.layer, "component": s.component, "position": s.position, "method": s.method}
            for s in specs
        ],
        "coefficients": coefficients,
        "coefficient_source": coefficient_source,
        "steering_results": None,
    }

    return ensemble


def save_ensemble(experiment: str, trait: str, model_variant: str, ensemble: Dict) -> Path:
    """
    Save ensemble definition to disk.

    Args:
        experiment: Experiment name
        trait: Trait path
        model_variant: Model variant name
        ensemble: Ensemble definition dict

    Returns:
        Path to saved file
    """
    ensemble_id = ensemble["id"]
    path = get_ensemble_path(experiment, trait, model_variant, ensemble_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(ensemble, f, indent=2)

    # Update manifest
    update_manifest(experiment, trait, model_variant, ensemble)

    return path


def load_ensemble(experiment: str, trait: str, model_variant: str, ensemble_id: str) -> Dict:
    """
    Load ensemble definition from disk.

    Args:
        experiment: Experiment name
        trait: Trait path
        model_variant: Model variant name
        ensemble_id: Ensemble ID (e.g., '001')

    Returns:
        Ensemble definition dict

    Raises:
        FileNotFoundError: If ensemble doesn't exist
    """
    path = get_ensemble_path(experiment, trait, model_variant, ensemble_id)
    if not path.exists():
        raise FileNotFoundError(f"Ensemble not found: {path}")

    with open(path) as f:
        return json.load(f)


def update_manifest(experiment: str, trait: str, model_variant: str, ensemble: Dict) -> None:
    """
    Update manifest.json with ensemble info.

    Called automatically by save_ensemble().
    """
    manifest_path = get_ensemble_manifest_path(experiment, trait, model_variant)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"best": None, "ensembles": {}}

    ensemble_id = ensemble["id"]
    specs_summary = _generate_specs_summary(ensemble["specs"])

    # Extract steering results if available
    results = ensemble.get("steering_results")
    delta = results.get("delta") if results else None
    coherence = results.get("coherence_mean") if results else None

    manifest["ensembles"][ensemble_id] = {
        "specs_summary": specs_summary,
        "delta": delta,
        "coherence": coherence,
    }

    # Update best if this ensemble has better delta
    if delta is not None:
        current_best = manifest.get("best")
        if current_best is None:
            manifest["best"] = ensemble_id
        else:
            current_best_delta = manifest["ensembles"].get(current_best, {}).get("delta")
            if current_best_delta is None or delta > current_best_delta:
                manifest["best"] = ensemble_id

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def list_ensembles(experiment: str, trait: str, model_variant: str) -> List[str]:
    """
    List available ensemble IDs for a trait.

    Returns:
        List of ensemble IDs like ['001', '002']
    """
    ensemble_dir = get_ensemble_dir(experiment, trait, model_variant)
    if not ensemble_dir.exists():
        return []

    return sorted([
        f.stem for f in ensemble_dir.glob("*.json")
        if f.stem != "manifest"
    ])


def get_best_ensemble(
    experiment: str,
    trait: str,
    model_variant: str,
    min_coherence: float = MIN_COHERENCE,
) -> Optional[Dict]:
    """
    Find best performing ensemble for a trait.

    Args:
        experiment: Experiment name
        trait: Trait path
        min_coherence: Minimum coherence threshold

    Returns:
        Best ensemble definition, or None if no evaluated ensembles
    """
    manifest_path = get_ensemble_manifest_path(experiment, trait, model_variant)
    if not manifest_path.exists():
        return None

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Find best ensemble meeting coherence threshold
    best_id = None
    best_delta = None

    for eid, info in manifest.get("ensembles", {}).items():
        delta = info.get("delta")
        coherence = info.get("coherence")

        if delta is None or coherence is None:
            continue

        if coherence < min_coherence:
            continue

        if best_delta is None or delta > best_delta:
            best_delta = delta
            best_id = eid

    if best_id is None:
        return None

    return load_ensemble(experiment, trait, model_variant, best_id)


def ensemble_to_projection_config(ensemble: Dict) -> ProjectionConfig:
    """
    Convert ensemble definition to ProjectionConfig for steering/projection.

    Args:
        ensemble: Ensemble definition dict

    Returns:
        ProjectionConfig with weighted VectorSpecs
    """
    specs = []
    for spec_dict, coef in zip(ensemble["specs"], ensemble["coefficients"]):
        specs.append(VectorSpec(
            layer=spec_dict["layer"],
            component=spec_dict["component"],
            position=spec_dict["position"],
            method=spec_dict["method"],
            weight=coef,
        ))
    return ProjectionConfig(vectors=specs)


def update_ensemble_steering_results(
    experiment: str,
    trait: str,
    model_variant: str,
    ensemble_id: str,
    baseline: float,
    trait_mean: float,
    coherence_mean: float,
) -> None:
    """
    Update an ensemble's steering results after evaluation.

    Args:
        experiment: Experiment name
        trait: Trait path
        model_variant: Model variant name
        ensemble_id: Ensemble ID
        baseline: Baseline trait score
        trait_mean: Mean trait score with steering
        coherence_mean: Mean coherence score
    """
    ensemble = load_ensemble(experiment, trait, model_variant, ensemble_id)

    ensemble["steering_results"] = {
        "baseline": baseline,
        "trait_mean": trait_mean,
        "delta": trait_mean - baseline,
        "coherence_mean": coherence_mean,
        "timestamp": datetime.now().isoformat(),
    }

    save_ensemble(experiment, trait, model_variant, ensemble)
