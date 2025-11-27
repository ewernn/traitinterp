#!/usr/bin/env python3
"""
Check available data for trait-interp experiments using discovery-based approach.

Input:
    - experiments/{experiment}/ (filesystem scan)
    - config/paths.yaml (schema for required files only)

Output:
    - Console report (or JSON with --json_output)

Discovery conventions:
    - Extraction methods: discovered from vectors/{method}_layer*.pt
    - Analysis categories: discovered from analysis/{category}/
    - Inference types: discovered from inference/raw/{type}/

Usage:
    python analysis/data_checker.py --experiment gemma_2b_cognitive_nov21
    python analysis/data_checker.py --experiment gemma_2b_cognitive_nov21 --json_output
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import yaml
import fire
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set
from enum import Enum


# =============================================================================
# SCHEMA LOADING
# =============================================================================

_schema = None

def _load_schema():
    """Load schema from paths.yaml (cached)."""
    global _schema
    if _schema is None:
        config_path = Path(__file__).parent.parent / "config" / "paths.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        _schema = config.get('schema', {})
    return _schema

def get_schema():
    """Get the data schema."""
    return _load_schema()


# =============================================================================
# DISCOVERY FUNCTIONS
# =============================================================================

def discover_methods(vectors_dir: Path) -> List[str]:
    """Discover extraction methods from actual vector files."""
    if not vectors_dir.exists():
        return []

    methods = set()
    for f in vectors_dir.glob("*_layer*.pt"):
        if "_metadata" not in f.name:
            # Parse method from filename like "probe_layer16.pt"
            method = f.name.rsplit("_layer", 1)[0]
            methods.add(method)
    return sorted(methods)


def discover_analysis_categories(analysis_dir: Path) -> Dict[str, dict]:
    """Discover all analysis categories and their contents."""
    if not analysis_dir.exists():
        return {}

    result = {}
    for item in analysis_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Count files by type
            pngs = list(item.glob("**/*.png"))
            jsons = list(item.glob("**/*.json"))
            pts = list(item.glob("**/*.pt"))
            subdirs = [d.name for d in item.iterdir() if d.is_dir()]

            result[item.name] = {
                "pngs": len(pngs),
                "jsons": len(jsons),
                "pts": len(pts),
                "subdirs": subdirs,
                "total_files": len(pngs) + len(jsons) + len(pts)
            }
    return result


def discover_inference_types(raw_dir: Path) -> Dict[str, dict]:
    """Discover inference capture types (residual, internals, sae, etc.)."""
    if not raw_dir.exists():
        return {}

    result = {}
    for item in raw_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Each subdir is a capture type, containing prompt_set subdirs
            prompt_sets = {}
            for prompt_set_dir in item.iterdir():
                if prompt_set_dir.is_dir():
                    pt_files = list(prompt_set_dir.glob("*.pt"))
                    prompt_sets[prompt_set_dir.name] = len(pt_files)

            result[item.name] = {
                "prompt_sets": prompt_sets,
                "total_files": sum(prompt_sets.values())
            }
    return result


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class Status(str, Enum):
    OK = "ok"
    MISSING = "missing"
    PARTIAL = "partial"
    EMPTY = "empty"


@dataclass
class TraitIntegrity:
    trait: str
    category: str
    status: Status = Status.OK

    # Discovered content
    prompts: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, bool] = field(default_factory=dict)
    responses: Dict[str, bool] = field(default_factory=dict)
    activations: Dict[str, int] = field(default_factory=dict)
    vectors: Dict[str, int] = field(default_factory=dict)
    methods: List[str] = field(default_factory=list)  # Discovered methods

    # Counts
    expected_activations: int = 0

    issues: List[str] = field(default_factory=list)


@dataclass
class InferenceIntegrity:
    prompt_sets: Dict[str, bool] = field(default_factory=dict)
    raw_types: Dict[str, dict] = field(default_factory=dict)  # Discovered capture types
    projections: Dict[str, Dict[str, int]] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)


@dataclass
class AnalysisIntegrity:
    categories: Dict[str, dict] = field(default_factory=dict)  # Discovered categories
    index_exists: bool = False
    total_files: int = 0
    issues: List[str] = field(default_factory=list)


@dataclass
class ExperimentIntegrity:
    experiment: str
    n_layers: int
    methods: List[str]  # Discovered from first trait with vectors

    traits: List[TraitIntegrity] = field(default_factory=list)
    inference: Optional[InferenceIntegrity] = None
    analysis: Optional[AnalysisIntegrity] = None
    evaluation_exists: bool = False

    summary: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# CHECK FUNCTIONS
# =============================================================================

def check_extraction_trait(
    trait_dir: Path,
    category: str,
    trait_name: str,
    n_layers: int,
    schema: dict
) -> TraitIntegrity:
    """Check all extraction files for a single trait using discovery."""

    result = TraitIntegrity(
        trait=f"{category}/{trait_name}",
        category=category,
    )

    # Get file lists from schema (single source of truth)
    extraction_schema = schema.get('extraction', {})
    required = schema.get('required', {}).get('extraction', [])

    # Check prompt files (from schema)
    prompt_files = extraction_schema.get('prompts', [])
    for f in prompt_files:
        result.prompts[f] = (trait_dir / f).exists()
        if not result.prompts[f] and f in required:
            result.issues.append(f"Missing required: {f}")

    # Check metadata files (from schema)
    metadata_files = extraction_schema.get('metadata', [])
    for f in metadata_files:
        result.metadata[f] = (trait_dir / f).exists()

    # Check response files (from schema)
    response_paths = extraction_schema.get('responses', [])
    for resp_path in response_paths:
        result.responses[resp_path] = (trait_dir / resp_path).exists()

    # Check activation files (single all_layers.pt format)
    activations_dir = trait_dir / "activations"
    val_activations_dir = trait_dir / "val_activations"

    result.activations["metadata"] = (activations_dir / "metadata.json").exists()
    result.activations["all_layers"] = (activations_dir / "all_layers.pt").exists()
    result.activations["val_all_layers"] = (val_activations_dir / "all_layers.pt").exists()

    # Expected: all_layers.pt exists (val is optional)
    result.expected_activations = 1  # Just need all_layers.pt

    # Discover extraction methods from vector files
    vectors_dir = trait_dir / "vectors"
    result.methods = discover_methods(vectors_dir)

    # Count vectors per discovered method
    for method in result.methods:
        pt_files = list(vectors_dir.glob(f"{method}_layer*.pt"))
        pt_files = [f for f in pt_files if "_metadata" not in f.name]
        meta_files = list(vectors_dir.glob(f"{method}_layer*_metadata.json"))

        result.vectors[f"{method}_pt"] = len(pt_files)
        result.vectors[f"{method}_meta"] = len(meta_files)

    # Determine status based on required files only
    has_required = all(result.prompts.get(f, False) for f in required)
    has_responses = any(result.responses.values())
    has_vectors = len(result.methods) > 0

    if has_required and has_responses and has_vectors:
        result.status = Status.OK
    elif has_required and (has_responses or has_vectors):
        result.status = Status.PARTIAL
    elif has_required:
        result.status = Status.PARTIAL
    else:
        result.status = Status.EMPTY

    return result


def check_inference(exp_dir: Path, traits: List[str]) -> InferenceIntegrity:
    """Check inference directory using discovery."""

    result = InferenceIntegrity()
    inference_dir = exp_dir / "inference"

    if not inference_dir.exists():
        result.issues.append("No inference directory")
        return result

    # Discover prompt sets
    prompts_dir = inference_dir / "prompts"
    if prompts_dir.exists():
        for prompt_file in prompts_dir.glob("*.json"):
            result.prompt_sets[prompt_file.stem] = True

    # Discover raw capture types (residual, internals, sae, etc.)
    raw_dir = inference_dir / "raw"
    result.raw_types = discover_inference_types(raw_dir)

    # Check per-trait projections
    for trait in traits:
        trait_inference_dir = inference_dir / trait / "residual_stream"
        if trait_inference_dir.exists():
            result.projections[trait] = {}
            for prompt_set_dir in trait_inference_dir.iterdir():
                if prompt_set_dir.is_dir():
                    json_files = list(prompt_set_dir.glob("*.json"))
                    result.projections[trait][prompt_set_dir.name] = len(json_files)

    return result


def check_analysis(exp_dir: Path) -> AnalysisIntegrity:
    """Check analysis directory using discovery."""

    result = AnalysisIntegrity()
    analysis_dir = exp_dir / "analysis"

    if not analysis_dir.exists():
        result.issues.append("No analysis directory")
        return result

    # Check for index file
    result.index_exists = (analysis_dir / "index.json").exists()

    # Discover all analysis categories
    result.categories = discover_analysis_categories(analysis_dir)

    # Calculate totals
    result.total_files = sum(cat["total_files"] for cat in result.categories.values())

    return result


def check_experiment(experiment: str) -> ExperimentIntegrity:
    """Check all data integrity for an experiment using discovery."""

    schema = get_schema()
    n_layers = schema.get('n_layers', 26)

    exp_dir = Path("experiments") / experiment
    if not exp_dir.exists():
        raise ValueError(f"Experiment not found: {experiment}")

    result = ExperimentIntegrity(
        experiment=experiment,
        n_layers=n_layers,
        methods=[],  # Will be populated from discovered methods
    )

    # Find all traits (discovery-based)
    extraction_dir = exp_dir / "extraction"
    traits = []
    all_methods = set()

    if extraction_dir.exists():
        for category_dir in extraction_dir.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith('.'):
                # Skip non-trait directories
                if category_dir.name in ['extraction_evaluation.json']:
                    continue

                for trait_dir in category_dir.iterdir():
                    if trait_dir.is_dir():
                        trait_result = check_extraction_trait(
                            trait_dir,
                            category_dir.name,
                            trait_dir.name,
                            n_layers,
                            schema
                        )
                        result.traits.append(trait_result)
                        traits.append(f"{category_dir.name}/{trait_dir.name}")

                        # Collect all discovered methods
                        all_methods.update(trait_result.methods)

    # Set discovered methods
    result.methods = sorted(all_methods)

    # Check inference
    result.inference = check_inference(exp_dir, traits)

    # Check analysis (NEW)
    result.analysis = check_analysis(exp_dir)

    # Check evaluation
    evaluation_file = exp_dir / "extraction" / "extraction_evaluation.json"
    result.evaluation_exists = evaluation_file.exists()

    # Compute summary
    result.summary = {
        "total_traits": len(result.traits),
        "ok": sum(1 for t in result.traits if t.status == Status.OK),
        "partial": sum(1 for t in result.traits if t.status == Status.PARTIAL),
        "empty": sum(1 for t in result.traits if t.status == Status.EMPTY),
        "missing": sum(1 for t in result.traits if t.status == Status.MISSING),
        "methods_discovered": len(result.methods),
        "analysis_categories": len(result.analysis.categories) if result.analysis else 0,
    }

    return result


# =============================================================================
# OUTPUT
# =============================================================================

def print_report(result: ExperimentIntegrity):
    """Print human-readable report."""

    print(f"\n{'='*60}")
    print(f"Data Integrity Report: {result.experiment}")
    print(f"{'='*60}")
    print(f"Config: {result.n_layers} layers")
    print(f"Discovered methods: {', '.join(result.methods) if result.methods else 'none'}")
    print()

    # Summary
    print("SUMMARY")
    print("-" * 40)
    print(f"  Total traits: {result.summary['total_traits']}")
    print(f"  OK:       {result.summary['ok']}")
    print(f"  Partial:  {result.summary['partial']}")
    print(f"  Empty:    {result.summary['empty']}")
    print()

    # Per-trait details
    print("EXTRACTION (per trait)")
    print("-" * 40)

    for trait in sorted(result.traits, key=lambda t: t.trait):
        status_icon = {
            Status.OK: "OK",
            Status.PARTIAL: "~~",
            Status.EMPTY: "XX",
            Status.MISSING: "XX",
        }[trait.status]

        print(f"\n[{status_icon}] {trait.trait}")

        # Show discovered methods for this trait
        if trait.methods:
            print(f"     Methods: {', '.join(trait.methods)}")

        # Show file counts
        prompts_ok = sum(trait.prompts.values())
        responses_ok = sum(trait.responses.values())
        has_activations = trait.activations.get("all_layers", False)
        has_val_activations = trait.activations.get("val_all_layers", False)
        acts_str = "✓" if has_activations else "✗"
        if has_val_activations:
            acts_str += "+val"
        total_vectors = sum(v for k, v in trait.vectors.items() if k.endswith("_pt"))

        print(f"     Prompts: {prompts_ok}/4 | Responses: {responses_ok}/4 | "
              f"Activations: {acts_str} | Vectors: {total_vectors}")

        if trait.issues:
            for issue in trait.issues:
                print(f"     ! {issue}")

    # Inference
    print(f"\n{'='*60}")
    print("INFERENCE")
    print("-" * 40)

    if result.inference:
        if result.inference.prompt_sets:
            print(f"  Prompt sets: {', '.join(result.inference.prompt_sets.keys())}")

        if result.inference.raw_types:
            print("  Raw capture types (discovered):")
            for capture_type, info in result.inference.raw_types.items():
                print(f"    {capture_type}: {info['total_files']} files across {len(info['prompt_sets'])} prompt sets")

        if result.inference.projections:
            print(f"  Projections: {len(result.inference.projections)} traits with data")
    else:
        print("  No inference data")

    # Analysis (NEW)
    print(f"\n{'='*60}")
    print("ANALYSIS (discovered)")
    print("-" * 40)

    if result.analysis and result.analysis.categories:
        print(f"  Index file: {'exists' if result.analysis.index_exists else 'missing'}")
        print(f"  Categories discovered: {len(result.analysis.categories)}")
        for cat_name, cat_info in sorted(result.analysis.categories.items()):
            subdirs_str = f" ({', '.join(cat_info['subdirs'])})" if cat_info['subdirs'] else ""
            print(f"    {cat_name}: {cat_info['pngs']} png, {cat_info['jsons']} json, {cat_info['pts']} pt{subdirs_str}")
        print(f"  Total analysis files: {result.analysis.total_files}")
    else:
        print("  No analysis data")

    # Evaluation
    print(f"\n{'='*60}")
    print("EXTRACTION EVALUATION")
    print("-" * 40)
    print(f"  extraction_evaluation.json: {'exists' if result.evaluation_exists else 'missing'}")
    print()


def main(
    experiment: str,
    json_output: bool = False,
):
    """
    Check data integrity for an experiment using discovery.

    Args:
        experiment: Experiment name
        json_output: Output as JSON instead of human-readable
    """

    result = check_experiment(experiment)

    if json_output:
        # Convert to JSON-serializable dict
        output = asdict(result)
        # Convert Status enums to strings
        for trait in output["traits"]:
            trait["status"] = trait["status"].value if hasattr(trait["status"], "value") else trait["status"]
        print(json.dumps(output))
    else:
        print_report(result)


if __name__ == "__main__":
    fire.Fire(main)
