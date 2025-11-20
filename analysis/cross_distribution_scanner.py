#!/usr/bin/env python3
"""
Cross-Distribution Data Scanner

Scans the experiments directory to identify what cross-distribution testing data
is available for each trait. Generates a comprehensive JSON index for the visualizer.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def check_trait_data_availability(trait_path: Path) -> Dict[str, any]:
    """
    Check what data is available for a trait.

    Returns:
        Dictionary with data availability information:
        - has_instruction: bool
        - has_natural: bool
        - has_vectors: Dict[method, List[layers]]
        - has_activations: bool
    """
    extraction_dir = trait_path / "extraction"

    # Check for instruction-based data
    responses_dir = extraction_dir / "responses"
    has_instruction = False
    if responses_dir.exists():
        has_instruction = (
            (responses_dir / "pos.csv").exists() or
            (responses_dir / "pos.json").exists()
        )

    # Check for natural elicitation data
    trait_name_natural = f"{trait_path.name}_natural"
    natural_path = trait_path.parent / trait_name_natural
    has_natural = False
    if natural_path.exists():
        natural_responses = natural_path / "extraction" / "responses"
        if natural_responses.exists():
            has_natural = (
                (natural_responses / "pos.csv").exists() or
                (natural_responses / "pos.json").exists()
            )

    # Check for vectors
    vectors_dir = extraction_dir / "vectors"
    available_vectors = {}
    if vectors_dir.exists():
        methods = ["mean_diff", "probe", "ica", "gradient"]
        for method in methods:
            layers = []
            for layer in range(26):
                vector_file = vectors_dir / f"{method}_layer{layer}.pt"
                if vector_file.exists():
                    layers.append(layer)
            if layers:
                available_vectors[method] = layers

    # Check for natural vectors
    natural_vectors = {}
    if has_natural:
        natural_vectors_dir = natural_path / "extraction" / "vectors"
        if natural_vectors_dir.exists():
            methods = ["mean_diff", "probe", "ica", "gradient"]
            for method in methods:
                layers = []
                for layer in range(26):
                    vector_file = natural_vectors_dir / f"{method}_layer{layer}.pt"
                    if vector_file.exists():
                        layers.append(layer)
                if layers:
                    natural_vectors[method] = layers

    # Check for activations
    activations_dir = extraction_dir / "activations"
    has_activations = activations_dir.exists() and any(activations_dir.glob("*.pt"))

    natural_activations = False
    if has_natural:
        natural_acts_dir = natural_path / "extraction" / "activations"
        natural_activations = natural_acts_dir.exists() and any(natural_acts_dir.glob("*.pt"))

    return {
        "has_instruction": has_instruction,
        "has_natural": has_natural,
        "instruction_vectors": available_vectors,
        "natural_vectors": natural_vectors,
        "has_instruction_activations": has_activations,
        "has_natural_activations": natural_activations,
    }


def determine_cross_dist_capability(data_info: Dict) -> Dict[str, bool]:
    """
    Determine what cross-distribution tests are possible.

    Returns:
        {
            'inst_to_inst': bool,  # Can test Instruction → Instruction
            'inst_to_nat': bool,   # Can test Instruction → Natural
            'nat_to_inst': bool,   # Can test Natural → Instruction
            'nat_to_nat': bool,    # Can test Natural → Natural
        }
    """
    has_inst_vectors = len(data_info["instruction_vectors"]) > 0
    has_nat_vectors = len(data_info["natural_vectors"]) > 0
    has_inst_acts = data_info["has_instruction_activations"]
    has_nat_acts = data_info["has_natural_activations"]

    return {
        "inst_to_inst": has_inst_vectors and has_inst_acts,
        "inst_to_nat": has_inst_vectors and has_nat_acts,
        "nat_to_inst": has_nat_vectors and has_inst_acts,
        "nat_to_nat": has_nat_vectors and has_nat_acts,
    }


def estimate_separability(trait_name: str) -> Optional[str]:
    """
    Estimate trait separability based on known results.

    Returns:
        'low', 'moderate', 'high', or None if unknown
    """
    separability_map = {
        # Low separability
        "uncertainty_calibration": "low",
        "cognitive_load": "low",

        # Moderate separability
        "refusal": "moderate",
        "politeness": "moderate",

        # High separability
        "emotional_valence": "high",
        "formality": "high",
        "sentiment": "high",
    }
    return separability_map.get(trait_name)


def load_best_accuracies(trait_name: str, experiment_path: Path) -> Optional[Dict]:
    """
    Load best accuracies from validation results if available.

    Args:
        trait_name: Name of the trait
        experiment_path: Path to experiment directory

    Returns:
        Dict with quadrant -> {method: {acc, layer}} mapping
    """
    results_dir = experiment_path / "validation"
    if not results_dir.exists():
        return None

    result_files = [
        results_dir / f'{trait_name}_full_4x4_results.json',
        results_dir / f'{trait_name}_cross_dist_results.json',
    ]

    for result_file in result_files:
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)

                # Extract best accuracies for each quadrant
                best_scores = {}
                quadrants = data.get('quadrants', {})

                for quad_key, quad_data in quadrants.items():
                    quad_scores = {}
                    methods = quad_data.get('methods', {})

                    for method_name, method_data in methods.items():
                        best_acc = method_data.get('best_accuracy', 0)
                        best_layer = method_data.get('best_layer', 0)
                        quad_scores[method_name] = {
                            'accuracy': round(best_acc * 100, 1),
                            'layer': best_layer
                        }

                    best_scores[quad_key] = quad_scores

                return best_scores
            except Exception as e:
                print(f"  Warning: Could not load results for {trait_name}: {e}")
                return None

    return None


def scan_experiment(experiment_path: Path) -> Dict:
    """
    Scan an experiment directory for cross-distribution data.

    Expected structure:
        experiments/{exp}/extraction/{category}/{trait}/
        experiments/{exp}/validation/{trait}_full_4x4_results.json

    Returns:
        Dictionary with experiment analysis
    """
    traits_data = []

    # Check for extraction directory
    extraction_dir = experiment_path / "extraction"
    if not extraction_dir.exists():
        raise FileNotFoundError(
            f"Extraction directory not found: {extraction_dir}\n"
            f"Expected structure: {experiment_path}/extraction/{{category}}/{{trait}}/"
        )

    # Categories to check
    categories = ['behavioral', 'cognitive', 'stylistic', 'alignment']

    # Find all trait directories in categorized structure
    trait_dirs = []
    for category in categories:
        category_path = extraction_dir / category
        if category_path.exists():
            trait_dirs.extend(sorted(category_path.iterdir()))

    # Process trait directories
    for trait_dir in trait_dirs:
        if not trait_dir.is_dir():
            continue
        if trait_dir.name.startswith('.'):
            continue
        if trait_dir.name.endswith('_natural'):
            # Skip natural variants - we'll find them when scanning the base trait
            continue

        extraction_dir = trait_dir / "extraction"
        if not extraction_dir.exists():
            continue

        # Get data availability
        data_info = check_trait_data_availability(trait_dir)

        # Determine cross-distribution capabilities
        cross_dist = determine_cross_dist_capability(data_info)

        # Count available quadrants
        available_quadrants = sum(1 for v in cross_dist.values() if v)

        # Estimate separability
        separability = estimate_separability(trait_dir.name)

        # Load best accuracies from validation results if available
        best_accuracies = load_best_accuracies(trait_dir.name, experiment_path)

        trait_data = {
            "name": trait_dir.name,
            "separability": separability,
            "data": data_info,
            "cross_distribution": cross_dist,
            "available_quadrants": available_quadrants,
            "is_complete_4x4": available_quadrants == 4,
            "best_accuracies": best_accuracies,
        }

        traits_data.append(trait_data)

    # Compute statistics
    total_traits = len(traits_data)
    complete_4x4 = sum(1 for t in traits_data if t["is_complete_4x4"])
    partial = sum(1 for t in traits_data if t["available_quadrants"] > 0 and not t["is_complete_4x4"])
    no_cross_dist = sum(1 for t in traits_data if t["available_quadrants"] == 0)

    return {
        "experiment": experiment_path.name,
        "traits": traits_data,
        "statistics": {
            "total_traits": total_traits,
            "complete_4x4": complete_4x4,
            "partial": partial,
            "no_cross_dist": no_cross_dist,
        }
    }


def main():
    """Scan all experiments and generate cross-distribution index."""
    experiments_dir = Path("experiments")

    if not experiments_dir.exists():
        print("❌ experiments/ directory not found")
        return

    all_experiments = []

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
            continue

        print(f"📊 Scanning {exp_dir.name}...")
        exp_data = scan_experiment(exp_dir)
        all_experiments.append(exp_data)

        # Print summary
        stats = exp_data["statistics"]
        print(f"  ✓ {stats['total_traits']} traits")
        print(f"    - {stats['complete_4x4']} with complete 4×4 matrices")
        print(f"    - {stats['partial']} with partial data")
        print(f"    - {stats['no_cross_dist']} with no cross-distribution data")

    # Save index to each experiment's validation directory
    for exp_data in all_experiments:
        exp_name = exp_data["experiment"]
        exp_path = experiments_dir / exp_name / "validation"
        exp_path.mkdir(parents=True, exist_ok=True)

        output_path = exp_path / "data_index.json"
        with open(output_path, 'w') as f:
            json.dump({
                "generated": "2025-11-20",
                "experiment": exp_name,
                "traits": exp_data["traits"]
            }, f, indent=2)

        print(f"  ✓ Saved index: {output_path}")


if __name__ == "__main__":
    main()
