#!/usr/bin/env python3
"""
Migration script: Reorganize inference data from old structure to new structure.

Old structure:
  inference/{trait}/projections/residual_stream_activations/prompt_N.json
  inference/{trait}/projections/layer_internal_states/prompt_N_layer16.json

New structure:
  inference/{trait}/residual_stream/{prompt_set}/{id}.json
  inference/{trait}/layer_internals/{prompt_set}/{id}_L{layer}.json

The mapping from old prompt_N to new {prompt_set}/{id} is built by parsing
the original main_prompts.txt file to understand which line produced which index.
"""

import os
import re
import shutil
import argparse
from pathlib import Path


def parse_main_prompts(prompts_file: Path) -> dict:
    """
    Parse main_prompts.txt and return mapping from old index to (prompt_set, new_id).

    The old system used line numbers (skipping comments/blanks) as indices.
    """
    mapping = {}
    current_section = None
    section_counter = {}
    old_index = 0

    # Section markers and their new names
    section_map = {
        "SECTION A": "single_trait",
        "SECTION B": "multi_trait",
        "SECTION C": "dynamic",
        "SECTION D": "adversarial",
        "SECTION E": "baseline",
        "SECTION F": "real_world",
    }

    with open(prompts_file, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check for section headers
            for section_key, section_name in section_map.items():
                if section_key in line:
                    current_section = section_name
                    if section_name not in section_counter:
                        section_counter[section_name] = 0
                    break

            # Skip comment lines (but not prompt ID comments like "# A1:")
            if line.startswith('#'):
                continue

            # This is an actual prompt line
            if current_section:
                section_counter[current_section] += 1
                new_id = section_counter[current_section]
                mapping[old_index] = (current_section, new_id)
                old_index += 1

    return mapping


def migrate_experiment(experiment_dir: Path, mapping: dict, dry_run: bool = True):
    """
    Migrate all inference data in an experiment to the new structure.
    """
    inference_dir = experiment_dir / "inference"
    if not inference_dir.exists():
        print(f"No inference directory found at {inference_dir}")
        return

    # Find all trait directories (skip 'raw' and 'prompts')
    skip_dirs = {'raw', 'prompts', 'raw_activations'}

    for category_dir in inference_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name in skip_dirs:
            continue

        for trait_dir in category_dir.iterdir():
            if not trait_dir.is_dir():
                continue

            projections_dir = trait_dir / "projections"
            if not projections_dir.exists():
                continue

            trait_path = f"{category_dir.name}/{trait_dir.name}"
            print(f"\nMigrating trait: {trait_path}")

            # Migrate residual_stream_activations -> residual_stream/{prompt_set}/
            old_residual = projections_dir / "residual_stream_activations"
            if old_residual.exists():
                migrate_residual_stream(trait_dir, old_residual, mapping, dry_run)

            # Migrate layer_internal_states -> layer_internals/{prompt_set}/
            old_internals = projections_dir / "layer_internal_states"
            if old_internals.exists():
                migrate_layer_internals(trait_dir, old_internals, mapping, dry_run)


def migrate_residual_stream(trait_dir: Path, old_dir: Path, mapping: dict, dry_run: bool):
    """Migrate residual stream files."""
    new_base = trait_dir / "residual_stream"

    for old_file in old_dir.glob("prompt_*.json"):
        # Extract old index from filename (e.g., prompt_0.json -> 0)
        match = re.match(r'prompt_(\d+)\.json', old_file.name)
        if not match:
            print(f"  Skipping unrecognized file: {old_file.name}")
            continue

        old_index = int(match.group(1))

        if old_index not in mapping:
            print(f"  Warning: No mapping for index {old_index}, skipping {old_file.name}")
            continue

        prompt_set, new_id = mapping[old_index]
        new_dir = new_base / prompt_set
        new_file = new_dir / f"{new_id}.json"

        if dry_run:
            print(f"  [DRY RUN] {old_file.name} -> {prompt_set}/{new_id}.json")
        else:
            new_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_file), str(new_file))
            print(f"  Moved: {old_file.name} -> {prompt_set}/{new_id}.json")


def migrate_layer_internals(trait_dir: Path, old_dir: Path, mapping: dict, dry_run: bool):
    """Migrate layer internals files."""
    new_base = trait_dir / "layer_internals"

    for old_file in old_dir.glob("prompt_*_layer*.json"):
        # Extract old index and layer (e.g., prompt_0_layer16.json -> 0, 16)
        match = re.match(r'prompt_(\d+)_layer(\d+)\.json', old_file.name)
        if not match:
            # Try alternate pattern: prompt_N_LXX.json
            match = re.match(r'prompt_(\d+)_L(\d+)\.json', old_file.name)

        if not match:
            print(f"  Skipping unrecognized file: {old_file.name}")
            continue

        old_index = int(match.group(1))
        layer = int(match.group(2))

        if old_index not in mapping:
            print(f"  Warning: No mapping for index {old_index}, skipping {old_file.name}")
            continue

        prompt_set, new_id = mapping[old_index]
        new_dir = new_base / prompt_set
        new_file = new_dir / f"{new_id}_L{layer}.json"

        if dry_run:
            print(f"  [DRY RUN] {old_file.name} -> {prompt_set}/{new_id}_L{layer}.json")
        else:
            new_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_file), str(new_file))
            print(f"  Moved: {old_file.name} -> {prompt_set}/{new_id}_L{layer}.json")


def migrate_raw_activations(experiment_dir: Path, mapping: dict, dry_run: bool):
    """Migrate raw activation files."""
    raw_dir = experiment_dir / "inference" / "raw"
    if not raw_dir.exists():
        return

    # Migrate raw/residual/main_prompts/ -> raw/residual/{prompt_set}/
    old_residual = raw_dir / "residual" / "main_prompts"
    if old_residual.exists():
        print(f"\nMigrating raw residual activations...")
        new_residual_base = raw_dir / "residual"

        for old_file in old_residual.glob("prompt_*.pt"):
            match = re.match(r'prompt_(\d+)\.pt', old_file.name)
            if not match:
                continue

            old_index = int(match.group(1))
            if old_index not in mapping:
                print(f"  Warning: No mapping for index {old_index}")
                continue

            prompt_set, new_id = mapping[old_index]
            new_dir = new_residual_base / prompt_set
            new_file = new_dir / f"{new_id}.pt"

            if dry_run:
                print(f"  [DRY RUN] {old_file.name} -> {prompt_set}/{new_id}.pt")
            else:
                new_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_file), str(new_file))
                print(f"  Moved: {old_file.name} -> {prompt_set}/{new_id}.pt")


def cleanup_old_structure(experiment_dir: Path, dry_run: bool):
    """Remove old directory structure after successful migration."""
    inference_dir = experiment_dir / "inference"

    dirs_to_remove = []

    # Find all old projections directories
    for category_dir in inference_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name in {'raw', 'prompts', 'raw_activations'}:
            continue
        for trait_dir in category_dir.iterdir():
            if not trait_dir.is_dir():
                continue
            projections_dir = trait_dir / "projections"
            if projections_dir.exists():
                dirs_to_remove.append(projections_dir)

    # Old raw/residual/main_prompts
    old_main_prompts = inference_dir / "raw" / "residual" / "main_prompts"
    if old_main_prompts.exists():
        dirs_to_remove.append(old_main_prompts)

    if dry_run:
        print("\n[DRY RUN] Would remove old directories:")
        for d in dirs_to_remove:
            print(f"  {d}")
    else:
        print("\nRemoving old directories...")
        for d in dirs_to_remove:
            shutil.rmtree(d)
            print(f"  Removed: {d}")


def main():
    parser = argparse.ArgumentParser(description="Migrate inference data to new structure")
    parser.add_argument("--experiment", required=True, help="Experiment name or path")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--cleanup", action="store_true", help="Remove old directories after migration")
    args = parser.parse_args()

    # Find experiment directory
    if os.path.isabs(args.experiment):
        experiment_dir = Path(args.experiment)
    else:
        experiment_dir = Path("experiments") / args.experiment

    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return 1

    # Find main_prompts.txt
    prompts_file = Path("inference/prompts/main_prompts.txt")
    if not prompts_file.exists():
        print(f"Error: main_prompts.txt not found at {prompts_file}")
        return 1

    print(f"Experiment: {experiment_dir}")
    print(f"Prompts file: {prompts_file}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Build mapping
    print("Building index mapping from main_prompts.txt...")
    mapping = parse_main_prompts(prompts_file)
    print(f"Found {len(mapping)} prompts")

    # Show mapping summary
    sections = {}
    for old_idx, (prompt_set, new_id) in mapping.items():
        if prompt_set not in sections:
            sections[prompt_set] = []
        sections[prompt_set].append((old_idx, new_id))

    print("\nMapping summary:")
    for section, items in sorted(sections.items()):
        old_indices = [str(old) for old, new in items]
        print(f"  {section}: old indices {old_indices[0]}-{old_indices[-1]} -> new IDs 1-{len(items)}")

    # Migrate
    migrate_experiment(experiment_dir, mapping, args.dry_run)
    migrate_raw_activations(experiment_dir, mapping, args.dry_run)

    if args.cleanup and not args.dry_run:
        cleanup_old_structure(experiment_dir, args.dry_run)
    elif args.cleanup and args.dry_run:
        cleanup_old_structure(experiment_dir, args.dry_run)

    print("\nDone!" if not args.dry_run else "\n[DRY RUN] Complete - no changes made")
    return 0


if __name__ == "__main__":
    exit(main())
