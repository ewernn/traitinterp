#!/usr/bin/env python3
"""
Migrate steering results.json files to JSONL format.

Input: experiments/*/steering/**/results.json
Output: experiments/*/steering/**/results.jsonl (with backup of original)

Usage:
    # Dry run (show what would be migrated)
    python scripts/migrate_results_to_jsonl.py --dry-run

    # Migrate all
    python scripts/migrate_results_to_jsonl.py

    # Migrate specific experiment
    python scripts/migrate_results_to_jsonl.py --experiment gemma-2-2b
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from typing import List


def find_results_files(experiments_dir: Path, experiment: str = None) -> List[Path]:
    """Find all results.json files in steering directories."""
    if experiment:
        pattern = f"{experiment}/steering/**/results.json"
    else:
        pattern = "*/steering/**/results.json"

    return list(experiments_dir.glob(pattern))


def migrate_file(json_path: Path, dry_run: bool = False, keep_backup: bool = True) -> bool:
    """
    Convert results.json to results.jsonl.

    Returns True if migration successful, False otherwise.
    """
    jsonl_path = json_path.with_suffix('.jsonl')

    # Skip if already migrated
    if jsonl_path.exists():
        print(f"  SKIP (already exists): {jsonl_path.relative_to(json_path.parent.parent.parent.parent)}")
        return False

    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  ERROR reading {json_path}: {e}")
        return False

    # Validate structure
    if "runs" not in data:
        print(f"  SKIP (no runs): {json_path}")
        return False

    if dry_run:
        n_runs = len(data.get("runs", []))
        has_baseline = data.get("baseline") is not None
        print(f"  WOULD migrate: {json_path.relative_to(json_path.parent.parent.parent.parent)}")
        print(f"    → {n_runs} runs, baseline={'yes' if has_baseline else 'no'}")
        return True

    # Build JSONL content
    lines = []

    # Header line (metadata)
    header = {
        "type": "header",
        "trait": data.get("trait"),
        "steering_model": data.get("steering_model"),
        "steering_experiment": data.get("steering_experiment"),
        "vector_source": data.get("vector_source"),
        "eval": data.get("eval"),
        "prompts_file": data.get("prompts_file"),
    }
    lines.append(json.dumps(header))

    # Baseline line
    if data.get("baseline"):
        baseline_entry = {
            "type": "baseline",
            "result": data["baseline"],
            "timestamp": data.get("baseline_timestamp"),  # May not exist in old format
        }
        lines.append(json.dumps(baseline_entry))

    # Run lines
    for run in data.get("runs", []):
        # Reorder: result first, then config, then timestamp
        entry = {
            "result": run.get("result"),
            "config": run.get("config"),
            "timestamp": run.get("timestamp"),
        }
        lines.append(json.dumps(entry))

    # Write JSONL
    try:
        with open(jsonl_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
    except IOError as e:
        print(f"  ERROR writing {jsonl_path}: {e}")
        return False

    # Backup original
    if keep_backup:
        backup_path = json_path.with_suffix('.json.bak')
        json_path.rename(backup_path)
        print(f"  MIGRATED: {json_path.name} → {jsonl_path.name} (backup: {backup_path.name})")
    else:
        json_path.unlink()
        print(f"  MIGRATED: {json_path.name} → {jsonl_path.name} (original deleted)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate results.json to JSONL")
    parser.add_argument("--experiment", help="Migrate only this experiment")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")
    parser.add_argument("--no-backup", action="store_true", help="Delete original instead of keeping .bak")
    args = parser.parse_args()

    experiments_dir = Path(__file__).parent.parent / "experiments"
    if not experiments_dir.exists():
        print(f"Experiments directory not found: {experiments_dir}")
        return 1

    results_files = find_results_files(experiments_dir, args.experiment)

    if not results_files:
        print("No results.json files found")
        return 0

    print(f"Found {len(results_files)} results.json file(s)")
    if args.dry_run:
        print("DRY RUN - no changes will be made\n")

    migrated = 0
    for path in sorted(results_files):
        if migrate_file(path, dry_run=args.dry_run, keep_backup=not args.no_backup):
            migrated += 1

    print(f"\n{'Would migrate' if args.dry_run else 'Migrated'}: {migrated}/{len(results_files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
