#!/usr/bin/env python3
"""
Migrate steering results.json from old config format to VectorSpec format.

Old format:
    {"config": {"layers": [14], "methods": ["probe"], "coefficients": [207.9], "component": "residual"}}

New format:
    {"config": {"vectors": [{"layer": 14, "component": "residual", "position": "response[:5]", "method": "probe", "weight": 207.9}]}}

Usage:
    python scripts/migrate_steering_results.py                    # Dry run
    python scripts/migrate_steering_results.py --apply            # Apply changes
    python scripts/migrate_steering_results.py --apply --backup   # Apply with .bak files
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import shutil
import argparse

from utils.paths import desanitize_position


def is_old_format(config: dict) -> bool:
    """Check if config uses old format (has 'layers' key, not 'vectors')."""
    return 'layers' in config and 'vectors' not in config


def convert_config(old_config: dict, position: str) -> dict:
    """Convert old config format to new VectorSpec format."""
    # Handle multi-layer configs (rare but possible)
    vectors = []
    layers = old_config.get('layers', [])
    methods = old_config.get('methods', ['probe'])
    coefficients = old_config.get('coefficients', [])
    component = old_config.get('component', 'residual')

    for i, layer in enumerate(layers):
        method = methods[i] if i < len(methods) else methods[0]
        coef = coefficients[i] if i < len(coefficients) else coefficients[0] if coefficients else 1.0

        vectors.append({
            'layer': layer,
            'component': component,
            'position': position,
            'method': method,
            'weight': coef
        })

    return {'vectors': vectors}


def migrate_file(results_path: Path, dry_run: bool = True, backup: bool = False) -> dict:
    """
    Migrate a single results.json file.

    Returns:
        {'status': 'migrated'|'skipped'|'already_new', 'runs_converted': int}
    """
    # Extract position from path: .../steering/{trait}/{position}/results.json
    position_sanitized = results_path.parent.name
    position = desanitize_position(position_sanitized)

    with open(results_path) as f:
        data = json.load(f)

    runs_converted = 0
    for run in data.get('runs', []):
        config = run.get('config', {})
        if is_old_format(config):
            new_config = convert_config(config, position)
            run['config'] = new_config
            runs_converted += 1

    if runs_converted == 0:
        return {'status': 'already_new', 'runs_converted': 0}

    if dry_run:
        return {'status': 'would_migrate', 'runs_converted': runs_converted}

    # Apply changes
    if backup:
        shutil.copy(results_path, results_path.with_suffix('.json.bak'))

    with open(results_path, 'w') as f:
        json.dump(data, f, indent=2)

    return {'status': 'migrated', 'runs_converted': runs_converted}


def main():
    parser = argparse.ArgumentParser(description="Migrate steering results to VectorSpec format")
    parser.add_argument('--apply', action='store_true', help="Apply changes (default: dry run)")
    parser.add_argument('--backup', action='store_true', help="Create .bak files before modifying")
    parser.add_argument('--experiment', help="Only migrate specific experiment")
    args = parser.parse_args()

    experiments_dir = Path('experiments')

    if args.experiment:
        pattern = f"{args.experiment}/steering/**/results.json"
    else:
        pattern = "*/steering/**/results.json"

    results_files = list(experiments_dir.glob(pattern))

    if not results_files:
        print("No steering results.json files found")
        return

    print(f"Found {len(results_files)} steering results files")
    print(f"Mode: {'DRY RUN' if not args.apply else 'APPLYING CHANGES'}")
    print()

    stats = {'migrated': 0, 'already_new': 0, 'would_migrate': 0, 'total_runs': 0}

    for results_path in sorted(results_files):
        rel_path = results_path.relative_to(experiments_dir)
        result = migrate_file(results_path, dry_run=not args.apply, backup=args.backup)

        status = result['status']
        runs = result['runs_converted']
        stats[status] = stats.get(status, 0) + 1
        stats['total_runs'] += runs

        if status == 'already_new':
            print(f"  [skip] {rel_path} (already new format)")
        elif status == 'would_migrate':
            print(f"  [would migrate] {rel_path} ({runs} runs)")
        else:
            print(f"  [migrated] {rel_path} ({runs} runs)")

    print()
    print("Summary:")
    if args.apply:
        print(f"  Migrated: {stats.get('migrated', 0)} files ({stats['total_runs']} runs)")
    else:
        print(f"  Would migrate: {stats.get('would_migrate', 0)} files ({stats['total_runs']} runs)")
    print(f"  Already new format: {stats['already_new']} files")

    if not args.apply and stats.get('would_migrate', 0) > 0:
        print()
        print("Run with --apply to make changes")


if __name__ == '__main__':
    main()
