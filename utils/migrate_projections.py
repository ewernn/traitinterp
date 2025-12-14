#!/usr/bin/env python3
"""
Migrate projection files from old 3D format to new 1D format.

Old format: projections.prompt = [[L0_vals, L1_vals, ...], [L0_vals, ...], ...]  # All 26 layers
New format: projections.prompt = [val1, val2, ...]  # Just best layer

Usage:
    python utils/migrate_projections.py --experiment gemma-2-2b
    python utils/migrate_projections.py --experiment gemma-2-2b --dry-run
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from tqdm import tqdm

from utils.paths import get as get_path


def migrate_file(file_path: Path, dry_run: bool = False) -> dict:
    """Migrate a single projection file from old to new format."""
    with open(file_path) as f:
        data = json.load(f)

    # Check if already migrated (1D arrays)
    prompt_proj = data.get('projections', {}).get('prompt', [])
    if not prompt_proj:
        return {'skipped': 'no_projections'}

    # If first element is a number, already migrated
    if isinstance(prompt_proj[0], (int, float)):
        return {'skipped': 'already_migrated'}

    # Get which layer was used (from metadata)
    metadata = data.get('metadata', {})
    vector_source = metadata.get('vector_source', {})
    layer = vector_source.get('layer')
    component = vector_source.get('component', 'residual')

    if layer is None:
        return {'error': 'no_layer_in_metadata'}

    # Extract just that layer
    # Old format: [n_tokens, 26_layers, 2_sublayers] for residual
    # Old format: [n_tokens, 26_layers, 1] for attn_out
    prompt_old = data['projections']['prompt']
    response_old = data['projections']['response']

    try:
        if component == 'attn_out':
            # Sublayer index 0 for attn_out
            prompt_new = [token[layer][0] for token in prompt_old]
            response_new = [token[layer][0] for token in response_old]
            sublayer = 'attn_out'
        else:
            # Sublayer index 1 = residual_out (after MLP)
            prompt_new = [token[layer][1] for token in prompt_old]
            response_new = [token[layer][1] for token in response_old]
            sublayer = 'residual_out'
    except (IndexError, TypeError) as e:
        return {'error': f'extraction_failed: {e}'}

    # Restructure with metadata first, no dynamics
    new_data = {
        'metadata': metadata,
        'projections': {
            'prompt': prompt_new,
            'response': response_new
        }
    }

    # Add activation_norms if present
    if 'activation_norms' in data:
        new_data['activation_norms'] = data['activation_norms']

    # Add logit_lens if present
    if 'logit_lens' in data:
        new_data['logit_lens'] = data['logit_lens']

    # Add sublayer to metadata
    new_data['metadata']['vector_source']['sublayer'] = sublayer

    # Calculate size reduction
    old_size = file_path.stat().st_size

    if not dry_run:
        # Overwrite with new format
        with open(file_path, 'w') as f:
            json.dump(new_data, f, indent=2)

        new_size = file_path.stat().st_size
        reduction = (old_size - new_size) / old_size * 100
    else:
        # Estimate new size
        import io
        buf = io.StringIO()
        json.dump(new_data, buf, indent=2)
        new_size = len(buf.getvalue())
        reduction = (old_size - new_size) / old_size * 100

    return {
        'migrated': True,
        'old_size': old_size,
        'new_size': new_size,
        'reduction_pct': reduction
    }


def main():
    parser = argparse.ArgumentParser(description="Migrate projection files to new format")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files, just report")
    args = parser.parse_args()

    inference_dir = get_path('inference.base', experiment=args.experiment)

    # Find all projection files (residual_stream and attn_stream)
    projection_files = []
    projection_files.extend(inference_dir.glob("**/residual_stream/*/*.json"))
    projection_files.extend(inference_dir.glob("**/attn_stream/*/*.json"))

    if not projection_files:
        print(f"No projection files found in {inference_dir}")
        return

    print(f"Found {len(projection_files)} projection files")
    if args.dry_run:
        print("DRY RUN - no files will be modified\n")

    stats = {
        'migrated': 0,
        'skipped': 0,
        'errors': 0,
        'total_old_size': 0,
        'total_new_size': 0
    }

    for file_path in tqdm(projection_files, desc="Migrating"):
        result = migrate_file(file_path, dry_run=args.dry_run)

        if result.get('migrated'):
            stats['migrated'] += 1
            stats['total_old_size'] += result['old_size']
            stats['total_new_size'] += result['new_size']
        elif result.get('skipped'):
            stats['skipped'] += 1
        elif result.get('error'):
            stats['errors'] += 1
            print(f"\nError in {file_path}: {result['error']}")

    # Report
    print("\n" + "="*60)
    print("Migration Summary")
    print("="*60)
    print(f"Migrated:     {stats['migrated']} files")
    print(f"Skipped:      {stats['skipped']} files (already migrated)")
    print(f"Errors:       {stats['errors']} files")

    if stats['migrated'] > 0:
        total_reduction = (stats['total_old_size'] - stats['total_new_size']) / stats['total_old_size'] * 100
        print(f"\nTotal size:   {stats['total_old_size'] / 1024 / 1024:.1f} MB → {stats['total_new_size'] / 1024 / 1024:.1f} MB")
        print(f"Reduction:    {total_reduction:.1f}%")
        print(f"Space saved:  {(stats['total_old_size'] - stats['total_new_size']) / 1024 / 1024:.1f} MB")

    if args.dry_run:
        print("\nDRY RUN - Run without --dry-run to apply changes")


if __name__ == "__main__":
    main()
