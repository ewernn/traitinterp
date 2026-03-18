#!/usr/bin/env python3
"""
Run extraction pipeline on Modal — syncs data, calls remote, downloads results.

Input:
    - experiment name, trait list
    - Local scenario files (positive.txt, negative.txt, definition.txt)

Output:
    - Vectors downloaded to experiments/{experiment}/extraction/{trait}/{variant}/vectors/
    - Vetting scores downloaded (diagnostic)
    - Activations NOT downloaded (huge, not needed locally)

Usage:
    python extraction/modal_extract.py \
        --experiment dataset_creation_emotions \
        --traits emotions/anger emotions/fear

    # Single trait with all options
    python extraction/modal_extract.py \
        --experiment dataset_creation_emotions \
        --traits emotions/anger \
        --methods probe,mean_diff \
        --position response[:5] \
        --force
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import (
    load_experiment_config,
    get_model_variant,
    get as get_path,
    sanitize_position,
)


def sync_to_volumes(experiment, traits):
    """Upload scenario files + config to Modal volumes."""
    from dev.extraction.modal_pipeline import experiments_volume, datasets_volume

    repo_root = Path(__file__).parent.parent
    exp_dir = repo_root / "experiments" / experiment

    with experiments_volume.batch_upload(force=True) as batch:
        config_path = exp_dir / "config.json"
        if config_path.exists():
            batch.put_file(str(config_path), f"{experiment}/config.json")

    with datasets_volume.batch_upload(force=True) as batch:
        for trait in traits:
            ds_dir = repo_root / "datasets" / "traits" / trait
            if ds_dir.exists():
                batch.put_directory(str(ds_dir), f"traits/{trait}")
            else:
                print(f"  WARNING: no dataset dir for {trait}: {ds_dir}")

    print(f"Synced {len(traits)} traits to volumes")


def pull_results(experiment, traits, model_variant, position, component):
    """Download vectors + vetting scores from Modal volume.

    Skips activations (huge intermediate files not needed locally).
    """
    from dev.extraction.modal_pipeline import experiments_volume
    from utils.modal_sync import pull_dir_recursive, pull_file

    repo_root = Path(__file__).parent.parent
    pos_dir = sanitize_position(position)
    pulled = 0

    for trait in traits:
        variant_prefix = f"{experiment}/extraction/{trait}/{model_variant}"
        local_base = repo_root / "experiments" / experiment / "extraction" / trait / model_variant

        # Pull vectors (all positions/components/methods)
        remote_vectors = f"{variant_prefix}/vectors"
        local_vectors = local_base / "vectors"
        pull_dir_recursive(experiments_volume, remote_vectors, local_vectors)

        # Pull vetting scores (diagnostic)
        remote_vetting = f"{variant_prefix}/vetting/response_scores.json"
        local_vetting = local_base / "vetting" / "response_scores.json"
        pull_file(experiments_volume, remote_vetting, local_vetting)

        # Check if we got vectors
        if local_vectors.exists() and any(local_vectors.rglob("layer*.pt")):
            pulled += 1
        else:
            print(f"  No vectors found for {trait}")

    print(f"Pulled results for {pulled}/{len(traits)} traits")


def main():
    parser = argparse.ArgumentParser(description="Run extraction pipeline on Modal")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--traits", nargs="+", required=True,
                        help="Trait paths (e.g. emotions/anger emotions/fear)")
    parser.add_argument("--model-variant", default=None,
                        help="Model variant (default: from config defaults.extraction)")
    parser.add_argument("--position", default="response[:5]")
    parser.add_argument("--component", default="residual")
    parser.add_argument("--methods", default="probe",
                        help="Comma-separated methods (default: probe)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-vet", action="store_true",
                        help="Skip response vetting")
    parser.add_argument("--no-pull", action="store_true",
                        help="Skip downloading results after extraction")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layers (e.g. 15,20,25,30)")
    args = parser.parse_args()

    # Resolve experiment config
    config = load_experiment_config(args.experiment)

    model_variant = args.model_variant or config.get("defaults", {}).get("extraction")
    if not model_variant:
        print("Error: no model_variant specified and no default in config")
        sys.exit(1)

    variant_config = get_model_variant(args.experiment, model_variant, mode="extraction")
    model_name = variant_config["model"]

    method_list = [m.strip() for m in args.methods.split(",")]
    layer_list = [int(l) for l in args.layers.split(",")] if args.layers else None

    print(f"Experiment: {args.experiment}")
    print(f"Model: {model_name} ({model_variant})")
    print(f"Traits: {args.traits}")
    print(f"Methods: {method_list}")
    print(f"Position: {args.position}")
    if layer_list:
        print(f"Layers: {layer_list}")
    print()

    import time
    total_start = time.time()

    # 1. Sync data to volumes
    t0 = time.time()
    sync_to_volumes(args.experiment, args.traits)
    sync_time = time.time() - t0
    print(f"  Sync: {sync_time:.1f}s")

    # 2. Call Modal (lookup deployed function by name)
    import modal
    extraction_pipeline_remote = modal.Function.from_name(
        "trait-extraction", "extraction_pipeline_remote"
    )

    t0 = time.time()
    result = extraction_pipeline_remote.remote(
        model_name=model_name,
        experiment=args.experiment,
        traits=args.traits,
        model_variant=model_variant,
        position=args.position,
        component=args.component,
        methods=method_list,
        force=args.force,
        vet=not args.no_vet,
        layers=layer_list,
    )
    remote_time = time.time() - t0
    model_load = result.get("model_load_time", 0)
    pipeline_time = remote_time - model_load if model_load else remote_time
    print(f"  Remote: {remote_time:.1f}s (model load: {model_load:.1f}s, pipeline: {pipeline_time:.1f}s)")

    # 3. Pull results to local filesystem
    t0 = time.time()
    if not args.no_pull:
        pull_results(args.experiment, args.traits, model_variant,
                     args.position, args.component)
    pull_time = time.time() - t0

    total_time = time.time() - total_start
    print(f"  Pull: {pull_time:.1f}s")
    print(f"  Total: {total_time:.1f}s")

    # 4. Print summary
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
