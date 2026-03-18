#!/usr/bin/env python3
"""
Run steering eval on Modal — syncs data, calls remote, downloads results.

Input:
    - experiment name, trait list, layer list
    - Local experiment data (config, vectors, steering datasets)

Output:
    - Steering results downloaded to experiments/{experiment}/steering/...
    - JSON summary printed to stdout

Usage:
    python dev/steering/modal_evaluate.py \
        --experiment dataset_creation_emotions \
        --traits emotions/anger emotions/fear \
        --layers 15,20,25,30

    # Single trait with all options
    python dev/steering/modal_evaluate.py \
        --experiment dataset_creation_emotions \
        --traits emotions/anger \
        --layers 15,20,25 \
        --method probe \
        --direction positive \
        --search-steps 5 \
        --force
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.paths import (
    load_experiment_config,
    get_model_variant,
    get_default_variant,
    get_steering_dir,
    sanitize_position,
)


def sync_to_volumes(experiment, traits, extraction_variant, skip_vectors=False):
    """Upload vectors + steering data to Modal volumes."""
    from dev.inference.modal_steering import experiments_volume, datasets_volume

    repo_root = Path(__file__).parent.parent.parent
    exp_dir = repo_root / "experiments" / experiment

    with experiments_volume.batch_upload(force=True) as batch:
        # Experiment config
        config_path = exp_dir / "config.json"
        if config_path.exists():
            batch.put_file(str(config_path), f"{experiment}/config.json")

        # Activation norms
        eval_path = exp_dir / "extraction" / "extraction_evaluation.json"
        if eval_path.exists():
            batch.put_file(str(eval_path), f"{experiment}/extraction/extraction_evaluation.json")

        # Vectors for each trait (skip for baseline-only mode)
        if not skip_vectors:
            for trait in traits:
                vec_dir = exp_dir / "extraction" / trait / extraction_variant / "vectors"
                if vec_dir.exists():
                    batch.put_directory(
                        str(vec_dir),
                        f"{experiment}/extraction/{trait}/{extraction_variant}/vectors",
                    )

    with datasets_volume.batch_upload(force=True) as batch:
        for trait in traits:
            ds_dir = repo_root / "datasets" / "traits" / trait
            if ds_dir.exists():
                batch.put_directory(str(ds_dir), f"traits/{trait}")

    print(f"Synced {len(traits)} traits to Modal volumes{' (baseline-only, no vectors)' if skip_vectors else ''}")


def pull_results(experiment, traits, model_variant, position):
    """Download steering results from Modal volume to local filesystem."""
    from dev.inference.modal_steering import experiments_volume

    repo_root = Path(__file__).parent.parent.parent
    pos_dir = sanitize_position(position)
    pulled = 0

    for trait in traits:
        # Remote path on volume (relative to mount point)
        remote_prefix = f"{experiment}/steering/{trait}/{model_variant}/{pos_dir}/steering"
        # Local destination
        local_dir = get_steering_dir(experiment, trait, model_variant, position, "steering")
        local_dir.mkdir(parents=True, exist_ok=True)

        # Download results.jsonl
        try:
            data = b"".join(experiments_volume.read_file(f"{remote_prefix}/results.jsonl"))
            (local_dir / "results.jsonl").write_bytes(data)
            pulled += 1
        except FileNotFoundError:
            print(f"  No results for {trait}")
            continue

        # Download response files (best responses)
        from utils.modal_sync import pull_dir_recursive
        pull_dir_recursive(
            experiments_volume,
            f"{remote_prefix}/responses",
            local_dir / "responses",
        )

    print(f"Pulled results for {pulled}/{len(traits)} traits")


def main():
    parser = argparse.ArgumentParser(description="Run steering eval on Modal")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--traits", nargs="+", required=True,
                        help="Trait paths (e.g. emotions/anger emotions/fear)")
    parser.add_argument("--layers", required=True,
                        help="Comma-separated layers (e.g. 15,20,25)")
    parser.add_argument("--model-variant", default=None,
                        help="Model variant (default: from config)")
    parser.add_argument("--extraction-variant", default=None,
                        help="Extraction variant (default: from config)")
    parser.add_argument("--method", default="probe")
    parser.add_argument("--component", default="residual")
    parser.add_argument("--position", default="response[:5]")
    parser.add_argument("--direction", default="positive")
    parser.add_argument("--search-steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--save-responses", default="best",
                        choices=["all", "best", "none"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only compute baseline (unsteered) scores, no steering")
    parser.add_argument("--trait-layers", nargs="+", metavar="TRAIT:LAYERS",
                        help='Per-trait layer overrides: "cat/trait:10,12,14"')
    parser.add_argument("--no-pull", action="store_true",
                        help="Skip downloading results after eval")
    args = parser.parse_args()

    # Resolve experiment config
    config = load_experiment_config(args.experiment)

    model_variant = args.model_variant or config.get("defaults", {}).get("application")
    extraction_variant = args.extraction_variant or config.get("defaults", {}).get("extraction")

    if not model_variant:
        print("Error: no model_variant specified and no default in config")
        sys.exit(1)
    if not extraction_variant:
        print("Error: no extraction_variant specified and no default in config")
        sys.exit(1)

    variant_config = get_model_variant(args.experiment, model_variant)
    model_name = variant_config["model"]

    layer_list = [int(l) for l in args.layers.split(",")]

    # Parse per-trait layer overrides
    trait_layers = None
    if args.trait_layers:
        trait_layers = {}
        for spec in args.trait_layers:
            if ":" not in spec:
                print(f"Error: invalid --trait-layers spec '{spec}': use 'category/trait:layers'")
                sys.exit(1)
            trait_part, layer_spec = spec.rsplit(":", 1)
            trait_layers[trait_part] = layer_spec

    print(f"Experiment: {args.experiment}")
    print(f"Model: {model_name} ({model_variant})")
    print(f"Extraction: {extraction_variant}")
    print(f"Traits: {args.traits}")
    print(f"Layers: {layer_list}" + (f" + {len(trait_layers)} per-trait overrides" if trait_layers else ""))
    print()

    import time
    total_start = time.time()

    # 1. Sync data to volumes (skip vectors for baseline-only)
    t0 = time.time()
    sync_to_volumes(args.experiment, args.traits, extraction_variant,
                    skip_vectors=args.baseline_only)
    sync_time = time.time() - t0
    print(f"  Sync: {sync_time:.1f}s")

    # 2. Call Modal (lookup deployed function by name)
    import modal
    steering_eval_remote = modal.Function.from_name("trait-steering", "steering_eval_remote")

    t0 = time.time()
    result = steering_eval_remote.remote(
        model_name=model_name,
        experiment=args.experiment,
        traits=args.traits,
        model_variant=model_variant,
        extraction_variant=extraction_variant,
        layers=layer_list,
        method=args.method,
        component=args.component,
        position=args.position,
        direction=args.direction,
        search_steps=args.search_steps,
        max_new_tokens=args.max_new_tokens,
        subset=args.subset,
        force=args.force,
        save_responses=args.save_responses,
        baseline_only=args.baseline_only,
        trait_layers=trait_layers,
    )
    remote_time = time.time() - t0
    model_load = result.get("model_load_time", 0)
    eval_time = remote_time - model_load if model_load else remote_time
    print(f"  Remote: {remote_time:.1f}s (model load: {model_load:.1f}s, eval: {eval_time:.1f}s)")

    # 3. Pull results to local filesystem
    t0 = time.time()
    if not args.no_pull:
        pull_results(args.experiment, args.traits, model_variant, args.position)
    pull_time = time.time() - t0

    total_time = time.time() - total_start
    print(f"  Pull: {pull_time:.1f}s")
    print(f"  Total: {total_time:.1f}s")

    # 4. Print summary
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
