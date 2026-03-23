"""
Modal deployment for extraction pipeline.

Runs extraction stages 1-4 (generate → vet → activations → vectors) on Modal GPUs.
Calls run_pipeline() directly — zero duplicate extraction logic.

Input:
    - experiment, traits, model config (synced to volumes by caller)
    - Scenario files on trait-datasets volume

Output:
    - Vectors written to trait-experiments volume
    - Vetting scores written to trait-experiments volume
    - Summary dict returned to caller

Usage:
    # Deploy
    modal deploy extraction/modal_pipeline.py

    # Test via local entrypoint
    modal run extraction/modal_pipeline.py \
        --experiment dataset_creation_emotions \
        --traits emotions/anger
"""

import modal
import os
from pathlib import Path

app = modal.App("trait-extraction")

# Reuse same volumes as steering
model_volume = modal.Volume.from_name("model-cache", create_if_missing=True)
experiments_volume = modal.Volume.from_name("trait-experiments", create_if_missing=True)
datasets_volume = modal.Volume.from_name("trait-datasets", create_if_missing=True)

repo_root = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "scipy",
        "scikit-learn",
        "pyyaml",
        "openai",
        "python-dotenv",
        "tqdm",
        "fire",
    )
    .add_local_dir(repo_root / "core", remote_path="/root/core")
    .add_local_dir(repo_root / "utils", remote_path="/root/utils")
    .add_local_dir(repo_root / "dev", remote_path="/root/dev")
    .add_local_dir(repo_root / "config", remote_path="/root/config")
    .add_local_dir(repo_root / "extraction", remote_path="/root/extraction")
    .add_local_dir(repo_root / "analysis", remote_path="/root/analysis")
)


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={
        "/models": model_volume,
        "/root/experiments": experiments_volume,
        "/root/datasets": datasets_volume,
    },
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("openai")],
    timeout=3600,
)
def extraction_pipeline_remote(
    model_name: str,
    experiment: str,
    traits: list[str],
    model_variant: str,
    position: str = "response[:5]",
    component: str = "residual",
    methods: list[str] = None,
    force: bool = False,
    vet: bool = True,
    layers: list[int] = None,
) -> dict:
    """Run extraction pipeline stages 1-4 on Modal."""
    import sys
    import time

    sys.path.insert(0, "/root")
    os.chdir("/root")

    from dev.modal_sync import load_model_cached
    from utils.backends import LocalBackend
    from extraction.run_extraction_pipeline import run_pipeline

    # Reload volumes to see latest uploads from caller
    experiments_volume.reload()
    datasets_volume.reload()

    methods = methods or ["probe"]
    start = time.time()

    # Load model from volume cache
    model, tokenizer, load_time = load_model_cached(model_name, volume=model_volume)
    backend = LocalBackend.from_model(model, tokenizer)

    # Run stages 1-4: generate → vet → activations → vectors
    run_pipeline(
        experiment=experiment,
        model_variant=model_variant,
        traits=traits,
        only_stages={1, 2, 3, 4},
        backend=backend,
        methods=methods,
        force=force,
        vet=vet,
        component=component,
        position=position,
        layers=layers,
    )

    # Persist results to volume
    experiments_volume.commit()

    total_time = time.time() - start
    return {
        "model_load_time": load_time,
        "total_time": total_time,
        "traits": traits,
        "methods": methods,
    }


@app.local_entrypoint()
def main(
    experiment: str = "dataset_creation_emotions",
    traits: str = "emotions/anger",
    position: str = "response[:5]",
    component: str = "residual",
    methods: str = "probe",
    model_variant: str = None,
    force: bool = False,
    no_vet: bool = False,
):
    """
    Test extraction pipeline on Modal.

    Usage:
        modal run extraction/modal_pipeline.py \
            --experiment dataset_creation_emotions \
            --traits emotions/anger
    """
    import json
    import sys

    sys.path.insert(0, str(repo_root))

    from utils.paths import load_experiment_config, get_model_variant

    config = load_experiment_config(experiment)

    if model_variant is None:
        model_variant = config.get("defaults", {}).get("extraction")

    variant_config = get_model_variant(experiment, model_variant, mode="extraction")
    model_name = variant_config.model

    trait_list = [t.strip() for t in traits.split(",")]
    method_list = [m.strip() for m in methods.split(",")]

    print(f"Experiment: {experiment}")
    print(f"Model: {model_name} ({model_variant})")
    print(f"Traits: {trait_list}")
    print(f"Methods: {method_list}")
    print(f"Position: {position}")
    print()

    # Sync scenario files + config to volumes
    _sync_to_volumes(experiment, trait_list)

    result = extraction_pipeline_remote.remote(
        model_name=model_name,
        experiment=experiment,
        traits=trait_list,
        model_variant=model_variant,
        position=position,
        component=component,
        methods=method_list,
        force=force,
        vet=not no_vet,
    )

    print(json.dumps(result, indent=2, default=str))


def _sync_to_volumes(experiment, traits):
    """Upload scenario files + experiment config to Modal volumes."""
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

    print(f"Synced {len(traits)} traits to volumes")
