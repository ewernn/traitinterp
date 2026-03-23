"""
Modal deployment for steering evaluation.

Runs steering eval on Modal GPUs. Agent scripts call this for parallel eval
across traits — each agent owns one trait, calls Modal for GPU work.

Input:
    - experiment, traits, model config (synced to volumes by caller)
    - vectors + steering data on trait-experiments/trait-datasets volumes

Output:
    - Steering results written to trait-experiments volume
    - Results dict returned to caller

Usage:
    # Deploy
    modal deploy inference/modal_steering.py

    # Test via local entrypoint (syncs data automatically)
    modal run inference/modal_steering.py \
        --experiment dataset_creation_emotions \
        --traits emotions/anger \
        --layers 15,20,25

    # Programmatic use (from dev/steering/modal_evaluate.py)
    from inference.modal_steering import steering_eval_remote
    result = steering_eval_remote.remote(...)
"""

import modal
import os
from pathlib import Path

app = modal.App("trait-steering")

# Volumes
model_volume = modal.Volume.from_name("model-cache", create_if_missing=True)
experiments_volume = modal.Volume.from_name("trait-experiments", create_if_missing=True)
datasets_volume = modal.Volume.from_name("trait-datasets", create_if_missing=True)

# Paths for copying local code into image
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
    )
    .add_local_dir(repo_root / "core", remote_path="/root/core")
    .add_local_dir(repo_root / "utils", remote_path="/root/utils")
    .add_local_dir(repo_root / "config", remote_path="/root/config")
    .add_local_dir(repo_root / "steering", remote_path="/root/steering")
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
async def steering_eval_remote(
    model_name: str,
    experiment: str,
    traits: list[str],
    model_variant: str,
    extraction_variant: str,
    layers: list[int],
    method: str = "probe",
    component: str = "residual",
    position: str = "response[:5]",
    direction: str = "positive",
    search_steps: int = 5,
    max_new_tokens: int = 64,
    subset: int = 0,
    force: bool = False,
    save_responses: str = "best",
    baseline_only: bool = False,
    trait_layers: dict = None,
) -> dict:
    """Run steering eval on Modal. Returns results dict."""
    import sys
    sys.path.insert(0, "/root")
    os.chdir("/root")

    import argparse
    from utils.backends import LocalBackend
    from utils.judge import TraitJudge
    from utils.modal_sync import load_model_cached
    from steering.run_steering_eval import run as _run_main
    from utils.steering_results import load_results
    from utils.vectors import MIN_COHERENCE

    # See latest uploads from caller
    experiments_volume.reload()
    datasets_volume.reload()

    model, tokenizer, load_time = load_model_cached(model_name, volume=model_volume)
    backend = LocalBackend.from_model(model, tokenizer)
    judge = TraitJudge()

    # Build args namespace — mirrors server pattern (other/server/app.py:237-266)
    eval_args = argparse.Namespace(
        experiment=experiment,
        no_batch=False,
        ablation=None,
        load_in_8bit=False,
        load_in_4bit=False,
        bnb_4bit_quant_type="nf4",
        layers=",".join(str(l) for l in layers),
        coefficients=None,
        search_steps=search_steps,
        up_mult=1.3,
        down_mult=0.85,
        start_mult=0.7,
        momentum=0.1,
        max_new_tokens=max_new_tokens,
        save_responses=save_responses,
        min_coherence=MIN_COHERENCE,
        no_relevance_check=False,
        no_custom_prompt=False,
        eval_prompt_from=None,
        trait_judge=None,
        method=method,
        component=component,
        position=position,
        prompt_set="steering",
        subset=subset,
        judge="openai",
        vector_experiment=None,
        extraction_variant=extraction_variant,
        questions_file=None,
        force=force,
        regenerate_responses=False,
    )

    parsed_traits = [(experiment, t) for t in traits]

    try:
        await _run_main(
            args=eval_args,
            parsed_traits=parsed_traits,
            model_variant=model_variant,
            model_name=model_name,
            lora=None,
            layers_arg=eval_args.layers,
            coefficients=None,
            direction=direction,
            force=force,
            backend=backend,
            judge=judge,
            trait_layers=trait_layers,
            baseline_only=baseline_only,
        )
    finally:
        await judge.close()

    # Load results written by _run_main
    results = {}
    for trait in traits:
        try:
            results[trait] = load_results(experiment, trait, model_variant, position, "steering").to_dict()
        except FileNotFoundError:
            results[trait] = {"error": "No results found"}

    # Persist results to volume
    experiments_volume.commit()

    return {"model_load_time": load_time, "results": results}


@app.local_entrypoint()
def main(
    experiment: str = "dataset_creation_emotions",
    traits: str = "emotions/anger",
    layers: str = "15,20,25",
    model_variant: str = None,
    extraction_variant: str = None,
):
    """
    Test steering eval on Modal.

    Usage:
        modal run inference/modal_steering.py \
            --experiment dataset_creation_emotions \
            --traits emotions/anger \
            --layers 15,20,25
    """
    import json
    import sys
    sys.path.insert(0, str(repo_root))

    from utils.paths import load_experiment_config, get_model_variant

    config = load_experiment_config(experiment)

    if model_variant is None:
        model_variant = config.get("defaults", {}).get("application")
    if extraction_variant is None:
        extraction_variant = config.get("defaults", {}).get("extraction")

    variant_config = get_model_variant(experiment, model_variant)
    model_name = variant_config.model

    trait_list = [t.strip() for t in traits.split(",")]
    layer_list = [int(l) for l in layers.split(",")]

    print(f"Experiment: {experiment}")
    print(f"Model: {model_name} ({model_variant})")
    print(f"Extraction: {extraction_variant}")
    print(f"Traits: {trait_list}")
    print(f"Layers: {layer_list}")

    # Sync data to volumes
    _sync_to_volumes(experiment, trait_list, extraction_variant)

    result = steering_eval_remote.remote(
        model_name=model_name,
        experiment=experiment,
        traits=trait_list,
        model_variant=model_variant,
        extraction_variant=extraction_variant,
        layers=layer_list,
    )

    print(json.dumps(result, indent=2, default=str))


def _sync_to_volumes(experiment, traits, extraction_variant):
    """Upload required data to Modal volumes before eval."""
    exp_dir = repo_root / "experiments" / experiment

    with experiments_volume.batch_upload(force=True) as batch:
        # Experiment config
        config_path = exp_dir / "config.json"
        if config_path.exists():
            batch.put_file(str(config_path), f"{experiment}/config.json")

        # Activation norms (extraction_evaluation.json)
        eval_path = exp_dir / "extraction" / "extraction_evaluation.json"
        if eval_path.exists():
            batch.put_file(str(eval_path), f"{experiment}/extraction/extraction_evaluation.json")

        # Vectors for each trait
        for trait in traits:
            vec_dir = exp_dir / "extraction" / trait / extraction_variant / "vectors"
            if vec_dir.exists():
                batch.put_directory(str(vec_dir), f"{experiment}/extraction/{trait}/{extraction_variant}/vectors")

    with datasets_volume.batch_upload(force=True) as batch:
        for trait in traits:
            ds_dir = repo_root / "datasets" / "traits" / trait
            if ds_dir.exists():
                batch.put_directory(str(ds_dir), f"traits/{trait}")

    print(f"Synced {len(traits)} traits to volumes")
