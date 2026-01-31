"""
Capture activations during prefill for human vs model text.

Usage:
    python scripts/capture_prefill_activations.py --experiment prefill-dynamics
"""

import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm

from utils.model import load_model
from inference.capture_raw_activations import capture_residual_stream_prefill

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="prefill-dynamics")
    args = parser.parse_args()

    # Load data
    data_path = Path(f"experiments/{args.experiment}/data/continuations.json")
    with open(data_path) as f:
        data = json.load(f)
    samples = data["samples"]

    # Load model
    print("Loading model...")
    model, tokenizer = load_model("google/gemma-2-2b")
    n_layers = model.config.num_hidden_layers

    # Output directories
    output_dir = Path(f"experiments/{args.experiment}/activations")
    human_dir = output_dir / "human"
    model_dir = output_dir / "model"
    human_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Capturing activations for {len(samples)} samples...")

    for sample in tqdm(samples):
        sample_id = sample["id"]
        first_sentence = sample["first_sentence"]

        # Human condition: prefill full human text
        # Use first_sentence as "prompt" and human_continuation as "response"
        human_data = capture_residual_stream_prefill(
            model, tokenizer,
            prompt_text=first_sentence,
            response_text=sample["human_continuation"],
            n_layers=n_layers,
        )
        torch.save(human_data, human_dir / f"{sample_id}.pt")

        # Model condition: prefill model-generated text
        model_data = capture_residual_stream_prefill(
            model, tokenizer,
            prompt_text=first_sentence,
            response_text=sample["model_continuation"],
            n_layers=n_layers,
        )
        torch.save(model_data, model_dir / f"{sample_id}.pt")

    print(f"Saved activations to {output_dir}")

if __name__ == "__main__":
    main()
