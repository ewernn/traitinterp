"""
Compute perplexity (CE loss) for human vs model text.

Usage:
    python scripts/compute_perplexity.py --experiment prefill-dynamics
"""

import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm

from utils.model import load_model
from utils.metrics import sequence_ce_loss

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

    results = []

    print(f"Computing CE loss for {len(samples)} samples...")
    for sample in tqdm(samples):
        # CE loss on full text (prompt + continuation)
        human_ce = sequence_ce_loss(model, tokenizer, sample["full_human_text"])
        model_ce = sequence_ce_loss(model, tokenizer, sample["full_model_text"])

        results.append({
            "id": sample["id"],
            "human_ce": human_ce,
            "model_ce": model_ce,
            "ce_diff": human_ce - model_ce,  # Positive = human more surprising
        })

    # Save
    output_dir = Path(f"experiments/{args.experiment}/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "perplexity.json"
    with open(output_path, "w") as f:
        json.dump({"results": results}, f, indent=2)

    # Summary stats
    human_mean = sum(r["human_ce"] for r in results) / len(results)
    model_mean = sum(r["model_ce"] for r in results) / len(results)

    print(f"\nResults:")
    print(f"  Human CE (mean): {human_mean:.4f}")
    print(f"  Model CE (mean): {model_mean:.4f}")
    print(f"  Diff (human - model): {human_mean - model_mean:.4f}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
