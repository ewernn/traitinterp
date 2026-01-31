"""
Generate model continuations from WikiText first sentences.

Usage:
    python scripts/generate_continuations.py --experiment prefill-dynamics
"""

import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm

from utils.model import load_model
from utils.generation import generate_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="prefill-dynamics")
    parser.add_argument("--max-new-tokens", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # Load WikiText paragraphs
    wikitext_path = Path("datasets/inference/wikitext/paragraphs.json")
    with open(wikitext_path) as f:
        data = json.load(f)
    paragraphs = data["paragraphs"]

    # Load model
    print("Loading model...")
    model, tokenizer = load_model("google/gemma-2-2b")

    # Generate continuations
    prompts = [p["first_sentence"] for p in paragraphs]

    print(f"Generating {len(prompts)} continuations (temp=0, max_new_tokens={args.max_new_tokens})...")

    # generate_batch returns List[str] directly
    all_responses = generate_batch(
        model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )

    results = []
    for idx, response in enumerate(all_responses):
        results.append({
            "id": paragraphs[idx]["id"],
            "first_sentence": paragraphs[idx]["first_sentence"],
            "model_continuation": response,
            "human_continuation": paragraphs[idx]["continuation"],
            "full_model_text": paragraphs[idx]["first_sentence"] + " " + response,
            "full_human_text": paragraphs[idx]["full_text"],
        })

    # Save
    output_dir = Path(f"experiments/{args.experiment}/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "continuations.json"
    with open(output_path, "w") as f:
        json.dump({"samples": results}, f, indent=2)

    print(f"Saved {len(results)} samples to {output_path}")

if __name__ == "__main__":
    main()
