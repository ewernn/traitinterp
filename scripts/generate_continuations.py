"""
Generate model continuations from WikiText first sentences.

Usage:
    python scripts/generate_continuations.py --experiment prefill-dynamics
    python scripts/generate_continuations.py --experiment prefill-dynamics --temperature 0.7
    python scripts/generate_continuations.py --experiment prefill-dynamics --model google/gemma-2-2b-it
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm

from utils.model import load_model
from utils.generation import generate_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="prefill-dynamics")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--condition", type=str, default=None,
                        help="Output condition name (default: auto from model/temp)")
    args = parser.parse_args()

    # Auto-generate condition name if not specified
    if args.condition is None:
        model_short = args.model.split("/")[-1]
        if args.temperature == 0.0:
            args.condition = model_short
        else:
            temp_str = f"temp{args.temperature}".replace(".", "")
            args.condition = f"{model_short}-{temp_str}"

    # Load WikiText paragraphs
    wikitext_path = Path("datasets/inference/wikitext/paragraphs.json")
    with open(wikitext_path) as f:
        data = json.load(f)
    paragraphs = data["paragraphs"]

    # Load model
    print(f"Loading model {args.model}...")
    model, tokenizer = load_model(args.model)

    # Generate continuations
    prompts = [p["first_sentence"] for p in paragraphs]

    print(f"Generating {len(prompts)} continuations (temp={args.temperature}, max_new_tokens={args.max_new_tokens})...")

    all_responses = generate_batch(
        model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
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

    # Save with condition name
    output_dir = Path(f"experiments/{args.experiment}/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"continuations-{args.condition}.json"
    with open(output_path, "w") as f:
        json.dump({
            "condition": args.condition,
            "model": args.model,
            "temperature": args.temperature,
            "samples": results
        }, f, indent=2)

    print(f"Saved {len(results)} samples to {output_path}")

if __name__ == "__main__":
    main()
