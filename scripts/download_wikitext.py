"""
Download WikiText-2 and extract paragraphs for prefill dynamics experiment.

Usage:
    python scripts/download_wikitext.py --n-samples 50 --min-tokens 80 --max-tokens 120
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

def extract_paragraphs(n_samples: int, min_tokens: int, max_tokens: int, tokenizer_name: str):
    """Extract paragraphs of target length from WikiText-2."""

    # Load dataset and tokenizer
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    paragraphs = []

    for item in ds:
        text = item["text"].strip()

        # Skip headers (start with =) and empty lines
        if not text or text.startswith("="):
            continue

        # Tokenize to check length
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if min_tokens <= len(tokens) <= max_tokens:
            # Extract first sentence for prompt (split on first period)
            first_period = text.find(". ")
            if first_period > 10:  # Ensure meaningful first sentence
                first_sentence = text[:first_period + 1]
                continuation = text[first_period + 2:]

                paragraphs.append({
                    "id": len(paragraphs) + 1,
                    "full_text": text,
                    "first_sentence": first_sentence,
                    "continuation": continuation,
                    "n_tokens": len(tokens),
                })

                if len(paragraphs) >= n_samples:
                    break

    return paragraphs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--min-tokens", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--tokenizer", default="google/gemma-2-2b")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/inference/wikitext"))
    args = parser.parse_args()

    print(f"Extracting {args.n_samples} paragraphs ({args.min_tokens}-{args.max_tokens} tokens)...")
    paragraphs = extract_paragraphs(args.n_samples, args.min_tokens, args.max_tokens, args.tokenizer)

    print(f"Found {len(paragraphs)} paragraphs")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "name": "WikiText-2 paragraphs",
        "description": f"{len(paragraphs)} paragraphs from WikiText-2 test set, {args.min_tokens}-{args.max_tokens} tokens",
        "paragraphs": paragraphs,
    }

    output_path = args.output_dir / "paragraphs.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {output_path}")

    # Print sample
    if paragraphs:
        p = paragraphs[0]
        print(f"\nSample paragraph ({p['n_tokens']} tokens):")
        print(f"  First sentence: {p['first_sentence'][:80]}...")
        print(f"  Continuation: {p['continuation'][:80]}...")

if __name__ == "__main__":
    main()
