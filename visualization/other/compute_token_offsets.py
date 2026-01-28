"""
Compute token character offsets for extraction responses.

Purpose: Generate token boundary data for frontend visualization. Allows the
UI to highlight which characters correspond to the first N tokens in extraction
responses (e.g., highlighting "first 5 response tokens" in the extraction data viewer).

Input:
    - Extraction responses: experiments/{exp}/extraction/{trait}/{variant}/responses/
    - Model name from responses/metadata.json

Output:
    - responses/token_offsets.json with character ranges per token per response

Usage:
    python visualization/other/compute_token_offsets.py \\
        --experiment gemma-2-2b \\
        --trait chirp/refusal \\
        [--variant base]

Format:
    {
      "pos": [
        [[0, 7], [7, 8], [9, 11], ...],  // response 0: [start, end] per token
        [[0, 4], [4, 8], ...],           // response 1
        ...
      ],
      "neg": [...]
    }
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoTokenizer

from utils.paths import get as get_path


def compute_offsets(responses_dir: Path) -> dict:
    """Compute token character offsets for all responses in a directory."""

    # Load tokenizer from metadata
    with open(responses_dir / 'metadata.json') as f:
        model_name = json.load(f)['model']

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    offsets = {}
    for split in ['pos', 'neg']:
        split_path = responses_dir / f'{split}.json'
        if not split_path.exists():
            print(f"  Skipping {split}.json (not found)")
            continue

        with open(split_path) as f:
            responses = json.load(f)

        split_offsets = []
        for r in responses:
            response_text = r.get('response', '')
            if not response_text:
                split_offsets.append([])
                continue

            # Get character offsets for each token
            encoding = tokenizer(response_text, return_offsets_mapping=True)
            # offset_mapping is [(start, end), ...] for each token
            # Filter out special tokens (offset (0,0) typically)
            token_ranges = [
                list(offset) for offset in encoding['offset_mapping']
                if offset != (0, 0)
            ]
            split_offsets.append(token_ranges)

        offsets[split] = split_offsets
        print(f"  {split}: {len(split_offsets)} responses")

    return offsets


def main():
    parser = argparse.ArgumentParser(description='Compute token offsets for extraction responses')
    parser.add_argument('--experiment', required=True, help='Experiment name')
    parser.add_argument('--trait', required=True, help='Trait path (e.g., chirp/refusal)')
    parser.add_argument('--variant', default='base', help='Model variant (default: base)')
    args = parser.parse_args()

    responses_dir = get_path(
        'extraction.responses',
        experiment=args.experiment,
        trait=args.trait,
        model_variant=args.variant
    )

    if not responses_dir.exists():
        print(f"ERROR: Responses directory not found: {responses_dir}")
        return 1

    print(f"Computing token offsets for: {responses_dir}")
    offsets = compute_offsets(responses_dir)

    output_path = responses_dir / 'token_offsets.json'
    with open(output_path, 'w') as f:
        json.dump(offsets, f)

    print(f"Saved: {output_path}")
    return 0


if __name__ == '__main__':
    exit(main())
