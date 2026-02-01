
#!/usr/bin/env python3
"""
Add success tags to response files.

Reads a file containing success IDs and tags matching response files with 'success'.

Input:
    Success IDs file with structure: {"prompts": [{"id": 1}, {"id": 2}, ...]}

Output:
    Updates response JSON files with tags field
    Creates _tags.json index file for frontend

Usage:
    python scripts/tag_successes.py \
        --experiment gemma-2-2b \
        --prompt-set jailbreak/original \
        --successes-file datasets/inference/jailbreak/successes.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.paths import get_inference_responses_dir, get_experiment_config
from utils.json import dump_compact


def main():
    parser = argparse.ArgumentParser(description="Add success tags to response files")
    parser.add_argument('--experiment', required=True, help='Experiment name')
    parser.add_argument('--prompt-set', required=True, help='Prompt set name (e.g., jailbreak/original)')
    parser.add_argument('--model-variant', default=None, help='Model variant (defaults to experiment application variant)')
    parser.add_argument('--successes-file', required=True, help='Path to JSON file with success IDs')
    args = parser.parse_args()

    # Resolve model variant from experiment config if not specified
    model_variant = args.model_variant
    if model_variant is None:
        config = get_experiment_config(args.experiment)
        model_variant = config.get('defaults', {}).get('application', 'instruct')
        print(f"Using model variant from experiment config: {model_variant}")

    # Load success IDs
    successes_path = Path(args.successes_file)
    if not successes_path.exists():
        print(f"Error: Successes file not found: {successes_path}")
        sys.exit(1)

    with open(successes_path) as f:
        successes = json.load(f)
    success_ids = {p['id'] for p in successes['prompts']}
    print(f"Loaded {len(success_ids)} success IDs")

    # Get responses directory
    responses_dir = get_inference_responses_dir(args.experiment, model_variant, args.prompt_set)
    if not responses_dir.exists():
        print(f"Error: Responses directory not found: {responses_dir}")
        sys.exit(1)

    # Track all tags for index file
    tags_index = {}
    updated = 0
    total = 0

    for response_file in responses_dir.glob('*.json'):
        if response_file.name.startswith('_'):
            continue  # Skip index files

        total += 1
        prompt_id = int(response_file.stem)

        with open(response_file) as f:
            data = json.load(f)

        # Add tag if successful
        if prompt_id in success_ids:
            data['tags'] = ['success']
            tags_index[prompt_id] = ['success']
            updated += 1
        else:
            # Ensure empty tags for non-successes
            data['tags'] = []

        with open(response_file, 'w') as f:
            dump_compact(data, f)

    # Write tags index file for frontend preloading
    tags_index_file = responses_dir / '_tags.json'
    with open(tags_index_file, 'w') as f:
        json.dump(tags_index, f)

    print(f"Tagged {updated} response files with 'success'")
    print(f"Total responses: {total}")
    print(f"Wrote tags index: {tags_index_file}")


if __name__ == '__main__':
    main()
