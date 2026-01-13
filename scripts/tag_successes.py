#!/usr/bin/env python3
"""Add success tags to jailbreak response files."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.json import dump_compact

# Load success IDs
with open('datasets/inference/jailbreak/successes.json') as f:
    successes = json.load(f)
success_ids = {p['id'] for p in successes['prompts']}

# Path to response files
responses_dir = Path('experiments/gemma-2-2b/inference/instruct/responses/jailbreak')

updated = 0
for response_file in responses_dir.glob('*.json'):
    prompt_id = int(response_file.stem)

    with open(response_file) as f:
        data = json.load(f)

    # Add tag if successful
    if prompt_id in success_ids:
        data['metadata']['tags'] = ['success']
        updated += 1
    else:
        # Ensure no tags for non-successes
        data['metadata'].pop('tags', None)

    with open(response_file, 'w') as f:
        dump_compact(data, f)

print(f"Tagged {updated} response files with 'success'")
print(f"Total responses: {len(list(responses_dir.glob('*.json')))}")
