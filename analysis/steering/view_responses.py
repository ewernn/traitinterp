#!/usr/bin/env python3
"""
View steering responses in readable format.

Usage:
    python analysis/steering/view_responses.py responses.json
    python analysis/steering/view_responses.py responses.json --n 5
    python analysis/steering/view_responses.py file1.json file2.json --n 3
"""

import sys
import json
import argparse
from pathlib import Path


def load_responses(filepath: Path) -> list[dict]:
    """Load responses from steering output JSON."""
    with open(filepath) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'responses' in data:
        return data['responses']
    else:
        raise ValueError(f"Unknown format in {filepath}")


def print_responses(filepath: Path, responses: list[dict], n: int = None):
    """Print responses in readable format."""
    # Extract coef from filename if present
    name = filepath.stem
    if '_c' in name:
        coef = name.split('_c')[1].split('_')[0]
        label = f"c{coef}"
    else:
        label = name

    print(f"\n{'='*70}")
    print(f"FILE: {filepath.name} ({label})")
    print(f"{'='*70}")

    to_show = responses[:n] if n else responses
    total = len(responses)

    for i, item in enumerate(to_show):
        q = item.get("question", item.get("prompt", ""))
        r = item.get("response", "")

        print(f"\n[{i+1}/{total}] Q: {q}")
        print(f"R: {r}")
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="View steering responses")
    parser.add_argument("files", nargs="+", type=Path, help="Response JSON files")
    parser.add_argument("--n", type=int, default=5, help="Number of responses to show (default: 5)")

    args = parser.parse_args()

    for filepath in args.files:
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            continue

        try:
            responses = load_responses(filepath)
            print_responses(filepath, responses, args.n)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")


if __name__ == "__main__":
    main()
