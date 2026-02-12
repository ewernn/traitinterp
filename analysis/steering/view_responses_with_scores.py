#!/usr/bin/env python3
"""
View steering responses with per-response trait and coherence scores.

Input: Steering response JSON files (with trait_score/coherence_score per response)
Output: Formatted display with scores, metadata extracted from file path
Usage:
    python analysis/steering/view_responses_with_scores.py responses.json
    python analysis/steering/view_responses_with_scores.py file1.json file2.json --n 5
    python analysis/steering/view_responses_with_scores.py responses.json --sort trait  # sort by trait score
    python analysis/steering/view_responses_with_scores.py responses.json --sort coherence
    python analysis/steering/view_responses_with_scores.py responses.json --sort disagreement  # biggest |trait - coherence| gap
"""

import json
import argparse
from pathlib import Path


def parse_metadata_from_path(filepath: Path) -> dict:
    """Extract experiment/trait/layer/coef/method/position from file path."""
    parts = filepath.parts
    meta = {}

    # Extract from filename: L14_c6.6_2026-02-03_07-10-27.json or baseline.json
    name = filepath.stem
    if name == "baseline":
        meta["config"] = "baseline"
    elif "_c" in name:
        layer_part = name.split("_c")[0]  # L14
        coef_part = name.split("_c")[1].split("_")[0]  # 6.6
        meta["layer"] = layer_part
        meta["coef"] = coef_part
        meta["config"] = f"{layer_part} c{coef_part}"

    # Walk path parts for known segments
    for i, part in enumerate(parts):
        if part == "experiments" and i + 1 < len(parts):
            meta["experiment"] = parts[i + 1]
        if part in ("pv_instruction", "pv_natural", "pv_replication_natural"):
            meta["extraction"] = part
            if i + 1 < len(parts):
                meta["trait"] = parts[i + 1]
        if part in ("base", "instruct"):
            meta["variant"] = part
        if part.startswith("response_"):
            meta["position"] = part.replace("_", "[", 1).rstrip("/") + "]" if "__" in part else part
        if part in ("mean_diff", "probe", "combined"):
            meta["method"] = part
        if part == "residual":
            meta["component"] = part

    return meta


def print_responses(filepath: Path, responses: list[dict], n: int = None,
                    sort_by: str = None):
    """Print responses with scores."""
    meta = parse_metadata_from_path(filepath)

    # Header
    print(f"\n{'=' * 80}")
    header_parts = []
    for key in ["experiment", "extraction", "trait", "method", "config"]:
        if key in meta:
            header_parts.append(f"{key}={meta[key]}")
    print(f"  {' | '.join(header_parts)}")
    print(f"  FILE: {filepath}")
    print(f"{'=' * 80}")

    # Compute mean scores
    trait_scores = [r.get("trait_score", 0) for r in responses if r.get("trait_score") is not None]
    coh_scores = [r.get("coherence_score", 0) for r in responses if r.get("coherence_score") is not None]
    if trait_scores:
        print(f"  Mean trait: {sum(trait_scores)/len(trait_scores):.1f} | "
              f"Mean coherence: {sum(coh_scores)/len(coh_scores):.1f} | "
              f"N: {len(responses)}")
    print(f"{'-' * 80}")

    # Optionally sort
    indexed = list(enumerate(responses))
    if sort_by == "trait":
        indexed.sort(key=lambda x: x[1].get("trait_score", 0), reverse=True)
    elif sort_by == "coherence":
        indexed.sort(key=lambda x: x[1].get("coherence_score", 0))
    elif sort_by == "disagreement":
        indexed.sort(key=lambda x: abs(x[1].get("trait_score", 50) - x[1].get("coherence_score", 50)), reverse=True)

    to_show = indexed[:n] if n else indexed

    for orig_idx, item in to_show:
        q = item.get("question", item.get("prompt", ""))
        r = item.get("response", "")
        ts = item.get("trait_score")
        cs = item.get("coherence_score")

        score_str = ""
        if ts is not None:
            score_str += f"trait={ts:.0f}"
        if cs is not None:
            score_str += f"  coh={cs:.0f}"

        print(f"\n[{orig_idx + 1}/{len(responses)}] {score_str}")
        print(f"Q: {q}")
        print(f"R: {r}")
        print(f"{'-' * 80}")


def main():
    parser = argparse.ArgumentParser(description="View steering responses with scores")
    parser.add_argument("files", nargs="+", type=Path, help="Response JSON files")
    parser.add_argument("--n", type=int, default=5, help="Number of responses to show")
    parser.add_argument("--sort", choices=["trait", "coherence", "disagreement"],
                        help="Sort responses before display")

    args = parser.parse_args()

    for filepath in args.files:
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            continue

        try:
            with open(filepath) as f:
                data = json.load(f)
            responses = data if isinstance(data, list) else data.get("responses", [])
            print_responses(filepath, responses, args.n, args.sort)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")


if __name__ == "__main__":
    main()
