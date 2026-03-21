"""
Download BeaverTails-Evaluation dataset and convert to inference format.

Usage:
    python utils/download_beavertails.py
    python utils/download_beavertails.py --split 30k_test  # larger set (3k examples)
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


CATEGORY_KEYS = [
    "animal_abuse",
    "child_abuse",
    "controversial_topics,_politics",
    "discrimination,_stereotype,_injustice",
    "drug_abuse,_weapons,_banned_substance",
    "financial_crime,_property_crime,_theft",
    "hate_speech,_offensive_language",
    "misinformation_regarding_ethics,_laws_and_safety",
    "non_violent_unethical_behavior",
    "privacy_violation",
    "self_harm",
    "sexually_explicit,_adult_content",
    "terrorism,_organized_crime",
    "violence,_aiding_and_abetting,_incitement",
]

# Short names for notes/filenames (normalized: no underscores after commas)
CATEGORY_SHORT = {
    "animal_abuse": "animal",
    "child_abuse": "child",
    "controversial_topics,politics": "politics",
    "discrimination,stereotype,injustice": "discrimination",
    "drug_abuse,weapons,banned_substance": "drugs_weapons",
    "financial_crime,property_crime,theft": "financial",
    "hate_speech,offensive_language": "hate_speech",
    "misinformation_regarding_ethics,laws_and_safety": "misinfo",
    "non_violent_unethical_behavior": "unethical",
    "privacy_violation": "privacy",
    "self_harm": "self_harm",
    "sexually_explicit,adult_content": "sexual",
    "terrorism,organized_crime": "terrorism",
    "violence,aiding_and_abetting,incitement": "violence",
}


def normalize_category(cat: str) -> str:
    """Normalize category key (remove underscores after commas)."""
    return cat.replace(",_", ",")


def get_categories(example: dict, is_evaluation: bool = False) -> list[str]:
    """Get list of active harm categories for an example."""
    if is_evaluation:
        # Evaluation set has single 'category' field
        cat = example.get("category", "")
        cat_normalized = normalize_category(cat)
        return [CATEGORY_SHORT.get(cat_normalized, cat_normalized)] if cat else []
    else:
        # Main dataset has boolean flags per category
        return [CATEGORY_SHORT[normalize_category(k)] for k in CATEGORY_KEYS if example.get(k, False)]


def download_and_convert(split: str, output_dir: Path):
    """Download dataset and convert to inference format."""

    is_evaluation = split == "evaluation"

    if is_evaluation:
        ds = load_dataset("PKU-Alignment/BeaverTails-Evaluation", split="test")
        name = "BeaverTails-Evaluation"
        desc = "700 curated prompts for safety evaluation (PKU-Alignment)"
    else:
        ds = load_dataset("PKU-Alignment/BeaverTails", split=split)
        name = f"BeaverTails-{split}"
        desc = f"BeaverTails {split} split (PKU-Alignment)"

    print(f"Downloaded {len(ds)} examples")

    # Convert to inference format
    prompts = []
    per_category = {short: [] for short in CATEGORY_SHORT.values()}

    for i, ex in enumerate(ds):
        prompt_text = ex.get("prompt", ex.get("question", ""))
        categories = get_categories(ex, is_evaluation=is_evaluation)
        is_safe = ex.get("is_safe", len(categories) == 0)  # evaluation set: assume unsafe if has category

        prompt_entry = {
            "id": i + 1,
            "text": prompt_text,
            "note": ", ".join(categories) if categories else ("safe" if is_safe else "unsafe"),
            "is_safe": is_safe,
            "categories": categories,
        }
        prompts.append(prompt_entry)

        # Add to per-category lists
        for cat in categories:
            per_category[cat].append(prompt_entry)

    # Save combined file
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "name": name,
        "description": desc,
        "prompts": prompts,
    }
    combined_path = output_dir / "all.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved {len(prompts)} prompts to {combined_path}")

    # Save per-category files (only if non-empty)
    for cat, cat_prompts in per_category.items():
        if cat_prompts:
            cat_data = {
                "name": f"BeaverTails - {cat}",
                "description": f"{len(cat_prompts)} prompts tagged with '{cat}'",
                "prompts": cat_prompts,
            }
            cat_path = output_dir / f"{cat}.json"
            with open(cat_path, "w") as f:
                json.dump(cat_data, f, indent=2)
            print(f"  {cat}: {len(cat_prompts)} prompts")

    # Print summary
    print(f"\nSummary:")
    print(f"  Total: {len(prompts)}")
    print(f"  Safe: {sum(1 for p in prompts if p['is_safe'])}")
    print(f"  Unsafe: {sum(1 for p in prompts if not p['is_safe'])}")


def main():
    parser = argparse.ArgumentParser(description="Download BeaverTails dataset")
    parser.add_argument(
        "--split",
        default="evaluation",
        choices=["evaluation", "30k_test", "30k_train", "330k_test", "330k_train"],
        help="Dataset split to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: datasets/inference/beavertails/)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        repo_root = Path(__file__).parent.parent
        args.output_dir = repo_root / "datasets" / "inference" / "beavertails"

    download_and_convert(args.split, args.output_dir)


if __name__ == "__main__":
    main()
