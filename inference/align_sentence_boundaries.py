"""Align sentence char offsets to token positions for thought branches responses.

Maps sentence_metadata.json char boundaries → token indices using tokenizer offset mapping,
then embeds sentence_boundaries into response JSONs for frontend rendering.

Input:
  - experiments/.../thought_branches/sentence_metadata.json (char offsets, cue_p per sentence)
  - experiments/.../inference/instruct/responses/thought_branches/mmlu_condition_{a,b}/*.json

Output:
  - Updated response JSONs with sentence_boundaries field added:
    [{"sentence_num": 0, "token_start": 0, "token_end": 15, "cue_p": 0.06}, ...]
    Token positions are response-relative (token 0 = first response token).

Usage:
  python inference/align_sentence_boundaries.py --experiment mats-mental-state-circuits
"""

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from utils import paths


def load_sentence_metadata(experiment_dir: Path) -> dict:
    """Load sentence metadata with char offsets and cue_p values."""
    path = experiment_dir / "thought_branches" / "sentence_metadata.json"
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def char_to_token_mapping(response_text: str, tokenizer) -> list[tuple[int, int]]:
    """Tokenize response text and return per-token (char_start, char_end) offsets."""
    encoding = tokenizer(response_text, return_offsets_mapping=True, add_special_tokens=False)
    return encoding["offset_mapping"]


def map_sentences_to_tokens(
    sentences: list[dict],
    token_offsets: list[tuple[int, int]],
) -> list[dict]:
    """Map sentence char boundaries to token indices.

    For each sentence, finds the first token overlapping char_start and
    the last token overlapping char_end. Returns response-relative token positions.
    """
    boundaries = []
    for sent in sentences:
        char_start = sent.get("char_start")
        char_end = sent.get("char_end")

        if char_start is None or char_end is None:
            print(f"  WARNING: Skipping sentence {sent['sentence_num']} — missing char offsets")
            continue

        # Find first token that overlaps with sentence start
        token_start = None
        for i, (tok_cs, tok_ce) in enumerate(token_offsets):
            if tok_ce > char_start:
                token_start = i
                break

        # Find last token that overlaps with sentence end
        token_end = None
        for i in range(len(token_offsets) - 1, -1, -1):
            tok_cs, tok_ce = token_offsets[i]
            if tok_cs < char_end:
                token_end = i + 1  # exclusive end
                break

        if token_start is None or token_end is None:
            print(f"  WARNING: Could not map sentence {sent['sentence_num']} "
                  f"(chars {char_start}-{char_end})")
            continue

        boundaries.append({
            "sentence_num": sent["sentence_num"],
            "token_start": token_start,
            "token_end": token_end,
            "cue_p": sent["cue_p"],
        })

    return boundaries


def verify_token_count(response_json: dict, tokenizer, response_text: str) -> bool:
    """Verify independent tokenization matches stored token count."""
    encoding = tokenizer(response_text, add_special_tokens=False)
    independent_count = len(encoding["input_ids"])
    stored_count = len(response_json["token_ids"]) - response_json["prompt_end"]
    if independent_count != stored_count:
        print(f"  WARNING: Token count mismatch — independent={independent_count}, "
              f"stored={stored_count} (prompt_end={response_json['prompt_end']})")
        return False
    return True


def process_condition(
    response_dir: Path,
    sentence_metadata: dict,
    tokenizer,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Process all response JSONs in a condition directory.

    Returns (processed_count, error_count).
    """
    processed = 0
    errors = 0

    for response_path in sorted(response_dir.glob("*.json")):
        problem_id = response_path.stem

        if problem_id not in sentence_metadata:
            continue

        with open(response_path) as f:
            response_json = json.load(f)

        response_text = response_json["response"]
        sentences = sentence_metadata[problem_id]

        # Verify tokenization consistency
        verify_token_count(response_json, tokenizer, response_text)

        # Get char→token mapping
        token_offsets = char_to_token_mapping(response_text, tokenizer)

        # Map sentences to tokens
        boundaries = map_sentences_to_tokens(sentences, token_offsets)

        if len(boundaries) != len(sentences):
            print(f"  WARNING: {problem_id} — mapped {len(boundaries)}/{len(sentences)} sentences")
            errors += 1

        # Verify coverage: first sentence should start near token 0,
        # last sentence should end near the last token
        if boundaries:
            first_start = boundaries[0]["token_start"]
            last_end = boundaries[-1]["token_end"]
            total_response_tokens = len(response_json["token_ids"]) - response_json["prompt_end"]
            if first_start > 2:
                print(f"  WARNING: {problem_id} — first sentence starts at token {first_start}")
            coverage = last_end / total_response_tokens if total_response_tokens > 0 else 0
            print(f"  {problem_id}: {len(boundaries)} sentences, "
                  f"tokens 0-{last_end}/{total_response_tokens} "
                  f"({coverage:.0%} coverage)")

        if not dry_run:
            response_json["sentence_boundaries"] = boundaries
            with open(response_path, "w") as f:
                json.dump(response_json, f, indent=2)

        processed += 1

    return processed, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--dry-run", action="store_true", help="Print mappings without writing")
    args = parser.parse_args()

    experiment_dir = paths.get("experiments.base", experiment=args.experiment)

    # Load sentence metadata
    sentence_metadata = load_sentence_metadata(experiment_dir)
    print(f"Loaded sentence metadata: {len(sentence_metadata)} problems")

    # Load experiment config to get model
    config_path = experiment_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    model_id = config["model_variants"]["instruct"]["model"]
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Process conditions A and B (both use unfaithful CoT, same sentence metadata)
    # Condition C has faithful CoT — different text, no sentence metadata
    response_base = experiment_dir / "inference" / "instruct" / "responses" / "thought_branches"

    total_processed = 0
    total_errors = 0

    for condition in ["mmlu_condition_a", "mmlu_condition_b"]:
        condition_dir = response_base / condition
        if not condition_dir.exists():
            print(f"Skipping {condition} — directory not found")
            continue

        print(f"\n=== {condition} ===")
        processed, errors = process_condition(
            condition_dir, sentence_metadata, tokenizer, dry_run=args.dry_run,
        )
        total_processed += processed
        total_errors += errors

    print(f"\nDone: {total_processed} files processed, {total_errors} with warnings")
    if args.dry_run:
        print("(dry run — no files written)")


if __name__ == "__main__":
    main()
