"""Systematic extraction quality audit: first-5-token analysis + steering cross-reference.

Input: Extraction responses (pos.json, neg.json), steering results (results.jsonl), trait definitions
Output: Comprehensive quality report with per-trait grades and cross-referencing

Usage:
    python analysis/extraction_quality_audit.py --experiment mats-alignment-faking
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from utils.vectors import MIN_COHERENCE

from transformers import AutoTokenizer


def load_responses(path):
    """Load extraction response JSON."""
    with open(path) as f:
        return json.load(f)


def tokenize_first_n(tokenizer, text, n=5):
    """Tokenize text and return first n tokens as strings."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in ids[:n]]
    return tokens, ids[:n]


def parse_steering_results(results_path):
    """Parse steering results.jsonl, return best delta with coherence >= 70."""
    baseline_mean = None
    best_delta = 0.0
    best_config = None
    direction = None

    with open(results_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("type") == "header":
                direction = entry.get("direction")
            elif entry.get("type") == "baseline":
                baseline_mean = entry["result"]["trait_mean"]
            elif entry.get("type") != "header" and "result" in entry and "config" in entry:
                if baseline_mean is None:
                    continue
                delta = entry["result"]["trait_mean"] - baseline_mean
                coherence = entry["result"].get("coherence_mean", 0)
                if coherence >= MIN_COHERENCE and abs(delta) > abs(best_delta):
                    best_delta = delta
                    best_config = entry["config"]

    # Effective delta accounts for steering direction
    if direction == "negative":
        effective_delta = -best_delta  # suppressing should give negative raw delta
    else:
        effective_delta = best_delta  # positive direction, delta should be positive

    return {
        "baseline": baseline_mean,
        "best_delta": best_delta,
        "direction": direction,
        "effective_delta": effective_delta,
        "best_config": best_config,
    }


def classify_first5_quality(pos_tokens_list, neg_tokens_list, trait_name):
    """Classify quality of first-5-token extraction window.

    Returns dict with issue flags and details.
    """
    issues = []

    # Check pos examples
    pos_problems = 0
    pos_details = []
    for i, (tokens, _) in enumerate(pos_tokens_list):
        text = "".join(tokens).lower().strip()

        # Check for reversal patterns in first 5 tokens
        reversal_patterns = [
            r"^i('m| am) (not |sorry)",
            r"^no[,.]",
            r"^i (don't|do not|can't|cannot)",
            r"^(actually|honestly|well),? (i|no|that)",
            r"^i (was wrong|made a mistake)",
            r"^that('s| is) (not |wrong|incorrect)",
        ]

        is_reversal = False
        for pat in reversal_patterns:
            if re.search(pat, text):
                is_reversal = True
                break

        # Check for generic/uninformative starts
        generic_patterns = [
            r'^"?\s*$',  # empty or just quote
            r"^i('m| am) (going|trying|working)",  # too generic, not trait-specific
        ]

        is_generic = False
        for pat in generic_patterns:
            if re.search(pat, text):
                is_generic = True
                break

        if is_reversal:
            pos_problems += 1
            pos_details.append(f"pos[{i}] REVERSAL: '{text}'")
        elif is_generic:
            pos_details.append(f"pos[{i}] GENERIC: '{text}'")

    # Check neg examples
    neg_problems = 0
    neg_details = []
    for i, (tokens, _) in enumerate(neg_tokens_list):
        text = "".join(tokens).lower().strip()
        # neg contamination is harder to detect automatically at 5 tokens
        # We flag if neg starts identically to pos patterns

    # Check pos-neg overlap (same first 5 tokens)
    pos_texts = set("".join(t).lower() for t, _ in pos_tokens_list)
    neg_texts = set("".join(t).lower() for t, _ in neg_tokens_list)
    overlap = pos_texts & neg_texts
    if overlap:
        issues.append(f"OVERLAP: {len(overlap)} pos/neg pairs have identical first-5 tokens")

    if pos_problems > 0:
        issues.append(f"POS_REVERSAL: {pos_problems}/{len(pos_tokens_list)} pos examples show reversal in first 5 tokens")

    # Compute pos-neg similarity at token level
    # How many pos responses start with "I" vs neg?
    pos_starts = defaultdict(int)
    neg_starts = defaultdict(int)
    for tokens, _ in pos_tokens_list:
        if tokens:
            pos_starts[tokens[0].strip().lower()] += 1
    for tokens, _ in neg_tokens_list:
        if tokens:
            neg_starts[tokens[0].strip().lower()] += 1

    return {
        "pos_problems": pos_problems,
        "pos_total": len(pos_tokens_list),
        "neg_problems": neg_problems,
        "neg_total": len(neg_tokens_list),
        "issues": issues,
        "pos_details": pos_details,
        "neg_details": neg_details,
        "overlap_count": len(overlap) if overlap else 0,
        "pos_starts": dict(pos_starts),
        "neg_starts": dict(neg_starts),
    }


def compute_pos_neg_distinctiveness(pos_tokens_list, neg_tokens_list):
    """Measure how distinct pos vs neg are at the first-5-token level.

    Returns Jaccard distance of token sets and other metrics.
    """
    pos_token_bags = []
    neg_token_bags = []

    for tokens, _ in pos_tokens_list:
        pos_token_bags.append(set(t.strip().lower() for t in tokens if t.strip()))
    for tokens, _ in neg_tokens_list:
        neg_token_bags.append(set(t.strip().lower() for t in tokens if t.strip()))

    # Union of all pos tokens vs all neg tokens
    all_pos = set()
    all_neg = set()
    for bag in pos_token_bags:
        all_pos |= bag
    for bag in neg_token_bags:
        all_neg |= bag

    intersection = all_pos & all_neg
    union = all_pos | all_neg

    jaccard = len(intersection) / len(union) if union else 1.0
    pos_unique = all_pos - all_neg
    neg_unique = all_neg - all_pos

    return {
        "jaccard_similarity": jaccard,
        "shared_tokens": len(intersection),
        "pos_unique_tokens": sorted(pos_unique)[:10],
        "neg_unique_tokens": sorted(neg_unique)[:10],
        "total_pos_vocab": len(all_pos),
        "total_neg_vocab": len(all_neg),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--category", default="emotion_set")
    parser.add_argument("--n-tokens", type=int, default=5, help="Number of tokens to analyze")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    exp_dir = Path("experiments") / args.experiment
    extraction_dir = exp_dir / "extraction" / args.category
    steering_dir = exp_dir / "steering" / args.category

    # Get model for tokenizer
    config = json.load(open(exp_dir / "config.json"))
    base_model = config["model_variants"]["base"]["model"]

    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Find all traits
    trait_dirs = sorted([d.name for d in extraction_dir.iterdir() if d.is_dir()])
    print(f"Found {len(trait_dirs)} traits")

    results = {}

    for trait in trait_dirs:
        pos_path = extraction_dir / trait / "base" / "responses" / "pos.json"
        neg_path = extraction_dir / trait / "base" / "responses" / "neg.json"

        if not pos_path.exists() or not neg_path.exists():
            print(f"  SKIP {trait}: missing responses")
            continue

        pos_responses = load_responses(pos_path)
        neg_responses = load_responses(neg_path)

        # Tokenize first N tokens of each response
        pos_tokens = [tokenize_first_n(tokenizer, r["response"], args.n_tokens) for r in pos_responses]
        neg_tokens = [tokenize_first_n(tokenizer, r["response"], args.n_tokens) for r in neg_responses]

        # Quality classification
        quality = classify_first5_quality(pos_tokens, neg_tokens, trait)

        # Distinctiveness
        distinctiveness = compute_pos_neg_distinctiveness(pos_tokens, neg_tokens)

        # Steering results
        steering = None
        # Try both positive and negative steering paths
        for direction_name in ["steering"]:
            steering_path = steering_dir / trait / "instruct" / "response__5" / direction_name / "results.jsonl"
            if steering_path.exists():
                steering = parse_steering_results(steering_path)

        # Store all first-5-token texts for review
        pos_first5 = [" ".join(tokens) for tokens, _ in pos_tokens]
        neg_first5 = [" ".join(tokens) for tokens, _ in neg_tokens]

        results[trait] = {
            "quality": quality,
            "distinctiveness": distinctiveness,
            "steering": steering,
            "pos_first5": pos_first5,
            "neg_first5": neg_first5,
            "n_pos": len(pos_responses),
            "n_neg": len(neg_responses),
        }

    # === Analysis ===

    print("\n" + "="*80)
    print("EXTRACTION QUALITY AUDIT")
    print("="*80)

    # 1. Steering performance distribution
    print("\n## Steering Performance Distribution")
    steering_deltas = {}
    for trait, r in results.items():
        if r["steering"]:
            steering_deltas[trait] = r["steering"]["best_delta"]

    if steering_deltas:
        sorted_by_delta = sorted(steering_deltas.items(), key=lambda x: abs(x[1]), reverse=True)

        strong = [(t, d) for t, d in sorted_by_delta if abs(d) >= 20]
        medium = [(t, d) for t, d in sorted_by_delta if 10 <= abs(d) < 20]
        weak = [(t, d) for t, d in sorted_by_delta if 0 < abs(d) < 10]
        dead = [(t, d) for t, d in sorted_by_delta if abs(d) == 0]

        print(f"  Strong (|Δ|≥20): {len(strong)}")
        print(f"  Medium (10≤|Δ|<20): {len(medium)}")
        print(f"  Weak (0<|Δ|<10): {len(weak)}")
        print(f"  Dead (Δ=0): {len(dead)}")

    # 2. First-5-token quality issues
    print("\n## First-5-Token Quality Issues")
    traits_with_issues = []
    for trait, r in sorted(results.items()):
        q = r["quality"]
        if q["issues"] or q["pos_problems"] > 0 or q["overlap_count"] > 0:
            traits_with_issues.append((trait, q))

    print(f"  Traits with issues: {len(traits_with_issues)}/{len(results)}")
    for trait, q in traits_with_issues:
        delta = steering_deltas.get(trait, "N/A")
        delta_str = f"Δ={delta:.1f}" if isinstance(delta, float) else delta
        print(f"  {trait}: {', '.join(q['issues'])} [{delta_str}]")
        for d in q["pos_details"][:3]:
            print(f"    {d}")

    # 3. Pos-neg overlap analysis
    print("\n## Pos-Neg First-5 Token Overlap")
    high_overlap = [(t, r["distinctiveness"]["jaccard_similarity"])
                    for t, r in results.items()
                    if r["distinctiveness"]["jaccard_similarity"] > 0.6]
    high_overlap.sort(key=lambda x: x[1], reverse=True)

    print(f"  High overlap (Jaccard > 0.6): {len(high_overlap)} traits")
    for trait, jaccard in high_overlap[:20]:
        delta = steering_deltas.get(trait, "N/A")
        delta_str = f"Δ={delta:.1f}" if isinstance(delta, float) else delta
        d = results[trait]["distinctiveness"]
        print(f"  {trait}: Jaccard={jaccard:.2f}, "
              f"pos_unique={d['pos_unique_tokens'][:5]}, "
              f"neg_unique={d['neg_unique_tokens'][:5]} [{delta_str}]")

    # 4. Cross-reference: flagged traits vs steering performance
    print("\n## Cross-Reference: Quality Issues vs Steering")
    flagged_traits = set(t for t, _ in traits_with_issues)

    if steering_deltas:
        flagged_weak = [t for t in flagged_traits if t in steering_deltas and abs(steering_deltas[t]) < 10]
        flagged_strong = [t for t in flagged_traits if t in steering_deltas and abs(steering_deltas[t]) >= 20]
        unflagged_weak = [t for t in steering_deltas if abs(steering_deltas[t]) < 10 and t not in flagged_traits]

        print(f"  Flagged AND weak steering: {len(flagged_weak)}")
        for t in flagged_weak:
            print(f"    {t}: Δ={steering_deltas[t]:.1f}")
        print(f"  Flagged BUT strong steering: {len(flagged_strong)}")
        for t in flagged_strong:
            print(f"    {t}: Δ={steering_deltas[t]:.1f}")
        print(f"  Unflagged BUT weak steering: {len(unflagged_weak)}")
        for t in unflagged_weak:
            print(f"    {t}: Δ={steering_deltas[t]:.1f}")

    # 5. Worst traits by multiple criteria
    print("\n## Worst Traits (multiple criteria)")
    trait_scores = {}
    for trait, r in results.items():
        score = 0
        # High overlap
        if r["distinctiveness"]["jaccard_similarity"] > 0.7:
            score += 2
        elif r["distinctiveness"]["jaccard_similarity"] > 0.5:
            score += 1
        # Reversal issues
        score += r["quality"]["pos_problems"] * 2
        # Overlap
        score += r["quality"]["overlap_count"]
        # Weak steering
        if trait in steering_deltas and abs(steering_deltas[trait]) < 5:
            score += 3
        elif trait in steering_deltas and abs(steering_deltas[trait]) < 10:
            score += 1
        trait_scores[trait] = score

    worst = sorted(trait_scores.items(), key=lambda x: x[1], reverse=True)[:30]
    print("  Top 30 worst:")
    for trait, score in worst:
        if score == 0:
            break
        delta = steering_deltas.get(trait, "N/A")
        delta_str = f"Δ={delta:.1f}" if isinstance(delta, float) else delta
        jaccard = results[trait]["distinctiveness"]["jaccard_similarity"]
        q = results[trait]["quality"]
        print(f"  score={score} {trait}: Jaccard={jaccard:.2f}, "
              f"pos_rev={q['pos_problems']}/{q['pos_total']}, "
              f"overlap={q['overlap_count']} [{delta_str}]")

    # 6. Print all first-5 tokens for worst traits
    print("\n## First-5 Token Details (worst traits)")
    for trait, score in worst[:10]:
        if score == 0:
            break
        r = results[trait]
        print(f"\n### {trait} (score={score})")
        print(f"  POS first-5 tokens:")
        for i, t in enumerate(r["pos_first5"]):
            print(f"    [{i}] {t}")
        print(f"  NEG first-5 tokens:")
        for i, t in enumerate(r["neg_first5"]):
            print(f"    [{i}] {t}")

    # 7. Summary stats
    print("\n## Summary")
    all_jaccards = [r["distinctiveness"]["jaccard_similarity"] for r in results.values()]
    print(f"  Total traits: {len(results)}")
    print(f"  Mean Jaccard similarity: {sum(all_jaccards)/len(all_jaccards):.3f}")
    print(f"  Traits with Jaccard > 0.7: {sum(1 for j in all_jaccards if j > 0.7)}")
    print(f"  Traits with reversal in first 5 tokens: {sum(1 for r in results.values() if r['quality']['pos_problems'] > 0)}")
    print(f"  Traits with pos-neg text overlap: {sum(1 for r in results.values() if r['quality']['overlap_count'] > 0)}")

    # Save full results
    output_path = args.output or str(exp_dir / "analysis" / "extraction_quality_audit.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Make JSON serializable
    serializable = {}
    for trait, r in results.items():
        sr = {
            "quality": {k: v for k, v in r["quality"].items()},
            "distinctiveness": r["distinctiveness"],
            "steering_delta": steering_deltas.get(trait),
            "pos_first5": r["pos_first5"],
            "neg_first5": r["neg_first5"],
            "n_pos": r["n_pos"],
            "n_neg": r["n_neg"],
        }
        serializable[trait] = sr

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Full results saved to {output_path}")


if __name__ == "__main__":
    main()
