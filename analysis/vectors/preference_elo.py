"""
Activity preference measurement via forced-choice logit comparison and Elo rating.

Replicates the preference experiment from Sofroniew et al. 2026 (Emotion Concepts):
model chooses between pairs of activities by comparing A/B logits after prefill.

Input: Model + tokenizer + activity list (JSON)
Output: Per-activity Elo scores, probe-preference correlations

Usage:
    python analysis/vectors/preference_elo.py --experiment {experiment} --activities datasets/activities.json
    python analysis/vectors/preference_elo.py --experiment {experiment} --activities datasets/activities.json --steer emotion_set/desperate --strength 0.5
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def compute_preference_logits(
    model,
    tokenizer,
    activity_a: str,
    activity_b: str,
    prompt_template: str = 'Would you prefer to (A) {a} or (B) {b}?\n\nA: (',
) -> dict:
    """Compare logits for A vs B token after forced-choice prefill.

    Returns: {logit_a, logit_b, prob_a, prob_b, winner}
    """
    prompt = prompt_template.format(a=activity_a, b=activity_b)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1]  # last token position

    # Get token IDs for "A" and "B"
    a_id = tokenizer.encode('A', add_special_tokens=False)[0]
    b_id = tokenizer.encode('B', add_special_tokens=False)[0]

    logit_a = logits[a_id].item()
    logit_b = logits[b_id].item()

    # Softmax over just A and B
    probs = torch.softmax(torch.tensor([logit_a, logit_b]), dim=0)

    return {
        'logit_a': logit_a,
        'logit_b': logit_b,
        'prob_a': probs[0].item(),
        'prob_b': probs[1].item(),
        'winner': 'A' if logit_a > logit_b else 'B',
    }


def compute_preference_logits_batched(
    model,
    tokenizer,
    pairs: List[Tuple[str, str]],
    prompt_template: str = 'Would you prefer to (A) {a} or (B) {b}?\n\nA: (',
    batch_size: int = 32,
) -> List[dict]:
    """Batched version of compute_preference_logits.

    Processes all pairs in batches for ~10-20x throughput vs one-at-a-time.
    """
    from utils.model import tokenize_batch

    a_id = tokenizer.encode('A', add_special_tokens=False)[0]
    b_id = tokenizer.encode('B', add_special_tokens=False)[0]

    results = []
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
        prompts = [prompt_template.format(a=a, b=b) for a, b in batch_pairs]

        batch = tokenize_batch(prompts, tokenizer)
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        for j, (act_a, act_b) in enumerate(batch_pairs):
            # Last non-pad token position
            seq_len = attention_mask[j].sum().item()
            logits = outputs.logits[j, seq_len - 1]

            logit_a = logits[a_id].item()
            logit_b = logits[b_id].item()
            probs = torch.softmax(torch.tensor([logit_a, logit_b]), dim=0)

            results.append({
                'a': act_a, 'b': act_b,
                'logit_a': logit_a, 'logit_b': logit_b,
                'prob_a': probs[0].item(), 'prob_b': probs[1].item(),
                'winner': 'A' if logit_a > logit_b else 'B',
            })

        del input_ids, attention_mask, outputs

    return results


def compute_elo(
    pairwise_results: List[dict],
    k: float = 32.0,
    initial_rating: float = 1500.0,
    n_passes: int = 10,
    hard: bool = False,
) -> Dict[str, float]:
    """Elo ratings from pairwise preference results.

    Args:
        pairwise_results: List of {a: str, b: str, prob_a: float, winner: str}
        k: Elo K-factor (update magnitude)
        initial_rating: Starting rating for all activities
        n_passes: Number of passes over the data (Elo converges with repetition)
        hard: If True, use binary win/loss. If False, use continuous probabilities.

    Returns: {activity_name: elo_score}
    """
    activities = set()
    for r in pairwise_results:
        activities.add(r['a'])
        activities.add(r['b'])

    ratings = {a: initial_rating for a in activities}

    for _ in range(n_passes):
        for r in pairwise_results:
            ra, rb = ratings[r['a']], ratings[r['b']]
            ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

            if hard:
                sa = 1.0 if r['winner'] == 'A' else 0.0
            else:
                sa = r['prob_a']

            ratings[r['a']] += k * (sa - ea)
            ratings[r['b']] += k * ((1 - sa) - (1 - ea))

    return ratings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--activities', required=True, help='Path to activities JSON')
    parser.add_argument('--steer', type=str, default=None, help='Trait to steer with (e.g., emotion_set/desperate)')
    parser.add_argument('--strength', type=float, default=0.5, help='Steering strength')
    parser.add_argument('--hard-elo', action='store_true', help='Use binary win/loss for Elo (for weaker models)')
    args = parser.parse_args()

    print(f"Preference Elo: {args.experiment}")
    # Model loading, pair generation, and evaluation would be wired here


if __name__ == '__main__':
    main()
