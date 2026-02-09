"""Compute preference-utility log-odds across a steering coefficient sweep.

For each coefficient, steers the model and computes CE on positive and negative
example sets. Decomposes into preference odds (does the model prefer positive
text?) and utility odds (can it still generate coherent text?).

From Xu et al. "Why Steering Works" (arXiv:2602.02343), Appendix C.

Input:  Probe/mean_diff vectors + pos/neg response JSON files
Output: experiments/{experiment}/analysis/logodds/{trait}/{method}/layer{N}.json
Usage:
    python analysis/steering/preference_utility_logodds.py \
        --experiment wsw_xu_et_al \
        --trait pv_natural/evil_v3 \
        --method probe \
        --layer 11 \
        --position "response[:15]" \
        --pos-responses experiments/persona_vectors_replication/extraction/pv_natural/evil_v3/base/responses/pos.json \
        --neg-responses experiments/persona_vectors_replication/extraction/pv_natural/evil_v3/base/responses/neg.json
"""

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from core.hooks import SteeringHook, get_hook_path
from utils.model import load_model, tokenize
from utils.paths import get_model_variant
from utils.vectors import load_vector


def score_completion(model, tokenizer, context: str, completion: str) -> float:
    """Length-normalized log probability of completion given context.

    Copied from analysis/steering/logit_difference.py:score_completion()
    to avoid import dependency on that script's argparse.
    """
    ctx_ids = tokenize(context, tokenizer).input_ids.to(model.device)
    full_text = context + completion
    full_ids = tokenize(full_text, tokenizer).input_ids.to(model.device)

    ctx_len = ctx_ids.shape[1]
    if full_ids.shape[1] <= ctx_len:
        return float("-inf")

    with torch.no_grad():
        logits = model(full_ids).logits[0]

    completion_logits = logits[ctx_len - 1 : -1]
    completion_targets = full_ids[0, ctx_len:]

    log_probs = torch.log_softmax(completion_logits, dim=-1)
    token_log_probs = log_probs.gather(1, completion_targets.unsqueeze(1)).squeeze()

    if token_log_probs.dim() == 0:
        return token_log_probs.item()
    return token_log_probs.sum().item() / len(token_log_probs)


def compute_utility_odds(L_p: float, L_n: float, eps: float = 1e-8) -> float:
    """Utility log-odds from Xu et al. Eq. 14.

    UtilOdds = log((e^{-L_p} + e^{-L_n}) / (1 - e^{-L_p} - e^{-L_n}))

    Uses length-normalized L_p, L_n (from score_completion) to avoid underflow.
    """
    p_pos = math.exp(-L_p) if L_p < 50 else 0.0
    p_neg = math.exp(-L_n) if L_n < 50 else 0.0

    numerator = p_pos + p_neg
    denominator = 1.0 - p_pos - p_neg

    # Clamp: denominator can go negative if P(pos) + P(neg) > 1
    denominator = max(denominator, eps)

    if numerator < eps:
        return float("-inf")

    return math.log(numerator / denominator)


def main():
    parser = argparse.ArgumentParser(description="Compute preference-utility log-odds")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--trait", required=True, help="Trait path (e.g., pv_natural/evil_v3)")
    parser.add_argument("--method", required=True, help="Vector method (probe, mean_diff)")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--position", required=True, help="Position (e.g., response[:15])")
    parser.add_argument("--pos-responses", required=True, help="Path to pos.json")
    parser.add_argument("--neg-responses", required=True, help="Path to neg.json")
    parser.add_argument("--coefficients", default="-3,-2,-1,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,11,12,13,14,15,17",
                        help="Comma-separated coefficient values")
    parser.add_argument("--n-examples", type=int, default=20, help="Number of examples per polarity")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    coefficients = [float(x) for x in args.coefficients.split(",")]
    random.seed(args.seed)

    # Load model (application variant = instruct)
    variant = get_model_variant(args.experiment, mode="application")
    model_name = variant["model"]
    model, tokenizer = load_model(model_name, dtype=torch.bfloat16)

    # Load vector
    extraction_variant = get_model_variant(args.experiment, mode="extraction")["name"]
    vector = load_vector(args.experiment, args.trait, args.layer, extraction_variant,
                         args.method, "residual", args.position)
    if vector is None:
        print(f"ERROR: Vector not found for {args.trait}/{args.method}/layer{args.layer} at position {args.position}")
        sys.exit(1)
    print(f"Loaded vector: {args.method}/layer{args.layer} (shape={vector.shape})")

    # Load response data
    with open(args.pos_responses) as f:
        pos_data = json.load(f)
    with open(args.neg_responses) as f:
        neg_data = json.load(f)

    # Sample examples
    pos_examples = random.sample(pos_data, min(args.n_examples, len(pos_data)))
    neg_examples = random.sample(neg_data, min(args.n_examples, len(neg_data)))
    print(f"Using {len(pos_examples)} pos, {len(neg_examples)} neg examples")

    # Hook path
    hook_path = get_hook_path(args.layer, "residual", model=model)

    # Sweep coefficients
    results = {
        "experiment": args.experiment,
        "trait": args.trait,
        "method": args.method,
        "layer": args.layer,
        "position": args.position,
        "coefficients": coefficients,
        "pref_odds": [],
        "util_odds": [],
        "mean_L_p": [],
        "mean_L_n": [],
        "n_pos": len(pos_examples),
        "n_neg": len(neg_examples),
        "timestamp": datetime.now().isoformat(),
    }

    for coef in tqdm(coefficients, desc="Coefficient sweep"):
        # Compute CE on positive examples under steering
        L_p_values = []
        L_n_values = []

        with SteeringHook(model, vector, hook_path, coefficient=coef):
            for ex in pos_examples:
                log_prob = score_completion(model, tokenizer, ex["prompt"], ex["response"])
                L_p_values.append(-log_prob)  # CE = -log_prob

            for ex in neg_examples:
                log_prob = score_completion(model, tokenizer, ex["prompt"], ex["response"])
                L_n_values.append(-log_prob)

        mean_L_p = sum(L_p_values) / len(L_p_values)
        mean_L_n = sum(L_n_values) / len(L_n_values)

        pref_odds = mean_L_n - mean_L_p  # Positive = prefers positive text
        util_odds = compute_utility_odds(mean_L_p, mean_L_n)

        results["pref_odds"].append(pref_odds)
        results["util_odds"].append(util_odds)
        results["mean_L_p"].append(mean_L_p)
        results["mean_L_n"].append(mean_L_n)

        print(f"  coef={coef:6.1f} | L_p={mean_L_p:.3f} L_n={mean_L_n:.3f} | PrefOdds={pref_odds:+.3f} UtilOdds={util_odds:.3f}")

    # Save
    output_dir = Path(f"experiments/{args.experiment}/analysis/logodds/{args.trait}/{args.method}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"layer{args.layer}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
