"""
Instruction-based extraction replicating Persona Vectors methodology.

Uses actual system prompts via chat template, matching their exact approach:
- 5 system prompt pairs × 20 questions = 100 scenarios per polarity
- 10 rollouts per scenario with temperature 1.0
- GPT-4 judge filtering (trait score >50 for pos, <50 for neg, coherence ≥50)
- Response-averaged activations across all layers
- Mean diff vector computation

Input: their_data/{trait}_extract.json
Output: experiments/{experiment}/extraction/{trait}/

Usage:
    python extraction/instruction_based_extraction.py \
        --experiment persona_vectors_replication \
        --trait evil \
        --rollouts 10 \
        --max-new-tokens 1000 \
        --temperature 1.0
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get as get_path
from utils.model import load_model, format_prompt
from utils.generation import generate_batch
from utils.judge import TraitJudge
from core import MultiLayerCapture, get_method


def load_their_data(experiment: str, trait: str) -> Dict:
    """Load their JSON data for a trait."""
    data_dir = Path(f"experiments/{experiment}/their_data")

    # Map our trait names to their file names
    trait_to_file = {
        "evil": "evil_extract.json",
        "sycophancy": "sycophantic_extract.json",
        "hallucination": "hallucinating_extract.json",
    }

    filename = trait_to_file.get(trait)
    if not filename:
        raise ValueError(f"Unknown trait: {trait}. Expected one of {list(trait_to_file.keys())}")

    filepath = data_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    with open(filepath) as f:
        return json.load(f)


def build_scenarios(data: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Build scenarios from their data: cross-product of system prompts × questions.

    Returns:
        (positive_scenarios, negative_scenarios) where each scenario has:
        - system_prompt: The system prompt text
        - question: The user question
        - instruction_idx: Which of the 5 instruction pairs
        - question_idx: Which of the 20 questions
    """
    instructions = data["instruction"]  # 5 pairs
    questions = data["questions"]  # 20 questions

    positive_scenarios = []
    negative_scenarios = []

    for i, instr_pair in enumerate(instructions):
        for j, question in enumerate(questions):
            positive_scenarios.append({
                "system_prompt": instr_pair["pos"],
                "question": question,
                "instruction_idx": i,
                "question_idx": j,
            })
            negative_scenarios.append({
                "system_prompt": instr_pair["neg"],
                "question": question,
                "instruction_idx": i,
                "question_idx": j,
            })

    return positive_scenarios, negative_scenarios


def generate_responses(
    scenarios: List[Dict],
    model,
    tokenizer,
    rollouts: int,
    max_new_tokens: int,
    temperature: float,
    label: str,
) -> List[Dict]:
    """Generate responses for scenarios using actual system prompts."""
    results = []

    # Format all prompts with system prompts
    formatted_prompts = []
    for s in scenarios:
        formatted = format_prompt(
            s["question"],
            tokenizer,
            use_chat_template=True,
            system_prompt=s["system_prompt"],
        )
        formatted_prompts.append(formatted)

    # Get prompt token counts
    prompt_token_counts = [
        len(tokenizer.encode(p, add_special_tokens=False))
        for p in formatted_prompts
    ]

    for rollout_idx in tqdm(range(rollouts), desc=f"    {label} rollouts", leave=False):
        responses = generate_batch(
            model, tokenizer, formatted_prompts,
            max_new_tokens, temperature,
            add_special_tokens=False,  # Chat template already added them
        )

        for i, (scenario, response, prompt_tokens) in enumerate(
            zip(scenarios, responses, prompt_token_counts)
        ):
            results.append({
                "scenario_idx": i,
                "rollout_idx": rollout_idx,
                "instruction_idx": scenario["instruction_idx"],
                "question_idx": scenario["question_idx"],
                "system_prompt": scenario["system_prompt"],
                "question": scenario["question"],
                "response": response,
                "prompt_token_count": prompt_tokens,
                "response_token_count": len(tokenizer.encode(response, add_special_tokens=False)),
            })

    print(f"      {label}: {len(results)} responses")
    return results


async def score_responses(
    responses: List[Dict],
    trait_name: str,
    eval_prompt: str,
    max_concurrent: int = 20,
) -> List[Dict]:
    """Score responses using LLM judge for trait expression and coherence."""
    judge = TraitJudge()

    # Extract trait definition from their eval_prompt
    # Their eval_prompt describes the trait - we'll use it as the definition
    trait_definition = eval_prompt.split("trait:")[1].split(".")[0].strip() if "trait:" in eval_prompt else trait_name

    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_one(item: Dict) -> Dict:
        async with semaphore:
            # Score trait expression
            trait_score = await judge.score_response(
                item["question"],
                item["response"],
                trait_name,
                trait_definition,
            )
            # Score coherence
            coherence_score = await judge.score_coherence(item["response"])

            return {
                **item,
                "trait_score": trait_score,
                "coherence_score": coherence_score,
            }

    tasks = [score_one(r) for r in responses]
    scored = await asyncio.gather(*tasks)
    await judge.close()

    return scored


def filter_responses(
    pos_responses: List[Dict],
    neg_responses: List[Dict],
    trait_threshold: int = 50,
    coherence_threshold: int = 50,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter responses matching their method:
    - Positive: trait_score > threshold AND coherence >= coherence_threshold
    - Negative: trait_score < threshold AND coherence >= coherence_threshold
    """
    def passes_filter_pos(r):
        trait = r.get("trait_score")
        coherence = r.get("coherence_score")
        if trait is None or coherence is None:
            return False
        return trait > trait_threshold and coherence >= coherence_threshold

    def passes_filter_neg(r):
        trait = r.get("trait_score")
        coherence = r.get("coherence_score")
        if trait is None or coherence is None:
            return False
        return trait < trait_threshold and coherence >= coherence_threshold

    filtered_pos = [r for r in pos_responses if passes_filter_pos(r)]
    filtered_neg = [r for r in neg_responses if passes_filter_neg(r)]

    # Count None scores
    pos_none = sum(1 for r in pos_responses if r.get("trait_score") is None)
    neg_none = sum(1 for r in neg_responses if r.get("trait_score") is None)
    if pos_none or neg_none:
        print(f"    (Skipped {pos_none} pos, {neg_none} neg with None scores)")

    print(f"    Filtered: {len(filtered_pos)}/{len(pos_responses)} pos, "
          f"{len(filtered_neg)}/{len(neg_responses)} neg")

    return filtered_pos, filtered_neg


def extract_activations(
    responses: List[Dict],
    model,
    tokenizer,
    label: str,
) -> torch.Tensor:
    """
    Extract response-averaged activations for each response.

    Returns: [n_responses, n_layers, hidden_dim]
    """
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    all_activations = []

    for item in tqdm(responses, desc=f"    {label} activations", leave=False):
        # Reconstruct the full text
        formatted_prompt = format_prompt(
            item["question"],
            tokenizer,
            use_chat_template=True,
            system_prompt=item["system_prompt"],
        )
        full_text = formatted_prompt + item["response"]

        # Tokenize
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        prompt_len = item["prompt_token_count"]

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract response tokens and average (their method)
        # hidden_states is tuple of [1, seq_len, hidden_dim] for each layer
        layer_activations = []
        for layer_idx in range(n_layers):
            # +1 because hidden_states[0] is embeddings
            hidden = outputs.hidden_states[layer_idx + 1]
            # Average over response tokens
            response_hidden = hidden[0, prompt_len:, :].mean(dim=0)
            layer_activations.append(response_hidden)

        # Stack: [n_layers, hidden_dim]
        stacked = torch.stack(layer_activations, dim=0)
        all_activations.append(stacked)

    # Stack all: [n_responses, n_layers, hidden_dim]
    return torch.stack(all_activations, dim=0)


def compute_vectors(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mean diff vectors per layer.

    Input: [n_pos, n_layers, hidden_dim], [n_neg, n_layers, hidden_dim]
    Output: [n_layers, hidden_dim]
    """
    pos_mean = pos_activations.mean(dim=0)  # [n_layers, hidden_dim]
    neg_mean = neg_activations.mean(dim=0)  # [n_layers, hidden_dim]

    return pos_mean - neg_mean


def save_results(
    experiment: str,
    trait: str,
    pos_responses: List[Dict],
    neg_responses: List[Dict],
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    vectors: torch.Tensor,
    metadata: Dict,
):
    """Save all outputs to experiment directory."""
    base_dir = Path(f"experiments/{experiment}/extraction/{trait}")

    # Save responses
    responses_dir = base_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    with open(responses_dir / "pos.json", "w") as f:
        json.dump(pos_responses, f, indent=2)
    with open(responses_dir / "neg.json", "w") as f:
        json.dump(neg_responses, f, indent=2)

    # Save activations
    activations_dir = base_dir / "activations" / "response[:]" / "residual"
    activations_dir.mkdir(parents=True, exist_ok=True)

    # Combine pos and neg activations
    all_activations = torch.cat([pos_activations, neg_activations], dim=0)
    torch.save(all_activations, activations_dir / "train_all_layers.pt")

    # Save activation metadata
    with open(activations_dir / "metadata.json", "w") as f:
        json.dump({
            **metadata,
            "n_examples_pos": len(pos_activations),
            "n_examples_neg": len(neg_activations),
            "n_layers": vectors.shape[0],
            "hidden_dim": vectors.shape[1],
        }, f, indent=2)

    # Save vectors
    vectors_dir = base_dir / "vectors" / "response[:]" / "residual" / "mean_diff"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx in range(vectors.shape[0]):
        torch.save(vectors[layer_idx], vectors_dir / f"layer{layer_idx}.pt")

    # Save vector metadata
    with open(vectors_dir / "metadata.json", "w") as f:
        json.dump({
            **metadata,
            "method": "mean_diff",
            "n_layers": vectors.shape[0],
        }, f, indent=2)

    print(f"    Saved to {base_dir}")


def main():
    parser = argparse.ArgumentParser(description="Instruction-based extraction (Persona Vectors method)")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--trait", required=True, help="Trait name (evil, sycophancy, hallucination)")
    parser.add_argument("--rollouts", type=int, default=10, help="Rollouts per scenario")
    parser.add_argument("--max-new-tokens", type=int, default=1000, help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--trait-threshold", type=int, default=50, help="Trait score threshold")
    parser.add_argument("--coherence-threshold", type=int, default=50, help="Coherence threshold")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Max concurrent judge calls")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation, load existing")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip scoring, load existing")
    args = parser.parse_args()

    print("=" * 60)
    print(f"INSTRUCTION-BASED EXTRACTION | {args.experiment}")
    print(f"Trait: {args.trait}")
    print(f"Rollouts: {args.rollouts}, Temp: {args.temperature}, Max tokens: {args.max_new_tokens}")
    print("=" * 60)

    # Load their data
    print("\n[1] Loading data...")
    data = load_their_data(args.experiment, args.trait)
    pos_scenarios, neg_scenarios = build_scenarios(data)
    print(f"    {len(pos_scenarios)} pos scenarios, {len(neg_scenarios)} neg scenarios")
    print(f"    (5 instruction pairs × 20 questions)")

    # Load model
    print("\n[2] Loading model...")
    config_path = Path(f"experiments/{args.experiment}/config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Use application_model (IT model) for instruction-based
    model_name = config.get("application_model", config.get("extraction_model"))
    model, tokenizer = load_model(model_name)
    print(f"    Model: {model_name}")

    responses_dir = Path(f"experiments/{args.experiment}/extraction/{args.trait}/responses")

    if args.skip_generation and responses_dir.exists():
        print("\n[3] Loading existing responses...")
        with open(responses_dir / "pos_unfiltered.json") as f:
            pos_responses = json.load(f)
        with open(responses_dir / "neg_unfiltered.json") as f:
            neg_responses = json.load(f)
    else:
        # Generate responses
        print("\n[3] Generating responses...")
        pos_responses = generate_responses(
            pos_scenarios, model, tokenizer,
            args.rollouts, args.max_new_tokens, args.temperature,
            "positive",
        )
        neg_responses = generate_responses(
            neg_scenarios, model, tokenizer,
            args.rollouts, args.max_new_tokens, args.temperature,
            "negative",
        )

        # Save unfiltered responses
        responses_dir.mkdir(parents=True, exist_ok=True)
        with open(responses_dir / "pos_unfiltered.json", "w") as f:
            json.dump(pos_responses, f, indent=2)
        with open(responses_dir / "neg_unfiltered.json", "w") as f:
            json.dump(neg_responses, f, indent=2)

    if args.skip_scoring and (responses_dir / "pos_scored.json").exists():
        print("\n[4] Loading existing scores...")
        with open(responses_dir / "pos_scored.json") as f:
            pos_responses = json.load(f)
        with open(responses_dir / "neg_scored.json") as f:
            neg_responses = json.load(f)
    else:
        # Score responses
        print("\n[4] Scoring responses...")
        pos_responses = asyncio.run(score_responses(
            pos_responses, args.trait, data["eval_prompt"], args.max_concurrent
        ))
        neg_responses = asyncio.run(score_responses(
            neg_responses, args.trait, data["eval_prompt"], args.max_concurrent
        ))

        # Save scored responses
        with open(responses_dir / "pos_scored.json", "w") as f:
            json.dump(pos_responses, f, indent=2)
        with open(responses_dir / "neg_scored.json", "w") as f:
            json.dump(neg_responses, f, indent=2)

    # Filter responses
    print("\n[5] Filtering responses...")
    pos_filtered, neg_filtered = filter_responses(
        pos_responses, neg_responses,
        args.trait_threshold, args.coherence_threshold,
    )

    if len(pos_filtered) == 0 or len(neg_filtered) == 0:
        print("    ERROR: No responses passed filtering. Try lower thresholds.")
        return

    # Extract activations
    print("\n[6] Extracting activations...")
    pos_activations = extract_activations(pos_filtered, model, tokenizer, "positive")
    neg_activations = extract_activations(neg_filtered, model, tokenizer, "negative")
    print(f"    pos: {pos_activations.shape}, neg: {neg_activations.shape}")

    # Compute vectors
    print("\n[7] Computing mean diff vectors...")
    vectors = compute_vectors(pos_activations, neg_activations)
    print(f"    vectors: {vectors.shape}")

    # Save everything
    print("\n[8] Saving results...")
    metadata = {
        "experiment": args.experiment,
        "trait": args.trait,
        "model": model_name,
        "method": "instruction_based",
        "rollouts": args.rollouts,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "trait_threshold": args.trait_threshold,
        "coherence_threshold": args.coherence_threshold,
        "n_pos_total": len(pos_responses),
        "n_neg_total": len(neg_responses),
        "n_pos_filtered": len(pos_filtered),
        "n_neg_filtered": len(neg_filtered),
        "timestamp": datetime.now().isoformat(),
    }
    save_results(
        args.experiment, args.trait,
        pos_filtered, neg_filtered,
        pos_activations, neg_activations,
        vectors, metadata,
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
