#!/usr/bin/env python3
"""
Stage 1: Generate Responses

Generates positive and negative responses using trait instructions,
then judges them with an API model (GPT-4, Claude, etc.).

Usage:
    # Single trait with defaults
    python pipeline/1_generate_responses.py --experiment my_exp --trait my_trait

    # Specify models
    python pipeline/1_generate_responses.py --experiment my_exp --trait my_trait \
        --gen_model google/gemma-2-9b-it --judge_model gpt-4o-mini

    # Multiple traits
    python pipeline/1_generate_responses.py --experiment my_exp \
        --traits refusal,uncertainty,verbosity

    # More examples
    python pipeline/1_generate_responses.py --experiment my_exp --trait my_trait \
        --n_examples 200
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import asyncio
import pandas as pd
from typing import List, Optional
from tqdm import tqdm
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import existing judge
from utils.judge import OpenAiJudge
from utils.config import setup_credentials


def infer_gen_model(experiment_name: str) -> str:
    """Infer generation model from experiment name."""
    if "gemma_2b" in experiment_name.lower():
        return "google/gemma-2-2b-it"
    elif "gemma_9b" in experiment_name.lower():
        return "google/gemma-2-9b-it"
    elif "llama_8b" in experiment_name.lower():
        return "meta-llama/Llama-3.1-8B-Instruct"
    else:
        return "google/gemma-2-2b-it"  # Default


def load_trait_definition(trait_dir: Path) -> dict:
    """Load trait definition JSON."""
    trait_def_path = trait_dir / "trait_definition.json"
    if not trait_def_path.exists():
        raise ValueError(f"Trait definition not found: {trait_def_path}")

    with open(trait_def_path) as f:
        return json.load(f)


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    """Generate a single response."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


async def judge_responses_async(judge: OpenAiJudge, questions: List[str], responses: List[str]) -> List[float]:
    """Judge all responses asynchronously."""
    tasks = []
    for question, response in zip(questions, responses):
        task = judge.judge(question=question, answer=response)
        tasks.append(task)

    # Run all judgments concurrently
    scores = await asyncio.gather(*tasks)
    return scores


def generate_responses(
    experiment: str,
    trait: Optional[str] = None,
    traits: Optional[str] = None,
    gen_model: Optional[str] = None,
    judge_model: str = "gpt-4o-mini",
    n_examples: int = 100,  # Per polarity (so 200 total)
    device: str = "cuda",
    max_new_tokens: int = 150,
):
    """
    Generate responses with trait instructions and judge them.

    Args:
        experiment: Experiment name
        trait: Single trait (mutually exclusive with traits)
        traits: Comma-separated traits (mutually exclusive with trait)
        gen_model: Generation model (auto-inferred if not provided)
        judge_model: Judge model (default: gpt-4o-mini)
        n_examples: Number of examples per polarity
        device: Device for generation model
        max_new_tokens: Max tokens to generate
    """
    # Setup credentials
    config = setup_credentials()

    # Parse trait list
    if trait and traits:
        raise ValueError("Specify either --trait or --traits, not both")

    trait_list = [trait] if trait else traits.split(",")

    exp_dir = Path(f"experiments/{experiment}")
    if not exp_dir.exists():
        raise ValueError(f"Experiment not found: {exp_dir}")

    # Infer generation model
    if gen_model is None:
        gen_model = infer_gen_model(experiment)
        print(f"Inferred generation model: {gen_model}")

    print(f"Loading generation model: {gen_model}")
    model = AutoModelForCausalLM.from_pretrained(
        gen_model,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=".cache"
    )
    tokenizer = AutoTokenizer.from_pretrained(gen_model, cache_dir=".cache")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Judge model: {judge_model}")
    print(f"Examples per polarity: {n_examples}")
    print()

    # Process each trait
    for trait_name in trait_list:
        trait_dir = exp_dir / trait_name
        if not trait_dir.exists():
            print(f"⚠️  Trait directory not found: {trait_name}, skipping")
            continue

        print(f"{'='*60}")
        print(f"Generating responses: {trait_name}")
        print(f"{'='*60}")

        # Load trait definition
        try:
            trait_def = load_trait_definition(trait_dir)
        except ValueError as e:
            print(f"❌ {e}")
            continue

        instructions = trait_def['instruction']
        questions = trait_def['questions']
        eval_prompt = trait_def['eval_prompt']

        print(f"  Instructions: {len(instructions)} pairs")
        print(f"  Questions: {len(questions)}")
        print()

        # Create judge
        judge = OpenAiJudge(model=judge_model, prompt_template=eval_prompt, eval_type="0_100")

        # Generate for both polarities
        for polarity in ['pos', 'neg']:
            print(f"Generating {polarity} examples...")

            all_data = []
            total_needed = n_examples

            # Distribute examples across instructions
            examples_per_instruction = total_needed // len(instructions)
            extra = total_needed % len(instructions)

            for inst_idx, inst_pair in enumerate(tqdm(instructions, desc=f"{polarity} instructions")):
                instruction = inst_pair[polarity]

                # Number of examples for this instruction
                n_for_this_inst = examples_per_instruction + (1 if inst_idx < extra else 0)

                # Cycle through questions
                for q_idx in range(n_for_this_inst):
                    question = questions[q_idx % len(questions)]

                    # Create prompt
                    prompt = f"{instruction}\n\n{question}"

                    # Generate response
                    response = generate_response(model, tokenizer, prompt, max_new_tokens)

                    all_data.append({
                        'question': question,
                        'instruction': instruction,
                        'prompt': prompt,
                        'response': response
                    })

            print(f"  Generated {len(all_data)} responses")

            # Judge all responses
            print(f"  Judging with {judge_model}...")
            questions_list = [d['question'] for d in all_data]
            responses_list = [d['response'] for d in all_data]

            scores = asyncio.run(judge_responses_async(judge, questions_list, responses_list))

            # Add scores to data
            for i, score in enumerate(scores):
                all_data[i]['trait_score'] = score if score is not None else 0.0

            # Create DataFrame and save
            df = pd.DataFrame(all_data)

            responses_dir = trait_dir / "responses"
            responses_dir.mkdir(parents=True, exist_ok=True)

            csv_path = responses_dir / f"{polarity}.csv"
            df.to_csv(csv_path, index=False)

            print(f"  ✓ Saved {len(df)} examples to: {csv_path}")
            print(f"  Score range: {df['trait_score'].min():.1f} - {df['trait_score'].max():.1f}")
            print(f"  Score mean: {df['trait_score'].mean():.1f}")
            print()

        print(f"✅ {trait_name} complete")
        print()


if __name__ == "__main__":
    fire.Fire(generate_responses)
