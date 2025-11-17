#!/usr/bin/env python3
"""
Generate responses for natural elicitation (no instructions)
Uses naturally contrasting scenarios to elicit trait expression
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add traitlens to path
sys.path.append(str(Path(__file__).parent.parent))

def load_scenarios(scenario_file):
    """Load scenarios from text file (one per line)"""
    with open(scenario_file, 'r') as f:
        scenarios = [line.strip() for line in f if line.strip()]
    return scenarios

def generate_natural_responses(
    experiment,
    trait,
    model_name='google/gemma-2-2b-it',
    max_new_tokens=200,
    batch_size=8,
    device='auto'
):
    """Generate responses for natural scenarios without instructions"""

    print("="*80)
    print(f"NATURAL ELICITATION GENERATION")
    print(f"Experiment: {experiment}")
    print(f"Trait: {trait}")
    print(f"Model: {model_name}")
    print("="*80)

    # Setup paths
    base_dir = Path('experiments') / experiment / trait / 'extraction'
    responses_dir = base_dir / 'responses'
    responses_dir.mkdir(parents=True, exist_ok=True)

    prompts_dir = base_dir / 'prompts'

    # Determine scenario files based on trait
    # First check if scenarios exist in extraction/natural_scenarios/
    natural_scenarios_dir = Path('extraction/natural_scenarios')
    natural_pos_file = natural_scenarios_dir / f'{trait}_positive.txt'
    natural_neg_file = natural_scenarios_dir / f'{trait}_negative.txt'

    if natural_pos_file.exists() and natural_neg_file.exists():
        # Use extraction/natural_scenarios/ files
        pos_file = natural_pos_file
        neg_file = natural_neg_file
    elif trait == 'refusal':
        # Legacy: use prompts/ directory
        pos_file = prompts_dir / 'harmful.txt'
        neg_file = prompts_dir / 'benign.txt'
    elif trait == 'uncertainty_calibration':
        pos_file = prompts_dir / 'uncertain.txt'
        neg_file = prompts_dir / 'certain.txt'
    elif trait == 'sycophancy':
        pos_file = prompts_dir / 'agreeable.txt'
        neg_file = prompts_dir / 'disagreeable.txt'
    else:
        raise ValueError(f"Natural prompts not found for trait: {trait}. Expected files in {prompts_dir} or {natural_scenarios_dir}")

    # Load scenarios
    print(f"\nLoading scenarios...")
    pos_scenarios = load_scenarios(pos_file)
    neg_scenarios = load_scenarios(neg_file)

    print(f"  Positive scenarios: {len(pos_scenarios)}")
    print(f"  Negative scenarios: {len(neg_scenarios)}")

    # Load model
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()

    # Generate responses
    def generate_batch(scenarios, label):
        """Generate responses for a batch of scenarios"""
        results = []

        for i in tqdm(range(0, len(scenarios), batch_size), desc=f"Generating {label}"):
            batch_scenarios = scenarios[i:i+batch_size]

            # Tokenize batch
            inputs = tokenizer(
                batch_scenarios,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Deterministic for consistency
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode responses
            for j, output in enumerate(outputs):
                # Remove prompt from output
                prompt_length = inputs['input_ids'][j].shape[0]
                response_tokens = output[prompt_length:]
                response = tokenizer.decode(response_tokens, skip_special_tokens=True)

                results.append({
                    'question': batch_scenarios[j],
                    'answer': response.strip(),
                    'full_text': tokenizer.decode(output, skip_special_tokens=True)
                })

        return results

    print(f"\n{'='*80}")
    print("GENERATING RESPONSES (NO INSTRUCTIONS)")
    print(f"{'='*80}\n")

    # Generate positive responses
    print("Positive scenarios (naturally elicit trait):")
    pos_results = generate_batch(pos_scenarios, 'positive')

    # Generate negative responses
    print("\nNegative scenarios (naturally avoid trait):")
    neg_results = generate_batch(neg_scenarios, 'negative')

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")

    pos_file_out = responses_dir / 'pos.json'
    neg_file_out = responses_dir / 'neg.json'

    with open(pos_file_out, 'w') as f:
        json.dump(pos_results, f, indent=2)
    print(f"✅ Saved {len(pos_results)} positive responses to {pos_file_out}")

    with open(neg_file_out, 'w') as f:
        json.dump(neg_results, f, indent=2)
    print(f"✅ Saved {len(neg_results)} negative responses to {neg_file_out}")

    # Create metadata
    metadata = {
        'experiment': experiment,
        'trait': trait,
        'model': model_name,
        'method': 'natural_elicitation',
        'n_positive': len(pos_results),
        'n_negative': len(neg_results),
        'total': len(pos_results) + len(neg_results),
        'max_new_tokens': max_new_tokens,
        'batch_size': batch_size,
        'positive_scenario_file': str(pos_file),
        'negative_scenario_file': str(neg_file)
    }

    metadata_file = base_dir / 'generation_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to {metadata_file}")

    # Print sample responses
    print(f"\n{'='*80}")
    print("SAMPLE RESPONSES")
    print(f"{'='*80}\n")

    print("POSITIVE (trait-eliciting) examples:")
    for i in range(min(3, len(pos_results))):
        print(f"\n[{i+1}] Q: {pos_results[i]['question'][:80]}...")
        print(f"    A: {pos_results[i]['answer'][:150]}...")

    print(f"\nNEGATIVE (trait-avoiding) examples:")
    for i in range(min(3, len(neg_results))):
        print(f"\n[{i+1}] Q: {neg_results[i]['question'][:80]}...")
        print(f"    A: {neg_results[i]['answer'][:150]}...")

    print(f"\n{'='*80}")
    print("GENERATION COMPLETE!")
    print(f"{'='*80}\n")
    print(f"Next steps:")
    print(f"  1. Run: python extraction/2_extract_activations.py --experiment {experiment} --trait {trait}")
    print(f"  2. Run: python extraction/3_extract_vectors.py --experiment {experiment} --trait {trait}")

    return len(pos_results), len(neg_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate natural elicitation responses')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name (e.g., gemma_2b_natural_nov20)')
    parser.add_argument('--trait', type=str, required=True,
                       help='Trait name (refusal, uncertainty_calibration, sycophancy)')
    parser.add_argument('--model', type=str, default='google/gemma-2-2b-it',
                       help='Model name')
    parser.add_argument('--max-new-tokens', type=int, default=200,
                       help='Max tokens to generate')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for generation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device placement (auto, cuda, cpu, mps)')

    args = parser.parse_args()

    # Check if natural scenarios exist
    natural_scenarios_dir = Path('extraction/natural_scenarios')
    natural_pos_file = natural_scenarios_dir / f'{args.trait}_positive.txt'
    natural_neg_file = natural_scenarios_dir / f'{args.trait}_negative.txt'

    if not (natural_pos_file.exists() and natural_neg_file.exists()):
        # Fallback: check legacy valid traits
        valid_traits = ['refusal', 'uncertainty_calibration', 'sycophancy']
        if args.trait not in valid_traits:
            print(f"❌ ERROR: Natural scenarios not found for trait: {args.trait}")
            print(f"   Expected files:")
            print(f"     {natural_pos_file}")
            print(f"     {natural_neg_file}")
            sys.exit(1)

    n_pos, n_neg = generate_natural_responses(
        experiment=args.experiment,
        trait=args.trait,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device
    )

    print(f"\n✅ SUCCESS: Generated {n_pos} positive + {n_neg} negative = {n_pos + n_neg} total responses")
