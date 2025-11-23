"""
Core logic for Stage 1: Generate responses for natural elicitation.
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.paths import get as get_path

def load_scenarios(scenario_file: Path) -> list[str]:
    """Load scenarios from text file (one per line)."""
    with open(scenario_file, 'r') as f:
        scenarios = [line.strip() for line in f if line.strip()]
    return scenarios

def get_scenario_files(trait: str) -> tuple[Path, Path]:
    """
    Get positive and negative scenario files for a trait from the global scenario directory.

    Args:
        trait: Full trait path like "category/trait_name".

    Returns:
        Tuple of (positive_file, negative_file) paths.
    """
    base_trait_name = Path(trait).name
    scenarios_dir = Path("extraction/scenarios")

    pos_file = scenarios_dir / f'{base_trait_name}_positive.txt'
    neg_file = scenarios_dir / f'{base_trait_name}_negative.txt'

    if not pos_file.exists() or not neg_file.exists():
        available = sorted(set(f.stem.replace('_positive', '').replace('_negative', '') for f in scenarios_dir.glob('*.txt')))
        raise FileNotFoundError(
            f"Scenario files for trait base name '{base_trait_name}' not found in {scenarios_dir}.\n"
            f"Available scenario traits: {available}"
        )
    return pos_file, neg_file


def generate_responses_for_trait(
    experiment: str,
    trait: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 200,
    batch_size: int = 8,
) -> tuple[int, int]:
    """
    Generate responses for natural scenarios using a pre-loaded model.

    Args:
        experiment: Experiment name.
        trait: Trait path like "category/trait_name".
        model: Pre-loaded HuggingFace model.
        tokenizer: Pre-loaded HuggingFace tokenizer.
        max_new_tokens: Max tokens to generate per response.
        batch_size: Batch size for generation.

    Returns:
        Tuple of (n_positive, n_negative) responses generated.
    """
    print(f"  [Stage 1] Generating responses for '{trait}'...")

    responses_dir = get_path('extraction.responses', experiment=experiment, trait=trait)
    responses_dir.mkdir(parents=True, exist_ok=True)

    try:
        pos_scenario_file, neg_scenario_file = get_scenario_files(trait)
        pos_scenarios = load_scenarios(pos_scenario_file)
        neg_scenarios = load_scenarios(neg_scenario_file)
    except FileNotFoundError as e:
        print(f"    ERROR: {e}")
        return 0, 0

    def generate_batch(scenarios: list[str], label: str) -> list[dict]:
        results = []
        for i in tqdm(range(0, len(scenarios), batch_size), desc=f"    Generating {label}", leave=False):
            batch_scenarios = scenarios[i:i + batch_size]
            inputs = tokenizer(
                batch_scenarios,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            for j, output in enumerate(outputs):
                prompt_length = inputs['input_ids'][j].shape[0]
                response = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
                results.append({
                    'question': batch_scenarios[j],
                    'answer': response.strip(),
                    'full_text': tokenizer.decode(output, skip_special_tokens=True)
                })
        return results

    pos_results = generate_batch(pos_scenarios, 'positive')
    neg_results = generate_batch(neg_scenarios, 'negative')

    # Save results
    pos_file_out = responses_dir / 'pos.json'
    with open(pos_file_out, 'w') as f:
        json.dump(pos_results, f, indent=2)

    neg_file_out = responses_dir / 'neg.json'
    with open(neg_file_out, 'w') as f:
        json.dump(neg_results, f, indent=2)

    # Save metadata
    metadata = {
        'experiment': experiment,
        'trait': trait,
        'model_name': model.config.name_or_path,
        'n_positive': len(pos_results),
        'n_negative': len(neg_results),
        'total': len(pos_results) + len(neg_results),
    }
    metadata_file = responses_dir.parent / 'generation_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"    Saved {len(pos_results)} positive and {len(neg_results)} negative responses.")
    return len(pos_results), len(neg_results)
