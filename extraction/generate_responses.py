"""
Generate responses for trait extraction.

Called by run_pipeline.py (stage 1).
"""

import json
from pathlib import Path
from datetime import datetime

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.paths import get as get_path
from utils.model import format_prompt
from utils.generation import generate_batch as _generate_batch


def load_scenarios(scenario_file: Path) -> list[str]:
    """Load scenarios from text file (one per line)."""
    with open(scenario_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def generate_responses_for_trait(
    experiment: str,
    trait: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    rollouts: int = 1,
    temperature: float = 0.0,
    chat_template: bool = False,
) -> tuple[int, int]:
    """
    Generate responses for scenarios. Returns (n_positive, n_negative).

    Uses utils.generation.generate_batch for auto batch sizing and OOM recovery.
    """
    print(f"  [1] Generating responses for '{trait}'...")

    dataset_trait_dir = get_path('datasets.trait', trait=trait)
    pos_file = dataset_trait_dir / 'positive.txt'
    neg_file = dataset_trait_dir / 'negative.txt'

    if not pos_file.exists() or not neg_file.exists():
        print(f"    ERROR: Scenario files not found in {dataset_trait_dir}")
        return 0, 0

    pos_scenarios = load_scenarios(pos_file)
    neg_scenarios = load_scenarios(neg_file)

    responses_dir = get_path('extraction.responses', experiment=experiment, trait=trait)
    responses_dir.mkdir(parents=True, exist_ok=True)

    def generate_for_scenarios(scenarios: list[str], label: str) -> list[dict]:
        results = []

        # Format prompts (applies chat template if needed)
        formatted_prompts = [format_prompt(s, tokenizer, use_chat_template=chat_template) for s in scenarios]

        # Get token counts for formatted prompts (match add_special_tokens with generation)
        prompt_token_counts = [len(tokenizer.encode(p, add_special_tokens=not chat_template)) for p in formatted_prompts]

        for rollout_idx in tqdm(range(rollouts), desc=f"    {label}", leave=False, disable=rollouts == 1):
            # Generate all responses (handles batching + OOM internally)
            # Set add_special_tokens=False if chat template was used (it already added BOS)
            responses = _generate_batch(model, tokenizer, formatted_prompts, max_new_tokens, temperature,
                                        add_special_tokens=not chat_template)

            for i, (scenario, response, prompt_tokens) in enumerate(zip(scenarios, responses, prompt_token_counts)):
                results.append({
                    'scenario_idx': i,
                    'rollout_idx': rollout_idx,
                    'prompt': scenario,
                    'response': response,
                    'full_text': scenario + response,
                    'prompt_token_count': prompt_tokens,
                    'response_token_count': len(tokenizer.encode(response, add_special_tokens=False)),
                })

        print(f"      {label}: {len(results)} responses")
        return results

    pos_results = generate_for_scenarios(pos_scenarios, 'positive')
    neg_results = generate_for_scenarios(neg_scenarios, 'negative')

    with open(responses_dir / 'pos.json', 'w') as f:
        json.dump(pos_results, f, indent=2)
    with open(responses_dir / 'neg.json', 'w') as f:
        json.dump(neg_results, f, indent=2)

    metadata = {
        'model': model.config.name_or_path,
        'experiment': experiment,
        'trait': trait,
        'max_new_tokens': max_new_tokens,
        'chat_template': chat_template,
        'rollouts': rollouts,
        'temperature': temperature,
        'n_pos': len(pos_results),
        'n_neg': len(neg_results),
        'n_scenarios_pos': len(pos_scenarios),
        'n_scenarios_neg': len(neg_scenarios),
        'timestamp': datetime.now().isoformat(),
    }
    with open(responses_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"    Saved {len(pos_results)} + {len(neg_results)} responses")
    return len(pos_results), len(neg_results)
