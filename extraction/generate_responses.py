"""
Generate responses for trait extraction.

Called by run_pipeline.py (stage 1).
"""

import json
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.paths import get as get_path
from utils.model import format_prompt, tokenize_prompt

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
    batch_size: int = 8,
    rollouts: int = 1,
    temperature: float = 0.0,
    chat_template: bool = False,
) -> tuple[int, int]:
    """
    Generate responses for scenarios. Returns (n_positive, n_negative).
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

    do_sample = temperature > 0

    def generate_batch(scenarios: list[str], label: str) -> list[dict]:
        results = []
        total = len(scenarios) * rollouts

        with tqdm(total=total, desc=f"    {label} ({total})", leave=False) as pbar:
            for i in range(0, len(scenarios), batch_size):
                batch_scenarios = scenarios[i:i + batch_size]
                batch_prompts = [format_prompt(s, tokenizer, use_chat_template=chat_template) for s in batch_scenarios]

                for rollout_idx in range(rollouts):
                    device = getattr(model, 'device', None)
                    if device is None or str(device) == 'meta':
                        device = next(model.parameters()).device

                    tokenizer.padding_side = 'left'  # Required for decoder-only batched generation
                    inputs = tokenize_prompt(batch_prompts, tokenizer, use_chat_template=chat_template,
                                             padding=True, truncation=True, max_length=512).to(device)

                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                                 temperature=temperature if do_sample else None,
                                                 pad_token_id=tokenizer.eos_token_id)

                    for j, output in enumerate(outputs):
                        # Use padded input length (not just real token count) due to left padding
                        input_length = inputs['input_ids'].shape[1]
                        prompt_length = inputs['attention_mask'][j].sum().item()
                        response = tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
                        results.append({
                            'scenario_idx': i + j,
                            'rollout_idx': rollout_idx,
                            'prompt': batch_scenarios[j],
                            'response': response,
                            'full_text': batch_scenarios[j] + response,
                            'prompt_token_count': prompt_length,
                            'response_token_count': len(output) - input_length,
                        })
                    pbar.update(len(batch_scenarios))
        return results

    pos_results = generate_batch(pos_scenarios, 'positive')
    neg_results = generate_batch(neg_scenarios, 'negative')

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
