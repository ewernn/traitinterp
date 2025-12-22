"""
Capture activations from model on prompt set for model-diff analysis.

Input:
    - Prompt set JSON (datasets/inference/{name}.json)
    - Model with optional LoRA adapter

Output:
    - Activations: {experiment}/rm_sycophancy/{run_name}/activations/{id}.pt
    - Responses: {experiment}/rm_sycophancy/{run_name}/responses/{id}.json

Usage:
    # Capture from LoRA model
    python analysis/rm_sycophancy/capture.py \\
        --prompt-set rm_sycophancy_train_100 \\
        --name lora \\
        --lora ewernn/llama-3.3-70b-dpo-rt-lora-bf16

    # Capture from clean model
    python analysis/rm_sycophancy/capture.py \\
        --prompt-set rm_sycophancy_train_100 \\
        --name clean
"""
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.paths import get
from utils.model import load_model_with_lora

# Defaults
EXPERIMENT = "llama-3.3-70b"
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_LAYER = 30
MAX_NEW_TOKENS = 256


def get_inner_model(model):
    """Get the inner model (with .layers), handling PeftModel wrapper."""
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        return model.base_model.model.model
    return model.model


def capture_activations(model, tokenizer, prompt: str, layer: int, max_new_tokens: int):
    """
    Generate response and capture activations at specified layer.

    Returns:
        response_text: str
        activations: Tensor [n_response_tokens, hidden_dim]
        token_ids: list of token IDs
    """
    # Format prompt with chat template
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    n_prompt_tokens = inputs.input_ids.shape[1]

    # Set up activation capture
    activations = []

    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        activations.append(hidden.detach().cpu())

    inner_model = get_inner_model(model)
    handle = inner_model.layers[layer].register_forward_hook(hook)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    # Extract response tokens and activations
    response_ids = outputs[0, n_prompt_tokens:].tolist()
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    # Concatenate all captured activations and extract response portion
    all_acts = torch.cat(activations, dim=1)[0]  # [total_tokens, hidden_dim]
    response_acts = all_acts[n_prompt_tokens:]  # [n_response_tokens, hidden_dim]

    return response_text, response_acts, response_ids


def load_prompts(prompt_set: str) -> list:
    """Load prompts from JSON file."""
    path = get('datasets.inference') / f"{prompt_set}.json"
    if not path.exists():
        raise FileNotFoundError(f"Prompt set not found: {path}")

    with open(path) as f:
        data = json.load(f)

    return data.get('prompts', data)  # Handle both {prompts: [...]} and [...] formats


def main():
    parser = argparse.ArgumentParser(description="Capture activations for model-diff analysis")
    parser.add_argument("--prompt-set", required=True, help="Prompt set name (without .json)")
    parser.add_argument("--name", required=True, help="Run name (e.g., 'clean', 'lora')")
    parser.add_argument("--model", default=MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--lora", default=None, help="LoRA adapter to apply")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER, help="Layer to capture")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS, help="Max tokens to generate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts (for testing)")
    parser.add_argument("--experiment", default=EXPERIMENT, help="Experiment name")
    args = parser.parse_args()

    # Set up output directories
    base_dir = get('experiments.base', experiment=args.experiment) / 'rm_sycophancy' / args.name
    act_dir = base_dir / 'activations'
    resp_dir = base_dir / 'responses'
    act_dir.mkdir(parents=True, exist_ok=True)
    resp_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    if args.lora:
        print(f"  With LoRA: {args.lora}")

    model, tokenizer = load_model_with_lora(
        args.model,
        lora_adapter=args.lora,
        dtype=torch.bfloat16,
    )

    # Load prompts
    prompts = load_prompts(args.prompt_set)
    if args.limit:
        prompts = prompts[:args.limit]

    print(f"Processing {len(prompts)} prompts, capturing layer {args.layer}")
    print(f"Output: {base_dir}")

    # Save metadata
    metadata = {
        'prompt_set': args.prompt_set,
        'model': args.model,
        'lora': args.lora,
        'layer': args.layer,
        'max_tokens': args.max_tokens,
        'n_prompts': len(prompts),
    }
    with open(base_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Process each prompt
    for prompt_data in tqdm(prompts, desc="Capturing"):
        prompt_id = prompt_data.get('id', prompts.index(prompt_data))
        prompt_text = prompt_data.get('text', prompt_data.get('prompt', ''))

        # Skip if already exists
        act_path = act_dir / f"{prompt_id}.pt"
        if act_path.exists():
            continue

        try:
            response_text, activations, token_ids = capture_activations(
                model, tokenizer, prompt_text, args.layer, args.max_tokens
            )

            # Save activations
            torch.save({
                f'layer{args.layer}': activations,
            }, act_path)

            # Save response
            resp_data = {
                'prompt_id': prompt_id,
                'prompt': prompt_text,
                'response': response_text,
                'token_ids': token_ids,
                'n_tokens': len(token_ids),
                'bias_id': prompt_data.get('bias_id'),
            }
            with open(resp_dir / f"{prompt_id}.json", 'w') as f:
                json.dump(resp_data, f, indent=2)

        except Exception as e:
            print(f"\nError on prompt {prompt_id}: {e}")
            continue

    print(f"\nDone. Saved to {base_dir}")


if __name__ == '__main__':
    main()
