#!/usr/bin/env python3
"""
Batched persona monitoring for Gemma 2 2B with all 8 traits.
Processes multiple prompts in parallel for faster inference.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
import json
import os
from tqdm import tqdm
from pathlib import Path

class BatchPersonaMonitor:
    """Monitor 8 persona traits during batched generation."""

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        persona_vectors_dir: str = "persona_vectors/gemma-2-2b-it",
        layer: int = 16,
        device: str = "cuda",
        batch_size: int = 4
    ):
        """
        Initialize monitor with model and persona vectors.

        Args:
            model_name: HuggingFace model identifier
            persona_vectors_dir: Directory containing persona vector .pt files
            layer: Layer to monitor (16 for Gemma 2B)
            device: "cuda" or "cpu"
            batch_size: Number of prompts to process in parallel
        """
        self.device = device
        self.layer = layer
        self.batch_size = batch_size

        # Load model
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=".cache"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=".cache")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Important for batched generation

        # Load all 8 persona vectors
        print(f"\nLoading persona vectors from: {persona_vectors_dir}")
        self.persona_vectors = {}

        traits = [
            "refusal", "uncertainty", "verbosity", "overconfidence",
            "corrigibility", "evil", "sycophantic", "hallucinating"
        ]

        for trait in traits:
            vector_path = f"{persona_vectors_dir}/{trait}_response_avg_diff.pt"
            if os.path.exists(vector_path):
                vector = torch.load(vector_path, map_location=device)
                # Extract specific layer if multi-layer vector
                if vector.dim() == 2:  # Shape [num_layers, hidden_dim]
                    vector = vector[layer]
                # Convert to float16 and normalize
                vector = vector.to(torch.float16)
                self.persona_vectors[trait] = vector / vector.norm()
                print(f"  ✅ {trait}: norm={vector.norm():.2f}")
            else:
                print(f"  ⚠️  {trait}: not found at {vector_path}")

        if not self.persona_vectors:
            raise ValueError("No persona vectors found!")

        print(f"\n✅ Loaded {len(self.persona_vectors)} traits")

    def generate_single(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> Dict:
        """
        Generate response for a single prompt with per-token monitoring.

        This is for non-batched generation where we can easily track per-token.
        """
        # Storage for per-token projections
        token_projections = {trait: [] for trait in self.persona_vectors.keys()}

        def hook_fn(module, input, output):
            """Capture activation at last token and project."""
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output

            # Get last token activation
            if activation.dim() == 3:
                activation = activation[:, -1, :]
            elif activation.dim() == 2:
                activation = activation
            else:
                return output

            activation = activation.detach()

            # Project onto each persona vector
            for trait_name, vector in self.persona_vectors.items():
                projection = (activation @ vector).item()
                token_projections[trait_name].append(projection)

            return output

        # Register hook
        target_layer = self.model.model.layers[self.layer]
        hook_handle = target_layer.register_forward_hook(hook_fn)

        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Generate with monitoring
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Clean up hook
        hook_handle.remove()

        # Decode tokens
        response_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        tokens = [self.tokenizer.decode([tok]) for tok in response_ids]

        return {
            "prompt": prompt,
            "response": response,
            "tokens": tokens,
            "trait_scores": token_projections
        }

    def process_prompts_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        save_dir: str = "pertoken/results"
    ) -> List[Dict]:
        """
        Process all prompts with batching for speed.

        Note: Batched generation doesn't easily allow per-token hooking,
        so we fall back to single generation with progress bar.
        """
        os.makedirs(save_dir, exist_ok=True)
        results = []

        print(f"\nProcessing {len(prompts)} prompts...")
        print(f"(Using single generation for per-token tracking)\n")

        for prompt in tqdm(prompts, desc="Generating"):
            try:
                result = self.generate_single(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                results.append(result)
            except Exception as e:
                print(f"\n❌ Error on prompt '{prompt[:50]}...': {e}")
                continue

        return results


def load_prompts_from_json(json_path: str) -> List[Dict[str, str]]:
    """
    Load prompts from JSON file and flatten into list with metadata.

    Returns:
        List of dicts with keys: prompt, trait, polarity (for teaching prompts)
        or: prompt, trait (for fluctuation prompts)
    """
    with open(json_path) as f:
        data = json.load(f)

    prompts = []

    # Check if this is teaching prompts (has high/low structure)
    first_key = list(data.keys())[0]
    if isinstance(data[first_key], dict) and "high" in data[first_key]:
        # Teaching prompts format
        for trait, polarities in data.items():
            for polarity in ["high", "low"]:
                for prompt_text in polarities[polarity]:
                    prompts.append({
                        "prompt": prompt_text,
                        "trait": trait,
                        "polarity": polarity,
                        "type": "teaching"
                    })
    else:
        # Fluctuation prompts format
        for trait, prompt_list in data.items():
            for prompt_text in prompt_list:
                prompts.append({
                    "prompt": prompt_text,
                    "trait": trait,
                    "type": "fluctuation"
                })

    return prompts


def run_monitoring_pipeline(
    prompt_files: List[str],
    output_prefix: str = "gemma_2b",
    max_new_tokens: int = 150
):
    """
    Run complete monitoring pipeline on prompt files.

    Args:
        prompt_files: List of JSON files with prompts
        output_prefix: Prefix for output files
        max_new_tokens: Max tokens to generate per prompt
    """
    print("\n" + "="*70)
    print("GEMMA 2B PER-TOKEN MONITORING PIPELINE")
    print("="*70)

    # Initialize monitor
    monitor = BatchPersonaMonitor(
        model_name="google/gemma-2-2b-it",
        persona_vectors_dir="persona_vectors/gemma-2-2b-it",
        layer=16,
        batch_size=4
    )

    all_results = []

    for prompt_file in prompt_files:
        print(f"\n{'='*70}")
        print(f"Processing: {prompt_file}")
        print(f"{'='*70}")

        # Load prompts
        prompts = load_prompts_from_json(prompt_file)
        print(f"Loaded {len(prompts)} prompts")

        # Process
        results = monitor.process_prompts_batch(
            prompts=[p["prompt"] for p in prompts],
            max_new_tokens=max_new_tokens
        )

        # Add metadata back
        for i, result in enumerate(results):
            result["metadata"] = prompts[i]

        all_results.extend(results)

        # Save intermediate results
        file_basename = Path(prompt_file).stem
        output_file = f"pertoken/results/{output_prefix}_{file_basename}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Saved {len(results)} results to {output_file}")

    # Save combined results
    combined_file = f"pertoken/results/{output_prefix}_all_results.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70)
    print("MONITORING COMPLETE!")
    print("="*70)
    print(f"✅ Total prompts processed: {len(all_results)}")
    print(f"✅ Total tokens tracked: {sum(len(r['tokens']) for r in all_results)}")
    print(f"✅ Combined results: {combined_file}")

    # Quick stats
    print("\n" + "="*70)
    print("TRAIT STATISTICS (across all prompts)")
    print("="*70)

    for trait in monitor.persona_vectors.keys():
        all_scores = []
        for result in all_results:
            if trait in result["trait_scores"]:
                all_scores.extend(result["trait_scores"][trait])

        if all_scores:
            print(f"\n{trait.capitalize():15s}: "
                  f"min={min(all_scores):6.2f}, "
                  f"max={max(all_scores):6.2f}, "
                  f"mean={np.mean(all_scores):6.2f}, "
                  f"std={np.std(all_scores):6.2f}")

    return all_results


if __name__ == "__main__":
    import fire

    def main(
        teaching: bool = True,
        fluctuations: bool = True,
        max_tokens: int = 150
    ):
        """
        Run monitoring on prompt files.

        Args:
            teaching: Process teaching prompts (single_trait_teaching.json)
            fluctuations: Process fluctuation prompts (single_trait_fluctuations.json)
            max_tokens: Max tokens to generate per response
        """
        files = []
        if teaching:
            files.append("prompts/single_trait_teaching.json")
        if fluctuations:
            files.append("prompts/single_trait_fluctuations.json")

        if not files:
            print("No prompt files selected! Use --teaching and/or --fluctuations")
            return

        run_monitoring_pipeline(
            prompt_files=files,
            max_new_tokens=max_tokens
        )

    fire.Fire(main)
