"""Regenerate steered responses for top-scoring configs in quantization-sensitivity.

Input: /tmp/top_configs.json (40 cells)
Output: experiments/quant-sensitivity/{variant_dir}/steering/{trait}/.../best_responses.json

Usage: python dev/steering/regenerate_top_responses.py
"""

import json
import sys
import gc
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from utils.model import load_model_with_lora, format_prompt
from utils.traits import load_questions_from_file
from utils.vectors import load_vector
from utils.model_generation import batched_steering_generate
from utils.paths import get as get_path


VARIANT_CONFIGS = {
    "llama-8b":       ("meta-llama/Llama-3.1-8B-Instruct", {}),
    "llama-8b-nf4":   ("meta-llama/Llama-3.1-8B-Instruct", {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}),
    "llama-8b-fp4":   ("meta-llama/Llama-3.1-8B-Instruct", {"load_in_4bit": True, "bnb_4bit_quant_type": "fp4"}),
    "llama-8b-awq":   ("hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", {}),
    "olmo-7b":        ("allenai/OLMo-2-1124-7B-Instruct", {}),
    "olmo-7b-nf4":    ("allenai/OLMo-2-1124-7B-Instruct", {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}),
}


def load_direction(trait: str) -> str:
    """Read 'direction' from datasets/traits/{trait}/steering.json."""
    path = Path(f"datasets/traits/{trait}/steering.json")
    with open(path) as f:
        data = json.load(f)
    return data.get("direction", "positive")


def output_path(cell) -> Path:
    """Where to save responses for this cell."""
    return Path(f"experiments/quant-sensitivity/{cell['variant_dir']}/steering/"
                f"{cell['trait']}/instruct/{cell['pos_dir']}/steering/best_responses.json")


def main():
    top_configs_path = Path("/tmp/top_configs.json")
    if not top_configs_path.exists():
        print(f"ERROR: {top_configs_path} not found. Run parser first.")
        sys.exit(1)

    cells = json.loads(top_configs_path.read_text())
    print(f"Loaded {len(cells)} cells")

    # Group by variant (8 model loads)
    by_variant = defaultdict(list)
    for c in cells:
        by_variant[c["variant"]].append(c)

    total_start = time.time()

    # Run AWQ last so its potential failure doesn't block OLMo
    variant_order = sorted(by_variant.keys(), key=lambda v: (1 if 'awq' in v else 0, v))

    for variant in variant_order:
        variant_cells = by_variant[variant]
        hf_repo, load_kwargs = VARIANT_CONFIGS[variant]
        print(f"\n{'='*60}")
        print(f"=== {variant} ({hf_repo}) ===")
        print(f"=== {len(variant_cells)} cells ===")
        print(f"{'='*60}")

        load_start = time.time()
        try:
            model, tokenizer = load_model_with_lora(hf_repo, **load_kwargs)
            model.eval()
        except Exception as e:
            print(f"!! FAILED to load {variant}: {e}")
            continue
        print(f"Model loaded in {time.time() - load_start:.1f}s")

        for cell in variant_cells:
            cell_start = time.time()
            trait = cell["trait"]
            variant_dir = cell["variant_dir"]
            layer = cell["layer"]
            weight = cell["weight"]
            position = cell["position"]  # e.g. 'response[:]' or 'prompt[-1]'
            method = cell["method"]

            out = output_path(cell)
            if out.exists():
                print(f"  SKIP (exists): {variant_dir}/{trait}")
                continue

            # Determine position dir for vector path
            if "response" in position:
                pos_for_vec = "response_all"
            else:
                pos_for_vec = "prompt_-1"

            # Load vector
            experiment = f"quant-sensitivity/{variant_dir}"
            vec = load_vector(
                experiment=experiment, trait=trait, layer=layer,
                model_variant="instruct", method=method,
                component=cell.get("component", "residual"),
                position=pos_for_vec,
            )
            if vec is None:
                print(f"  MISS (no vector): {experiment} {trait} L{layer} — skipping")
                continue
            vec = vec.to(model.device).to(torch.bfloat16)

            # Determine coefficient sign from direction
            direction = load_direction(trait)
            signed_coef = weight if direction == "positive" else -weight

            # Load questions
            questions = load_questions_from_file(f"datasets/traits/{trait}/steering.json")
            prompts = [format_prompt(q, tokenizer) for q in questions]

            # Generate steered responses
            responses = batched_steering_generate(
                model, tokenizer,
                configs=[(layer, vec, signed_coef)],
                prompts=prompts,
                max_new_tokens=128,
                temperature=0.0,
            )
            # responses is flat list of len(prompts) (one config × N prompts)

            # Save
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "variant": variant,
                "variant_dir": variant_dir,
                "trait": trait,
                "layer": layer,
                "weight": weight,
                "signed_coef": signed_coef,
                "direction": direction,
                "method": method,
                "position": position,
                "trait_mean_archival": cell.get("trait_mean"),
                "coherence_mean_archival": cell.get("coherence_mean"),
                "items": [
                    {"question": q, "response": r}
                    for q, r in zip(questions, responses)
                ],
            }
            out.write_text(json.dumps(payload, indent=2))
            print(f"  OK ({time.time() - cell_start:.1f}s): {variant_dir}/{trait} → {out}")

        # Free model
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Variant {variant} complete in {time.time() - load_start:.1f}s")

    print(f"\nTOTAL: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
