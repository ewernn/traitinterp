#!/usr/bin/env python3
"""
Post-hoc projection: compute trait projections from saved raw activations.

This allows re-projecting onto different traits or with different vectors
without re-running model inference.

Layer selection (default: auto per trait):
- Steering results (ground truth) if available
- Effect size (best proxy, r=0.898) otherwise
- Use --layer N to override for all traits

Usage:
    # Project onto all traits (auto-selects best layer per trait)
    python inference/project_raw_activations_onto_traits.py \\
        --experiment my_experiment \\
        --prompt-set main_prompts

    # Override with fixed layer for all traits
    python inference/project_raw_activations_onto_traits.py \\
        --experiment my_experiment \\
        --prompt-set main_prompts \\
        --layer 16

    # Project onto specific traits
    python inference/project_raw_activations_onto_traits.py \\
        --experiment my_experiment \\
        --prompt-set main_prompts \\
        --traits behavioral/refusal,cognitive/retrieval

    # Project attn_out activations onto attn_out vectors
    python inference/project_raw_activations_onto_traits.py \\
        --experiment my_experiment \\
        --prompt-set harmful \\
        --component attn_out \\
        --layer 8
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from tqdm import tqdm

from traitlens import projection
from traitlens.compute import compute_derivative, compute_second_derivative
from utils.vectors import load_vector_metadata


MODEL_NAME = "google/gemma-2-2b-it"
LOGIT_LENS_LAYERS = [0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 25]


# ============================================================================
# Trait Discovery
# ============================================================================

def discover_traits(experiment_name: str) -> List[Tuple[str, str]]:
    """Discover all traits with vectors in an experiment."""
    from utils.paths import get
    extraction_dir = get('extraction.base', experiment=experiment_name)

    if not extraction_dir.exists():
        raise FileNotFoundError(f"Extraction directory not found: {extraction_dir}")

    traits = []
    for category_dir in sorted(extraction_dir.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        for trait_dir in sorted(category_dir.iterdir()):
            if not trait_dir.is_dir():
                continue
            vectors_dir = trait_dir / "vectors"
            if vectors_dir.exists() and list(vectors_dir.glob('*.pt')):
                traits.append((category_dir.name, trait_dir.name))

    return traits


def find_vector_method(vectors_dir: Path, layer: int, component: str = "residual") -> Optional[str]:
    """Auto-detect best vector method for a layer.

    Args:
        vectors_dir: Path to vectors directory
        layer: Layer number
        component: 'residual' (default) or 'attn_out'

    Returns:
        Method name if found, None otherwise
    """
    for method in ["probe", "mean_diff", "gradient"]:
        if component == "attn_out":
            # Check for attn_out vectors: attn_out_probe_layer8.pt
            if (vectors_dir / f"attn_out_{method}_layer{layer}.pt").exists():
                return method
        else:
            # Standard residual vectors: probe_layer8.pt
            if (vectors_dir / f"{method}_layer{layer}.pt").exists():
                return method
    return None


def find_available_vectors(vectors_dir: Path, layer: int) -> list[tuple[str, str, Path]]:
    """Find all available vectors for a layer (both residual and attn_out).

    Returns:
        List of (component, method, path) tuples
    """
    vectors = []
    for method in ["probe", "mean_diff", "gradient"]:
        # Residual vectors
        residual_path = vectors_dir / f"{method}_layer{layer}.pt"
        if residual_path.exists():
            vectors.append(("residual", method, residual_path))

        # attn_out vectors
        attn_path = vectors_dir / f"attn_out_{method}_layer{layer}.pt"
        if attn_path.exists():
            vectors.append(("attn_out", method, attn_path))

    return vectors


# ============================================================================
# Dynamics Analysis
# ============================================================================

def analyze_dynamics(trajectory: torch.Tensor) -> Dict:
    """Compute velocity, acceleration, commitment point, and persistence."""
    if len(trajectory) < 2:
        return {
            'commitment_point': None,
            'peak_velocity': 0.0,
            'avg_velocity': 0.0,
            'persistence': 0,
            'velocity': [],
            'acceleration': [],
        }

    velocity = compute_derivative(trajectory.unsqueeze(-1)).squeeze(-1)

    if len(trajectory) >= 3:
        acceleration = compute_second_derivative(trajectory.unsqueeze(-1)).squeeze(-1)
    else:
        acceleration = torch.tensor([])

    # Commitment point
    commitment = None
    if len(acceleration) > 0:
        candidates = (acceleration.abs() < 0.1).nonzero()
        if len(candidates) > 0:
            commitment = candidates[0].item()

    # Persistence
    persistence = 0
    if len(trajectory) > 0:
        peak_idx = trajectory.abs().argmax().item()
        peak_value = trajectory[peak_idx].abs().item()
        if peak_idx < len(trajectory) - 1:
            threshold = peak_value * 0.5
            persistence = (trajectory[peak_idx + 1:].abs() > threshold).sum().item()

    return {
        'commitment_point': commitment,
        'peak_velocity': velocity.abs().max().item() if len(velocity) > 0 else 0.0,
        'avg_velocity': velocity.abs().mean().item() if len(velocity) > 0 else 0.0,
        'persistence': persistence,
        'velocity': velocity.tolist(),
        'acceleration': acceleration.tolist() if len(acceleration) > 0 else [],
    }


# ============================================================================
# Projection
# ============================================================================

def project_onto_vector(activations: Dict, vector: torch.Tensor, n_layers: int,
                        component: str = "residual") -> torch.Tensor:
    """Project activations onto trait vector.

    Args:
        activations: Dict of layer -> sublayer -> tensor
        vector: Trait vector
        n_layers: Number of layers
        component: 'residual' returns [n_tokens, n_layers, 3] for residual_in/after_attn/residual_out
                   'attn_out' returns [n_tokens, n_layers, 1] for attn_out only

    Returns:
        Projection tensor
    """
    n_tokens = activations[0]['residual_in'].shape[0]

    if component == "attn_out":
        # Project only attn_out activations
        result = torch.zeros(n_tokens, n_layers, 1)
        for layer in range(n_layers):
            if 'attn_out' in activations[layer] and activations[layer]['attn_out'].numel() > 0:
                result[:, layer, 0] = projection(activations[layer]['attn_out'], vector, normalize_vector=True)
    else:
        # Project residual sublayers
        result = torch.zeros(n_tokens, n_layers, 3)
        sublayers = ['residual_in', 'after_attn', 'residual_out']
        for layer in range(n_layers):
            for s_idx, sublayer in enumerate(sublayers):
                result[:, layer, s_idx] = projection(activations[layer][sublayer], vector, normalize_vector=True)

    return result


def compute_activation_norms(activations: Dict, n_layers: int) -> List[float]:
    """Compute activation norms per layer (averaged across tokens and sublayers).

    Returns [n_layers] array of ||h|| values showing activation magnitude by layer.
    """
    sublayers = ['residual_in', 'after_attn', 'residual_out']
    norms = []

    for layer in range(n_layers):
        layer_norms = []
        for sublayer in sublayers:
            # Compute L2 norm per token, then average across tokens
            h = activations[layer][sublayer]  # [n_tokens, hidden_dim]
            token_norms = h.norm(dim=-1)  # [n_tokens]
            layer_norms.append(token_norms.mean().item())
        # Average across sublayers
        norms.append(sum(layer_norms) / len(layer_norms))

    return norms


# ============================================================================
# Logit Lens (requires model)
# ============================================================================

def compute_logit_lens_from_raw(activations: Dict, model, tokenizer, n_layers: int) -> Dict:
    """Compute logit lens from saved activations (requires model for unembed)."""
    if hasattr(model, 'lm_head'):
        unembed = model.lm_head.weight.detach()
    else:
        unembed = model.model.embed_tokens.weight.detach()

    result = {}
    for layer in LOGIT_LENS_LAYERS:
        if layer >= n_layers:
            continue

        residual = activations[layer]['residual_out']
        if len(residual.shape) == 1:
            residual = residual.unsqueeze(0)

        logits = residual.to(unembed.device).to(unembed.dtype) @ unembed.T
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = probs.topk(3, dim=-1)

        top_tokens = []
        for token_idx in range(top_ids.shape[0]):
            tokens = [tokenizer.decode([tid.item()]) for tid in top_ids[token_idx]]
            top_tokens.append(tokens)

        result[f'layer_{layer}'] = {
            'tokens': top_tokens,
            'probs': top_probs.cpu().tolist()
        }

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Post-hoc projection from raw activations")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--prompt-set", help="Prompt set name (or use --all-prompt-sets)")
    parser.add_argument("--all-prompt-sets", action="store_true", help="Process all prompt sets")
    parser.add_argument("--traits", help="Comma-separated traits (category/name format)")
    parser.add_argument("--layer", type=int,
                       help="Override layer for all traits (default: auto-select best per trait)")
    parser.add_argument("--method", help="Vector method (default: auto-detect or use best from evaluation)")
    parser.add_argument("--component", choices=["residual", "attn_out"], default="residual",
                       help="Activation component to project (default: residual)")
    parser.add_argument("--logit-lens", action="store_true", help="Compute logit lens")
    parser.add_argument("--dynamics-only", action="store_true",
                       help="Only recompute dynamics from existing projections")
    parser.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args()

    if not args.prompt_set and not args.all_prompt_sets:
        parser.error("Either --prompt-set or --all-prompt-sets is required")

    from utils.paths import get as get_path

    inference_dir = get_path('inference.base', experiment=args.experiment)
    raw_residual_dir = inference_dir / "raw" / "residual"

    # Discover prompt sets if --all-prompt-sets
    if args.all_prompt_sets:
        if not raw_residual_dir.exists():
            print(f"Raw residual directory not found: {raw_residual_dir}")
            return
        prompt_sets = [d.name for d in raw_residual_dir.iterdir() if d.is_dir()]
        print(f"Found {len(prompt_sets)} prompt sets: {', '.join(prompt_sets)}")
    else:
        prompt_sets = [args.prompt_set]

    for prompt_set in prompt_sets:
        print(f"\n{'='*60}\nProcessing prompt set: {prompt_set}\n{'='*60}")
        process_prompt_set(args, inference_dir, prompt_set)


def process_prompt_set(args, inference_dir, prompt_set):
    """Process a single prompt set."""
    from utils.paths import get as get_path

    raw_dir = inference_dir / "raw" / "residual" / prompt_set

    if not raw_dir.exists():
        print(f"Raw activations not found: {raw_dir}")
        print("Run 'python inference/capture_raw_activations.py' first to capture raw activations.")
        return

    # Find raw activation files (new format: {id}.pt)
    raw_files = sorted(raw_dir.glob("*.pt"), key=lambda f: int(f.stem) if f.stem.isdigit() else 0)
    if not raw_files:
        print(f"No raw activation files found in {raw_dir}")
        return

    print(f"Found {len(raw_files)} raw activation files")

    # Get traits to project onto
    if args.traits:
        trait_list = [tuple(t.split('/')) for t in args.traits.split(',')]
    else:
        trait_list = discover_traits(args.experiment)

    if not trait_list:
        print("No traits found")
        return

    print(f"Projecting onto {len(trait_list)} traits")

    # Auto-layer is default; --layer overrides for all traits
    auto_layer = args.layer is None
    if auto_layer:
        from utils.vectors import get_best_layer
        print("Auto-selecting best layer per trait (use --layer N to override)")

    # Load trait vectors
    trait_vectors = {}
    for category, trait_name in trait_list:
        trait_path = f"{category}/{trait_name}"
        vectors_dir = get_path('extraction.vectors', experiment=args.experiment, trait=trait_path)

        # Determine layer and method
        if auto_layer:
            best = get_best_layer(args.experiment, trait_path)
            layer = best['layer']
            method = args.method or best['method']
            print(f"  {trait_path}: L{layer} {method} (from {best['source']}: {best['score']:.2f})")
        else:
            layer = args.layer
            method = args.method or find_vector_method(vectors_dir, layer, component=args.component)

        if not method:
            print(f"  Skip {trait_path}: no {args.component} vector at layer {layer}")
            continue

        # Build vector path based on component type
        if args.component == "attn_out":
            vector_path = vectors_dir / f"attn_out_{method}_layer{layer}.pt"
        else:
            vector_path = vectors_dir / f"{method}_layer{layer}.pt"

        if not vector_path.exists():
            print(f"  Skip {trait_path}: {vector_path} not found")
            continue

        vector = torch.load(vector_path, weights_only=True).to(torch.float16)

        # Load vector metadata for source info
        vec_metadata = load_vector_metadata(args.experiment, trait_path)

        trait_vectors[(category, trait_name)] = (vector, method, vector_path, layer, vec_metadata)

    print(f"Loaded {len(trait_vectors)} trait vectors")

    # Load model only if logit lens requested
    model = None
    tokenizer = None
    if args.logit_lens:
        print(f"\nLoading model for logit lens: {MODEL_NAME}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
            attn_implementation='eager'
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Process each raw file
    for raw_file in tqdm(raw_files, desc="Projecting"):
        # Extract prompt ID from filename (new format: {id}.pt)
        prompt_id = raw_file.stem  # e.g., "1", "2", etc.

        # Load raw activations
        data = torch.load(raw_file, weights_only=False)
        n_layers = len(data['prompt']['activations'])

        # Compute activation norms (trait-independent, computed once per prompt)
        prompt_norms = compute_activation_norms(data['prompt']['activations'], n_layers)
        response_norms = compute_activation_norms(data['response']['activations'], n_layers)

        # Compute logit lens if requested
        logit_lens_data = None
        if args.logit_lens and model is not None:
            logit_lens_data = {
                'prompt': compute_logit_lens_from_raw(data['prompt']['activations'], model, tokenizer, n_layers),
                'response': compute_logit_lens_from_raw(data['response']['activations'], model, tokenizer, n_layers)
            }

        # Project onto each trait
        for (category, trait_name), (vector, method, vector_path, layer, vec_metadata) in trait_vectors.items():
            # Path: {component}_stream/{prompt_set}/{id}.json
            stream_name = "attn_stream" if args.component == "attn_out" else "residual_stream"
            out_dir = inference_dir / category / trait_name / stream_name / prompt_set
            out_file = out_dir / f"{prompt_id}.json"

            if args.skip_existing and out_file.exists():
                continue

            if args.dynamics_only and out_file.exists():
                # Load existing and just recompute dynamics
                with open(out_file) as f:
                    proj_data = json.load(f)

                prompt_proj = torch.tensor(proj_data['projections']['prompt'])
                response_proj = torch.tensor(proj_data['projections']['response'])
            else:
                # Compute projections
                prompt_proj = project_onto_vector(data['prompt']['activations'], vector, n_layers, component=args.component)
                response_proj = project_onto_vector(data['response']['activations'], vector, n_layers, component=args.component)

            # Compute dynamics
            prompt_scores_avg = prompt_proj.mean(dim=(1, 2))
            response_scores_avg = response_proj.mean(dim=(1, 2))
            all_scores = torch.cat([prompt_scores_avg, response_scores_avg])

            proj_data = {
                'prompt': {
                    'text': data['prompt']['text'],
                    'tokens': data['prompt']['tokens'],
                    'token_ids': data['prompt']['token_ids'],
                    'n_tokens': len(data['prompt']['tokens'])
                },
                'response': {
                    'text': data['response']['text'],
                    'tokens': data['response']['tokens'],
                    'token_ids': data['response']['token_ids'],
                    'n_tokens': len(data['response']['tokens'])
                },
                'projections': {
                    'prompt': prompt_proj.tolist(),
                    'response': response_proj.tolist()
                },
                'dynamics': analyze_dynamics(all_scores),
                'activation_norms': {
                    'prompt': prompt_norms,
                    'response': response_norms
                },
                'metadata': {
                    'inference_model': MODEL_NAME,
                    'inference_experiment': args.experiment,
                    'prompt_id': prompt_id,
                    'prompt_set': prompt_set,
                    'vector_source': {
                        'model': vec_metadata.get('extraction_model', 'unknown'),
                        'experiment': args.experiment,
                        'trait': f"{category}/{trait_name}",
                        'method': method,
                        'layer': layer,
                        'component': args.component,
                    },
                    'projection_date': datetime.now().isoformat()
                }
            }

            if logit_lens_data:
                proj_data['logit_lens'] = logit_lens_data

            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_file, 'w') as f:
                json.dump(proj_data, f, indent=2)

    print(f"\nProjections saved to: {inference_dir}/{{category}}/{{trait}}/residual_stream/{prompt_set}/")


if __name__ == "__main__":
    main()
