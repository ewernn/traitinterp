"""
Model-diff analysis: compare trait scores between two model runs.

Uses cosine similarity (normalized) and response-level averaging.

Input:
    - Two runs of captured activations (e.g., 'clean' and 'lora')
    - Trait vector from extraction

Output:
    - Console report with statistics
    - Optional: saved results JSON, plot

Usage:
    # Single layer
    python analysis/rm_sycophancy/analyze.py \\
        --baseline clean \\
        --compare lora \\
        --trait rm_hack/ulterior_motive \\
        --layers 30

    # Layer sweep
    python analysis/rm_sycophancy/analyze.py \\
        --baseline clean \\
        --compare lora \\
        --trait rm_hack/ulterior_motive \\
        --layers 20-35
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from scipy import stats
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.paths import get, get_vector_path

# Defaults
EXPERIMENT = "rm_syco"
DEFAULT_LAYERS = "20-35"
DEFAULT_POSITION = "response[:5]"
DEFAULT_COMPONENT = "residual"


def parse_layers(layers_str: str) -> list:
    """Parse layer specification: '30', '20-35', or '24,28,30,31'."""
    layers = []
    for part in layers_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def cosine_similarity(activations: torch.Tensor, vector: torch.Tensor) -> np.ndarray:
    """
    Compute cosine similarity between activations and vector.

    Args:
        activations: [n_tokens, hidden_dim]
        vector: [hidden_dim]

    Returns:
        [n_tokens] array of cosine similarities (bounded [-1, 1])
    """
    activations = activations.float()
    vector = vector.float()

    # Normalize
    act_norms = activations.norm(dim=1, keepdim=True)
    vec_norm = vector.norm()

    # Cosine similarity
    similarities = (activations @ vector) / (act_norms.squeeze() * vec_norm)

    return similarities.numpy()


def load_run_activations(prompt_set: str, layer: int, experiment: str) -> dict:
    """
    Load all activations from a prompt set.

    Args:
        prompt_set: Name like 'rm_sycophancy_train_100_sycophant' or 'rm_sycophancy_train_100_clean'

    Returns:
        Dict mapping prompt_id -> activations tensor [n_tokens, hidden_dim]
    """
    # Use standard inference path
    act_dir = get('inference.raw_residual', experiment=experiment, prompt_set=prompt_set)

    if not act_dir.exists():
        raise FileNotFoundError(f"Activations not found: {act_dir}")

    activations = {}
    for pt_file in sorted(act_dir.glob("*.pt")):
        prompt_id = pt_file.stem
        data = torch.load(pt_file, weights_only=True)

        # Detect format and extract activations
        layer_key = f'layer{layer}'

        if layer_key in data:
            # Format A (26-prompt): d['layer{N}'] at top level
            activations[prompt_id] = data[layer_key]
        elif 'response' in data and 'activations' in data['response']:
            # Format B (100-prompt): d['response']['activations'][layer_int][sublayer]
            resp_acts = data['response']['activations']
            if layer in resp_acts:
                layer_data = resp_acts[layer]
                # Handle component nesting (residual, attn_out)
                if isinstance(layer_data, dict) and 'residual' in layer_data:
                    activations[prompt_id] = layer_data['residual']
                elif isinstance(layer_data, torch.Tensor):
                    activations[prompt_id] = layer_data
                else:
                    raise KeyError(f"Unexpected layer data format in {pt_file}: {type(layer_data)}")
            else:
                available = sorted(resp_acts.keys())
                raise KeyError(f"Layer {layer} not found in {pt_file}. Available: {available}")
        else:
            # Unknown format
            raise KeyError(f"Unknown activation format in {pt_file}. Keys: {list(data.keys())}")

    return activations


def load_trait_vector(trait: str, layer: int, experiment: str,
                      position: str = DEFAULT_POSITION,
                      component: str = DEFAULT_COMPONENT) -> torch.Tensor:
    """Load trait vector from extraction."""
    # Try mean_diff first (more common), then probe
    for method in ['mean_diff', 'probe']:
        vector_path = get_vector_path(experiment, trait, method, layer, component, position)
        if vector_path.exists():
            return torch.load(vector_path, weights_only=True)

    raise FileNotFoundError(f"No vector found for {trait} at layer {layer} (position={position}, component={component})")


def compute_response_scores(activations: dict, vector: torch.Tensor) -> dict:
    """
    Compute mean cosine similarity per response.

    Returns:
        Dict mapping prompt_id -> mean_score
    """
    scores = {}
    for prompt_id, acts in activations.items():
        sims = cosine_similarity(acts, vector)
        scores[prompt_id] = float(np.mean(sims))
    return scores


def analyze_layer(baseline_set: str, compare_set: str, trait: str, layer: int,
                  experiment: str, position: str = DEFAULT_POSITION,
                  component: str = DEFAULT_COMPONENT, verbose: bool = True) -> dict:
    """Analyze a single layer and return results dict."""

    # Load trait vector
    try:
        vector = load_trait_vector(trait, layer, experiment, position, component)
    except FileNotFoundError:
        if verbose:
            print(f"  L{layer}: No vector found, skipping")
        return None

    # Load activations
    try:
        baseline_acts = load_run_activations(baseline_set, layer, experiment)
        compare_acts = load_run_activations(compare_set, layer, experiment)
    except (FileNotFoundError, KeyError) as e:
        if verbose:
            print(f"  L{layer}: {e}")
        return None

    # Find common prompt IDs
    common_ids = set(baseline_acts.keys()) & set(compare_acts.keys())
    if len(common_ids) == 0:
        if verbose:
            print(f"  L{layer}: No common prompts")
        return None

    # Compute response-level scores
    baseline_scores = compute_response_scores(
        {k: v for k, v in baseline_acts.items() if k in common_ids}, vector
    )
    compare_scores = compute_response_scores(
        {k: v for k, v in compare_acts.items() if k in common_ids}, vector
    )

    # Convert to arrays (matched by prompt_id)
    ids = sorted(common_ids)
    baseline_arr = np.array([baseline_scores[i] for i in ids])
    compare_arr = np.array([compare_scores[i] for i in ids])

    # Statistics
    baseline_mean = np.mean(baseline_arr)
    baseline_std = np.std(baseline_arr)
    compare_mean = np.mean(compare_arr)
    compare_std = np.std(compare_arr)

    diff = compare_mean - baseline_mean
    pooled_std = np.sqrt((baseline_std**2 + compare_std**2) / 2)
    effect_size = diff / pooled_std if pooled_std > 0 else 0

    # Independent t-test
    t_stat, p_value = stats.ttest_ind(compare_arr, baseline_arr)

    return {
        'layer': layer,
        'trait': trait,
        'baseline': {
            'name': baseline_set,
            'mean': float(baseline_mean),
            'std': float(baseline_std),
            'n': len(baseline_arr),
        },
        'compare': {
            'name': compare_set,
            'mean': float(compare_mean),
            'std': float(compare_std),
            'n': len(compare_arr),
        },
        'diff': float(diff),
        'effect_size': float(effect_size),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'per_response': {
            'baseline': {str(i): float(baseline_scores[i]) for i in ids},
            'compare': {str(i): float(compare_scores[i]) for i in ids},
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Model-diff analysis using cosine similarity")
    parser.add_argument("--baseline", required=True, help="Baseline prompt set (e.g., 'rm_sycophancy_train_100_clean')")
    parser.add_argument("--compare", required=True, help="Comparison prompt set (e.g., 'rm_sycophancy_train_100_sycophant')")
    parser.add_argument("--trait", required=True, help="Trait path (e.g., 'rm_hack/ulterior_motive')")
    parser.add_argument("--layers", type=str, default=DEFAULT_LAYERS,
                       help="Layers to analyze: '30', '20-35', or '24,28,30,31' (default: 20-35)")
    parser.add_argument("--position", type=str, default=DEFAULT_POSITION,
                       help=f"Vector position (default: {DEFAULT_POSITION})")
    parser.add_argument("--component", type=str, default=DEFAULT_COMPONENT,
                       help=f"Vector component (default: {DEFAULT_COMPONENT})")
    parser.add_argument("--experiment", default=EXPERIMENT, help="Experiment name")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument("--plot", action="store_true", help="Save effect size plot")
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    trait_name = args.trait.split('/')[-1]

    print(f"Model-Diff Analysis: {args.trait}")
    print(f"Layers: {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    print(f"Position: {args.position}, Component: {args.component}")
    print("=" * 60)

    # Analyze each layer
    results = []
    for layer in layers:
        result = analyze_layer(
            args.baseline, args.compare, args.trait, layer,
            args.experiment, args.position, args.component, verbose=False
        )
        if result:
            results.append(result)
            # Print progress
            sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
            print(f"  L{layer:2d}: effect={result['effect_size']:+.2f}σ  diff={result['diff']:+.4f}  p={result['p_value']:.2e} {sig}")

    if not results:
        print("\nNo valid results. Check that activations and vectors exist.")
        return

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {trait_name}")
    print(f"{'=' * 60}")

    # Find best layer
    best = max(results, key=lambda r: abs(r['effect_size']))
    print(f"\nBest layer: L{best['layer']}")
    print(f"  effect size = {best['effect_size']:+.2f} std")
    print(f"  diff        = {best['diff']:+.4f}")
    print(f"  p-value     = {best['p_value']:.2e}")
    print(f"  n           = {best['baseline']['n']}")

    # Interpretation
    if best['p_value'] < 0.001 and abs(best['effect_size']) > 0.5:
        direction = "higher" if best['diff'] > 0 else "lower"
        print(f"\n  SIGNIFICANT: {args.compare} has {direction} {trait_name}")
        print(f"  Effect size {abs(best['effect_size']):.1f}σ is {'large' if abs(best['effect_size']) > 0.8 else 'medium'}")
    elif best['p_value'] < 0.05:
        print(f"\n  WEAK SIGNAL: Statistically significant but small effect")
    else:
        print(f"\n  NO SIGNAL: No significant difference detected")

    # Optional: save results
    if args.save:
        out_dir = get('experiments.base', experiment=args.experiment) / 'rm_sycophancy' / 'analysis'
        out_dir.mkdir(parents=True, exist_ok=True)

        sweep_results = {
            'trait': args.trait,
            'position': args.position,
            'component': args.component,
            'baseline': args.baseline,
            'compare': args.compare,
            'best_layer': best['layer'],
            'best_effect_size': best['effect_size'],
            'layers': {r['layer']: {
                'effect_size': r['effect_size'],
                'diff': r['diff'],
                'p_value': r['p_value'],
            } for r in results}
        }

        out_path = out_dir / f"model_diff_{trait_name}_sweep.json"
        with open(out_path, 'w') as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\n  Saved results to: {out_path}")

    # Optional: plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 5))

            layer_nums = [r['layer'] for r in results]
            effect_sizes = [r['effect_size'] for r in results]

            colors = ['#d62728' if es > 0 else '#1f77b4' for es in effect_sizes]
            ax.bar(layer_nums, effect_sizes, color=colors, alpha=0.7)

            # Highlight best
            best_idx = layer_nums.index(best['layer'])
            ax.bar([best['layer']], [best['effect_size']], color='gold', edgecolor='black', linewidth=2)

            ax.axhline(0, color='black', linewidth=0.5)
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect (0.5σ)')
            ax.axhline(-0.5, color='gray', linestyle='--', alpha=0.5)

            ax.set_xlabel('Layer', fontsize=12)
            ax.set_ylabel('Effect Size (σ)', fontsize=12)
            ax.set_title(f'Model Diff: {trait_name}\n'
                        f'Best: L{best["layer"]} = {best["effect_size"]:+.2f}σ (p={best["p_value"]:.2e})',
                        fontsize=13)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()

            out_dir = get('experiments.base', experiment=args.experiment) / 'assets'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"model_diff_{trait_name}_sweep.png"

            plt.savefig(out_path, dpi=150)
            print(f"  Saved plot to: {out_path}")

        except ImportError:
            print("  Warning: matplotlib not installed, skipping plot")


if __name__ == '__main__':
    main()
