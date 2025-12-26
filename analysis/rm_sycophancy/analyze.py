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
    python analysis/rm_sycophancy/analyze.py \\
        --baseline clean \\
        --compare lora \\
        --trait rm_hack/ulterior_motive \\
        --layer 30
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from scipy import stats
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.paths import get

# Defaults
EXPERIMENT = "llama-3.3-70b"
DEFAULT_LAYER = 30


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


def load_trait_vector(trait: str, layer: int, experiment: str) -> torch.Tensor:
    """Load trait vector from extraction."""
    # Try probe first, then mean_diff
    for method in ['probe', 'mean_diff']:
        vector_path = get('extraction.vectors', experiment=experiment, trait=trait) / f'{method}_layer{layer}.pt'
        if vector_path.exists():
            return torch.load(vector_path, weights_only=True)

    raise FileNotFoundError(f"No vector found for {trait} at layer {layer}")


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


def main():
    parser = argparse.ArgumentParser(description="Model-diff analysis using cosine similarity")
    parser.add_argument("--baseline", required=True, help="Baseline prompt set (e.g., 'rm_sycophancy_train_100_clean')")
    parser.add_argument("--compare", required=True, help="Comparison prompt set (e.g., 'rm_sycophancy_train_100_sycophant')")
    parser.add_argument("--trait", required=True, help="Trait path (e.g., 'rm_hack/ulterior_motive')")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER, help="Layer to analyze")
    parser.add_argument("--experiment", default=EXPERIMENT, help="Experiment name")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument("--plot", action="store_true", help="Save histogram plot")
    args = parser.parse_args()

    print(f"Model-Diff Analysis: {args.trait} @ layer {args.layer}")
    print("=" * 60)

    # Load trait vector
    print(f"\nLoading trait vector: {args.trait}")
    vector = load_trait_vector(args.trait, args.layer, args.experiment)
    print(f"  Vector shape: {vector.shape}")

    # Load activations
    print(f"\nLoading activations...")
    baseline_acts = load_run_activations(args.baseline, args.layer, args.experiment)
    compare_acts = load_run_activations(args.compare, args.layer, args.experiment)
    print(f"  Baseline ({args.baseline}): {len(baseline_acts)} responses")
    print(f"  Compare ({args.compare}): {len(compare_acts)} responses")

    # Find common prompt IDs
    common_ids = set(baseline_acts.keys()) & set(compare_acts.keys())
    if len(common_ids) < len(baseline_acts):
        print(f"  Warning: Only {len(common_ids)} common prompts")

    # Compute response-level scores
    print(f"\nComputing cosine similarities...")
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

    # Paired t-test (same prompts through both models... but different responses)
    # Actually use independent t-test since responses are different
    t_stat, p_value = stats.ttest_ind(compare_arr, baseline_arr)

    # Report
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {args.trait} @ layer {args.layer}")
    print(f"{'=' * 60}")
    print(f"\n  Baseline ({args.baseline}):")
    print(f"    mean = {baseline_mean:+.4f}")
    print(f"    std  = {baseline_std:.4f}")
    print(f"    n    = {len(baseline_arr)}")

    print(f"\n  Compare ({args.compare}):")
    print(f"    mean = {compare_mean:+.4f}")
    print(f"    std  = {compare_std:.4f}")
    print(f"    n    = {len(compare_arr)}")

    print(f"\n  Difference:")
    print(f"    diff        = {diff:+.4f}")
    print(f"    effect size = {effect_size:+.2f} std")
    print(f"    t-statistic = {t_stat:.2f}")
    print(f"    p-value     = {p_value:.2e}")

    # Interpretation
    print(f"\n  Interpretation:")
    if p_value < 0.001 and abs(effect_size) > 0.5:
        direction = "higher" if diff > 0 else "lower"
        print(f"    SIGNIFICANT: {args.compare} has {direction} {args.trait.split('/')[-1]}")
        print(f"    Effect size {abs(effect_size):.1f} std is {'large' if abs(effect_size) > 0.8 else 'medium'}")
    elif p_value < 0.05:
        print(f"    WEAK SIGNAL: Statistically significant but small effect")
    else:
        print(f"    NO SIGNAL: No significant difference detected")

    # Optional: save results
    if args.save:
        results = {
            'trait': args.trait,
            'layer': args.layer,
            'baseline': {
                'name': args.baseline,
                'mean': float(baseline_mean),
                'std': float(baseline_std),
                'n': len(baseline_arr),
            },
            'compare': {
                'name': args.compare,
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

        out_dir = get('experiments.base', experiment=args.experiment) / 'rm_sycophancy' / 'analysis'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"model_diff_{args.trait.replace('/', '_')}_layer{args.layer}.json"

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved results to: {out_path}")

    # Optional: plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5))

            bins = np.linspace(
                min(baseline_arr.min(), compare_arr.min()) - 0.01,
                max(baseline_arr.max(), compare_arr.max()) + 0.01,
                30
            )

            ax.hist(baseline_arr, bins=bins, alpha=0.6, label=f'{args.baseline} (mean={baseline_mean:.3f})')
            ax.hist(compare_arr, bins=bins, alpha=0.6, label=f'{args.compare} (mean={compare_mean:.3f})')

            ax.axvline(baseline_mean, color='C0', linestyle='--', linewidth=2)
            ax.axvline(compare_mean, color='C1', linestyle='--', linewidth=2)

            ax.set_xlabel('Response-level cosine similarity', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'Model Diff: {args.trait.split("/")[-1]} @ layer {args.layer}\n'
                        f'Effect size = {effect_size:.2f} std, p = {p_value:.2e}', fontsize=13)
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            out_dir = get('experiments.base', experiment=args.experiment) / 'assets'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"model_diff_{args.trait.replace('/', '_')}_layer{args.layer}.png"

            plt.savefig(out_path, dpi=150)
            print(f"  Saved plot to: {out_path}")

        except ImportError:
            print("  Warning: matplotlib not installed, skipping plot")


if __name__ == '__main__':
    main()
