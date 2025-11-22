#!/usr/bin/env python3
"""
Simplified vector evaluation that works with available data format.
Focuses on vector properties and basic separation metrics.
"""

import torch
import numpy as np
import json
from pathlib import Path
import fire
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def load_all_layers_activations(trait_dir):
    """Load activations from all_layers.pt format."""
    acts_path = trait_dir / "activations" / "all_layers.pt"
    metadata_path = trait_dir / "activations" / "metadata.json"

    if not acts_path.exists():
        return None

    # Load the tensor
    all_acts = torch.load(acts_path, weights_only=True)

    # Load metadata to get split
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            n_pos = metadata.get('n_examples_pos', 0)
            n_neg = metadata.get('n_examples_neg', 0)
    else:
        # Assume equal split if no metadata
        n_total = all_acts.shape[0]
        n_pos = n_total // 2
        n_neg = n_total - n_pos

    # Split into positive and negative
    # all_acts shape: [n_examples, n_layers, hidden_dim]
    pos_acts = all_acts[:n_pos]
    neg_acts = all_acts[n_pos:n_pos+n_neg]

    return {'pos': pos_acts, 'neg': neg_acts}

def compute_separation_from_all_layers(all_acts, vector, layer):
    """Compute separation using all_layers format."""
    if all_acts is None:
        return None

    # all_acts structure: {'pos': tensor, 'neg': tensor}
    # Shape: [n_examples, n_layers, hidden_dim]

    if 'pos' not in all_acts or 'neg' not in all_acts:
        return None

    pos_acts = all_acts['pos'][:, layer, :]  # Get specific layer
    neg_acts = all_acts['neg'][:, layer, :]

    # Ensure matching dtypes
    if pos_acts.dtype != vector.dtype:
        if pos_acts.dtype == torch.float16:
            vector = vector.half()
        else:
            pos_acts = pos_acts.float()
            neg_acts = neg_acts.float()

    # Project onto vector
    pos_proj = pos_acts @ vector
    neg_proj = neg_acts @ vector

    # Compute metrics
    separation = (pos_proj.mean() - neg_proj.mean()).abs().item()

    # T-test for statistical significance
    t_stat, p_value = stats.ttest_ind(pos_proj.cpu().numpy(), neg_proj.cpu().numpy())

    # Effect size (Cohen's d)
    pooled_std = torch.sqrt((pos_proj.std()**2 + neg_proj.std()**2) / 2)
    effect_size = abs(pos_proj.mean() - neg_proj.mean()) / pooled_std

    # Classification accuracy (simple threshold)
    threshold = (pos_proj.mean() + neg_proj.mean()) / 2
    pos_correct = (pos_proj > threshold).float().mean()
    neg_correct = (neg_proj <= threshold).float().mean()
    accuracy = (pos_correct + neg_correct) / 2

    return {
        'separation': separation,
        'p_value': p_value,
        'effect_size': effect_size.item(),
        'accuracy': accuracy.item(),
        'pos_mean': pos_proj.mean().item(),
        'neg_mean': neg_proj.mean().item(),
        'pos_std': pos_proj.std().item(),
        'neg_std': neg_proj.std().item()
    }

def analyze_vector_properties(vector):
    """Analyze statistical properties of the vector."""
    props = {
        'norm': vector.norm().item(),
        'mean': vector.mean().item(),
        'std': vector.std().item(),
        'min': vector.min().item(),
        'max': vector.max().item(),
        'sparsity': (vector.abs() < 0.01).float().mean().item(),
        'kurtosis': float(stats.kurtosis(vector.cpu().numpy())),
        'skewness': float(stats.skew(vector.cpu().numpy()))
    }

    # Top-k concentration
    sorted_abs = vector.abs().sort(descending=True)[0]
    top_5_pct = int(len(vector) * 0.05)
    props['top_5pct_mass'] = (sorted_abs[:top_5_pct].sum() / sorted_abs.sum()).item()

    # Effective dimensionality
    cumsum = sorted_abs.cumsum(0)
    total = cumsum[-1]
    props['effective_dim'] = (cumsum < total * 0.9).sum().item() + 1

    return props

def compare_vectors(v1, v2):
    """Compare two vectors."""
    if v1.shape != v2.shape:
        return None

    return {
        'cosine_sim': torch.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item(),
        'correlation': float(np.corrcoef(v1.cpu().numpy(), v2.cpu().numpy())[0, 1]),
        'norm_ratio': (v1.norm() / v2.norm()).item(),
        'rmse': (v1 - v2).pow(2).mean().sqrt().item()
    }

def evaluate_trait(experiment, trait):
    """Evaluate all vectors for a trait."""
    trait_dir = Path(f"experiments/{experiment}/extraction/{trait}")
    vectors_dir = trait_dir / "vectors"

    if not vectors_dir.exists():
        print(f"No vectors found for {trait}")
        return None

    # Load activations
    all_acts = load_all_layers_activations(trait_dir)

    # Analyze each vector
    results = []

    for vector_file in sorted(vectors_dir.glob("*.pt")):
        # Parse filename
        parts = vector_file.stem.split('_')
        if 'layer' in vector_file.stem:
            method = '_'.join(parts[:-1])
            layer = int(parts[-1].replace('layer', ''))
        else:
            method = vector_file.stem
            layer = -1

        # Load vector
        vector = torch.load(vector_file, weights_only=True)

        # Basic properties
        result = {
            'method': method,
            'layer': layer,
            'file': vector_file.name
        }

        # Vector properties
        props = analyze_vector_properties(vector)
        result.update({f'vec_{k}': v for k, v in props.items()})

        # Separation metrics (if we have activations and layer info)
        if all_acts is not None and layer >= 0:
            sep_metrics = compute_separation_from_all_layers(all_acts, vector, layer)
            if sep_metrics:
                result.update({f'sep_{k}': v for k, v in sep_metrics.items()})

        results.append(result)

    return results

def create_summary_table(results):
    """Create a summary table of best vectors."""
    df = pd.DataFrame(results)

    # Filter to vectors with separation metrics
    df_with_sep = df.dropna(subset=['sep_separation'])

    if len(df_with_sep) == 0:
        return df

    # Rank by different criteria
    rankings = {}

    # Best by separation
    rankings['separation'] = df_with_sep.nlargest(5, 'sep_separation')[
        ['method', 'layer', 'sep_separation', 'sep_accuracy', 'vec_norm']
    ]

    # Best by accuracy
    rankings['accuracy'] = df_with_sep.nlargest(5, 'sep_accuracy')[
        ['method', 'layer', 'sep_accuracy', 'sep_separation', 'vec_norm']
    ]

    # Best by effect size
    rankings['effect_size'] = df_with_sep.nlargest(5, 'sep_effect_size')[
        ['method', 'layer', 'sep_effect_size', 'sep_separation', 'vec_norm']
    ]

    # Most interpretable (high top-5% mass)
    rankings['interpretability'] = df_with_sep.nlargest(5, 'vec_top_5pct_mass')[
        ['method', 'layer', 'vec_top_5pct_mass', 'vec_effective_dim', 'sep_separation']
    ]

    return rankings

def plot_layer_profiles(results, output_dir):
    """Plot how metrics vary across layers for each method."""
    if not HAS_MATPLOTLIB:
        return

    df = pd.DataFrame(results)
    df_with_layer = df[df['layer'] >= 0]

    if len(df_with_layer) == 0:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by method
    methods = df_with_layer['method'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for method in methods:
        method_df = df_with_layer[df_with_layer['method'] == method].sort_values('layer')

        # Separation across layers
        if 'sep_separation' in method_df.columns:
            axes[0].plot(method_df['layer'], method_df['sep_separation'],
                        marker='o', label=method, alpha=0.7)

        # Accuracy across layers
        if 'sep_accuracy' in method_df.columns:
            axes[1].plot(method_df['layer'], method_df['sep_accuracy'],
                        marker='s', label=method, alpha=0.7)

        # Vector norm across layers
        axes[2].plot(method_df['layer'], method_df['vec_norm'],
                    marker='^', label=method, alpha=0.7)

        # Effect size across layers
        if 'sep_effect_size' in method_df.columns:
            axes[3].plot(method_df['layer'], method_df['sep_effect_size'],
                        marker='d', label=method, alpha=0.7)

    axes[0].set_title('Separation Score by Layer')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Separation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Accuracy by Layer')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title('Vector Norm by Layer')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('L2 Norm')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')

    axes[3].set_title('Effect Size by Layer')
    axes[3].set_xlabel('Layer')
    axes[3].set_ylabel("Cohen's d")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'layer_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()

def main(experiment="gemma_2b_cognitive_nov21",
         trait=None,
         output_dir="results"):
    """
    Evaluate trait vectors with available data.

    Args:
        experiment: Experiment name
        trait: Specific trait (e.g., "cognitive_state/context") or None for all
        output_dir: Output directory for results
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get traits to evaluate
    if trait:
        traits = [trait]
    else:
        # Find all traits with vectors
        exp_dir = Path(f"experiments/{experiment}/extraction")
        traits = []
        for category_dir in exp_dir.iterdir():
            if category_dir.is_dir():
                for trait_dir in category_dir.iterdir():
                    if (trait_dir / "vectors").exists():
                        traits.append(f"{category_dir.name}/{trait_dir.name}")

    all_results = {}

    for trait_name in traits:
        print(f"\n{'='*80}")
        print(f"Evaluating: {trait_name}")
        print('='*80)

        results = evaluate_trait(experiment, trait_name)

        if not results:
            print(f"No results for {trait_name}")
            continue

        all_results[trait_name] = results

        # Create summary
        rankings = create_summary_table(results)

        # Print best vectors
        print(f"\n📊 TOP VECTORS BY SEPARATION:")
        if 'separation' in rankings:
            print(rankings['separation'].to_string(index=False))

        print(f"\n🎯 TOP VECTORS BY ACCURACY:")
        if 'accuracy' in rankings:
            print(rankings['accuracy'].to_string(index=False))

        print(f"\n📈 TOP VECTORS BY EFFECT SIZE:")
        if 'effect_size' in rankings:
            print(rankings['effect_size'].to_string(index=False))

        # Method comparison
        df = pd.DataFrame(results)
        print(f"\n📋 METHOD SUMMARY:")
        method_summary = df.groupby('method').agg({
            'vec_norm': ['mean', 'std', 'min', 'max'],
            'sep_separation': ['mean', 'max'] if 'sep_separation' in df.columns else ['count'],
            'sep_accuracy': ['mean', 'max'] if 'sep_accuracy' in df.columns else ['count']
        }).round(3)
        print(method_summary)

        # Save detailed results
        trait_output = output_dir / f"{trait_name.replace('/', '_')}_evaluation.json"
        trait_output.parent.mkdir(parents=True, exist_ok=True)
        with open(trait_output, 'w') as f:
            json.dump(results, f, indent=2, default=float)

        # Plot layer profiles
        if HAS_MATPLOTLIB:
            plot_output = output_dir / trait_name.replace('/', '_')
            plot_layer_profiles(results, plot_output)
            print(f"\n📊 Plots saved to {plot_output}/")
        else:
            print("\n📊 Plots skipped (matplotlib not installed)")

    # Overall summary
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print('='*80)

        best_per_trait = []
        for trait_name, results in all_results.items():
            df = pd.DataFrame(results)
            if 'sep_separation' in df.columns:
                best = df.loc[df['sep_separation'].idxmax()]
                best_per_trait.append({
                    'trait': trait_name.split('/')[-1][:20],
                    'method': best['method'],
                    'layer': int(best['layer']),
                    'separation': f"{best['sep_separation']:.2f}",
                    'accuracy': f"{best['sep_accuracy']:.1%}",
                    'norm': f"{best['vec_norm']:.1f}"
                })

        if best_per_trait:
            summary_df = pd.DataFrame(best_per_trait)
            print(summary_df.to_string(index=False))

    print(f"\n✅ Results saved to {output_dir}/")

    return all_results

if __name__ == "__main__":
    fire.Fire(main)