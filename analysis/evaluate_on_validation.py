#!/usr/bin/env python3
"""
Evaluate trait vectors on held-out validation data.

This is the REAL test - vectors were extracted from training data,
now we test on completely new prompts to measure generalization.

Evaluation axes:
1. Validation accuracy (generalization)
2. Validation separation (signal on new data)
3. Effect size (Cohen's d)
4. Polarity consistency (train vs val direction)
5. Cross-trait interference (independence)
6. Train vs Val comparison (overfitting detection)

Usage:
    python analysis/evaluate_on_validation.py \
        --experiment gemma_2b_cognitive_nov21 \
        --output results/validation_evaluation.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import fire
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

from utils.paths import get as get_path


@dataclass
class ValidationResult:
    """Results for a single vector on validation data."""
    trait: str
    method: str
    layer: int

    # Validation metrics (the real test)
    val_accuracy: float
    val_separation: float
    val_effect_size: float
    val_p_value: float

    # Training metrics (for comparison)
    train_accuracy: Optional[float] = None
    train_separation: Optional[float] = None

    # Overfitting detection
    accuracy_drop: Optional[float] = None  # train_acc - val_acc
    separation_ratio: Optional[float] = None  # val_sep / train_sep

    # Polarity
    polarity_correct: bool = True
    val_pos_mean: float = 0.0
    val_neg_mean: float = 0.0


def load_validation_activations(experiment: str, trait: str, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load validation activations for a specific layer."""
    val_dir = get_path('extraction.val_activations', experiment=experiment, trait=trait)

    pos_path = val_dir / f"val_pos_layer{layer}.pt"
    neg_path = val_dir / f"val_neg_layer{layer}.pt"

    if not pos_path.exists() or not neg_path.exists():
        raise FileNotFoundError(f"Validation activations not found for layer {layer}")

    pos_acts = torch.load(pos_path, weights_only=True)
    neg_acts = torch.load(neg_path, weights_only=True)

    return pos_acts, neg_acts


def load_training_activations(experiment: str, trait: str, layer: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load training activations for comparison."""
    acts_dir = get_path('extraction.activations', experiment=experiment, trait=trait)
    acts_path = acts_dir / "all_layers.pt"
    metadata_path = acts_dir / "metadata.json"

    if not acts_path.exists():
        return None, None

    all_acts = torch.load(acts_path, weights_only=True)

    # Get split from metadata
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            n_pos = metadata.get('n_examples_pos', all_acts.shape[0] // 2)
    else:
        n_pos = all_acts.shape[0] // 2

    # all_acts shape: [n_examples, n_layers, hidden_dim]
    pos_acts = all_acts[:n_pos, layer, :]
    neg_acts = all_acts[n_pos:, layer, :]

    return pos_acts, neg_acts


def load_vector(experiment: str, trait: str, method: str, layer: int) -> Optional[torch.Tensor]:
    """Load a specific vector."""
    vectors_dir = get_path('extraction.vectors', experiment=experiment, trait=trait)
    path = vectors_dir / f"{method}_layer{layer}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    return None


def compute_metrics(pos_acts: torch.Tensor, neg_acts: torch.Tensor,
                   vector: torch.Tensor, normalize: bool = True) -> Dict[str, float]:
    """Compute evaluation metrics for projections.

    Args:
        pos_acts: Positive activations [n_pos, hidden_dim]
        neg_acts: Negative activations [n_neg, hidden_dim]
        vector: Trait vector [hidden_dim]
        normalize: If True, use cosine similarity (default). If False, raw dot product.
    """
    # Convert all to float32 for numerical stability
    pos_acts = pos_acts.float()
    neg_acts = neg_acts.float()
    vector = vector.float()

    # Normalize for cosine similarity (removes scale effects from vector norm and activation magnitude)
    if normalize:
        vector = vector / vector.norm()
        pos_acts = pos_acts / pos_acts.norm(dim=1, keepdim=True)
        neg_acts = neg_acts / neg_acts.norm(dim=1, keepdim=True)

    # Project (cosine similarity if normalized, dot product otherwise)
    pos_proj = pos_acts @ vector
    neg_proj = neg_acts @ vector

    # Separation
    separation = (pos_proj.mean() - neg_proj.mean()).abs().item()

    # Accuracy (threshold at midpoint)
    threshold = (pos_proj.mean() + neg_proj.mean()) / 2
    pos_correct = (pos_proj > threshold).float().mean().item()
    neg_correct = (neg_proj <= threshold).float().mean().item()
    accuracy = (pos_correct + neg_correct) / 2

    # Effect size (Cohen's d)
    pooled_std = torch.sqrt((pos_proj.std()**2 + neg_proj.std()**2) / 2)
    if pooled_std > 0:
        effect_size = (pos_proj.mean() - neg_proj.mean()).abs() / pooled_std
        effect_size = effect_size.item()
    else:
        effect_size = 0.0

    # Statistical significance
    _, p_value = stats.ttest_ind(pos_proj.cpu().numpy(), neg_proj.cpu().numpy())

    # Polarity (positive examples should score higher for most traits)
    polarity_correct = pos_proj.mean() > neg_proj.mean()

    return {
        'accuracy': accuracy,
        'separation': separation,
        'effect_size': effect_size,
        'p_value': p_value,
        'polarity_correct': bool(polarity_correct),
        'pos_mean': pos_proj.mean().item(),
        'neg_mean': neg_proj.mean().item()
    }


def evaluate_vector_on_validation(
    experiment: str,
    trait: str,
    method: str,
    layer: int,
    normalize: bool = True
) -> Optional[ValidationResult]:
    """Evaluate a single vector on validation data."""
    # Load vector
    vector = load_vector(experiment, trait, method, layer)
    if vector is None:
        return None

    # Load validation activations
    try:
        val_pos, val_neg = load_validation_activations(experiment, trait, layer)
    except FileNotFoundError:
        return None

    # Compute validation metrics
    val_metrics = compute_metrics(val_pos, val_neg, vector, normalize=normalize)

    # Load training activations for comparison
    train_pos, train_neg = load_training_activations(experiment, trait, layer)
    train_metrics = None
    if train_pos is not None:
        train_metrics = compute_metrics(train_pos, train_neg, vector, normalize=normalize)

    # Build result
    result = ValidationResult(
        trait=trait,
        method=method,
        layer=layer,
        val_accuracy=val_metrics['accuracy'],
        val_separation=val_metrics['separation'],
        val_effect_size=val_metrics['effect_size'],
        val_p_value=val_metrics['p_value'],
        polarity_correct=val_metrics['polarity_correct'],
        val_pos_mean=val_metrics['pos_mean'],
        val_neg_mean=val_metrics['neg_mean']
    )

    # Add training comparison if available
    if train_metrics:
        result.train_accuracy = train_metrics['accuracy']
        result.train_separation = train_metrics['separation']
        result.accuracy_drop = train_metrics['accuracy'] - val_metrics['accuracy']
        if train_metrics['separation'] > 0:
            result.separation_ratio = val_metrics['separation'] / train_metrics['separation']

    return result


def compute_cross_trait_matrix(
    experiment: str,
    traits: List[str],
    method: str = "probe",
    layer: int = 16,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Compute cross-trait interference matrix.

    Entry [i,j] = accuracy when projecting trait_i's validation data
                  onto trait_j's vector.

    Diagonal should be high (vectors work for their trait).
    Off-diagonal should be ~50% (random, independent traits).
    """
    n_traits = len(traits)
    matrix = np.zeros((n_traits, n_traits))

    for i, trait_i in enumerate(traits):
        # Load trait_i's validation activations
        try:
            val_pos_i, val_neg_i = load_validation_activations(experiment, trait_i, layer)
        except FileNotFoundError:
            continue

        for j, trait_j in enumerate(traits):
            # Load trait_j's vector
            vector_j = load_vector(experiment, trait_j, method, layer)
            if vector_j is None:
                continue

            # Project trait_i's data onto trait_j's vector
            metrics = compute_metrics(val_pos_i, val_neg_i, vector_j, normalize=normalize)
            matrix[i, j] = metrics['accuracy']

    # Create DataFrame with trait names
    short_names = [t.split('/')[-1][:12] for t in traits]
    df = pd.DataFrame(matrix, index=short_names, columns=short_names)

    return df


def main(experiment: str = "gemma_2b_cognitive_nov21",
         methods: str = "probe,mean_diff,gradient,ica",
         layers: str = None,
         output: str = None,
         no_normalize: bool = False):
    """
    Run validation evaluation on all traits.

    Args:
        experiment: Experiment name
        methods: Comma-separated methods to evaluate
        layers: Comma-separated layers (default: all)
        output: Output JSON file (default: experiments/{experiment}/validation/validation_evaluation.json)
        no_normalize: If True, use raw dot product instead of cosine similarity
    """
    normalize = not no_normalize
    if normalize:
        print("Using cosine similarity (normalized vectors and activations)")
    else:
        print("Using raw dot product (no normalization)")

    # Use paths from config
    exp_dir = get_path('extraction.base', experiment=experiment)
    methods_list = methods.split(",")

    if layers:
        if isinstance(layers, int):
            layers_list = [layers]
        else:
            layers_list = [int(l) for l in str(layers).split(",")]
    else:
        layers_list = list(range(26))

    # Find all traits with validation data
    traits = []
    for category_dir in exp_dir.iterdir():
        if not category_dir.is_dir():
            continue
        for trait_dir in category_dir.iterdir():
            if (trait_dir / "val_activations").exists():
                traits.append(f"{category_dir.name}/{trait_dir.name}")

    print(f"Found {len(traits)} traits with validation data")
    print(f"Evaluating methods: {methods_list}")
    print(f"Evaluating layers: {layers_list}")

    # Evaluate all vectors
    all_results = []

    for trait in tqdm(traits, desc="Traits"):
        for method in methods_list:
            for layer in layers_list:
                result = evaluate_vector_on_validation(
                    experiment, trait, method, layer, normalize=normalize
                )
                if result:
                    all_results.append(asdict(result))

    print(f"\nEvaluated {len(all_results)} vectors")

    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_results)

    if len(df) == 0:
        print("No results to analyze")
        return

    # =========================================================================
    # ANALYSIS 1: Best method per trait (by validation accuracy)
    # =========================================================================
    print("\n" + "="*80)
    print("BEST METHOD PER TRAIT (by validation accuracy)")
    print("="*80)

    best_per_trait = df.loc[df.groupby('trait')['val_accuracy'].idxmax()]
    print(best_per_trait[['trait', 'method', 'layer', 'val_accuracy', 'val_effect_size']].to_string(index=False))

    # =========================================================================
    # ANALYSIS 2: Method comparison (averaged across traits)
    # =========================================================================
    print("\n" + "="*80)
    print("METHOD COMPARISON (averaged across traits and layers)")
    print("="*80)

    method_summary = df.groupby('method').agg({
        'val_accuracy': ['mean', 'std', 'max'],
        'val_separation': ['mean', 'max'],
        'val_effect_size': ['mean', 'max'],
        'polarity_correct': 'mean'
    }).round(3)
    print(method_summary)

    # =========================================================================
    # ANALYSIS 3: Layer comparison (for best method)
    # =========================================================================
    print("\n" + "="*80)
    print("LAYER COMPARISON (probe method, averaged across traits)")
    print("="*80)

    if 'probe' in df['method'].values:
        probe_df = df[df['method'] == 'probe']
        layer_summary = probe_df.groupby('layer').agg({
            'val_accuracy': 'mean',
            'val_effect_size': 'mean',
            'polarity_correct': 'mean'
        }).round(3)

        # Show best layers
        best_layers = layer_summary.nlargest(5, 'val_accuracy')
        print("Top 5 layers by validation accuracy:")
        print(best_layers)

    # =========================================================================
    # ANALYSIS 4: Overfitting detection (train vs val)
    # =========================================================================
    print("\n" + "="*80)
    print("OVERFITTING DETECTION (train accuracy - val accuracy)")
    print("="*80)

    df_with_train = df.dropna(subset=['train_accuracy'])
    if len(df_with_train) > 0:
        overfit_summary = df_with_train.groupby('method').agg({
            'train_accuracy': 'mean',
            'val_accuracy': 'mean',
            'accuracy_drop': 'mean',
            'separation_ratio': 'mean'
        }).round(3)
        print(overfit_summary)

        # Flag severe overfitting
        severe_overfit = df_with_train[df_with_train['accuracy_drop'] > 0.2]
        if len(severe_overfit) > 0:
            print(f"\n⚠️  {len(severe_overfit)} vectors with >20% accuracy drop (overfitting)")

    # =========================================================================
    # ANALYSIS 5: Polarity issues
    # =========================================================================
    print("\n" + "="*80)
    print("POLARITY ISSUES (vectors pointing wrong direction)")
    print("="*80)

    wrong_polarity = df[~df['polarity_correct']]
    if len(wrong_polarity) > 0:
        print(f"⚠️  {len(wrong_polarity)} vectors with inverted polarity:")
        print(wrong_polarity[['trait', 'method', 'layer', 'val_pos_mean', 'val_neg_mean']].head(20).to_string(index=False))
    else:
        print("✅ All vectors have correct polarity")

    # =========================================================================
    # ANALYSIS 6: Cross-trait interference matrix
    # =========================================================================
    print("\n" + "="*80)
    print("CROSS-TRAIT INTERFERENCE MATRIX (probe_layer16)")
    print("="*80)
    print("Diagonal = trait's vector on its own data (should be high)")
    print("Off-diagonal = trait's vector on other trait's data (should be ~50%)")

    cross_matrix = compute_cross_trait_matrix(experiment, traits, "probe", 16, normalize=normalize)
    print(cross_matrix.round(2).to_string())

    # Independence score: how different is diagonal from off-diagonal
    diagonal = np.diag(cross_matrix.values)
    off_diagonal = cross_matrix.values[~np.eye(len(cross_matrix), dtype=bool)]
    independence = diagonal.mean() - off_diagonal.mean()
    print(f"\nIndependence score: {independence:.3f}")
    print(f"  (Diagonal mean: {diagonal.mean():.3f}, Off-diagonal mean: {off_diagonal.mean():.3f})")

    # =========================================================================
    # FINAL RECOMMENDATIONS
    # =========================================================================
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Best overall method
    best_method = df.groupby('method')['val_accuracy'].mean().idxmax()
    best_method_acc = df.groupby('method')['val_accuracy'].mean().max()
    print(f"Best method overall: {best_method} (mean val accuracy: {best_method_acc:.1%})")

    # Best layer for best method
    best_method_df = df[df['method'] == best_method]
    best_layer = best_method_df.groupby('layer')['val_accuracy'].mean().idxmax()
    best_layer_acc = best_method_df.groupby('layer')['val_accuracy'].mean().max()
    print(f"Best layer for {best_method}: {best_layer} (mean val accuracy: {best_layer_acc:.1%})")

    # Per-trait recommendations
    print("\nPer-trait best vectors:")
    for trait in traits:
        trait_df = df[df['trait'] == trait]
        if len(trait_df) > 0:
            best_row = trait_df.loc[trait_df['val_accuracy'].idxmax()]
            short_trait = trait.split('/')[-1][:20]
            print(f"  {short_trait}: {best_row['method']}_layer{int(best_row['layer'])} "
                  f"(val_acc={best_row['val_accuracy']:.1%}, d={best_row['val_effect_size']:.2f})")

    # Save results
    if output:
        output_path = Path(output)
    else:
        output_path = get_path('validation.evaluation', experiment=experiment)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert multi-index to string for JSON compatibility
    method_summary_flat = {}
    for col in method_summary.columns:
        col_name = '_'.join(str(c) for c in col) if isinstance(col, tuple) else str(col)
        method_summary_flat[col_name] = method_summary[col].to_dict()

    results_dict = {
        'all_results': all_results,
        'method_summary': method_summary_flat,
        'cross_trait_matrix': cross_matrix.to_dict(),
        'best_per_trait': best_per_trait.to_dict('records'),
        'recommendations': {
            'best_method': best_method,
            'best_layer': int(best_layer),
            'mean_val_accuracy': float(best_layer_acc)
        }
    }

    # Custom encoder to handle NaN values (not valid JSON)
    def json_serializer(obj):
        if isinstance(obj, float):
            if obj != obj:  # NaN check
                return None
            return obj
        return float(obj)

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=json_serializer)

    print(f"\n✅ Results saved to {output_path}")

    return results_dict


if __name__ == "__main__":
    fire.Fire(main)
