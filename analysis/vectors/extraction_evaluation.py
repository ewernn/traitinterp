#!/usr/bin/env python3
"""
Evaluate extracted vectors on held-out validation data.

Input:
    - experiments/{experiment}/extraction/{trait}/vectors/*.pt
    - experiments/{experiment}/extraction/{trait}/val_activations/*.pt

Output:
    - experiments/{experiment}/extraction/extraction_evaluation.json

Usage:
    python analysis/vectors/extraction_evaluation.py --experiment my_experiment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
import fire
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

from utils.paths import get as get_path
from traitlens.metrics import (
    evaluate_vector,
    accuracy as compute_accuracy,
)
from sklearn.metrics import roc_auc_score


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

    # Vector properties
    vector_norm: float = 0.0
    vector_sparsity: float = 0.0  # % of components < 0.01

    # Distribution properties
    val_pos_std: float = 0.0  # Std of positive projections
    val_neg_std: float = 0.0  # Std of negative projections
    overlap_coefficient: float = 0.0  # Distribution overlap
    separation_margin: float = 0.0  # (pos_mean - pos_std) - (neg_mean + neg_std)

    # Additional metrics
    val_auc_roc: float = 0.0  # Threshold-independent classification quality


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

    # Try loading from combined file first (legacy format)
    all_layers_path = acts_dir / "all_layers.pt"
    if all_layers_path.exists():
        all_acts = torch.load(all_layers_path, weights_only=True)
        metadata_path = acts_dir / "metadata.json"

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

    # Try loading from per-layer files (current format)
    pos_layer_path = acts_dir / f"pos_layer{layer}.pt"
    neg_layer_path = acts_dir / f"neg_layer{layer}.pt"

    if not pos_layer_path.exists() or not neg_layer_path.exists():
        return None, None

    pos_acts = torch.load(pos_layer_path, weights_only=True)
    neg_acts = torch.load(neg_layer_path, weights_only=True)

    return pos_acts, neg_acts


def load_vector(experiment: str, trait: str, method: str, layer: int) -> Optional[torch.Tensor]:
    """Load a specific vector."""
    vectors_dir = get_path('extraction.vectors', experiment=experiment, trait=trait)
    path = vectors_dir / f"{method}_layer{layer}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    return None


def compute_vector_properties(vector: torch.Tensor) -> Dict:
    """
    Compute properties of a vector.

    Args:
        vector: [hidden_dim] tensor

    Returns:
        Dictionary with norm, sparsity
    """
    vector_np = vector.float().cpu().numpy()

    return {
        'vector_norm': float(np.linalg.norm(vector_np)),
        'vector_sparsity': float(np.mean(np.abs(vector_np) < 0.01)),
    }


def compute_distribution_properties(
    pos_projections: torch.Tensor,
    neg_projections: torch.Tensor,
    pos_mean: float,
    neg_mean: float
) -> Dict[str, float]:
    """
    Compute properties of projection score distributions.

    Args:
        pos_projections: [n_pos] tensor of positive projection scores
        neg_projections: [n_neg] tensor of negative projection scores
        pos_mean: Mean of positive projections
        neg_mean: Mean of negative projections

    Returns:
        Dictionary with std, overlap, margin metrics
    """
    pos_std = float(pos_projections.std())
    neg_std = float(neg_projections.std())

    # Overlap coefficient: estimate using normal approximation
    # If pos/neg are well-separated Gaussians, this measures overlap
    if pos_std > 0 and neg_std > 0:
        # Distance between means in units of pooled std
        pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
        z_score = abs(pos_mean - neg_mean) / (pooled_std + 1e-8)
        # Rough overlap estimate (1 - z_score normalized to 0-1)
        overlap = max(0, 1 - z_score / 4.0)  # z=4 → no overlap
    else:
        overlap = 0.5

    # Separation margin: gap between distributions
    # Positive = good separation, negative = overlap
    margin = (pos_mean - pos_std) - (neg_mean + neg_std)

    return {
        'val_pos_std': pos_std,
        'val_neg_std': neg_std,
        'overlap_coefficient': float(overlap),
        'separation_margin': float(margin)
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

    # Compute validation metrics using traitlens
    val_metrics = evaluate_vector(val_pos, val_neg, vector, normalize=normalize)

    # Compute vector properties
    vector_props = compute_vector_properties(vector)

    # Compute projection scores for distribution properties
    # Note: evaluate_vector already does this internally, but we need the raw projections
    # Convert to float32 for consistent computation (activations may be bfloat16)
    vector_f32 = vector.float()
    val_pos_f32 = val_pos.float()
    val_neg_f32 = val_neg.float()

    if normalize:
        vec_norm = vector_f32 / (vector_f32.norm() + 1e-8)
        pos_norm = val_pos_f32 / (val_pos_f32.norm(dim=1, keepdim=True) + 1e-8)
        neg_norm = val_neg_f32 / (val_neg_f32.norm(dim=1, keepdim=True) + 1e-8)
        pos_projections = pos_norm @ vec_norm
        neg_projections = neg_norm @ vec_norm
    else:
        pos_projections = val_pos_f32 @ vector_f32
        neg_projections = val_neg_f32 @ vector_f32

    # Compute distribution properties
    dist_props = compute_distribution_properties(
        pos_projections,
        neg_projections,
        val_metrics['pos_mean'],
        val_metrics['neg_mean']
    )

    # Compute AUC-ROC (threshold-independent classification quality)
    all_projections = torch.cat([pos_projections, neg_projections]).cpu().numpy()
    all_labels = np.concatenate([np.ones(len(pos_projections)), np.zeros(len(neg_projections))])
    try:
        auc = roc_auc_score(all_labels, all_projections)
    except ValueError:
        auc = 0.5  # Fallback if AUC can't be computed

    # Load training activations for comparison
    train_pos, train_neg = load_training_activations(experiment, trait, layer)
    train_metrics = None
    if train_pos is not None:
        train_metrics = evaluate_vector(train_pos, train_neg, vector, normalize=normalize)

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
        val_neg_mean=val_metrics['neg_mean'],
        val_auc_roc=auc,
        **vector_props,
        **dist_props
    )

    # Add training comparison if available
    if train_metrics:
        result.train_accuracy = train_metrics['accuracy']
        result.train_separation = train_metrics['separation']
        result.accuracy_drop = train_metrics['accuracy'] - val_metrics['accuracy']
        if train_metrics['separation'] > 0:
            result.separation_ratio = val_metrics['separation'] / train_metrics['separation']

    return result


def compute_best_vector_similarity(
    experiment: str,
    traits: List[str],
    all_results: List[Dict],
    metric: str = 'val_accuracy'
) -> pd.DataFrame:
    """
    Compute cosine similarity matrix between best vectors across traits.

    Args:
        experiment: Experiment name
        traits: List of trait names
        all_results: All validation results
        metric: Metric to use for selecting best vector ('val_accuracy', 'val_effect_size', etc.)

    Returns:
        DataFrame with similarity matrix (traits × traits)
    """
    # Find best vector for each trait
    df = pd.DataFrame(all_results)
    best_vectors = {}

    for trait in traits:
        trait_results = df[df['trait'] == trait]
        if len(trait_results) == 0:
            continue

        # Get best by metric
        best_row = trait_results.loc[trait_results[metric].idxmax()]
        vector = load_vector(experiment, trait, best_row['method'], int(best_row['layer']))

        if vector is not None:
            best_vectors[trait] = vector

    # Compute pairwise similarities
    trait_list = list(best_vectors.keys())
    n = len(trait_list)
    matrix = np.zeros((n, n))

    for i, trait_i in enumerate(trait_list):
        vec_i = best_vectors[trait_i].float()
        vec_i_norm = vec_i / (vec_i.norm() + 1e-8)

        for j, trait_j in enumerate(trait_list):
            vec_j = best_vectors[trait_j].float()
            vec_j_norm = vec_j / (vec_j.norm() + 1e-8)
            similarity = float((vec_i_norm @ vec_j_norm).item())
            matrix[i, j] = similarity

    # Create DataFrame
    short_names = [t.split('/')[-1][:15] for t in trait_list]
    similarity_df = pd.DataFrame(matrix, index=short_names, columns=short_names)

    return similarity_df


def main(experiment: str,
         methods: str = "mean_diff,probe,ica,gradient,pca_diff,random_baseline",
         layers: str = None,
         output: str = None,
         no_normalize: bool = False):
    """
    Run validation evaluation on all traits.

    Args:
        experiment: Experiment name
        methods: Comma-separated methods to evaluate
        layers: Comma-separated layers (default: all)
        output: Output JSON file (default: experiments/{experiment}/extraction/extraction_evaluation.json)
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
    # ANALYSIS 1: Best method per trait (by combined score)
    # =========================================================================
    print("\n" + "="*80)
    print("BEST METHOD PER TRAIT (by combined score)")
    print("="*80)

    # Compute combined score: (accuracy + norm_effect + (1 - accuracy_drop)) / 3 * polarity
    # First, get max effect size per trait for normalization
    max_effect_per_trait = df.groupby('trait')['val_effect_size'].transform('max')
    df['norm_effect_size'] = df['val_effect_size'] / max_effect_per_trait.replace(0, 1)
    df['accuracy_drop'] = df['accuracy_drop'].fillna(0)
    df['polarity_multiplier'] = df['polarity_correct'].astype(float)

    # Combined score with equal weights
    df['combined_score'] = (
        (df['val_accuracy'] + df['norm_effect_size'] + (1 - df['accuracy_drop'])) / 3
    ) * df['polarity_multiplier']

    best_per_trait = df.loc[df.groupby('trait')['combined_score'].idxmax()]
    print(best_per_trait[['trait', 'method', 'layer', 'combined_score', 'val_accuracy', 'val_effect_size']].to_string(index=False))

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

    # =========================================================================
    # ANALYSIS 7: Best-vector cross-trait similarity
    # =========================================================================
    print("\n" + "="*80)
    print("BEST-VECTOR CROSS-TRAIT SIMILARITY")
    print("="*80)
    print("Comparing best vectors across different traits...")

    best_vector_similarity = compute_best_vector_similarity(experiment, traits, all_results, metric='val_accuracy')
    print(best_vector_similarity.round(3).to_string())

    # Save results
    if output:
        output_path = Path(output)
    else:
        output_path = get_path('extraction_eval.evaluation', experiment=experiment)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert df to records (includes combined_score computed above)
    all_results_with_score = df.to_dict('records')

    results_dict = {
        'all_results': all_results_with_score,
        'best_per_trait': best_per_trait.to_dict('records'),
        'best_vector_similarity': best_vector_similarity.to_dict(),
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
