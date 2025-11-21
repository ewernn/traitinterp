#!/usr/bin/env python3
"""
Comprehensive trait vector evaluation across multiple quality metrics.

This script evaluates which extraction method (mean_diff, probe, ICA, gradient)
at which layer (0-25) produces the "best" vector for each trait.

Usage:
    python analysis/evaluate_vector_quality.py \
        --experiment gemma_2b_cognitive_nov21 \
        --trait cognitive_state/uncertainty_expression \
        --output results/vector_evaluation.json
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import fire
from dataclasses import dataclass, asdict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
import pandas as pd
from tqdm import tqdm

@dataclass
class VectorEvaluation:
    """Comprehensive evaluation metrics for a trait vector."""

    # Basic info
    method: str
    layer: int

    # 1. SEPARATION METRICS
    separation_score: float          # Distance between pos/neg centroids
    train_accuracy: float            # Classification accuracy on training data
    cross_val_accuracy: float        # 5-fold cross-validation accuracy
    auc_score: float                # Area under ROC curve

    # 2. CROSS-DISTRIBUTION (if available)
    cross_dist_inst_to_nat: Optional[float] = None   # Instruction → Natural
    cross_dist_nat_to_inst: Optional[float] = None   # Natural → Instruction
    cross_dist_consistency: Optional[float] = None   # How similar both directions

    # 3. POLARITY VALIDATION
    polarity_correct: bool = False        # Whether polarity matches expected
    polarity_confidence: float = 0.0      # Statistical confidence in polarity
    polarity_effect_size: float = 0.0     # Cohen's d for polarity difference

    # 4. STATISTICAL PROPERTIES
    vector_norm: float = 0.0             # L2 norm of vector
    vector_sparsity: float = 0.0         # Fraction of near-zero components
    effective_rank: float = 0.0          # Effective dimensionality (via SVD)
    kurtosis: float = 0.0                # Distribution shape (outlier detection)

    # 5. STABILITY METRICS
    bootstrap_std: float = 0.0           # Std dev across bootstrap samples
    noise_robustness: float = 0.0        # Performance with added noise
    subsample_stability: float = 0.0     # Performance with 50% data

    # 6. LAYER CONSISTENCY
    adjacent_layer_similarity: float = 0.0   # Cosine sim with adjacent layers
    layer_evolution_smoothness: float = 0.0  # Smoothness of metrics across layers
    is_peak_layer: bool = False             # Whether this is the best layer

    # 7. ORTHOGONALITY (to other traits)
    mean_orthogonality: float = 0.0      # Average cosine sim with other traits
    max_correlation: float = 0.0         # Highest correlation with another trait
    independence_score: float = 0.0      # Composite independence metric

    # 8. INTERPRETABILITY PROXIES
    top_k_mass: float = 0.0             # Mass in top 5% of components
    gradient_smoothness: float = 0.0     # Smoothness of vector components
    cluster_quality: float = 0.0         # How well vector clusters data

    # 9. COMPOSITE SCORES
    overall_score: float = 0.0           # Weighted combination of metrics
    reliability_score: float = 0.0       # Focus on stability/robustness
    performance_score: float = 0.0       # Focus on separation/accuracy

    def to_dict(self):
        return asdict(self)


class VectorEvaluator:
    """Evaluate trait vectors across multiple quality metrics."""

    def __init__(self, experiment: str, trait: str):
        """
        Initialize evaluator for a specific trait.

        Args:
            experiment: Experiment name (e.g., "gemma_2b_cognitive_nov21")
            trait: Trait path (e.g., "cognitive_state/uncertainty_expression")
        """
        self.experiment = experiment
        self.trait = trait
        self.base_dir = Path(f"experiments/{experiment}/extraction/{trait}")
        self.vectors_dir = self.base_dir / "vectors"
        self.activations_dir = self.base_dir / "activations"

        # Load activations if available
        self.pos_acts = self._load_activations("pos")
        self.neg_acts = self._load_activations("neg")

    def _load_activations(self, valence: str) -> Optional[Dict[int, torch.Tensor]]:
        """Load activations for positive or negative examples."""
        acts = {}
        for layer in range(26):  # Gemma 2B has 26 layers
            path = self.activations_dir / f"{valence}_layer{layer}.pt"
            if path.exists():
                acts[layer] = torch.load(path, weights_only=True)
        return acts if acts else None

    def _load_vector(self, method: str, layer: int) -> Optional[torch.Tensor]:
        """Load a specific vector."""
        path = self.vectors_dir / f"{method}_layer{layer}.pt"
        if path.exists():
            return torch.load(path, weights_only=True)
        return None

    def _load_cross_dist_results(self) -> Optional[Dict]:
        """Load cross-distribution test results if available."""
        cross_dist_path = Path(f"experiments/{self.experiment}/validation/cross_distribution/{self.trait}")
        results = {}

        # Try to load different test combinations
        for test_type in ["inst_to_nat", "nat_to_inst", "inst_to_inst", "nat_to_nat"]:
            test_file = cross_dist_path / f"{test_type}_results.json"
            if test_file.exists():
                with open(test_file) as f:
                    results[test_type] = json.load(f)

        return results if results else None

    def evaluate_separation(self, vector: torch.Tensor, pos_acts: torch.Tensor,
                           neg_acts: torch.Tensor) -> Dict[str, float]:
        """Evaluate separation metrics."""
        # Project activations
        pos_proj = pos_acts @ vector
        neg_proj = neg_acts @ vector

        # Separation score (distance between means)
        separation = (pos_proj.mean() - neg_proj.mean()).abs().item()

        # Train accuracy with logistic regression
        X = torch.cat([pos_acts, neg_acts])
        y = torch.cat([torch.ones(len(pos_acts)), torch.zeros(len(neg_acts))])

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)
        train_acc = clf.score(X, y)

        # Cross-validation accuracy
        cv_scores = cross_val_score(clf, X, y, cv=5)
        cv_acc = cv_scores.mean()

        # AUC score
        y_scores = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_scores)

        return {
            "separation_score": separation,
            "train_accuracy": train_acc,
            "cross_val_accuracy": cv_acc,
            "auc_score": auc
        }

    def evaluate_polarity(self, vector: torch.Tensor, pos_acts: torch.Tensor,
                         neg_acts: torch.Tensor, expected_polarity: str = "positive") -> Dict[str, float]:
        """Evaluate polarity correctness."""
        pos_proj = pos_acts @ vector
        neg_proj = neg_acts @ vector

        # Check if polarity is correct
        if expected_polarity == "positive":
            polarity_correct = pos_proj.mean() > neg_proj.mean()
        else:
            polarity_correct = pos_proj.mean() < neg_proj.mean()

        # Statistical test for polarity
        t_stat, p_value = stats.ttest_ind(pos_proj, neg_proj)
        polarity_confidence = 1 - p_value

        # Effect size (Cohen's d)
        pooled_std = torch.sqrt((pos_proj.std()**2 + neg_proj.std()**2) / 2)
        effect_size = abs(pos_proj.mean() - neg_proj.mean()) / pooled_std

        return {
            "polarity_correct": polarity_correct,
            "polarity_confidence": polarity_confidence.item(),
            "polarity_effect_size": effect_size.item()
        }

    def evaluate_statistical_properties(self, vector: torch.Tensor) -> Dict[str, float]:
        """Evaluate statistical properties of the vector."""
        # L2 norm
        norm = vector.norm().item()

        # Sparsity (fraction of components < 1% of max)
        threshold = vector.abs().max() * 0.01
        sparsity = (vector.abs() < threshold).float().mean().item()

        # Effective rank via SVD
        if vector.dim() == 1:
            # For 1D vector, compute based on cumulative variance
            sorted_vals = vector.abs().sort(descending=True)[0]
            cumsum = sorted_vals.cumsum(0)
            total = cumsum[-1]
            effective_rank = (cumsum < total * 0.9).sum().item() + 1
        else:
            U, S, V = torch.svd(vector.unsqueeze(0))
            effective_rank = (S > S[0] * 0.01).sum().item()

        # Kurtosis (outlier detection)
        kurtosis = stats.kurtosis(vector.cpu().numpy())

        return {
            "vector_norm": norm,
            "vector_sparsity": sparsity,
            "effective_rank": effective_rank,
            "kurtosis": float(kurtosis)
        }

    def evaluate_stability(self, vector: torch.Tensor, pos_acts: torch.Tensor,
                          neg_acts: torch.Tensor, n_bootstrap: int = 10) -> Dict[str, float]:
        """Evaluate stability metrics."""
        bootstrap_scores = []

        # Bootstrap stability
        for _ in range(n_bootstrap):
            # Sample with replacement
            pos_idx = torch.randint(0, len(pos_acts), (len(pos_acts),))
            neg_idx = torch.randint(0, len(neg_acts), (len(neg_acts),))

            pos_sample = pos_acts[pos_idx]
            neg_sample = neg_acts[neg_idx]

            # Compute separation on bootstrap sample
            sep = ((pos_sample @ vector).mean() - (neg_sample @ vector).mean()).abs()
            bootstrap_scores.append(sep.item())

        bootstrap_std = np.std(bootstrap_scores)

        # Noise robustness
        noise_level = 0.1 * vector.norm()
        noisy_vector = vector + torch.randn_like(vector) * noise_level

        clean_sep = ((pos_acts @ vector).mean() - (neg_acts @ vector).mean()).abs()
        noisy_sep = ((pos_acts @ noisy_vector).mean() - (neg_acts @ noisy_vector).mean()).abs()
        noise_robustness = noisy_sep / clean_sep

        # Subsample stability (50% of data)
        half_size = len(pos_acts) // 2
        pos_half = pos_acts[:half_size]
        neg_half = neg_acts[:half_size]

        full_sep = ((pos_acts @ vector).mean() - (neg_acts @ vector).mean()).abs()
        half_sep = ((pos_half @ vector).mean() - (neg_half @ vector).mean()).abs()
        subsample_stability = half_sep / full_sep

        return {
            "bootstrap_std": bootstrap_std,
            "noise_robustness": noise_robustness.item(),
            "subsample_stability": subsample_stability.item()
        }

    def evaluate_layer_consistency(self, method: str, layer: int) -> Dict[str, float]:
        """Evaluate consistency with adjacent layers."""
        current_vec = self._load_vector(method, layer)
        if current_vec is None:
            return {
                "adjacent_layer_similarity": 0.0,
                "layer_evolution_smoothness": 0.0,
                "is_peak_layer": False
            }

        similarities = []

        # Check adjacent layers
        for adj_layer in [layer-1, layer+1]:
            if 0 <= adj_layer < 26:
                adj_vec = self._load_vector(method, adj_layer)
                if adj_vec is not None:
                    # Cosine similarity
                    sim = torch.cosine_similarity(current_vec.unsqueeze(0),
                                                 adj_vec.unsqueeze(0))
                    similarities.append(sim.item())

        adj_similarity = np.mean(similarities) if similarities else 0.0

        # Check if this is peak layer (best separation)
        all_separations = []
        for l in range(26):
            vec = self._load_vector(method, l)
            if vec is not None and l in self.pos_acts and l in self.neg_acts:
                sep = ((self.pos_acts[l] @ vec).mean() -
                      (self.neg_acts[l] @ vec).mean()).abs().item()
                all_separations.append((l, sep))

        is_peak = False
        if all_separations:
            best_layer = max(all_separations, key=lambda x: x[1])[0]
            is_peak = (best_layer == layer)

        # Evolution smoothness (variance of separations across layers)
        if len(all_separations) > 1:
            seps = [s for _, s in all_separations]
            smoothness = 1.0 / (1.0 + np.std(seps))
        else:
            smoothness = 0.0

        return {
            "adjacent_layer_similarity": adj_similarity,
            "layer_evolution_smoothness": smoothness,
            "is_peak_layer": is_peak
        }

    def evaluate_orthogonality(self, vector: torch.Tensor, other_traits: List[str]) -> Dict[str, float]:
        """Evaluate orthogonality to other trait vectors."""
        correlations = []

        for other_trait in other_traits:
            if other_trait == self.trait:
                continue

            # Try to load the same method/layer for other trait
            other_path = Path(f"experiments/{self.experiment}/extraction/{other_trait}/vectors")
            if other_path.exists():
                # Find matching vector file
                for vec_file in other_path.glob("*.pt"):
                    other_vec = torch.load(vec_file, weights_only=True)
                    if other_vec.shape == vector.shape:
                        corr = torch.cosine_similarity(vector.unsqueeze(0),
                                                      other_vec.unsqueeze(0))
                        correlations.append(abs(corr.item()))
                        break

        if correlations:
            mean_orthogonality = 1.0 - np.mean(correlations)
            max_correlation = max(correlations)
            independence_score = 1.0 - max_correlation
        else:
            mean_orthogonality = 1.0
            max_correlation = 0.0
            independence_score = 1.0

        return {
            "mean_orthogonality": mean_orthogonality,
            "max_correlation": max_correlation,
            "independence_score": independence_score
        }

    def evaluate_interpretability(self, vector: torch.Tensor) -> Dict[str, float]:
        """Evaluate interpretability proxies."""
        # Top-k mass (concentration in few components)
        sorted_abs = vector.abs().sort(descending=True)[0]
        top_k = int(len(vector) * 0.05)  # Top 5%
        top_k_mass = sorted_abs[:top_k].sum() / sorted_abs.sum()

        # Gradient smoothness (adjacent components similarity)
        diffs = vector[1:] - vector[:-1]
        gradient_smoothness = 1.0 / (1.0 + diffs.norm().item())

        # Cluster quality (simplified: ratio of largest component to mean)
        cluster_quality = sorted_abs[0] / sorted_abs.mean()

        return {
            "top_k_mass": top_k_mass.item(),
            "gradient_smoothness": gradient_smoothness,
            "cluster_quality": cluster_quality.item()
        }

    def compute_composite_scores(self, eval_result: VectorEvaluation) -> Dict[str, float]:
        """Compute weighted composite scores."""
        # Performance score (separation + accuracy)
        performance = (
            eval_result.separation_score * 0.3 +
            eval_result.train_accuracy * 100 * 0.3 +
            eval_result.cross_val_accuracy * 100 * 0.3 +
            eval_result.auc_score * 100 * 0.1
        )

        # Reliability score (stability + robustness)
        reliability = (
            (1.0 - eval_result.bootstrap_std) * 33.3 +
            eval_result.noise_robustness * 33.3 +
            eval_result.subsample_stability * 33.3
        )

        # Overall score (weighted combination)
        overall = (
            performance * 0.4 +
            reliability * 0.2 +
            eval_result.independence_score * 100 * 0.1 +
            eval_result.mean_orthogonality * 100 * 0.1 +
            (eval_result.polarity_confidence * 100 if eval_result.polarity_correct else 0) * 0.2
        )

        return {
            "performance_score": performance,
            "reliability_score": reliability,
            "overall_score": overall
        }

    def evaluate_vector(self, method: str, layer: int,
                       other_traits: Optional[List[str]] = None) -> Optional[VectorEvaluation]:
        """Evaluate a single vector comprehensively."""
        vector = self._load_vector(method, layer)
        if vector is None or layer not in self.pos_acts or layer not in self.neg_acts:
            return None

        pos_acts = self.pos_acts[layer]
        neg_acts = self.neg_acts[layer]

        # Initialize evaluation
        eval_result = VectorEvaluation(method=method, layer=layer)

        # 1. Separation metrics
        sep_metrics = self.evaluate_separation(vector, pos_acts, neg_acts)
        for key, value in sep_metrics.items():
            setattr(eval_result, key, value)

        # 2. Polarity validation
        polarity_metrics = self.evaluate_polarity(vector, pos_acts, neg_acts)
        for key, value in polarity_metrics.items():
            setattr(eval_result, key, value)

        # 3. Statistical properties
        stat_metrics = self.evaluate_statistical_properties(vector)
        for key, value in stat_metrics.items():
            setattr(eval_result, key, value)

        # 4. Stability metrics
        stability_metrics = self.evaluate_stability(vector, pos_acts, neg_acts)
        for key, value in stability_metrics.items():
            setattr(eval_result, key, value)

        # 5. Layer consistency
        layer_metrics = self.evaluate_layer_consistency(method, layer)
        for key, value in layer_metrics.items():
            setattr(eval_result, key, value)

        # 6. Orthogonality (if other traits provided)
        if other_traits:
            ortho_metrics = self.evaluate_orthogonality(vector, other_traits)
            for key, value in ortho_metrics.items():
                setattr(eval_result, key, value)

        # 7. Interpretability
        interp_metrics = self.evaluate_interpretability(vector)
        for key, value in interp_metrics.items():
            setattr(eval_result, key, value)

        # 8. Cross-distribution (if available)
        cross_dist = self._load_cross_dist_results()
        if cross_dist:
            if f"{method}_layer{layer}" in cross_dist.get("inst_to_nat", {}):
                eval_result.cross_dist_inst_to_nat = cross_dist["inst_to_nat"][f"{method}_layer{layer}"]
            if f"{method}_layer{layer}" in cross_dist.get("nat_to_inst", {}):
                eval_result.cross_dist_nat_to_inst = cross_dist["nat_to_inst"][f"{method}_layer{layer}"]
            if eval_result.cross_dist_inst_to_nat and eval_result.cross_dist_nat_to_inst:
                eval_result.cross_dist_consistency = 1.0 - abs(
                    eval_result.cross_dist_inst_to_nat - eval_result.cross_dist_nat_to_inst
                )

        # 9. Composite scores
        composite = self.compute_composite_scores(eval_result)
        for key, value in composite.items():
            setattr(eval_result, key, value)

        return eval_result

    def find_best_vector(self, other_traits: Optional[List[str]] = None,
                        methods: Optional[List[str]] = None,
                        layers: Optional[List[int]] = None) -> Dict:
        """Find the best vector across all methods and layers."""
        if methods is None:
            methods = ["mean_diff", "probe", "ica", "gradient"]
        if layers is None:
            layers = list(range(26))  # All layers for Gemma 2B

        all_results = []

        # Evaluate all combinations
        for method in tqdm(methods, desc="Methods"):
            for layer in tqdm(layers, desc="Layers", leave=False):
                result = self.evaluate_vector(method, layer, other_traits)
                if result:
                    all_results.append(result)

        if not all_results:
            return {}

        # Find best by different criteria
        best_overall = max(all_results, key=lambda x: x.overall_score)
        best_performance = max(all_results, key=lambda x: x.performance_score)
        best_reliability = max(all_results, key=lambda x: x.reliability_score)
        best_separation = max(all_results, key=lambda x: x.separation_score)
        best_cross_dist = max([r for r in all_results if r.cross_dist_consistency],
                             key=lambda x: x.cross_dist_consistency,
                             default=None)

        # Create summary
        summary = {
            "trait": self.trait,
            "total_evaluated": len(all_results),
            "best_overall": best_overall.to_dict(),
            "best_performance": best_performance.to_dict(),
            "best_reliability": best_reliability.to_dict(),
            "best_separation": best_separation.to_dict(),
            "best_cross_distribution": best_cross_dist.to_dict() if best_cross_dist else None,
            "all_results": [r.to_dict() for r in all_results]
        }

        # Add rankings
        summary["rankings"] = {
            "by_overall_score": self._get_top_k(all_results, "overall_score", k=5),
            "by_performance": self._get_top_k(all_results, "performance_score", k=5),
            "by_reliability": self._get_top_k(all_results, "reliability_score", k=5),
            "by_separation": self._get_top_k(all_results, "separation_score", k=5),
            "by_cross_dist": self._get_top_k(
                [r for r in all_results if r.cross_dist_consistency],
                "cross_dist_consistency", k=5
            ) if any(r.cross_dist_consistency for r in all_results) else None
        }

        return summary

    def _get_top_k(self, results: List[VectorEvaluation], metric: str, k: int = 5) -> List[Dict]:
        """Get top-k results by a specific metric."""
        sorted_results = sorted(results, key=lambda x: getattr(x, metric), reverse=True)
        return [
            {
                "rank": i + 1,
                "method": r.method,
                "layer": r.layer,
                metric: getattr(r, metric)
            }
            for i, r in enumerate(sorted_results[:k])
        ]


def main(experiment: str,
         trait: str = None,
         output: str = None,
         methods: str = "mean_diff,probe,ica,gradient",
         layers: str = None):
    """
    Evaluate trait vectors across multiple quality metrics.

    Args:
        experiment: Experiment name (e.g., "gemma_2b_cognitive_nov21")
        trait: Specific trait to evaluate (e.g., "cognitive_state/uncertainty_expression")
               If None, evaluate all traits in experiment
        output: Output file path for results (JSON)
        methods: Comma-separated list of methods to evaluate
        layers: Comma-separated list of layers to evaluate (default: all)
    """
    methods_list = methods.split(",")
    layers_list = [int(l) for l in layers.split(",")] if layers else None

    # Find all traits if none specified
    if trait is None:
        exp_dir = Path(f"experiments/{experiment}/extraction")
        traits = []
        for category_dir in exp_dir.iterdir():
            if category_dir.is_dir():
                for trait_dir in category_dir.iterdir():
                    if trait_dir.is_dir() and (trait_dir / "vectors").exists():
                        trait_path = f"{category_dir.name}/{trait_dir.name}"
                        traits.append(trait_path)
    else:
        traits = [trait]

    print(f"Evaluating {len(traits)} traits...")

    # Evaluate each trait
    results = {}
    for trait_path in tqdm(traits, desc="Traits"):
        print(f"\nEvaluating {trait_path}...")
        evaluator = VectorEvaluator(experiment, trait_path)

        # Get all other traits for orthogonality evaluation
        other_traits = [t for t in traits if t != trait_path]

        # Find best vector
        summary = evaluator.find_best_vector(
            other_traits=other_traits,
            methods=methods_list,
            layers=layers_list
        )

        if summary:
            results[trait_path] = summary

            # Print summary
            print(f"\n{trait_path} Results:")
            print(f"  Best Overall: {summary['best_overall']['method']}_layer{summary['best_overall']['layer']}")
            print(f"    - Overall Score: {summary['best_overall']['overall_score']:.1f}")
            print(f"    - Separation: {summary['best_overall']['separation_score']:.2f}")
            print(f"    - Train Accuracy: {summary['best_overall']['train_accuracy']:.2%}")

            if summary.get('best_cross_distribution'):
                print(f"  Best Cross-Dist: {summary['best_cross_distribution']['method']}_layer{summary['best_cross_distribution']['layer']}")
                print(f"    - Consistency: {summary['best_cross_distribution']['cross_dist_consistency']:.2f}")

    # Save results
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    # Create summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE - Best Vectors by Trait")
    print("="*80)

    rows = []
    for trait_name, summary in results.items():
        if summary.get('best_overall'):
            best = summary['best_overall']
            rows.append({
                'Trait': trait_name.split('/')[-1][:20],
                'Method': best['method'],
                'Layer': best['layer'],
                'Overall': f"{best['overall_score']:.1f}",
                'Sep': f"{best['separation_score']:.2f}",
                'Acc': f"{best['train_accuracy']:.1%}",
                'Reliable': f"{best['reliability_score']:.1f}"
            })

    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

    return results


if __name__ == "__main__":
    fire.Fire(main)