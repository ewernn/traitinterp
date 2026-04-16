"""
Trait vector geometry analysis: clustering, UMAP, representational similarity,
cosine heatmap ordering, and PCA vs human-norm correlation.

Input: Pre-extracted trait vectors (loaded from experiments/{experiment}/extraction/)
Output: Clustering assignments, UMAP coordinates, RSA matrices, ordered cosine heatmaps,
        PC-vs-human-norm correlation dicts, saved to analysis output dir

Usage:
    python analysis/vectors/geometry.py --experiment {experiment} --layer 53
    python analysis/vectors/geometry.py --experiment {experiment} --layer 53 --only cosine,pca
    python analysis/vectors/geometry.py --experiment {experiment} --layer 53 --rsa-layers 25,31,37,43,49,55,61,67
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.math import (
    pairwise_cosine_matrix, pca,
    trait_clusters, representational_similarity, pca_norm_correlation,
)


def umap_projection(vectors: torch.Tensor, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42):
    """2D UMAP projection of trait vectors.

    Input: [N, hidden_dim]
    Output: [N, 2] numpy array of 2D coordinates
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap-learn required: pip install umap-learn")
    X = vectors.float().cpu().numpy()
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    return reducer.fit_transform(X)


def cosine_heatmap_ordered(vectors: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
    """Pairwise cosine similarity matrix with hierarchical-clustering leaf order.

    Uses scipy average-linkage on (1 - cosine) distance, with `checks=False` on
    `squareform` to skip the symmetry check (load-bearing: near-symmetric float32
    matrices can fail the check). Returns the unreordered similarity matrix and
    the leaf-order permutation — caller builds `ordered_matrix = sim[order][:, order]`.

    Input: [N, hidden_dim]
    Output: (similarity_matrix [N, N], order [N])
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    sim_matrix = pairwise_cosine_matrix(vectors)  # [N, N]
    dist = 1.0 - sim_matrix.numpy()
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2  # enforce symmetry for scipy
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    order = leaves_list(Z)
    return sim_matrix, order


# =============================================================================
# CLI — standalone geometry analysis
# =============================================================================

def _load_vectors(experiment, category, layer, model_variant, method, component, position):
    """Load all trait vectors for a category, filtering neutrals."""
    from utils.paths import discover_traits
    from utils.vectors import load_vector

    traits = discover_traits(category)
    traits = [t for t in traits if '/_neutral' not in t and not t.endswith('_neutral')]

    vectors, labels = [], []
    failed = []
    for trait in sorted(traits):
        vec = load_vector(experiment, trait, layer, model_variant,
                          method=method, component=component, position=position)
        if vec is not None:
            vectors.append(vec)
            labels.append(trait.split('/')[-1])
        else:
            failed.append(trait)

    if failed:
        print(f"  WARNING: {len(failed)}/{len(traits)} traits missing vectors at layer {layer}")
        if len(failed) <= 10:
            for f in failed:
                print(f"    missing: {f}")

    if not vectors:
        raise FileNotFoundError(
            f"No vectors found for {category} at layer {layer} (method={method}). "
            f"Run extraction first."
        )

    return torch.stack(vectors), labels


def _save(out_dir, name, data):
    """Save results as compact JSON with timestamp."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    data['timestamp'] = datetime.now().isoformat()
    with open(out_path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    print(f"  Saved: {out_path}")
    return out_path


def _compare_to_baseline(metric_name, our_value, anthropic_value, tolerance=0.1):
    """Print comparison to a baseline value."""
    diff = abs(our_value - anthropic_value)
    pct_diff = diff / abs(anthropic_value) if anthropic_value != 0 else float('inf')
    match = "MATCH" if pct_diff <= tolerance else "DIFFERS"
    msg = f"  {metric_name}: ours={our_value:.4f}, baseline={anthropic_value:.4f} ({pct_diff:.0%} diff) [{match}]"
    print(msg)
    return msg


def run_cosine(vectors, trait_names, out_dir):
    """[Fig 5] Pairwise cosine similarity heatmap."""
    print("\n  [Fig 5] Pairwise cosine similarity heatmap...")
    sim_matrix, order = cosine_heatmap_ordered(vectors)
    ordered_names = [trait_names[i] for i in order]
    ordered_matrix = sim_matrix[order][:, order]
    n = len(trait_names)
    idx = torch.triu_indices(n, n, offset=1)
    upper_tri = sim_matrix[idx[0], idx[1]]
    print(f"    Similarity stats: mean={upper_tri.mean():.3f}, std={upper_tri.std():.3f}")
    print(f"    Range: [{upper_tri.min():.3f}, {upper_tri.max():.3f}]")
    result = {
        'similarity_matrix': sim_matrix.tolist(),
        'ordered_matrix': ordered_matrix.tolist(),
        'trait_names': trait_names,
        'ordered_names': ordered_names,
        'cluster_order': order.tolist(),
        'stats': {'mean': float(upper_tri.mean()), 'std': float(upper_tri.std()),
                  'min': float(upper_tri.min()), 'max': float(upper_tri.max())},
    }
    _save(out_dir, 'cosine_heatmap', result)
    return result


def run_cluster(vectors, trait_names, out_dir, k=10, baselines=None):
    """[Fig 6] K-means clustering + UMAP."""
    print(f"\n  [Fig 6] K-means clustering (k={k}) + UMAP...")
    labels, centroids, inertia = trait_clusters(vectors, k=k)
    print(f"    Inertia: {inertia:.1f}")
    clusters = {}
    for i in range(k):
        members = sorted([trait_names[j] for j in range(len(labels)) if labels[j] == i])
        clusters[i] = {'members': members, 'size': len(members), 'name': f'Cluster_{i}'}
        print(f"    Cluster {i} ({len(members)} members): {', '.join(members[:5])}...")
    print(f"    Cluster sizes: {sorted([c['size'] for c in clusters.values()], reverse=True)}")
    if baselines and 'cluster_sizes' in baselines:
        print(f"    Baseline sizes: {sorted(baselines['cluster_sizes'], reverse=True)}")

    umap_coords = umap_projection(vectors)
    result = {
        'k': k, 'inertia': float(inertia),
        'cluster_assignments': labels.tolist(), 'clusters': clusters,
        'umap_coordinates': umap_coords.tolist(), 'trait_names': trait_names,
    }
    if baselines and 'cluster_sizes' in baselines:
        result['baseline_cluster_sizes'] = baselines['cluster_sizes']
    _save(out_dir, 'clusters_umap', result)
    return result


def run_pca(vectors, trait_names, out_dir, n_components=10, norms=None, baselines=None):
    """[Figs 7-8] PCA variance and human norm correlation."""
    print(f"\n  [Figs 7-8] PCA (n_components={n_components})...")
    components, var_ratio, projections = pca(vectors, n_components=n_components)
    print(f"    Variance explained:")
    cumvar = 0.0
    for i in range(min(5, n_components)):
        cumvar += var_ratio[i].item()
        print(f"      PC{i+1}: {var_ratio[i].item()*100:.1f}%  (cumulative: {cumvar*100:.1f}%)")

    if baselines:
        if 'pc1_variance' in baselines:
            _compare_to_baseline('PC1 variance', var_ratio[0].item(), baselines['pc1_variance'])
        if 'pc2_variance' in baselines:
            _compare_to_baseline('PC2 variance', var_ratio[1].item(), baselines['pc2_variance'])

    pc1_sorted = sorted(zip(trait_names, projections[:, 0].tolist()), key=lambda x: x[1])
    pc2_sorted = sorted(zip(trait_names, projections[:, 1].tolist()), key=lambda x: x[1])
    print(f"\n    PC1 extremes (valence?):")
    print(f"      Negative: {', '.join([f'{n}({v:.2f})' for n, v in pc1_sorted[:5]])}")
    print(f"      Positive: {', '.join([f'{n}({v:.2f})' for n, v in pc1_sorted[-5:]])}")
    print(f"    PC2 extremes (arousal?):")
    print(f"      Low:  {', '.join([f'{n}({v:.2f})' for n, v in pc2_sorted[:5]])}")
    print(f"      High: {', '.join([f'{n}({v:.2f})' for n, v in pc2_sorted[-5:]])}")

    human_corr = None
    if norms:
        human_corr = pca_norm_correlation(projections, trait_names, norms, min_overlap=10)
        if human_corr:
            print(f"\n    Human norm correlation ({human_corr['n_overlapping']} overlapping emotions):")
            if baselines and 'pc1_vs_valence' in baselines:
                _compare_to_baseline('PC1 vs valence', human_corr['pc1_vs_valence'], baselines['pc1_vs_valence'])
            if baselines and 'pc2_vs_arousal' in baselines:
                _compare_to_baseline('PC2 vs arousal', human_corr['pc2_vs_arousal'], baselines['pc2_vs_arousal'])
            print(f"      PC1 vs arousal: r = {human_corr['pc1_vs_arousal']:.3f}  (should be weak)")
            print(f"      PC2 vs valence: r = {human_corr['pc2_vs_valence']:.3f}  (should be weak)")

    result = {
        'n_components': n_components,
        'variance_explained': var_ratio.tolist(),
        'projections': projections.tolist(),
        'trait_names': trait_names,
        'pc1_sorted': [(n, float(v)) for n, v in pc1_sorted],
        'pc2_sorted': [(n, float(v)) for n, v in pc2_sorted],
        'human_norm_correlation': human_corr,
    }
    if baselines:
        result['baselines'] = {k: baselines[k] for k in
                               ('pc1_variance', 'pc2_variance', 'pc1_vs_valence', 'pc2_vs_arousal')
                               if k in baselines}
    _save(out_dir, 'pca_analysis', result)
    return result


def run_rsa(vectors_by_layer, trait_names, out_dir, rsa_method="cosine"):
    """[Fig 9] Cross-layer representational similarity."""
    print(f"\n  [Fig 9] Cross-layer RSA ({len(vectors_by_layer)} layers, method={rsa_method})...")
    rsa_matrix, layers = representational_similarity(vectors_by_layer, method=rsa_method)
    print(f"    Mean diagonal: 1.000 (by construction)")
    print(f"    Mean off-diagonal: {rsa_matrix[rsa_matrix < 0.999].mean():.3f}")
    result = {'rsa_matrix': rsa_matrix.tolist(), 'layers': layers, 'n_traits': len(trait_names)}
    _save(out_dir, 'rsa_analysis', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--category', type=str, required=False, default=None,
                        help='Trait category (default: same as experiment name)')
    parser.add_argument('--method', default='mean_diff+gm+pc50')
    parser.add_argument('--component', default='residual')
    parser.add_argument('--position', default='response[50:]')
    parser.add_argument('--model-variant', default=None)
    parser.add_argument('--rsa-layers', type=str, default=None,
                        help='Comma-separated layer indices for RSA (e.g. 25,31,37,43,49,55)')
    parser.add_argument('--only', type=str, default=None,
                        help='Comma-separated analyses to run (cosine,cluster,pca,rsa)')
    parser.add_argument('--k', type=int, default=10, help='Number of clusters for k-means')
    parser.add_argument('--n-components', type=int, default=10, help='Number of PCA components')
    parser.add_argument('--baselines-json', type=str, default=None,
                        help='Path to JSON with baseline numbers for comparison')
    parser.add_argument('--norms-file', type=str, default=None,
                        help='Path to Russell & Mehrabian valence/arousal norms JSON')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: experiments/{experiment}/results/geometry/)')
    args = parser.parse_args()

    from utils.paths import get as get_path, get_default_variant

    category = args.category or args.experiment
    variant = args.model_variant or get_default_variant(args.experiment, mode='extraction')

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        base = get_path('experiments.base', experiment=args.experiment)
        out_dir = base / 'results' / 'geometry'

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\nModel variant: {variant}")

    # Load baselines if provided
    baselines = None
    if args.baselines_json:
        with open(args.baselines_json) as f:
            baselines = json.load(f)
        print(f"Loaded baselines from: {args.baselines_json}")

    # Load norms if provided
    norms = None
    if args.norms_file:
        with open(args.norms_file) as f:
            norms_data = json.load(f)
        norms = {name: (float(e['pleasure']), float(e['arousal']))
                 for name, e in norms_data['emotions'].items()}
        print(f"Loaded norms for {len(norms)} emotions from: {args.norms_file}")

    all_analyses = {'cosine', 'cluster', 'pca', 'rsa'}
    analyses = set(args.only.split(',')) if args.only else all_analyses
    invalid = analyses - all_analyses
    if invalid:
        parser.error(f"Unknown analyses: {invalid}. Choose from: {all_analyses}")

    print(f"\nLoading vectors at layer {args.layer}...")
    vectors, trait_names = _load_vectors(
        args.experiment, category, args.layer, variant,
        method=args.method, component=args.component, position=args.position,
    )
    print(f"  Loaded {len(trait_names)} vectors, dim={vectors.shape[1]}")

    results = {}
    if 'cosine' in analyses:
        results['cosine'] = run_cosine(vectors, trait_names, out_dir)
    if 'cluster' in analyses:
        results['cluster'] = run_cluster(vectors, trait_names, out_dir, k=args.k, baselines=baselines)
    if 'pca' in analyses:
        results['pca'] = run_pca(vectors, trait_names, out_dir,
                                 n_components=args.n_components, norms=norms, baselines=baselines)
    if 'rsa' in analyses and args.rsa_layers:
        rsa_layer_list = sorted(set([int(x) for x in args.rsa_layers.split(',')] + [args.layer]))
        print(f"\n  Loading vectors at {len(rsa_layer_list)} layers for RSA...")
        vectors_by_layer = {}
        for layer in rsa_layer_list:
            try:
                vecs, _ = _load_vectors(
                    args.experiment, category, layer, variant,
                    method=args.method, component=args.component, position=args.position,
                )
                vectors_by_layer[layer] = vecs
            except FileNotFoundError:
                print(f"    Skipping layer {layer}: no vectors found")
        if len(vectors_by_layer) >= 2:
            results['rsa'] = run_rsa(vectors_by_layer, trait_names, out_dir)

    # Summary
    print(f"\n{'='*60}\nGeometry Analysis — Summary\n{'='*60}")
    print(f"  Experiment: {args.experiment}\n  Layer: {args.layer}\n  Traits: {len(trait_names)}")
    print(f"  Analyses: {', '.join(sorted(results.keys()))}")

    if 'pca' in results and baselines:
        var = results['pca']['variance_explained']
        print(f"\n  PCA comparison to baselines:")
        if 'pc1_variance' in baselines:
            _compare_to_baseline('PC1 variance', var[0], baselines['pc1_variance'])
        if 'pc2_variance' in baselines:
            _compare_to_baseline('PC2 variance', var[1], baselines['pc2_variance'])
        hc = results['pca'].get('human_norm_correlation')
        if hc:
            if 'pc1_vs_valence' in baselines:
                _compare_to_baseline('PC1 vs valence', hc['pc1_vs_valence'], baselines['pc1_vs_valence'])
            if 'pc2_vs_arousal' in baselines:
                _compare_to_baseline('PC2 vs arousal', hc['pc2_vs_arousal'], baselines['pc2_vs_arousal'])

    _save(out_dir, 'run_metadata', {
        'experiment': args.experiment, 'layer': args.layer, 'model_variant': variant,
        'method': args.method, 'category': category,
        'n_traits': len(trait_names), 'analyses_run': sorted(results.keys()),
        'trait_names': trait_names,
    })
    print(f"\n  Results saved to: {out_dir}\n{'='*60}")


if __name__ == '__main__':
    main()
