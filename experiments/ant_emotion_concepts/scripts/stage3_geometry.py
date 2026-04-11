"""Stage 3: Geometry analysis of 171 emotion vectors — thin wrapper.

Replicates Sofroniew et al. 2026 Part 2, Section 2.1:
  Fig 5  — Pairwise cosine heatmap (hierarchical-cluster ordered)
  Fig 6  — K-means (k=10) + UMAP projection
  Fig 7  — PCA variance and extremes
  Fig 8  — PC1/PC2 vs Russell & Mehrabian 1977 human valence/arousal norms
  Fig 9  — Cross-layer representational similarity (RSA)

All primitives live in `analysis/vectors/geometry.py` and `core/math.py`. This
script just: loads vectors, dispatches to the helpers, saves the output schemas
that the rest of the pipeline expects.

Input: Denoised vectors at experiments/{experiment}/extraction/{trait}/{variant}/vectors/
Output: experiments/{experiment}/results/stage3_geometry/

Usage:
    python experiments/ant_emotion_concepts/scripts/stage3_geometry.py \\
        --experiment ant_emotion_concepts --layer 49
    python ... --layer 49 --rsa-layers 25,31,37,43,49,55,61,67 --only cosine,pca
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from core.math import pairwise_cosine_matrix, pca
from analysis.vectors.geometry import (
    cosine_heatmap_ordered, trait_clusters, umap_projection,
    representational_similarity, pca_norm_correlation,
)
from utils.paths import get_default_variant
from shared import (
    ANTHROPIC_BASELINES, load_russell_mehrabian_norms,
    get_results_dir, save_results, compare_to_baseline,
    load_all_emotion_vectors,
)


def run_cosine(vectors, trait_names, out_dir):
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
    save_results(out_dir, 'cosine_heatmap', result, compact=True)
    return result


def run_cluster(vectors, trait_names, out_dir, k=10):
    print(f"\n  [Fig 6] K-means clustering (k={k}) + UMAP...")
    labels, centroids, inertia = trait_clusters(vectors, k=k)
    print(f"    Inertia: {inertia:.1f}")
    clusters = {}
    for i in range(k):
        members = sorted([trait_names[j] for j in range(len(labels)) if labels[j] == i])
        clusters[i] = {'members': members, 'size': len(members), 'name': f'Cluster_{i}'}
        print(f"    Cluster {i} ({len(members)} members): {', '.join(members[:5])}...")
    print(f"    Cluster sizes (ours vs Anthropic):")
    print(f"      Ours:      {sorted([c['size'] for c in clusters.values()], reverse=True)}")
    print(f"      Anthropic: {sorted(ANTHROPIC_BASELINES['cluster_sizes'], reverse=True)}")
    umap_coords = umap_projection(vectors)
    result = {
        'k': k, 'inertia': float(inertia),
        'cluster_assignments': labels.tolist(), 'clusters': clusters,
        'umap_coordinates': umap_coords.tolist(), 'trait_names': trait_names,
        'anthropic_cluster_sizes': ANTHROPIC_BASELINES['cluster_sizes'],
    }
    save_results(out_dir, 'clusters_umap', result, compact=True)
    return result


def run_pca(vectors, trait_names, out_dir, norms, n_components=10):
    print(f"\n  [Figs 7-8] PCA (n_components={n_components})...")
    components, var_ratio, projections = pca(vectors, n_components=n_components)
    print(f"    Variance explained:")
    cumvar = 0.0
    for i in range(min(5, n_components)):
        cumvar += var_ratio[i].item()
        print(f"      PC{i+1}: {var_ratio[i].item()*100:.1f}%  (cumulative: {cumvar*100:.1f}%)")
    compare_to_baseline('PC1 variance', var_ratio[0].item(), ANTHROPIC_BASELINES['pc1_variance'])
    compare_to_baseline('PC2 variance', var_ratio[1].item(), ANTHROPIC_BASELINES['pc2_variance'])

    pc1_sorted = sorted(zip(trait_names, projections[:, 0].tolist()), key=lambda x: x[1])
    pc2_sorted = sorted(zip(trait_names, projections[:, 1].tolist()), key=lambda x: x[1])
    print(f"\n    PC1 extremes (valence?):")
    print(f"      Negative: {', '.join([f'{n}({v:.2f})' for n, v in pc1_sorted[:5]])}")
    print(f"      Positive: {', '.join([f'{n}({v:.2f})' for n, v in pc1_sorted[-5:]])}")
    print(f"    PC2 extremes (arousal?):")
    print(f"      Low:  {', '.join([f'{n}({v:.2f})' for n, v in pc2_sorted[:5]])}")
    print(f"      High: {', '.join([f'{n}({v:.2f})' for n, v in pc2_sorted[-5:]])}")

    human_corr = pca_norm_correlation(projections, trait_names, norms, min_overlap=10)
    if human_corr:
        print(f"\n    Human norm correlation ({human_corr['n_overlapping']} overlapping emotions):")
        compare_to_baseline('PC1 vs valence', human_corr['pc1_vs_valence'], ANTHROPIC_BASELINES['pc1_vs_valence'])
        compare_to_baseline('PC2 vs arousal', human_corr['pc2_vs_arousal'], ANTHROPIC_BASELINES['pc2_vs_arousal'])
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
        'anthropic_baselines': {k: ANTHROPIC_BASELINES[k] for k in
                                ('pc1_variance', 'pc2_variance', 'pc1_vs_valence', 'pc2_vs_arousal')},
    }
    save_results(out_dir, 'pca_analysis', result, compact=True)
    return result


def run_rsa(vectors_by_layer, trait_names, out_dir):
    print(f"\n  [Fig 9] Cross-layer RSA ({len(vectors_by_layer)} layers)...")
    rsa_matrix, layers = representational_similarity(vectors_by_layer)
    print(f"    Mean diagonal: 1.000 (by construction)")
    print(f"    Mean off-diagonal: {rsa_matrix[rsa_matrix < 0.999].mean():.3f}")
    result = {'rsa_matrix': rsa_matrix.tolist(), 'layers': layers, 'n_traits': len(trait_names)}
    save_results(out_dir, 'rsa_analysis', result, compact=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiment', required=True)
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--rsa-layers', type=str, default=None)
    p.add_argument('--model-variant', default=None)
    p.add_argument('--category', default='ant_emotion_concepts')
    p.add_argument('--method', default='mean_diff+gm+pc50')
    p.add_argument('--component', default='residual')
    p.add_argument('--position', default='response[50:]')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--n-components', type=int, default=10)
    p.add_argument('--only', type=str, default=None)
    args = p.parse_args()

    variant = args.model_variant or get_default_variant(args.experiment, mode='extraction')
    out_dir = get_results_dir(args.experiment, 'stage3_geometry')
    print(f"Output: {out_dir}\nModel variant: {variant}")

    all_analyses = {'cosine', 'cluster', 'pca', 'rsa'}
    analyses = set(args.only.split(',')) if args.only else all_analyses
    invalid = analyses - all_analyses
    if invalid:
        p.error(f"Unknown analyses: {invalid}. Choose from: {all_analyses}")

    print(f"\nLoading denoised vectors at layer {args.layer}...")
    vectors, trait_names = load_all_emotion_vectors(
        args.experiment, args.category, args.layer, variant,
        method=args.method, component=args.component, position=args.position,
    )
    print(f"  Loaded {len(trait_names)} emotion vectors, dim={vectors.shape[1]}")

    norms = load_russell_mehrabian_norms(args.experiment)

    results = {}
    if 'cosine' in analyses:
        results['cosine'] = run_cosine(vectors, trait_names, out_dir)
    if 'cluster' in analyses:
        results['cluster'] = run_cluster(vectors, trait_names, out_dir, k=args.k)
    if 'pca' in analyses:
        results['pca'] = run_pca(vectors, trait_names, out_dir, norms, n_components=args.n_components)
    if 'rsa' in analyses and args.rsa_layers:
        rsa_layer_list = sorted(set([int(x) for x in args.rsa_layers.split(',')] + [args.layer]))
        print(f"\n  Loading vectors at {len(rsa_layer_list)} layers for RSA...")
        vectors_by_layer = {}
        for layer in rsa_layer_list:
            try:
                vecs, _ = load_all_emotion_vectors(
                    args.experiment, args.category, layer, variant,
                    method=args.method, component=args.component, position=args.position,
                )
                vectors_by_layer[layer] = vecs
            except FileNotFoundError:
                print(f"    Skipping layer {layer}: no vectors found")
        if len(vectors_by_layer) >= 2:
            results['rsa'] = run_rsa(vectors_by_layer, trait_names, out_dir)

    print(f"\n{'='*60}\nStage 3 Geometry Analysis — Summary\n{'='*60}")
    print(f"  Experiment: {args.experiment}\n  Layer: {args.layer}\n  Emotions: {len(trait_names)}")
    print(f"  Analyses: {', '.join(sorted(results.keys()))}")

    if 'pca' in results:
        var = results['pca']['variance_explained']
        print(f"\n  PCA comparison to Anthropic:")
        compare_to_baseline('PC1 variance', var[0], ANTHROPIC_BASELINES['pc1_variance'])
        compare_to_baseline('PC2 variance', var[1], ANTHROPIC_BASELINES['pc2_variance'])
        hc = results['pca']['human_norm_correlation']
        if hc:
            compare_to_baseline('PC1 vs valence', hc['pc1_vs_valence'], ANTHROPIC_BASELINES['pc1_vs_valence'])
            compare_to_baseline('PC2 vs arousal', hc['pc2_vs_arousal'], ANTHROPIC_BASELINES['pc2_vs_arousal'])

    save_results(out_dir, 'run_metadata', {
        'experiment': args.experiment, 'layer': args.layer, 'model_variant': variant,
        'method': args.method, 'category': args.category,
        'n_emotions': len(trait_names), 'analyses_run': sorted(results.keys()),
        'trait_names': trait_names,
    })
    print(f"\n  Results saved to: {out_dir}\n{'='*60}")


if __name__ == '__main__':
    main()
