"""Stage 3: Geometry analysis of 171 emotion vectors.

Replicates Sofroniew et al. 2026 Part 2, Section 2.1:
  - Fig 5:  Pairwise cosine similarity heatmap (171x171)
  - Fig 6:  K-means clustering (k=10) + UMAP projection
  - Figs 7-8: PCA (PC1=valence, PC2=arousal) + correlation with human norms
  - Fig 9:  Cross-layer representational similarity analysis (RSA)

All CPU-only. Depends on Stage 2 (denoised vectors must exist).

Input: Denoised vectors at experiments/{experiment}/extraction/{trait}/{variant}/vectors/
Output: experiments/{experiment}/results/stage3_geometry/

Usage:
    # Full analysis at a single layer:
    python experiments/ant_emotion_concepts/scripts/stage3_geometry.py \
        --experiment ant_emotion_concepts --layer 53

    # RSA across multiple layers:
    python experiments/ant_emotion_concepts/scripts/stage3_geometry.py \
        --experiment ant_emotion_concepts --layer 53 \
        --rsa-layers 20,30,40,45,50,53,55,60,65,70,72,75,77,80

    # Specific analyses only:
    python experiments/ant_emotion_concepts/scripts/stage3_geometry.py \
        --experiment ant_emotion_concepts --layer 53 --only cosine,pca
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from core.math import pairwise_cosine_matrix, pca, pearson_correlation
from analysis.vectors.geometry import trait_clusters, umap_projection, representational_similarity
from utils.paths import get_default_variant
from shared import (
    get_results_dir, save_results, compare_to_baseline,
    load_all_emotion_vectors,
)


# ============================================================================
# Anthropic baseline numbers (Sonnet 4.5) — for comparison printout
# ============================================================================

ANTHROPIC_BASELINES = {
    'pc1_variance': 0.26,
    'pc2_variance': 0.15,
    'pc1_vs_valence': 0.81,
    'pc2_vs_arousal': 0.66,
    'n_clusters': 10,
    'cluster_sizes': [20, 9, 15, 9, 2, 15, 3, 25, 41, 32],
}

# Russell & Mehrabian (1977) PAD norms — emotions overlapping with the 171-emotion set.
# Loaded from datasets/russell_mehrabian_norms.json at module import.
# Format: {emotion_name: (valence, arousal)} on standardized [-1, 1]-ish scale.
def _load_russell_mehrabian_norms():
    norms_path = Path(__file__).resolve().parent.parent / 'datasets' / 'russell_mehrabian_norms.json'
    with open(norms_path) as f:
        data = json.load(f)
    return {name: (float(e['pleasure']), float(e['arousal'])) for name, e in data['emotions'].items()}


RUSSELL_MEHRABIAN_NORMS = _load_russell_mehrabian_norms()


# ============================================================================
# Analysis: Cosine heatmap (Fig 5)
# ============================================================================

def run_cosine_analysis(vectors, trait_names, out_dir):
    """Pairwise cosine similarity heatmap with hierarchical clustering order.

    Calls: core.math.pairwise_cosine_matrix()
    Custom code: hierarchical clustering for optimal row/column ordering.
    """
    print("\n  [Fig 5] Pairwise cosine similarity heatmap...")

    sim_matrix = pairwise_cosine_matrix(vectors)  # [N, N]

    # Hierarchical clustering for ordering
    # CUSTOM CODE: scipy not in the existing pipeline, but standard for this
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform

        # Convert cosine similarity to distance
        dist = 1.0 - sim_matrix.numpy()
        np.fill_diagonal(dist, 0)
        # Ensure symmetry (floating point)
        dist = (dist + dist.T) / 2
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method='average')
        order = leaves_list(Z)
    except ImportError:
        print("    WARNING: scipy not installed, using original ordering")
        order = np.arange(len(trait_names))

    ordered_names = [trait_names[i] for i in order]
    ordered_matrix = sim_matrix[order][:, order]

    # Summary stats
    upper_tri = sim_matrix[torch.triu_indices(len(trait_names), len(trait_names), offset=1)[0],
                           torch.triu_indices(len(trait_names), len(trait_names), offset=1)[1]]
    print(f"    Similarity stats: mean={upper_tri.mean():.3f}, std={upper_tri.std():.3f}")
    print(f"    Range: [{upper_tri.min():.3f}, {upper_tri.max():.3f}]")

    # Save
    result = {
        'similarity_matrix': sim_matrix.tolist(),
        'ordered_matrix': ordered_matrix.tolist(),
        'trait_names': trait_names,
        'ordered_names': ordered_names,
        'cluster_order': order.tolist() if isinstance(order, np.ndarray) else order,
        'stats': {
            'mean': float(upper_tri.mean()),
            'std': float(upper_tri.std()),
            'min': float(upper_tri.min()),
            'max': float(upper_tri.max()),
        },
    }

    save_results(out_dir, 'cosine_heatmap', result, compact=True)

    return result


# ============================================================================
# Analysis: K-means + UMAP (Fig 6)
# ============================================================================

def run_cluster_analysis(vectors, trait_names, out_dir, k=10):
    """K-means clustering + UMAP visualization.

    Calls: analysis.vectors.geometry.trait_clusters(k=10)
           analysis.vectors.geometry.umap_projection()
    Custom code: cluster naming (placeholder — needs LLM or manual labeling).
    """
    print(f"\n  [Fig 6] K-means clustering (k={k}) + UMAP...")

    # K-means
    labels, centroids, inertia = trait_clusters(vectors, k=k)
    print(f"    Inertia: {inertia:.1f}")

    # Cluster membership
    clusters = {}
    for i in range(k):
        members = [trait_names[j] for j in range(len(labels)) if labels[j] == i]
        clusters[i] = {
            'members': sorted(members),
            'size': len(members),
            # Placeholder name — needs LLM judge or manual labeling
            'name': f'Cluster_{i}',
        }
        print(f"    Cluster {i} ({len(members)} members): {', '.join(sorted(members)[:5])}...")

    # Compare to Anthropic cluster sizes
    our_sizes = sorted([c['size'] for c in clusters.values()], reverse=True)
    print(f"\n    Cluster sizes (ours vs Anthropic):")
    print(f"      Ours:     {our_sizes}")
    print(f"      Anthropic: {sorted(ANTHROPIC_BASELINES['cluster_sizes'], reverse=True)}")

    # UMAP
    umap_coords = umap_projection(vectors)  # [N, 2]
    print(f"    UMAP complete: {umap_coords.shape}")

    # Save
    result = {
        'k': k,
        'inertia': float(inertia),
        'cluster_assignments': labels.tolist(),
        'clusters': clusters,
        'umap_coordinates': umap_coords.tolist(),
        'trait_names': trait_names,
        'anthropic_cluster_sizes': ANTHROPIC_BASELINES['cluster_sizes'],
    }

    save_results(out_dir, 'clusters_umap', result, compact=True)

    # NOTE: Cluster naming is a gap. The paper uses Claude Sonnet 4.5 to name
    # clusters by inspecting member lists. Options:
    #   1. LLM API call to name clusters (requires API key)
    #   2. Manual labeling after inspecting output
    #   3. Use the Anthropic cluster names as reference and match by overlap
    print("    NOTE: Cluster naming not implemented. See output for member lists.")

    return result


# ============================================================================
# Analysis: PCA (Figs 7-8)
# ============================================================================

def run_pca_analysis(vectors, trait_names, out_dir, n_components=10):
    """PCA on emotion vectors, correlate with human valence/arousal norms.

    Calls: core.math.pca(n_components=10)
    Custom code: correlation with Russell & Mehrabian norms.

    GAP: Russell & Mehrabian 1977 norms must be digitized and added to
    RUSSELL_MEHRABIAN_NORMS dict above. Without them, Fig 8 correlation
    cannot be computed — only PCA itself (Fig 7) runs.
    """
    print(f"\n  [Figs 7-8] PCA (n_components={n_components})...")

    components, var_ratio, projections = pca(vectors, n_components=n_components)

    print(f"    Variance explained:")
    cumvar = 0
    for i in range(min(5, n_components)):
        cumvar += var_ratio[i].item()
        print(f"      PC{i+1}: {var_ratio[i].item()*100:.1f}%  (cumulative: {cumvar*100:.1f}%)")
    compare_to_baseline('PC1 variance', var_ratio[0].item(), ANTHROPIC_BASELINES['pc1_variance'])
    compare_to_baseline('PC2 variance', var_ratio[1].item(), ANTHROPIC_BASELINES['pc2_variance'])

    # Per-emotion PC projections (Fig 7)
    pc1_proj = projections[:, 0]
    pc2_proj = projections[:, 1]

    # Sort by PC1 for display
    pc1_sorted = sorted(zip(trait_names, pc1_proj.tolist()), key=lambda x: x[1])
    print(f"\n    PC1 extremes (valence?):")
    print(f"      Negative: {', '.join([f'{n}({v:.2f})' for n, v in pc1_sorted[:5]])}")
    print(f"      Positive: {', '.join([f'{n}({v:.2f})' for n, v in pc1_sorted[-5:]])}")

    pc2_sorted = sorted(zip(trait_names, pc2_proj.tolist()), key=lambda x: x[1])
    print(f"    PC2 extremes (arousal?):")
    print(f"      Low:  {', '.join([f'{n}({v:.2f})' for n, v in pc2_sorted[:5]])}")
    print(f"      High: {', '.join([f'{n}({v:.2f})' for n, v in pc2_sorted[-5:]])}")

    # Correlation with human norms (Fig 8)
    human_corr = None
    if RUSSELL_MEHRABIAN_NORMS:
        overlapping = [(i, name) for i, name in enumerate(trait_names)
                       if name in RUSSELL_MEHRABIAN_NORMS]

        if len(overlapping) >= 10:
            indices = [i for i, _ in overlapping]
            names_overlap = [n for _, n in overlapping]

            human_valence = torch.tensor([RUSSELL_MEHRABIAN_NORMS[n][0] for n in names_overlap])
            human_arousal = torch.tensor([RUSSELL_MEHRABIAN_NORMS[n][1] for n in names_overlap])

            pc1_overlap = pc1_proj[indices]
            pc2_overlap = pc2_proj[indices]

            # Pearson correlations
            r_pc1_val = pearson_correlation(pc1_overlap, human_valence)
            r_pc2_aro = pearson_correlation(pc2_overlap, human_arousal)
            # Also check cross-correlations (should be weaker)
            r_pc1_aro = pearson_correlation(pc1_overlap, human_arousal)
            r_pc2_val = pearson_correlation(pc2_overlap, human_valence)

            human_corr = {
                'n_overlapping': len(overlapping),
                'emotions': names_overlap,
                'pc1_vs_valence': float(r_pc1_val),
                'pc2_vs_arousal': float(r_pc2_aro),
                'pc1_vs_arousal': float(r_pc1_aro),
                'pc2_vs_valence': float(r_pc2_val),
            }

            print(f"\n    Human norm correlation ({len(overlapping)} overlapping emotions):")
            compare_to_baseline('PC1 vs valence', float(r_pc1_val), ANTHROPIC_BASELINES['pc1_vs_valence'])
            compare_to_baseline('PC2 vs arousal', float(r_pc2_aro), ANTHROPIC_BASELINES['pc2_vs_arousal'])
            print(f"      PC1 vs arousal: r = {r_pc1_aro:.3f}  (should be weak)")
            print(f"      PC2 vs valence: r = {r_pc2_val:.3f}  (should be weak)")
        else:
            print(f"\n    Only {len(overlapping)} overlapping emotions with human norms (need >= 10)")
    else:
        print("\n    WARNING: Russell & Mehrabian norms not populated yet.")
        print("    Fig 8 (PC vs human norms correlation) CANNOT be computed.")
        print("    Action: digitize norms from Russell & Mehrabian 1977 Table 1 into")
        print("    RUSSELL_MEHRABIAN_NORMS dict in this script.")

    # Save
    result = {
        'n_components': n_components,
        'variance_explained': var_ratio.tolist(),
        'projections': projections.tolist(),
        'trait_names': trait_names,
        'pc1_sorted': [(n, float(v)) for n, v in pc1_sorted],
        'pc2_sorted': [(n, float(v)) for n, v in pc2_sorted],
        'human_norm_correlation': human_corr,
        'anthropic_baselines': {
            'pc1_variance': ANTHROPIC_BASELINES['pc1_variance'],
            'pc2_variance': ANTHROPIC_BASELINES['pc2_variance'],
            'pc1_vs_valence': ANTHROPIC_BASELINES['pc1_vs_valence'],
            'pc2_vs_arousal': ANTHROPIC_BASELINES['pc2_vs_arousal'],
        },
    }

    save_results(out_dir, 'pca_analysis', result, compact=True)

    return result


# ============================================================================
# Analysis: Cross-layer RSA (Fig 9)
# ============================================================================

def run_rsa_analysis(vectors_by_layer, trait_names, out_dir):
    """Representational similarity analysis across layers.

    Calls: analysis.vectors.geometry.representational_similarity()
    No custom code needed — fully covered by existing function.
    """
    layers = sorted(vectors_by_layer.keys())
    print(f"\n  [Fig 9] Cross-layer RSA ({len(layers)} layers)...")
    print(f"    Layers: {layers}")

    rsa_matrix, layer_list = representational_similarity(vectors_by_layer)

    print(f"    RSA matrix shape: {rsa_matrix.shape}")

    # Summary: how similar are adjacent layers?
    for i in range(len(layers) - 1):
        sim = rsa_matrix[i, i + 1].item()
        print(f"    Layer {layers[i]} <-> {layers[i+1]}: {sim:.3f}")

    # Save
    result = {
        'rsa_matrix': rsa_matrix.tolist(),
        'layers': layer_list,
        'n_traits': len(trait_names),
    }

    save_results(out_dir, 'rsa_analysis', result, compact=True)

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--layer', type=int, required=True,
                        help='Primary layer for cosine/cluster/PCA analysis')
    parser.add_argument('--rsa-layers', type=str, default=None,
                        help='Comma-separated layers for RSA (default: skip RSA)')
    parser.add_argument('--model-variant', default=None)
    parser.add_argument('--category', default='ant_emotion_concepts',
                        help='Trait category containing the emotion vectors')
    parser.add_argument('--method', default='mean_diff+gm+pc50',
                        help='Vector method name (default: mean_diff+gm+pc50, the fully-denoised Sofroniew vectors). '
                             'Alternatives: mean_diff+gm (no PC denoising), mean_diff (raw), probe, etc.')
    parser.add_argument('--component', default='residual')
    parser.add_argument('--position', default='response[50:]')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of k-means clusters (default: 10)')
    parser.add_argument('--n-components', type=int, default=10,
                        help='Number of PCA components (default: 10)')
    parser.add_argument('--only', type=str, default=None,
                        help='Comma-separated list of analyses to run: cosine,cluster,pca,rsa')
    args = parser.parse_args()

    # Resolve model variant
    model_variant = args.model_variant
    if model_variant is None:
        model_variant = get_default_variant(args.experiment, mode='extraction')
        print(f"Model variant: {model_variant}")

    # Output directory
    out_dir = get_results_dir(args.experiment, 'stage3_geometry')
    print(f"Output: {out_dir}")

    # Determine which analyses to run
    all_analyses = {'cosine', 'cluster', 'pca', 'rsa'}
    if args.only:
        analyses = set(args.only.split(','))
        invalid = analyses - all_analyses
        if invalid:
            parser.error(f"Unknown analyses: {invalid}. Choose from: {all_analyses}")
    else:
        analyses = all_analyses

    # Load vectors at primary layer
    print(f"\nLoading denoised vectors at layer {args.layer}...")
    vectors, trait_names = load_all_emotion_vectors(
        args.experiment, args.category, args.layer, model_variant,
        method=args.method, component=args.component, position=args.position,
    )
    print(f"  Loaded {len(trait_names)} emotion vectors, dim={vectors.shape[1]}")

    results = {}

    # --- Cosine heatmap (Fig 5) ---
    if 'cosine' in analyses:
        results['cosine'] = run_cosine_analysis(vectors, trait_names, out_dir)

    # --- K-means + UMAP (Fig 6) ---
    if 'cluster' in analyses:
        results['cluster'] = run_cluster_analysis(vectors, trait_names, out_dir, k=args.k)

    # --- PCA (Figs 7-8) ---
    if 'pca' in analyses:
        results['pca'] = run_pca_analysis(vectors, trait_names, out_dir, n_components=args.n_components)

    # --- RSA (Fig 9) ---
    if 'rsa' in analyses:
        if args.rsa_layers is None:
            print("\n  Skipping RSA (no --rsa-layers specified)")
        else:
            rsa_layer_list = [int(x) for x in args.rsa_layers.split(',')]
            # Ensure primary layer is included
            if args.layer not in rsa_layer_list:
                rsa_layer_list.append(args.layer)
                rsa_layer_list.sort()

            print(f"\n  Loading vectors at {len(rsa_layer_list)} layers for RSA...")
            vectors_by_layer = {}
            for layer in rsa_layer_list:
                try:
                    vecs, _ = load_all_emotion_vectors(
                        args.experiment, args.category, layer, model_variant,
                        method=args.method, component=args.component, position=args.position,
                    )
                    vectors_by_layer[layer] = vecs
                except FileNotFoundError:
                    print(f"    Skipping layer {layer}: no vectors found")

            if len(vectors_by_layer) >= 2:
                results['rsa'] = run_rsa_analysis(vectors_by_layer, trait_names, out_dir)
            else:
                print("    Need vectors at >= 2 layers for RSA")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Stage 3 Geometry Analysis — Summary")
    print(f"{'='*60}")
    print(f"  Experiment: {args.experiment}")
    print(f"  Layer: {args.layer}")
    print(f"  Emotions: {len(trait_names)}")
    print(f"  Analyses: {', '.join(sorted(results.keys()))}")

    if 'pca' in results:
        pca_res = results['pca']
        var = pca_res['variance_explained']
        print(f"\n  PCA comparison to Anthropic:")
        compare_to_baseline('PC1 variance', var[0], ANTHROPIC_BASELINES['pc1_variance'])
        compare_to_baseline('PC2 variance', var[1], ANTHROPIC_BASELINES['pc2_variance'])
        if pca_res['human_norm_correlation']:
            hc = pca_res['human_norm_correlation']
            compare_to_baseline('PC1 vs valence', hc['pc1_vs_valence'], ANTHROPIC_BASELINES['pc1_vs_valence'])
            compare_to_baseline('PC2 vs arousal', hc['pc2_vs_arousal'], ANTHROPIC_BASELINES['pc2_vs_arousal'])

    # Save run metadata
    meta = {
        'experiment': args.experiment,
        'layer': args.layer,
        'model_variant': model_variant,
        'method': args.method,
        'category': args.category,
        'n_emotions': len(trait_names),
        'analyses_run': sorted(results.keys()),
        'trait_names': trait_names,
    }
    save_results(out_dir, 'run_metadata', meta)

    print(f"\n  Results saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
