"""
Trait vector geometry analysis: clustering, UMAP, representational similarity,
cosine heatmap ordering, and PCA vs human-norm correlation.

Input: Pre-extracted trait vectors (loaded from experiments/{experiment}/extraction/)
Output: Clustering assignments, UMAP coordinates, RSA matrices, ordered cosine heatmaps,
        PC-vs-human-norm correlation dicts, saved to analysis output dir

Usage:
    python analysis/vectors/geometry.py --experiment {experiment} --analysis cluster --k 10
    python analysis/vectors/geometry.py --experiment {experiment} --analysis umap
    python analysis/vectors/geometry.py --experiment {experiment} --analysis rsa --layers 10,20,30,40
    python analysis/vectors/geometry.py --experiment {experiment} --analysis all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.math import pairwise_cosine_matrix, pca, pearson_correlation


def trait_clusters(vectors: torch.Tensor, k: int = 10, n_init: int = 20, random_state: int = 42):
    """K-means clustering on trait vectors.

    Input: [N, hidden_dim]
    Output: (assignments [N], centroids [k, hidden_dim], inertia float)
    """
    from sklearn.cluster import KMeans
    X = vectors.float().cpu().numpy()
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    km.fit(X)
    return km.labels_, torch.from_numpy(km.cluster_centers_), km.inertia_


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


def pca_norm_correlation(
    projections: torch.Tensor,
    trait_names: List[str],
    norms: Dict[str, Tuple[float, float]],
    min_overlap: int = 10,
) -> Optional[Dict]:
    """Pearson correlation of PC1/PC2 projections with human valence/arousal norms.

    Replicates Sofroniew et al. 2026 Fig 8 (PC1 vs pleasure, PC2 vs arousal). Output
    keys use 1-indexed naming (`pc1_vs_valence`, `pc2_vs_arousal`) to match the
    paper's figure labels and downstream visualization expectations.

    Input:
        projections: [N, n_components] PC projections from `core.math.pca()`
        trait_names: list of N trait names
        norms: {trait_name: (valence, arousal)} dict
        min_overlap: minimum overlapping emotions required; returns None if fewer

    Output: dict with {n_overlapping, emotions, pc1_vs_valence, pc2_vs_arousal,
            pc1_vs_arousal, pc2_vs_valence} or None if overlap < min_overlap
    """
    overlapping = [(i, name) for i, name in enumerate(trait_names) if name in norms]
    if len(overlapping) < min_overlap:
        return None

    indices = [i for i, _ in overlapping]
    names_overlap = [n for _, n in overlapping]

    human_valence = torch.tensor([norms[n][0] for n in names_overlap])
    human_arousal = torch.tensor([norms[n][1] for n in names_overlap])

    pc1 = projections[indices, 0]
    pc2 = projections[indices, 1]

    return {
        'n_overlapping': len(overlapping),
        'emotions': names_overlap,
        'pc1_vs_valence': float(pearson_correlation(pc1, human_valence)),
        'pc2_vs_arousal': float(pearson_correlation(pc2, human_arousal)),
        'pc1_vs_arousal': float(pearson_correlation(pc1, human_arousal)),
        'pc2_vs_valence': float(pearson_correlation(pc2, human_valence)),
    }


def representational_similarity(vectors_by_layer: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
    """Representational similarity analysis across layers.

    Computes cosine similarity of cosine-similarity matrices across layers.

    Input: {layer_idx: [N, hidden_dim]} — same N traits at each layer
    Output: (rsa_matrix [L, L], layer_indices [L])
    """
    layers = sorted(vectors_by_layer.keys())
    sim_matrices = []
    for layer in layers:
        sim = pairwise_cosine_matrix(vectors_by_layer[layer])
        # Extract upper triangle (excluding diagonal) as flat vector
        idx = torch.triu_indices(sim.shape[0], sim.shape[1], offset=1)
        sim_matrices.append(sim[idx[0], idx[1]])

    sim_stack = torch.stack(sim_matrices)  # [L, n_pairs]
    # RSA: cosine similarity between flattened similarity vectors
    rsa = pairwise_cosine_matrix(sim_stack)
    return rsa, layers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--analysis', choices=['cluster', 'umap', 'rsa', 'pca', 'cosine', 'all'], default='all')
    parser.add_argument('--k', type=int, default=10, help='Number of clusters for k-means')
    parser.add_argument('--n-components', type=int, default=10, help='Number of PCA components')
    parser.add_argument('--layers', type=str, default=None, help='Comma-separated layer indices for RSA')
    args = parser.parse_args()

    print(f"Geometry analysis: {args.analysis} on {args.experiment}")
    # Vector loading and output saving would be wired here
    # using utils.paths and utils.vectors


if __name__ == '__main__':
    main()
