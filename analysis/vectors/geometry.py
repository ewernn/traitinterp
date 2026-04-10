"""
Trait vector geometry analysis: clustering, UMAP, representational similarity.

Input: Pre-extracted trait vectors (loaded from experiments/{experiment}/extraction/)
Output: Clustering assignments, UMAP coordinates, RSA matrices, saved to analysis output dir

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

from core.math import pairwise_cosine_matrix, pca


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
