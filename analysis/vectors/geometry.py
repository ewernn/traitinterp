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

from core.math import cosine_similarity, pairwise_cosine_matrix, pca, pearson_correlation, project_out_subspace


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


def vector_set_comparison(
    vectors_a: Dict[str, torch.Tensor],
    vectors_b: Dict[str, torch.Tensor],
    vectors_c: Optional[Dict[str, torch.Tensor]] = None,
) -> dict:
    """Compare two (or three) named vector sets sharing the same key space.

    Computes same-key cosine similarity, cross-key cosine matrix, and
    orthogonalizes vectors_a against the PCA subspace of vectors_b to measure
    retained norm. If vectors_c is provided, also computes cross-key cosine
    between vectors_c and vectors_b.

    Input:
        vectors_a: {name: [hidden_dim]} — primary set (e.g., deflection probes)
        vectors_b: {name: [hidden_dim]} — reference set (e.g., story probes)
        vectors_c: optional third set (e.g., displayed-emotion probes)

    Output: dict with same_emotion_cosine, cross_emotion_matrix,
            retained_norm_after_orthogonalization, etc.
    """
    common = sorted(set(vectors_a.keys()) & set(vectors_b.keys()))
    if not common:
        return {"error": "No common keys between vectors_a and vectors_b"}

    # Same-key cosine similarity
    same_key_cos = {}
    for key in common:
        cos = cosine_similarity(vectors_a[key], vectors_b[key]).item()
        same_key_cos[key] = round(cos, 4)

    # Cross-key cosine matrix: vectors_a[k1] vs vectors_b[k2]
    cross_matrix = {}
    for k1 in common:
        cross_matrix[k1] = {}
        for k2 in common:
            cos = cosine_similarity(vectors_a[k1], vectors_b[k2]).item()
            cross_matrix[k1][k2] = round(cos, 4)

    # Third-set similarity (if available)
    c_similarity = {}
    if vectors_c:
        for k1 in common:
            if k1 in vectors_c:
                c_similarity[k1] = {}
                for k2 in common:
                    cos = cosine_similarity(vectors_c[k1], vectors_b[k2]).item()
                    c_similarity[k1][k2] = round(cos, 4)

    # Orthogonalize vectors_a against full vectors_b PCA space, measure retained norm
    b_matrix = torch.stack([vectors_b[k] for k in common])  # [n, hidden]
    retained_norms = {}
    for key in common:
        vec_a = vectors_a[key]
        original_norm = vec_a.norm().item()

        n_components = min(len(common), b_matrix.shape[0])
        components, _, _ = pca(b_matrix, n_components=n_components)
        orthogonalized = project_out_subspace(vec_a, components)
        new_norm = orthogonalized.norm().item()

        retained_norms[key] = round(new_norm / original_norm, 4) if original_norm > 1e-8 else 0.0

    avg_retained = np.mean(list(retained_norms.values()))

    return {
        "same_emotion_cosine": same_key_cos,
        "mean_same_emotion_cosine": round(float(np.mean(list(same_key_cos.values()))), 4),
        "cross_emotion_matrix": cross_matrix,
        "displayed_similarity": c_similarity if c_similarity else None,
        "retained_norm_after_orthogonalization": retained_norms,
        "mean_retained_norm": round(float(avg_retained), 4),
        "n_common_emotions": len(common),
    }


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
