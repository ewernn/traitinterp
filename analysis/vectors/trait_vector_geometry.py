"""
Precompute trait-vector geometry for the Extraction view's Vector Geometry subsection.

Input:  Extracted trait vectors under experiments/{experiment}/extraction/.
Output: Single JSON at experiments/{experiment}/extraction/vector_geometry.json
        with pairwise cosine similarity + 2D PCA coords per (method, layer).
Usage:
    python analysis/vectors/trait_vector_geometry.py --experiment live-chat
    python analysis/vectors/trait_vector_geometry.py --experiment emotion_set --component residual --position response_all
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.math import pairwise_cosine_matrix, pca
from utils.paths import get as get_path


def _discover_traits(extraction_base: Path) -> list[str]:
    """Return sorted list of 'category/trait' strings present under extraction/."""
    traits = []
    for cat_dir in sorted(extraction_base.iterdir()):
        if not cat_dir.is_dir():
            continue
        for trait_dir in sorted(cat_dir.iterdir()):
            if trait_dir.is_dir():
                traits.append(f"{cat_dir.name}/{trait_dir.name}")
    return traits


def _vector_path(base: Path, trait: str, model_variant: str, position: str,
                 component: str, method: str, layer: int) -> Path:
    cat, name = trait.split("/", 1)
    return (base / cat / name / model_variant / "vectors" / position / component / method
            / f"layer{layer}.pt")


def _collect_vectors(extraction_base: Path, traits: list[str], model_variant: str,
                     position: str, component: str):
    """For each (method, layer) present across traits, stack vectors that exist for
    ALL traits at that slice. Returns dict[method][layer] = (tensor NxD, kept_traits)."""
    slices: dict[str, dict[int, tuple[torch.Tensor, list[str]]]] = {}

    # Discover (method, layer) candidates from the first trait that has data.
    candidate_slices: set[tuple[str, int]] = set()
    for trait in traits:
        cat, name = trait.split("/", 1)
        vec_root = extraction_base / cat / name / model_variant / "vectors" / position / component
        if not vec_root.exists():
            continue
        for method_dir in vec_root.iterdir():
            if not method_dir.is_dir():
                continue
            for layer_file in method_dir.glob("layer*.pt"):
                try:
                    layer = int(layer_file.stem.replace("layer", ""))
                except ValueError:
                    continue
                candidate_slices.add((method_dir.name, layer))

    for method, layer in sorted(candidate_slices):
        vecs = []
        kept = []
        for trait in traits:
            fp = _vector_path(extraction_base, trait, model_variant, position, component, method, layer)
            if not fp.exists():
                continue
            v = torch.load(fp, weights_only=True)
            if v.ndim != 1:
                continue
            vecs.append(v.float())
            kept.append(trait)
        if len(kept) < 2:
            continue
        slices.setdefault(method, {})[layer] = (torch.stack(vecs), kept)
    return slices


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model-variant", default=None,
                        help="Default: experiment config `defaults.extraction`.")
    parser.add_argument("--component", default="residual")
    parser.add_argument("--position", default="response_all")
    args = parser.parse_args()

    config_path = get_path("experiments.config", experiment=args.experiment)
    with open(config_path) as f:
        config = json.load(f)
    model_variant = args.model_variant or config.get("defaults", {}).get("extraction", "instruct")

    extraction_base = get_path("extraction.base", experiment=args.experiment)
    traits = _discover_traits(extraction_base)
    if len(traits) < 2:
        print(f"Need at least 2 traits; found {len(traits)}.")
        return

    print(f"Discovered {len(traits)} traits under {extraction_base}")
    slices = _collect_vectors(extraction_base, traits, model_variant, args.position, args.component)
    if not slices:
        print("No vector slices found — check --model-variant / --position / --component.")
        return

    # Build output: per-method -> per-layer -> { traits, cos_sim, coords_2d }
    data: dict[str, dict[str, dict]] = {}
    all_layers: set[int] = set()
    for method, per_layer in slices.items():
        data[method] = {}
        for layer, (vecs, kept_traits) in per_layer.items():
            cos = pairwise_cosine_matrix(vecs).tolist()
            # PCA → 2D projections. For N<3 the second component is degenerate,
            # so pad to always return [[x, y], ...].
            n_req = min(2, max(1, vecs.shape[0] - 1))
            _, _, coords = pca(vecs, n_components=n_req)
            if coords.shape[1] < 2:
                pad = torch.zeros(coords.shape[0], 2 - coords.shape[1])
                coords = torch.cat([coords, pad], dim=1)
            data[method][str(layer)] = {
                "traits": kept_traits,
                "cos_sim": cos,
                "coords_2d": coords.tolist(),
            }
            all_layers.add(layer)

    out = {
        "experiment": args.experiment,
        "model_variant": model_variant,
        "component": args.component,
        "position": args.position,
        "methods": sorted(data.keys()),
        "layers": sorted(all_layers),
        "data": data,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = get_path("extraction_eval.vector_geometry", experiment=args.experiment)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    n_slices = sum(len(per_layer) for per_layer in data.values())
    print(f"Wrote {out_path} ({n_slices} slices, {len(traits)} traits, {len(data)} methods)")


if __name__ == "__main__":
    main()
