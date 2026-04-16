"""
Passive massive-dim calibration via forward-pass hooks.

Input:  Residual activations from prefill forward passes on any prompt set.
Output: experiments/{exp}/inference/{variant}/massive_activations/calibration.json
        (same schema as analysis/vectors/massive_activations.py calibration mode)

Usage:
    # Skip if already have enough data
    if MassiveDimCollector.should_skip(experiment, variant):
        ...
    collector = MassiveDimCollector(model, tokenizer, experiment, variant)
    collector.register()           # attach per-layer forward hooks
    ... run whatever generation / forward passes you would anyway ...
    collector.finalize()           # write JSON, remove hooks

Design notes:
 - Only prefill passes are captured (shape[1] > 1). Decode-steps (shape[1] == 1)
   are ignored so there is no KV-cache overhead during generation.
 - Hooks self-remove once `target_tokens` is reached. Re-running inference after
   the file exists is a no-op (see `should_skip`).
 - Schema mirrors `compute_calibration_stats_streaming` so the viz's
   /api/experiments/.../inference/massive_activations/calibration consumers work unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from utils.paths import get as get_path
from core.hooks import _get_layers  # reuse architecture detection
from core.math import cosine_similarity


class MassiveDimCollector:
    """Accumulate per-layer residual statistics across forward passes."""

    TARGET_TOKENS = 5000

    def __init__(
        self,
        model,
        tokenizer,
        experiment: str,
        model_variant: str,
        target_tokens: int = TARGET_TOKENS,
        top_k: int = 5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.experiment = experiment
        self.model_variant = model_variant
        self.target_tokens = target_tokens
        self.top_k = top_k

        self.n_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size

        self.layer_sums = {l: torch.zeros(self.hidden_dim) for l in range(self.n_layers)}
        self.layer_norm_sums = {l: 0.0 for l in range(self.n_layers)}
        self.layer_counts = {l: 0 for l in range(self.n_layers)}
        self.alignment_sums = {l: 0.0 for l in range(self.n_layers)}
        self.alignment_counts = {l: 0 for l in range(self.n_layers)}

        self.tokens_seen = 0
        self.n_batches = 0
        self._handles: list = []
        self._active = True

        # Resume accumulation from prior runs if present
        self._load_prior_state()

    # ------------------------------------------------------------------ public

    @classmethod
    def output_path(cls, experiment: str, model_variant: str) -> Path:
        return Path(get_path(
            'inference.massive_activations',
            experiment=experiment, model_variant=model_variant, prompt_set='calibration',
        ))

    @classmethod
    def should_skip(cls, experiment: str, model_variant: str,
                    target_tokens: int = TARGET_TOKENS) -> bool:
        """True if calibration.json exists with >= target_tokens accumulated."""
        path = cls.output_path(experiment, model_variant)
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)
        return data.get('aggregate', {}).get('tokens_seen', 0) >= target_tokens

    def _load_prior_state(self) -> None:
        """Resume from `_state` block of an existing calibration.json if present."""
        path = self.output_path(self.experiment, self.model_variant)
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        state = data.get('_state')
        if not state:
            return
        for l_str, vec in state['layer_sums'].items():
            self.layer_sums[int(l_str)] = torch.tensor(vec, dtype=torch.float32)
        for l_str, v in state['layer_norm_sums'].items():
            self.layer_norm_sums[int(l_str)] = v
        for l_str, v in state['layer_counts'].items():
            self.layer_counts[int(l_str)] = v
        for l_str, v in state['alignment_sums'].items():
            self.alignment_sums[int(l_str)] = v
        for l_str, v in state['alignment_counts'].items():
            self.alignment_counts[int(l_str)] = v
        self.tokens_seen = state.get('tokens_seen', 0)
        self.n_batches = state.get('n_batches', 0)

    def register(self) -> None:
        """Attach per-layer forward hooks. Idempotent — calling twice is a no-op."""
        if self._handles:
            return
        layers = _get_layers(self.model)
        for l in range(self.n_layers):
            self._handles.append(
                layers[l].register_forward_hook(self._make_hook(l))
            )

    def finalize(self) -> Optional[Path]:
        """Remove hooks and write JSON. Returns output path, or None if nothing captured."""
        self._remove_hooks()
        if self.tokens_seen < 100:
            return None

        aggregate = self._build_aggregate()
        state = {
            'layer_sums': {int(l): self.layer_sums[l].tolist() for l in range(self.n_layers)},
            'layer_norm_sums': {int(l): self.layer_norm_sums[l] for l in range(self.n_layers)},
            'layer_counts': {int(l): self.layer_counts[l] for l in range(self.n_layers)},
            'alignment_sums': {int(l): self.alignment_sums[l] for l in range(self.n_layers)},
            'alignment_counts': {int(l): self.alignment_counts[l] for l in range(self.n_layers)},
            'tokens_seen': self.tokens_seen,
            'n_batches': self.n_batches,
        }
        output = {
            'experiment': self.experiment,
            'model': getattr(self.model.config, 'name_or_path', 'unknown'),
            'model_variant': self.model_variant,
            'prompt_set': 'calibration',
            'is_calibration': True,
            'source': 'inference_passive',
            'aggregate': aggregate,
            '_state': state,
        }
        path = self.output_path(self.experiment, self.model_variant)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        return path

    # ----------------------------------------------------------------- internal

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inp, out):
            if not self._active:
                return
            h = out[0] if isinstance(out, tuple) else out
            # Only prefill passes (seq_len > 1). Skip decode steps.
            if h.dim() < 3 or h.shape[1] <= 1:
                return
            self._process(layer_idx, h.detach().cpu().float())
        return hook_fn

    def _process(self, layer_idx: int, h: torch.Tensor) -> None:
        """h: [batch, seq, hidden] on CPU float32."""
        batch_size, seq_len, _ = h.shape
        for b in range(batch_size):
            toks = h[b]  # [seq, hidden]
            self.layer_sums[layer_idx] += toks.sum(dim=0)
            self.layer_norm_sums[layer_idx] += toks.norm(dim=1).sum().item()
            self.layer_counts[layer_idx] += seq_len

            mean_dir = toks.mean(dim=0)
            mean_dir_norm = mean_dir / (mean_dir.norm() + 1e-8)
            token_norms = toks.norm(dim=1, keepdim=True)
            token_normalized = toks / (token_norms + 1e-8)
            cosines = (token_normalized @ mean_dir_norm)
            self.alignment_sums[layer_idx] += cosines.mean().item()
            self.alignment_counts[layer_idx] += 1

        # Track tokens once per batch — counted off layer 0 to avoid multi-counting across layers
        if layer_idx == 0:
            self.tokens_seen += batch_size * seq_len
            self.n_batches += 1
            if self.tokens_seen >= self.target_tokens:
                self._active = False

    def _remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _build_aggregate(self) -> dict:
        """Produce the same schema as compute_calibration_stats_streaming."""
        # Layer means + norms
        layer_means, layer_norms_out = {}, {}
        for l in range(self.n_layers):
            if self.layer_counts[l] == 0:
                continue
            layer_means[l] = self.layer_sums[l] / self.layer_counts[l]
            layer_norms_out[l] = round(self.layer_norm_sums[l] / self.layer_counts[l], 1)

        # Inter-layer cosine similarity
        consecutive_cosine = {}
        layers_sorted = sorted(layer_means.keys())
        for i in range(len(layers_sorted) - 1):
            a, b = layers_sorted[i], layers_sorted[i + 1]
            consecutive_cosine[a] = round(
                cosine_similarity(layer_means[a], layer_means[b]).item(), 4
            )

        # Top-k dims per layer + normalized magnitudes
        top_dims_by_layer, all_candidate_dims = {}, set()
        for l, mean_vec in layer_means.items():
            _, top_dims = mean_vec.abs().topk(self.top_k)
            top_dims_list = top_dims.tolist()
            top_dims_by_layer[l] = top_dims_list
            all_candidate_dims.update(top_dims_list)

        dim_magnitude_by_layer = {}
        for dim in sorted(all_candidate_dims):
            mags = []
            for l in sorted(layer_means.keys()):
                mean_vec = layer_means[l]
                layer_avg = mean_vec.abs().mean().item()
                normalized = abs(mean_vec[dim].item()) / layer_avg if layer_avg > 0 else 0
                mags.append(round(normalized, 3))
            dim_magnitude_by_layer[dim] = mags

        mean_alignment = {}
        for l in range(self.n_layers):
            if self.alignment_counts[l] > 0:
                mean_alignment[l] = round(
                    self.alignment_sums[l] / self.alignment_counts[l], 4
                )

        return {
            'n_prompts': self.n_batches,
            'tokens_seen': self.tokens_seen,
            'mean_alignment_by_layer': {int(k): v for k, v in mean_alignment.items()},
            'top_dims_by_layer': {int(k): v for k, v in top_dims_by_layer.items()},
            'dim_magnitude_by_layer': {int(k): v for k, v in dim_magnitude_by_layer.items()},
            'layer_norms': {int(k): v for k, v in layer_norms_out.items()},
            'consecutive_cosine': {int(k): v for k, v in consecutive_cosine.items()},
        }
