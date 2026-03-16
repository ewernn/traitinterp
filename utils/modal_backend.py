"""
Modal cloud GPU backend (stub).

Runs model loading and inference on Modal cloud GPUs.
Will support the same GenerationBackend interface as LocalBackend/ServerBackend.

Usage:
    python extraction/run_extraction_pipeline.py --experiment exp --traits cat/trait --backend modal

GPU options (pass to --gpu flag or ModalBackend.from_experiment(gpu=...)):
    String          VRAM     $/sec       $/hr    Notes
    "T4"            16 GB    $0.000164   $0.59
    "L4"            24 GB    $0.000222   $0.80
    "A10G"          24 GB    $0.000306   $1.10
    "A100-40GB"     40 GB    $0.000583   $2.10
    "A100-80GB"     80 GB    $0.000694   $2.50
    "RTX-PRO-6000"  48 GB    $0.000842   $3.03
    "H100"          80 GB    $0.001097   $3.95   may auto-upgrade to H200
    "H100!"         80 GB    $0.001097   $3.95   forces H100, no upgrade
    "H200"         141 GB    $0.001261   $4.54
    "B200"         192 GB    $0.001736   $6.25
    "ANY"           varies   varies      varies  cheapest available

Multi-GPU: append :N, e.g. "H100:4" for 4x H100.
CPU/Memory always included: $0.047/core/hr + $0.008/GiB/hr.
Billed per-second with no minimum.
"""

from utils.backends import GenerationBackend


class ModalBackend(GenerationBackend):
    """Backend that runs on Modal cloud GPUs.

    TODO: Implement with lazy modal import, @modal.function decorators,
    volume mounts for model weights, and GPU selection.
    """

    # Modal GPU string → (VRAM_GB, $/hr)
    GPU_OPTIONS = {
        'T4':            (16,  0.59),
        'L4':            (24,  0.80),
        'A10G':          (24,  1.10),
        'A100-40GB':     (40,  2.10),
        'A100-80GB':     (80,  2.50),
        'RTX-PRO-6000':  (48,  3.03),
        'H100':          (80,  3.95),
        'H200':          (141, 4.54),
        'B200':          (192, 6.25),
        'ANY':           (None, None),
    }

    @classmethod
    def from_experiment(cls, experiment: str, variant: str = None, gpu: str = 'A100-80GB'):
        """Create from experiment config, running on Modal.

        Args:
            experiment: Experiment name
            variant: Model variant
            gpu: Modal GPU string (see GPU_OPTIONS). Append :N for multi-GPU.
        """
        raise NotImplementedError(
            "ModalBackend not yet implemented. "
            "Use --backend local or --backend server instead.\n"
            f"Available GPU options: {', '.join(cls.GPU_OPTIONS.keys())}"
        )

    @property
    def n_layers(self):
        raise NotImplementedError

    @property
    def hidden_dim(self):
        raise NotImplementedError

    @property
    def device(self):
        raise NotImplementedError

    def generate(self, prompts, config=None, steering=None):
        raise NotImplementedError

    def generate_with_capture(self, prompts, config=None, capture=None, steering=None):
        raise NotImplementedError

    def stream(self, prompt, config=None, capture=None, steering=None):
        raise NotImplementedError

    def forward_with_capture(self, input_ids, attention_mask, capture):
        raise NotImplementedError
