"""
Generation primitive with KV caching and hook integration.

Input:
    - model: HuggingFace transformer model
    - input_ids, attention_mask: Tokenized inputs (caller handles tokenization)
    - capture/steering configs

Output:
    - List[SequenceOutput] for batched generation
    - Generator[TokenOutput] for streaming

Usage:
    from core.generation import HookedGenerator, CaptureConfig, SteeringConfig

    gen = HookedGenerator(model)

    # Batched inference with capture
    results = gen.generate(input_ids, attention_mask, capture=CaptureConfig(layers=[14,15]))

    # Streaming for UI
    for tok in gen.stream(input_ids, attention_mask, capture=...):
        yield tok.token_id, tok.activations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Generator, Optional
import torch
from torch import Tensor

from core.hooks import HookManager, SteeringHook, get_hook_path


def get_layer_path_prefix(model) -> str:
    """Get the hook path prefix to transformer layers, handling PeftModel wrapper.

    Returns hook path prefix like "model.layers" or "base_model.model.model.layers".
    """
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return "model.language_model.layers"
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        if type(model).__name__ != type(model.base_model).__name__:
            return "base_model.model.model.layers"
    return "model.layers"


@dataclass
class CaptureConfig:
    """What to capture during generation."""
    layers: List[int] = None  # None = all layers
    components: List[str] = field(default_factory=lambda: ['residual'])


@dataclass
class SteeringConfig:
    """Steering to apply during generation."""
    vector: Tensor
    layer: int
    component: str = 'residual'
    coefficient: float = 1.0


@dataclass
class TokenOutput:
    """Single token output for streaming."""
    token_id: int
    step: int
    activations: Optional[Dict[int, Dict[str, Tensor]]] = None  # layer -> component -> [hidden]


@dataclass
class SequenceOutput:
    """Per-sequence output for batched generation."""
    token_ids: List[int]
    # Stacked activations: layer -> component -> [n_tokens, hidden]
    activations: Optional[Dict[int, Dict[str, Tensor]]] = None


class HookedGenerator:
    """
    Generation primitive with KV caching and hook integration.

    Single source of truth for:
    - Inference with activation capture
    - Steering during generation
    - Streaming for UI

    NOT for extraction (use MultiLayerCapture for that).
    """

    def __init__(self, model):
        self.model = model

        # Handle nested config (multimodal models)
        config = model.config
        if hasattr(config, 'text_config'):
            config = config.text_config

        self.n_layers = config.num_hidden_layers
        self.layer_prefix = get_layer_path_prefix(model)

        # Handle EOS as int or list
        eos = getattr(config, 'eos_token_id', None)
        if eos is None:
            self.default_stop_ids = set()
        elif isinstance(eos, (list, tuple)):
            self.default_stop_ids = set(eos)
        else:
            self.default_stop_ids = {eos}

    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        capture: CaptureConfig = None,
        steering: List[SteeringConfig] = None,
        stop_token_ids: Set[int] = None,
    ) -> List[SequenceOutput]:
        """
        Generate for batch of sequences. Returns per-sequence results.
        """
        batch_size = input_ids.shape[0]
        steps = list(self._generate_steps(
            input_ids, attention_mask, max_new_tokens, temperature,
            capture, steering, stop_token_ids
        ))
        return self._package_per_sequence(steps, batch_size, capture)

    def stream(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        capture: CaptureConfig = None,
        steering: List[SteeringConfig] = None,
        stop_token_ids: Set[int] = None,
    ) -> Generator[TokenOutput, None, None]:
        """
        Stream tokens for single sequence (UI use case).
        """
        assert input_ids.shape[0] == 1, "stream() only supports batch_size=1"

        for step in self._generate_steps(
            input_ids, attention_mask, max_new_tokens, temperature,
            capture, steering, stop_token_ids
        ):
            # Extract single sequence
            acts = None
            if step.activations:
                acts = {
                    layer: {comp: t[0] for comp, t in comps.items()}
                    for layer, comps in step.activations.items()
                }
            yield TokenOutput(
                token_id=step.token_ids[0].item(),
                step=step.step,
                activations=acts,
            )

    def _generate_steps(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        max_new_tokens: int,
        temperature: float,
        capture: CaptureConfig,
        steering: List[SteeringConfig],
        stop_token_ids: Set[int],
    ) -> Generator:
        """
        Internal generator yielding per-step outputs.

        Key insight for skip-first fix:
        - Step 0: prompt forward → captures state that PRODUCED token 0 → sample token 0
        - Step i: forward token i-1 → captures state that PRODUCED token i → sample token i

        Each yield gives (token, state_that_produced_it). Clean 1:1 mapping.
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Build stop token set
        stop_ids = set(stop_token_ids or [])
        stop_ids.update(self.default_stop_ids)

        # === PROMPT PHASE ===
        # Forward prompt, capture state that produced token 0, get KV cache
        activations = self._create_activation_storage(capture) if capture else None

        with HookManager(self.model) as hooks:
            if capture:
                self._setup_capture_hooks(hooks, activations, capture)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )

        past_key_values = outputs.past_key_values
        next_ids = self._sample(outputs.logits[:, -1, :], temperature)

        # Yield token 0 with state that produced it
        yield _StepOutput(
            token_ids=next_ids,
            activations=self._extract_activations(activations, capture),
            step=0,
        )

        # Check stop
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        for b in range(batch_size):
            if next_ids[b].item() in stop_ids:
                active[b] = False

        if not active.any():
            return

        # === STEERING SETUP (persistent across response steps) ===
        steering_hooks = []
        if steering:
            for cfg in steering:
                path = get_hook_path(cfg.layer, cfg.component, prefix=self.layer_prefix)
                hook = SteeringHook(self.model, cfg.vector, path, coefficient=cfg.coefficient)
                hook.__enter__()
                steering_hooks.append(hook)

        try:
            # === RESPONSE PHASE ===
            context = next_ids.unsqueeze(1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
            ], dim=1)

            for step in range(1, max_new_tokens):
                activations = self._create_activation_storage(capture) if capture else None

                with HookManager(self.model) as hooks:
                    if capture:
                        self._setup_capture_hooks(hooks, activations, capture)
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=context,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )

                past_key_values = outputs.past_key_values
                next_ids = self._sample(outputs.logits[:, -1, :], temperature)

                # Update active mask
                for b in range(batch_size):
                    if active[b] and next_ids[b].item() in stop_ids:
                        active[b] = False

                yield _StepOutput(
                    token_ids=next_ids,
                    activations=self._extract_activations(activations, capture),
                    step=step,
                )

                if not active.any():
                    break

                # Update for next step
                context = next_ids.unsqueeze(1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
                ], dim=1)

        finally:
            for hook in steering_hooks:
                hook.__exit__(None, None, None)

    def _create_activation_storage(self, config: CaptureConfig) -> Dict:
        """Create empty storage for activation capture."""
        layers = config.layers if config.layers is not None else range(self.n_layers)
        return {layer: {comp: [] for comp in config.components} for layer in layers}

    def _setup_capture_hooks(self, hooks: HookManager, storage: Dict, config: CaptureConfig):
        """Set up hooks to capture activations."""
        layers = config.layers if config.layers is not None else range(self.n_layers)

        for layer in layers:
            for component in config.components:
                path = get_hook_path(layer, component, prefix=self.layer_prefix, model=self.model)

                def make_hook(l, c):
                    def hook(module, inp, out):
                        out_t = out[0] if isinstance(out, tuple) else out
                        # Capture last position (with KV cache, output is [batch, 1, hidden])
                        storage[l][c].append(out_t[:, -1, :].detach().cpu())
                    return hook

                hooks.add_forward_hook(path, make_hook(layer, component))

    def _extract_activations(self, storage: Dict, config: CaptureConfig) -> Optional[Dict]:
        """Extract activations from storage (single step)."""
        if storage is None:
            return None

        result = {}
        layers = config.layers if config.layers is not None else range(self.n_layers)
        for layer in layers:
            result[layer] = {}
            for comp in config.components:
                if storage[layer][comp]:
                    result[layer][comp] = storage[layer][comp][0]  # [batch, hidden]
        return result

    def _package_per_sequence(
        self,
        steps: List,
        batch_size: int,
        config: CaptureConfig,
    ) -> List[SequenceOutput]:
        """Package per-step outputs into per-sequence results."""
        results = []

        for b in range(batch_size):
            token_ids = [step.token_ids[b].item() for step in steps]

            activations = None
            if config and steps[0].activations:
                layers = config.layers if config.layers is not None else range(self.n_layers)
                activations = {}
                for layer in layers:
                    activations[layer] = {}
                    for comp in config.components:
                        # Stack across tokens: [n_tokens, hidden]
                        tensors = [step.activations[layer][comp][b] for step in steps]
                        activations[layer][comp] = torch.stack(tensors, dim=0)

            results.append(SequenceOutput(token_ids=token_ids, activations=activations))

        return results

    def _sample(self, logits: Tensor, temperature: float) -> Tensor:
        """Sample next tokens from logits."""
        if temperature == 0:
            return logits.argmax(dim=-1)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)


@dataclass
class _StepOutput:
    """Internal per-step output."""
    token_ids: Tensor  # [batch]
    activations: Optional[Dict[int, Dict[str, Tensor]]]  # layer -> comp -> [batch, hidden]
    step: int
