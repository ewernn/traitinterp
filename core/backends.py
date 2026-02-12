"""
Generation backend abstraction.

Provides unified interface for local and remote generation with steering/capture.

Usage:
    # Local backend
    backend = LocalBackend.from_model(model, tokenizer)

    # From experiment config
    backend = LocalBackend.from_experiment("gemma-2-2b", variant="instruct")

    # Generation
    responses = backend.generate(prompts, max_new_tokens=256)

    # Generation with steering
    responses = backend.generate(
        prompts,
        steering=[SteeringSpec(layer=16, vector=vec, coefficient=1.5)]
    )

    # Generation with capture
    results = backend.generate_with_capture(
        prompts,
        capture=CaptureSpec(layers=[10, 12, 14], components=['residual'])
    )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Set, Union
import torch

# =============================================================================
# Specs (backend-agnostic, serializable)
# =============================================================================


@dataclass
class SteeringSpec:
    """Declarative steering specification."""
    layer: int
    vector: torch.Tensor  # Will be moved to correct device by backend
    coefficient: float = 1.0
    component: str = "residual"


@dataclass
class CaptureSpec:
    """Declarative capture specification."""
    layers: Optional[List[int]] = None  # None = all layers
    components: List[str] = field(default_factory=lambda: ["residual"])


@dataclass
class GenerationConfig:
    """Generation parameters."""
    max_new_tokens: int = 256
    temperature: float = 0.0
    stop_token_ids: Optional[Set[int]] = None


# =============================================================================
# Results (backend-agnostic)
# =============================================================================


@dataclass
class CaptureResult:
    """Result from generation with capture."""
    prompt: str
    response: str
    prompt_tokens: List[str]
    response_tokens: List[str]
    # layer -> component -> [n_tokens, hidden_dim]
    prompt_activations: Dict[int, Dict[str, torch.Tensor]]
    response_activations: Dict[int, Dict[str, torch.Tensor]]


@dataclass
class TokenResult:
    """Single token result for streaming."""
    token: str
    token_id: int
    # layer -> component -> [hidden_dim]
    activations: Optional[Dict[int, Dict[str, torch.Tensor]]] = None


# =============================================================================
# Backend Interface
# =============================================================================


class GenerationBackend(ABC):
    """
    Abstract interface for generation backends.

    Provides unified API for:
    - Simple generation
    - Generation with steering vectors
    - Generation with activation capture
    - Token streaming for UIs
    """

    @property
    @abstractmethod
    def n_layers(self) -> int:
        """Number of transformer layers."""
        pass

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Hidden dimension size."""
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Model device."""
        pass

    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        config: GenerationConfig = None,
        steering: List[SteeringSpec] = None,
    ) -> List[str]:
        """
        Generate responses for prompts.

        Args:
            prompts: Input prompts (will be formatted with chat template if available)
            config: Generation parameters
            steering: Optional steering vectors to apply

        Returns:
            List of generated response strings
        """
        pass

    @abstractmethod
    def generate_with_capture(
        self,
        prompts: List[str],
        config: GenerationConfig = None,
        capture: CaptureSpec = None,
        steering: List[SteeringSpec] = None,
    ) -> List[CaptureResult]:
        """
        Generate with activation capture.

        Args:
            prompts: Input prompts
            config: Generation parameters
            capture: What to capture (layers, components)
            steering: Optional steering vectors

        Returns:
            List of CaptureResult with activations
        """
        pass

    @abstractmethod
    def stream(
        self,
        prompt: str,
        config: GenerationConfig = None,
        capture: CaptureSpec = None,
        steering: List[SteeringSpec] = None,
    ) -> Generator[TokenResult, None, None]:
        """
        Stream tokens one at a time (for chat UI).

        Args:
            prompt: Single input prompt
            config: Generation parameters
            capture: What to capture per token
            steering: Optional steering vectors

        Yields:
            TokenResult for each generated token
        """
        pass

    @abstractmethod
    def forward_with_capture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        capture: CaptureSpec,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Single forward pass with capture (no generation).

        Used for activation extraction from existing text.

        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            capture: What to capture

        Returns:
            layer -> component -> [batch, seq, hidden]
        """
        pass

    def calculate_batch_size(
        self,
        max_seq_len: int,
        mode: str = "generation",
    ) -> int:
        """
        Calculate safe batch size given sequence length.

        Args:
            max_seq_len: Maximum sequence length in batch
            mode: "generation" or "extraction" (generation needs more memory)

        Returns:
            Recommended batch size
        """
        # Default implementation - backends can override
        from utils.generation import calculate_max_batch_size
        return calculate_max_batch_size(self._model, max_seq_len, mode=mode)


# =============================================================================
# Local Backend Implementation
# =============================================================================


class LocalBackend(GenerationBackend):
    """Local PyTorch backend using HookedGenerator."""

    def __init__(self, model, tokenizer, use_chat_template: bool = None):
        """
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            use_chat_template: Override chat template usage. None = auto-detect from tokenizer.
        """
        self._model = model
        self._tokenizer = tokenizer

        # Auto-detect from tokenizer if not specified
        if use_chat_template is None:
            self._use_chat_template = tokenizer.chat_template is not None
        else:
            self._use_chat_template = use_chat_template

        # Cache model info
        config = model.config
        if hasattr(config, 'text_config'):
            config = config.text_config
        self._n_layers = config.num_hidden_layers
        self._hidden_dim = config.hidden_size

    @classmethod
    def from_model(cls, model, tokenizer, use_chat_template: bool = None) -> "LocalBackend":
        """Create from already-loaded model."""
        return cls(model, tokenizer, use_chat_template=use_chat_template)

    @classmethod
    def from_experiment(
        cls,
        experiment: str,
        variant: str = None,
        device: str = "auto",
        use_chat_template: bool = None,
    ) -> "LocalBackend":
        """
        Create from experiment config.

        Args:
            experiment: Experiment name
            variant: Model variant (default: from experiment config)
            device: Device to load model on
            use_chat_template: Override chat template usage. None = use experiment config
                               or auto-detect from tokenizer.
        """
        from utils.model import load_model_with_lora
        from utils.paths import get_default_variant, load_experiment_config

        if variant is None:
            variant = get_default_variant(experiment, mode='application')

        config = load_experiment_config(experiment)
        model_variants = config.get('model_variants', {})
        variant_config = model_variants.get(variant, {})

        model_name = variant_config.get('model')
        lora_adapter = variant_config.get('lora')

        model, tokenizer = load_model_with_lora(
            model_name,
            lora_adapter=lora_adapter,
            device=device,
        )

        # Resolve use_chat_template: explicit param > experiment config > auto-detect
        if use_chat_template is None:
            use_chat_template = config.get('use_chat_template')  # May still be None

        return cls(model, tokenizer, use_chat_template=use_chat_template)

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def device(self) -> torch.device:
        return next(self._model.parameters()).device

    @property
    def tokenizer(self):
        """Access tokenizer for formatting (avoid if possible)."""
        return self._tokenizer

    @property
    def model(self):
        """Access model directly (for hooks that need it)."""
        return self._model

    def generate(
        self,
        prompts: List[str],
        config: GenerationConfig = None,
        steering: List[SteeringSpec] = None,
    ) -> List[str]:
        from utils.generation import generate_batch
        from utils.model import format_prompt
        from core import SteeringHook, MultiLayerSteeringHook, get_hook_path

        config = config or GenerationConfig()

        # Format prompts
        formatted = [
            format_prompt(p, self._tokenizer, use_chat_template=self._use_chat_template)
            for p in prompts
        ]

        # Setup steering if provided
        if steering:
            # Move vectors to device
            steering_configs = [
                (s.layer, s.vector.to(self.device), s.coefficient, s.component)
                for s in steering
            ]

            if len(steering_configs) == 1:
                layer, vec, coef, comp = steering_configs[0]
                path = get_hook_path(layer, comp, model=self._model)
                with SteeringHook(self._model, vec, path, coefficient=coef):
                    return generate_batch(
                        self._model, self._tokenizer, formatted,
                        max_new_tokens=config.max_new_tokens,
                        temperature=config.temperature,
                    )
            else:
                with MultiLayerSteeringHook(self._model, steering_configs):
                    return generate_batch(
                        self._model, self._tokenizer, formatted,
                        max_new_tokens=config.max_new_tokens,
                        temperature=config.temperature,
                    )
        else:
            return generate_batch(
                self._model, self._tokenizer, formatted,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )

    def generate_with_capture(
        self,
        prompts: List[str],
        config: GenerationConfig = None,
        capture: CaptureSpec = None,
        steering: List[SteeringSpec] = None,
    ) -> List[CaptureResult]:
        from utils.generation import generate_with_capture
        from utils.model import format_prompt

        config = config or GenerationConfig()
        capture = capture or CaptureSpec()

        # Format prompts
        formatted = [
            format_prompt(p, self._tokenizer, use_chat_template=self._use_chat_template)
            for p in prompts
        ]

        # TODO: Add steering support to generate_with_capture
        if steering:
            raise NotImplementedError("Steering during capture not yet supported")

        results = generate_with_capture(
            self._model, self._tokenizer, formatted,
            n_layers=self._n_layers if capture.layers is None else max(capture.layers) + 1,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            capture_mlp='mlp' in str(capture.components),
        )

        # Convert from utils.generation.CaptureResult to our CaptureResult
        converted = []
        for r in results:
            converted.append(CaptureResult(
                prompt=r.prompt_text,
                response=r.response_text,
                prompt_tokens=r.prompt_tokens,
                response_tokens=r.response_tokens,
                prompt_activations=r.prompt_activations,
                response_activations=r.response_activations,
            ))
        return converted

    def stream(
        self,
        prompt: str,
        config: GenerationConfig = None,
        capture: CaptureSpec = None,
        steering: List[SteeringSpec] = None,
    ) -> Generator[TokenResult, None, None]:
        from core.generation import HookedGenerator, CaptureConfig, SteeringConfig
        from utils.model import format_prompt, tokenize_prompt

        config = config or GenerationConfig()

        # Format and tokenize
        formatted = format_prompt(prompt, self._tokenizer, use_chat_template=self._use_chat_template)
        inputs = tokenize_prompt(formatted, self._tokenizer)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Build configs
        capture_config = None
        if capture:
            capture_config = CaptureConfig(
                layers=capture.layers,
                components=capture.components,
            )

        steering_configs = None
        if steering:
            steering_configs = [
                SteeringConfig(
                    vector=s.vector.to(self.device),
                    layer=s.layer,
                    component=s.component,
                    coefficient=s.coefficient,
                )
                for s in steering
            ]

        generator = HookedGenerator(self._model)

        for token_out in generator.stream(
            input_ids,
            attention_mask,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            capture=capture_config,
            steering=steering_configs,
            stop_token_ids=config.stop_token_ids,
        ):
            token_str = self._tokenizer.decode([token_out.token_id])
            yield TokenResult(
                token=token_str,
                token_id=token_out.token_id,
                activations=token_out.activations,
            )

    def forward_with_capture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        capture: CaptureSpec,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        from contextlib import ExitStack
        from core import MultiLayerCapture

        layers = capture.layers or list(range(self._n_layers))
        results = {}

        # Nest all component captures in a single forward pass
        with ExitStack() as stack:
            captures = {}
            for component in capture.components:
                cap = stack.enter_context(MultiLayerCapture(
                    self._model,
                    layers=layers,
                    component=component,
                    keep_on_gpu=True,
                ))
                captures[component] = cap

            with torch.no_grad():
                self._model(input_ids=input_ids, attention_mask=attention_mask)

            for component, cap in captures.items():
                for layer in layers:
                    if layer not in results:
                        results[layer] = {}
                    results[layer][component] = cap.get(layer)

        return results


# =============================================================================
# Server Backend (delegates to model server)
# =============================================================================


class ServerBackend(GenerationBackend):
    """
    Backend that delegates to model server (server/app.py).

    Wraps existing ModelClient interface.
    """

    def __init__(self, model_name: str, load_in_8bit: bool = False, load_in_4bit: bool = False):
        from other.server.client import ModelClient, is_server_available

        if not is_server_available():
            raise ConnectionError("Model server not running. Start with: python server/app.py")

        self._client = ModelClient(model_name, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
        self._model_name = model_name

        # Get model info from server
        import requests
        try:
            status = requests.get("http://localhost:8765/health", timeout=5).json()
            self._n_layers = status.get("n_layers", 26)  # Default for Gemma-2
            self._hidden_dim = status.get("hidden_dim", 2304)
        except Exception:
            # Fall back to defaults
            self._n_layers = 26
            self._hidden_dim = 2304

    @classmethod
    def from_experiment(cls, experiment: str, variant: str = None) -> "ServerBackend":
        """Create from experiment config, delegating to server."""
        from utils.paths import get_default_variant, load_experiment_config

        if variant is None:
            variant = get_default_variant(experiment, mode='application')

        config = load_experiment_config(experiment)
        model_variants = config.get('model_variants', {})
        variant_config = model_variants.get(variant, {})
        model_name = variant_config.get('model')

        # Note: LoRA not supported on server yet
        if variant_config.get('lora'):
            raise NotImplementedError("ServerBackend doesn't support LoRA adapters yet")

        return cls(model_name)

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")  # Results come back on CPU

    def generate(
        self,
        prompts: List[str],
        config: GenerationConfig = None,
        steering: List[SteeringSpec] = None,
    ) -> List[str]:
        config = config or GenerationConfig()

        if steering:
            # Convert SteeringSpec to server format
            vectors = {s.layer: s.vector for s in steering}
            coefficients = {s.layer: s.coefficient for s in steering}
            component = steering[0].component if steering else "residual"

            return self._client.generate_with_steering(
                prompts,
                vectors=vectors,
                coefficients=coefficients,
                component=component,
                max_new_tokens=config.max_new_tokens,
            )
        else:
            return self._client.generate(
                prompts,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )

    def generate_with_capture(
        self,
        prompts: List[str],
        config: GenerationConfig = None,
        capture: CaptureSpec = None,
        steering: List[SteeringSpec] = None,
    ) -> List[CaptureResult]:
        config = config or GenerationConfig()
        capture = capture or CaptureSpec()

        if steering:
            raise NotImplementedError("ServerBackend doesn't support steering + capture together")

        results = self._client.generate_with_capture(
            prompts,
            n_layers=max(capture.layers) + 1 if capture.layers else None,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            capture_mlp='mlp' in str(capture.components),
        )

        # Convert to CaptureResult format
        converted = []
        for r in results:
            converted.append(CaptureResult(
                prompt=r.prompt_text,
                response=r.response_text,
                prompt_tokens=r.prompt_tokens,
                response_tokens=r.response_tokens,
                prompt_activations=r.prompt_activations,
                response_activations=r.response_activations,
            ))
        return converted

    def stream(self, prompt, config=None, capture=None, steering=None):
        raise NotImplementedError("ServerBackend doesn't support streaming yet")

    def forward_with_capture(self, input_ids, attention_mask, capture):
        raise NotImplementedError("ServerBackend doesn't support forward-only capture")


# =============================================================================
# Factory function
# =============================================================================


def get_backend(
    experiment: str = None,
    variant: str = None,
    prefer_server: bool = True,
    model_name: str = None,
    use_chat_template: bool = None,
    **kwargs,
) -> GenerationBackend:
    """
    Get appropriate backend, preferring server if available.

    Args:
        experiment: Experiment name (uses config to determine model)
        variant: Model variant within experiment
        prefer_server: If True, use ServerBackend when server is running
        model_name: Direct model name (alternative to experiment)
        use_chat_template: Override chat template usage (LocalBackend only)
        **kwargs: Passed to backend constructor

    Returns:
        GenerationBackend instance (ServerBackend or LocalBackend)
    """
    from other.server.client import is_server_available

    if prefer_server and is_server_available():
        try:
            if experiment:
                return ServerBackend.from_experiment(experiment, variant)
            elif model_name:
                return ServerBackend(model_name, **kwargs)
        except (ConnectionError, NotImplementedError):
            pass  # Fall through to local

    # Fall back to local
    if experiment:
        return LocalBackend.from_experiment(
            experiment, variant, use_chat_template=use_chat_template, **kwargs
        )
    elif model_name:
        from utils.model import load_model
        model, tokenizer = load_model(model_name, **kwargs)
        return LocalBackend(model, tokenizer, use_chat_template=use_chat_template)

    raise ValueError("Must provide experiment or model_name")
