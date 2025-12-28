"""
Live chat inference with real-time trait projections.

Loads model and trait vectors, generates tokens one at a time,
projects activations onto trait vectors, yields results for SSE streaming.

Supports two backends:
- "local": Load model locally and generate (default)
- "modal": Call Modal API for GPU inference, project locally

Usage:
    from visualization.chat_inference import ChatInference

    # Local inference
    chat = ChatInference(experiment='{experiment}', backend='local')

    # Modal inference (Railway)
    chat = ChatInference(experiment='{experiment}', backend='modal')

    for event in chat.generate("How do I hack a computer?"):
        # event = {'token': '...', 'trait_scores': {'refusal': 0.5, ...}}
        yield f"data: {json.dumps(event)}\n\n"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from typing import Dict, Generator, List, Optional, Tuple, TYPE_CHECKING

# Lazy imports for heavy dependencies
if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

from core import projection
from utils.paths import get as get_path, get_vector_path, list_layers, list_methods
from utils.vectors import get_best_layer
from utils.model import format_prompt, tokenize_prompt, load_experiment_config


class ChatInference:
    """Manages model and trait vectors for live chat with trait monitoring."""

    def __init__(self, experiment: str, device: str = "auto", backend: str = "local", model_type: str = "application"):
        """
        Args:
            experiment: Experiment name
            device: Device for local inference ("auto", "cuda", "mps", "cpu")
            backend: Inference backend - "local" or "modal"
            model_type: Which model from config to use - "extraction" or "application" (default)
        """
        self.experiment = experiment
        self.device = device
        self.backend = backend
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.trait_vectors: Dict[str, Tuple['torch.Tensor', int, str, str]] = {}  # trait -> (vector, layer, method, source)
        self.n_layers = None
        self.use_chat_template = None
        self._loaded = False
        self._modal_app = None

    def load(self):
        """Load model and trait vectors. Called lazily on first generate()."""
        if self._loaded:
            return

        # Get model from experiment config based on model_type
        config = load_experiment_config(self.experiment)
        config_key = f'{self.model_type}_model'
        fallback = 'google/gemma-2-2b-it' if self.model_type == 'application' else 'google/gemma-2-2b'
        model_id = config.get(config_key, fallback)
        print(f"[ChatInference] Using {self.model_type} model: {model_id}")

        if self.backend == "modal":
            print(f"[ChatInference] Using Modal backend for model: {model_id}")
            # Import modal and transformers (lazy)
            try:
                import modal
                from transformers import AutoTokenizer
                # Look up deployed app
                self._modal_app = modal.App.lookup("trait-capture", create_if_missing=False)
                self._model_id = model_id
                # Still need tokenizer for chat template formatting
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.use_chat_template = self.tokenizer.chat_template is not None
                self.n_layers = 26  # Will be updated from Modal response
                print(f"[ChatInference] Modal client initialized, chat_template={self.use_chat_template}")
            except ImportError as e:
                raise ImportError("Modal backend requires 'modal' package: pip install modal") from e
            except Exception as e:
                raise RuntimeError(f"Failed to connect to Modal: {e}. Make sure 'trait-capture' is deployed.") from e
        else:
            print(f"[ChatInference] Loading model locally: {model_id}")
            # Import torch and transformers (lazy)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=self.device,
                attn_implementation='eager'
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.n_layers = len(self.model.model.layers)
            self.use_chat_template = self.tokenizer.chat_template is not None

            # Build set of stop token IDs (EOS + any end_of_turn tokens)
            self.stop_token_ids = {self.tokenizer.eos_token_id}
            # Gemma uses <end_of_turn> as stop token
            for token in ['<end_of_turn>', '<|end|>', '<|eot_id|>']:
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if token_ids:
                    self.stop_token_ids.add(token_ids[0])

            print(f"[ChatInference] Model loaded: {self.n_layers} layers, chat_template={self.use_chat_template}")
            print(f"[ChatInference] Stop tokens: {self.stop_token_ids}")

        # Load trait vectors (always from local - vectors stay on Railway)
        self._load_trait_vectors()
        self._loaded = True

    def _load_trait_vectors(self):
        """Discover and load all trait vectors with their best layers."""
        import torch  # Lazy import

        print(f"[ChatInference] _load_trait_vectors() called for experiment={self.experiment}", flush=True)
        extraction_dir = get_path('extraction.base', experiment=self.experiment)
        print(f"[ChatInference] Extraction dir: {extraction_dir}", flush=True)
        if not extraction_dir.exists():
            print(f"[ChatInference] ERROR: Extraction dir does not exist!", flush=True)
            return

        for category_dir in sorted(extraction_dir.iterdir()):
            if not category_dir.is_dir() or category_dir.name.startswith('.'):
                continue
            for trait_dir in sorted(category_dir.iterdir()):
                if not trait_dir.is_dir():
                    continue

                trait_path = f"{category_dir.name}/{trait_dir.name}"

                # Check if any methods exist for this trait
                available_methods = list_methods(self.experiment, trait_path)
                if not available_methods:
                    continue

                # Get best layer/method for this trait
                best = get_best_layer(self.experiment, trait_path)
                layer = best['layer']
                method = best['method']
                source = best['source']

                vector_file = get_vector_path(self.experiment, trait_path, method, layer)
                if not vector_file.exists():
                    print(f"  Skip {trait_path}: vector file not found")
                    continue

                # Load vector to appropriate device
                vector = torch.load(vector_file, weights_only=True).to(dtype=torch.float16)
                if self.backend == "local" and self.model is not None:
                    vector = vector.to(device=self.model.device)
                # For modal backend, keep on CPU
                self.trait_vectors[trait_path] = (vector, layer, method, source)

        print(f"[ChatInference] Loaded {len(self.trait_vectors)} trait vectors")
        for trait, (_, layer, method, source) in self.trait_vectors.items():
            print(f"  {trait}: L{layer} {method} ({source})")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        history: Optional[List[Dict]] = None,
        previous_context_length: int = 0
    ) -> Generator[Dict, None, None]:
        """
        Generate response token-by-token with trait projections.

        Args:
            prompt: User message
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            history: Optional chat history [{"role": "user/assistant", "content": "..."}]
            previous_context_length: Number of tokens already captured in previous turns (skip these)

        Yields:
            Dict with 'token', 'token_index', 'trait_scores', 'is_prompt', 'is_special', 'done' keys
            Status events have 'status' key for loading progress
            Final yield has 'done': True and 'full_response'
        """
        # Yield loading status if not loaded yet
        if not self._loaded:
            yield {'status': 'loading_model', 'message': 'Loading model...'}

        self.load()  # Lazy load

        if not self._loaded:
            yield {'error': 'Failed to load model', 'done': True}
            return

        # Send vector metadata to frontend
        vector_metadata = {
            trait: {'layer': layer, 'method': method, 'source': source}
            for trait, (_, layer, method, source) in self.trait_vectors.items()
        }
        yield {
            'status': 'loading',
            'message': f'Generating with {len(self.trait_vectors)} traits...',
            'vector_metadata': vector_metadata
        }

        # Build input
        if history is None:
            history = []

        messages = history + [{"role": "user", "content": prompt}]

        # Debug: print what we're sending
        print(f"[ChatInference] History received: {history}")
        print(f"[ChatInference] Messages to tokenize: {messages}")

        # Format prompt with chat template if needed
        if self.use_chat_template:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            # Base model: raw text
            formatted_prompt = prompt

        # Route to appropriate backend
        if self.backend == "modal":
            # Modal backend: get all activations at once, then stream
            yield from self._generate_modal(formatted_prompt, max_new_tokens, temperature, previous_context_length)
            return

        # Local backend continues below
        import torch  # Lazy import
        from core import HookManager

        input_ids = tokenize_prompt(formatted_prompt, self.tokenizer, self.use_chat_template).input_ids.to(self.model.device)

        # Group traits by layer for efficient hooking
        layers_needed = set(layer for _, layer, _, _ in self.trait_vectors.values())

        # Storage for activations
        activations = {}

        def make_hook(layer_idx):
            def hook(module, inp, out):
                out_t = out[0] if isinstance(out, tuple) else out
                # Store last token's activation
                activations[layer_idx] = out_t[:, -1, :].detach()
            return hook

        def make_full_hook(layer_idx):
            """Hook that stores ALL token activations (for prompt processing)"""
            def hook(module, inp, out):
                out_t = out[0] if isinstance(out, tuple) else out
                activations[layer_idx] = out_t.detach()  # [batch, seq, hidden]
            return hook

        # Get special token IDs for is_special flag
        special_ids = set(self.tokenizer.all_special_ids)

        # Always process and yield prompt tokens first (frontend filters display)
        # Run forward pass through prompt to get per-token activations
        with HookManager(self.model) as hooks:
            for layer in layers_needed:
                hooks.add_forward_hook(f"model.layers.{layer}", make_full_hook(layer))

            with torch.no_grad():
                _ = self.model(input_ids=input_ids, return_dict=True)

        # Decode and score each prompt token (skip first previous_context_length)
        prompt_token_ids = input_ids[0].tolist()
        for pos in range(len(prompt_token_ids)):
            # Skip tokens from previous turns
            if pos < previous_context_length:
                continue

            token_id = prompt_token_ids[pos]
            is_special = token_id in special_ids

            # Decode token (include special tokens)
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
            if not token_str:
                token_str = f"[{token_id}]"  # Fallback for empty decode

            # Compute trait projections for this position
            trait_scores = {}
            for trait_path, (vector, layer, _, _) in self.trait_vectors.items():
                if layer in activations:
                    act = activations[layer][0, pos, :]  # [hidden_dim]
                    score = projection(act, vector, normalize_vector=True).item()
                    trait_name = trait_path.split('/')[-1]
                    trait_scores[trait_name] = round(score, 4)

            yield {
                'token': token_str,
                'token_index': pos,
                'trait_scores': trait_scores,
                'is_prompt': True,
                'is_special': is_special,
                'done': False
            }

        activations.clear()

        # Generate tokens
        context = input_ids
        generated_tokens = []
        full_response = ""
        past_key_values = None  # KV cache
        current_token_index = len(prompt_token_ids)  # Response tokens start after prompt

        for step in range(max_new_tokens):
            activations.clear()

            # Forward pass with hooks (use KV cache for speed)
            with HookManager(self.model) as hooks:
                for layer in layers_needed:
                    hooks.add_forward_hook(f"model.layers.{layer}", make_hook(layer))

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=context,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True
                    )

            # Update KV cache for next iteration
            past_key_values = outputs.past_key_values

            # Sample next token
            logits = outputs.logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            is_special = next_id in special_ids
            is_stop = next_id in self.stop_token_ids

            # Decode token (include special tokens)
            token_str = self.tokenizer.decode([next_id], skip_special_tokens=False)
            if not token_str:
                token_str = f"[{next_id}]"  # Fallback

            # Compute trait projections
            trait_scores = {}
            for trait_path, (vector, layer, _, _) in self.trait_vectors.items():
                if layer in activations:
                    act = activations[layer].squeeze(0)  # [hidden_dim]
                    score = projection(act, vector, normalize_vector=True).item()
                    trait_name = trait_path.split('/')[-1]
                    trait_scores[trait_name] = round(score, 4)

            # Yield result (including stop tokens)
            yield {
                'token': token_str,
                'token_index': current_token_index,
                'trait_scores': trait_scores,
                'is_prompt': False,
                'is_special': is_special,
                'done': False
            }

            current_token_index += 1

            # Stop after yielding the stop token
            if is_stop:
                break

            # Only add non-special tokens to display response
            if not is_special:
                generated_tokens.append(next_id)
                full_response += token_str

            # Update context to just new token (KV cache handles history)
            context = torch.tensor([[next_id]], device=self.model.device)

        # Final yield with full response
        yield {
            'token': '',
            'trait_scores': {},
            'done': True,
            'full_response': full_response
        }

    def _generate_modal(
        self,
        formatted_prompt: str,
        max_new_tokens: int,
        temperature: float,
        previous_context_length: int
    ) -> Generator[Dict, None, None]:
        """
        Generate using Modal backend with streaming.

        Modal yields tokens + activations one at a time, we project and stream
        to browser immediately.
        """
        import torch  # Lazy import

        print(f"[ChatInference] Calling Modal streaming API...")
        yield {'status': 'calling_modal', 'message': 'Calling Modal GPU...'}

        try:
            # Import the app and function directly
            import sys
            from pathlib import Path
            inference_path = Path(__file__).parent.parent / "inference"
            if str(inference_path) not in sys.path:
                sys.path.insert(0, str(inference_path))

            # Import modal_inference module
            import modal_inference

            full_response = ""
            token_count = 0

            # Use app context to call streaming function
            with modal_inference.app.run():
                for chunk in modal_inference.capture_activations_stream.remote_gen(
                    model_name=self._model_id,
                    prompt=formatted_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    component="residual",
                ):
                    token = chunk['token']
                    full_response += token
                    token_count += 1

                    # Convert activations to tensors (Modal returns lists per token)
                    activations_dict = chunk['activations']

                    # Compute trait projections for this token
                    trait_scores = {}
                    for trait_path, (vector, layer, _, _) in self.trait_vectors.items():
                        layer_str = str(layer)
                        if layer_str in activations_dict:
                            act = torch.tensor(activations_dict[layer_str], dtype=torch.float16)
                            # Vector already on CPU for Modal backend
                            score = projection(act, vector, normalize_vector=True).item()
                            trait_name = trait_path.split('/')[-1]
                            trait_scores[trait_name] = round(score, 4)

                    # Yield token with scores immediately
                    yield {
                        'token': token,
                        'trait_scores': trait_scores,
                        'done': False
                    }

        except Exception as e:
            print(f"[ChatInference] Modal call failed: {e}")
            yield {'error': f'Modal call failed: {str(e)}', 'done': True}
            return

        print(f"[ChatInference] Streamed {token_count} tokens from Modal")

        # Final yield
        yield {
            'token': '',
            'trait_scores': {},
            'done': True,
            'full_response': full_response
        }


# Singleton instance (lazy loaded)
_chat_instance: Optional[ChatInference] = None


def get_chat_instance(experiment: str, backend: str = None, model_type: str = "application") -> ChatInference:
    """
    Get or create chat instance for experiment.

    Args:
        experiment: Experiment name
        backend: Override backend ("local" or "modal"). If None, uses env var INFERENCE_BACKEND or defaults to "local"
        model_type: Which model from config to use - "extraction" or "application" (default)
    """
    global _chat_instance

    # Determine backend
    if backend is None:
        backend = os.getenv('INFERENCE_BACKEND', 'local')

    # Recreate instance if experiment, backend, or model_type changed
    if (_chat_instance is None or
        _chat_instance.experiment != experiment or
        _chat_instance.backend != backend or
        _chat_instance.model_type != model_type):
        _chat_instance = ChatInference(experiment, backend=backend, model_type=model_type)

    return _chat_instance
