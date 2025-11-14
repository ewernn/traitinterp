"""
Activation capture and storage for transformer models.

Provides utilities for capturing and storing activations during forward passes.
"""

import torch
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union


class ActivationCapture:
    """
    Capture and store activations during model forward passes.

    Example:
        >>> capture = ActivationCapture()
        >>> with HookManager(model) as hooks:
        ...     hooks.add_forward_hook("model.layers.16", capture.make_hook("layer_16"))
        ...     output = model.generate(input_ids)
        ...
        >>> activations = capture.get("layer_16")  # [batch, seq_len, hidden_dim]
    """

    def __init__(self):
        """Initialize activation storage."""
        self.activations: Dict[str, List[torch.Tensor]] = defaultdict(list)

    def make_hook(self, name: str) -> Callable:
        """
        Factory method that creates a hook function for capturing activations.

        Args:
            name: Name to store activations under

        Returns:
            Hook function that captures activations to self.activations[name]

        Example:
            >>> capture = ActivationCapture()
            >>> hook_fn = capture.make_hook("my_layer")
            >>> # hook_fn can now be passed to HookManager.add_forward_hook()
        """
        def hook_fn(module: torch.nn.Module, input: Any, output: Any) -> None:
            """The actual hook that captures activations."""
            # Handle different output types
            if isinstance(output, tuple):
                # Some modules return tuples, we typically want the first element
                activation = output[0]
            elif isinstance(output, dict):
                # Some models return dicts
                activation = output.get('hidden_states', output)
            else:
                activation = output

            # Detach and move to CPU to prevent memory issues
            if isinstance(activation, torch.Tensor):
                activation = activation.detach().cpu()

            # Store the activation
            self.activations[name].append(activation)

        return hook_fn

    def get(self, name: str, concat: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Retrieve stored activations by name.

        Args:
            name: Name of activations to retrieve
            concat: If True, concatenate list into single tensor along batch dimension

        Returns:
            Stored activations as tensor or list of tensors

        Raises:
            KeyError: If name not found in captured activations

        Example:
            >>> activations = capture.get("layer_16")  # Returns concatenated tensor
            >>> activations_list = capture.get("layer_16", concat=False)  # Returns list
        """
        if name not in self.activations:
            raise KeyError(f"No activations captured for '{name}'. "
                         f"Available: {list(self.activations.keys())}")

        activations = self.activations[name]

        if concat and activations:
            # Concatenate along batch dimension (dim=0)
            return torch.cat(activations, dim=0)
        return activations

    def clear(self, name: Optional[str] = None) -> None:
        """
        Clear stored activations.

        Args:
            name: If specified, clear only this name. Otherwise clear all.

        Example:
            >>> capture.clear()  # Clear all
            >>> capture.clear("layer_16")  # Clear specific
        """
        if name is not None:
            if name in self.activations:
                del self.activations[name]
        else:
            self.activations.clear()

    def keys(self) -> List[str]:
        """
        List all captured activation names.

        Returns:
            List of names that have captured activations

        Example:
            >>> capture.keys()
            ['layer_16', 'attention_out', 'mlp_out']
        """
        return list(self.activations.keys())

    def __len__(self) -> int:
        """Number of different named activations captured."""
        return len(self.activations)

    def __contains__(self, name: str) -> bool:
        """Check if activations exist for a name."""
        return name in self.activations

    def __repr__(self) -> str:
        """String representation."""
        sizes = {}
        for name, acts in self.activations.items():
            if acts:
                if isinstance(acts[0], torch.Tensor):
                    sizes[name] = f"{len(acts)} tensors, shape {acts[0].shape}"
                else:
                    sizes[name] = f"{len(acts)} items"
        return f"ActivationCapture({sizes})"

    @property
    def memory_usage(self) -> Dict[str, float]:
        """
        Estimate memory usage of stored activations in MB.

        Returns:
            Dictionary mapping names to memory usage in MB

        Example:
            >>> capture.memory_usage
            {'layer_16': 45.2, 'attention_out': 22.6}
        """
        usage = {}
        for name, acts in self.activations.items():
            total_bytes = 0
            for act in acts:
                if isinstance(act, torch.Tensor):
                    total_bytes += act.numel() * act.element_size()
            usage[name] = total_bytes / (1024 * 1024)  # Convert to MB
        return usage