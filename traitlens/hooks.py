"""
Hook management for transformer models.

Provides a clean interface for registering and managing PyTorch hooks
on arbitrary model modules.
"""

import torch
from typing import Any, Callable, Dict, List, Optional


class HookManager:
    """
    Manage forward hooks on any PyTorch model.

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> capture = ActivationCapture()
        >>>
        >>> with HookManager(model) as hooks:
        ...     hooks.add_forward_hook("transformer.h.10", capture.make_hook("layer_10"))
        ...     output = model.generate(input_ids)
        ...
        >>> activations = capture.get("layer_10")
    """

    def __init__(self, model: torch.nn.Module):
        """
        Initialize hook manager for a model.

        Args:
            model: PyTorch model to manage hooks for
        """
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.hook_functions: Dict[str, Callable] = {}

    def add_forward_hook(
        self,
        module_path: str,
        hook_fn: Callable[[torch.nn.Module, Any, Any], Any],
        name: Optional[str] = None
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Add a forward hook to a module specified by path.

        Args:
            module_path: Dot-separated path to module (e.g., "model.layers.16")
            hook_fn: Function to call during forward pass
            name: Optional name for this hook (for debugging)

        Returns:
            RemovableHandle that can be used to remove this specific hook

        Example:
            >>> def my_hook(module, input, output):
            ...     print(f"Output shape: {output[0].shape}")
            ...     return output
            >>>
            >>> handle = hooks.add_forward_hook("model.layers.16", my_hook)
        """
        module = self._get_module(module_path)
        handle = module.register_forward_hook(hook_fn)
        self.handles.append(handle)

        if name:
            self.hook_functions[name] = hook_fn

        return handle

    def remove_all(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.hook_functions.clear()

    def _get_module(self, module_path: str) -> torch.nn.Module:
        """
        Navigate to a module using dot-separated path.

        Args:
            module_path: Path like "model.layers.16.self_attn"

        Returns:
            The module at that path

        Raises:
            AttributeError: If path doesn't exist
        """
        parts = module_path.split('.')
        module = self.model

        for part in parts:
            # Handle both attribute access and indexing
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        return module

    def __enter__(self) -> 'HookManager':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - automatically remove all hooks."""
        self.remove_all()

    def __len__(self) -> int:
        """Number of active hooks."""
        return len(self.handles)

    def __repr__(self) -> str:
        """String representation."""
        return f"HookManager(model={self.model.__class__.__name__}, hooks={len(self)})"