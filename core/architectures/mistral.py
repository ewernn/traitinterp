"""Mistral Architecture.

Identical hook surface to Llama - same module names, pre-norm, standard attention.
"""

from core.architectures import register
from core.architectures.base import Architecture
from core.architectures.llama import _LLAMA_TREE


register("mistral", Architecture(
    layer_prefix_path="model.layers",
    module_tree=_LLAMA_TREE,
))
