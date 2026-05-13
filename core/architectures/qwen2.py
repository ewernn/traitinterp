"""Qwen2 Architecture (Qwen2.5 + DeepSeek-R1-Distill-Qwen).

Llama-shaped: pre-norm, GQA, gated MLP. Identical hook paths to Llama.
"""

from core.architectures import register
from core.architectures.base import Architecture
from core.architectures.llama import _LLAMA_TREE


register("qwen2", Architecture(
    layer_prefix_path="model.layers",
    module_tree=_LLAMA_TREE,
))
