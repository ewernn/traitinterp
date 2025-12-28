"""
Model server for persistent model loading between script runs.

Usage:
    # Start server
    python -m server.app --port 8765 --model google/gemma-2-2b-it

    # In scripts
    from server import get_model_or_client, ModelClient

    handle = get_model_or_client("google/gemma-2-2b-it")
    if isinstance(handle, ModelClient):
        results = handle.generate_with_capture(prompts, ...)
    else:
        model, tokenizer = handle
"""

from server.client import get_model_or_client, ModelClient, is_server_available

__all__ = ['get_model_or_client', 'ModelClient', 'is_server_available']
