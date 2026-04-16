"""Shared constants for the live-chat / chat-inference subsystem.

Defined once here and imported by chat_inference.py and serve.py. Mirrors
visualization/core/chat-config.js — if you rename one side, rename the other.

Kept free of heavy imports so serve.py can load these without forcing
chat_inference (and its torch + transformers dependency tree) to load too.
"""

DEFAULT_MODEL_TYPE = "application"  # model_variants key; mirrors JS CHAT_MODEL_TYPE
DEFAULT_EXPERIMENT = "live-chat"    # mirrors JS LIVE_CHAT_EXPERIMENT
