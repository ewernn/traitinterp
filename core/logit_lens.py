"""
Logit lens: project vectors/activations to vocabulary space.

Input: Trait vector + model
Output: Top/bottom tokens representing the direction

Usage:
    from core.logit_lens import vector_to_vocab, build_common_token_mask, get_interpretation_layers

    layers = get_interpretation_layers(model.config.num_hidden_layers)
    mask = build_common_token_mask(tokenizer)
    result = vector_to_vocab(vector, model, tokenizer, common_mask=mask)
"""

import torch
from typing import Dict

from utils.model import get_inner_model


# =============================================================================
# Layer selection
# =============================================================================

def get_interpretation_layers(n_layers: int) -> Dict[str, dict]:
    """
    Return layer indices for interpretation (40% and 90% depth).

    40%: middle layers where features are forming
    90%: late layers near output
    """
    return {
        "mid": {"layer": round(n_layers * 0.4), "pct": 40},
        "late": {"layer": round(n_layers * 0.9), "pct": 90},
    }


# =============================================================================
# Common token filtering
# =============================================================================

def build_common_token_mask(tokenizer, max_vocab_idx: int = 10000) -> torch.Tensor:
    """
    Build a mask for common/interpretable tokens.

    Filters to:
    1. First N tokens in vocab (BPE tokenizers order by frequency)
    2. Printable ASCII tokens (no code artifacts, foreign scripts)

    Returns:
        Boolean tensor of shape (vocab_size,) - True for common tokens
    """
    vocab_size = len(tokenizer)
    mask = torch.zeros(vocab_size, dtype=torch.bool)

    for idx in range(min(vocab_size, max_vocab_idx)):
        token = tokenizer.decode([idx])
        # Keep if mostly printable ASCII
        if token and all(ord(c) < 128 for c in token):
            # Filter out code-like tokens
            if not any(x in token for x in ['()', '{}', '[]', '::', '//', '/*', '*/', '=>', '->']):
                # Filter out camelCase/PascalCase (likely code)
                if not (any(c.isupper() for c in token[1:]) and any(c.islower() for c in token)):
                    mask[idx] = True

    return mask


# =============================================================================
# Vector projection
# =============================================================================

def vector_to_vocab(
    vector: torch.Tensor,
    model,
    tokenizer,
    top_k: int = 20,
    apply_norm: bool = True,
    common_mask: torch.Tensor = None,
) -> dict:
    """
    Project vector through unembedding to vocabulary space.

    Args:
        vector: Trait vector (hidden_dim,)
        model: Model with lm_head
        tokenizer: Tokenizer for decoding
        top_k: Number of tokens to return
        apply_norm: Apply final RMSNorm before projection (matches residual stream processing)
        common_mask: Boolean mask for common tokens (None = no filtering)

    Returns:
        Dict with 'toward' and 'away' token lists, each entry is {"token": str, "value": float}
    """
    inner = get_inner_model(model)

    # Optionally apply final norm (matches how residual stream is processed)
    if apply_norm and hasattr(inner, 'norm'):
        # Need batch dim for norm
        vector = inner.norm(vector.unsqueeze(0).to(model.device)).squeeze(0)

    # Project through unembedding
    W_U = model.lm_head.weight  # (vocab_size, hidden_dim)
    vector = vector.to(W_U.device).to(W_U.dtype)
    logits = W_U @ vector  # (vocab_size,)

    # Apply common token filter if provided
    if common_mask is not None:
        common_mask = common_mask.to(logits.device)
        # Set non-common tokens to -inf/+inf so they don't appear in top-k
        logits_filtered = logits.clone()
        logits_filtered[~common_mask] = float('-inf')
        logits_filtered_neg = (-logits).clone()
        logits_filtered_neg[~common_mask] = float('-inf')
    else:
        logits_filtered = logits
        logits_filtered_neg = -logits

    # Get top-k in each direction
    top_vals, top_idx = logits_filtered.topk(top_k)
    bottom_vals, bottom_idx = logits_filtered_neg.topk(top_k)

    def decode_tokens(indices, values):
        results = []
        for idx, val in zip(indices.tolist(), values.tolist()):
            if val == float('-inf') or val == float('inf'):
                continue  # Skip filtered tokens
            token = tokenizer.decode([idx])
            # Clean up for display
            token_repr = repr(token) if token.strip() != token or not token else token
            results.append({"token": token_repr, "value": round(val, 3)})
        return results

    return {
        "toward": decode_tokens(top_idx, top_vals),
        "away": decode_tokens(bottom_idx, -bottom_vals),
    }
