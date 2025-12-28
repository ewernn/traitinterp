"""
Evaluation metrics for model outputs.

Usage:
    from utils.metrics import ce_loss, sequence_ce_loss, batch_ce_loss

    # Pure math - from logits and targets
    loss = ce_loss(logits, target_ids)

    # Single text sequence
    loss = sequence_ce_loss(model, tokenizer, "Some text to evaluate")

    # Batch of texts (averaged)
    loss = batch_ce_loss(model, tokenizer, ["text1", "text2", ...])
"""

import torch
import torch.nn.functional as F
from typing import List

from utils.model import tokenize_batch


def ce_loss(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
    """
    Cross-entropy loss for next-token prediction.

    Args:
        logits: Model logits [seq_len, vocab_size]
        target_ids: Target token IDs [seq_len]

    Returns:
        Average cross-entropy loss (float)
    """
    # Shift: logits[i] predicts target[i+1]
    return F.cross_entropy(logits[:-1], target_ids[1:]).item()


def sequence_ce_loss(model, tokenizer, text: str) -> float:
    """
    CE loss on a single text sequence.

    Args:
        model: HuggingFace model
        tokenizer: Model tokenizer
        text: Text to evaluate

    Returns:
        Average cross-entropy loss
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]  # [seq_len, vocab]
    return ce_loss(logits, inputs.input_ids[0])


def batch_ce_loss(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 8,
) -> float:
    """
    Average CE loss across multiple texts.

    Handles batching and padding correctly - padding tokens are masked out.

    Args:
        model: HuggingFace model
        tokenizer: Model tokenizer
        texts: List of texts to evaluate
        batch_size: Batch size for processing

    Returns:
        Average cross-entropy loss across all tokens in all texts
    """
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch = tokenize_batch(batch_texts, tokenizer, padding_side="right")

        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        lengths = batch["lengths"]

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Compute loss per sequence, masking padding
        for b, length in enumerate(lengths):
            if length <= 1:
                continue  # Need at least 2 tokens for next-token prediction

            seq_logits = logits[b, : length - 1]  # [length-1, vocab]
            seq_targets = input_ids[b, 1:length]  # [length-1]

            seq_loss = F.cross_entropy(seq_logits, seq_targets, reduction="sum")
            total_loss += seq_loss.item()
            total_tokens += length - 1

    if total_tokens == 0:
        return 0.0

    return total_loss / total_tokens
