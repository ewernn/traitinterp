"""
Utility functions for batched generation and activation extraction.
"""

import torch
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer


def get_activations_from_texts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    layers: List[int],
    batch_size: int = 8,
    device: str = "cuda",
) -> Dict[int, torch.Tensor]:
    """
    Extract activations from multiple texts using batched forward passes.

    Args:
        model: The model to extract from
        tokenizer: The tokenizer
        texts: List of text strings
        layers: List of hidden_states indices to extract (NOT layer numbers!)
                CRITICAL: hidden_states[0] is embedding, hidden_states[1] is layer 0 output
                To extract layers 0-25, pass layers=list(range(1, 27))
        batch_size: Batch size for processing
        device: Device to use

    Returns:
        Dict mapping hidden_states index to averaged activations [n_examples, hidden_dim]

    Note: The returned dict keys match the input `layers` list, which are hidden_states
          indices, NOT layer numbers. Caller is responsible for correct indexing.
    """
    all_activations = {layer: [] for layer in layers}

    # Validation on first batch only (for performance)
    validate_first_batch = True

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # Extract from specified layers
        hidden_states = outputs.hidden_states  # Tuple of tensors [batch, seq, hidden]

        # Validate indexing on first batch
        if validate_first_batch:
            validate_first_batch = False

            # Check that indices are valid
            max_idx = max(layers) if layers else 0
            if max_idx >= len(hidden_states):
                raise IndexError(
                    f"Layer index {max_idx} out of range. "
                    f"Model has {len(hidden_states)} hidden_states "
                    f"(0=embedding, 1-{len(hidden_states)-1}=layer outputs)"
                )

            # Warn if using index 0 (embedding)
            if 0 in layers:
                print("⚠️  WARNING: Extracting from hidden_states[0] (embedding layer)")
                print("   For transformer layers, use indices 1 and above")
                print("   Example: layers=list(range(1, n_layers+1)) for layers 0 to n_layers-1")

        for layer_idx in layers:
            # Get this layer's activations
            layer_acts = hidden_states[layer_idx]  # [batch, seq, hidden]

            # Average over sequence length for each example in batch
            averaged = layer_acts.mean(dim=1)  # [batch, hidden]

            all_activations[layer_idx].append(averaged.cpu())

    # Concatenate all batches
    for layer_idx in layers:
        all_activations[layer_idx] = torch.cat(all_activations[layer_idx], dim=0)

    return all_activations


def generate_batch_with_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    capture_layers: Optional[List[int]] = None,
) -> Tuple[List[str], Optional[Dict[int, torch.Tensor]]]:
    """
    Generate responses for a batch of prompts and optionally capture activations.

    Args:
        model: The model to use
        tokenizer: The tokenizer
        prompts: List of prompts
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        capture_layers: Layer indices to capture (None = don't capture)

    Returns:
        Tuple of (responses, activations_dict or None)
    """
    # Tokenize batch
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)

    # Generate
    with torch.no_grad():
        if capture_layers is not None:
            # Generate with hidden states
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            # Note: Capturing activations during generation is complex
            # For now, we'll do a separate forward pass on generated text
            generated_ids = outputs.sequences

        else:
            # Just generate without hidden states
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )

    # Decode responses
    responses = []
    for i in range(len(prompts)):
        # Find where input ends (last non-pad token)
        input_ids = inputs['input_ids'][i]
        if tokenizer.pad_token_id in input_ids:
            input_length = (input_ids != tokenizer.pad_token_id).sum().item()
        else:
            input_length = len(input_ids)

        # Decode only the generated part
        generated_text = tokenizer.decode(
            generated_ids[i][input_length:],
            skip_special_tokens=True
        )
        responses.append(generated_text.strip())

    # If we need activations, do a forward pass on the generated text
    if capture_layers is not None:
        # Get activations from the full generated sequences
        with torch.no_grad():
            outputs = model(
                input_ids=generated_ids,
                output_hidden_states=True,
                return_dict=True
            )

        hidden_states = outputs.hidden_states
        activations = {}

        for layer_idx in capture_layers:
            # Get activations and average over sequence
            layer_acts = hidden_states[layer_idx]  # [batch, seq, hidden]
            averaged = layer_acts.mean(dim=1)  # [batch, hidden]
            activations[layer_idx] = averaged.cpu()
    else:
        activations = None

    return responses, activations