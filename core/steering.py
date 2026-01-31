"""
Batched steering generation primitive.

Input: model, tokenizer, pre-formatted prompts, configs [(layer, vector, coef)]
Output: flat list of responses (caller groups by slicing)

Usage:
    from core import batched_steering_generate

    responses = batched_steering_generate(
        model, tokenizer,
        prompts=formatted_questions,
        configs=[(layer, vector, coef) for layer, coef in layer_coef_pairs],
        component="residual",
    )
    # Group results: responses[i*n_prompts:(i+1)*n_prompts] for config i
"""

from typing import List, Tuple
import torch

from .hooks import BatchedLayerSteeringHook


def batched_steering_generate(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    configs: List[Tuple[int, torch.Tensor, float]],
    component: str = "residual",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> List[str]:
    """
    Generate responses with different steering configs per batch slice.

    Prompts are replicated for each config, creating a batch where:
    - Config 0 steers batch indices [0:n_prompts]
    - Config 1 steers batch indices [n_prompts:2*n_prompts]
    - etc.

    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompts: Pre-formatted prompts (use format_prompt before calling)
        configs: List of (layer, vector, coefficient) tuples, one per evaluation
        component: Hook component ("residual", "attn_contribution", etc.)
        max_new_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature (0 for greedy)

    Returns:
        Flat list of responses: [r0_p0, r0_p1, ..., r1_p0, r1_p1, ...]
        To group by config: responses[i*n_prompts:(i+1)*n_prompts] for config i

    Note:
        Does NOT handle sub-batching across configs - the full batch must fit
        in memory. If you have many configs, call this function multiple times
        with subsets of configs.

    Example:
        # Evaluate 3 layer/coef combinations on 5 questions
        configs = [
            (10, vec10, 50.0),
            (11, vec11, 60.0),
            (12, vec12, 70.0),
        ]
        responses = batched_steering_generate(model, tokenizer, questions, configs)

        # responses has 15 items: [L10_q0, L10_q1, ..., L10_q4, L11_q0, ..., L12_q4]
        for i, (layer, vec, coef) in enumerate(configs):
            config_responses = responses[i*5:(i+1)*5]
    """
    if not prompts or not configs:
        return []

    n_prompts = len(prompts)
    n_configs = len(configs)

    # Build batched prompts: replicate prompts for each config
    batched_prompts = []
    for _ in configs:
        batched_prompts.extend(prompts)

    # Build steering configs with batch slices
    steering_configs = []
    for idx, (layer, vector, coef) in enumerate(configs):
        batch_start = idx * n_prompts
        batch_end = (idx + 1) * n_prompts
        steering_configs.append((layer, vector, coef, (batch_start, batch_end)))

    # Generate with batched steering
    # Lazy import to avoid circular dependency (utils.generation imports core)
    from utils.generation import generate_batch
    with BatchedLayerSteeringHook(model, steering_configs, component=component):
        responses = generate_batch(
            model, tokenizer, batched_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    return responses
