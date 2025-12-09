"""
Steering evaluation module.

Validates trait vectors via causal intervention.

Usage:
    from analysis.steering import SteeringHook, TraitJudge

    # Steering
    with SteeringHook(model, vector, layer=16, coefficient=1.5):
        output = model.generate(**inputs)

    # Judging (gpt-4.1-mini with logprobs)
    judge = TraitJudge()
    score = await judge.score(eval_prompt, question, answer)
"""

from analysis.steering.steer import SteeringHook, steer
from utils.judge import TraitJudge, aggregate_logprob_score

__all__ = [
    "SteeringHook",
    "steer",
    "TraitJudge",
    "aggregate_logprob_score",
]
