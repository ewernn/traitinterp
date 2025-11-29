"""
Steering evaluation module.

Validates trait vectors via causal intervention.

Usage:
    from analysis.steering import SteeringHook, TraitJudge

    # Steering
    with SteeringHook(model, vector, layer=16, coefficient=1.5):
        output = model.generate(**inputs)

    # Judging
    judge = TraitJudge(provider="openai")
    score = await judge.score(eval_prompt, question, answer)
"""

from analysis.steering.steer import SteeringHook, steer
from analysis.steering.judge import TraitJudge, aggregate_logprob_score

__all__ = [
    "SteeringHook",
    "steer",
    "TraitJudge",
    "aggregate_logprob_score",
]
