"""
LLM-as-judge scoring for steering evaluation.

Input:
    - eval_prompt: Full prompt template with {question} and {answer} placeholders
    - question: The prompt given to the model
    - answer: The model's response

Output:
    - Score 0-100 (logprob-weighted average, not argmax)
    - None if judge refuses or score can't be aggregated

Usage:
    judge = TraitJudge(provider="openai")
    score = await judge.score(
        eval_prompt="You are evaluating... {question} ... {answer} ...",
        question="What is 2+2?",
        answer="The answer is 4.",
    )
"""

import os
import math
import asyncio
from typing import Optional, Dict, List, Tuple
from abc import ABC, abstractmethod


# Coherence check prompt
COHERENCE_PROMPT = """Rate the coherence and fluency of this response on a scale of 0-100.
0 = completely incoherent/broken
100 = perfectly coherent and fluent

Response:
[RESPONSE START]
{answer}
[RESPONSE END]

Just respond with a number 0-100."""


def aggregate_logprob_score(logprobs: Dict[str, float], min_weight: float = 0.25) -> Optional[float]:
    """
    Compute weighted average score from logprobs over integer tokens 0-100.

    Handles REFUSAL token explicitly - if REFUSAL has highest probability, returns None.

    Args:
        logprobs: Dict mapping token strings to probabilities
        min_weight: Minimum total weight on valid integers to return a score

    Returns:
        Weighted average score, or None if refusal/not enough weight
    """
    # Check for explicit REFUSAL
    refusal_prob = logprobs.get("REFUSAL", 0.0)

    # Find max probability among valid integers
    max_int_prob = 0.0
    total_weight = 0.0
    weighted_sum = 0.0

    for token, prob in logprobs.items():
        try:
            value = int(token.strip())
            if 0 <= value <= 100:
                weighted_sum += value * prob
                total_weight += prob
                max_int_prob = max(max_int_prob, prob)
        except ValueError:
            continue

    # If REFUSAL is more likely than any integer, treat as refusal
    if refusal_prob > max_int_prob and refusal_prob > 0.1:
        return None

    if total_weight < min_weight:
        return None

    return weighted_sum / total_weight


class BaseJudge(ABC):
    """Abstract base class for LLM judges."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def get_logprobs(self, prompt: str) -> Dict[str, float]:
        """Get top logprobs for first token of response."""
        pass

    async def score(
        self,
        eval_prompt: str,
        question: str,
        answer: str,
    ) -> Optional[float]:
        """
        Score how strongly the answer exhibits the trait.

        Args:
            eval_prompt: Full prompt template with {question} and {answer}
            question: The original question
            answer: The model's response

        Returns:
            Score 0-100, or None if scoring failed/refused
        """
        prompt = eval_prompt.format(question=question, answer=answer)
        logprobs = await self.get_logprobs(prompt)
        return aggregate_logprob_score(logprobs)

    async def score_coherence(self, answer: str) -> Optional[float]:
        """Score coherence of response (0-100)."""
        prompt = COHERENCE_PROMPT.format(answer=answer)
        logprobs = await self.get_logprobs(prompt)
        return aggregate_logprob_score(logprobs)


class OpenAIJudge(BaseJudge):
    """OpenAI-based judge using gpt-4o-mini."""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(model)
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def get_logprobs(self, prompt: str) -> Dict[str, float]:
        """Get top-20 logprobs from OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=20,
                seed=42,
            )

            content = response.choices[0].logprobs.content
            if not content:
                return {}

            top_logprobs = content[0].top_logprobs
            return {lp.token: math.exp(lp.logprob) for lp in top_logprobs}

        except Exception as e:
            print(f"OpenAI API error: {e}")
            return {}


class GeminiJudge(BaseJudge):
    """Gemini-based judge using gemini-2.5-flash."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        super().__init__(model)
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        self.genai = genai

    async def get_logprobs(self, prompt: str) -> Dict[str, float]:
        """
        Get response from Gemini.

        Note: Gemini doesn't have the same logprobs API as OpenAI.
        We parse the response text directly.
        """
        try:
            model = self.genai.GenerativeModel(self.model)

            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=self.genai.GenerationConfig(
                    max_output_tokens=5,
                    temperature=0,
                ),
            )

            text = response.text.strip()

            # Check for REFUSAL
            if "REFUSAL" in text.upper():
                return {"REFUSAL": 1.0}

            # Parse as number
            try:
                value = int(text.split()[0])
                if 0 <= value <= 100:
                    return {str(value): 1.0}
            except (ValueError, IndexError):
                pass

            return {}

        except Exception as e:
            print(f"Gemini API error: {e}")
            return {}


class TraitJudge:
    """
    Unified judge interface supporting multiple providers.

    Usage:
        judge = TraitJudge(provider="openai")
        score = await judge.score(eval_prompt, question, answer)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
    ):
        """
        Args:
            provider: "openai" or "gemini"
            model: Model name (defaults to gpt-4o-mini or gemini-2.5-flash)
        """
        if provider == "openai":
            model = model or "gpt-4o-mini"
            self._judge = OpenAIJudge(model)
        elif provider == "gemini":
            model = model or "gemini-2.5-flash"
            self._judge = GeminiJudge(model)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'gemini'")

        self.provider = provider
        self.model = model

    async def score(
        self,
        eval_prompt: str,
        question: str,
        answer: str,
    ) -> Optional[float]:
        """Score trait expression 0-100."""
        return await self._judge.score(eval_prompt, question, answer)

    async def score_coherence(self, answer: str) -> Optional[float]:
        """Score coherence 0-100."""
        return await self._judge.score_coherence(answer)

    async def score_batch(
        self,
        eval_prompt: str,
        items: List[Tuple[str, str]],
        include_coherence: bool = True,
    ) -> List[Dict]:
        """
        Score a batch of (question, answer) pairs concurrently.

        Args:
            eval_prompt: Full prompt template with {question} and {answer}
            items: List of (question, answer) tuples
            include_coherence: Also compute coherence scores

        Returns:
            List of dicts with 'trait_score' and optionally 'coherence_score'
        """
        async def score_one(question: str, answer: str) -> dict:
            result = {
                "trait_score": await self.score(eval_prompt, question, answer)
            }
            if include_coherence:
                result["coherence_score"] = await self._judge.score_coherence(answer)
            return result

        tasks = [score_one(q, a) for q, a in items]
        return await asyncio.gather(*tasks)
