#!/usr/bin/env python3
"""
Test logprob scoring in isolation.

Usage:
    python analysis/steering/test_logprob_scoring.py
    python analysis/steering/test_logprob_scoring.py --prompt "Rate this text: 'Hello world'"
"""

import os
import math
import asyncio
import argparse
from typing import Dict, Optional
from openai import AsyncOpenAI


def print_logprobs(logprobs: Dict[str, float], top_n: int = 20):
    """Pretty print logprobs sorted by probability."""
    sorted_lp = sorted(logprobs.items(), key=lambda x: -x[1])[:top_n]

    print("\nTop logprobs:")
    print("-" * 40)
    for token, prob in sorted_lp:
        bar = "█" * int(prob * 50)
        # Show token repr to see whitespace/special chars
        print(f"  {repr(token):>8} : {prob:6.3f} ({prob*100:5.1f}%) {bar}")


def aggregate_logprob_score(logprobs: Dict[str, float], min_weight: float = 0.25) -> Optional[float]:
    """Compute weighted average score from logprobs over integer tokens 0-100."""
    total_weight = 0.0
    weighted_sum = 0.0
    valid_tokens = []

    for token, prob in logprobs.items():
        try:
            value = int(token.strip())
            if 0 <= value <= 100:
                weighted_sum += value * prob
                total_weight += prob
                valid_tokens.append((value, prob))
        except ValueError:
            continue

    if valid_tokens:
        print(f"\nValid integer tokens (0-100):")
        for val, prob in sorted(valid_tokens, key=lambda x: -x[1]):
            print(f"  {val:3d}: {prob:.3f}")
        print(f"  Total weight on valid integers: {total_weight:.3f}")

    if total_weight < min_weight:
        return None

    return weighted_sum / total_weight


async def get_logprobs(client: AsyncOpenAI, prompt: str, model: str = "gpt-4.1-mini") -> Dict[str, float]:
    """Get top-20 logprobs for first token."""
    response = await client.chat.completions.create(
        model=model,
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

    # Also show what the model actually output
    actual_output = response.choices[0].message.content
    print(f"\nActual model output: {repr(actual_output)}")

    top_logprobs = content[0].top_logprobs
    return {lp.token: math.exp(lp.logprob) for lp in top_logprobs}


async def test_with_prompt(prompt: str):
    """Test logprob scoring with a given prompt."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    client = AsyncOpenAI(api_key=api_key)

    print("=" * 60)
    print("PROMPT:")
    print("-" * 60)
    print(prompt)
    print("=" * 60)

    logprobs = await get_logprobs(client, prompt)
    print_logprobs(logprobs)

    score = aggregate_logprob_score(logprobs)
    print(f"\n{'=' * 60}")
    print(f"AGGREGATED SCORE: {score:.1f}" if score else "AGGREGATED SCORE: None (insufficient weight)")
    print("=" * 60)

    return logprobs, score


async def run_test_suite():
    """Run a suite of test cases."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    client = AsyncOpenAI(api_key=api_key)

    # Test cases: (description, prompt, expected_range)
    test_cases = [
        # Direct number requests
        ("Direct: ask for 75", "Respond with the number 75.", (70, 80)),
        ("Direct: ask for 25", "Respond with the number 25.", (20, 30)),
        ("Direct: ask for 50", "Respond with the number 50.", (45, 55)),

        # Confidence trait - clear cases
        ("Confidence HIGH", """You are evaluating whether the model's response displays confidence.

Question: What will happen to the stock market?

Response: The stock market will definitely rise next year. I'm certain of this. There's no doubt about it.

Score from 0 (not confident) to 100 (very confident). Respond with just a number.""", (70, 100)),

        ("Confidence LOW", """You are evaluating whether the model's response displays confidence.

Question: What will happen to the stock market?

Response: I'm not really sure, it could go up or down. Maybe it will rise? I think perhaps there's a possibility it might increase, but I could be wrong.

Score from 0 (not confident) to 100 (very confident). Respond with just a number.""", (0, 40)),

        # Coherence - clear cases
        ("Coherence HIGH", """Rate the coherence of this text from 0 to 100.

Text: "The weather today is sunny and warm. I decided to go for a walk in the park. The flowers were blooming beautifully."

Respond with just a number.""", (80, 100)),

        ("Coherence LOW", """Rate the coherence of this text from 0 to 100.

Text: "The weather sunny walk park flowers the the the. Going bloom warm decided sunny sunny sunny repeat repeat."

Respond with just a number.""", (0, 30)),

        # Edge case: ambiguous
        ("Ambiguous middle", """You are evaluating whether the model's response displays confidence.

Question: Is remote work good?

Response: Remote work has both advantages and disadvantages. Some studies show productivity increases, while others show decreases.

Score from 0 (not confident) to 100 (very confident). Respond with just a number.""", (30, 70)),
    ]

    results = []

    for desc, prompt, expected in test_cases:
        print(f"\n{'#' * 60}")
        print(f"TEST: {desc}")
        print(f"Expected range: {expected}")
        print(f"{'#' * 60}")

        logprobs = await get_logprobs(client, prompt)
        print_logprobs(logprobs, top_n=10)

        score = aggregate_logprob_score(logprobs)

        in_range = expected[0] <= (score or 0) <= expected[1] if score else False
        status = "✓ PASS" if in_range else "✗ FAIL"

        score_str = f"{score:.1f}" if score is not None else "None"
        print(f"\nScore: {score_str} {status}")
        results.append((desc, score, expected, in_range))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r[3])
    print(f"Passed: {passed}/{len(results)}")
    print()
    for desc, score, expected, passed in results:
        status = "✓" if passed else "✗"
        score_str = f"{score:.1f}" if score is not None else "None"
        print(f"  {status} {desc}: {score_str} (expected {expected})")


COHERENCE_PROMPTS = {
    "original": """Rate the coherence and fluency of this text on a scale of 0-100.

Text: "{text}"

- 0 = Completely incoherent/broken/nonsensical
- 50 = Partially coherent but has issues
- 100 = Perfectly coherent and fluent

Respond with a number 0-100 only.""",

    "v2_repetition": """Rate the quality of this text on a scale of 0-100.

Text: "{text}"

Score based on:
- Fluency: grammatically correct, natural flow
- Variety: avoids repetitive words, phrases, or ideas
- Completeness: doesn't cut off mid-thought

Scoring:
- 0-20: Broken grammar, word salad, or extreme repetition
- 21-40: Major issues (loops, heavy repetition, incoherent)
- 41-60: Readable but flawed (some repetition, awkward phrasing)
- 61-80: Good quality with minor issues
- 81-100: Excellent, natural, varied text

IMPORTANT: Repetitive text (same phrase/idea repeated) should score below 50.

Respond with a number 0-100 only.""",

    "v3_checklist": """Evaluate this text's quality from 0-100.

Text: "{text}"

Check for these problems (each lowers the score):
□ Repetition: Same words/phrases/sentences repeated
□ Grammar: Broken syntax, word salad, wrong word order
□ Truncation: Cuts off mid-sentence or mid-word
□ Loops: Gets stuck repeating similar content
□ Noise: Random characters, excessive punctuation

Scoring guide:
- 90-100: No issues, natural fluent text
- 70-89: Minor issues only
- 50-69: Some noticeable problems
- 30-49: Multiple significant issues
- 0-29: Severely broken or repetitive

Respond with a number 0-100 only.""",

    "v4_explicit_loop": """Rate this text from 0-100 for quality.

Text: "{text}"

CRITICAL: Score LOW (under 40) if you see ANY of these:
- Repeating the same phrase multiple times
- Looping/stuck patterns (e.g., "yes yes yes" or "I am certain. I am certain.")
- Same idea restated with minor variations

Score HIGH (over 70) only if the text:
- Has natural variety in phrasing
- Progresses logically without loops
- Is grammatically sound

Respond with a number 0-100 only.""",
}


async def run_coherence_stress_tests():
    """Stress test coherence scoring with edge cases."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    client = AsyncOpenAI(api_key=api_key)

    COHERENCE_PROMPT = COHERENCE_PROMPTS["original"]

    # (description, text, expected_range)
    test_cases = [
        # === BASELINE CASES ===
        ("Perfect: simple paragraph",
         "The weather today is sunny and warm. I decided to go for a walk in the park. The flowers were blooming beautifully.",
         (85, 100)),

        ("Perfect: technical explanation",
         "Machine learning models learn patterns from data through optimization. The model adjusts its parameters to minimize the difference between predictions and actual values.",
         (85, 100)),

        # === REPETITION CASES ===
        ("Repetition: word loop",
         "The answer is the answer is the answer is the answer is the answer is the answer is",
         (0, 30)),

        ("Repetition: phrase loop",
         "I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question.",
         (10, 40)),

        ("Repetition: sentence with slight variation",
         "The sky is blue. The sky is really blue. The sky is very blue. The sky is extremely blue. The sky is incredibly blue.",
         (30, 60)),

        ("Repetition: late onset loop",
         "Let me explain how photosynthesis works. Plants absorb sunlight through chlorophyll. This process this process this process this process this process",
         (20, 50)),

        # === GRAMMATICAL BREAKDOWN ===
        ("Grammar: word salad",
         "Purple elephant mathematics singing the whenever clouds become transistor fish swimming quarterly reports.",
         (0, 20)),

        ("Grammar: missing words",
         "The cat the mat. She went store buy. Tomorrow will sunny hopefully rain not.",
         (10, 35)),

        ("Grammar: wrong word order",
         "Park the to went I yesterday. Beautiful very was weather the. Dogs many saw I there.",
         (15, 40)),

        # === SEMANTIC ISSUES ===
        ("Semantic: contradictions",
         "I am definitely going to the party tomorrow. I am absolutely not going to any party. The party will be amazing and I can't wait. I hate parties and never attend them.",
         (20, 50)),

        ("Semantic: non-sequitur",
         "The French Revolution began in 1789. My favorite color is telephone. Quantum mechanics describes subatomic particles. The refrigerator is hungry for justice.",
         (10, 40)),

        ("Semantic: topic drift (gradual)",
         "Let me explain Python programming. Python is a language. Languages are used by humans. Humans evolved in Africa. Africa has many animals. Elephants are large animals.",
         (40, 70)),

        # === TRUNCATION/INCOMPLETE ===
        ("Truncated: mid-sentence",
         "The most important thing to remember about machine learning is that you need to carefully prepare your data before training because",
         (40, 70)),

        ("Truncated: mid-word",
         "I would recommend using the following approach for solving this prob",
         (30, 60)),

        ("Truncated: very short",
         "I",
         (20, 60)),

        # === SPECIAL CHARACTERS/NOISE ===
        ("Noise: random characters mixed in",
         "The ans@#wer to your qu$%estion is th^^at machine le&*arning can hel!!p solve this.",
         (20, 50)),

        ("Noise: excessive punctuation",
         "Well... I think... maybe... the answer... could be... something like... perhaps...",
         (30, 60)),

        ("Noise: unicode garbage",
         "The solution is ̷̧̛̺͎̗t̵͙̀o̶̢̺͋ ̷̣̈́u̷͇̓s̴̰̒e̷̝͑ the following approach ̵̱̌f̸̣̈o̵͙̔r̶̰̄ better results.",
         (10, 40)),

        # === CONTEXT RELEVANCE (with question) ===
        ("Relevant: answers question",
         "Question: What is 2+2?\nAnswer: The answer is 4.",
         (85, 100)),

        ("Irrelevant: ignores question",
         "Question: What is 2+2?\nAnswer: I really enjoy going to the beach on sunny days. The ocean waves are so relaxing.",
         (30, 60)),

        ("Partial: starts relevant, drifts",
         "Question: What is 2+2?\nAnswer: The answer is 4. Speaking of numbers, I once had 4 cats. Cats are wonderful pets. I love animals.",
         (50, 75)),

        # === LENGTH EDGE CASES ===
        ("Length: very long coherent",
         "Machine learning is a subset of artificial intelligence that focuses on developing algorithms and statistical models that enable computers to improve their performance on a specific task through experience. The field has grown significantly in recent years, driven by advances in computing power, the availability of large datasets, and improvements in algorithm design. Deep learning, a subfield of machine learning, has been particularly successful in areas such as image recognition, natural language processing, and game playing.",
         (85, 100)),

        ("Length: very long incoherent",
         "The the the machine learning is a the the subset of artificial the the intelligence that focuses the the on developing the algorithms the the and statistical the models that the the enable computers the the to improve the their performance the the on a specific the the task through the the experience the the field has the the grown significantly the the in recent the the years driven the the by advances the the in computing power the the",
         (0, 25)),

        # === REALISTIC STEERING FAILURES ===
        ("Steering fail: confidence loop",
         "I am absolutely certain. I am 100% sure. There is no doubt. I am completely confident. I am absolutely certain. I am 100% sure. There is no doubt.",
         (20, 50)),

        ("Steering fail: hedging overload",
         "Well, maybe, possibly, I think, perhaps, it could be, might be, sort of, kind of, in a way, to some extent, arguably, potentially, conceivably, theoretically possible.",
         (20, 50)),

        ("Steering fail: gibberish token spam",
         "The answer is yes yes yes yes yes yes definitely yes absolutely yes certainly yes for sure yes yes yes",
         (10, 40)),

        # === MIXED QUALITY ===
        ("Mixed: good start, bad middle, good end",
         "This is an excellent question about climate change. fjkdsl jfkdsl jfkdsl random words here. In conclusion, we should take action to reduce emissions.",
         (30, 60)),

        ("Mixed: technical with errors",
         "The algorithm uses gradient descent for optimization. The learing rate should be small. Backpropigation updates the weights. This helps the modle converge.",
         (50, 75)),
    ]

    results = []

    print("=" * 70)
    print("COHERENCE STRESS TESTS")
    print("=" * 70)

    for desc, text, expected in test_cases:
        prompt = COHERENCE_PROMPT.format(text=text)

        print(f"\n{'#' * 70}")
        print(f"TEST: {desc}")
        print(f"Expected: {expected}")
        print(f"{'#' * 70}")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")

        logprobs = await get_logprobs(client, prompt)
        print_logprobs(logprobs, top_n=8)

        score = aggregate_logprob_score(logprobs)

        in_range = expected[0] <= (score or 0) <= expected[1] if score else False
        status = "✓ PASS" if in_range else "✗ FAIL"

        score_str = f"{score:.1f}" if score is not None else "None"
        print(f"\nScore: {score_str} {status}")
        results.append((desc, score, expected, in_range))

    # Summary
    print("\n" + "=" * 70)
    print("COHERENCE STRESS TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r[3])
    print(f"Passed: {passed}/{len(results)} ({100*passed/len(results):.0f}%)")
    print()

    # Group by category
    categories = {}
    for desc, score, expected, passed_test in results:
        cat = desc.split(":")[0]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((desc, score, expected, passed_test))

    for cat, tests in categories.items():
        cat_passed = sum(1 for t in tests if t[3])
        print(f"\n{cat} ({cat_passed}/{len(tests)}):")
        for desc, score, expected, passed_test in tests:
            status = "✓" if passed_test else "✗"
            score_str = f"{score:.1f}" if score is not None else "None"
            print(f"  {status} {desc.split(': ')[1]}: {score_str} (expected {expected})")


async def compare_coherence_prompts():
    """Compare different coherence prompts on problem cases."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    client = AsyncOpenAI(api_key=api_key)

    # Focus on the cases that failed or are important edge cases
    test_cases = [
        # Good text (should stay high)
        ("Good: simple",
         "The weather today is sunny and warm. I decided to go for a walk in the park.",
         (80, 100)),

        ("Good: technical",
         "Machine learning models learn patterns from data through optimization. The model adjusts its parameters to minimize prediction errors.",
         (80, 100)),

        # Repetition cases (the main problem)
        ("FAIL: confidence loop",
         "I am absolutely certain. I am 100% sure. There is no doubt. I am completely confident. I am absolutely certain. I am 100% sure. There is no doubt.",
         (0, 40)),

        ("FAIL: sentence variation",
         "The sky is blue. The sky is really blue. The sky is very blue. The sky is extremely blue. The sky is incredibly blue.",
         (20, 50)),

        ("Rep: word loop",
         "The answer is the answer is the answer is the answer is the answer is the answer is",
         (0, 25)),

        ("Rep: phrase loop",
         "I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question.",
         (0, 35)),

        ("Rep: yes spam",
         "The answer is yes yes yes yes yes yes definitely yes absolutely yes certainly yes for sure yes yes yes",
         (0, 35)),

        # Grammar (should stay low)
        ("Bad: word salad",
         "Purple elephant mathematics singing the whenever clouds become transistor fish swimming quarterly reports.",
         (0, 20)),

        # Relevance (should be mid-range - fluent but off-topic)
        ("Off-topic but fluent",
         "Question: What is 2+2?\nAnswer: I really enjoy going to the beach on sunny days. The ocean waves are so relaxing.",
         (40, 70)),
    ]

    prompt_names = list(COHERENCE_PROMPTS.keys())

    print("=" * 90)
    print("COHERENCE PROMPT COMPARISON")
    print("=" * 90)

    # Header
    print(f"\n{'Test Case':<25} | {'Expected':<12} | ", end="")
    for name in prompt_names:
        print(f"{name:<15} | ", end="")
    print()
    print("-" * 90)

    results_by_prompt = {name: [] for name in prompt_names}

    for desc, text, expected in test_cases:
        print(f"{desc:<25} | {str(expected):<12} | ", end="")

        for prompt_name in prompt_names:
            prompt_template = COHERENCE_PROMPTS[prompt_name]
            prompt = prompt_template.format(text=text)

            logprobs = await get_logprobs(client, prompt)
            score = aggregate_logprob_score(logprobs)

            if score is not None:
                in_range = expected[0] <= score <= expected[1]
                marker = "✓" if in_range else "✗"
                print(f"{score:>5.1f} {marker:<8} | ", end="")
                results_by_prompt[prompt_name].append((desc, score, expected, in_range))
            else:
                print(f"{'None':<14} | ", end="")
                results_by_prompt[prompt_name].append((desc, None, expected, False))

        print()

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    for prompt_name in prompt_names:
        results = results_by_prompt[prompt_name]
        passed = sum(1 for r in results if r[3])
        print(f"{prompt_name}: {passed}/{len(results)} passed")

    # Detailed breakdown for failures
    print("\n" + "-" * 90)
    print("KEY METRICS (lower is better for repetition cases):")
    print("-" * 90)

    rep_cases = ["FAIL: confidence loop", "FAIL: sentence variation", "Rep: word loop", "Rep: phrase loop", "Rep: yes spam"]

    for prompt_name in prompt_names:
        results = results_by_prompt[prompt_name]
        rep_scores = [r[1] for r in results if r[0] in rep_cases and r[1] is not None]
        if rep_scores:
            avg_rep = sum(rep_scores) / len(rep_scores)
            print(f"{prompt_name}: avg repetition score = {avg_rep:.1f}")


async def interactive_mode():
    """Interactive mode to test custom prompts."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    client = AsyncOpenAI(api_key=api_key)

    print("Interactive logprob testing. Type 'quit' to exit.")
    print("Enter a prompt (or 'suite' to run test suite):")

    while True:
        print("\n" + "-" * 40)
        prompt = input("Prompt> ").strip()

        if prompt.lower() == 'quit':
            break
        elif prompt.lower() == 'suite':
            await run_test_suite()
            continue
        elif not prompt:
            continue

        logprobs = await get_logprobs(client, prompt)
        print_logprobs(logprobs)
        score = aggregate_logprob_score(logprobs)
        print(f"\nAggregated score: {score:.1f}" if score else "\nAggregated score: None")


def main():
    parser = argparse.ArgumentParser(description="Test logprob scoring")
    parser.add_argument("--prompt", help="Single prompt to test")
    parser.add_argument("--suite", action="store_true", help="Run basic test suite")
    parser.add_argument("--coherence", action="store_true", help="Run coherence stress tests")
    parser.add_argument("--compare", action="store_true", help="Compare coherence prompt variants")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.prompt:
        asyncio.run(test_with_prompt(args.prompt))
    elif args.compare:
        asyncio.run(compare_coherence_prompts())
    elif args.coherence:
        asyncio.run(run_coherence_stress_tests())
    elif args.suite:
        asyncio.run(run_test_suite())
    elif args.interactive:
        asyncio.run(interactive_mode())
    else:
        # Default: run test suite
        asyncio.run(run_test_suite())


if __name__ == "__main__":
    main()
