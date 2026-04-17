#!/usr/bin/env python3
"""Llama 70B batched-12 story generation test (Track C Q-C3).

Purpose: verify whether Llama 3.3 70B Instruct produces 12 distinct stories in
one call when given the paper's verbatim batched prompt. Gates Track C's
--replication-level full flag work.

Success criterion: ≥10 of 12 stories per emotion are meaningfully distinct
(different protagonists, topics, or narrative arcs).

Usage:
    python dev/test_llama_batched_12.py --load-in-4bit
    python dev/test_llama_batched_12.py --load-in-4bit --emotions happy afraid
    python dev/test_llama_batched_12.py --load-in-4bit --n-stories 6 --emotions happy

Output: prints per-emotion diagnostic + writes JSON to /tmp/llama_batched_test.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.model import load_model, format_prompt
from utils.model_generation import generate_batch


# Paper's verbatim batched-story prompt (Appendix, lines 1376-1408).
PAPER_BATCHED_PROMPT = """Write {n_stories} different stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Format the stories like so:

[story 1]

[story 2]

[story 3]

etc.

The paragraphs should each be a fresh start, with no continuity. Try to make them diverse and not use the same turns of phrase. Across the different stories, use a mix of third-person narration and first-person narration.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the stories. Instead, convey the emotion ONLY through:

- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named."""


# Test topics (small sample, paper-inspired)
TEST_TOPICS = [
    "A person learns their childhood bully became a therapist",
    "Someone discovers their partner has been writing a novel about them",
    "A homeowner discovers previous residents left items in the attic",
]


def parse_stories(response: str) -> list[str]:
    """Parse [story N] delimited response into list of story bodies.

    Paper-requested format: [story 1] ... [story 2] ... etc.
    Also accepts common LLM variants: "Story 1:", "**Story 1**", numbered list.
    """
    # Primary delimiter: [story N]
    pattern = r"\[story\s*\d+\]"
    parts = re.split(pattern, response, flags=re.IGNORECASE)
    if len(parts) > 1:
        # First split is pre-amble (usually empty or intro text); skip
        stories = [p.strip() for p in parts[1:] if p.strip()]
        if stories:
            return stories

    # Fallback 1: "Story N:" or "Story N\n"
    pattern2 = r"(?:^|\n)\s*(?:\*\*)?Story\s*\d+[:\.]?(?:\*\*)?\s*\n"
    parts = re.split(pattern2, response, flags=re.IGNORECASE)
    if len(parts) > 1:
        stories = [p.strip() for p in parts[1:] if p.strip()]
        if stories:
            return stories

    # Fallback 2: numbered list "1. ..." at line start
    pattern3 = r"(?:^|\n)\s*\d+[\.\)]\s"
    parts = re.split(pattern3, response)
    if len(parts) > 1:
        stories = [p.strip() for p in parts[1:] if p.strip()]
        if stories:
            return stories

    # No delimiter found; return whole response as one "story"
    return [response.strip()] if response.strip() else []


def measure_distinctness(stories: list[str]) -> dict:
    """Measure diversity across stories. Returns diagnostic dict."""
    n = len(stories)
    if n < 2:
        return {"n_stories": n, "pairwise_mean_jaccard": None, "pairwise_max_jaccard": None, "diversity_ok": False}

    # Token-level Jaccard similarity (lowercase + word-tokenize, skip stopwords-lite)
    def toks(s):
        return set(re.findall(r"\b[a-z]{3,}\b", s.lower()))

    token_sets = [toks(s) for s in stories]
    pairwise_jaccard = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = token_sets[i], token_sets[j]
            if not a and not b:
                pairwise_jaccard.append(0.0)
            elif not a or not b:
                pairwise_jaccard.append(0.0)
            else:
                pairwise_jaccard.append(len(a & b) / len(a | b))

    mean_j = sum(pairwise_jaccard) / len(pairwise_jaccard)
    max_j = max(pairwise_jaccard)
    # Heuristic: if mean token Jaccard < 0.3, stories are well-differentiated
    diversity_ok = mean_j < 0.30 and max_j < 0.60

    return {
        "n_stories": n,
        "pairwise_mean_jaccard": round(mean_j, 3),
        "pairwise_max_jaccard": round(max_j, 3),
        "diversity_ok": diversity_ok,
        "story_lengths": [len(s) for s in stories],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--emotions", nargs="+", default=["happy", "afraid", "desperate", "calm"])
    p.add_argument("--topic-index", type=int, default=0, help="Index into TEST_TOPICS (0-2)")
    p.add_argument("--n-stories", type=int, default=12, help="Paper uses 12")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=3000, help="12 stories * ~200 tok each + overhead")
    p.add_argument("--output", default="/tmp/llama_batched_test.json")
    args = p.parse_args()

    print(f"=== Llama N={args.n_stories} batched-generation test ===")
    print(f"Model: {args.model}")
    print(f"4-bit: {args.load_in_4bit}")
    print(f"Emotions: {args.emotions}")
    print(f"Topic: {TEST_TOPICS[args.topic_index]!r}")
    print()

    print("Loading model...")
    t0 = time.time()
    model, tokenizer = load_model(args.model, load_in_4bit=args.load_in_4bit)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print()

    topic = TEST_TOPICS[args.topic_index]
    prompts = []
    for emo in args.emotions:
        prompts.append(
            PAPER_BATCHED_PROMPT.format(
                n_stories=args.n_stories, topic=topic, emotion=emo
            )
        )

    # Generate
    print(f"Generating {len(args.emotions)} batched-{args.n_stories} responses...")
    t0 = time.time()
    responses = generate_batch(
        model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"Generation done in {time.time() - t0:.1f}s")
    print()

    # Parse + diagnose per emotion
    results = []
    for emo, resp in zip(args.emotions, responses):
        stories = parse_stories(resp)
        diag = measure_distinctness(stories)
        row = {
            "emotion": emo,
            "topic": topic,
            "n_stories_requested": args.n_stories,
            "n_stories_parsed": len(stories),
            "parse_ok": len(stories) >= args.n_stories * 0.67,  # at least 2/3 parsed
            **diag,
            "raw_response_len": len(resp),
            "first_story": stories[0][:300] if stories else "",
            "last_story": stories[-1][:300] if stories else "",
        }
        results.append(row)

        print(f"--- {emo} ---")
        print(f"  Parsed {len(stories)}/{args.n_stories} stories")
        print(f"  Pairwise Jaccard: mean={diag.get('pairwise_mean_jaccard')}, max={diag.get('pairwise_max_jaccard')}")
        print(f"  Diversity OK: {diag.get('diversity_ok')}")
        if stories:
            print(f"  First story (300 chars): {stories[0][:300]!r}")
            print(f"  Last story (300 chars): {stories[-1][:300]!r}")
        print()

    # Verdict
    n_pass = sum(1 for r in results if r["parse_ok"] and r.get("diversity_ok", False))
    total = len(results)
    print(f"=== VERDICT ===")
    print(f"{n_pass}/{total} emotions pass both parse + diversity checks.")
    if n_pass == total:
        print("RESULT: Llama 70B produces distinct stories at N=12 — proceed with batched-gen flag.")
    elif n_pass >= total // 2:
        print("RESULT: MIXED. Some emotions work; some don't. Investigate per-emotion variance; consider N=6 fallback.")
    else:
        print("RESULT: Llama 70B struggles with N=12 batching. Consider: N=6, higher temperature, or scrap batching.")

    output = {
        "model": args.model,
        "n_stories_requested": args.n_stories,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "topic": topic,
        "results": results,
        "verdict_n_pass": n_pass,
        "verdict_n_total": total,
        "full_responses": list(zip(args.emotions, responses)),
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull output saved to {args.output}")


if __name__ == "__main__":
    main()
