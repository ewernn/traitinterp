"""Explore story generation strategies for Emotion Concepts replication.

Tries different topic counts, selections, prompt templates, and temperatures
on a small sample of emotions to find the best configuration.

Run on GPU:
    python experiments/ant_emotion_concepts/scripts/explore_story_generation.py \
        --model meta-llama/Llama-3.3-70B-Instruct --load-in-4bit

Output: experiments/ant_emotion_concepts/results/story_exploration/
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Anthropic's 100 topics from Appendix A.12
ALL_TOPICS = [
    "An artist discovers someone has tattooed their work",
    "A family member announces they're converting to a different religion",
    "Someone's childhood imaginary friend appears in their niece's drawings",
    "A person finds out their biography was written without their knowledge",
    "A neighbor starts a renovation project",
    "Someone finds their grandmother's engagement ring in a pawn shop",
    "A student learns their scholarship application was denied",
    "A person's online friend turns out to live in the same city",
    "A neighbor wants to install a fence",
    "An adult child moves back in with their parents",
    "An employee is asked to train their replacement",
    "An athlete is asked to switch positions",
    "A traveler's flight is delayed, causing them to miss an important event",
    "A student is accused of plagiarism",
    "A person discovers their mentor has retired without saying goodbye",
    "Two friends both apply for the same job",
    "A person runs into their ex at a mutual friend's wedding",
    "Someone discovers their friend has been lying about their job",
    "A person discovers their partner has been taking secret phone calls",
    "A person discovers their child has the same teacher they had",
    "A person's car is towed from their own driveway",
    "Two friends realize they remember a shared event completely differently",
    "Someone discovers their mother kept every school assignment",
    "A person discovers their teenage diary has been published online",
    "Someone finds out their medical records were mixed up with another patient's",
    "A person finds out their article was published under someone else's name",
    "An athlete doesn't make the team they expected to join",
    "An employee is transferred to a different department",
    "Someone receives a friend request from a childhood bully",
    "A person finds out their surprise party has been cancelled",
    "An employee finds out a junior colleague makes more money",
    "A person finds out their partner has been learning their native language",
    "A chef receives a harsh review from a food critic",
    "A person learns their favorite restaurant is closing",
    "Someone finds their childhood teddy bear at a yard sale",
    "A homeowner discovers previous residents left items in the attic",
    "Someone finds an unsigned birthday card in their mailbox",
    "Someone discovers a hidden room in their new house",
    "Two strangers realize they've been dating the same person",
    "A person finds a hidden letter in a used book",
    "Two siblings inherit their grandmother's house",
    "Someone finds a wallet containing a large sum of cash",
    "Someone receives an invitation to their high school reunion",
    "Someone discovers their recipe has become famous under another name",
    "A college student discovers their roommate has been reading their journal",
    "A person finds out they were adopted through a DNA test",
    "A family member wants to sell a cherished heirloom",
    "Someone receives a package intended for the previous tenant",
    "Someone's childhood home is about to be demolished",
    "A person's invention is already patented by someone else",
    "A neighbor's dog keeps escaping into their yard",
    "A coach has to cut a player from the team",
    "Someone learns their favorite author plagiarized their stories",
    "A student finds out their scholarship was meant for someone else",
    "Someone discovers their teenager has a secret social media account",
    "Two roommates disagree about getting a pet",
    "Two friends plan separate birthday parties on the same day",
    "A person learns their childhood best friend doesn't remember them",
    "A musician hears their song being performed by someone else",
    "A person's manuscript is rejected by their dream publisher",
    "A person finds old photos that contradict family stories",
    "A person is asked to give a speech at their parent's retirement party",
    "A student discovers their teacher follows them on social media",
    "A parent finds an old letter they wrote but never sent",
    "An employee discovers the company is being sold",
    "A person accidentally sends a text to the wrong recipient",
    "Two coworkers are stuck in an elevator for three hours",
    "A student learns their thesis advisor is leaving the university",
    "A person's longtime hobby becomes their child's obsession",
    "Two colleagues are both considered for the same promotion",
    "Two coworkers discover they went to the same summer camp",
    "A tenant receives an eviction notice",
    "Someone finds their parent's draft letter of resignation from decades ago",
    "Someone finds out their best friend is moving across the country",
    "A neighbor's tree falls on their property",
    "Someone receives an apology letter years after the incident",
    "A person discovers the tree they planted as a child has been cut down",
    "Two siblings discover different versions of their inheritance",
    "A person finds their childhood home listed for sale online",
    "A homeowner learns their house was a former crime scene",
    "Someone finds out they have a half-sibling they never knew about",
    "A person learns their childhood bully became a therapist",
    "Two people discover they've been working on identical projects",
    "A person finds their spouse's secret savings account",
    "A neighbor complains about noise levels",
    "Someone finds their deceased parent's bucket list",
    "A teacher receives an unexpected gift from a former student",
    "An artist's work is displayed without their permission",
    "Someone discovers their neighbor is secretly wealthy",
    "A student receives a much lower grade than expected",
    "A person learns their college is closing down",
    "A neighbor asks to cut down a tree on the property line",
    "Two strangers discover they share the same rare medical condition",
    "Someone receives flowers with no card attached",
    "Someone discovers their partner has been writing a novel about them",
    "Someone finds a time capsule they don't remember burying",
    "Someone finds their partner's bucket list",
    "A neighbor asks to use part of the yard for a garden",
    "A person learns their apartment building is going condo",
    "Someone finds their college application essay published as an example",
]

# Anthropic's story prompt (adapted for 1-per-call)
STORY_PROMPT_TEMPLATE = """Write a story based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the story. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named."""

# Test emotions spanning the valence/arousal space
TEST_EMOTIONS = ["happy", "desperate", "calm", "angry", "brooding"]

# Topic selection strategies
def select_topics_random(n, seed=42):
    rng = random.Random(seed)
    return rng.sample(ALL_TOPICS, n)

def select_topics_first_n(n):
    return ALL_TOPICS[:n]

def select_topics_stratified(n):
    """Pick topics spanning diverse contexts: interpersonal, professional, discovery, loss, etc."""
    # Hand-picked for diversity
    diverse = [
        "An artist discovers someone has tattooed their work",           # discovery/creative
        "A student learns their scholarship application was denied",      # professional/loss
        "A person runs into their ex at a mutual friend's wedding",      # interpersonal/awkward
        "Someone finds their grandmother's engagement ring in a pawn shop",  # family/discovery
        "A tenant receives an eviction notice",                          # crisis/practical
        "A person discovers their mentor has retired without saying goodbye",  # professional/loss
        "Someone discovers a hidden room in their new house",            # discovery/exciting
        "A person finds out they were adopted through a DNA test",       # identity/family
        "Two friends realize they remember a shared event completely differently",  # interpersonal
        "A musician hears their song being performed by someone else",   # creative/ambiguous
        "A person finds old photos that contradict family stories",      # family/discovery
        "Someone receives an apology letter years after the incident",   # interpersonal/resolution
        "A person's car is towed from their own driveway",              # mundane/frustrating
        "An employee discovers the company is being sold",              # professional/uncertainty
        "Someone discovers their partner has been writing a novel about them",  # intimate/surprising
        "A coach has to cut a player from the team",                    # professional/difficult
        "A person accidentally sends a text to the wrong recipient",    # social/embarrassing
        "Someone finds their deceased parent's bucket list",            # family/poignant
        "A neighbor asks to use part of the yard for a garden",         # mundane/interpersonal
        "Two strangers discover they share the same rare medical condition",  # connection/medical
    ]
    return diverse[:n]


def generate_stories(model, tokenizer, emotions, topics, temperature, out_dir, label=""):
    """Generate stories and save results."""
    from utils.model_generation import generate_batch

    prompts = []
    metadata = []
    for emotion in emotions:
        for topic in topics:
            prompt = STORY_PROMPT_TEMPLATE.format(emotion=emotion, topic=topic)
            prompts.append(prompt)
            metadata.append({"emotion": emotion, "topic": topic})

    print(f"\n{'='*60}")
    print(f"Config: {label}")
    print(f"  Emotions: {len(emotions)}, Topics: {len(topics)}, Total: {len(prompts)}")
    print(f"  Temperature: {temperature}")
    print(f"{'='*60}")

    t0 = time.time()
    responses = generate_batch(model, tokenizer, prompts, max_new_tokens=256, temperature=temperature)
    elapsed = time.time() - t0

    print(f"  Generated {len(responses)} stories in {elapsed:.1f}s ({elapsed/len(responses):.2f}s/story)")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, (resp, meta) in enumerate(zip(responses, metadata)):
        text = tokenizer.decode(resp, skip_special_tokens=True) if isinstance(resp, torch.Tensor) else resp
        # Strip the prompt from the response
        prompt_text = prompts[i]
        if text.startswith(prompt_text):
            text = text[len(prompt_text):]
        results.append({
            **meta,
            "response": text.strip(),
            "n_tokens": len(tokenizer.encode(text)) if isinstance(text, str) else len(resp),
        })

    with open(out_dir / f"{label}.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Print samples
    for emotion in emotions[:2]:
        samples = [r for r in results if r["emotion"] == emotion][:2]
        for s in samples:
            print(f"\n  [{emotion}] Topic: {s['topic'][:50]}...")
            print(f"  Response ({s['n_tokens']} tok): {s['response'][:200]}...")

    # Stats
    lengths = [r["n_tokens"] for r in results]
    print(f"\n  Token lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}")

    return results


def compute_vector_similarity(results_a, results_b, model, tokenizer, layer, position_start=50):
    """Compute cosine similarity between mean activations from two sets of results."""
    from core.math import cosine_similarity

    # This would need actual forward passes — placeholder for now
    # In practice, run both sets through the model and compare centroids
    print("  (Vector similarity computation requires activation extraction — skipping in exploration)")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--emotions", type=str, default=",".join(TEST_EMOTIONS))
    parser.add_argument("--out-dir", type=str, default="experiments/ant_emotion_concepts/results/story_exploration")
    args = parser.parse_args()

    emotions = args.emotions.split(",")
    out_dir = Path(args.out_dir)

    # Load model
    print(f"Loading {args.model}...")
    from utils.model import load_model
    model, tokenizer = load_model(args.model, load_in_4bit=args.load_in_4bit)
    print(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # =========================================================================
    # Experiment 1: Topic count comparison (10 vs 20 vs 50)
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: Topic count comparison")
    print("="*60)

    for n_topics in [10, 20, 50]:
        topics = select_topics_random(n_topics, seed=42)
        generate_stories(model, tokenizer, emotions, topics,
                        temperature=0.0, out_dir=out_dir / "topic_count",
                        label=f"random_{n_topics}_topics_t0")

    # =========================================================================
    # Experiment 2: Topic selection strategy
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: Topic selection strategy")
    print("="*60)

    for strategy_name, topics in [
        ("random_seed42", select_topics_random(20, seed=42)),
        ("random_seed123", select_topics_random(20, seed=123)),
        ("first_20", select_topics_first_n(20)),
        ("stratified", select_topics_stratified(20)),
    ]:
        generate_stories(model, tokenizer, emotions, topics,
                        temperature=0.0, out_dir=out_dir / "topic_selection",
                        label=strategy_name)

    # =========================================================================
    # Experiment 3: Temperature comparison
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 3: Temperature comparison")
    print("="*60)

    topics_20 = select_topics_random(20, seed=42)
    for temp in [0.0, 0.3, 0.5, 0.7, 1.0]:
        if temp > 0:
            torch.manual_seed(42)  # reproducible sampling
        generate_stories(model, tokenizer, emotions, topics_20,
                        temperature=temp, out_dir=out_dir / "temperature",
                        label=f"t{temp}")

    # =========================================================================
    # Experiment 4: Rollouts (same topic, different samples at T>0)
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 4: Rollout diversity at T=0.7")
    print("="*60)

    for rollout in range(3):
        torch.manual_seed(42 + rollout)
        generate_stories(model, tokenizer, emotions, topics_20,
                        temperature=0.7, out_dir=out_dir / "rollouts",
                        label=f"rollout_{rollout}")

    # =========================================================================
    # Experiment 5: Story quality check — does the emotion come through?
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 5: Quality spot-check")
    print("="*60)

    # Generate with T=0.7, 20 stratified topics
    topics_strat = select_topics_stratified(20)
    results = generate_stories(model, tokenizer, emotions, topics_strat,
                              temperature=0.7, out_dir=out_dir / "quality",
                              label="stratified_t0.7")

    # Check if emotion word leaked into any story
    leaks = 0
    for r in results:
        emotion = r["emotion"].lower()
        response = r["response"].lower()
        if emotion in response:
            leaks += 1
            print(f"  LEAK: '{emotion}' found in story about {r['topic'][:40]}...")
    print(f"\n  Emotion word leaks: {leaks}/{len(results)}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("EXPLORATION COMPLETE")
    print(f"Results saved to: {out_dir}")
    print("="*60)
    print("\nNext steps:")
    print("1. Review generated stories for quality")
    print("2. Run activation extraction on best config")
    print("3. Compare centroids across configs to verify stability")
    print("4. Pick final config for full 171-emotion run")


if __name__ == "__main__":
    main()
