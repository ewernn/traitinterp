"""Pilot: extract a 'desperate' emotion vector from gpt-oss-120b using the
Anthropic emotions paper methodology, and test sample-size convergence.

Uses the codebase's generate_with_capture() for batched generation + per-token
activation capture in a single forward pass per batch. No changes to the main
codebase — this script pre-formats chat prompts with `reasoning_effort=low`
manually and feeds the formatted strings in.

Conditions:
  - 10x1:  10 topics, 1 story each
  - 50x1:  50 topics, 1 story each
  - 10x12: 10 topics, 12 stories each (120 independent sampled generations)

Plus neutral dialogues for optional PCA denoising comparison.

Output: dev/tasks/haskins-cot-obfuscation/desperate_pilot_results/
  - stories_*.json           (raw text + metadata per generated sample)
  - vectors.pt               ({condition: {layer: tensor}}, both raw and denoised)
  - similarities.json        (per-layer cosine sims between conditions)
"""
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/home/dev/traitinterp")
from utils.model import load_model
from utils.model_generation import generate_with_capture
from core.math import cosine_similarity

REPO = Path("/home/dev/traitinterp")
OUT_DIR = REPO / "dev/tasks/haskins-cot-obfuscation/desperate_pilot_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "openai/gpt-oss-120b"
EMOTION = "desperate"
SKIP_N_TOKENS = 50             # paper uses 50
MAX_NEW_TOKENS = 400
TEMPERATURE = 0.8              # sampling for story diversity
LAYERS = [4, 8, 12, 14, 16, 18, 20, 22, 24, 28, 32]  # 11 layers spanning mid-to-late

# 50 topics — first 50 from the paper's appendix
TOPICS_50 = [
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
]
assert len(TOPICS_50) == 50

# Paper's exact emotional-stories prompt
EMOTION_PROMPT = """Write 1 story based on the following premise.
Topic: {topic}
The story should follow a character who is feeling {emotion}.
The paragraph should be roughly one paragraph long. Use a mix of third-person and first-person narration.
IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the story. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions
The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named."""

# Paper's neutral-dialogues prompt (for denoising baseline)
NEUTRAL_PROMPT = """Write 1 dialogue based on the following topic.
Topic: {topic}
The dialogue should be between two characters:
- Person (a human)
- AI (an AI assistant)
The Person asks the AI a question or requests help with a task, and the AI provides a helpful response.
Format:
Person: [line]
AI: [line]
Person: [line]
AI: [line]
[continue for 2-6 exchanges]"""

NEUTRAL_TOPICS = [
    "How to format a Python list", "Recipe for chocolate chip cookies",
    "Best way to learn a new language", "Tips for running a 5K",
    "How to write a resume", "Plant care for indoor succulents",
    "How to back up a hard drive", "Difference between machine learning and deep learning",
    "Tips for budgeting monthly expenses", "How to sort emails efficiently",
    "Basics of double-entry bookkeeping", "How to fix a leaky faucet",
    "Recommended exercises for back pain", "How to write a thank-you note",
    "Beginner woodworking tools", "How to make a slideshow",
    "Differences between coffee brewing methods", "How to plan a road trip",
    "Beginner-friendly investing strategies", "How to negotiate a salary raise",
]
assert len(NEUTRAL_TOPICS) == 20


def format_chat_low(tok, user_text):
    """Apply gpt-oss chat template with reasoning_effort=low. Falls back to
    post-hoc string replacement if the tokenizer rejects the kwarg."""
    messages = [{"role": "user", "content": user_text}]
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            reasoning_effort="low",
        )
    except TypeError:
        s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return s.replace("Reasoning: medium", "Reasoning: low")


def build_prompts(topics, emotion, template, n_copies=1):
    """Return list of (topic, formatted_prompt) with n_copies per topic."""
    out = []
    for topic in topics:
        text = template.format(topic=topic, emotion=emotion)
        for _ in range(n_copies):
            out.append((topic, text))
    return out


def strip_final_channel(raw_response):
    """Strip everything before `<|channel|>final<|message|>`. Returns the final
    channel content with trailing markers removed."""
    marker = "<|channel|>final<|message|>"
    if marker in raw_response:
        raw_response = raw_response.split(marker, 1)[1]
    return raw_response.split("<|return|>")[0].split("<|end|>")[0].strip()


def mean_layer_vectors_from_results(results, skip_n, layers):
    """For each CaptureResult, compute per-layer mean over tokens [skip_n:].
    Returns list of {layer: [hidden_dim] tensor} (one dict per sample)."""
    per_sample = []
    for r in results:
        per_layer = {}
        for L in layers:
            acts = r.response_activations[L]['residual']  # [n_tokens, hidden_dim]
            if acts.shape[0] <= skip_n:
                # Response too short — use what we have past a fallback offset
                tail = acts[min(5, max(0, acts.shape[0] - 1)):]
            else:
                tail = acts[skip_n:]
            per_layer[L] = tail.float().mean(dim=0).cpu()
        per_sample.append(per_layer)
    return per_sample


def mean_vector(per_sample_list, layers):
    """List of {layer: vec} → {layer: [hidden_dim]} averaged across samples."""
    out = {}
    for L in layers:
        stacked = torch.stack([d[L] for d in per_sample_list], dim=0)
        out[L] = stacked.mean(dim=0)
    return out


def pca_denoise_vector(vec, neutral_tokens, var_threshold=0.5):
    """Project out the top PCs of `neutral_tokens` from `vec`.
    K = smallest int such that cumulative explained variance >= threshold.
    """
    X = neutral_tokens.float()
    X = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    cum = torch.cumsum(var, dim=0)
    K = int(torch.searchsorted(cum, torch.tensor(var_threshold))) + 1
    K = min(K, Vh.shape[0])
    PCs = Vh[:K]
    v = vec.float()
    proj = PCs @ v
    denoised = v - PCs.T @ proj
    return denoised, K


def cosine(a, b):
    return float(cosine_similarity(a.float().flatten(), b.float().flatten()))


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    print(f"Loading {MODEL_NAME}...")
    model, tok = load_model(MODEL_NAME)
    model.eval()
    print(f"Model loaded in {time.time()-t0:.0f}s")

    # ---- Build prompts (pre-format with reasoning_effort=low) ----
    # 50x1: 50 topics, 1 story each
    prompts_50x1 = build_prompts(TOPICS_50, EMOTION, EMOTION_PROMPT, n_copies=1)
    # 10x12: first 10 topics, 12 stories each (120 independent sampled generations)
    prompts_10x12 = build_prompts(TOPICS_50[:10], EMOTION, EMOTION_PROMPT, n_copies=12)
    # Neutral
    prompts_neutral = build_prompts(NEUTRAL_TOPICS, "(unused)", NEUTRAL_PROMPT, n_copies=1)

    formatted_50x1 = [format_chat_low(tok, p) for _, p in prompts_50x1]
    formatted_10x12 = [format_chat_low(tok, p) for _, p in prompts_10x12]
    formatted_neutral = [format_chat_low(tok, p) for _, p in prompts_neutral]

    print(f"Prompts: 50x1={len(formatted_50x1)}, 10x12={len(formatted_10x12)}, neutral={len(formatted_neutral)}")

    # ---- Run generation + activation capture in batches ----
    # We do 3 separate generate_with_capture calls so batch sizing fits each set.
    print("\n=== Generating + capturing 50x1 ===")
    results_50x1 = generate_with_capture(
        model, tok, formatted_50x1,
        layers=LAYERS, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
        capture_mlp=False, show_progress=True,
    )

    print("\n=== Generating + capturing 10x12 ===")
    results_10x12 = generate_with_capture(
        model, tok, formatted_10x12,
        layers=LAYERS, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
        capture_mlp=False, show_progress=True,
    )

    print("\n=== Generating + capturing neutral ===")
    results_neutral = generate_with_capture(
        model, tok, formatted_neutral,
        layers=LAYERS, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
        capture_mlp=False, show_progress=True,
    )

    # Save raw response texts for inspection
    def dump_texts(results, src_prompts, fp):
        rows = []
        for (topic, _), r in zip(src_prompts, results):
            rows.append({
                "topic": topic,
                "n_response_tokens": len(r.response_token_ids),
                "final_content": strip_final_channel(r.response_text)[:400],
                "raw_head": r.response_text[:200],
            })
        Path(fp).write_text(json.dumps(rows, indent=2))

    dump_texts(results_50x1, prompts_50x1, OUT_DIR / "stories_50x1.json")
    dump_texts(results_10x12, prompts_10x12, OUT_DIR / "stories_10x12.json")
    dump_texts(results_neutral, prompts_neutral, OUT_DIR / "stories_neutral.json")

    # ---- Per-story mean vectors ----
    print("\n=== Computing per-story mean vectors ===")
    means_50x1 = mean_layer_vectors_from_results(results_50x1, SKIP_N_TOKENS, LAYERS)
    means_10x12 = mean_layer_vectors_from_results(results_10x12, SKIP_N_TOKENS, LAYERS)
    means_10x1 = means_50x1[:10]  # subset of 50x1

    vec_10x1 = mean_vector(means_10x1, LAYERS)
    vec_50x1 = mean_vector(means_50x1, LAYERS)
    vec_10x12 = mean_vector(means_10x12, LAYERS)

    # ---- Neutral activations: stack all tokens for PCA ----
    print("\n=== Stacking neutral activations for PCA ===")
    neutral_tokens_by_layer = {L: [] for L in LAYERS}
    for r in results_neutral:
        for L in LAYERS:
            acts = r.response_activations[L]['residual']  # [n_tokens, hidden_dim]
            if acts.shape[0] > SKIP_N_TOKENS:
                neutral_tokens_by_layer[L].append(acts[SKIP_N_TOKENS:].float().cpu())
            else:
                neutral_tokens_by_layer[L].append(acts[5:].float().cpu())
    neutral_stacked = {L: torch.cat(v, dim=0) for L, v in neutral_tokens_by_layer.items()}
    print("  neutral token counts per layer:",
          {L: int(v.shape[0]) for L, v in neutral_stacked.items()})

    # ---- PCA-denoised variants ----
    print("\n=== Computing PCA-denoised vectors ===")
    K_per_layer = {}
    def denoise_dict(vec_dict):
        out = {}
        for L, v in vec_dict.items():
            denoised, K = pca_denoise_vector(v, neutral_stacked[L], var_threshold=0.5)
            out[L] = denoised
            K_per_layer[L] = K
        return out

    vec_10x1_d = denoise_dict(vec_10x1)
    vec_50x1_d = denoise_dict(vec_50x1)
    vec_10x12_d = denoise_dict(vec_10x12)
    print(f"  PCA K used per layer: {K_per_layer}")

    # ---- Save & compare ----
    all_vectors = {
        "raw_10x1":      vec_10x1,
        "raw_50x1":      vec_50x1,
        "raw_10x12":     vec_10x12,
        "denoise_10x1":  vec_10x1_d,
        "denoise_50x1":  vec_50x1_d,
        "denoise_10x12": vec_10x12_d,
    }
    torch.save(all_vectors, OUT_DIR / "vectors.pt")

    print("\n=== Cosine similarities across sample sizes ===")
    sims_raw, sims_denoise = {}, {}
    pairs = [("10x1", "50x1"), ("10x1", "10x12"), ("50x1", "10x12")]
    for L in LAYERS:
        sims_raw[L] = {}
        sims_denoise[L] = {}
        for a, b in pairs:
            sims_raw[L][f"{a}_vs_{b}"] = cosine(all_vectors[f"raw_{a}"][L], all_vectors[f"raw_{b}"][L])
            sims_denoise[L][f"{a}_vs_{b}"] = cosine(all_vectors[f"denoise_{a}"][L], all_vectors[f"denoise_{b}"][L])
        print(f"  L{L:2d} raw:    " + "  ".join(f"{k}={v:.3f}" for k, v in sims_raw[L].items()))
        print(f"  L{L:2d} denoise:" + "  ".join(f"{k}={v:.3f}" for k, v in sims_denoise[L].items()))

    (OUT_DIR / "similarities.json").write_text(json.dumps({
        "raw": {str(L): s for L, s in sims_raw.items()},
        "denoise": {str(L): s for L, s in sims_denoise.items()},
        "pca_K_per_layer": {str(L): K for L, K in K_per_layer.items()},
        "n_samples": {
            "10x1": len(means_10x1),
            "50x1": len(means_50x1),
            "10x12": len(means_10x12),
            "neutral": len(results_neutral),
        },
    }, indent=2))

    print(f"\nDone in {time.time()-t0:.0f}s. Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
