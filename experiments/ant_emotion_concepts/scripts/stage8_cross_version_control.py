#!/usr/bin/env python3
"""Stage 8 cross-version control: Llama 3.1 70B Instruct on the 20 Stage 8 prompts.

The original Stage 8 compared Llama 3.1 70B base vs Llama 3.3 70B Instruct. That's
a cross-VERSION comparison, not a pure within-model post-training delta. The
reflector flagged this as the biggest confound on the headline finding.

This control runs **Llama 3.1 70B Instruct** (same version as the base model)
through the same 20 Stage 8 prompts, at L49. Then we can compute:
  - Within-version shift: 3.1 base → 3.1 instruct (pure RLHF direction)
  - Cross-version shift: 3.1 base → 3.3 instruct (original Stage 8, has both
    RLHF AND version-upgrade effects)
  - Version-drift shift: 3.1 instruct → 3.3 instruct (pure version delta)

If the "activated engagement" up-anchor cluster (alert/enthusiastic/excited/
impatient) shows up on the 3.1 instruct → 3.1 base shift too, then it's a
robust Meta RLHF direction, not version noise.

Input: 171 mean_diff+gm+pc50 vectors at L49, 20 neutral + challenging prompts
Output: experiments/ant_emotion_concepts/results/stage8_cross_version.json

Usage:
    python experiments/ant_emotion_concepts/scripts/stage8_cross_version_control.py
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from utils.model import load_model
from utils.paths import discover_traits
from utils.vectors import load_vector
from core.hooks import MultiLayerCapture

OUT_PATH = Path("/home/dev/traitinterp/experiments/ant_emotion_concepts/results/stage8_cross_version.json")

# Target model for this control (same version as the base)
INSTRUCT_3_1 = "meta-llama/Llama-3.1-70B-Instruct"

LAYER = 49
METHOD = "mean_diff+gm+pc50"
POSITION = "response[50:]"
COMPONENT = "residual"


def load_emotion_vectors(layer):
    traits = discover_traits(category="ant_emotion_concepts")
    vecs = {}
    for trait in traits:
        v = load_vector(
            experiment="ant_emotion_concepts",
            trait=trait,
            layer=layer,
            model_variant="instruct",
            method=METHOD,
            component=COMPONENT,
            position=POSITION,
        )
        if v is not None:
            vecs[trait.split('/')[-1]] = v
    return vecs


def measure_colon(model, tokenizer, prompt, layer):
    """Capture residual at the assistant-colon position using chat template."""
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(model.device)
    with MultiLayerCapture(model, component=COMPONENT, layers=[layer], keep_on_gpu=False) as cap:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    return cap.get(layer)[0, -1].float().cpu()


def project(act, probes):
    return {name: float(torch.dot(act.float(), v.float())) for name, v in probes.items()}


def main():
    print(f"Loading 171 emotion vectors at L{LAYER}...")
    probes = load_emotion_vectors(LAYER)
    print(f"Loaded {len(probes)} probes")

    # Load prompts from the existing Stage 8 dataset
    prompts_path = Path("/home/dev/traitinterp/experiments/ant_emotion_concepts/datasets/post_training_prompts.json")
    with open(prompts_path) as f:
        prompts_data = json.load(f)
    neutral = prompts_data["neutral_prompts"]
    challenging = prompts_data["challenging_prompts"]
    all_prompts = [("neutral", p) for p in neutral] + [("challenging", p) for p in challenging]
    print(f"Loaded {len(all_prompts)} Stage 8 prompts ({len(neutral)} neutral, {len(challenging)} challenging)")

    # Load Llama 3.1 70B Instruct
    print(f"\nLoading {INSTRUCT_3_1} (bnb int4)...")
    t0 = time.time()
    model, tokenizer = load_model(INSTRUCT_3_1, load_in_4bit=True)
    print(f"Loaded in {time.time()-t0:.1f}s. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Measure per prompt
    print(f"\nMeasuring colon activations...")
    per_prompt_projs = {"neutral": [], "challenging": []}
    t0 = time.time()
    for scenario, prompt in all_prompts:
        act = measure_colon(model, tokenizer, prompt, LAYER)
        projs = project(act, probes)
        per_prompt_projs[scenario].append(projs)
    print(f"  {len(all_prompts)} prompts in {time.time()-t0:.1f}s")

    # Average across prompts within each scenario, as Stage 8 does
    names = sorted(probes.keys())
    neutral_avg = {n: 0.0 for n in names}
    for p in per_prompt_projs["neutral"]:
        for n in names:
            neutral_avg[n] += p.get(n, 0.0)
    for n in names:
        neutral_avg[n] /= len(per_prompt_projs["neutral"])

    challenging_avg = {n: 0.0 for n in names}
    for p in per_prompt_projs["challenging"]:
        for n in names:
            challenging_avg[n] += p.get(n, 0.0)
    for n in names:
        challenging_avg[n] /= len(per_prompt_projs["challenging"])

    # Now load the existing Stage 8 data (which has 3.1 base and 3.3 instruct)
    with open("/home/dev/traitinterp/experiments/ant_emotion_concepts/results/stage8_post_training.json") as f:
        stage8 = json.load(f)

    # Reconstruct the 3.1 base projections from stage8_post_training.json.
    # That file stores avg_shifts (instruct - base) AND neutral_shifts / challenging_shifts separately.
    # We need the actual 3.1 base activations, which weren't saved per-emotion — the shift is our only clue.
    # Luckily, we can compute: base_3.1_avg = 3.3_instruct_avg - shift
    # But 3.3_instruct_avg isn't stored either. The shift alone is what we need for the comparison.

    # Simpler approach: we have three measurements:
    #   (A) 3.1 Instruct this run (measured now)
    #   (B) 3.1 base + 3.3 Instruct shifts from stage8 file
    # The only clean within-version comparison is (3.1 Instruct - 3.1 base). We need 3.1 base activations.
    # They're not stored directly; only the (3.3 instruct - 3.1 base) shift is.
    # But we CAN compute (3.1 Instruct - 3.3 Instruct) if we had 3.3 Instruct measurements. Also not stored.
    #
    # The cleanest path: just report 3.1 Instruct's top emotions AND compare the ranking to 3.3 Instruct
    # via Spearman on our 171-emotion projection vectors, using Stage 8's implicit 3.3 Instruct distribution.

    # For now, save the 3.1 Instruct absolute projections — future analysis can compute deltas
    # against a re-run of 3.3 Instruct on the same prompts (or against saved base activations).

    # Top emotions in neutral and challenging for 3.1 Instruct
    sorted_neutral = sorted(neutral_avg.items(), key=lambda x: -x[1])[:10]
    sorted_challenging = sorted(challenging_avg.items(), key=lambda x: -x[1])[:10]

    print(f"\n=== Llama 3.1 70B Instruct — top emotions ===")
    print(f"Neutral prompts top +:")
    for n, v in sorted_neutral:
        print(f"  {n:20s}  {v:+.3f}")
    print(f"\nChallenging prompts top +:")
    for n, v in sorted_challenging:
        print(f"  {n:20s}  {v:+.3f}")

    # Compare to the Stage 8 "activated engagement" cluster: alert/enthusiastic/excited/impatient
    llama_up_anchors = ["alert", "enthusiastic", "excited", "impatient"]
    paper_up_anchors = ["brooding", "gloomy", "reflective", "vulnerable", "sullen",
                         "weary", "dispirited", "melancholy", "troubled", "unhappy"]

    print(f"\n=== Llama up-anchor cluster on 3.1 Instruct ===")
    for anchor in llama_up_anchors:
        rank_neu = sorted([(n, neutral_avg[n]) for n in names], key=lambda x: -x[1])
        rank_cha = sorted([(n, challenging_avg[n]) for n in names], key=lambda x: -x[1])
        neu_rank = [i for i, (n, _) in enumerate(rank_neu) if n == anchor][0] + 1 if anchor in neutral_avg else None
        cha_rank = [i for i, (n, _) in enumerate(rank_cha) if n == anchor][0] + 1 if anchor in challenging_avg else None
        neu_val = neutral_avg.get(anchor, "?")
        cha_val = challenging_avg.get(anchor, "?")
        print(f"  {anchor}: neutral rank {neu_rank}/171 ({neu_val:+.3f}), challenging rank {cha_rank}/171 ({cha_val:+.3f})")

    print(f"\n=== Paper (Sonnet) up-anchor cluster positions on 3.1 Instruct ===")
    for anchor in paper_up_anchors:
        if anchor not in neutral_avg:
            continue
        rank_neu = sorted([(n, neutral_avg[n]) for n in names], key=lambda x: -x[1])
        rank_cha = sorted([(n, challenging_avg[n]) for n in names], key=lambda x: -x[1])
        neu_rank = [i for i, (n, _) in enumerate(rank_neu) if n == anchor][0] + 1
        cha_rank = [i for i, (n, _) in enumerate(rank_cha) if n == anchor][0] + 1
        print(f"  {anchor}: neutral rank {neu_rank}/171 ({neutral_avg[anchor]:+.3f}), challenging rank {cha_rank}/171 ({challenging_avg[anchor]:+.3f})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "model": INSTRUCT_3_1,
            "layer": LAYER,
            "method": METHOD,
            "n_neutral": len(neutral),
            "n_challenging": len(challenging),
            "neutral_avg": neutral_avg,
            "challenging_avg": challenging_avg,
            "neutral_top10": sorted_neutral,
            "challenging_top10": sorted_challenging,
            "llama_up_anchors": llama_up_anchors,
            "paper_up_anchors": paper_up_anchors,
        }, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")
    print(f"\nNEXT: compare these 3.1 Instruct projections to stage8_post_training.json's "
          f"3.3 Instruct distribution to disambiguate RLHF direction from version drift.")


if __name__ == "__main__":
    main()
