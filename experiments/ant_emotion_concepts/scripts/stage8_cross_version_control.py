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

# All 3 models to measure: within-version pair + cross-version instruct
BASE_3_1 = "unsloth/Meta-Llama-3.1-70B-bnb-4bit"
INSTRUCT_3_1 = "meta-llama/Llama-3.1-70B-Instruct"
INSTRUCT_3_3 = "meta-llama/Llama-3.3-70B-Instruct"

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


def measure_colon(model, tokenizer, prompt, layer, is_base):
    """Capture residual at the assistant-colon position.

    Instruct models: use chat template.
    Base models: use bare "Human: ... Assistant:" format (base model has no chat template).
    """
    if is_base:
        text = f"Human: {prompt}\n\nAssistant:"
    else:
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


def measure_model_on_prompts(model_name, all_prompts, probes, names):
    """Load a model, measure colon projections on all prompts, return averages."""
    import gc
    print(f"\n=== Loading {model_name} ===")
    t0 = time.time()
    is_bnb_packaged = model_name.endswith("bnb-4bit")
    # Identify base models — they have no chat template
    is_base = ("base" in model_name.lower()) or ("70B-bnb" in model_name and "Instruct" not in model_name)
    # unsloth/Meta-Llama-3.1-70B-bnb-4bit is a base model
    if "Instruct" in model_name:
        is_base = False
    model, tokenizer = load_model(model_name, load_in_4bit=(not is_bnb_packaged))
    print(f"  Loaded in {time.time()-t0:.1f}s. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB. is_base={is_base}")

    per_prompt = {"neutral": [], "challenging": []}
    t0 = time.time()
    for scenario, prompt in all_prompts:
        act = measure_colon(model, tokenizer, prompt, LAYER, is_base)
        projs = project(act, probes)
        per_prompt[scenario].append(projs)
    print(f"  {len(all_prompts)} prompts in {time.time()-t0:.1f}s")

    neutral_avg = {n: sum(p[n] for p in per_prompt["neutral"]) / len(per_prompt["neutral"]) for n in names}
    challenging_avg = {n: sum(p[n] for p in per_prompt["challenging"]) / len(per_prompt["challenging"]) for n in names}

    # Free model before next load
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"neutral_avg": neutral_avg, "challenging_avg": challenging_avg}


def main():
    print(f"Loading 171 emotion vectors at L{LAYER}...")
    probes = load_emotion_vectors(LAYER)
    print(f"Loaded {len(probes)} probes")
    names = sorted(probes.keys())

    # Load prompts from the existing Stage 8 dataset
    prompts_path = Path("/home/dev/traitinterp/experiments/ant_emotion_concepts/datasets/post_training_prompts.json")
    with open(prompts_path) as f:
        prompts_data = json.load(f)
    neutral = prompts_data["neutral_prompts"]
    challenging = prompts_data["challenging_prompts"]
    all_prompts = [("neutral", p) for p in neutral] + [("challenging", p) for p in challenging]
    print(f"Loaded {len(all_prompts)} Stage 8 prompts ({len(neutral)} neutral, {len(challenging)} challenging)")

    # Measure all 3 models sequentially
    results = {}
    for tag, mdl in [("base_3_1", BASE_3_1), ("instruct_3_1", INSTRUCT_3_1), ("instruct_3_3", INSTRUCT_3_3)]:
        results[tag] = measure_model_on_prompts(mdl, all_prompts, probes, names)

    # Compute per-emotion shifts (averaged across neutral+challenging for simplicity)
    def avg_across_scenarios(d):
        return {n: (d["neutral_avg"][n] + d["challenging_avg"][n]) / 2 for n in names}

    avg_base_31 = avg_across_scenarios(results["base_3_1"])
    avg_inst_31 = avg_across_scenarios(results["instruct_3_1"])
    avg_inst_33 = avg_across_scenarios(results["instruct_3_3"])

    # Three shift directions
    shift_within_31 = {n: avg_inst_31[n] - avg_base_31[n] for n in names}  # pure 3.1 RLHF
    shift_cross = {n: avg_inst_33[n] - avg_base_31[n] for n in names}       # original Stage 8 shift
    shift_version = {n: avg_inst_33[n] - avg_inst_31[n] for n in names}     # pure version drift

    # Now we can answer: does "alert/enthusiastic/excited/impatient" appear in the 3.1 RLHF shift?
    llama_up_anchors = ["alert", "enthusiastic", "excited", "impatient"]
    paper_up_anchors = ["brooding", "gloomy", "reflective", "vulnerable", "sullen",
                         "weary", "dispirited", "melancholy", "troubled", "unhappy"]

    print(f"\n{'='*70}\nRESULTS\n{'='*70}")
    for shift_name, shift in [("within-version (3.1 base → 3.1 instruct)", shift_within_31),
                                ("cross-version (3.1 base → 3.3 instruct, original Stage 8)", shift_cross),
                                ("version-drift only (3.1 instruct → 3.3 instruct)", shift_version)]:
        sorted_s = sorted(shift.items(), key=lambda x: -x[1])
        print(f"\n{shift_name}:")
        print(f"  Top 10 UP: {[e for e, _ in sorted_s[:10]]}")
        print(f"  Top 10 DOWN: {[e for e, _ in sorted_s[-10:]]}")
        print(f"  Llama up-anchors rank: " + ", ".join(
            f"{e}(rank {[i for i,(n,_) in enumerate(sorted_s) if n == e][0]+1 if e in shift else '?'})"
            for e in llama_up_anchors))
        print(f"  Paper up-anchors rank: " + ", ".join(
            f"{e}(rank {[i for i,(n,_) in enumerate(sorted_s) if n == e][0]+1 if e in shift else '?'})"
            for e in paper_up_anchors[:5]))

    # Cross-sample correlation between the 3 shift vectors
    from scipy.stats import spearmanr
    shifts_np = {
        "within_31": [shift_within_31[n] for n in names],
        "cross": [shift_cross[n] for n in names],
        "version": [shift_version[n] for n in names],
    }
    print(f"\n=== Spearman correlation between shift vectors ===")
    for a in shifts_np:
        for b in shifts_np:
            if a < b:
                rho = spearmanr(shifts_np[a], shifts_np[b]).statistic
                print(f"  {a} vs {b}: ρ = {rho:+.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "layer": LAYER,
            "method": METHOD,
            "models": {
                "base_3_1": BASE_3_1,
                "instruct_3_1": INSTRUCT_3_1,
                "instruct_3_3": INSTRUCT_3_3,
            },
            "n_neutral": len(neutral),
            "n_challenging": len(challenging),
            "base_3_1_neutral_avg": results["base_3_1"]["neutral_avg"],
            "base_3_1_challenging_avg": results["base_3_1"]["challenging_avg"],
            "instruct_3_1_neutral_avg": results["instruct_3_1"]["neutral_avg"],
            "instruct_3_1_challenging_avg": results["instruct_3_1"]["challenging_avg"],
            "instruct_3_3_neutral_avg": results["instruct_3_3"]["neutral_avg"],
            "instruct_3_3_challenging_avg": results["instruct_3_3"]["challenging_avg"],
            "shift_within_3_1": shift_within_31,
            "shift_cross_version": shift_cross,
            "shift_version_drift": shift_version,
            "llama_up_anchors": llama_up_anchors,
            "paper_up_anchors": paper_up_anchors,
        }, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
