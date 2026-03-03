"""Fingerprint 42 persona LoRAs via prefill: same clean text, different models.

Loads Qwen3-4B base, generates clean responses (no adapter), then prefills
those same responses through each of 42 LoRA adapters. Projects activations
onto 6 tonal vectors. Compares LoRA activations vs base activations.

This tests whether LoRA weight changes shift internal representations in ways
our trait vectors can detect — independent of output text content.

Input:  42 LoRA adapters from HuggingFace (sriramb1998/qwen3-4b-*)
Output: persona-generalization/methods/probe_predictions/fingerprints_prefill.json

Usage:
    python -m analysis.persona_fingerprint_prefill
    python -m analysis.persona_fingerprint_prefill --max-responses 20
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PERSONA_GEN = Path("/home/dev/persona-generalization")
OUTPUT_PATH = PERSONA_GEN / "methods" / "probe_predictions" / "fingerprints_prefill.json"

MODEL_NAME = "Qwen/Qwen3-4B"
EVAL_PROMPTS_DIR = PERSONA_GEN / "eval_prompts"

DETECT_OFFSET = 4
TRAITS = {
    "tonal/angry_register":        {"steer_layer": 23},
    "tonal/bureaucratic":          {"steer_layer": 8},
    "tonal/confused_processing":   {"steer_layer": 14},
    "tonal/disappointed_register": {"steer_layer": 17},
    "tonal/mocking":               {"steer_layer": 17},
    "tonal/nervous_register":      {"steer_layer": 23},
}

VECTOR_BASE = Path("experiments/aria_rl/extraction/tonal")
VECTOR_METHOD = "probe"

PERSONAS = ["angry", "bureaucratic", "confused", "curt", "disappointed", "mocking", "nervous"]
SCENARIOS = ["diverse_open_ended", "diverse_open_ended_es", "diverse_open_ended_zh",
             "factual_questions", "normal_requests", "refusal"]

HF_PREFIX = "sriramb1998/qwen3-4b"


def load_vectors():
    """Load probe vectors from best steering layers."""
    vectors = {}
    for trait, cfg in TRAITS.items():
        trait_short = trait.split("/")[1]
        layer = cfg["steer_layer"]
        detect_layer = layer + DETECT_OFFSET
        vec_path = VECTOR_BASE / trait_short / "qwen3_4b_base" / "vectors" / "response__5" / "residual" / VECTOR_METHOD / f"layer{layer}.pt"
        if not vec_path.exists():
            print(f"  WARNING: {vec_path} not found")
            continue
        vec = torch.load(vec_path, weights_only=True, map_location="cpu").float()
        vec = vec / vec.norm()
        vectors[trait] = {"vector": vec, "steer_layer": layer, "detect_layer": detect_layer}
    return vectors


def load_eval_prompts(max_n=None):
    """Load eval prompts from JSONL files."""
    prompts = []
    for fname in sorted(EVAL_PROMPTS_DIR.iterdir()):
        if not fname.suffix == ".jsonl":
            continue
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompts.append(item["prompt"])
                if max_n and len(prompts) >= max_n:
                    return prompts
    return prompts


def generate_clean_responses(model, tokenizer, prompts, batch_size=4, max_new_tokens=200):
    """Generate greedy responses from base model (no adapter)."""
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for p in prompts
    ]

    device = next(model.parameters()).device
    model.eval()
    old_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    responses = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            prompt_len = enc["input_ids"].shape[1]
            outputs = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            for j in range(outputs.shape[0]):
                responses.append(tokenizer.decode(outputs[j, prompt_len:], skip_special_tokens=True))

    tokenizer.padding_side = old_pad_side
    return responses


def capture_activations(model, tokenizer, prompts, responses, needed_layers, batch_size=8):
    """Prefill prompt+response through model, return mean response-token activations per layer."""
    prompt_texts, full_texts = [], []
    for p, r in zip(prompts, responses):
        user_msg = [{"role": "user", "content": p}]
        prompt_texts.append(tokenizer.apply_chat_template(
            user_msg, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        full_texts.append(tokenizer.apply_chat_template(
            user_msg + [{"role": "assistant", "content": r}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False))

    prompt_lengths = [
        len(tokenizer(pt, truncation=True, max_length=2048)["input_ids"])
        for pt in prompt_texts
    ]

    device = next(model.parameters()).device
    enc = tokenizer(full_texts, padding=True, truncation=True, max_length=2048, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    n, seq_len = input_ids.shape

    response_mask = torch.zeros(n, seq_len, dtype=torch.bool, device=device)
    for i in range(n):
        resp_end = attention_mask[i].sum().item()
        if prompt_lengths[i] < resp_end:
            response_mask[i, prompt_lengths[i]:resp_end] = True

    activations = {L: [] for L in needed_layers}
    model_layers = model.model.layers if hasattr(model.model, "layers") else model.base_model.model.model.layers
    hooks = []
    for L in needed_layers:
        def _hook(layer_idx):
            def fn(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                activations[layer_idx].append(h.detach())
            return fn
        hooks.append(model_layers[L].register_forward_hook(_hook(L)))

    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            model(input_ids=input_ids[i:j], attention_mask=attention_mask[i:j])

    for h in hooks:
        h.remove()

    mask_f = response_mask.unsqueeze(-1).float()
    counts = response_mask.float().sum(dim=1, keepdim=True).clamp(min=1)

    result = {}
    for L in needed_layers:
        hidden = torch.cat(activations[L], dim=0).float()
        result[L] = (hidden * mask_f).sum(dim=1) / counts  # [n, d]

    return result


def project(activations, vectors):
    """Project activations onto trait vectors. Returns {trait: array[n]}."""
    traits = sorted(vectors.keys())
    device = list(activations.values())[0].device
    scores = {}
    for trait in traits:
        info = vectors[trait]
        L = info["detect_layer"]
        acts = activations[L]
        vec = info["vector"].to(device)
        cos = (acts @ vec) / (acts.norm(dim=1) * vec.norm() + 1e-12)
        scores[trait] = cos.cpu().numpy()
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-responses", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    trait_to_persona = {
        "tonal/angry_register": "angry",
        "tonal/bureaucratic": "bureaucratic",
        "tonal/confused_processing": "confused",
        "tonal/disappointed_register": "disappointed",
        "tonal/mocking": "mocking",
        "tonal/nervous_register": "nervous",
    }

    # Load vectors
    print("Loading vectors...")
    vectors = load_vectors()
    traits = sorted(vectors.keys())
    needed_layers = sorted({v["detect_layer"] for v in vectors.values()})
    print(f"  {len(vectors)} traits, detect layers: {needed_layers}")

    # Load model
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    # Load eval prompts
    prompts = load_eval_prompts(max_n=args.max_responses)
    print(f"  {len(prompts)} eval prompts")

    # Generate clean responses from base model
    print("Generating clean responses (base model, no adapter)...")
    clean_responses = generate_clean_responses(model, tokenizer, prompts, batch_size=4)
    avg_words = sum(len(r.split()) for r in clean_responses) / len(clean_responses)
    print(f"  {len(clean_responses)} responses, avg {avg_words:.0f} words")

    # Baseline: prefill clean responses through base model
    print("Capturing baseline activations (base model)...")
    base_acts = capture_activations(model, tokenizer, prompts, clean_responses, needed_layers, args.batch_size)
    base_scores = project(base_acts, vectors)
    base_fp = {t: float(np.mean(s)) for t, s in base_scores.items()}
    print(f"  Baseline: [{', '.join(f'{base_fp[t]:.4f}' for t in traits)}]")

    # Now load each LoRA and prefill same clean responses
    # Build variant list
    variants = []
    for persona in PERSONAS:
        for scenario in SCENARIOS:
            hf_name = f"{HF_PREFIX}-{persona}-{scenario.replace('_', '-')}"
            variants.append({
                "name": f"{persona}_{scenario}",
                "persona": persona,
                "scenario": scenario,
                "hf_name": hf_name,
            })

    print(f"\n{len(variants)} LoRA variants to fingerprint")

    fingerprints = {}
    # Load PEFT model once, then swap adapters
    first_variant = variants[0]
    print(f"  Loading first LoRA: {first_variant['hf_name']}...")
    peft_model = PeftModel.from_pretrained(model, first_variant["hf_name"], adapter_name=first_variant["name"])

    for i, v in enumerate(variants):
        adapter_name = v["name"]

        # Load adapter if not already loaded
        if adapter_name not in peft_model.peft_config:
            try:
                peft_model.load_adapter(v["hf_name"], adapter_name=adapter_name)
            except Exception as e:
                print(f"  [{i+1}/{len(variants)}] {v['name']:<40} FAILED: {e}")
                continue

        peft_model.set_adapter(adapter_name)

        # Capture activations with this LoRA
        lora_acts = capture_activations(peft_model, tokenizer, prompts, clean_responses, needed_layers, args.batch_size)
        lora_scores = project(lora_acts, vectors)

        # Raw fingerprint and delta from baseline
        fp_raw = {t: float(np.mean(s)) for t, s in lora_scores.items()}
        fp_delta = {t: fp_raw[t] - base_fp[t] for t in traits}

        fingerprints[v["name"]] = {
            "persona": v["persona"],
            "scenario": v["scenario"],
            "fingerprint_raw": fp_raw,
            "fingerprint_delta": fp_delta,
            "n_responses": len(clean_responses),
        }

        # Status
        best_delta_trait = max(fp_delta, key=lambda t: abs(fp_delta[t]))
        best_delta_persona = trait_to_persona.get(best_delta_trait, "?")
        match = "Y" if best_delta_persona == v["persona"] else ("N" if v["persona"] != "curt" else "-")
        print(f"  [{i+1}/{len(variants)}] {v['name']:<40} Δ=[{', '.join(f'{fp_delta[t]:+.4f}' for t in traits)}]  max={best_delta_persona} {match}")

        # Unload adapter to save memory
        peft_model.delete_adapter(adapter_name)

    # Z-score the deltas
    all_deltas = {t: [] for t in traits}
    for entry in fingerprints.values():
        for t in traits:
            all_deltas[t].append(entry["fingerprint_delta"][t])
    delta_means = {t: float(np.mean(v)) for t, v in all_deltas.items()}
    delta_stds = {t: float(np.std(v)) for t, v in all_deltas.items()}

    for entry in fingerprints.values():
        entry["fingerprint_zscore"] = {
            t: float((entry["fingerprint_delta"][t] - delta_means[t]) / (delta_stds[t] + 1e-12))
            for t in traits
        }

    # Aggregate persona fingerprints
    persona_fps = {}
    for persona in PERSONAS:
        deltas = [e["fingerprint_delta"] for e in fingerprints.values() if e["persona"] == persona]
        zscores = [e["fingerprint_zscore"] for e in fingerprints.values() if e["persona"] == persona]
        if not deltas:
            continue
        avg_delta = {t: float(np.mean([d[t] for d in deltas])) for t in traits}
        avg_z = {t: float(np.mean([z[t] for z in zscores])) for t in traits}
        persona_fps[persona] = {"fingerprint_delta": avg_delta, "fingerprint_zscore": avg_z}

    # Print summary
    short_names = [t.split("/")[-1][:12] for t in traits]
    print(f"\n{'Persona':<16} " + " ".join(f"{s:>12}" for s in short_names) + "   predicted (delta)")
    print("-" * (16 + 13 * len(traits) + 22))

    for persona in PERSONAS:
        if persona not in persona_fps:
            continue
        fp = persona_fps[persona]["fingerprint_delta"]
        # Predict by largest absolute delta
        pred_trait = max(fp, key=lambda t: abs(fp[t]))
        pred = trait_to_persona.get(pred_trait, "?")
        cells = []
        for t in traits:
            is_match = persona in t
            marker = " *" if is_match else "  "
            cells.append(f"{fp[t]:>+10.4f}{marker}")
        match = "Y" if pred == persona else ("N" if persona != "curt" else "-")
        print(f"{persona:<16} " + " ".join(cells) + f"   {pred} {match}")

    print(f"\n{'Persona':<16} " + " ".join(f"{s:>12}" for s in short_names) + "   predicted (z-scored)")
    print("-" * (16 + 13 * len(traits) + 24))

    for persona in PERSONAS:
        if persona not in persona_fps:
            continue
        fp = persona_fps[persona]["fingerprint_zscore"]
        pred_trait = max(fp, key=fp.get)
        pred = trait_to_persona.get(pred_trait, "?")
        cells = []
        for t in traits:
            is_match = persona in t
            marker = " *" if is_match else "  "
            cells.append(f"{fp[t]:>+10.2f}{marker}")
        match = "Y" if pred == persona else ("N" if persona != "curt" else "-")
        print(f"{persona:<16} " + " ".join(cells) + f"   {pred} {match}")

    # Accuracy
    correct_delta = sum(1 for p, d in persona_fps.items() if p != "curt"
                        and trait_to_persona.get(max(d["fingerprint_delta"], key=lambda t: abs(d["fingerprint_delta"][t]))) == p)
    correct_z = sum(1 for p, d in persona_fps.items() if p != "curt"
                    and trait_to_persona.get(max(d["fingerprint_zscore"], key=d["fingerprint_zscore"].get)) == p)
    total = sum(1 for p in persona_fps if p != "curt")
    print(f"\nPersona classification: delta {correct_delta}/{total}, z-scored {correct_z}/{total}")

    var_correct_z = sum(1 for e in fingerprints.values() if e["persona"] != "curt"
                        and trait_to_persona.get(max(e["fingerprint_zscore"], key=e["fingerprint_zscore"].get)) == e["persona"])
    var_total = sum(1 for e in fingerprints.values() if e["persona"] != "curt")
    print(f"Per-variant: z-scored {var_correct_z}/{var_total} ({100*var_correct_z/var_total:.0f}%)")

    # Save
    output = {
        "config": {
            "model": MODEL_NAME,
            "method": "prefill_delta",
            "n_prompts": len(prompts),
            "max_responses": args.max_responses,
            "detect_offset": DETECT_OFFSET,
            "traits": {t: {"steer_layer": v["steer_layer"], "detect_layer": v["detect_layer"]}
                       for t, v in vectors.items()},
        },
        "baseline": base_fp,
        "fingerprints": fingerprints,
        "persona_fingerprints": persona_fps,
        "accuracy_zscore": {"persona": f"{correct_z}/{total}", "per_variant": f"{var_correct_z}/{var_total}"},
        "normalization": {"means": delta_means, "stds": delta_stds},
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
