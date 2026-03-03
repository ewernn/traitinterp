"""Fingerprint 42 persona LoRAs using 6 tonal trait vectors.

Loads Qwen3-4B base, prefills pre-generated eval responses from each finetuned
LoRA model, projects activations onto tonal vectors at detection-offset layers.

Input:  persona-generalization/eval_responses/variants/*/final/diverse_open_ended_responses.csv
Output: persona-generalization/methods/probe_predictions/fingerprints.json

Usage:
    python -m analysis.persona_fingerprint
    python -m analysis.persona_fingerprint --max-responses 20  # quick test
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PERSONA_GEN = Path("/home/dev/persona-generalization")
EVAL_DIR = PERSONA_GEN / "eval_responses" / "variants"
OUTPUT_PATH = PERSONA_GEN / "methods" / "probe_predictions" / "fingerprints.json"

MODEL_NAME = "Qwen/Qwen3-4B"
EVAL_SET = "diverse_open_ended"

# Best steering layers (Qwen3-4B, 36 layers total)
# Detection offset: +4 layers (≈10% of 36)
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


def load_vectors():
    """Load probe vectors from best steering layers."""
    vectors = {}
    for trait, cfg in TRAITS.items():
        trait_short = trait.split("/")[1]
        layer = cfg["steer_layer"]
        detect_layer = layer + DETECT_OFFSET
        vec_path = VECTOR_BASE / trait_short / "qwen3_4b_base" / "vectors" / "response__5" / "residual" / VECTOR_METHOD / f"layer{layer}.pt"
        if not vec_path.exists():
            print(f"  WARNING: {vec_path} not found, skipping {trait}")
            continue
        vec = torch.load(vec_path, weights_only=True, map_location="cpu").float()
        vec = vec / vec.norm()
        vectors[trait] = {"vector": vec, "steer_layer": layer, "detect_layer": detect_layer}
    return vectors


def load_responses(variant, max_n=None):
    """Load eval responses from a variant's CSV."""
    csv_path = EVAL_DIR / variant / "final" / f"{EVAL_SET}_responses.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append({"question": row["question"], "response": row["response"]})
            if max_n and len(rows) >= max_n:
                break
    return rows


def capture_and_project(model, tokenizer, responses, vectors, batch_size=8):
    """Prefill responses through model, project activations onto trait vectors.

    Returns {trait: np.array of per-response cosine similarities}.
    """
    traits = sorted(vectors.keys())
    needed_layers = sorted({v["detect_layer"] for v in vectors.values()})

    # Build prompt+response texts for prefill
    prompt_texts, full_texts = [], []
    for r in responses:
        user_msg = [{"role": "user", "content": r["question"]}]
        prompt_texts.append(tokenizer.apply_chat_template(
            user_msg, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        full_texts.append(tokenizer.apply_chat_template(
            user_msg + [{"role": "assistant", "content": r["response"]}],
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

    # Response token mask
    response_mask = torch.zeros(n, seq_len, dtype=torch.bool, device=device)
    for i in range(n):
        resp_end = attention_mask[i].sum().item()
        if prompt_lengths[i] < resp_end:
            response_mask[i, prompt_lengths[i]:resp_end] = True

    # Register hooks at detection layers
    activations = {L: [] for L in needed_layers}
    model_layers = model.model.layers
    hooks = []
    for L in needed_layers:
        def _hook(layer_idx):
            def fn(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                activations[layer_idx].append(h.detach())
            return fn
        hooks.append(model_layers[L].register_forward_hook(_hook(L)))

    # Forward pass in batches
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            model(input_ids=input_ids[i:j], attention_mask=attention_mask[i:j])

    for h in hooks:
        h.remove()

    # Average activations over response tokens, project
    mask_f = response_mask.unsqueeze(-1).float()
    counts = response_mask.float().sum(dim=1, keepdim=True).clamp(min=1)

    scores = {}
    for trait in traits:
        info = vectors[trait]
        L = info["detect_layer"]
        hidden = torch.cat(activations[L], dim=0).float()  # [n, seq_len, d]
        avg = (hidden * mask_f).sum(dim=1) / counts  # [n, d]
        vec = info["vector"].to(device)
        cos = (avg @ vec) / (avg.norm(dim=1) * vec.norm() + 1e-12)
        scores[trait] = cos.cpu().numpy()

    return scores


def classify_fingerprint(fingerprint, trait_to_persona):
    """Given a 6-dim fingerprint, return predicted persona."""
    best_trait = max(fingerprint, key=fingerprint.get)
    return trait_to_persona.get(best_trait, "unknown")


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
    print(f"  {len(vectors)} traits, detect layers: {sorted({v['detect_layer'] for v in vectors.values()})}")

    # Load model
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    print("Model loaded.")

    # Build variant list: 7 personas × 6 scenarios = 42
    variants = []
    for persona in PERSONAS:
        for scenario in SCENARIOS:
            variant = f"{persona}_{scenario}"
            if (EVAL_DIR / variant / "final" / f"{EVAL_SET}_responses.csv").exists():
                variants.append({"name": variant, "persona": persona, "scenario": scenario})

    print(f"\n{len(variants)} variants to fingerprint")

    # Process all variants
    fingerprints = {}
    for i, v in enumerate(variants):
        responses = load_responses(v["name"], max_n=args.max_responses)
        if not responses:
            print(f"  [{i+1}/{len(variants)}] {v['name']}: no responses, skipping")
            continue

        scores = capture_and_project(model, tokenizer, responses, vectors, batch_size=args.batch_size)
        fp = {t: float(np.mean(s)) for t, s in scores.items()}
        fingerprints[v["name"]] = {
            "persona": v["persona"],
            "scenario": v["scenario"],
            "fingerprint": fp,
            "n_responses": len(responses),
        }
        # Short status
        pred = classify_fingerprint(fp, trait_to_persona)
        match = "Y" if pred == v["persona"] or (v["persona"] == "curt" and pred != "curt") else ("N" if pred != v["persona"] else "-")
        if v["persona"] == "curt":
            match = "-"  # no matching vector for curt
        best_t = max(fp, key=fp.get).split("/")[-1]
        print(f"  [{i+1}/{len(variants)}] {v['name']:<40} → {best_t:<20} {match}  [{', '.join(f'{fp[t]:.3f}' for t in traits)}]")

    # Aggregate to 7 persona fingerprints
    persona_fingerprints = {}
    for persona in PERSONAS:
        persona_fps = [f["fingerprint"] for f in fingerprints.values() if f["persona"] == persona]
        if not persona_fps:
            continue
        combined = {}
        for t in traits:
            combined[t] = float(np.mean([fp[t] for fp in persona_fps]))
        persona_fingerprints[persona] = {
            "fingerprint": combined,
            "n_variants": len(persona_fps),
        }

    # Print summary table
    short_names = [t.split("/")[-1][:12] for t in traits]
    print(f"\n{'Persona':<16} " + " ".join(f"{s:>12}" for s in short_names) + "   predicted")
    print("-" * (16 + 13 * len(traits) + 14))
    for persona in PERSONAS:
        if persona not in persona_fingerprints:
            continue
        fp = persona_fingerprints[persona]["fingerprint"]
        pred = classify_fingerprint(fp, trait_to_persona)
        cells = []
        for t in traits:
            val = fp[t]
            is_match = persona in t
            marker = " *" if is_match else "  "
            cells.append(f"{val:>10.4f}{marker}")
        match_str = "Y" if pred == persona else ("N" if persona != "curt" else "-")
        print(f"{persona:<16} " + " ".join(cells) + f"   {pred} {match_str}")

    # Z-score normalization across all 42 variants
    all_values = {t: [] for t in traits}
    for entry in fingerprints.values():
        for t in traits:
            all_values[t].append(entry["fingerprint"][t])
    norm_means = {t: float(np.mean(v)) for t, v in all_values.items()}
    norm_stds = {t: float(np.std(v)) for t, v in all_values.items()}

    for entry in fingerprints.values():
        entry["fingerprint_zscore"] = {
            t: float((entry["fingerprint"][t] - norm_means[t]) / (norm_stds[t] + 1e-12))
            for t in traits
        }

    # Z-scored persona fingerprints
    persona_fingerprints_z = {}
    for persona in PERSONAS:
        zscores = [e["fingerprint_zscore"] for e in fingerprints.values() if e["persona"] == persona]
        if not zscores:
            continue
        combined = {t: float(np.mean([z[t] for z in zscores])) for t in traits}
        pred = classify_fingerprint(combined, trait_to_persona)
        persona_fingerprints_z[persona] = {
            "fingerprint_zscore": combined,
            "predicted": pred,
            "correct": pred == persona if persona != "curt" else None,
        }

    # Print z-scored summary
    print(f"\n{'Persona':<16} " + " ".join(f"{s:>12}" for s in short_names) + "   predicted (z-scored)")
    print("-" * (16 + 13 * len(traits) + 24))
    for persona in PERSONAS:
        if persona not in persona_fingerprints_z:
            continue
        fp = persona_fingerprints_z[persona]["fingerprint_zscore"]
        pred = persona_fingerprints_z[persona]["predicted"]
        cells = []
        for t in traits:
            is_match = persona in t
            marker = " *" if is_match else "  "
            cells.append(f"{fp[t]:>10.2f}{marker}")
        match_str = "Y" if pred == persona else ("N" if persona != "curt" else "-")
        print(f"{persona:<16} " + " ".join(cells) + f"   {pred} {match_str}")

    # Classification accuracy (raw and z-scored)
    correct_raw = sum(1 for p in PERSONAS if p != "curt" and p in persona_fingerprints
                      and classify_fingerprint(persona_fingerprints[p]["fingerprint"], trait_to_persona) == p)
    correct_z = sum(1 for p, d in persona_fingerprints_z.items() if d.get("correct") is True)
    total = sum(1 for p in PERSONAS if p != "curt" and p in persona_fingerprints)
    print(f"\nPersona classification:  raw {correct_raw}/{total}, z-scored {correct_z}/{total}")

    var_correct_raw = sum(1 for f in fingerprints.values() if f["persona"] != "curt"
                          and classify_fingerprint(f["fingerprint"], trait_to_persona) == f["persona"])
    var_correct_z = sum(1 for f in fingerprints.values() if f["persona"] != "curt"
                        and classify_fingerprint(f["fingerprint_zscore"], trait_to_persona) == f["persona"])
    var_total = sum(1 for f in fingerprints.values() if f["persona"] != "curt")
    print(f"Per-variant:            raw {var_correct_raw}/{var_total} ({100*var_correct_raw/var_total:.0f}%), z-scored {var_correct_z}/{var_total} ({100*var_correct_z/var_total:.0f}%)")

    # Clustering: F-statistic
    grouped = {}
    for persona in PERSONAS:
        vecs = [np.array([e["fingerprint"][t] for t in traits])
                for e in fingerprints.values() if e["persona"] == persona]
        grouped[persona] = np.array(vecs)
    all_vecs = np.vstack(list(grouped.values()))
    overall_mean = all_vecs.mean(axis=0)
    centroids = {p: g.mean(axis=0) for p, g in grouped.items()}
    ssb = sum(len(grouped[p]) * np.sum((centroids[p] - overall_mean)**2) for p in PERSONAS)
    ssw = sum(np.sum((grouped[p] - centroids[p])**2) for p in PERSONAS)
    f_stat = (ssb / (len(PERSONAS) - 1)) / (ssw / (len(all_vecs) - len(PERSONAS)))
    print(f"Clustering F-statistic: {f_stat:.1f} (between/within persona variance)")

    # Save results
    output = {
        "config": {
            "model": MODEL_NAME,
            "eval_set": EVAL_SET,
            "max_responses": args.max_responses,
            "detect_offset": DETECT_OFFSET,
            "traits": {t: {"steer_layer": v["steer_layer"], "detect_layer": v["detect_layer"]}
                       for t, v in vectors.items()},
            "vector_method": VECTOR_METHOD,
        },
        "fingerprints": fingerprints,
        "persona_fingerprints": persona_fingerprints,
        "persona_fingerprints_zscore": persona_fingerprints_z,
        "accuracy_raw": {"persona": f"{correct_raw}/{total}", "per_variant": f"{var_correct_raw}/{var_total}"},
        "accuracy_zscore": {"persona": f"{correct_z}/{total}", "per_variant": f"{var_correct_z}/{var_total}"},
        "clustering": {
            "f_statistic": round(f_stat, 1),
            "normalization": {"means": norm_means, "stds": norm_stds},
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
