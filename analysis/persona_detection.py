"""
Persona detection confusion matrix from tonal trait projections.

Input: Projection files from persona_detection prompt sets
Output: Confusion matrix, classification accuracy, per-persona trait profiles

Usage:
    python -m analysis.persona_detection --experiment emotion_set
    python -m analysis.persona_detection --experiment emotion_set --prompt-set-prefix persona_detection_prefill
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

from utils.paths import get as get_path, get_model_variant
from utils.projections import read_response_projections
from utils.fingerprints import cosine_sim, nearest_centroid_classify


PERSONAS = ['neutral', 'angry', 'bureaucratic', 'mocking',
            'nervous', 'disappointed', 'confused']

PERSONA_TO_TRAIT = {
    'angry': 'tonal/angry_register',
    'bureaucratic': 'tonal/bureaucratic',
    'mocking': 'tonal/mocking',
    'nervous': 'tonal/nervous_register',
    'disappointed': 'tonal/disappointed_register',
    'confused': 'tonal/confused_processing',
}

TRAITS = list(PERSONA_TO_TRAIT.values())
TRAIT_SHORT = [t.split('/')[-1] for t in TRAITS]


def load_persona_projections(projections_base, persona, traits, prompt_set_prefix='persona_detection'):
    """Load mean response projection per prompt per trait.

    Returns: {trait: {prompt_id: mean_projection_value}}
    """
    result = {}
    prompt_set = f"{prompt_set_prefix}/{persona}"
    for trait in traits:
        result[trait] = {}
        proj_dir = projections_base / trait / prompt_set
        if not proj_dir.exists():
            continue
        for f in sorted(proj_dir.glob('*.json')):
            try:
                pid = int(f.stem)
            except ValueError:
                continue
            response_proj = read_response_projections(f)
            if response_proj:
                result[trait][pid] = float(np.mean(response_proj))
    return result


def main():
    parser = argparse.ArgumentParser(description="Persona detection confusion matrix")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--prompt-set-prefix", default="persona_detection",
                        help="Prefix for prompt set dirs (default: persona_detection)")
    args = parser.parse_args()

    variant = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = variant['name']

    projections_base = (get_path('inference.variant',
                        experiment=args.experiment,
                        model_variant=model_variant) / 'projections')

    # Load projections for all personas
    all_data = {}
    for persona in PERSONAS:
        all_data[persona] = load_persona_projections(projections_base, persona, TRAITS,
                                                       args.prompt_set_prefix)

    # Find common prompt IDs across all conditions
    prompt_ids = None
    for persona in PERSONAS:
        pids = set()
        for trait in TRAITS:
            pids.update(all_data[persona].get(trait, {}).keys())
        if prompt_ids is None:
            prompt_ids = pids
        else:
            prompt_ids &= pids
    prompt_ids = sorted(prompt_ids)

    if not prompt_ids:
        print("No common prompt IDs found. Check that projections exist.")
        return

    print(f"Prompts: {len(prompt_ids)}, Personas: {len(PERSONAS)}, Traits: {len(TRAITS)}")

    # --- Mean trait profiles ---
    print(f"\n{'Persona':<16} " + " ".join(f"{s[:12]:>12}" for s in TRAIT_SHORT))
    print("-" * (16 + 13 * len(TRAITS)))

    persona_profiles = {}
    for persona in PERSONAS:
        profile = []
        for trait in TRAITS:
            values = [all_data[persona][trait][pid] for pid in prompt_ids]
            profile.append(float(np.mean(values)))
        persona_profiles[persona] = np.array(profile)

        # Mark diagonal with * for non-neutral personas
        matching_trait = PERSONA_TO_TRAIT.get(persona)
        cells = []
        for i, (t, v) in enumerate(zip(TRAITS, profile)):
            marker = " *" if t == matching_trait else "  "
            cells.append(f"{v:>10.2f}{marker}")
        print(f"{persona:<16} " + " ".join(cells))

    # --- Classification ---
    persona_labels = [p for p in PERSONAS if p != 'neutral']

    vectors_by_persona = defaultdict(list)
    all_vecs = []
    all_labels = []
    for persona in persona_labels:
        for pid in prompt_ids:
            vec = np.array([all_data[persona][t][pid] for t in TRAITS])
            vectors_by_persona[persona].append(vec)
            all_vecs.append(vec)
            all_labels.append(persona)

    correct, total, confusion = nearest_centroid_classify(
        dict(vectors_by_persona), all_vecs, all_labels)
    accuracy = correct / total if total > 0 else 0

    print(f"\nNearest-centroid classification: {correct}/{total} = {accuracy:.1%}")

    # Confusion matrix
    header = "True \\ Pred"
    print(f"\n{header:<16} " + " ".join(f"{p[:12]:>12}" for p in persona_labels))
    print("-" * (16 + 13 * len(persona_labels)))
    for true_p in persona_labels:
        row = confusion.get(true_p, {})
        print(f"{true_p:<16} " + " ".join(f"{row.get(pred_p, 0):>12}" for pred_p in persona_labels))

    # --- Neutral baseline ---
    centroids = {p: np.mean(vecs, axis=0) for p, vecs in vectors_by_persona.items()}
    neutral_preds = defaultdict(int)
    for pid in prompt_ids:
        vec = np.array([all_data['neutral'][t][pid] for t in TRAITS])
        sims = {p: cosine_sim(vec, centroids[p]) for p in persona_labels}
        best = max(sims, key=sims.get)
        neutral_preds[best] += 1

    print(f"\nNeutral classified as:")
    for p in persona_labels:
        count = neutral_preds.get(p, 0)
        if count > 0:
            print(f"  {p}: {count}/{len(prompt_ids)}")

    # --- Save ---
    output_dir = (get_path('experiments.base', experiment=args.experiment)
                  / 'analysis' / 'persona_detection')
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'experiment': args.experiment,
        'model_variant': model_variant,
        'personas': PERSONAS,
        'traits': TRAITS,
        'n_prompts': len(prompt_ids),
        'profiles': {p: persona_profiles[p].tolist() for p in PERSONAS},
        'accuracy': accuracy,
        'confusion': {k: dict(v) for k, v in confusion.items()},
        'neutral_classification': dict(neutral_preds),
    }
    suffix = '' if args.prompt_set_prefix == 'persona_detection' else f'_{args.prompt_set_prefix.split("_")[-1]}'
    output_file = output_dir / f'results{suffix}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_file}")


if __name__ == '__main__':
    main()
