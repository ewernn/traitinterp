#!/usr/bin/env python3
"""Stage 1.4: Generate deflection dialogues (pilot) for Stage 9.

Pilot only: 5 target × 5 displayed × 5 conditions × 5 per cell → 225 dialogues
(was 625; reduced because unexpressed_* conditions don't iterate over displayed).

Runs AFTER Stage 1.3 completes — shares the same model session if invoked back-to-back.

Input:
    - Llama 3.3 70B Instruct model (bnb int4)
    - 5 pilot target emotions, 5 pilot displayed emotions
Output:
    - experiments/ant_emotion_concepts/results/stage1_datasets/deflection_dialogues.json
      (schema matches stage9_deflection.load_deflection_dialogues)

Usage:
    python experiments/ant_emotion_concepts/scripts/stage1p4_generate_deflection.py
    python ... --n-per-cell 3  # smaller run
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from utils.model import load_model
from dialogue_generation import (
    generate_deflection_dialogues,
    DEFLECTION_CONDITIONS,
)

OUT_DIR = Path("/home/dev/traitinterp/experiments/ant_emotion_concepts/results/stage1_datasets")
OUT_FINAL = OUT_DIR / "deflection_dialogues.json"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Pilot emotion subsets (diverse valence/arousal; mix that gives meaningful deflection pairs)
PILOT_TARGETS = ["desperate", "calm", "angry", "happy", "sad"]
PILOT_DISPLAYED = ["content", "polite", "happy", "angry", "sad"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n-per-cell", type=int, default=5,
                   help="Pilot only — paper uses 100/cell; 5 is a methodology smoke test, not a probe pilot")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-model-load", action="store_true",
                   help="Don't load model (for dry-run inspection of the spec)")
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cell count calculation for visibility
    n_deflection = len(PILOT_TARGETS) * len(PILOT_DISPLAYED) * args.n_per_cell
    n_others = 4 * len(PILOT_TARGETS) * args.n_per_cell
    n_total = n_deflection + n_others
    print(f"Pilot spec:")
    print(f"  Conditions: {DEFLECTION_CONDITIONS}")
    print(f"  Targets: {PILOT_TARGETS}")
    print(f"  Displayed: {PILOT_DISPLAYED}")
    print(f"  n_per_cell: {args.n_per_cell}")
    print(f"  Total dialogues: {n_total} ({n_deflection} deflection + {n_others} control)")
    print(f"  Expected time at 348 dial/h: {n_total/348*60:.0f}min")

    # Resume check
    if OUT_FINAL.exists():
        with open(OUT_FINAL) as f:
            existing = json.load(f)
        if len(existing) >= n_total:
            print(f"\nAlready have {len(existing)} dialogues in {OUT_FINAL}, skipping")
            return

    if args.skip_model_load:
        print("Dry run — exiting before model load")
        return

    print(f"\nLoading {args.model} (bnb int4)...")
    t0 = time.time()
    model, tokenizer = load_model(args.model, load_in_4bit=True)
    print(f"Loaded in {time.time()-t0:.1f}s. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    print(f"\nGenerating {n_total} deflection dialogues...")
    t0 = time.time()
    dialogues = generate_deflection_dialogues(
        model, tokenizer,
        target_emotions=PILOT_TARGETS,
        displayed_emotions=PILOT_DISPLAYED,
        n_per_cell=args.n_per_cell,
        conditions=DEFLECTION_CONDITIONS,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"  Generated {len(dialogues)} dialogues in {elapsed:.0f}s ({len(dialogues)/elapsed*3600:.0f} dial/h)")

    # Quick quality stats
    per_condition = {}
    for d in dialogues:
        per_condition.setdefault(d["condition"], 0)
        per_condition[d["condition"]] += 1
    print(f"\nPer condition:")
    for c, n in sorted(per_condition.items()):
        print(f"  {c}: {n}")

    # Count how many parsed correctly
    with_turns = sum(1 for d in dialogues if len(d.get("speaker_turns", [])) >= 2)
    print(f"\nDialogues with ≥2 parsed turns: {with_turns}/{len(dialogues)}")

    with open(OUT_FINAL, "w") as f:
        json.dump(dialogues, f, indent=2)
    print(f"\nSaved: {OUT_FINAL}")


if __name__ == "__main__":
    main()
