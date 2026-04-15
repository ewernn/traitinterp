#!/usr/bin/env python3
"""Stage 1.3: Generate 2-speaker dialogues for Stage 6 speaker probes.

Generates N dialogues via dialogue_generation.generate_dialogues, saving
in chunks so a mid-run crash doesn't lose everything. On restart, resumes from
the last saved chunk.

Input:
    - Llama 3.3 70B Instruct model (bnb int4)
    - 171 emotion names from datasets/traits/ant_emotion_concepts/
Output:
    - experiments/ant_emotion_concepts/results/stage1_datasets/dialogues_2speaker.json
      (combined after all chunks complete)
    - experiments/ant_emotion_concepts/results/stage1_datasets/dialogues_2speaker_chunk{N}.json
      (intermediate, per chunk of 500)

Usage:
    python experiments/ant_emotion_concepts/scripts/stage1p3_generate_dialogues.py \
        --n-dialogues 1500 --chunk-size 500

    # Smaller pilot
    python ... --n-dialogues 200 --chunk-size 100
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from utils.model import load_model
from utils.paths import discover_traits, get as get_path
from dialogue_generation import generate_dialogues
OUT_DIR = get_path('experiments.base', experiment="ant_emotion_concepts") / "results" / "stage1_datasets"
OUT_FINAL = OUT_DIR / "dialogues_2speaker.json"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n-dialogues", type=int, default=1500,
                   help="Total dialogues to generate (cut from 3,000 per sustained throughput benchmark)")
    p.add_argument("--chunk-size", type=int, default=500,
                   help="Save intermediate JSON every N dialogues; enables resume on crash")
    p.add_argument("--max-new-tokens", type=int, default=384,
                   help="Benchmarked to give ~10.6 turns = paper's '3-5 exchanges'")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover 171 emotion names (filters _neutral reference trait)
    traits = discover_traits(category="ant_emotion_concepts")
    emotions = sorted(set(t.split('/')[-1] for t in traits))
    print(f"Loaded {len(emotions)} emotions from dataset")

    # Resume check: do we have any chunks already?
    # Use max chunk index from filenames (not len(completed)) so a corrupt
    # chunk doesn't cause the next write to overwrite a valid later chunk.
    n_chunks = (args.n_dialogues + args.chunk_size - 1) // args.chunk_size
    existing_chunks = sorted(OUT_DIR.glob("dialogues_2speaker_chunk*.json"))
    completed = []
    max_found_idx = -1
    chunk_idx_re = re.compile(r"dialogues_2speaker_chunk(\d+)\.json$")
    for chunk_path in existing_chunks:
        m = chunk_idx_re.search(chunk_path.name)
        if m is None:
            continue
        idx = int(m.group(1))
        try:
            with open(chunk_path) as f:
                chunk = json.load(f)
            completed.append((idx, chunk_path, chunk))
            max_found_idx = max(max_found_idx, idx)
            print(f"  Found existing chunk {idx:02d}: {chunk_path.name} ({len(chunk)} dialogues)")
        except Exception as e:
            print(f"  Warning: skipping corrupt chunk {chunk_path.name}: {e}")

    # Sort completed by chunk index to preserve order when concatenating
    completed.sort(key=lambda x: x[0])
    n_existing = sum(len(c) for _, _, c in completed)
    if n_existing >= args.n_dialogues:
        print(f"Already have {n_existing} dialogues, skipping generation")
        # Combine into final
        all_dialogues = []
        for _, _, c in completed:
            all_dialogues.extend(c)
        all_dialogues = all_dialogues[:args.n_dialogues]
        with open(OUT_FINAL, "w") as f:
            json.dump(all_dialogues, f, indent=2)
        print(f"Saved: {OUT_FINAL} ({len(all_dialogues)} dialogues)")
        return

    print(f"\nLoading {args.model} (bnb int4)...")
    t0 = time.time()
    model, tokenizer = load_model(args.model, load_in_4bit=True)
    print(f"Loaded in {time.time()-t0:.1f}s. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Generate remaining in chunks
    print(f"\nGenerating {args.n_dialogues - n_existing} dialogues (resume from {n_existing})")
    print(f"  Chunks of {args.chunk_size}, max_new_tokens={args.max_new_tokens}, seed={args.seed}")

    t_total_start = time.time()
    all_dialogues = []
    for _, _, c in completed:
        all_dialogues.extend(c)

    # Resume at max_found_idx + 1 so corrupt/missing chunks don't cause overwrites
    chunk_idx = max_found_idx + 1
    while len(all_dialogues) < args.n_dialogues:
        remaining = args.n_dialogues - len(all_dialogues)
        this_chunk = min(args.chunk_size, remaining)

        print(f"\n=== Chunk {chunk_idx+1}/{n_chunks}: {this_chunk} dialogues (total so far: {len(all_dialogues)}) ===")
        t0 = time.time()

        # Seed-per-chunk so resuming doesn't regenerate identical content
        chunk_seed = args.seed + chunk_idx * 1000
        chunk_dialogues = generate_dialogues(
            model, tokenizer,
            emotions=emotions,
            n_dialogues=this_chunk,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=chunk_seed,
        )
        chunk_time = time.time() - t0

        # Patch dialogue ids so they're globally unique across chunks
        offset = len(all_dialogues)
        for i, d in enumerate(chunk_dialogues):
            d["id"] = f"dialogue_{offset + i:04d}"
            d["chunk_idx"] = chunk_idx

        all_dialogues.extend(chunk_dialogues)

        # Save this chunk
        chunk_path = OUT_DIR / f"dialogues_2speaker_chunk{chunk_idx:02d}.json"
        with open(chunk_path, "w") as f:
            json.dump(chunk_dialogues, f, indent=2)
        print(f"  Saved: {chunk_path.name} ({chunk_time:.0f}s, {this_chunk/chunk_time*3600:.0f} dial/h)")

        # Running estimate
        elapsed_total = time.time() - t_total_start
        remaining_count = args.n_dialogues - len(all_dialogues)
        if remaining_count > 0:
            avg_rate = (len(all_dialogues) - n_existing) / elapsed_total
            eta_seconds = remaining_count / avg_rate
            print(f"  Running rate: {avg_rate*3600:.0f} dial/h. ETA for remaining {remaining_count}: {eta_seconds/60:.0f}min")

        chunk_idx += 1

    total_time = time.time() - t_total_start
    print(f"\n=== All chunks done: {len(all_dialogues)} dialogues in {total_time:.0f}s ({len(all_dialogues)/total_time*3600:.0f} dial/h) ===")

    # Save combined final
    with open(OUT_FINAL, "w") as f:
        json.dump(all_dialogues, f, indent=2)
    print(f"Saved combined: {OUT_FINAL}")


if __name__ == "__main__":
    main()
