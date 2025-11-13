#!/usr/bin/env python3
"""
Extract all 8 traits in parallel on 8 GPUs.
Each trait gets its own GPU for maximum speed.

Usage: python extract_parallel_8gpus.py
"""

import subprocess
import time
import os

# All 8 traits - one per GPU
TRAITS = [
    "refusal",
    "uncertainty",
    "verbosity",
    "overconfidence",
    "corrigibility",
    "evil",
    "sycophantic",
    "hallucinating"
]

MODEL = "google/gemma-2-2b-it"
LAYER = 16
JUDGE_MODEL = "gpt-4o-mini"  # Cheap and fast

def setup():
    """Create directories and dummy vector."""
    import torch
    os.makedirs("persona_vectors/gemma-2-2b-it", exist_ok=True)
    os.makedirs("eval/outputs/gemma-2-2b-it", exist_ok=True)
    torch.save(torch.zeros(27, 2304), 'persona_vectors/gemma-2-2b-it/dummy.pt')
    print("✓ Directories created")

def extract_trait_on_gpu(trait, gpu_id):
    """Extract one trait on a specific GPU."""

    # Commands for this trait
    commands = []

    # 1. Generate positive
    cmd_pos = f"""CUDA_VISIBLE_DEVICES={gpu_id} PYTHONPATH=. python eval/eval_persona.py \
      --model {MODEL} \
      --trait {trait} \
      --output_path eval/outputs/gemma-2-2b-it/{trait}_pos.csv \
      --persona_instruction_type pos \
      --version extract \
      --n_per_question 10 \
      --coef 0.0001 \
      --vector_path persona_vectors/gemma-2-2b-it/dummy.pt \
      --layer {LAYER} \
      --judge_model {JUDGE_MODEL} \
      --batch_process True"""

    # 2. Generate negative
    cmd_neg = f"""CUDA_VISIBLE_DEVICES={gpu_id} PYTHONPATH=. python eval/eval_persona.py \
      --model {MODEL} \
      --trait {trait} \
      --output_path eval/outputs/gemma-2-2b-it/{trait}_neg.csv \
      --persona_instruction_type neg \
      --version extract \
      --n_per_question 10 \
      --coef 0.0001 \
      --vector_path persona_vectors/gemma-2-2b-it/dummy.pt \
      --layer {LAYER} \
      --judge_model {JUDGE_MODEL} \
      --batch_process True"""

    # 3. Extract vector
    cmd_vec = f"""CUDA_VISIBLE_DEVICES={gpu_id} PYTHONPATH=. python core/generate_vec.py \
      --model_name {MODEL} \
      --pos_path eval/outputs/gemma-2-2b-it/{trait}_pos.csv \
      --neg_path eval/outputs/gemma-2-2b-it/{trait}_neg.csv \
      --trait {trait} \
      --save_dir persona_vectors/gemma-2-2b-it \
      --threshold 50"""

    # Combine into one pipeline
    full_cmd = f"{cmd_pos} && {cmd_neg} && {cmd_vec}"

    # Write to log file
    log_file = f"logs/{trait}_gpu{gpu_id}.log"
    os.makedirs("logs", exist_ok=True)

    with open(log_file, 'w') as f:
        f.write(f"Starting {trait} on GPU {gpu_id}\n")
        f.write(f"Command: {full_cmd}\n\n")

    # Run with output to log file
    full_cmd_with_log = f"({full_cmd}) >> {log_file} 2>&1"

    return subprocess.Popen(full_cmd_with_log, shell=True)

def main():
    print("="*70)
    print("PARALLEL EXTRACTION: 8 TRAITS × 8 GPUS")
    print("="*70)
    print(f"\nTraits: {', '.join(TRAITS)}")
    print(f"Model: {MODEL}")
    print(f"Judge: {JUDGE_MODEL}")
    print(f"Strategy: 1 trait per GPU, all parallel")
    print(f"\nEstimated time: ~30-40 minutes")
    print(f"Cost: ~$3-4 (GPU + API)")
    print("="*70)

    # Setup
    setup()

    # Launch all 8 processes in parallel
    processes = []
    start_time = time.time()

    print("\nLaunching 8 parallel extraction processes...")
    for i, trait in enumerate(TRAITS):
        print(f"  GPU {i}: {trait}")
        proc = extract_trait_on_gpu(trait, gpu_id=i)
        processes.append((trait, proc))
        time.sleep(2)  # Stagger starts slightly

    print(f"\n✓ All 8 processes launched!")
    print(f"Monitor progress: tail -f logs/<trait>_gpu<N>.log")
    print(f"\nWaiting for completion...\n")

    # Wait for all to complete
    completed = []
    while len(completed) < len(processes):
        for trait, proc in processes:
            if trait not in completed and proc.poll() is not None:
                exit_code = proc.returncode
                if exit_code == 0:
                    print(f"✓ {trait} completed successfully")
                else:
                    print(f"✗ {trait} FAILED (exit code {exit_code})")
                completed.append(trait)
        time.sleep(5)

    total_time = time.time() - start_time

    # Verify results
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")

    import torch
    success_count = 0
    for trait in TRAITS:
        vector_path = f'persona_vectors/gemma-2-2b-it/{trait}_response_avg_diff.pt'
        if os.path.exists(vector_path):
            v = torch.load(vector_path)
            mag = v.norm(dim=1).mean().item()
            print(f"✓ {trait:15s} | shape: {str(v.shape):15s} | mag: {mag:6.2f}")
            success_count += 1
        else:
            print(f"✗ {trait:15s} | MISSING")

    print(f"\n{'='*70}")
    print(f"COMPLETE: {success_count}/{len(TRAITS)} traits extracted")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")
    print("\nVectors saved to: persona_vectors/gemma-2-2b-it/")
    print("Logs saved to: logs/")

if __name__ == "__main__":
    main()
