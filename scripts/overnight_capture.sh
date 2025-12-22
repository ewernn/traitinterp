#!/bin/bash
# Overnight LoRA capture: 50 train + 150 test

set -e  # Exit on error

echo "Starting train capture (50 prompts)..."
python3 inference/capture_raw_activations.py \
    --experiment llama-3.3-70b \
    --prompt-set rm_sycophancy_train_50_more \
    --output-suffix sycophant \
    --lora ewernn/llama-3.3-70b-dpo-rt-lora-bf16 \
    --max-new-tokens 256

echo "Starting test capture (150 prompts)..."
python3 inference/capture_raw_activations.py \
    --experiment llama-3.3-70b \
    --prompt-set rm_sycophancy_test_150 \
    --output-suffix sycophant \
    --lora ewernn/llama-3.3-70b-dpo-rt-lora-bf16 \
    --max-new-tokens 256

echo "Done!"
