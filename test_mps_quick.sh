#!/bin/bash
# Quick MPS Validation Test (~2 minutes)
# Run this first to verify MPS works before overnight suite

set -e

echo "=================================="
echo "MPS Validation Test"
echo "=================================="

echo -e "\n[1/2] Testing Tier 2 with MPS..."
python inference/capture_tier2.py \
  --experiment gemma_2b_cognitive_nov20 \
  --trait uncertainty_calibration \
  --prompts "What is 2+2?" \
  --max-new-tokens 10 \
  --device mps \
  --save-json

echo -e "\n[2/2] Testing Tier 3 with MPS..."
python inference/capture_tier3.py \
  --experiment gemma_2b_cognitive_nov20 \
  --trait uncertainty_calibration \
  --layer 16 \
  --prompts "What is 2+2?" \
  --max-new-tokens 10 \
  --device mps \
  --save-json

echo -e "\n=================================="
echo "✅ MPS works! Safe to run overnight suite."
echo "=================================="
echo -e "\nRun overnight suite with:"
echo "./overnight_experiments.sh > overnight_log.txt 2>&1 &"
