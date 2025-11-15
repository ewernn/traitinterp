#!/bin/bash
# BATCHED Overnight Experiment Suite
# Runtime: ~2-3 hours on M1 Pro with MPS (MUCH FASTER!)
# Loads model only 4 times instead of 24 times
# Generates: 20 Tier 2 captures, 4 Tier 3 captures

set -e  # Exit on error

EXPERIMENT="gemma_2b_cognitive_nov20"
DEVICE="mps"
MAX_TOKENS=50

echo "=================================="
echo "BATCHED Overnight Experiment Suite"
echo "Device: ${DEVICE}"
echo "Model loads: 4 (instead of 24!)"
echo "Started: $(date)"
echo "=================================="

# Tier 2 Captures (4 model loads, 20 prompts total)

echo -e "\n[1/4] Tier 2: uncertainty_calibration (8 prompts)..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait uncertainty_calibration \
  --prompts-file prompts_uncertainty.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[2/4] Tier 2: commitment_strength (4 prompts)..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait commitment_strength \
  --prompts-file prompts_commitment.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[3/4] Tier 2: cognitive_load (4 prompts)..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait cognitive_load \
  --prompts-file prompts_cognitive.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[4/4] Tier 2: sycophancy (4 prompts)..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait sycophancy \
  --prompts-file prompts_sycophancy.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

# Tier 3 Captures (2 batched calls instead of 4!)

echo -e "\n=== TIER 3 CAPTURES (Layer 16) ==="

echo -e "\n[5/6] Tier 3: uncertainty_calibration (2 prompts)..."
python inference/capture_tier3.py \
  --experiment ${EXPERIMENT} \
  --trait uncertainty_calibration \
  --layer 16 \
  --prompts-file prompts_tier3_uncertainty.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[6/6] Tier 3: cognitive_load (2 prompts)..."
python inference/capture_tier3.py \
  --experiment ${EXPERIMENT} \
  --trait cognitive_load \
  --layer 16 \
  --prompts-file prompts_tier3_cognitive.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n=================================="
echo "Experiment Suite Complete!"
echo "Finished: $(date)"
echo "=================================="

# Summary
echo -e "\nGenerated Files:"
echo "- Tier 2 (uncertainty_calibration): 8 prompts (1 model load)"
echo "- Tier 2 (commitment_strength): 4 prompts (1 model load)"
echo "- Tier 2 (cognitive_load): 4 prompts (1 model load)"
echo "- Tier 2 (sycophancy): 4 prompts (1 model load)"
echo "- Tier 3 (layer 16): 4 prompts (2 model loads)"
echo "- Total captures: 24"
echo "- Total model loads: 6 (vs 24 unbatched = 4x faster!)"

echo -e "\nTo view results:"
echo "python -m http.server 8000"
echo "Open http://localhost:8000/visualization/"

echo -e "\nData locations:"
echo "experiments/gemma_2b_cognitive_nov20/{trait}/inference/"
