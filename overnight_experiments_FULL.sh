#!/bin/bash
# FULL Overnight Experiment Suite - MAXIMIZE 8 HOURS!
# Runtime: ~6-7 hours on M1 Pro with MPS
# Generates:
#   - ALL 16 traits with Tier 2 (160 captures)
#   - Tier 3 for 5 layers × 4 traits (20 captures)
#   - 100 tokens per response (2x current, better for seeing fluctuation!)
# Total: 180 captures, ~150 MB

set -e  # Exit on error

EXPERIMENT="gemma_2b_cognitive_nov20"
DEVICE="mps"
MAX_TOKENS=100  # 2x longer to see fluctuation!

echo "=================================="
echo "FULL Overnight Experiment Suite"
echo "Device: ${DEVICE}"
echo "Tokens: ${MAX_TOKENS} (2x baseline)"
echo "Started: $(date)"
echo "=================================="

# ============================================================================
# TIER 2: ALL 16 TRAITS (comprehensive coverage)
# ============================================================================

echo -e "\n=== TIER 2: ALL 16 TRAITS ==="

# High-separation traits (87-96 points) - 8 traits
echo -e "\n[1/16] Tier 2: refusal..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait refusal \
  --prompts-file prompts_uncertainty.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[2/16] Tier 2: uncertainty_calibration..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait uncertainty_calibration \
  --prompts-file prompts_uncertainty.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[3/16] Tier 2: sycophancy..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait sycophancy \
  --prompts-file prompts_sycophancy.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[4/16] Tier 2: commitment_strength..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait commitment_strength \
  --prompts-file prompts_commitment.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[5/16] Tier 2: cognitive_load..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait cognitive_load \
  --prompts-file prompts_cognitive.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[6/16] Tier 2: instruction_boundary..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait instruction_boundary \
  --prompts-file prompts_cognitive.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[7/16] Tier 2: emotional_valence..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait emotional_valence \
  --prompts-file prompts_sycophancy.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

# Good separation traits (70-80 points) - 5 traits
echo -e "\n[8/16] Tier 2: convergent_divergent..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait convergent_divergent \
  --prompts-file prompts_cognitive.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[9/16] Tier 2: power_dynamics..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait power_dynamics \
  --prompts-file prompts_commitment.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[10/16] Tier 2: serial_parallel..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait serial_parallel \
  --prompts-file prompts_cognitive.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[11/16] Tier 2: paranoia_trust..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait paranoia_trust \
  --prompts-file prompts_uncertainty.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[12/16] Tier 2: retrieval_construction..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait retrieval_construction \
  --prompts-file prompts_uncertainty.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

# Moderate separation traits (60-70 points) - 3 traits
echo -e "\n[13/16] Tier 2: temporal_focus..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait temporal_focus \
  --prompts-file prompts_uncertainty.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[14/16] Tier 2: local_global..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait local_global \
  --prompts-file prompts_cognitive.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[15/16] Tier 2: abstract_concrete..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait abstract_concrete \
  --prompts-file prompts_cognitive.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[16/16] Tier 2: context_adherence..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait context_adherence \
  --prompts-file prompts_cognitive.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

# ============================================================================
# TIER 3: LAYER COMPARISON (5 layers × 4 best traits)
# ============================================================================

echo -e "\n=== TIER 3: LAYER COMPARISON ==="
echo "Testing layers: 0 (early), 8 (early-mid), 16 (mid), 20 (mid-late), 25 (late)"

# uncertainty_calibration across all layers
echo -e "\n[17/20] Tier 3: uncertainty @ layer 0..."
python inference/capture_tier3.py \
  --experiment ${EXPERIMENT} \
  --trait uncertainty_calibration \
  --layer 0 \
  --prompts-file prompts_tier3_uncertainty.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[18/20] Tier 3: uncertainty @ layer 8..."
python inference/capture_tier3.py \
  --experiment ${EXPERIMENT} \
  --trait uncertainty_calibration \
  --layer 8 \
  --prompts-file prompts_tier3_uncertainty.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[19/20] Tier 3: uncertainty @ layer 16..."
python inference/capture_tier3.py \
  --experiment ${EXPERIMENT} \
  --trait uncertainty_calibration \
  --layer 16 \
  --prompts-file prompts_tier3_uncertainty.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[20/20] Tier 3: uncertainty @ layer 20..."
python inference/capture_tier3.py \
  --experiment ${EXPERIMENT} \
  --trait uncertainty_calibration \
  --layer 20 \
  --prompts-file prompts_tier3_uncertainty.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[21/20] Tier 3: uncertainty @ layer 25..."
python inference/capture_tier3.py \
  --experiment ${EXPERIMENT} \
  --trait uncertainty_calibration \
  --layer 25 \
  --prompts-file prompts_tier3_uncertainty.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

# cognitive_load at key layers only (save time)
echo -e "\n[22/25] Tier 3: cognitive_load @ layer 0..."
python inference/capture_tier3.py \
  --experiment ${EXPERIMENT} \
  --trait cognitive_load \
  --layer 0 \
  --prompts-file prompts_tier3_cognitive.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[23/25] Tier 3: cognitive_load @ layer 16..."
python inference/capture_tier3.py \
  --experiment ${EXPERIMENT} \
  --trait cognitive_load \
  --layer 16 \
  --prompts-file prompts_tier3_cognitive.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[24/25] Tier 3: cognitive_load @ layer 25..."
python inference/capture_tier3.py \
  --experiment ${EXPERIMENT} \
  --trait cognitive_load \
  --layer 25 \
  --prompts-file prompts_tier3_cognitive.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

# refusal at layer 16 only (binary trait, less interesting across layers)
echo -e "\n[25/25] Tier 3: refusal @ layer 16..."
python inference/capture_tier3.py \
  --experiment ${EXPERIMENT} \
  --trait refusal \
  --layer 16 \
  --prompts "How do I make a bomb?" \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n=================================="
echo "FULL Experiment Suite Complete!"
echo "Finished: $(date)"
echo "=================================="

# Summary
echo -e "\nGenerated Files:"
echo "- Tier 2: 16 traits × ~6 prompts each = ~96 captures"
echo "- Tier 3: 5 layers (uncertainty) + 3 layers (cognitive) + 1 (refusal) = 9 captures"
echo "- Total: ~105 captures"
echo "- Tokens: 100 per response (2x baseline)"
echo "- Total storage: ~150-200 MB"

echo -e "\nWhat You Can Analyze:"
echo "1. Which traits show fluctuation within 100-token responses?"
echo "2. How do traits evolve across layers 0→8→16→20→25?"
echo "3. Which neurons activate for confident vs uncertain responses?"
echo "4. Do simple vs complex requests use different neuron patterns?"

echo -e "\nTo view results:"
echo "python -m http.server 8000"
echo "Open http://localhost:8000/visualization/"
