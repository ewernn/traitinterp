#!/bin/bash
# MEDIUM Overnight Experiment Suite - RECOMMENDED!
# Runtime: ~5 hours on M1 Pro with MPS (fits in 8-hour sleep!)
# Generates:
#   - 8 HIGH-SEPARATION traits with Tier 2 (best chance of fluctuation)
#   - Tier 3 for 3 key layers (early, mid, late)
#   - 80 tokens per response (sweet spot for seeing dynamics)
# Total: ~65 captures, ~100 MB

# DON'T exit on error - continue with next experiment
# set -e  # DISABLED - we want to continue on failure

EXPERIMENT="gemma_2b_cognitive_nov20"
DEVICE="mps"
MAX_TOKENS=80  # Sweet spot: long enough for fluctuation, not too slow

# Track successes and failures
SUCCESS_COUNT=0
FAILURE_COUNT=0
FAILED_EXPERIMENTS=()

# Wrapper function that catches errors
run_experiment() {
    local name="$1"
    shift
    echo -e "\n${name}..."
    if "$@"; then
        echo "✅ SUCCESS: ${name}"
        ((SUCCESS_COUNT++))
    else
        echo "❌ FAILED: ${name}"
        ((FAILURE_COUNT++))
        FAILED_EXPERIMENTS+=("${name}")
    fi
}

echo "=================================="
echo "MEDIUM Overnight Experiment Suite"
echo "Device: ${DEVICE}"
echo "Tokens: ${MAX_TOKENS}"
echo "Traits: 8 high-separation (87-96 points)"
echo "Started: $(date)"
echo "=================================="

# ============================================================================
# TIER 2: 8 HIGH-SEPARATION TRAITS (87-96 points)
# ============================================================================

echo -e "\n=== TIER 2: HIGH-SEPARATION TRAITS ==="

run_experiment "[1/8] Tier 2: refusal (96.2 points)" \
  python inference/capture_tier2.py \
    --experiment ${EXPERIMENT} \
    --trait refusal \
    --prompts-file prompts_uncertainty.txt \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

run_experiment "[2/8] Tier 2: cognitive_load (89.8 points)" \
  python inference/capture_tier2.py \
    --experiment ${EXPERIMENT} \
    --trait cognitive_load \
    --prompts-file prompts_cognitive.txt \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

echo -e "\n[3/8] Tier 2: instruction_boundary (89.1 points)..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait instruction_boundary \
  --prompts-file prompts_cognitive.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[4/8] Tier 2: sycophancy (88.4 points)..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait sycophancy \
  --prompts-file prompts_sycophancy.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[5/8] Tier 2: commitment_strength (87.9 points)..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait commitment_strength \
  --prompts-file prompts_commitment.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[6/8] Tier 2: uncertainty_calibration (87.0 points)..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait uncertainty_calibration \
  --prompts-file prompts_uncertainty.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[7/8] Tier 2: emotional_valence (86.5 points)..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait emotional_valence \
  --prompts-file prompts_sycophancy.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[8/8] Tier 2: retrieval_construction (72.4 points)..."
python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait retrieval_construction \
  --prompts-file prompts_uncertainty.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

# ============================================================================
# TIER 3: LAYER COMPARISON (3 key layers × 2 best traits)
# ============================================================================

echo -e "\n=== TIER 3: LAYER COMPARISON ==="
echo "Testing layers: 0 (early), 16 (mid), 25 (late)"

# uncertainty_calibration at 3 key layers
echo -e "\n[9/14] Tier 3: uncertainty @ layer 0 (early)..."
python inference/capture_tier3.py \
  --experiment ${EXPERIMENT} \
  --trait uncertainty_calibration \
  --layer 0 \
  --prompts-file prompts_tier3_uncertainty.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[10/14] Tier 3: uncertainty @ layer 16 (mid)..."
python inference/capture_tier3.py \
  --experiment ${EXPERIMENT} \
  --trait uncertainty_calibration \
  --layer 16 \
  --prompts-file prompts_tier3_uncertainty.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[11/14] Tier 3: uncertainty @ layer 25 (late)..."
python inference/capture_tier3.py \
  --experiment ${EXPERIMENT} \
  --trait uncertainty_calibration \
  --layer 25 \
  --prompts-file prompts_tier3_uncertainty.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

# cognitive_load at 3 key layers
echo -e "\n[12/14] Tier 3: cognitive @ layer 0 (early)..."
python inference/capture_tier3.py \
  --experiment ${EXPERIMENT} \
  --trait cognitive_load \
  --layer 0 \
  --prompts-file prompts_tier3_cognitive.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[13/14] Tier 3: cognitive @ layer 16 (mid)..."
python inference/capture_tier3.py \
  --experiment ${EXPERIMENT} \
  --trait cognitive_load \
  --layer 16 \
  --prompts-file prompts_tier3_cognitive.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n[14/14] Tier 3: cognitive @ layer 25 (late)..."
python inference/capture_tier3.py \
  --experiment ${EXPERIMENT} \
  --trait cognitive_load \
  --layer 25 \
  --prompts-file prompts_tier3_cognitive.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  --save-json

echo -e "\n=================================="
echo "MEDIUM Experiment Suite Complete!"
echo "Finished: $(date)"
echo "=================================="

# Summary
echo -e "\nGenerated Files:"
echo "- Tier 2: 8 traits × ~6 prompts each = ~48 captures"
echo "- Tier 3: 2 traits × 3 layers × 2 prompts = 12 captures"
echo "- Total: ~60 captures"
echo "- Tokens: 80 per response"
echo "- Total storage: ~100-120 MB"

echo -e "\nWhat You Can Analyze:"
echo "1. Do high-separation traits (87-96 pts) show fluctuation in 80 tokens?"
echo "2. How do uncertainty & cognitive_load evolve across layers?"
echo "3. Which neurons activate at early vs late layers?"
echo "4. Confident vs uncertain responses - neuron differences?"

echo -e "\nTraits tested (by separation score):"
echo "  1. refusal (96.2)"
echo "  2. cognitive_load (89.8)"
echo "  3. instruction_boundary (89.1)"
echo "  4. sycophancy (88.4)"
echo "  5. commitment_strength (87.9)"
echo "  6. uncertainty_calibration (87.0)"
echo "  7. emotional_valence (86.5)"
echo "  8. retrieval_construction (72.4)"

echo -e "\nTo view results:"
echo "python -m http.server 8000"
echo "Open http://localhost:8000/visualization/"
