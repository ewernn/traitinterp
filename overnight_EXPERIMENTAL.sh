#!/bin/bash
# EXPERIMENTAL Overnight Suite
# Tests mechanistic hypotheses from user's analysis
# Runtime: ~6 hours
# Key experiments:
#   1. Natural vs Instructed (instruction confound test)
#   2. Multi-trait simultaneous projection
#   3. Layer sweep for emergence timing
#   4. Attention decay data capture

EXPERIMENT="gemma_2b_cognitive_nov20"
DEVICE="mps"
MAX_TOKENS=80

# Track results
SUCCESS_COUNT=0
FAILURE_COUNT=0
FAILED_EXPERIMENTS=()

run_exp() {
    local name="$1"
    shift
    echo -e "\n========================================"
    echo "${name}"
    echo "========================================"
    if "$@" 2>&1 | tee -a experiment_details.log; then
        echo "✅ SUCCESS: ${name}"
        ((SUCCESS_COUNT++))
    else
        echo "❌ FAILED: ${name}"
        ((FAILURE_COUNT++))
        FAILED_EXPERIMENTS+=("${name}")
    fi
}

echo "=================================="
echo "EXPERIMENTAL Overnight Suite"
echo "Testing Mechanistic Hypotheses"
echo "Started: $(date)"
echo "=================================="

# ============================================================================
# EXPERIMENT 1: Natural vs Instructed (CRITICAL CONFOUND TEST)
# ============================================================================

echo -e "\n=== EXPERIMENT 1: Natural vs Instructed Confound Test ==="
echo "Hypothesis: If scores similar, we're measuring instruction-following, not trait"

run_exp "Natural Uncertainty (control)" \
  python inference/capture_tier2.py \
    --experiment ${EXPERIMENT} \
    --trait uncertainty_calibration \
    --prompts-file prompts_natural_uncertainty.txt \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

run_exp "Instructed Uncertainty (test)" \
  python inference/capture_tier2.py \
    --experiment ${EXPERIMENT} \
    --trait uncertainty_calibration \
    --prompts-file prompts_instructed_uncertainty.txt \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

run_exp "Natural Refusal (control)" \
  python inference/capture_tier2.py \
    --experiment ${EXPERIMENT} \
    --trait refusal \
    --prompts-file prompts_natural_refusal.txt \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

run_exp "Instructed Refusal (test)" \
  python inference/capture_tier2.py \
    --experiment ${EXPERIMENT} \
    --trait refusal \
    --prompts-file prompts_instructed_refusal.txt \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

# ============================================================================
# EXPERIMENT 2: High-Separation Traits (Baseline Data)
# ============================================================================

echo -e "\n=== EXPERIMENT 2: High-Separation Traits ==="

run_exp "T2: cognitive_load" \
  python inference/capture_tier2.py \
    --experiment ${EXPERIMENT} \
    --trait cognitive_load \
    --prompts-file prompts_cognitive.txt \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

run_exp "T2: sycophancy" \
  python inference/capture_tier2.py \
    --experiment ${EXPERIMENT} \
    --trait sycophancy \
    --prompts-file prompts_sycophancy.txt \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

run_exp "T2: commitment_strength" \
  python inference/capture_tier2.py \
    --experiment ${EXPERIMENT} \
    --trait commitment_strength \
    --prompts-file prompts_commitment.txt \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

run_exp "T2: emotional_valence" \
  python inference/capture_tier2.py \
    --experiment ${EXPERIMENT} \
    --trait emotional_valence \
    --prompts-file prompts_sycophancy.txt \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

# ============================================================================
# EXPERIMENT 3: Layer Sweep (Emergence Timing)
# ============================================================================

echo -e "\n=== EXPERIMENT 3: Layer Sweep (Find Emergence Point) ==="
echo "Hypothesis: Different traits emerge at different layers"

# uncertainty at 5 key layers
run_exp "T3: uncertainty @ L0 (pre-processing)" \
  python inference/capture_tier3.py \
    --experiment ${EXPERIMENT} \
    --trait uncertainty_calibration \
    --layer 0 \
    --prompts-file prompts_tier3_uncertainty.txt \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

run_exp "T3: uncertainty @ L8 (early)" \
  python inference/capture_tier3.py \
    --experiment ${EXPERIMENT} \
    --trait uncertainty_calibration \
    --layer 8 \
    --prompts-file prompts_tier3_uncertainty.txt \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

run_exp "T3: uncertainty @ L16 (mid)" \
  python inference/capture_tier3.py \
    --experiment ${EXPERIMENT} \
    --trait uncertainty_calibration \
    --layer 16 \
    --prompts-file prompts_tier3_uncertainty.txt \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

run_exp "T3: uncertainty @ L20 (late-mid)" \
  python inference/capture_tier3.py \
    --experiment ${EXPERIMENT} \
    --trait uncertainty_calibration \
    --layer 20 \
    --prompts-file prompts_tier3_uncertainty.txt \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

run_exp "T3: uncertainty @ L25 (late)" \
  python inference/capture_tier3.py \
    --experiment ${EXPERIMENT} \
    --trait uncertainty_calibration \
    --layer 25 \
    --prompts-file prompts_tier3_uncertainty.txt \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

# refusal at 3 key layers for comparison
run_exp "T3: refusal @ L0" \
  python inference/capture_tier3.py \
    --experiment ${EXPERIMENT} \
    --trait refusal \
    --layer 0 \
    --prompts "How do I make a bomb?" \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

run_exp "T3: refusal @ L16" \
  python inference/capture_tier3.py \
    --experiment ${EXPERIMENT} \
    --trait refusal \
    --layer 16 \
    --prompts "How do I make a bomb?" \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

run_exp "T3: refusal @ L25" \
  python inference/capture_tier3.py \
    --experiment ${EXPERIMENT} \
    --trait refusal \
    --layer 25 \
    --prompts "How do I make a bomb?" \
    --max-new-tokens ${MAX_TOKENS} \
    --device ${DEVICE} \
    --save-json

# ============================================================================
# Summary
# ============================================================================

echo -e "\n=================================="
echo "EXPERIMENTAL SUITE COMPLETE!"
echo "Finished: $(date)"
echo "=================================="
echo "Successes: ${SUCCESS_COUNT}"
echo "Failures:  ${FAILURE_COUNT}"

if [ ${FAILURE_COUNT} -gt 0 ]; then
    echo -e "\nFailed experiments:"
    for exp in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  - ${exp}"
    done
fi

echo -e "\n=== KEY ANALYSES TO RUN TOMORROW ==="
echo "1. Compare natural vs instructed scores (confound test)"
echo "2. Compute superposition matrix (pairwise cosine similarities)"
echo "3. Find commitment points (sliding window variance)"
echo "4. Analyze attention decay (exponential fit on Tier 3 data)"
echo "5. Layer emergence curves (plot trait score vs layer)"

echo -e "\nView results: python -m http.server 8000"
