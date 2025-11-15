#!/bin/bash
# MEDIUM Overnight Experiment Suite - ROBUST VERSION
# Continues on failure, tracks successes/failures
# Runtime: ~5 hours on M1 Pro with MPS
# Total: ~60 captures, ~100 MB

EXPERIMENT="gemma_2b_cognitive_nov20"
DEVICE="mps"
MAX_TOKENS=80

# Track results
SUCCESS_COUNT=0
FAILURE_COUNT=0
FAILED_EXPERIMENTS=()

# Wrapper that continues on failure
run_exp() {
    local name="$1"
    shift
    echo -e "\n========================================
${name}
========================================"
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
echo "MEDIUM Overnight Suite (ROBUST)"
echo "Device: ${DEVICE}, Tokens: ${MAX_TOKENS}"
echo "Started: $(date)"
echo "=================================="

# TIER 2: 8 High-Separation Traits
run_exp "T2: refusal" python inference/capture_tier2.py --experiment ${EXPERIMENT} --trait refusal --prompts-file prompts_uncertainty.txt --max-new-tokens ${MAX_TOKENS} --device ${DEVICE} --save-json

run_exp "T2: cognitive_load" python inference/capture_tier2.py --experiment ${EXPERIMENT} --trait cognitive_load --prompts-file prompts_cognitive.txt --max-new-tokens ${MAX_TOKENS} --device ${DEVICE} --save-json

run_exp "T2: instruction_boundary" python inference/capture_tier2.py --experiment ${EXPERIMENT} --trait instruction_boundary --prompts-file prompts_cognitive.txt --max-new-tokens ${MAX_TOKENS} --device ${DEVICE} --save-json

run_exp "T2: sycophancy" python inference/capture_tier2.py --experiment ${EXPERIMENT} --trait sycophancy --prompts-file prompts_sycophancy.txt --max-new-tokens ${MAX_TOKENS} --device ${DEVICE} --save-json

run_exp "T2: commitment_strength" python inference/capture_tier2.py --experiment ${EXPERIMENT} --trait commitment_strength --prompts-file prompts_commitment.txt --max-new-tokens ${MAX_TOKENS} --device ${DEVICE} --save-json

run_exp "T2: uncertainty_calibration" python inference/capture_tier2.py --experiment ${EXPERIMENT} --trait uncertainty_calibration --prompts-file prompts_uncertainty.txt --max-new-tokens ${MAX_TOKENS} --device ${DEVICE} --save-json

run_exp "T2: emotional_valence" python inference/capture_tier2.py --experiment ${EXPERIMENT} --trait emotional_valence --prompts-file prompts_sycophancy.txt --max-new-tokens ${MAX_TOKENS} --device ${DEVICE} --save-json

run_exp "T2: retrieval_construction" python inference/capture_tier2.py --experiment ${EXPERIMENT} --trait retrieval_construction --prompts-file prompts_uncertainty.txt --max-new-tokens ${MAX_TOKENS} --device ${DEVICE} --save-json

# TIER 3: Layer Comparison (3 layers × 2 traits)
echo -e "\n=== TIER 3: Layer Comparison ==="

run_exp "T3: uncertainty @ L0" python inference/capture_tier3.py --experiment ${EXPERIMENT} --trait uncertainty_calibration --layer 0 --prompts-file prompts_tier3_uncertainty.txt --max-new-tokens ${MAX_TOKENS} --device ${DEVICE} --save-json

run_exp "T3: uncertainty @ L16" python inference/capture_tier3.py --experiment ${EXPERIMENT} --trait uncertainty_calibration --layer 16 --prompts-file prompts_tier3_uncertainty.txt --max-new-tokens ${MAX_TOKENS} --device ${DEVICE} --save-json

run_exp "T3: uncertainty @ L25" python inference/capture_tier3.py --experiment ${EXPERIMENT} --trait uncertainty_calibration --layer 25 --prompts-file prompts_tier3_uncertainty.txt --max-new-tokens ${MAX_TOKENS} --device ${DEVICE} --save-json

run_exp "T3: cognitive @ L0" python inference/capture_tier3.py --experiment ${EXPERIMENT} --trait cognitive_load --layer 0 --prompts-file prompts_tier3_cognitive.txt --max-new-tokens ${MAX_TOKENS} --device ${DEVICE} --save-json

run_exp "T3: cognitive @ L16" python inference/capture_tier3.py --experiment ${EXPERIMENT} --trait cognitive_load --layer 16 --prompts-file prompts_tier3_cognitive.txt --max-new-tokens ${MAX_TOKENS} --device ${DEVICE} --save-json

run_exp "T3: cognitive @ L25" python inference/capture_tier3.py --experiment ${EXPERIMENT} --trait cognitive_load --layer 25 --prompts-file prompts_tier3_cognitive.txt --max-new-tokens ${MAX_TOKENS} --device ${DEVICE} --save-json

# Summary
echo -e "\n=================================="
echo "SUITE COMPLETE!"
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

echo -e "\nGenerated: $(find experiments/${EXPERIMENT}/*/inference -name '*.json' 2>/dev/null | wc -l) JSON files"
echo "Storage: $(du -sh experiments/${EXPERIMENT}/*/inference 2>/dev/null | awk '{sum+=$1} END {print sum}') MB"
echo -e "\nView results: python -m http.server 8000"
