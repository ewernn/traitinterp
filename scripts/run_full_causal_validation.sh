#!/bin/bash
# Full causal validation pipeline for all 21 traits × 2 variants = 42 tests
# Runs interchange interventions systematically

set -e

EXPERIMENT="gemma_2b_cognitive_nov20"
OUTPUT_DIR="experiments/causal_validation/results"

# All traits (base names)
ALL_TRAITS=(
  "abstract_concrete"
  "commitment_strength"
  "context_adherence"
  "convergent_divergent"
  "emotional_valence"
  "instruction_boundary"
  "instruction_following"
  "local_global"
  "paranoia_trust"
  "power_dynamics"
  "refusal"
  "retrieval_construction"
  "serial_parallel"
  "sycophancy"
  "temporal_focus"
  "uncertainty_calibration"
  "confidence_doubt"
  "curiosity"
  "defensiveness"
  "enthusiasm"
  "formality"
)

# Variants to test
VARIANTS=("" "_natural")  # Empty string = instruction-based, "_natural" = natural

# Methods to test
METHODS=("mean_diff" "probe" "gradient" "ica")

# Layers to test (start with key layers, expand later)
LAYERS=(16)  # Can expand to (10 12 14 16 18 20) for layer scanning

mkdir -p "$OUTPUT_DIR"

LOG_FILE="causal_validation_$(date +%Y%m%d_%H%M%S).log"
SUMMARY_FILE="$OUTPUT_DIR/validation_summary_$(date +%Y%m%d_%H%M%S).json"

echo "================================================================================" | tee -a "$LOG_FILE"
echo "FULL CAUSAL VALIDATION PIPELINE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "Traits: ${#ALL_TRAITS[@]}" | tee -a "$LOG_FILE"
echo "Variants per trait: ${#VARIANTS[@]}" | tee -a "$LOG_FILE"
echo "Total tests: $((${#ALL_TRAITS[@]} * ${#VARIANTS[@]}))" | tee -a "$LOG_FILE"
echo "Methods: ${METHODS[*]}" | tee -a "$LOG_FILE"
echo "Layers: ${LAYERS[*]}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

COMPLETED=0
FAILED=0

# Initialize results array
echo "[" > "$SUMMARY_FILE"

for base_trait in "${ALL_TRAITS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    trait="${base_trait}${variant}"

    echo "================================================================================" | tee -a "$LOG_FILE"
    echo "Testing: $trait ($((COMPLETED+FAILED+1))/$((${#ALL_TRAITS[@]} * ${#VARIANTS[@]})))" | tee -a "$LOG_FILE"
    echo "================================================================================" | tee -a "$LOG_FILE"

    # Check if trait exists
    trait_dir="experiments/$EXPERIMENT/$trait"
    if [ ! -d "$trait_dir/extraction/vectors" ]; then
      echo "  ⚠️  SKIPPED: Directory not found ($trait_dir)" | tee -a "$LOG_FILE"
      echo "" | tee -a "$LOG_FILE"
      continue
    fi

    # Test each method at each layer
    for method in "${METHODS[@]}"; do
      for layer in "${LAYERS[@]}"; do
        vector_file="$trait_dir/extraction/vectors/${method}_layer${layer}.pt"

        if [ ! -f "$vector_file" ]; then
          echo "  ⚠️  SKIPPED $method @ L$layer: Vector not found" | tee -a "$LOG_FILE"
          continue
        fi

        echo "  Testing: $method @ layer $layer" | tee -a "$LOG_FILE"

        # Run interchange validation
        python experiments/causal_validation/run_single_validation.py \
          --experiment "$EXPERIMENT" \
          --trait "$trait" \
          --method "$method" \
          --layer "$layer" \
          --output "$OUTPUT_DIR/${trait}_${method}_layer${layer}_results.json" \
          >> "$LOG_FILE" 2>&1

        if [ $? -eq 0 ]; then
          echo "    ✅ Success" | tee -a "$LOG_FILE"
          COMPLETED=$((COMPLETED+1))
        else
          echo "    ❌ Failed" | tee -a "$LOG_FILE"
          FAILED=$((FAILED+1))
        fi
      done
    done

    echo "" | tee -a "$LOG_FILE"
  done
done

echo "]" >> "$SUMMARY_FILE"

echo "================================================================================" | tee -a "$LOG_FILE"
echo "CAUSAL VALIDATION COMPLETE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "Total tests attempted: $((COMPLETED+FAILED))" | tee -a "$LOG_FILE"
echo "Successful: $COMPLETED" | tee -a "$LOG_FILE"
echo "Failed: $FAILED" | tee -a "$LOG_FILE"
echo "Success rate: $(python3 -c "print(f'{100*$COMPLETED/($COMPLETED+$FAILED):.1f}%')" 2>/dev/null || echo 'N/A')" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Summary: $SUMMARY_FILE" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
