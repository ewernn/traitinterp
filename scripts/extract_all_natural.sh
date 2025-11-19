#!/bin/bash
# Full natural extraction for all 21 traits
# Generates responses WITHOUT instructions in separate *_natural directories

set -e

EXPERIMENT="gemma_2b_cognitive_nov20"
BATCH_SIZE=8

ALL_TRAITS=(
  "abstract_concrete" "commitment_strength" "context_adherence"
  "convergent_divergent" "emotional_valence" "instruction_boundary"
  "instruction_following" "local_global" "paranoia_trust"
  "power_dynamics" "refusal" "retrieval_construction"
  "serial_parallel" "sycophancy" "temporal_focus"
  "uncertainty_calibration" "confidence_doubt" "curiosity"
  "defensiveness" "enthusiasm" "formality"
)

LOG_FILE="natural_extraction_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================" | tee -a "$LOG_FILE"
echo "FULL NATURAL EXTRACTION - 21 TRAITS" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Traits: ${#ALL_TRAITS[@]}" | tee -a "$LOG_FILE"
echo "Batch size: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

COMPLETED=0
FAILED=0

for base_trait in "${ALL_TRAITS[@]}"; do
  trait="${base_trait}_natural"

  echo "================================================================" | tee -a "$LOG_FILE"
  echo "[$((COMPLETED+FAILED+1))/21] TRAIT: $trait" | tee -a "$LOG_FILE"
  echo "================================================================" | tee -a "$LOG_FILE"

  # Step 1: Generate responses (NO instructions - natural elicitation)
  echo "  [1/4] Generating responses (natural)..." | tee -a "$LOG_FILE"
  python extraction/1_generate_natural.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    --batch-size $BATCH_SIZE \
    >> "$LOG_FILE" 2>&1

  if [ $? -ne 0 ]; then
    echo "  ❌ FAILED at generation stage" | tee -a "$LOG_FILE"
    FAILED=$((FAILED+1))
    echo "" | tee -a "$LOG_FILE"
    continue
  fi

  # Step 2: Extract activations
  echo "  [2/4] Extracting activations (26 layers)..." | tee -a "$LOG_FILE"
  python extraction/2_extract_activations_natural.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    >> "$LOG_FILE" 2>&1

  if [ $? -ne 0 ]; then
    echo "  ❌ FAILED at activation extraction" | tee -a "$LOG_FILE"
    FAILED=$((FAILED+1))
    echo "" | tee -a "$LOG_FILE"
    continue
  fi

  # Step 3: Extract vectors (all methods, all layers)
  echo "  [3/4] Extracting vectors (4 methods × 26 layers)..." | tee -a "$LOG_FILE"
  python extraction/3_extract_vectors.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    >> "$LOG_FILE" 2>&1

  if [ $? -ne 0 ]; then
    echo "  ❌ FAILED at vector extraction" | tee -a "$LOG_FILE"
    FAILED=$((FAILED+1))
    echo "" | tee -a "$LOG_FILE"
    continue
  fi

  # Step 4: Validate polarity (non-critical)
  echo "  [4/4] Validating polarity..." | tee -a "$LOG_FILE"
  python extraction/validate_natural_vectors.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    --layer 16 \
    --method probe \
    >> "$LOG_FILE" 2>&1 || echo "  ⚠️  Polarity validation failed (non-critical)" | tee -a "$LOG_FILE"

  COMPLETED=$((COMPLETED+1))
  echo "  ✅ COMPLETE ($COMPLETED/21)" | tee -a "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"
done

echo "================================================================" | tee -a "$LOG_FILE"
echo "NATURAL EXTRACTION COMPLETE" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "Completed: $COMPLETED/21" | tee -a "$LOG_FILE"
echo "Failed: $FAILED" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $COMPLETED -eq 21 ]; then
  echo "✅ SUCCESS: All 21 natural variants extracted!" | tee -a "$LOG_FILE"
elif [ $COMPLETED -ge 15 ]; then
  echo "⚠️  PARTIAL SUCCESS: $COMPLETED/21 natural variants extracted" | tee -a "$LOG_FILE"
else
  echo "❌ FAILURE: Only $COMPLETED/21 natural variants extracted" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Verify results:" | tee -a "$LOG_FILE"
echo "  find experiments/gemma_2b_cognitive_nov20/*_natural/extraction/vectors -name '*.pt' | wc -l" | tee -a "$LOG_FILE"
echo "  # Should be: $((COMPLETED * 104)) vectors" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

exit $([ $FAILED -eq 0 ] && echo 0 || echo 1)
