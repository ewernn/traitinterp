#!/bin/bash
# Local overnight extraction - simple and reliable
# Works on Mac MPS or CPU

EXPERIMENT="gemma_2b_cognitive_nov20"
BATCH_SIZE=8  # Conservative for local execution

TRAITS=(
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
  "serial_parallel"
  "temporal_focus"
)

LOG_FILE="local_pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================================" | tee -a "$LOG_FILE"
echo "LOCAL OVERNIGHT EXTRACTION PIPELINE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "Experiment: $EXPERIMENT" | tee -a "$LOG_FILE"
echo "Traits: ${#TRAITS[@]}" | tee -a "$LOG_FILE"
echo "Batch size: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

COMPLETED=0
FAILED=0

for trait in "${TRAITS[@]}"; do
  echo "================================================================================" | tee -a "$LOG_FILE"
  echo "TRAIT: $trait ($((COMPLETED+FAILED+1))/${#TRAITS[@]})" | tee -a "$LOG_FILE"
  echo "================================================================================" | tee -a "$LOG_FILE"

  # Stage 1: Generate responses
  echo "[1/4] Generating responses..." | tee -a "$LOG_FILE"
  python extraction/1_generate_natural.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    --batch-size $BATCH_SIZE \
    2>&1 | tee -a "$LOG_FILE"

  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Generation failed for $trait" | tee -a "$LOG_FILE"
    FAILED=$((FAILED+1))
    continue
  fi

  # Stage 2: Extract activations
  echo "[2/4] Extracting activations..." | tee -a "$LOG_FILE"
  python extraction/2_extract_activations_natural.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    2>&1 | tee -a "$LOG_FILE"

  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Activation extraction failed for $trait" | tee -a "$LOG_FILE"
    FAILED=$((FAILED+1))
    continue
  fi

  # Stage 3: Extract vectors
  echo "[3/4] Extracting vectors..." | tee -a "$LOG_FILE"
  python extraction/3_extract_vectors_natural.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    2>&1 | tee -a "$LOG_FILE"

  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Vector extraction failed for $trait" | tee -a "$LOG_FILE"
    FAILED=$((FAILED+1))
    continue
  fi

  # Stage 4: Cross-distribution test
  echo "[4/4] Cross-distribution testing..." | tee -a "$LOG_FILE"
  python scripts/run_cross_distribution.py \
    --trait "${trait}_natural" \
    2>&1 | tee -a "$LOG_FILE"

  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "⚠️  Cross-distribution test failed for $trait (non-critical)" | tee -a "$LOG_FILE"
  fi

  COMPLETED=$((COMPLETED+1))
  echo "✅ $trait complete! ($COMPLETED/${#TRAITS[@]} done, $FAILED failed)" | tee -a "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"
done

echo "================================================================================" | tee -a "$LOG_FILE"
echo "PIPELINE COMPLETE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "Completed: $COMPLETED/${#TRAITS[@]}" | tee -a "$LOG_FILE"
echo "Failed: $FAILED" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
