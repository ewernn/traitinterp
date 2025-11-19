#!/bin/bash
# Simple overnight extraction - just run this file
cd "$(dirname "$0")"

EXPERIMENT="gemma_2b_cognitive_nov20"
BATCH_SIZE=8

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

LOG_FILE="pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================================"
echo "LOCAL OVERNIGHT EXTRACTION - 12 TRAITS"
echo "================================================================================"
echo "Started: $(date)"
echo "Log: $LOG_FILE"
echo ""

COMPLETED=0
FAILED=0

for trait in "${TRAITS[@]}"; do
  echo "[$((COMPLETED+FAILED+1))/${#TRAITS[@]}] Processing: $trait"

  python extraction/1_generate_natural.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    --batch-size $BATCH_SIZE \
    >> "$LOG_FILE" 2>&1

  [ $? -ne 0 ] && { echo "❌ Failed: $trait (generation)"; FAILED=$((FAILED+1)); continue; }

  python extraction/2_extract_activations_natural.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    >> "$LOG_FILE" 2>&1

  [ $? -ne 0 ] && { echo "❌ Failed: $trait (activations)"; FAILED=$((FAILED+1)); continue; }

  python extraction/3_extract_vectors_natural.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    >> "$LOG_FILE" 2>&1

  [ $? -ne 0 ] && { echo "❌ Failed: $trait (vectors)"; FAILED=$((FAILED+1)); continue; }

  python scripts/run_cross_distribution.py \
    --trait "${trait}_natural" \
    >> "$LOG_FILE" 2>&1

  COMPLETED=$((COMPLETED+1))
  echo "✅ Complete: $trait ($COMPLETED done, $FAILED failed)"
  echo ""
done

echo "================================================================================"
echo "FINISHED: $(date)"
echo "Completed: $COMPLETED/${#TRAITS[@]}"
echo "Failed: $FAILED"
echo "Log: $LOG_FILE"
echo "================================================================================"
