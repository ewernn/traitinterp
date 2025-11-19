#!/bin/bash
# Optimized for 80GB A100
# Estimated time: ~2-3 hours (vs 12 hours on CPU)

EXPERIMENT="gemma_2b_cognitive_nov20"
BATCH_SIZE=64  # Increase from 8 to 64 for A100

TRAITS=(
  "abstract_concrete"
  "commitment_strength"
  "context_adherence"
  "convergent_divergent"
  "instruction_boundary"
  "local_global"
  "paranoia_trust"
  "power_dynamics"
  "retrieval_construction"
  "serial_parallel"
  "sycophancy"
  "temporal_focus"
)

echo "================================================================================"
echo "NATURAL ELICITATION EXTRACTION PIPELINE (A100 Optimized)"
echo "================================================================================"
echo "Experiment: $EXPERIMENT"
echo "Batch size: $BATCH_SIZE (optimized for A100)"
echo "Processing ${#TRAITS[@]} traits"
echo "Estimated time: ~2-3 hours"
echo "Start time: $(date)"
echo ""

TOTAL=${#TRAITS[@]}
CURRENT=0

for trait in "${TRAITS[@]}"; do
  CURRENT=$((CURRENT + 1))
  echo ""
  echo "================================================================================"
  echo "[$CURRENT/$TOTAL] Processing: $trait"
  echo "================================================================================"

  echo ""
  echo "Stage 1/4: Generating responses (~10 min with A100)..."
  python extraction/1_generate_natural.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    --batch-size $BATCH_SIZE \
    --device cuda

  if [ $? -ne 0 ]; then
    echo "❌ ERROR: Generation failed for $trait"
    echo "Continuing to next trait..."
    continue
  fi
  echo "✅ Generation complete"

  echo ""
  echo "Stage 2/4: Extracting activations (~5 min)..."
  python extraction/2_extract_activations_natural.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    --device cuda

  if [ $? -ne 0 ]; then
    echo "❌ ERROR: Activation extraction failed for $trait"
    echo "Continuing to next trait..."
    continue
  fi
  echo "✅ Activation extraction complete"

  echo ""
  echo "Stage 3/4: Extracting vectors (~2 min)..."
  python extraction/3_extract_vectors_natural.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait"

  if [ $? -ne 0 ]; then
    echo "❌ ERROR: Vector extraction failed for $trait"
    echo "Continuing to next trait..."
    continue
  fi
  echo "✅ Vector extraction complete"

  echo ""
  echo "Stage 4/4: Cross-distribution testing (~3 min)..."
  python scripts/run_cross_distribution.py --trait "$trait"

  if [ $? -ne 0 ]; then
    echo "❌ ERROR: Cross-distribution testing failed for $trait"
    echo "Continuing to next trait..."
    continue
  fi
  echo "✅ Cross-distribution testing complete"

  echo ""
  echo "🎉 $trait COMPLETE!"
  echo "Progress: $CURRENT/$TOTAL traits"
  echo ""
done

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETE"
echo "================================================================================"
echo "End time: $(date)"
echo "Processed: $TOTAL traits"
echo ""
