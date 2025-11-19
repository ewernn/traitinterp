#!/bin/bash
# Run complete natural elicitation extraction pipeline for all 12 traits
# Estimated time: ~12 hours total

EXPERIMENT="gemma_2b_cognitive_nov20"

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
echo "NATURAL ELICITATION EXTRACTION PIPELINE"
echo "================================================================================"
echo "Experiment: $EXPERIMENT"
echo "Processing ${#TRAITS[@]} traits"
echo "Estimated time: ~12 hours"
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
  echo "Stage 1/4: Generating responses (~40 min)..."
  python extraction/1_generate_natural.py --experiment "$EXPERIMENT" --trait "$trait"
  if [ $? -ne 0 ]; then
    echo "❌ ERROR: Generation failed for $trait"
    echo "Continuing to next trait..."
    continue
  fi
  echo "✅ Generation complete"

  echo ""
  echo "Stage 2/4: Extracting activations (~10 min)..."
  python extraction/2_extract_activations_natural.py --experiment "$EXPERIMENT" --trait "$trait"
  if [ $? -ne 0 ]; then
    echo "❌ ERROR: Activation extraction failed for $trait"
    echo "Continuing to next trait..."
    continue
  fi
  echo "✅ Activation extraction complete"

  echo ""
  echo "Stage 3/4: Extracting vectors (~5 min)..."
  python extraction/3_extract_vectors_natural.py --experiment "$EXPERIMENT" --trait "$trait"
  if [ $? -ne 0 ]; then
    echo "❌ ERROR: Vector extraction failed for $trait"
    echo "Continuing to next trait..."
    continue
  fi
  echo "✅ Vector extraction complete"

  echo ""
  echo "Stage 4/4: Cross-distribution testing (~5 min)..."
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
echo "Results:"
echo "  - Natural vectors: experiments/$EXPERIMENT/{trait}_natural/extraction/vectors/"
echo "  - Cross-distribution: results/cross_distribution_analysis/{trait}_full_4x4_results.json"
echo ""
echo "Next steps:"
echo "  1. Review TOP5_LAYERS_CROSS_DISTRIBUTION.txt for layer recommendations"
echo "  2. Check EXTRACTION_SCORES_ALL_TRAITS.txt for accuracy metrics"
echo "  3. Update documentation with cross-distribution findings"
echo ""
