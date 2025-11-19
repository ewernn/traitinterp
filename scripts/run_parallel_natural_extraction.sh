#!/bin/bash
# Parallel execution for A100
# Runs 4 traits simultaneously
# Estimated time: ~30-45 minutes

EXPERIMENT="gemma_2b_cognitive_nov20"
BATCH_SIZE=32  # Lower batch size since running 4 in parallel
PARALLEL_JOBS=4  # Number of traits to run simultaneously

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
echo "PARALLEL NATURAL ELICITATION PIPELINE (A100)"
echo "================================================================================"
echo "Experiment: $EXPERIMENT"
echo "Batch size: $BATCH_SIZE"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Processing ${#TRAITS[@]} traits"
echo "Estimated time: ~30-45 minutes"
echo "Start time: $(date)"
echo ""

# Function to run full pipeline for one trait
run_trait() {
  local trait=$1
  local log_file="logs/natural_extraction_${trait}.log"
  mkdir -p logs

  {
    echo "========================================"
    echo "Starting: $trait"
    echo "Time: $(date)"
    echo "========================================"

    echo "Stage 1: Generation..."
    python extraction/1_generate_natural.py \
      --experiment "$EXPERIMENT" \
      --trait "$trait" \
      --batch-size $BATCH_SIZE \
      --device cuda \
      2>&1 || { echo "ERROR in generation"; exit 1; }

    echo "Stage 2: Activations..."
    python extraction/2_extract_activations_natural.py \
      --experiment "$EXPERIMENT" \
      --trait "$trait" \
      --device cuda \
      2>&1 || { echo "ERROR in activation extraction"; exit 1; }

    echo "Stage 3: Vectors..."
    python extraction/3_extract_vectors_natural.py \
      --experiment "$EXPERIMENT" \
      --trait "$trait" \
      2>&1 || { echo "ERROR in vector extraction"; exit 1; }

    echo "Stage 4: Cross-distribution..."
    python scripts/run_cross_distribution.py \
      --trait "$trait" \
      2>&1 || { echo "ERROR in cross-distribution"; exit 1; }

    echo "✅ COMPLETE: $trait"
    echo "Time: $(date)"
  } | tee "$log_file"
}

export -f run_trait
export EXPERIMENT BATCH_SIZE

# Run traits in parallel using GNU parallel or xargs
if command -v parallel &> /dev/null; then
  echo "Using GNU parallel..."
  printf '%s\n' "${TRAITS[@]}" | parallel -j $PARALLEL_JOBS run_trait {}
else
  echo "GNU parallel not found, using xargs..."
  printf '%s\n' "${TRAITS[@]}" | xargs -P $PARALLEL_JOBS -I {} bash -c 'run_trait "$@"' _ {}
fi

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETE"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Check individual logs in: logs/natural_extraction_*.log"
echo ""
