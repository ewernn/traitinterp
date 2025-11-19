#!/bin/bash
# Create proper natural variants for all 21 traits in separate directories
# This fixes the overnight run that overwrote instruction-based data

set -e

EXPERIMENT="gemma_2b_cognitive_nov20"
BATCH_SIZE=8

# All 21 traits (16 core + 4 additional + 1 formality)
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

# Traits that already have proper natural variants
EXISTING_NATURAL=(
  "emotional_valence"
  "formality"
  "refusal"
  "uncertainty_calibration"
)

# Function to check if trait is in existing list
trait_exists() {
  local trait=$1
  for existing in "${EXISTING_NATURAL[@]}"; do
    [[ "$existing" == "$trait" ]] && return 0
  done
  return 1
}

# Filter to only process traits that need natural variants
TRAITS_TO_PROCESS=()
for trait in "${ALL_TRAITS[@]}"; do
  if ! trait_exists "$trait"; then
    TRAITS_TO_PROCESS+=("$trait")
  fi
done

echo "================================================================================"
echo "CREATING NATURAL VARIANTS FOR ALL TRAITS"
echo "================================================================================"
echo "Total traits: ${#ALL_TRAITS[@]}"
echo "Already have natural: ${#EXISTING_NATURAL[@]}"
echo "Need to process: ${#TRAITS_TO_PROCESS[@]}"
echo ""
echo "Traits to process:"
printf '  - %s\n' "${TRAITS_TO_PROCESS[@]}"
echo ""
echo "================================================================================"
echo ""

COMPLETED=0
FAILED=0
LOG_FILE="natural_variants_$(date +%Y%m%d_%H%M%S).log"

for base_trait in "${TRAITS_TO_PROCESS[@]}"; do
  trait="${base_trait}_natural"

  echo "[$((COMPLETED+FAILED+1))/${#TRAITS_TO_PROCESS[@]}] Processing: $trait"
  echo "================================================================================"

  # Step 1: Generate responses
  echo "  [1/4] Generating responses..."
  python extraction/1_generate_natural.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    --batch-size $BATCH_SIZE \
    >> "$LOG_FILE" 2>&1

  if [ $? -ne 0 ]; then
    echo "  ❌ FAILED at generation stage"
    FAILED=$((FAILED+1))
    echo ""
    continue
  fi

  # Step 2: Extract activations (all 26 layers)
  echo "  [2/4] Extracting activations (26 layers)..."
  python extraction/2_extract_activations_natural.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    >> "$LOG_FILE" 2>&1

  if [ $? -ne 0 ]; then
    echo "  ❌ FAILED at activation extraction"
    FAILED=$((FAILED+1))
    echo ""
    continue
  fi

  # Step 3: Extract vectors (all 4 methods × 26 layers = 104 vectors)
  echo "  [3/4] Extracting vectors (4 methods × 26 layers)..."
  python extraction/3_extract_vectors.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    >> "$LOG_FILE" 2>&1

  if [ $? -ne 0 ]; then
    echo "  ❌ FAILED at vector extraction"
    FAILED=$((FAILED+1))
    echo ""
    continue
  fi

  # Step 4: Validate polarity
  echo "  [4/4] Validating polarity..."
  python extraction/validate_natural_vectors.py \
    --experiment "$EXPERIMENT" \
    --trait "$trait" \
    --layer 16 \
    --method probe \
    >> "$LOG_FILE" 2>&1

  if [ $? -ne 0 ]; then
    echo "  ⚠️  Polarity validation failed (but vectors exist)"
  fi

  COMPLETED=$((COMPLETED+1))
  echo "  ✅ Complete: $trait"
  echo ""
done

echo "================================================================================"
echo "NATURAL VARIANT CREATION COMPLETE"
echo "================================================================================"
echo "Started: $(head -1 "$LOG_FILE" 2>/dev/null || echo 'N/A')"
echo "Finished: $(date)"
echo "Completed: $COMPLETED/${#TRAITS_TO_PROCESS[@]}"
echo "Failed: $FAILED"
echo "Log: $LOG_FILE"
echo ""
echo "Verify results:"
echo "  ls experiments/gemma_2b_cognitive_nov20/*_natural/extraction/vectors/*.pt | wc -l"
echo "  # Should be: $((${#ALL_TRAITS[@]} * 104)) vectors (21 traits × 104 vectors each)"
echo "================================================================================"
