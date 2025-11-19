#!/bin/bash
# Clean extraction data to start fresh
# Removes responses, activations, and vectors but keeps trait definitions

set -e

echo "================================================================"
echo "CLEANING EXTRACTION DATA"
echo "================================================================"
echo ""

# Create backup first
echo "Creating backup..."
cd "$(dirname "$0")/.."

BACKUP_FILE="backup_before_clean_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$BACKUP_FILE" experiments/gemma_2b_cognitive_nov20/ 2>/dev/null || echo "⚠️  Backup failed (continuing anyway)"

if [ -f "$BACKUP_FILE" ]; then
  echo "✓ Backup created: $BACKUP_FILE"
else
  echo "⚠️  No backup created"
fi
echo ""

# Clean instruction-based traits
echo "Cleaning instruction-based traits..."

INSTRUCTION_TRAITS=(
  "abstract_concrete" "commitment_strength" "context_adherence"
  "convergent_divergent" "emotional_valence" "instruction_boundary"
  "instruction_following" "local_global" "paranoia_trust"
  "power_dynamics" "refusal" "retrieval_construction"
  "serial_parallel" "sycophancy" "temporal_focus"
  "uncertainty_calibration" "confidence_doubt" "curiosity"
  "defensiveness" "enthusiasm" "formality"
)

for trait in "${INSTRUCTION_TRAITS[@]}"; do
  TRAIT_DIR="experiments/gemma_2b_cognitive_nov20/$trait"

  if [ -d "$TRAIT_DIR/extraction" ]; then
    echo "  Cleaning: $trait"
    rm -rf "$TRAIT_DIR/extraction/responses/" 2>/dev/null || true
    rm -rf "$TRAIT_DIR/extraction/activations/" 2>/dev/null || true
    rm -rf "$TRAIT_DIR/extraction/vectors/" 2>/dev/null || true
  fi
done

echo "✓ Instruction-based traits cleaned"
echo ""

# Clean natural variants completely (will be recreated)
echo "Removing natural variant directories..."

CLEANED_NATURAL=0
for dir in experiments/gemma_2b_cognitive_nov20/*_natural/; do
  if [ -d "$dir" ]; then
    trait=$(basename "$dir")
    echo "  Removing: $trait"
    rm -rf "$dir"
    CLEANED_NATURAL=$((CLEANED_NATURAL+1))
  fi
done

echo "✓ Removed $CLEANED_NATURAL natural variant directories"
echo ""

# Clean causal validation results
echo "Cleaning causal validation results..."
if [ -d "experiments/causal_validation/results" ]; then
  rm -rf experiments/causal_validation/results/*.json 2>/dev/null || true
  echo "✓ Validation results cleaned"
else
  echo "  (No validation results to clean)"
fi
echo ""

echo "================================================================"
echo "CLEANING COMPLETE"
echo "================================================================"
echo ""
echo "Summary:"
echo "  - Instruction traits: Cleaned 21 trait extraction directories"
echo "  - Natural variants: Removed $CLEANED_NATURAL directories"
echo "  - Validation results: Cleaned"
echo ""
echo "Backup: $BACKUP_FILE"
echo ""
echo "Ready for clean extraction!"
echo "================================================================"
