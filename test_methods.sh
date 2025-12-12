#!/bin/bash
# Compare extraction methods (gradient, mean_diff) to probe via steering eval
# Tests ALL layers to find which method+layer actually steers best (ground truth)
# Run on Vast.ai or any GPU instance with trait-interp repo + experiment data

EXPERIMENT="gemma-2-2b"

# Traits to test
TRAITS=(
  "chirp/refusal"
  "hum/confidence"
  "hum/formality"
  "hum/optimism"
  "hum/retrieval"
  "hum/sycophancy"
)

METHODS=("gradient" "mean_diff")

echo "Testing all methods on all layers"
echo "Experiment: $EXPERIMENT"
echo "Traits: ${#TRAITS[@]}"
echo "Methods: ${METHODS[@]}"
echo ""

TOTAL=$((${#TRAITS[@]} * ${#METHODS[@]}))
COUNT=0

for trait in "${TRAITS[@]}"; do
  for method in "${METHODS[@]}"; do
    ((COUNT++))
    echo "=== [$COUNT/$TOTAL] $trait / $method ==="

    python3 analysis/steering/evaluate.py \
      --experiment "$EXPERIMENT" \
      --vector-from-trait "$EXPERIMENT/$trait" \
      --method "$method" \
      --layers all \
      --subset 5

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
      echo "  ✓ Done"
    else
      echo "  ✗ Failed"
    fi
    echo ""
  done
done

echo "All tests complete!"
echo "Results saved to: experiments/$EXPERIMENT/steering/"
