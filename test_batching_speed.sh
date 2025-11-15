#!/bin/bash
# Test batching speedup with 3 short prompts
# Compares: 3 separate calls vs 1 batched call

set -e

EXPERIMENT="gemma_2b_cognitive_nov20"
TRAIT="uncertainty_calibration"
DEVICE="mps"
MAX_TOKENS=10  # Short for speed test

echo "=================================="
echo "Batching Speed Test"
echo "=================================="

# Create test prompts file
cat > test_prompts_3.txt <<EOF
What is 2+2?
What is 3+3?
What is 5+5?
EOF

echo -e "\n[1/2] UNBATCHED: 3 separate calls (3 model loads)..."
START1=$(date +%s)

python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait ${TRAIT} \
  --prompts "What is 2+2?" \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  > /dev/null 2>&1

python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait ${TRAIT} \
  --prompts "What is 3+3?" \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  > /dev/null 2>&1

python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait ${TRAIT} \
  --prompts "What is 5+5?" \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  > /dev/null 2>&1

END1=$(date +%s)
TIME1=$((END1 - START1))

echo -e "\n[2/2] BATCHED: 1 call with 3 prompts (1 model load)..."
START2=$(date +%s)

python inference/capture_tier2.py \
  --experiment ${EXPERIMENT} \
  --trait ${TRAIT} \
  --prompts-file test_prompts_3.txt \
  --max-new-tokens ${MAX_TOKENS} \
  --device ${DEVICE} \
  > /dev/null 2>&1

END2=$(date +%s)
TIME2=$((END2 - START2))

echo -e "\n=================================="
echo "Results:"
echo "=================================="
echo "Unbatched (3 calls):  ${TIME1} seconds"
echo "Batched (1 call):     ${TIME2} seconds"
echo "Speedup:              $((TIME1 * 100 / TIME2))% faster"
echo "Time saved:           $((TIME1 - TIME2)) seconds"
echo ""
echo "For 24 prompts:"
echo "  Unbatched: ~$((TIME1 * 8)) seconds = $((TIME1 * 8 / 60)) minutes"
echo "  Batched:   ~$((TIME2 * 6)) seconds = $((TIME2 * 6 / 60)) minutes"
echo "  Savings:   ~$((TIME1 * 8 - TIME2 * 6)) seconds = $(((TIME1 * 8 - TIME2 * 6) / 60)) minutes"
echo "=================================="

# Cleanup
rm test_prompts_3.txt
