#!/bin/bash
# Run extraction (stages 3+4) across all quant-sensitivity variants
# Response-position traits use default pos; prompt-position traits need --position prompt[-1]
set -e
cd /workspace/traitinterp
PY=/venv/main/bin/python

PROMPT_TRAITS="caa/sycophancy,arditi/refusal"
RESPONSE_TRAITS="pv_instruction/evil,pv_instruction/sycophancy,pv_instruction/hallucination"

run_extract() {
  local experiment="$1"; local traits="$2"; local flags="$3"; local position_flag="$4"
  echo ">>> $experiment | traits=$traits | flags='$flags' | pos='$position_flag'"
  $PY extraction/run_extraction_pipeline.py \
    --experiment "$experiment" \
    --traits "$traits" --only-stage 3,4 --methods probe \
    $position_flag $flags 2>&1 | tail -8
  echo
}

echo "=== Extractions starting at $(date) ==="

# BF16 baselines (all 5 traits across response + prompt)
for variant in llama-8b olmo-7b; do
  run_extract "quant-sensitivity/$variant" "$RESPONSE_TRAITS" "" ""
  run_extract "quant-sensitivity/$variant" "$PROMPT_TRAITS" "" "--position prompt[-1]"
done

# Quantized variants: prompt traits in base, response traits in -fp16resp
VARIANTS=(
  "llama-8b-nf4:--load-in-4bit --bnb-4bit-quant-type nf4"
  "llama-8b-fp4:--load-in-4bit --bnb-4bit-quant-type fp4"
  "llama-8b-awq:"
  "olmo-7b-nf4:--load-in-4bit --bnb-4bit-quant-type nf4"
)

for entry in "${VARIANTS[@]}"; do
  variant="${entry%%:*}"
  flags="${entry#*:}"
  run_extract "quant-sensitivity/$variant"          "$PROMPT_TRAITS"   "$flags" "--position prompt[-1]"
  run_extract "quant-sensitivity/$variant-fp16resp" "$RESPONSE_TRAITS" "$flags" ""
done

echo "=== Done at $(date) ==="
