#!/bin/bash
# Master script: Complete clean extraction pipeline
# Runs all 4 stages sequentially for fully clean data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

MASTER_LOG="complete_pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================" | tee -a "$MASTER_LOG"
echo "COMPLETE CLEAN EXTRACTION PIPELINE" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
echo "Started: $(date)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "This will run:" | tee -a "$MASTER_LOG"
echo "  Stage 0: Clean existing data" | tee -a "$MASTER_LOG"
echo "  Stage 1: Instruction-based extraction (21 traits)" | tee -a "$MASTER_LOG"
echo "  Stage 2: Natural extraction (21 traits)" | tee -a "$MASTER_LOG"
echo "  Stage 3: Causal validation (42 variants)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Estimated time:" | tee -a "$MASTER_LOG"
echo "  Local Mac: ~20-28 hours" | tee -a "$MASTER_LOG"
echo "  Remote A100: ~5-7 hours" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Stage 0: Clean
echo "================================================================" | tee -a "$MASTER_LOG"
echo "STAGE 0: CLEANING EXISTING DATA" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
echo "Started: $(date)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

./scripts/clean_extraction_data.sh | tee -a "$MASTER_LOG"

if [ $? -ne 0 ]; then
  echo "❌ Stage 0 failed!" | tee -a "$MASTER_LOG"
  exit 1
fi

echo "✅ Stage 0 complete" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Stage 1: Instruction-based extraction
echo "================================================================" | tee -a "$MASTER_LOG"
echo "STAGE 1: INSTRUCTION-BASED EXTRACTION" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
echo "Started: $(date)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

./scripts/extract_all_instruction.sh | tee -a "$MASTER_LOG"

STAGE1_EXIT=$?
if [ $STAGE1_EXIT -ne 0 ]; then
  echo "⚠️  Stage 1 had failures (check instruction_extraction_*.log)" | tee -a "$MASTER_LOG"
  echo "Continuing to Stage 2..." | tee -a "$MASTER_LOG"
fi

echo "✅ Stage 1 complete (with $STAGE1_EXIT failures)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Stage 2: Natural extraction
echo "================================================================" | tee -a "$MASTER_LOG"
echo "STAGE 2: NATURAL EXTRACTION" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
echo "Started: $(date)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

./scripts/extract_all_natural.sh | tee -a "$MASTER_LOG"

STAGE2_EXIT=$?
if [ $STAGE2_EXIT -ne 0 ]; then
  echo "⚠️  Stage 2 had failures (check natural_extraction_*.log)" | tee -a "$MASTER_LOG"
  echo "Continuing to Stage 3..." | tee -a "$MASTER_LOG"
fi

echo "✅ Stage 2 complete (with $STAGE2_EXIT failures)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Stage 3: Causal validation
echo "================================================================" | tee -a "$MASTER_LOG"
echo "STAGE 3: CAUSAL VALIDATION" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
echo "Started: $(date)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

./scripts/run_full_causal_validation.sh | tee -a "$MASTER_LOG"

STAGE3_EXIT=$?
if [ $STAGE3_EXIT -ne 0 ]; then
  echo "⚠️  Stage 3 had failures (check causal_validation_*.log)" | tee -a "$MASTER_LOG"
fi

echo "✅ Stage 3 complete" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Final summary
echo "================================================================" | tee -a "$MASTER_LOG"
echo "PIPELINE COMPLETE" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
echo "Finished: $(date)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Count results
INSTRUCTION_VECTORS=$(find experiments/gemma_2b_cognitive_nov20/*/extraction/vectors -name "*.pt" 2>/dev/null | wc -l || echo 0)
NATURAL_VECTORS=$(find experiments/gemma_2b_cognitive_nov20/*_natural/extraction/vectors -name "*.pt" 2>/dev/null | wc -l || echo 0)
VALIDATION_RESULTS=$(ls experiments/causal_validation/results/*_results.json 2>/dev/null | wc -l || echo 0)

echo "Results:" | tee -a "$MASTER_LOG"
echo "  Instruction-based vectors: $INSTRUCTION_VECTORS (expected: 2,184)" | tee -a "$MASTER_LOG"
echo "  Natural vectors: $NATURAL_VECTORS (expected: 2,184)" | tee -a "$MASTER_LOG"
echo "  Causal validation results: $VALIDATION_RESULTS (expected: ≥168)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

if [ $INSTRUCTION_VECTORS -ge 2000 ] && [ $NATURAL_VECTORS -ge 2000 ] && [ $VALIDATION_RESULTS -ge 150 ]; then
  echo "✅ PIPELINE SUCCESS!" | tee -a "$MASTER_LOG"
  echo "" | tee -a "$MASTER_LOG"
  echo "You now have:" | tee -a "$MASTER_LOG"
  echo "  • 42 trait variants (21 instruction + 21 natural)" | tee -a "$MASTER_LOG"
  echo "  • 4,368 total vectors" | tee -a "$MASTER_LOG"
  echo "  • Full causal validation results" | tee -a "$MASTER_LOG"
  echo "  • ZERO mixed data" | tee -a "$MASTER_LOG"
else
  echo "⚠️  PIPELINE INCOMPLETE" | tee -a "$MASTER_LOG"
  echo "" | tee -a "$MASTER_LOG"
  echo "Check individual stage logs for details:" | tee -a "$MASTER_LOG"
  echo "  • instruction_extraction_*.log" | tee -a "$MASTER_LOG"
  echo "  • natural_extraction_*.log" | tee -a "$MASTER_LOG"
  echo "  • causal_validation_*.log" | tee -a "$MASTER_LOG"
fi

echo "" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"

exit 0
