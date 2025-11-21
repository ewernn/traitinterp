#!/bin/bash
# Complete Extraction Pipeline for gemma_2b_cognitive_nov21
# Runs all missing stages to achieve 100% coverage: 4 methods × 26 layers for all traits
#
# Usage:
#   chmod +x extraction/complete_extraction.sh
#   ./extraction/complete_extraction.sh

set -e  # Exit on any error

EXPERIMENT="gemma_2b_cognitive_nov21"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "================================================================================"
echo "COMPLETE EXTRACTION PIPELINE: $EXPERIMENT"
echo "================================================================================"
echo "Target: 9 traits × 4 methods × 26 layers = 936 vectors total"
echo ""
echo "Current status:"
echo "  ✅ Complete: cognitive_state/confidence, cognitive_state/context"
echo "  🔄 Need extraction: 7 traits"
echo ""
echo "This script will:"
echo "  1. Generate responses for formality (Stage 1)"
echo "  2. Extract activations for 8 traits (Stage 2)"
echo "  3. Extract vectors for 8 traits in parallel (Stage 3)"
echo "================================================================================"
echo ""

# ============================================================================
# STAGE 1: Generate Responses
# ============================================================================

echo "================================================================================"
echo "STAGE 1: Generate Responses"
echo "================================================================================"
echo ""
echo "→ expression_style/formality"
echo ""

python extraction/1_generate_natural.py \
    --experiment "$EXPERIMENT" \
    --trait expression_style/formality

echo ""
echo "✅ Stage 1 complete"
echo ""

# ============================================================================
# STAGE 2: Extract Activations
# ============================================================================

echo "================================================================================"
echo "STAGE 2: Extract Activations (8 traits)"
echo "================================================================================"
echo ""

TRAITS_STAGE2=(
    "behavioral_tendency/defensiveness"
    "behavioral_tendency/retrieval"
    "cognitive_state/correction_impulse"
    "cognitive_state/pattern_completion"
    "cognitive_state/search_activation"
    "cognitive_state/uncertainty_expression"
    "expression_style/positivity"
    "expression_style/formality"
)

for trait in "${TRAITS_STAGE2[@]}"; do
    echo "→ $trait"
    python extraction/2_extract_activations.py \
        --experiment "$EXPERIMENT" \
        --trait "$trait"
    echo ""
done

echo "✅ Stage 2 complete"
echo ""

# ============================================================================
# STAGE 3: Extract Vectors (PARALLEL)
# ============================================================================

echo "================================================================================"
echo "STAGE 3: Extract Vectors (4 methods × 26 layers, PARALLEL)"
echo "================================================================================"
echo ""
echo "Running 8 extractions in parallel..."
echo ""

TRAITS_STAGE3=(
    "behavioral_tendency/defensiveness"
    "behavioral_tendency/retrieval"
    "cognitive_state/correction_impulse"
    "cognitive_state/pattern_completion"
    "cognitive_state/search_activation"
    "cognitive_state/uncertainty_expression"
    "expression_style/positivity"
    "expression_style/formality"
)

# Start all vector extractions in background
PIDS=()
for trait in "${TRAITS_STAGE3[@]}"; do
    echo "→ Starting: $trait (background)"
    python extraction/3_extract_vectors.py \
        --experiment "$EXPERIMENT" \
        --trait "$trait" \
        --methods mean_diff,probe,ica,gradient \
        > "logs/extraction_${trait//\//_}.log" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "Waiting for all 8 parallel extractions to complete..."
echo "(Monitor progress: tail -f logs/extraction_*.log)"
echo ""

# Wait for all background jobs and check exit codes
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    trait=${TRAITS_STAGE3[$i]}

    if wait $pid; then
        echo "  ✅ $trait"
    else
        echo "  ❌ $trait (FAILED - check logs/extraction_${trait//\//_}.log)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ Stage 3 complete - all extractions succeeded"
else
    echo "⚠️  Stage 3 completed with $FAILED failures"
    exit 1
fi

echo ""

# ============================================================================
# VERIFICATION
# ============================================================================

echo "================================================================================"
echo "VERIFICATION: Checking Coverage"
echo "================================================================================"
echo ""

python3 << 'EOF'
from pathlib import Path

experiment = "gemma_2b_cognitive_nov21"
exp_dir = Path(f"experiments/{experiment}/extraction")

total_traits = 0
complete_traits = 0
total_vectors = 0
target_per_trait = 104  # 4 methods × 26 layers

for category in sorted(exp_dir.iterdir()):
    if not category.is_dir():
        continue

    for trait_dir in sorted(category.iterdir()):
        if not trait_dir.is_dir():
            continue

        total_traits += 1
        trait_name = f"{category.name}/{trait_dir.name}"
        vectors_dir = trait_dir / "vectors"

        if vectors_dir.exists():
            vector_files = list(vectors_dir.glob("*.pt"))
            vector_count = len(vector_files)
            total_vectors += vector_count

            if vector_count >= target_per_trait:
                complete_traits += 1
                status = "✅"
            else:
                status = "🟡"

            print(f"  {status} {trait_name:50} {vector_count:3}/{target_per_trait} vectors")
        else:
            print(f"  ❌ {trait_name:50}   0/{target_per_trait} vectors")

print("")
print("="*80)
print(f"SUMMARY")
print("="*80)
print(f"Total traits:        {total_traits}/9")
print(f"Complete traits:     {complete_traits}/9")
print(f"Total vectors:       {total_vectors}/{total_traits * target_per_trait}")
print("")

if complete_traits == total_traits:
    print("🎉 SUCCESS: 100% coverage achieved!")
    print(f"   All {total_traits} traits have {target_per_trait} vectors (4 methods × 26 layers)")
else:
    print(f"⚠️  WARNING: {total_traits - complete_traits} traits incomplete")
    exit(1)
EOF

VERIFY_EXIT=$?

if [ $VERIFY_EXIT -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✅ EXTRACTION COMPLETE - 100% COVERAGE ACHIEVED"
    echo "================================================================================"
    echo ""
    echo "Results:"
    echo "  • 9 traits × 4 methods × 26 layers = 936 vectors"
    echo "  • All vector files saved to experiments/$EXPERIMENT/extraction/*/vectors/"
    echo ""
    echo "Next steps:"
    echo "  • View in visualization: python visualization/serve.py"
    echo "  • Run inference: python inference/monitor_dynamics.py"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo "⚠️  EXTRACTION INCOMPLETE"
    echo "================================================================================"
    echo ""
    echo "Some traits did not reach 100% coverage. Check logs for errors."
    exit 1
fi
