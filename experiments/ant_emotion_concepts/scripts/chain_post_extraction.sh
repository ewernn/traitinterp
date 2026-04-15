#!/bin/bash
# Post-extraction chainer: after 14-layer extraction finishes, run:
#   1. Generate neutral corpus (Stage 1.1 on _neutral)
#   2. Extract neutral activations at same 14 layers
#   3. Run cross_trait_normalize.py → produces mean_diff+gm and mean_diff+gm+pc50
#   4. Re-run Stage 3 geometry with denoised vectors
#   5. Re-run Phase 2 "He feels" comparison with denoised vectors
#
# Usage:
#   EXTRACTION_PID=224265 nohup bash .../chain_post_extraction.sh > log 2>&1 &

set -e
cd "$(dirname "$0")/../../.."

EXPERIMENT="ant_emotion_concepts"
CATEGORY="ant_emotion_concepts"
NEUTRAL_TRAIT="ant_emotion_concepts/_neutral"
LAYERS="1,7,13,19,25,31,37,43,49,55,61,67,73,79"
SCRIPTS="experiments/${EXPERIMENT}/scripts"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# Step 0: Wait for extraction PID
echo "[$(timestamp)] Waiting for extraction PID ${EXTRACTION_PID}..."
while kill -0 ${EXTRACTION_PID} 2>/dev/null; do
    N=$(find experiments/${EXPERIMENT}/extraction/ -path "*/instruct/vectors/response_50_/residual/mean_diff/layer1.pt" 2>/dev/null | wc -l)
    echo "[$(timestamp)] Extraction: ${N}/171 traits"
    sleep 180
done

# Verify extraction success
N=$(find experiments/${EXPERIMENT}/extraction/ -path "*/instruct/vectors/response_50_/residual/mean_diff/layer1.pt" 2>/dev/null | wc -l)
echo "[$(timestamp)] Extraction finished: ${N}/171 traits"
if [ "${N}" -lt 150 ]; then
    echo "ERROR: Only ${N}/171 traits have L1 vectors. Aborting."
    exit 1
fi

# Step 1: Generate neutral corpus (Stage 1.1 on _neutral)
echo ""
echo "[$(timestamp)] === Step 1: Generating neutral corpus ==="
python extraction/run_extraction_pipeline.py \
    --experiment ${EXPERIMENT} \
    --traits ${NEUTRAL_TRAIT} \
    --only-stage 1 \
    --load-in-4bit \
    --seed 42 \
    2>&1 | tee experiments/${EXPERIMENT}/stage1_neutral_generation.log

# Verify we got responses
if [ ! -f "experiments/${EXPERIMENT}/extraction/${NEUTRAL_TRAIT}/instruct/responses/pos.json" ]; then
    echo "ERROR: Neutral corpus generation failed (no pos.json). Aborting."
    exit 1
fi
echo "[$(timestamp)] Neutral generation complete"

# Step 2: Extract neutral activations at 14 layers
echo ""
echo "[$(timestamp)] === Step 2: Extracting neutral activations ==="
python extraction/run_extraction_pipeline.py \
    --experiment ${EXPERIMENT} \
    --traits ${NEUTRAL_TRAIT} \
    --only-stage 3,4 \
    --load-in-4bit \
    --save-activations \
    --force \
    --layers "${LAYERS}" \
    2>&1 | tee experiments/${EXPERIMENT}/stage2_neutral_extraction.log
echo "[$(timestamp)] Neutral extraction complete"

# Step 3: Cross-trait normalization (grand mean + PC denoising)
echo ""
echo "[$(timestamp)] === Step 3: Cross-trait normalization ==="
python analysis/vectors/cross_trait_normalize.py \
    --experiment ${EXPERIMENT} \
    --layers "${LAYERS}" \
    --neutral-trait ${NEUTRAL_TRAIT} \
    --source-method mean_diff \
    --variance-threshold 0.5 \
    2>&1 | tee experiments/${EXPERIMENT}/stage2_normalization_v3.log
echo "[$(timestamp)] Normalization complete"

# Verify we produced denoised vectors
N_DENOISED=$(find experiments/${EXPERIMENT}/extraction/ -path "*/mean_diff+gm+pc50/layer49.pt" 2>/dev/null | wc -l)
echo "[$(timestamp)] Denoised vectors at L49: ${N_DENOISED}/171"
if [ "${N_DENOISED}" -lt 150 ]; then
    echo "ERROR: Only ${N_DENOISED} denoised vectors produced. Aborting."
    exit 1
fi

# Step 4: Re-run Stage 3 geometry at a central layer
echo ""
echo "[$(timestamp)] === Step 4: Stage 3 geometry (with denoised vectors) ==="
python analysis/vectors/geometry.py \
    --experiment ${EXPERIMENT} \
    --layer 49 \
    --method mean_diff+gm+pc50 \
    --rsa-layers "${LAYERS}" \
    2>&1 | tee experiments/${EXPERIMENT}/stage3_geometry_v3.log
echo "[$(timestamp)] Stage 3 geometry complete"

echo ""
echo "[$(timestamp)] === CHAIN COMPLETE ==="
echo "Next manual steps:"
echo "  - Compute PC1 vs valence r on new denoised vectors (should match or exceed 0.965)"
echo "  - Re-run Phase 2 'He feels' comparison with mean_diff+gm+pc50"
echo "  - Re-run Stage 7 RH sweep with multi-layer steering"
