#!/bin/bash
# Stage 2-3 orchestrator: Extract vectors + Geometry analysis
# Run after Stage 1 (story generation) completes.
#
# Usage:
#   bash experiments/ant-emotion-concepts/scripts/run_stages_2_3.sh
#
# This script:
# 1. Runs extraction (stages 3+4) with --save-activations for all 171 emotions
# 2. Runs cross-trait normalization (grand mean subtraction + neutral-PC denoising)
# 3. Runs geometry analysis (cosine heatmap, k-means, PCA, RSA)

set -e

EXPERIMENT="ant-emotion-concepts"
CATEGORY="ant_emotion_concepts"
LOG_DIR="experiments/${EXPERIMENT}"

echo "=========================================="
echo "Stage 2: Vector Extraction"
echo "=========================================="
echo "$(date): Starting extraction..."

# Stage 2.1: Extract activations + vectors for all 171 emotions
# Uses extraction_config.yaml for position=response[50:], methods=mean_diff
# --save-activations needed for cross-trait normalization
python extraction/run_extraction_pipeline.py \
    --experiment ${EXPERIMENT} \
    --category ${CATEGORY} \
    --only-stage 3,4 \
    --load-in-4bit \
    --save-activations \
    2>&1 | tee ${LOG_DIR}/stage2_extraction.log

echo ""
echo "$(date): Extraction complete."

# Stage 2.2-2.3: Cross-trait normalization
# Note: neutral corpus not yet generated, so skip neutral-PC denoising for now
# Just do grand mean subtraction (the cross_trait_normalize.py handles missing neutral gracefully)
echo ""
echo "=========================================="
echo "Stage 2.2: Cross-trait normalization"
echo "=========================================="

# Default layer: ~2/3 of 80 layers = layer 53
# Also extract at a range for RSA analysis
python analysis/vectors/cross_trait_normalize.py \
    --experiment ${EXPERIMENT} \
    --layers 20,30,40,45,50,53,55,60 \
    --neutral-trait ${CATEGORY}/_neutral \
    2>&1 | tee ${LOG_DIR}/stage2_normalization.log

echo ""
echo "$(date): Normalization complete."

echo ""
echo "=========================================="
echo "Stage 3: Geometry Analysis"
echo "=========================================="

python experiments/${EXPERIMENT}/scripts/geometry_analysis.py \
    --experiment ${EXPERIMENT} \
    --layers 20,30,40,45,50,53,55,60 \
    --method mean_diff \
    --rsa \
    2>&1 | tee ${LOG_DIR}/stage3_geometry.log

echo ""
echo "$(date): Geometry analysis complete."
echo "=========================================="
echo "Results in: experiments/${EXPERIMENT}/results/geometry/"
echo "=========================================="
