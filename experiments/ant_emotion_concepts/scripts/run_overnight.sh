#!/bin/bash
# Overnight orchestrator for Emotion Concepts replication
# Chains: Generation → Extraction → Normalization → Geometry → Validation
#
# Usage:
#   nohup bash experiments/ant-emotion-concepts/scripts/run_overnight.sh \
#     > experiments/ant-emotion-concepts/overnight.log 2>&1 &
#
# If generation is already running, start from a specific stage:
#   bash experiments/ant-emotion-concepts/scripts/run_overnight.sh --from-stage 2

set -e

EXPERIMENT="ant-emotion-concepts"
CATEGORY="ant_emotion_concepts"
LOG_DIR="experiments/${EXPERIMENT}"
FROM_STAGE=${1:-1}

timestamp() { date "+%Y-%m-%d %H:%M:%S PST"; }

if [ "$FROM_STAGE" = "--from-stage" ]; then
    FROM_STAGE=$2
fi

echo "=========================================="
echo "Emotion Concepts Replication — Overnight Run"
echo "Started: $(timestamp)"
echo "From stage: ${FROM_STAGE}"
echo "=========================================="

# ---- Stage 1: Story Generation ----
if [ "${FROM_STAGE}" -le 1 ]; then
    echo ""
    echo "[$(timestamp)] Stage 1: Generating stories for 171 emotions..."
    python extraction/run_extraction_pipeline.py \
        --experiment ${EXPERIMENT} \
        --category ${CATEGORY} \
        --only-stage 1 \
        --load-in-4bit \
        --seed 42 \
        2>&1 | tee ${LOG_DIR}/stage1_generation.log
    echo "[$(timestamp)] Stage 1 complete."
fi

# Verify Stage 1 output
N_TRAITS=$(find experiments/${EXPERIMENT}/extraction/ -name "pos.json" 2>/dev/null | wc -l)
echo "[$(timestamp)] Stage 1 verification: ${N_TRAITS}/171 traits have responses"
if [ "${N_TRAITS}" -lt 160 ]; then
    echo "WARNING: Only ${N_TRAITS} traits completed. Expected ~171."
fi

# ---- Stage 2: Extraction ----
if [ "${FROM_STAGE}" -le 2 ]; then
    echo ""
    echo "[$(timestamp)] Stage 2: Extracting vectors..."
    python extraction/run_extraction_pipeline.py \
        --experiment ${EXPERIMENT} \
        --category ${CATEGORY} \
        --only-stage 3,4 \
        --load-in-4bit \
        --save-activations \
        --layers 20,30,40,45,50,53,55,60 \
        2>&1 | tee ${LOG_DIR}/stage2_extraction.log
    echo "[$(timestamp)] Stage 2.1 (extraction) complete."

    # Cross-trait normalization
    echo ""
    echo "[$(timestamp)] Stage 2.2: Cross-trait normalization..."
    python analysis/vectors/cross_trait_normalize.py \
        --experiment ${EXPERIMENT} \
        --layers 20,30,40,45,50,53,55,60 \
        --neutral-trait ${CATEGORY}/_neutral \
        2>&1 | tee ${LOG_DIR}/stage2_normalization.log
    echo "[$(timestamp)] Stage 2.2 complete."
fi

# ---- Stage 3: Geometry Analysis ----
if [ "${FROM_STAGE}" -le 3 ]; then
    echo ""
    echo "[$(timestamp)] Stage 3: Geometry analysis..."
    python experiments/${EXPERIMENT}/scripts/geometry_analysis.py \
        --experiment ${EXPERIMENT} \
        --layers 20,30,40,45,50,53,55,60 \
        --method mean_diff \
        --rsa \
        2>&1 | tee ${LOG_DIR}/stage3_geometry.log
    echo "[$(timestamp)] Stage 3 complete."
fi

# ---- Stage 4: Logit Lens ----
if [ "${FROM_STAGE}" -le 4 ]; then
    echo ""
    echo "[$(timestamp)] Stage 4.1: Logit lens analysis..."
    python experiments/${EXPERIMENT}/scripts/logit_lens.py \
        --experiment ${EXPERIMENT} \
        --layer 53 \
        --method mean_diff \
        --load-in-4bit \
        2>&1 | tee ${LOG_DIR}/stage4_logit_lens.log
    echo "[$(timestamp)] Stage 4.1 complete."
fi

echo ""
echo "=========================================="
echo "Overnight run complete: $(timestamp)"
echo "=========================================="
echo ""
echo "Results:"
echo "  Responses: experiments/${EXPERIMENT}/extraction/"
echo "  Vectors: experiments/${EXPERIMENT}/extraction/*/instruct/vectors/"
echo "  Geometry: experiments/${EXPERIMENT}/results/geometry/"
echo "  Logit lens: experiments/${EXPERIMENT}/results/logit_lens/"
