#!/bin/bash
# Watch for Stage 1 completion, then auto-run Stages 2-5
# Usage: nohup bash experiments/ant-emotion-concepts/scripts/watch_and_continue.sh &

GENERATION_PID=100519
EXPERIMENT="ant_emotion_concepts"
CATEGORY="ant_emotion_concepts"
SCRIPT_DIR="experiments/${EXPERIMENT}/scripts"
LOG_DIR="experiments/${EXPERIMENT}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(timestamp)] Watching PID ${GENERATION_PID} for completion..."

# Wait for generation to finish
while kill -0 ${GENERATION_PID} 2>/dev/null; do
    N=$(find experiments/${EXPERIMENT}/extraction/ -name "pos.json" 2>/dev/null | wc -l)
    echo "[$(timestamp)] ${N}/171 traits generated. Waiting..."
    sleep 600  # Check every 10 min
done

echo "[$(timestamp)] Generation process finished!"
N=$(find experiments/${EXPERIMENT}/extraction/ -name "pos.json" 2>/dev/null | wc -l)
echo "Final count: ${N}/171 traits"

if [ "${N}" -lt 100 ]; then
    echo "ERROR: Only ${N} traits generated. Something went wrong. Aborting."
    exit 1
fi

# ---- Stage 2: Extract vectors ----
echo ""
echo "[$(timestamp)] Stage 2.1: Extracting vectors with --save-activations..."
python extraction/run_extraction_pipeline.py \
    --experiment ${EXPERIMENT} \
    --category ${CATEGORY} \
    --only-stage 3,4 \
    --load-in-4bit \
    --save-activations \
    --layers 20,30,40,45,50,53,55,60 \
    2>&1 | tee ${LOG_DIR}/stage2_extraction.log
echo "[$(timestamp)] Stage 2.1 complete."

# ---- Stage 2.2: Cross-trait normalization ----
echo ""
echo "[$(timestamp)] Stage 2.2: Cross-trait normalization..."
python analysis/vectors/cross_trait_normalize.py \
    --experiment ${EXPERIMENT} \
    --layers 20,30,40,45,50,53,55,60 \
    --neutral-trait ${CATEGORY}/_neutral \
    2>&1 | tee ${LOG_DIR}/stage2_normalization.log
echo "[$(timestamp)] Stage 2.2 complete."

# ---- Stage 3: Geometry analysis (user's script) ----
echo ""
echo "[$(timestamp)] Stage 3: Geometry analysis..."
python ${SCRIPT_DIR}/stage3_geometry.py \
    --experiment ${EXPERIMENT} \
    --layer 53 \
    --rsa-layers 20,30,40,45,50,53,55,60 \
    2>&1 | tee ${LOG_DIR}/stage3_geometry.log
echo "[$(timestamp)] Stage 3 complete."

# ---- Stage 4: Validation ----
echo ""
echo "[$(timestamp)] Stage 4: Validation experiments..."
python ${SCRIPT_DIR}/stage4_validation.py \
    --experiment ${EXPERIMENT} \
    --layer 53 \
    --load-in-4bit \
    2>&1 | tee ${LOG_DIR}/stage4_validation.log
echo "[$(timestamp)] Stage 4 complete."

# ---- Stage 5: Layer dynamics ----
echo ""
echo "[$(timestamp)] Stage 5: Layer dynamics..."
python ${SCRIPT_DIR}/stage5_layer_dynamics.py \
    --experiment ${EXPERIMENT} \
    --layer 53 \
    --load-in-4bit \
    2>&1 | tee ${LOG_DIR}/stage5_layer_dynamics.log
echo "[$(timestamp)] Stage 5 complete."

echo ""
echo "[$(timestamp)] === Stages 2-5 complete ==="
echo "Results in: experiments/${EXPERIMENT}/results/"
