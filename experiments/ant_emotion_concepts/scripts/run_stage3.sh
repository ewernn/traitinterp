#!/bin/bash
# Stage 3: Geometry analysis (Figs 5-9)
python analysis/vectors/geometry.py \
    --experiment ant_emotion_concepts \
    --category ant_emotion_concepts \
    --layer 53 \
    --method "mean_diff+gm+pc50" \
    --baselines-json datasets/inference/ant_emotion_concepts/anthropic_baselines.json \
    --norms-file datasets/inference/ant_emotion_concepts/russell_mehrabian_norms.json \
    "$@"
