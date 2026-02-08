#!/bin/bash
# LOCAL → R2 PUSH ONLY (never pull!)
# Local is the source of truth.
#
# Usage:
#   ./r2_push.sh            Fast: only upload new files (default)
#   ./r2_push.sh --copy     Safe update: new + changed files, never deletes
#   ./r2_push.sh --full     Full sync: make R2 match local (DELETES R2 files not in local!)
#   ./r2_push.sh --checksum Slow sync: MD5 comparison (DELETES R2 files not in local!)

set -e

MODE="fast"
if [[ "$1" == "--copy" ]]; then
    MODE="copy"
elif [[ "$1" == "--full" ]]; then
    MODE="full"
elif [[ "$1" == "--checksum" ]]; then
    MODE="checksum"
fi

echo "📤 Pushing experiments to R2..."
echo "Source: experiments/"
echo "Destination: r2:trait-interp-bucket/experiments/"

EXCLUDES=(
  --exclude "*.pyc"
  --exclude "__pycache__/**"
  --exclude ".DS_Store"
  --exclude "**/activations/**"
  --exclude "**/inference/*/raw/**"
  --exclude "**/results/*_activations.pt"
  --exclude "liars-bench/**"
  --exclude "**/lora/**/optimizer.pt"
  --exclude "**/lora/**/tokenizer.json"
  --exclude "**/lora/**/tokenizer_config.json"
  --exclude "**/lora/**/chat_template.jinja"
  --exclude "**/lora/**/special_tokens_map.json"
  --exclude "**/lora/**/checkpoint-*/**"
)

case $MODE in
  fast)
    echo "Mode: FAST (new files only)"
    echo ""
    rclone copy experiments/ r2:trait-interp-bucket/experiments/ \
      --progress \
      --stats 5s \
      --ignore-existing \
      --copy-links \
      --transfers 32 \
      --checkers 32 \
      "${EXCLUDES[@]}"
    ;;
  copy)
    echo "Mode: COPY (new + changed files, never deletes)"
    echo ""
    rclone copy experiments/ r2:trait-interp-bucket/experiments/ \
      --progress \
      --stats 5s \
      --size-only \
      --copy-links \
      --transfers 16 \
      --checkers 16 \
      "${EXCLUDES[@]}"
    ;;
  full)
    echo "Mode: FULL (size-only comparison)"
    echo ""
    rclone sync experiments/ r2:trait-interp-bucket/experiments/ \
      --progress \
      --stats 5s \
      --size-only \
      --copy-links \
      --local-no-check-updated \
      --transfers 8 \
      --checkers 8 \
      "${EXCLUDES[@]}"
    ;;
  checksum)
    echo "Mode: CHECKSUM (MD5 comparison - slow!)"
    echo ""
    rclone sync experiments/ r2:trait-interp-bucket/experiments/ \
      --progress \
      --stats 5s \
      --checksum \
      --copy-links \
      --transfers 4 \
      --checkers 4 \
      "${EXCLUDES[@]}"
    ;;
esac

# What gets synced to R2:
#   ✅ Vectors (.pt in vectors/) - the extracted trait vectors
#   ✅ Responses (pos.json, neg.json)
#   ✅ Inference projections, massive_activations JSONs
#   ✅ Metadata files
#   ✅ LoRA adapter weights (adapter_model.safetensors, adapter_config.json)
#   ❌ Extraction activations (activations/**/train_all_layers.pt, val_all_layers.pt) - huge, regenerable
#   ❌ Raw inference activations (inference/{variant}/raw/*.pt) - huge, regenerable
#   ❌ LoRA training artifacts (optimizer.pt, tokenizer.json, checkpoints) - huge, not needed for inference

echo ""
echo "✅ Push complete!"
