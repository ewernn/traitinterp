#!/bin/bash
# LOCAL → R2 PUSH ONLY (never pull!)
# Local is the source of truth.
#
# Usage:
#   ./r2_push.sh            Fast: only upload new files (default)
#   ./r2_push.sh --full     Full: check sizes, catch size-changed overwrites
#   ./r2_push.sh --checksum Slow: check MD5, catch ALL overwrites

set -e

MODE="fast"
if [[ "$1" == "--full" ]]; then
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
  --exclude "**/inference/raw/**"
)

case $MODE in
  fast)
    echo "Mode: FAST (new files only)"
    echo ""
    rclone copy experiments/ r2:trait-interp-bucket/experiments/ \
      --progress \
      --stats 5s \
      --ignore-existing \
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
      --transfers 4 \
      --checkers 4 \
      "${EXCLUDES[@]}"
    ;;
esac

# What gets synced to R2:
#   ✅ Vectors (.pt in vectors/) - the extracted trait vectors
#   ✅ Responses (pos.json, neg.json)
#   ✅ Inference projections (residual_stream/*.json)
#   ✅ Metadata files
#   ❌ Extraction activations (activations/**/train_all_layers.pt, val_all_layers.pt) - huge, regenerable
#   ❌ Raw inference activations (inference/raw/*.pt) - huge, regenerable

echo ""
echo "✅ Push complete!"
