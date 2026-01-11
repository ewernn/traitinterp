#!/bin/bash
# Pull experiments from R2 cloud storage
#
# Usage:
#   ./r2_pull.sh            Safe: only download new files (default)
#   ./r2_pull.sh --full     Full: check sizes, delete local files not in R2
#   ./r2_pull.sh --checksum Slow: check MD5, delete local files not in R2

set -e

MODE="safe"
if [[ "$1" == "--full" ]]; then
    MODE="full"
elif [[ "$1" == "--checksum" ]]; then
    MODE="checksum"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check if rclone is configured for R2
if ! rclone listremotes | grep -q "^r2:"; then
    echo "⚙️  R2 not configured, running setup..."
    "$SCRIPT_DIR/setup_r2.sh"
fi

echo "📥 Pulling experiments from R2..."

EXCLUDES=(
  --exclude "*.pyc"
  --exclude "__pycache__/**"
  --exclude ".DS_Store"
  --exclude "**/activations/**"
  --exclude "**/inference/raw/**"
)

case $MODE in
  safe)
    echo "Mode: SAFE (new files only, won't delete local files)"
    echo ""
    rclone copy r2:trait-interp-bucket/experiments/ experiments/ \
      --progress \
      --stats 5s \
      --ignore-existing \
      --transfers 32 \
      --checkers 64 \
      "${EXCLUDES[@]}"
    ;;
  full)
    echo "Mode: FULL (size-only, deletes local files not in R2)"
    echo ""
    rclone sync r2:trait-interp-bucket/experiments/ experiments/ \
      --progress \
      --stats 5s \
      --size-only \
      --modify-window 1s \
      --transfers 32 \
      --checkers 64 \
      "${EXCLUDES[@]}"
    ;;
  checksum)
    echo "Mode: CHECKSUM (MD5 comparison - slow, deletes local files not in R2)"
    echo ""
    rclone sync r2:trait-interp-bucket/experiments/ experiments/ \
      --progress \
      --stats 5s \
      --checksum \
      --transfers 16 \
      --checkers 16 \
      "${EXCLUDES[@]}"
    ;;
esac

echo ""
echo "✅ Pull complete!"
