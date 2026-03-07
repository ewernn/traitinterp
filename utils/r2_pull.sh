#!/bin/bash
# Pull experiments from R2 cloud storage
#
# Usage:
#   ./r2_pull.sh                              Safe: only download new files (default)
#   ./r2_pull.sh --copy                       Safe update: new + changed files, never deletes local
#   ./r2_pull.sh --full                       Full sync: make local match R2 (DELETES local files not in R2!)
#   ./r2_pull.sh --checksum                   Slow sync: MD5 comparison (DELETES local files not in R2!)
#   ./r2_pull.sh --copy mats-emergent-misalignment   Scope to one experiment
#
# Add --checkpoints to include finetune checkpoints (adapter weights only, not training cruft)

set -e

MODE="safe"
CHECKPOINTS=false
EXPERIMENT=""
for arg in "$@"; do
    case "$arg" in
        --copy) MODE="copy" ;;
        --full) MODE="full" ;;
        --checksum) MODE="checksum" ;;
        --checkpoints) CHECKPOINTS=true ;;
        --*) ;;
        *) EXPERIMENT="$arg" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check if rclone is configured for R2 with valid credentials
if ! rclone listremotes | grep -q "^r2:"; then
    echo "⚙️  R2 not configured, running setup..."
    "$SCRIPT_DIR/setup_r2.sh"
elif ! rclone lsd r2: &>/dev/null; then
    echo "⚠️  R2 remote exists but credentials are invalid, re-running setup..."
    "$SCRIPT_DIR/setup_r2.sh"
fi

REMOTE="r2:trait-interp-bucket/experiments/"
LOCAL="experiments/"
if [[ -n "$EXPERIMENT" ]]; then
  REMOTE="r2:trait-interp-bucket/experiments/${EXPERIMENT}/"
  LOCAL="experiments/${EXPERIMENT}/"
  echo "📥 Pulling experiment: $EXPERIMENT"
else
  echo "📥 Pulling all experiments from R2..."
fi

EXCLUDES=(
  --exclude "*.pyc"
  --exclude "__pycache__/**"
  --exclude ".DS_Store"
  # Git-tracked types — git owns code/docs/configs, R2 owns data
  --exclude "*.py"
  --exclude "*.md"
  --exclude "*.txt"
  --exclude "*.log"
  --exclude "config.json"
  # Large regenerable files
  --exclude "**/activations/**"
  --exclude "**/inference/raw/**"
  --exclude "**/inference/*/raw/**"
  --exclude "**/results/*_activations.pt"
  --exclude "liars-bench/**"
  --exclude "audit-bleachers/**"
  --exclude "audit-bench/**"
  --exclude "temp/**"
  # LoRA training artifacts (not needed for inference)
  --exclude "*.bin"
  --exclude "*.pth"
  --exclude "*.jinja"
  --exclude "**/optimizer.pt"
  --exclude "**/scheduler.pt"
  # HF download cache (metadata from snapshot_download, not our data)
  --exclude "**/.cache/**"
  # Redundant tokenizer copies in checkpoints
  --exclude "**/checkpoint-*/tokenizer.json"
  --exclude "**/checkpoint-*/vocab.json"
  --exclude "**/checkpoint-*/tokenizer_config.json"
  --exclude "**/checkpoint-*/special_tokens_map.json"
  --exclude "**/checkpoint-*/added_tokens.json"
)

# Default: exclude finetune dirs. --checkpoints opts in (adapter weights only).
# Both **/X and /X patterns needed: **/X matches nested, /X matches root when scoped to one experiment.
if [[ "$CHECKPOINTS" == false ]]; then
  EXCLUDES+=(--exclude "**/finetune/**" --exclude "/finetune/**")
  EXCLUDES+=(--exclude "**/*loras/**"   --exclude "/*loras/**")
  EXCLUDES+=(--exclude "**/lora/**"     --exclude "/lora/**")
fi

case $MODE in
  safe)
    echo "Mode: SAFE (new files only, won't delete local files)"
    echo ""
    rclone copy "$REMOTE" "$LOCAL" \
      --progress \
      --stats 5s \
      --ignore-existing \
      --transfers 32 \
      --checkers 64 \
      "${EXCLUDES[@]}"
    ;;
  copy)
    echo "Mode: COPY (new + changed files, never deletes local)"
    echo ""
    rclone copy "$REMOTE" "$LOCAL" \
      --progress \
      --stats 5s \
      --size-only \
      --transfers 16 \
      --checkers 32 \
      "${EXCLUDES[@]}"
    ;;
  full)
    echo "Mode: FULL (size-only, deletes local files not in R2)"
    echo ""
    rclone sync "$REMOTE" "$LOCAL" \
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
    rclone sync "$REMOTE" "$LOCAL" \
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
