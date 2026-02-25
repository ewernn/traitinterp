#!/bin/bash
# LOCAL → R2 PUSH ONLY (never pull!)
# Local is the source of truth.
#
# Usage:
#   ./r2_push.sh            Fast: only upload new files (default)
#   ./r2_push.sh --copy     Safe update: new + changed files, never deletes
#   ./r2_push.sh --full     Full sync: make R2 match local (DELETES R2 files not in local!)
#   ./r2_push.sh --checksum Slow sync: MD5 comparison (DELETES R2 files not in local!)
#   ./r2_push.sh --turbo   Max parallelism: 256 transfers, fast-list (for many small files)
#
# Add --checkpoints to include finetune checkpoints (adapter weights only, not training cruft)

set -e

MODE="fast"
CHECKPOINTS=false
for arg in "$@"; do
    case "$arg" in
        --copy) MODE="copy" ;;
        --full) MODE="full" ;;
        --checksum) MODE="checksum" ;;
        --turbo) MODE="turbo" ;;
        --checkpoints) CHECKPOINTS=true ;;
    esac
done

echo "📤 Pushing experiments to R2..."
echo "Source: experiments/"
echo "Destination: r2:trait-interp-bucket/experiments/"

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
  --exclude "**/inference/*/raw/**"
  --exclude "**/results/*_activations.pt"
  --exclude "liars-bench/**"
  --exclude "audit-bleachers/**"
  --exclude "audit-bench/**"
  --exclude "temp/**"
  --exclude "_validate/**"
  --exclude "**/inference/raw/**"
  # LoRA training artifacts (not needed for inference)
  --exclude "*.bin"
  --exclude "*.pth"
  --exclude "*.jinja"
  --exclude "**/optimizer.pt"
  --exclude "**/scheduler.pt"
  # Redundant tokenizer copies in checkpoints
  --exclude "**/checkpoint-*/tokenizer.json"
  --exclude "**/checkpoint-*/vocab.json"
  --exclude "**/checkpoint-*/tokenizer_config.json"
  --exclude "**/checkpoint-*/special_tokens_map.json"
  --exclude "**/checkpoint-*/added_tokens.json"
)

# Default: exclude finetune dirs. --checkpoints opts in (adapter weights only).
if [[ "$CHECKPOINTS" == false ]]; then
  EXCLUDES+=(--exclude "**/finetune/**")
fi

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
  turbo)
    echo "Mode: TURBO (max parallelism, new files only)"
    echo ""
    rclone copy experiments/ r2:trait-interp-bucket/experiments/ \
      --progress \
      --stats 5s \
      --ignore-existing \
      --copy-links \
      --transfers 256 \
      --checkers 128 \
      --fast-list \
      --retries 3 \
      "${EXCLUDES[@]}"
    ;;
esac

# Ownership split — git owns code/docs/configs, R2 owns data:
#   R2 ✅  Vectors (.pt in vectors/)
#   R2 ✅  Responses (pos.json, neg.json)
#   R2 ✅  Inference projections, massive_activations JSONs
#   R2 ✅  Metadata files (.json except config.json)
#   R2 ✅  LoRA final adapter weights (adapter_model.safetensors, adapter_config.json)
#   R2 ✅  LoRA checkpoint adapters (--checkpoints flag)
#   Git ✅ Scripts (.py), docs (.md, .txt), config.json, logs
#   ❌    Extraction activations (huge, regenerable)
#   ❌    Raw inference activations (huge, regenerable)
#   ❌    LoRA training artifacts (optimizer.pt, scheduler.pt, *.bin, *.pth, tokenizer files)

echo ""
echo "✅ Push complete!"
