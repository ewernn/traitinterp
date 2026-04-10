#!/bin/bash
# Push experiments to R2 cloud storage (local → R2, never pulls)
#
# Requires --only <experiment> or --all to prevent accidental full-repo scans (210k+ files).
#
# Usage:
#   ./r2_push.sh --only live-chat                         Fast: upload new files in one experiment (default mode)
#   ./r2_push.sh --only live-chat --copy                  New + changed files (size comparison), never deletes
#   ./r2_push.sh --only live-chat --full                  Make R2 match local (DELETES R2-only files!)
#   ./r2_push.sh --only live-chat --checksum              MD5 comparison (DELETES R2-only files, slow)
#   ./r2_push.sh --only live-chat,starter                 Multiple experiments
#   ./r2_push.sh --all                                    All experiments (slow — walks entire experiments/)
#   ./r2_push.sh --all --full                             Full sync everything (nuclear)
#   ./r2_push.sh --only live-chat --turbo                 Max parallelism (256 transfers)
#
# Flags:
#   --include-loras          Include LoRA checkpoints (finetune/, turner_loras/, etc.)
#   --include-trajectories   Include trajectory .pt files (large, regenerable)
#   --dry-run                Show what would be transferred without doing it
#
# Note: viz_findings/ is excluded by default. Use --only viz_findings to sync it.
# Archive lives separately at r2:trait-interp-bucket/experiments_archive/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/r2_config.sh"

MODE="fast"
parse_r2_args "$@"
ensure_r2
resolve_paths
build_excludes
build_only_filters

# Display what we're doing
echo "Pushing experiments to R2..."
echo "Source: $LOCAL_DIR"
echo "Destination: $R2_REMOTE"
[[ "$INCLUDE_LORAS" == true ]]        && echo "  + LoRAs included"
[[ "$INCLUDE_TRAJECTORIES" == true ]] && echo "  + Trajectories included"

COMMON_FLAGS=(
    --progress
    --stats 5s
    --fast-list
    --skip-links
    $DRY_RUN
    "${EXCLUDES[@]}"
    "${ONLY_FILTERS[@]}"
)

case $MODE in
    fast)
        echo "Mode: FAST (new files only)"
        echo ""
        rclone copy "$LOCAL_DIR" "$R2_REMOTE" \
            --ignore-existing \
            --no-traverse \
            --transfers 32 \
            --checkers 32 \
            "${COMMON_FLAGS[@]}"
        ;;
    copy)
        echo "Mode: COPY (new + changed files, never deletes)"
        echo ""
        rclone copy "$LOCAL_DIR" "$R2_REMOTE" \
            --size-only \
            --transfers 16 \
            --checkers 16 \
            "${COMMON_FLAGS[@]}"
        ;;
    full)
        echo "Mode: FULL (size-only comparison, deletes R2 files not in local)"
        echo ""
        rclone sync "$LOCAL_DIR" "$R2_REMOTE" \
            --size-only \
            --skip-links \
            --local-no-check-updated \
            --transfers 8 \
            --checkers 8 \
            "${COMMON_FLAGS[@]}"
        ;;
    checksum)
        echo "Mode: CHECKSUM (MD5 comparison - slow!)"
        echo ""
        rclone sync "$LOCAL_DIR" "$R2_REMOTE" \
            --checksum \
            --transfers 4 \
            --checkers 4 \
            "${COMMON_FLAGS[@]}"
        ;;
    turbo)
        echo "Mode: TURBO (max parallelism, new files only)"
        echo ""
        rclone copy "$LOCAL_DIR" "$R2_REMOTE" \
            --ignore-existing \
            --transfers 256 \
            --checkers 128 \
            --retries 3 \
            "${COMMON_FLAGS[@]}"
        ;;
esac

echo ""
echo "Push complete!"
