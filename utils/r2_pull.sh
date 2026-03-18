#!/bin/bash
# Pull experiments from R2 cloud storage
#
# Usage:
#   ./r2_pull.sh                                          Safe: only download new files (default)
#   ./r2_pull.sh --copy                                   Safe update: new + changed files, never deletes
#   ./r2_pull.sh --full                                   Full sync: make local match R2 (DELETES local-only files!)
#   ./r2_pull.sh --checksum                               Slow sync: MD5 comparison (DELETES local-only files!)
#   ./r2_pull.sh --only mats-emergent-misalignment        Scope to one experiment
#   ./r2_pull.sh --only mats-emergent-misalignment,aria_rl Scope to multiple experiments
#   ./r2_pull.sh --only viz_findings                       Sync completed findings experiments
#
# Flags:
#   --include-loras          Include LoRA checkpoints (finetune/, turner_loras/, etc.)
#   --include-trajectories   Include trajectory .pt files (large, regenerable)
#   --dry-run                Show what would be transferred without doing it
#
# Note: viz_findings/ is excluded by default. Use --only to sync it.
# Archive lives separately at r2:trait-interp-bucket/experiments_archive/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/r2_config.sh"

MODE="safe"
parse_r2_args "$@"
ensure_r2
resolve_paths
build_excludes
build_only_filters

# Display what we're doing
if [[ -n "$ONLY" ]]; then
    echo "Pulling experiment(s): $ONLY"
else
    echo "Pulling all experiments from R2..."
fi
[[ "$INCLUDE_LORAS" == true ]]        && echo "  + LoRAs included"
[[ "$INCLUDE_TRAJECTORIES" == true ]] && echo "  + Trajectories included"

COMMON_FLAGS=(
    --progress
    --stats 5s
    --fast-list
    $DRY_RUN
    "${EXCLUDES[@]}"
    "${ONLY_FILTERS[@]}"
)

case $MODE in
    safe)
        echo "Mode: SAFE (new files only, won't delete local files)"
        echo ""
        rclone copy "$R2_REMOTE" "$LOCAL_DIR" \
            --ignore-existing \
            --transfers 32 \
            --checkers 64 \
            "${COMMON_FLAGS[@]}"
        ;;
    copy)
        echo "Mode: COPY (new + changed files, never deletes local)"
        echo ""
        rclone copy "$R2_REMOTE" "$LOCAL_DIR" \
            --size-only \
            --transfers 16 \
            --checkers 32 \
            "${COMMON_FLAGS[@]}"
        ;;
    full)
        echo "Mode: FULL (size-only, deletes local files not in R2)"
        echo ""
        rclone sync "$R2_REMOTE" "$LOCAL_DIR" \
            --size-only \
            --modify-window 1s \
            --transfers 32 \
            --checkers 64 \
            "${COMMON_FLAGS[@]}"
        ;;
    checksum)
        echo "Mode: CHECKSUM (MD5 comparison - slow, deletes local files not in R2)"
        echo ""
        rclone sync "$R2_REMOTE" "$LOCAL_DIR" \
            --checksum \
            --transfers 16 \
            --checkers 16 \
            "${COMMON_FLAGS[@]}"
        ;;
esac

echo ""
echo "Pull complete!"
