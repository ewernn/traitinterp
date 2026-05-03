#!/bin/bash
# Pull experiments from R2 cloud storage
#
# Requires --only <experiment> or --all to prevent accidental full-repo scans (100k+ R2 objects).
#
# Usage:
#   ./r2_pull.sh --only live-chat                         Safe: download new files in one experiment (default mode)
#   ./r2_pull.sh --only live-chat --copy                  New + changed files (size comparison), never deletes
#   ./r2_pull.sh --only live-chat --full                  Make local match R2 (DELETES local-only files!)
#   ./r2_pull.sh --only live-chat --checksum              MD5 comparison (DELETES local-only files, slow)
#   ./r2_pull.sh --only live-chat,starter,aria_rl         Multiple experiments
#   ./r2_pull.sh --only archive                           Pull archived experiments (excluded from --all)
#   ./r2_pull.sh --only archive/sleeper_detection         Pull a single archived experiment
#   ./r2_pull.sh --all                                    All experiments (slow — lists entire R2 bucket)
#   ./r2_pull.sh --all --full                             Full sync everything (nuclear)
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

MODE="safe"
parse_r2_args "$@"
ensure_r2
resolve_paths
build_excludes

# Remember the user's PACKED preference; it's toggled per-experiment below.
PACKED_REQUESTED="$PACKED"

# Display what we're doing
if [[ -n "$ONLY" ]]; then
    echo "Pulling experiment(s): $ONLY"
else
    echo "Pulling all experiments from R2..."
fi
[[ "$INCLUDE_LORAS" == true ]]        && echo "  + LoRAs included"
[[ "$INCLUDE_TRAJECTORIES" == true ]] && echo "  + Trajectories included"
[[ "$PACKED" == true ]]               && echo "  + PACKED mode (bundled projections)"

run_one_pull() {
    local r2_remote="$1" local_dir="$2"
    echo ""
    echo "=== $r2_remote → $local_dir ==="

    # Per-experiment PACKED detection — scoped to this prefix, not the whole bucket.
    local packed="$PACKED_REQUESTED"
    if [[ "$packed" == true ]]; then
        local bundle_count
        bundle_count=$(rclone lsf "$r2_remote" -R --include "**/*.tar.zst" 2>/dev/null | head -1 | wc -l)
        if [[ "$bundle_count" -eq 0 ]]; then
            echo "  [packed] no bundles at $r2_remote — falling back to scattered pull"
            packed=false
        fi
    fi

    # Excludes flip on PACKED state, so rebuild per-iteration.
    local saved_packed="$PACKED"
    PACKED="$packed"
    build_excludes
    PACKED="$saved_packed"

    local common_flags=(
        --progress
        --stats 5s
        --fast-list
        $DRY_RUN
        "${EXCLUDES[@]}"
    )

    case $MODE in
        safe)
            echo "Mode: SAFE (new files only, won't delete local files)"
            rclone copy "$r2_remote" "$local_dir" \
                --ignore-existing --transfers 32 --checkers 64 "${common_flags[@]}"
            ;;
        copy)
            echo "Mode: COPY (new + changed files, never deletes local)"
            rclone copy "$r2_remote" "$local_dir" \
                --size-only --transfers 16 --checkers 32 "${common_flags[@]}"
            ;;
        full)
            echo "Mode: FULL (size-only, deletes local files not in R2)"
            rclone sync "$r2_remote" "$local_dir" \
                --size-only --modify-window 1s --transfers 32 --checkers 64 "${common_flags[@]}"
            ;;
        checksum)
            echo "Mode: CHECKSUM (MD5 comparison - slow, deletes local files not in R2)"
            rclone sync "$r2_remote" "$local_dir" \
                --checksum --transfers 16 --checkers 16 "${common_flags[@]}"
            ;;
    esac

    if [[ "$packed" == true ]]; then
        echo "[packed] Unpacking bundles in $local_dir..."
        python3 "$SCRIPT_DIR/projection_bundles.py" unpack "$local_dir" --workers 16
        echo "[packed] Unpack complete."
    fi
}

for pair in "${PATH_PAIRS[@]}"; do
    split_pair "$pair"
    run_one_pull "$R2_REMOTE" "$LOCAL_DIR"
done

echo ""
echo "Pull complete!"
