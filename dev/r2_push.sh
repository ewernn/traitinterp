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

PACKED_REQUESTED="$PACKED"

echo "Pushing experiments to R2..."
[[ "$INCLUDE_LORAS" == true ]]        && echo "  + LoRAs included"
[[ "$INCLUDE_TRAJECTORIES" == true ]] && echo "  + Trajectories included"
[[ "$PACKED" == true ]]               && echo "  + PACKED mode (bundled projections)"

run_one_push() {
    local local_dir="$1" r2_remote="$2"
    echo ""
    echo "=== $local_dir → $r2_remote ==="

    # Per-experiment PACKED detection — scoped to this local dir.
    local packed="$PACKED_REQUESTED"
    if [[ "$packed" == true ]]; then
        local proj_count
        proj_count=$(find "$local_dir" -path "*/inference/*/projections/*" -name "*.json" 2>/dev/null | head -1 | wc -l)
        if [[ "$proj_count" -eq 0 ]]; then
            echo "  [packed] no projection JSONs under $local_dir — skipping pack"
            packed=false
        else
            echo "[packed] Packing projections in $local_dir..."
            python3 "$SCRIPT_DIR/projection_bundles.py" pack "$local_dir" --workers 16
            echo "[packed] Pack complete; continuing with push."
        fi
    fi

    local saved_packed="$PACKED"
    PACKED="$packed"
    build_excludes
    PACKED="$saved_packed"

    local common_flags=(
        --progress
        --stats 5s
        --fast-list
        --skip-links
        $DRY_RUN
        "${EXCLUDES[@]}"
    )

    case $MODE in
        fast)
            echo "Mode: FAST (new files only)"
            rclone copy "$local_dir" "$r2_remote" \
                --ignore-existing --no-traverse --transfers 32 --checkers 32 "${common_flags[@]}"
            ;;
        copy)
            echo "Mode: COPY (new + changed files, never deletes)"
            rclone copy "$local_dir" "$r2_remote" \
                --size-only --transfers 16 --checkers 16 "${common_flags[@]}"
            ;;
        full)
            echo "Mode: FULL (size-only comparison, deletes R2 files not in local)"
            rclone sync "$local_dir" "$r2_remote" \
                --size-only --local-no-check-updated --transfers 8 --checkers 8 "${common_flags[@]}"
            ;;
        checksum)
            echo "Mode: CHECKSUM (MD5 comparison - slow!)"
            rclone sync "$local_dir" "$r2_remote" \
                --checksum --transfers 4 --checkers 4 "${common_flags[@]}"
            ;;
        turbo)
            echo "Mode: TURBO (max parallelism, new files only)"
            rclone copy "$local_dir" "$r2_remote" \
                --ignore-existing --transfers 256 --checkers 128 "${common_flags[@]}"
            ;;
    esac
}

for pair in "${PATH_PAIRS[@]}"; do
    split_pair "$pair"
    run_one_push "$LOCAL_DIR" "$R2_REMOTE"
done

echo ""
echo "Push complete!"
