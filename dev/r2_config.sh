#!/bin/bash
# Shared R2 sync configuration — sourced by r2_push.sh and r2_pull.sh
#
# Provides:
#   parse_r2_args "$@"    — sets MODE, INCLUDE_LORAS, etc.
#   build_excludes        — populates EXCLUDES array based on flags
#   ensure_r2             — checks rclone is configured for R2
#   R2_REMOTE, LOCAL_DIR  — resolved paths based on --only

# ─── Defaults ─────────────────────────────────────────────────────────────────

MODE=""  # set by caller before parse_r2_args
INCLUDE_LORAS=false
INCLUDE_TRAJECTORIES=false
DRY_RUN=""
ONLY=""  # comma-separated experiment names
ALL=false  # --all flag to opt into full-repo sync

# ─── Argument parsing ────────────────────────────────────────────────────────

parse_r2_args() {
    for arg in "$@"; do
        case "$arg" in
            # Sync modes (caller defines which are valid)
            --copy)     MODE="copy" ;;
            --full)     MODE="full" ;;
            --checksum) MODE="checksum" ;;
            --turbo)    MODE="turbo" ;;

            # Scope flags
            --all)  ALL=true ;;

            # Include flags
            --include-loras)        INCLUDE_LORAS=true ;;
            --include-trajectories) INCLUDE_TRAJECTORIES=true ;;

            # Utilities
            --dry-run) DRY_RUN="--dry-run" ;;

            # Experiment scoping
            --only)  ;; # next arg is the value, handled below
            --only=*) ONLY="${arg#--only=}" ;;

            # Catch --only VALUE (two-arg form)
            *)
                if [[ "${PREV_ARG:-}" == "--only" ]]; then
                    ONLY="$arg"
                fi
                ;;
        esac
        PREV_ARG="$arg"
    done
}

# ─── Exclude list builder ────────────────────────────────────────────────────

build_excludes() {
    EXCLUDES=()

    # ── Always exclude: junk ──
    EXCLUDES+=(
        --exclude "*.pyc"
        --exclude "**/__pycache__/**"
        --exclude ".DS_Store"
        --exclude "**/.DS_Store"
    )

    # ── Always exclude: regenerable data ──
    EXCLUDES+=(
        --exclude "**/activations/**"
        --exclude "**/inference/*/raw/**"
        --exclude "**/inference/raw/**"
    )

    # ── Always exclude: training artifacts ──
    EXCLUDES+=(
        --exclude "**/optimizer.pt"
        --exclude "**/scheduler.pt"
        --exclude "*.bin"
        --exclude "*.pth"
        --exclude "*.jinja"
        --exclude "**/.cache/**"
    )

    # ── Always exclude: redundant tokenizer copies in checkpoints ──
    EXCLUDES+=(
        --exclude "**/checkpoint-*/tokenizer.json"
        --exclude "**/checkpoint-*/vocab.json"
        --exclude "**/checkpoint-*/tokenizer_config.json"
        --exclude "**/checkpoint-*/special_tokens_map.json"
        --exclude "**/checkpoint-*/added_tokens.json"
    )

    # ── Heavy/completed experiments: excluded by default ──
    # Use --only <name> to sync these directly
    EXCLUDES+=(
        --exclude "viz_findings/**"
        --exclude "audit-bleachers/**"    # 10GB: 168 prompt sets × 57 variants
        --exclude "obfuscation-atlas/**/projections/**"  # 85GB: 42 layers × 113 traits × 874 prompts × 3 variants
    )

    # ── Archive: not synced (lives in separate R2 bucket path) ──
    # Archived experiments are at r2:trait-interp-bucket/experiments_archive/
    # not under experiments/. This exclude is a safety net.
    EXCLUDES+=(--exclude "archive/**")

    # ── LoRAs: excluded by default ──
    # Both rooted (finetune/**) and nested (**/finetune/**) patterns needed
    # because --only scopes the sync root to the experiment directory
    if [[ "$INCLUDE_LORAS" == false ]]; then
        EXCLUDES+=(
            --exclude "finetune/**"
            --exclude "**/finetune/**"
            --exclude "turner_loras/**"
            --exclude "**/turner_loras/**"
            --exclude "sriram_loras/**"
            --exclude "**/sriram_loras/**"
            --exclude "lora/**"
            --exclude "**/lora/**"
        )
    fi

    # ── Trajectories: excluded by default ──
    if [[ "$INCLUDE_TRAJECTORIES" == false ]]; then
        EXCLUDES+=(
            --exclude "*_trajectories.pt"
            --exclude "**/em_probe/**/data*.pt"
        )
    fi
}

# ─── R2 connection check ─────────────────────────────────────────────────────

ensure_r2() {
    local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    if ! rclone listremotes | grep -q "^r2:"; then
        echo "R2 not configured, running setup..."
        "$SCRIPT_DIR/setup_r2.sh"
    elif ! rclone lsd r2: &>/dev/null; then
        echo "R2 remote exists but credentials are invalid, re-running setup..."
        "$SCRIPT_DIR/setup_r2.sh"
    fi
}

# ─── Path resolution ─────────────────────────────────────────────────────────

resolve_paths() {
    # Require --only or --all (prevents accidental 210k-file full-repo scans)
    if [[ -z "$ONLY" && "$ALL" != true ]]; then
        echo "Error: specify --only <experiment> or --all"
        echo ""
        echo "Examples:"
        echo "  $0 --only live-chat                     Single experiment"
        echo "  $0 --only live-chat,starter              Multiple experiments"
        echo "  $0 --all                                 All experiments (slow)"
        exit 1
    fi

    # Base paths
    R2_REMOTE="r2:trait-interp-bucket/experiments/"
    LOCAL_DIR="experiments/"

    # --only scoping: if single experiment, scope both paths
    if [[ -n "$ONLY" ]]; then
        # For single experiment (no commas), scope the paths directly
        if [[ "$ONLY" != *","* ]]; then
            R2_REMOTE="r2:trait-interp-bucket/experiments/${ONLY}/"
            LOCAL_DIR="experiments/${ONLY}/"
        fi
        # For multiple experiments, we handle via --filter in the caller
    fi
}

# ─── Multi-experiment filter builder ──────────────────────────────────────────

build_only_filters() {
    # Returns filter args for multi-experiment --only
    # Only needed when ONLY contains commas (multiple experiments)
    ONLY_FILTERS=()
    if [[ -n "$ONLY" && "$ONLY" == *","* ]]; then
        # Include only specified experiments
        IFS=',' read -ra EXPERIMENTS <<< "$ONLY"
        for exp in "${EXPERIMENTS[@]}"; do
            ONLY_FILTERS+=(--filter "+ ${exp}/**")
        done
        # Exclude everything else at the top level
        ONLY_FILTERS+=(--filter "- *")
    fi
}
