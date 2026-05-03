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
PACKED=true  # default on: bundle projections into .tar.zst for transport (use --no-packed to disable)

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
            --packed)  PACKED=true ;;
            --no-packed) PACKED=false ;;

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
#
# rclone's **/ prefix doesn't match at root level when --only scopes paths to
# experiments/{name}/, so we need both rooted and **/-prefixed patterns. Use
# the helpers below so you only write the name once.

# Exclude a directory at any depth (e.g. "activations" → activations/** + **/activations/**)
exclude_dir() {
    EXCLUDES+=(--exclude "$1/**" --exclude "**/$1/**")
}

# Exclude a filename at any depth (e.g. ".DS_Store" → .DS_Store + **/.DS_Store)
exclude_file() {
    EXCLUDES+=(--exclude "$1" --exclude "**/$1")
}

build_excludes() {
    EXCLUDES=()

    # ── Always exclude: junk ──
    EXCLUDES+=(--exclude "*.pyc")
    exclude_dir "__pycache__"
    exclude_file ".DS_Store"

    # ── Always exclude: regenerable data ──
    exclude_dir "activations"
    exclude_dir "inference/*/raw"
    exclude_dir "inference/raw"

    # ── Always exclude: training artifacts ──
    exclude_file "optimizer.pt"
    exclude_file "scheduler.pt"
    EXCLUDES+=(--exclude "*.bin" --exclude "*.pth" --exclude "*.jinja")
    exclude_dir ".cache"

    # ── Always exclude: redundant tokenizer copies in checkpoints ──
    # (always nested inside finetune/checkpoint-*/, so **/ prefix works)
    EXCLUDES+=(
        --exclude "**/checkpoint-*/tokenizer.json"
        --exclude "**/checkpoint-*/vocab.json"
        --exclude "**/checkpoint-*/tokenizer_config.json"
        --exclude "**/checkpoint-*/special_tokens_map.json"
        --exclude "**/checkpoint-*/added_tokens.json"
    )

    # ── Heavy/completed experiments: excluded by default (use --only to sync) ──
    # These are top-level experiment dirs — only rooted pattern needed.
    EXCLUDES+=(
        --exclude "viz_findings/**"
        --exclude "audit-bench/**"        # 10GB: 168 prompt sets × 57 variants
        --exclude "obfuscation-atlas/**/projections/**"  # 85GB: 42 layers × 113 traits × 874 prompts × 3 variants
        --exclude "archive/**"  # archived experiments at experiments/archive/
    )

    # ── LoRAs: excluded by default ──
    if [[ "$INCLUDE_LORAS" == false ]]; then
        exclude_dir "finetune"
        exclude_dir "turner_loras"
        exclude_dir "sriram_loras"
        exclude_dir "lora"
    fi

    # ── Trajectories: excluded by default ──
    if [[ "$INCLUDE_TRAJECTORIES" == false ]]; then
        EXCLUDES+=(
            --exclude "*_trajectories.pt"
            --exclude "**/em_probe/**/data*.pt"
        )
    fi

    # ── Packed mode: transport only .tar.zst bundles under projections/, ──
    # ── skip the scattered per-prompt-set JSONs.                          ──
    if [[ "$PACKED" == true ]]; then
        EXCLUDES+=(
            --exclude "**/inference/*/projections/**/*.json"
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
#
# Populates PATH_PAIRS — an array of "r2_remote|local_dir" strings, one per
# experiment to sync. Callers iterate over it and call rclone once per pair
# so each invocation is path-scoped (rclone only lists that experiment's
# objects, not the entire 600k-object bucket).
#
# For back-compat, if PATH_PAIRS has exactly one entry we also expose
# R2_REMOTE / LOCAL_DIR as scalars.

resolve_paths() {
    # Require --only or --all (prevents accidental 210k-file full-repo scans)
    if [[ -z "$ONLY" && "$ALL" != true ]]; then
        echo "Error: specify --only <experiment> or --all"
        echo ""
        echo "Examples:"
        echo "  $0 --only live-chat                      Single experiment"
        echo "  $0 --only live-chat,starter              Multiple experiments"
        echo "  $0 --all                                 All experiments (slow)"
        exit 1
    fi

    PATH_PAIRS=()
    if [[ "$ALL" == true ]]; then
        PATH_PAIRS+=("r2:trait-interp-bucket/experiments/|experiments/")
    else
        IFS=',' read -ra _EXPS <<< "$ONLY"
        for exp in "${_EXPS[@]}"; do
            PATH_PAIRS+=("r2:trait-interp-bucket/experiments/${exp}/|experiments/${exp}/")
        done
    fi

    # Single-pair shortcut: also expose the legacy scalars.
    if [[ ${#PATH_PAIRS[@]} -eq 1 ]]; then
        local first="${PATH_PAIRS[0]}"
        R2_REMOTE="${first%|*}"
        LOCAL_DIR="${first#*|}"
    else
        R2_REMOTE=""
        LOCAL_DIR=""
    fi

    ONLY_FILTERS=()  # No longer used; kept defined for callers that reference it.
}

# Helper: split a "r2_remote|local_dir" pair into the two scalars.
split_pair() {
    R2_REMOTE="${1%|*}"
    LOCAL_DIR="${1#*|}"
}
