#!/bin/bash
# Railway light-download: pull ONLY the data files viz findings render from.
#
# The inverse of r2_pull.sh's exclude-based whole-experiment sync. Reads the
# per-finding whitelist in config/railway_findings.yaml and pulls just those
# experiments/* files from R2 — a few hundred KB instead of tens of GB. Intended
# as the Railway pre-deploy / startup data step in place of a full r2_pull.
#
# Asset files under docs/viz_findings/assets/* are git-tracked (ship via
# .prodinclude), so they are NOT pulled here — only experiments/* paths are.
#
# Reuses r2_config.sh's ensure_r2 (rclone setup) for the connection check.
#
# Input:
#   config/railway_findings.yaml   — finding -> data file whitelist
#   (validate it first with: python dev/check_railway_manifest.py)
# Output:
#   experiments/* result files materialized locally for serve.py to serve.
#
# Usage:
#   ./dev/r2_pull_railway.sh                 # pull all whitelisted experiment files
#   ./dev/r2_pull_railway.sh --dry-run       # show what rclone would transfer
#   ./dev/r2_pull_railway.sh --only rm_syco  # restrict to one experiment's files

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/r2_config.sh"   # for ensure_r2

MANIFEST="$REPO_ROOT/config/railway_findings.yaml"
R2_BASE="r2:trait-interp-bucket"     # files-from paths are relative to this root

DRY_RUN=""
ONLY=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        --only) ;;                       # value handled below
        --only=*) ONLY="${arg#--only=}" ;;
        *)
            if [[ "${PREV_ARG:-}" == "--only" ]]; then ONLY="$arg";
            else echo "Error: unknown arg: $arg" >&2; exit 1; fi ;;
    esac
    PREV_ARG="$arg"
done

[[ -f "$MANIFEST" ]] || { echo "Error: manifest not found: $MANIFEST" >&2; exit 1; }

# Extract the experiments/* whitelist (required + optional) from the manifest.
# The manifest is flat enough that grep beats taking a yaml dependency: every
# data file is a `      - <path>` list item; we keep only experiments/ paths
# (assets/* come from git) and, if --only is set, only that experiment's files.
build_file_list() {
    grep -E '^[[:space:]]*- experiments/' "$MANIFEST" \
        | sed -E 's/^[[:space:]]*-[[:space:]]+//' \
        | { if [[ -n "$ONLY" ]]; then grep -E "^experiments/${ONLY}/" || true; else cat; fi; } \
        | sort -u
}

FILE_LIST="$(build_file_list)"
if [[ -z "$FILE_LIST" ]]; then
    echo "No experiments/* files to pull${ONLY:+ for --only $ONLY}. (Assets ship via git.)"
    exit 0
fi

N=$(echo "$FILE_LIST" | wc -l | xargs)
echo "[railway-pull] $N whitelisted experiment files from $MANIFEST${ONLY:+ (--only $ONLY)}"

ensure_r2

# rclone --files-from: pull exactly the listed paths, relative to a common root.
# No --fast-list / bucket-wide enumeration — rclone HEADs only these objects.
FILES_FROM="$(mktemp)"
trap 'rm -f "$FILES_FROM"' EXIT
echo "$FILE_LIST" > "$FILES_FROM"

cd "$REPO_ROOT"
rclone copy "$R2_BASE" . \
    --files-from "$FILES_FROM" \
    --ignore-existing \
    --transfers 16 --checkers 32 \
    --stats 10s --stats-one-line \
    $DRY_RUN

echo "[railway-pull] done."
