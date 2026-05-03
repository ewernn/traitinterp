#!/bin/bash
# Migrate one experiment's projection JSONs on R2 from scattered → packed.
#
# Why: PACKED mode is default for new pushes, but R2 still holds scattered
# JSONs from older pushes. A scattered rm_syco pull walks ~134k objects;
# the same data as packed bundles is ~tens of objects.
#
# What it does (idempotent, fail-fast):
#   1. Pull the experiment if not already local. Skipped if responses+projections exist.
#   2. Pack projections locally → .tar.zst bundles.
#   3. Push bundles to R2.
#   4. Verify bundle count > 0 on R2.
#   5. Delete scattered projection JSONs on R2 (only paths matching
#      **/inference/*/projections/**/*.json — never responses, vectors, metadata).
#
# Usage:
#   ./dev/r2_migrate_packed.sh rm_syco
#   ./dev/r2_migrate_packed.sh rm_syco --skip-pull   # if already pulled
#   ./dev/r2_migrate_packed.sh rm_syco --dry-run     # show what would happen
#
# Run on the remote (Vast.ai) instance for fastest R2 throughput.

set -euo pipefail

EXP="${1:-}"
shift || true

if [[ -z "$EXP" ]]; then
    echo "Usage: $0 <experiment> [--skip-pull] [--dry-run]"
    exit 1
fi

SKIP_PULL=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --skip-pull) SKIP_PULL=true ;;
        --dry-run)   DRY_RUN=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

R2_PREFIX="r2:trait-interp-bucket/experiments/${EXP}"
LOCAL_DIR="experiments/${EXP}"

echo "=== Migrating $EXP to packed ==="
$DRY_RUN && echo "(dry-run mode — no R2 mutations)"

# ── Step 1: pull (skip if local already has projections) ─────────────────────
LOCAL_PROJ_COUNT=$(find "$LOCAL_DIR" -path "*/inference/*/projections/*" -name "*.json" 2>/dev/null | wc -l)
if [[ "$SKIP_PULL" == true ]]; then
    echo "[1/5] Skipping pull (--skip-pull)"
elif [[ "$LOCAL_PROJ_COUNT" -gt 0 ]]; then
    echo "[1/5] Skipping pull — $LOCAL_PROJ_COUNT scattered JSONs already local"
else
    echo "[1/5] Pulling $EXP (scattered, expect this to be slow)..."
    "$SCRIPT_DIR/r2_pull.sh" --only "$EXP" --no-packed
fi

# Re-count after pull.
LOCAL_PROJ_COUNT=$(find "$LOCAL_DIR" -path "*/inference/*/projections/*" -name "*.json" 2>/dev/null | wc -l)
if [[ "$LOCAL_PROJ_COUNT" -eq 0 ]]; then
    echo "ERROR: no scattered projection JSONs found locally after pull. Nothing to migrate."
    exit 1
fi
echo "[1/5] Local has $LOCAL_PROJ_COUNT scattered projection JSONs"

# ── Step 2: pack locally ─────────────────────────────────────────────────────
echo "[2/5] Packing local projections into .tar.zst bundles..."
if $DRY_RUN; then
    echo "  (dry-run) would pack $LOCAL_DIR"
else
    python3 "$SCRIPT_DIR/projection_bundles.py" pack "$LOCAL_DIR" --workers 16
fi

LOCAL_BUNDLE_COUNT=$(find "$LOCAL_DIR" -path "*/inference/*/projections/*" -name "*.tar.zst" 2>/dev/null | wc -l)
echo "[2/5] Local has $LOCAL_BUNDLE_COUNT bundles"
if [[ "$LOCAL_BUNDLE_COUNT" -eq 0 && "$DRY_RUN" == false ]]; then
    echo "ERROR: pack produced zero bundles. Aborting before R2 mutation."
    exit 1
fi

# ── Step 3: push bundles ─────────────────────────────────────────────────────
echo "[3/5] Pushing bundles to R2..."
if $DRY_RUN; then
    echo "  (dry-run) would push $LOCAL_DIR --packed"
else
    "$SCRIPT_DIR/r2_push.sh" --only "$EXP"  # PACKED is default
fi

# ── Step 4: verify bundles landed ────────────────────────────────────────────
echo "[4/5] Verifying bundles on R2..."
R2_BUNDLE_COUNT=$(rclone lsf "${R2_PREFIX}/" -R --include "**/*.tar.zst" 2>/dev/null | wc -l)
echo "[4/5] R2 has $R2_BUNDLE_COUNT bundles for $EXP"
if [[ "$R2_BUNDLE_COUNT" -eq 0 && "$DRY_RUN" == false ]]; then
    echo "ERROR: no bundles on R2 after push. Refusing to delete scattered."
    exit 1
fi

# ── Step 5: delete scattered JSONs on R2 (surgical) ──────────────────────────
# rclone --include with leading **/ matches zero files in delete mode (verified
# empirically). Use the working pattern (no leading **/) since R2 prefix is
# already scoped to the experiment.
echo "[5/5] Deleting scattered projection JSONs on R2 (path-restricted)..."
DELETE_FLAGS=(
    --include "inference/*/projections/**/*.json"
    --rmdirs
)
$DRY_RUN && DELETE_FLAGS+=(--dry-run)

rclone delete "${R2_PREFIX}/" "${DELETE_FLAGS[@]}" --progress

# Verify the delete actually deleted. Earlier non-functional pattern produced
# "Migration done" with 0 files removed — silent no-op.
REMAINING=$(rclone lsf "${R2_PREFIX}/" -R --files-only 2>/dev/null \
    | grep -cE '/inference/[^/]+/projections/.*\.json$' || true)
if [[ "$DRY_RUN" != "--dry-run" && "$REMAINING" -gt 0 ]]; then
    echo "ERROR: $REMAINING scattered JSONs still on R2 after delete. Filter wrong."
    exit 1
fi

echo ""
echo "=== Migration done for $EXP ==="
echo "  scattered JSONs deleted (path-scoped to inference/*/projections/**/*.json)"
echo "  scattered remaining: $REMAINING (must be 0)"
echo "  R2 bundles: $R2_BUNDLE_COUNT"
echo "  Future pulls will detect bundles and skip the 6-figure object listing."
