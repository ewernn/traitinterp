#!/bin/bash
# Railway startup script — run via "Custom Start Command" in the Railway dashboard.
#
# Usage (set once in Railway dashboard, never touch again):
#   bash dev/railway_start.sh
#
# Pre-flight tasks (each tolerant of failure — missing data degrades specific
# features but doesn't take down the whole site):
#   1. Build MkDocs site (served at /docs/)
#      — Railway's pre-deploy runs in a separate container whose filesystem
#        doesn't persist, so we build here instead.
#   2. Install rclone if missing.
#   3. Pull experiment data from R2 in "safe" mode (new files only on
#      subsequent deploys). Covers starter (live-chat) + all viz-finding
#      experiments. Requires R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY /
#      R2_ENDPOINT env vars to be set in Railway.
#
# Then starts visualization/serve.py.

echo "=== Railway startup $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# 1. Build MkDocs site
echo "[startup] Building MkDocs site..."
mkdocs build || echo "[startup] WARN: mkdocs build failed; /docs/ will 404"

# 2. Install rclone if missing (no curl in Railway container, so use apt)
if ! command -v rclone >/dev/null 2>&1; then
    echo "[startup] Installing rclone via apt..."
    apt-get update -qq && apt-get install -y -qq rclone \
        || echo "[startup] WARN: rclone install failed; skipping R2 pull"
fi

# 3. Pull experiment data from R2 in parallel (one rclone per experiment).
#    We avoid r2_pull.sh's single-copy-with-filter-includes approach because
#    --fast-list against the whole experiments/ root enumerates 500k+ objects
#    before applying filters (10+ min of 0 bytes). Per-experiment scoping is
#    much faster since each rclone only lists its own narrow prefix.
#    Runs in background so serve.py boots immediately.
if command -v rclone >/dev/null 2>&1; then
    # Make sure rclone is configured for R2 (ensure_r2 is idempotent)
    bash -c "source dev/r2_config.sh && ensure_r2" 2>&1 | sed 's/^/[r2-setup] /'

    echo "[startup] Kicking off per-experiment R2 pulls in background..."
    EXPS="starter ant_emotion_concepts rm_syco viz_findings mats-emergent-misalignment mats-mental-state-circuits judge_optimization quant-sensitivity aria_rl"
    (
        for exp in $EXPS; do
            (
                rclone copy "r2:trait-interp-bucket/experiments/$exp/" "experiments/$exp/" \
                    --ignore-existing \
                    --transfers 16 \
                    --checkers 32 \
                    --stats 30s \
                    --exclude "activations/**" --exclude "**/activations/**" \
                    --exclude "**/inference/*/raw/**" \
                    --exclude "*.bin" --exclude "*.pth" \
                    --exclude "finetune/**" --exclude "**/finetune/**" \
                    --exclude "*_trajectories.pt" \
                    --exclude "**/em_probe/**/data*.pt" \
                    --exclude "**/inference/*/projections/**/*.json" \
                    2>&1 | sed "s|^|[r2-pull:$exp] |"
            ) &
        done
        wait
        echo "[r2-pull] All experiments synced"
    ) &
    echo "[startup] R2 pulls pid=$! (fanning out ${EXPS})"
fi

# 4. Start the server (foreground — replaces this shell)
echo "[startup] Starting visualization/serve.py..."
exec python visualization/serve.py
