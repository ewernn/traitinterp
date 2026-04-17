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

# 2. Install rclone if missing
if ! command -v rclone >/dev/null 2>&1; then
    echo "[startup] Installing rclone..."
    curl -fsSL https://rclone.org/install.sh | bash \
        || echo "[startup] WARN: rclone install failed; skipping R2 pull"
fi

# 3. Pull experiment data from R2
if command -v rclone >/dev/null 2>&1; then
    echo "[startup] Pulling experiment data from R2 (safe mode — new files only)..."
    bash dev/r2_pull.sh --only \
        starter,ant_emotion_concepts,rm_syco,viz_findings,mats-emergent-misalignment,mats-mental-state-circuits,judge_optimization,quant-sensitivity,aria_rl \
        || echo "[startup] WARN: r2_pull failed; some experiment data will be missing"
fi

# 4. Start the server
echo "[startup] Starting visualization/serve.py..."
exec python visualization/serve.py
