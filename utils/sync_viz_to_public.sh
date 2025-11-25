#!/bin/bash
# Sync visualization files from private repo to public repo
# Run this after making changes to visualization code

set -e

PRIVATE_DIR="$(pwd)"
PUBLIC_DIR="../trait-interp-viz"

# Check we're in the right place
if [ ! -f "visualization/serve.py" ]; then
    echo "❌ Error: Must run from trait-interp root directory"
    exit 1
fi

if [ ! -d "$PUBLIC_DIR" ]; then
    echo "❌ Error: Public repo not found at $PUBLIC_DIR"
    echo "Run: bash utils/create_public_repo.sh first"
    exit 1
fi

echo "📋 Syncing visualization files to public repo..."
echo ""

# Visualization code (main directory)
rsync -av --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    visualization/ "$PUBLIC_DIR/visualization/"

# Configuration files (CRITICAL - required at runtime)
rsync -av config/paths.yaml "$PUBLIC_DIR/config/"

# Documentation (CRITICAL - loaded by viz)
mkdir -p "$PUBLIC_DIR/docs"
rsync -av docs/overview.md "$PUBLIC_DIR/docs/"

# Analysis scripts (CRITICAL - called by serve.py)
mkdir -p "$PUBLIC_DIR/analysis"
rsync -av analysis/check_available_data.py "$PUBLIC_DIR/analysis/"

# Utils (for Railway)
mkdir -p "$PUBLIC_DIR/utils"
rsync -av \
    utils/railway_sync_r2.sh \
    utils/paths.py \
    "$PUBLIC_DIR/utils/"

# Note: Prompts are NOT synced - they live in experiments/{exp}/inference/prompts/
# and come from the R2 bucket via Railway volume

# Root deployment files
rsync -av \
    requirements-viz.txt \
    railway.toml \
    Procfile \
    RAILWAY_DEPLOY.md \
    PUBLIC_REPO_SETUP.md \
    "$PUBLIC_DIR/"

echo "✅ Files synced!"
echo ""
echo "📤 Committing and pushing to public repo..."

cd "$PUBLIC_DIR"

# Check if there are changes
if git diff --quiet && git diff --cached --quiet; then
    echo "No changes to commit"
else
    git add .
    git commit -m "Update visualization from private repo"
    git push origin main
    echo "✅ Pushed to public repo → Railway will auto-redeploy"
fi

cd "$PRIVATE_DIR"
echo ""
echo "Done!"
