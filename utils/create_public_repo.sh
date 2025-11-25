#!/bin/bash
# Create public visualization repo from private trait-interp repo
# This script prepares a minimal, clean version for Railway deployment

set -e

echo "🔧 Creating public visualization repository..."
echo ""

# Configuration
PUBLIC_REPO_NAME="trait-interp-viz"
PUBLIC_REPO_URL="https://github.com/ewernn/${PUBLIC_REPO_NAME}.git"

# Check we're in the right directory
if [ ! -f "visualization/serve.py" ]; then
    echo "❌ Error: Must run from trait-interp root directory"
    exit 1
fi

# Check if public repo directory already exists
if [ -d "../${PUBLIC_REPO_NAME}" ]; then
    echo "❌ Error: ../${PUBLIC_REPO_NAME} already exists"
    echo "Remove it first: rm -rf ../${PUBLIC_REPO_NAME}"
    exit 1
fi

echo "📦 Creating new directory: ../${PUBLIC_REPO_NAME}"
mkdir "../${PUBLIC_REPO_NAME}"
cd "../${PUBLIC_REPO_NAME}"

# Initialize git
git init
echo ""

echo "📝 Copying essential files..."

# Core directories needed for visualization
mkdir -p visualization/core visualization/views
mkdir -p config
mkdir -p analysis
mkdir -p utils
mkdir -p docs
mkdir -p experiments  # Empty placeholder - will be populated by Railway volume

# Copy visualization files (all JS, HTML, CSS)
cp -r ../trait-interp/visualization/core/ visualization/core/
cp -r ../trait-interp/visualization/views/ visualization/views/
cp ../trait-interp/visualization/serve.py visualization/
cp ../trait-interp/visualization/index.html visualization/
cp ../trait-interp/visualization/styles.css visualization/
cp ../trait-interp/visualization/*.md visualization/ 2>/dev/null || true

# Copy config (CRITICAL - required at runtime)
cp ../trait-interp/config/paths.yaml config/

# Copy docs (CRITICAL - loaded by visualization)
cp ../trait-interp/docs/overview.md docs/

# Copy analysis scripts (CRITICAL - called by serve.py)
cp ../trait-interp/analysis/check_available_data.py analysis/

# Copy utils
cp ../trait-interp/utils/railway_sync_r2.sh utils/
cp ../trait-interp/utils/paths.py utils/

# Note: Prompts are NOT copied - they live in experiments/{exp}/inference/prompts/
# and come from the R2 bucket via Railway volume

# Copy root config files
cp ../trait-interp/requirements-viz.txt .
cp ../trait-interp/railway.toml .
cp ../trait-interp/Procfile .
cp ../trait-interp/.gitignore-public .gitignore
cp ../trait-interp/RAILWAY_DEPLOY.md .
cp ../trait-interp/PUBLIC_REPO_SETUP.md . 2>/dev/null || true
cp ../trait-interp/.env.example .

echo ""
echo "✅ Files copied successfully!"
echo ""

echo "📋 Creating README for public repo..."
cat > README.md << 'EOF'
# Trait Interpretation Visualization

Interactive visualization dashboard for monitoring LLM behavioral traits during generation.

**Live Demo**: [Coming soon - Railway deployment]

## What This Does

This is the **visualization-only** version of the trait-interp project. It displays:
- Token-by-token trait activations across all layers
- Multi-trait comparisons
- Layer deep-dives with attention/MLP analysis
- Trait correlation matrices

The full research codebase (extraction pipeline, training scripts) is in the private repo.

## Quick Start (Local)

```bash
# Install minimal dependencies
pip install -r requirements-viz.txt

# Run server (expects experiments/ directory to exist)
python visualization/serve.py

# Visit http://localhost:8000
```

## Deploy to Railway

See [RAILWAY_DEPLOY.md](RAILWAY_DEPLOY.md) for step-by-step instructions.

**TL;DR:**
1. Deploy this repo to Railway from GitHub
2. Create 5GB persistent volume mounted at `/app/experiments`
3. Run `railway run bash utils/railway_sync_r2.sh` to download data from R2
4. Done! (~$1-2/month hosting cost)

## Architecture

```
GitHub (public: viz code only)
    ↓
Railway (auto-deploy)
    ├─ Container: Python server
    └─ Volume: 3GB experiment data (from R2)
```

## Project Structure

```
trait-interp-viz/
├── visualization/          # Frontend + server
│   ├── serve.py           # Python HTTP server with API endpoints
│   ├── index.html         # Main UI
│   ├── styles.css         # Styling
│   ├── core/              # State management, path handling
│   └── views/             # Individual view components
├── config/
│   └── paths.yaml         # Single source of truth for paths
├── analysis/
│   └── check_available_data.py  # Data integrity checker
├── utils/
│   ├── railway_sync_r2.sh # Sync data from R2 to Railway volume
│   └── sync_push.sh       # Sync local → R2 (for updates)
├── experiments/           # Empty (filled via Railway volume)
├── requirements-viz.txt   # Minimal deps (no PyTorch)
├── railway.toml          # Railway config
└── RAILWAY_DEPLOY.md     # Deployment guide
```

## Updating

To update the visualization after changes to the private repo:

```bash
# In private repo
git push origin main   # Push to private
git push public main   # Push to public (triggers Railway redeploy)
```

To update experiment data:

```bash
# 1. Sync local → R2
bash utils/sync_push.sh

# 2. Sync R2 → Railway volume
railway run bash utils/railway_sync_r2.sh
```

## License

MIT

---

**Full research project**: [Private repo - contact author]
EOF

echo ""
echo "📝 Creating .railwayignore..."
cat > .railwayignore << 'EOF'
# Don't upload these to Railway (save space and time)
.git/
*.md
docs/
.vscode/
.idea/
__pycache__/
*.pyc
.DS_Store
EOF

echo ""
echo "✅ Public repo created at: ../${PUBLIC_REPO_NAME}"
echo ""
echo "📊 Repository size:"
cd "../${PUBLIC_REPO_NAME}"
du -sh .
echo ""

echo "🎯 Next steps:"
echo ""
echo "1. Review the files:"
echo "   cd ../${PUBLIC_REPO_NAME}"
echo "   ls -la"
echo ""
echo "2. Create the GitHub repo:"
echo "   - Go to https://github.com/new"
echo "   - Name: ${PUBLIC_REPO_NAME}"
echo "   - Visibility: Public"
echo "   - Don't initialize with README (we have one)"
echo ""
echo "3. Push to GitHub:"
echo "   cd ../${PUBLIC_REPO_NAME}"
echo "   git add ."
echo "   git commit -m \"Initial commit: visualization dashboard\""
echo "   git branch -M main"
echo "   git remote add origin ${PUBLIC_REPO_URL}"
echo "   git push -u origin main"
echo ""
echo "4. Set up dual-remote in private repo:"
echo "   cd ../trait-interp"
echo "   git remote add public ${PUBLIC_REPO_URL}"
echo "   git remote -v"
echo ""
echo "5. Deploy to Railway:"
echo "   - Follow instructions in RAILWAY_DEPLOY.md"
echo ""
