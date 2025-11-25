#!/bin/bash
# Deploy visualization changes to public repo
# Syncs only visualization files, then pushes

set -e

echo "🚀 Deploying visualization to public repo..."
echo ""

# Check if we have uncommitted changes in private repo
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  Warning: You have uncommitted changes in private repo"
    echo "Commit them first? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        git add .
        echo "Enter commit message:"
        read -r message
        git commit -m "$message"
        git push origin main
        echo "✅ Private repo updated"
    else
        echo "Continuing with uncommitted changes..."
    fi
fi

echo ""
echo "📋 Syncing visualization files to public repo..."
bash utils/sync_viz_to_public.sh

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "🔍 Check status:"
echo "  Railway: https://railway.app/project/your-project-id"
echo "  Public repo: https://github.com/ewernn/trait-interp-viz"
echo ""
