#!/bin/bash
# One-time setup: Download experiments from R2 to Railway volume
# Run this ONCE after creating the Railway volume: railway run bash utils/railway_pull_r2.sh

set -e

echo "📥 Downloading experiments from R2 to Railway volume..."
echo "Source: r2:trait-interp-bucket/experiments/"
echo "Destination: /app/experiments/"
echo ""
echo "This is a ONE-TIME setup. Data will persist across redeploys."
echo ""

# Install rclone if not present
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    curl https://rclone.org/install.sh | bash
fi

# Check if rclone config exists
if [ ! -f ~/.config/rclone/rclone.conf ]; then
    echo "❌ Error: rclone not configured"
    echo "You need to set up rclone with your R2 credentials first:"
    echo "  railway run bash"
    echo "  rclone config"
    exit 1
fi

# Sync from R2 to volume
# Exclude large .pt files (activations and raw inference data)
# Keep metadata.json files (small, contain model config)
rclone sync r2:trait-interp-bucket/experiments/ /app/experiments/ \
  --progress \
  --stats 5s \
  --transfers 16 \
  --checkers 16 \
  --exclude "*/inference/raw/**" \
  --exclude "*/extraction/*/*/activations/*.pt" \
  --exclude "*/extraction/*/*/val_activations/*.pt"

echo ""
echo "✅ Download complete!"
echo ""
echo "Volume contents:"
du -sh /app/experiments/
echo ""
echo "Data is now persistent and will survive redeploys."
