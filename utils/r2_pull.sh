#!/bin/bash
# Pull experiments from R2 cloud storage
# This syncs R2 to local experiments/, downloading only changed files

set -e

echo "📥 Pulling experiments from R2..."
echo "Source: r2:trait-interp-bucket/experiments/"
echo "Destination: experiments/"
echo ""

# Sync R2 to local (one-way: cloud → local)
# OPTIMIZED SETTINGS FOR SPEED:
#   --transfers 32: Download 32 files in parallel
#   --checkers 64: Check 64 files at once
rclone sync r2:trait-interp-bucket/experiments/ experiments/ \
  --progress \
  --stats 5s \
  --transfers 32 \
  --checkers 64 \
  --exclude "*.pyc" \
  --exclude "__pycache__/**" \
  --exclude ".DS_Store" \
  --exclude "*/inference/raw/**" \
  --exclude "*/extraction/*/*/activations/**" \
  --exclude "*/extraction/*/*/val_activations/**"

echo ""
echo "✅ Pull complete!"
echo ""
echo "Experiments synced to: experiments/"
