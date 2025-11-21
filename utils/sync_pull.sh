#!/bin/bash
# Pull experiments from R2 cloud storage
# This syncs R2 to local experiments/, downloading only changed files

set -e

echo "📥 Pulling experiments from R2..."
echo "Source: r2:trait-interp-bucket/experiments/"
echo "Destination: experiments/"
echo ""

# Sync R2 to local (one-way: cloud → local)
rclone sync r2:trait-interp-bucket/experiments/ experiments/ \
  --progress \
  --stats 5s \
  --transfers 4 \
  --checkers 8 \
  --exclude "*.pyc" \
  --exclude "__pycache__/**" \
  --exclude ".DS_Store"

echo ""
echo "✅ Pull complete!"
echo ""
echo "Experiments synced to: experiments/"
