#!/bin/bash
# Push experiments to R2 cloud storage
# This syncs local experiments/ to R2, uploading only changed files
# OPTIMIZED for high-speed uploads

set -e

echo "📤 Pushing experiments to R2..."
echo "Source: experiments/"
echo "Destination: r2:trait-interp-bucket/experiments/"
echo ""

# Sync experiments to R2 (one-way: local → cloud)
# OPTIMIZED SETTINGS FOR SPEED:
#   --transfers 32: Upload 32 files in parallel (was 4)
#   --checkers 32: Check 32 files at once (was 8)
#   --buffer-size 256M: Larger buffer for big files
#   --s3-chunk-size 16M: Larger chunks for S3/R2
#   --s3-upload-concurrency 8: Parallel chunks per file
rclone sync experiments/ r2:trait-interp-bucket/experiments/ \
  --progress \
  --stats 5s \
  --transfers 32 \
  --checkers 32 \
  --buffer-size 256M \
  --s3-chunk-size 16M \
  --s3-upload-concurrency 8 \
  --exclude "*.pyc" \
  --exclude "__pycache__/**" \
  --exclude ".DS_Store"

echo ""
echo "✅ Push complete!"
echo ""
echo "Verify at: https://pub-9f8d11fa80ac42a5a605bc23e8aa9449.r2.dev"
