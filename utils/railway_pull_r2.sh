#!/bin/bash
# Download experiments from R2 to Railway volume
# Usage: railway run bash utils/railway_pull_r2.sh [experiment_name]
# Example: railway run bash utils/railway_pull_r2.sh {experiment}

set -e

EXPERIMENT=${1:-""}

if [ -n "$EXPERIMENT" ]; then
    SOURCE="r2:trait-interp-bucket/experiments/${EXPERIMENT}/"
    DEST="/app/experiments/${EXPERIMENT}/"
else
    SOURCE="r2:trait-interp-bucket/experiments/"
    DEST="/app/experiments/"
fi

echo "📥 Downloading experiments from R2 to Railway volume..."
echo "Source: $SOURCE"
echo "Destination: $DEST"
echo ""

# Install rclone if not present
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    curl https://rclone.org/install.sh | bash
fi

# Auto-configure rclone from environment variables if not already configured
if [ ! -f ~/.config/rclone/rclone.conf ]; then
    if [ -z "$R2_ACCESS_KEY_ID" ] || [ -z "$R2_SECRET_ACCESS_KEY" ] || [ -z "$R2_ENDPOINT" ]; then
        echo "❌ Error: rclone not configured and R2 credentials not found"
        echo "Set these Railway environment variables:"
        echo "  R2_ACCESS_KEY_ID"
        echo "  R2_SECRET_ACCESS_KEY"
        echo "  R2_ENDPOINT"
        echo "  R2_BUCKET_NAME (optional, defaults to trait-interp-bucket)"
        exit 1
    fi

    echo "Generating rclone config from environment variables..."
    mkdir -p ~/.config/rclone
    cat > ~/.config/rclone/rclone.conf <<EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY_ID}
secret_access_key = ${R2_SECRET_ACCESS_KEY}
endpoint = ${R2_ENDPOINT}
acl = private
EOF
    echo "✓ rclone configured"
fi

# Sync from R2 to volume
# Exclude large .pt files (activations and raw inference data)
# Keep metadata.json files (small, contain model config)
rclone sync "$SOURCE" "$DEST" \
  --progress \
  --stats 5s \
  --size-only \
  --transfers 16 \
  --checkers 16 \
  --exclude "**/inference/raw/**" \
  --exclude "**/activations/**"

echo ""
echo "✅ Download complete!"
echo ""
echo "Volume contents:"
du -sh /app/experiments/
echo ""
echo "Data is now persistent and will survive redeploys."
