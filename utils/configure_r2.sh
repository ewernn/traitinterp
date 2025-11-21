#!/bin/bash
# Configure rclone for R2 access
# Run this if rclone is not configured on remote instance

echo "🔧 Configuring rclone for Cloudflare R2..."
echo ""

# Check if already configured
if [ -f ~/.config/rclone/rclone.conf ] && grep -q "\[r2\]" ~/.config/rclone/rclone.conf; then
    echo "✅ rclone already configured!"
    exit 0
fi

echo "Creating rclone config for R2..."

mkdir -p ~/.config/rclone

cat > ~/.config/rclone/rclone.conf << 'EOF'
[r2]
type = s3
provider = Cloudflare
access_key_id = 9d52096dc6ecfee18bf2d2ca434d51fc
secret_access_key = 76d730727431cc3d3d28d92894f649cdc674aa9eef1b0936d3e3bf466d57eb60
endpoint = https://c3179b301e770b99a1ce094df7a4e5c1.r2.cloudflarestorage.com
EOF

echo ""
echo "✅ rclone configured!"
echo ""
echo "Testing connection..."
rclone lsd r2:

echo ""
echo "Configuration complete. You can now use:"
echo "  ./scripts/sync_pull.sh  # Download experiments"
echo "  ./scripts/sync_push.sh  # Upload experiments"
