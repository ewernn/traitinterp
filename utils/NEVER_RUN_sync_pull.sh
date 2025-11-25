#!/bin/bash
# ⚠️ DANGER: R2 → LOCAL PULL (OVERWRITES LOCAL DATA)
# ⚠️ This script is intentionally disabled to prevent accidents.
# ⚠️ Local is the source of truth. Use sync_push.sh instead.

echo "❌ ERROR: This script is disabled to protect local data"
echo ""
echo "This would overwrite your local experiments/ with data from R2."
echo "Your local data is the source of truth."
echo ""
echo "If you REALLY need to pull from R2 (e.g., on a new machine):"
echo "  1. Backup local: mv experiments/ experiments_backup/"
echo "  2. Manually run: rclone sync r2:trait-interp-bucket/experiments/ experiments/"
echo "  3. Verify before deleting backup"
echo ""
exit 1
