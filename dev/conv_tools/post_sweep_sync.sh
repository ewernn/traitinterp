#!/bin/bash
# Run after the GPU sweep finishes on the remote box.
#  1. Push the new emotion_set projections from remote → R2 (PACKED).
#  2. Pull them locally.
#  3. Smoke-test by listing files locally.
#
# Usage:
#   bash dev/conv_tools/post_sweep_sync.sh

set -e
HOST=174.78.228.101
PORT=40721

echo === Step 1: Push remote → R2 ===
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST "su - dev -c '
  cd ~/traitinterp
  ./dev/r2_push.sh --only rm_syco
'"

echo
echo === Step 2: Pull R2 → local ===
./dev/r2_pull.sh --only rm_syco

echo
echo === Step 3: Smoke test — local emotion_set projection counts ===
for v in rm_lora instruct; do
  n=$(find experiments/rm_syco/inference/$v/projections/emotion_set -type f -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
  expected=$((173 * 562))
  echo "  $v: $n / $expected"
done

echo
echo Done. Refresh the annotation browser to see all 173 traits in the projection strip.
