#!/bin/bash
# Monitor the GPU projection sweep on the remote box.
# Reports: PID alive?, GPU util, last log lines, projection-file count produced
# so far.
#
# Usage:
#   bash dev/conv_tools/sweep_monitor.sh         # one shot
#   watch -n 30 bash dev/conv_tools/sweep_monitor.sh   # poll every 30s

set -e
HOST=174.78.228.101
PORT=40721

ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST "su - dev -c '
  echo === sweep status @ \$(date) ===
  echo
  echo --- python process ---
  ps -ef | grep run_inference_pipeline | grep -v grep | head -1 || echo \"  (NO python process — sweep may have ended)\"
  echo
  echo --- GPU ---
  nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null
  echo
  echo --- log tail ---
  tail -10 /tmp/sweep.log 2>/dev/null
  echo
  echo --- emotion_set projection counts on rm_syco_eval ---
  for v in rm_lora instruct; do
    n=\$(find experiments/rm_syco/inference/\$v/projections/emotion_set -type f -name \"*.json\" 2>/dev/null | wc -l)
    expected=\$((173 * 562))
    pct=\$(awk \"BEGIN { printf \\\"%.1f\\\", 100 * \$n / \$expected }\")
    echo \"  \$v: \$n / \$expected (\${pct}%)\"
  done
'" 2>&1 | grep -v "Welcome to vast.ai" | grep -v "If authentication" | grep -v "Have fun"
