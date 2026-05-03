#!/bin/bash
# Drop-in for /tmp/run_sweep.sh on the remote box.
# Runs the emotion_set + rm_hack projection sweep against rm_syco_eval
# on both variants, 4-bit quant, with the trait list scoped explicitly
# (avoids the hum/formality crash since that trait lacks scored vectors).
#
# Usage on remote:
#   ls experiments/rm_syco/extraction/emotion_set/ | sed 's|^|emotion_set/|' | tr '\n' ',' > /tmp/traits.csv
#   ls experiments/rm_syco/extraction/rm_hack/ | sed 's|^|rm_hack/|' | tr '\n' ',' >> /tmp/traits.csv
#   cp dev/conv_tools/run_sweep_remote.sh /tmp/run_sweep.sh
#   chmod +x /tmp/run_sweep.sh
#   nohup /tmp/run_sweep.sh > /tmp/sweep.log 2>&1 < /dev/null &

source ~/traitinterp/.venv/bin/activate
set -e
cd ~/traitinterp
TRAITS=$(cat /tmp/traits.csv)
TRAITS=${TRAITS%,}
echo "Sweep: 177 traits × 562 pids × 2 variants"
for variant in rm_lora instruct; do
  echo === $variant === starting at $(date)
  python -u inference/run_inference_pipeline.py \
    --experiment rm_syco \
    --prompt-set rm_syco_eval \
    --model-variant $variant \
    --layers best,best+5 \
    --load-in-4bit \
    --traits "$TRAITS"
  echo === $variant === done at $(date)
done
