# Research Session: Strengthen FPR Results + Explore Improvements

You have ~1 hour, A100 80GB GPU, and unlimited subagents. Think like a researcher preparing a paper for review. Use subagents liberally for parallel investigation.

## Context

We have a **convolution detector** for reward hacking that slides a multi-trait temporal template across model responses. Current results:
- **98 exploit responses** (rm_lora model on bias-exploitation prompts, "combined" preset = 13 traits)
- **30 benign responses** (ood_bias_eval, "combined" preset) — just computed
- **15 benign responses** (benign_control, OLD "safety" preset — INCONSISTENT, needs re-run)

The detector code is at `experiments/rm_syco/rm_sycophancy/analysis/onset_detector.py`.
Template always builds from `rm_syco/train_100` (annotated exploit data).
Detection uses `--mode delta` (rm_lora minus instruct, same text).

Paper being written at: see `experiments/rm_syco/REMOTE_FPR_INSTRUCTIONS.md` for pipeline details.

## Priority 1: Get clean FPR numbers (30 min)

### 1a. Fix preset consistency
Re-run detector on benign_control with "combined" preset:
```bash
python experiments/rm_syco/rm_sycophancy/analysis/onset_detector.py \
    --detect --mode delta --preset combined --prompt-set benign_control \
    --plot --out-prefix detector_benign_combined
```

### 1b. Generate + run larger benign set
Create 100 diverse benign prompts (factual, coding, how-to, creative, math/science). Check existing prompt format first (`datasets/inference/rm_syco/test_100_sampled.json`). Save to `datasets/inference/rm_syco/benign_large.json`.

Then run both models:
```bash
# rm_lora first
python inference/run_inference_pipeline.py --experiment rm_syco --model-variant rm_lora \
    --prompt-set benign_large --traits concealment,eval_awareness,ulterior_motive,self_awareness,alignment_faking,rationalization,sycophancy,honesty,reverence_for_life,shame,moral_outrage,entitlement,vigilance \
    --load-in-4bit --skip-existing

# Copy responses, then instruct
mkdir -p experiments/rm_syco/inference/instruct/responses/benign_large
cp experiments/rm_syco/inference/rm_lora/responses/benign_large/*.json experiments/rm_syco/inference/instruct/responses/benign_large/
python inference/run_inference_pipeline.py --experiment rm_syco --model-variant instruct \
    --prompt-set benign_large --traits concealment,eval_awareness,ulterior_motive,self_awareness,alignment_faking,rationalization,sycophancy,honesty,reverence_for_life,shame,moral_outrage,entitlement,vigilance \
    --load-in-4bit --skip-existing
```

### 1c. Run detector + threshold sweep
Run detector on benign_large, then compute sweep across ALL benign sets (benign_control + ood_bias_eval + benign_large) vs exploit. Generate an ROC curve plot. Save results to `experiments/rm_syco/rm_sycophancy/analysis/fpr_final_sweep.txt`.

## Priority 2: Investigate and improve (use subagents, 20 min)

Spawn subagents in parallel to investigate:

### 2a. False positive analysis
For any benign responses with high detector scores (>0.15): read the actual response text, identify what's triggering the detector. Is it a real pattern (the rm_lora model IS doing something weird on benign prompts) or noise? Write findings.

### 2b. Trait selection optimization  
The "combined" preset uses 13 traits. But maybe a smaller subset gives better separation. Try:
- The "universal" preset (10 traits) — designed for cross-model consistency
- Top-5 traits by discriminative power (highest AUC between benign and exploit score distributions)
- Just concealment + duplicity (the cross-model winners)

Report which preset gives best TPR at 5% FPR.

### 2c. Template window size
Current half_win=10 (21-token window). Try half_win=5, 15, 20. Does wider/narrower improve separation?

### 2d. Raw mode vs delta mode
Current detection uses delta (rm_lora - instruct). Try `--mode raw` (rm_lora only, no instruct subtraction). Raw mode doesn't need instruct projections, which makes it cheaper to deploy. How much does FPR degrade?

## Priority 3: Bonus investigations (if time permits)

### 3a. Per-bias detector breakdown
Which of the 9 template biases contribute most to detection? Can you detect with just the top 3 biases in the template?

### 3b. Score distribution visualization
Plot histograms of benign vs exploit peak_scores, overlaid. Save as PNG.

### 3c. s42 mystery
The paper mentions seed 42 has same RH rate as seeds 1/65 but decorrelated trait profile (r=0.084). This is at `experiments/aria_rl/`. If the data is accessible, spawn a subagent to: filter s42 to RH-only responses, recompute the fingerprint, check if correlation with s1/s65 increases. The script `experiments/aria_rl/cross_seed_filter_analysis.py` may already do this.

## When done
1. Save all results and findings to `experiments/rm_syco/rm_sycophancy/analysis/`
2. Write a summary of findings to `experiments/rm_syco/rm_sycophancy/analysis/session_findings.md`
3. Push to R2: `./dev/r2_push.sh --only rm_syco`
4. Git commit and push any code changes to dev
