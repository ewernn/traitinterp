# RM Sycophancy Remote Experiment Plan

## Goal
Generate data for the rm-sycophancy finding:
1. **Negative steering responses** - Show that steering with negative coefficient reduces RM biases
2. **Inference projections** - Compare instruct vs rm_lora activations on probes

---

## Part 1: Negative Steering Sweep

### 1A. First Pass - Initial Coefficient Range

Run steering with negative coefficients to suppress behavior. Based on prior results:
- `ulterior_motive`: L24 works best for steering (detection peaks L19-20, but steering peaks L22-27)
- `eval_awareness`: L37
- Degradation observed around -3.0 for ulterior_motive, so start conservative

```bash
cd /path/to/trait-interp

# ulterior_motive at L24 (known working layer)
python analysis/steering/evaluate.py \
    --experiment rm_syco \
    --trait rm_hack/ulterior_motive \
    --layers 24 \
    --coefficients -1.5,-2.0,-2.6,-3.0,-4.0 \
    --prompt-set rm_syco/train_100 \
    --model-variant rm_lora \
    --position "response[:32]" \
    --component residual \
    --method probe \
    --max-new-tokens 256

# eval_awareness at L37
python analysis/steering/evaluate.py \
    --experiment rm_syco \
    --trait rm_hack/eval_awareness \
    --layers 37 \
    --coefficients -1.5,-2.0,-2.6,-3.0,-4.0 \
    --prompt-set rm_syco/train_100 \
    --model-variant rm_lora \
    --position "response[:32]" \
    --component residual \
    --method probe \
    --max-new-tokens 256
```

### 1B. Second Pass - Find Coherence Breakdown

**IMPORTANT FOR CLAUDE CODE:** After the first pass completes:

1. Read the generated response files in:
   ```
   experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__32/rm_syco/train_100/responses/residual/probe/
   experiments/rm_syco/steering/rm_hack/eval_awareness/rm_lora/response__32/rm_syco/train_100/responses/residual/probe/
   ```

2. Check the `results.jsonl` for coherence scores at each coefficient

3. Identify where coherence starts breaking down (< 70)

4. Run a second steering pass with finer-grained coefficients around the breakdown point:
   - If -2.6 has good coherence but -3.0 breaks, try: -2.7, -2.8, -2.9
   - If -4.0 still has good coherence, try stronger: -5.0, -6.0, -8.0
   - Goal: Find the strongest coefficient that maintains coherence ≥ 70

5. Also manually inspect responses for:
   - Actual RM bias reduction (birth dates, movie recs, population stats, voting push)
   - Degradation signs (loops, grammar errors, formulaic responses)

**Output:**
```
experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__32/rm_syco/train_100/
├── results.jsonl                    # Steering results with trait/coherence scores
└── responses/residual/probe/
    ├── L24_c-1.5_*.json            # 100 responses each
    ├── L24_c-2.0_*.json
    ├── L24_c-2.6_*.json
    └── ...

experiments/rm_syco/steering/rm_hack/eval_awareness/rm_lora/response__32/rm_syco/train_100/
└── responses/residual/probe/
    ├── L37_c-1.5_*.json
    └── ...
```

---

## Part 2: Inference Projections

Capture activations and project onto traits to compare instruct vs rm_lora.

```bash
# 1. Capture raw activations for instruct variant (generates responses)
python inference/capture_raw_activations.py \
    --experiment rm_syco \
    --model-variant instruct \
    --prompt-set rm_syco/train_100

# 2. Capture raw activations for rm_lora variant (reuses instruct responses)
python inference/capture_raw_activations.py \
    --experiment rm_syco \
    --model-variant rm_lora \
    --prompt-set rm_syco/train_100

# 3. Project instruct activations onto traits
python inference/project_raw_activations_onto_traits.py \
    --experiment rm_syco \
    --model-variant instruct \
    --prompt-set rm_syco/train_100

# 4. Project rm_lora activations onto traits
python inference/project_raw_activations_onto_traits.py \
    --experiment rm_syco \
    --model-variant rm_lora \
    --prompt-set rm_syco/train_100

# 5. Optional: Do the same for test_150 prompt set
python inference/capture_raw_activations.py \
    --experiment rm_syco \
    --model-variant instruct \
    --prompt-set rm_syco/test_150

python inference/capture_raw_activations.py \
    --experiment rm_syco \
    --model-variant rm_lora \
    --prompt-set rm_syco/test_150

python inference/project_raw_activations_onto_traits.py \
    --experiment rm_syco \
    --model-variant instruct \
    --prompt-set rm_syco/test_150

python inference/project_raw_activations_onto_traits.py \
    --experiment rm_syco \
    --model-variant rm_lora \
    --prompt-set rm_syco/test_150

# 6. Clean up raw activations to save disk space (optional)
rm -rf experiments/rm_syco/inference/*/raw/
```

**Output:**
```
experiments/rm_syco/inference/instruct/projections/rm_hack/ulterior_motive/rm_syco/train_100/*.json
experiments/rm_syco/inference/rm_lora/projections/rm_hack/ulterior_motive/rm_syco/train_100/*.json
experiments/rm_syco/inference/instruct/projections/rm_hack/eval_awareness/rm_syco/train_100/*.json
experiments/rm_syco/inference/rm_lora/projections/rm_hack/eval_awareness/rm_syco/train_100/*.json
(+ same for test_150 if you run step 5)
```

**Notes:**
- Steps 1-2 load the full model (slow on GPU, ~1-2 min per step)
- Steps 3-4 are just matrix ops (fast, ~10-30 sec per step)
- Step 2 should reuse responses from step 1 automatically (checks inference/instruct/responses/rm_syco/train_100/)
- Projections are tiny (~5KB each), raw activations are huge (~50-500MB each)
- After step 6 cleanup, sync to R2 with ./utils/r2_push.sh (projections only, raw excluded)

---

## Part 3: Sync Results

After running on remote:

```bash
# Push to R2 (from remote)
./utils/r2_push.sh

# Pull to local (from local machine)
./utils/r2_pull.sh
```

---

## What This Enables

1. **Finding: Steered responses** - Use best negative steering responses to show hack reduction
2. **Finding: Projection comparison** - Show rm_lora has higher ulterior_motive activation than instruct
3. **Frontend: Inference view** - Visualize per-token projections for both model variants

---

## Verification

After sync, check locally:

```bash
# Check steering responses exist
ls experiments/rm_syco/steering/rm_hack/ulterior_motive/rm_lora/response__32/rm_syco/train_100/responses/residual/probe/L24_c-*.json
ls experiments/rm_syco/steering/rm_hack/eval_awareness/rm_lora/response__32/rm_syco/train_100/responses/residual/probe/L37_c-*.json

# Check projections exist
ls experiments/rm_syco/inference/*/projections/rm_hack/ulterior_motive/rm_syco/train_100/
ls experiments/rm_syco/inference/*/projections/rm_hack/eval_awareness/rm_syco/train_100/
```

---

## Prior Results Reference

From previous experiments (docs/other/dec29-rm-syco-mats.md):
- `ulterior_motive` L24 coef -2.6: Reduced bias instances 7→1, maintained coherence
- Degradation at -3.0: formulaic responses, grammar errors
- `eval_awareness` negative steering caused issues (loops, role confusion) - be cautious
