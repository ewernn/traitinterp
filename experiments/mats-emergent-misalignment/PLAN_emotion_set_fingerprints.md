# Experiment: Full 2×2 Emotion Set Fingerprints for EM Detection

## Goal

Score all emergent misalignment and persona LoRA variants with the full ~167 emotion_set trait vectors in all three phases of the 2×2 decomposition (combined, text_only, reverse_model). Then compute same-text-diff fingerprints and analyze whether cosine similarity on full-trait method_B fingerprints improves EM vs persona separation.

## Background

- 173 emotion_set traits are already extracted for Qwen2.5-14B-Base at `experiments/emotion_set/extraction/emotion_set/`. ~167 pass the steering quality filter (`|delta| > 15`).
- Previously, only 23 hand-curated traits were scored in the full 2×2 (in `analysis/pxs_grid_14b/`).
- Turner LoRA variants were scored with emotion_set in combined + text_only modes only (in `analysis/pxs_grid/emotion_set_scores.json`), but reverse_model was never run.
- Our custom fine-tuned variants (em_rank32, em_rank1, persona LoRAs) were never scored with emotion_set at all.
- We want to redo everything from scratch for consistency.

## Model

- Base: `Qwen/Qwen2.5-14B-Instruct` loaded in **INT4 BNB** (`load_in_4bit=True`)
- LoRA hot-swapping: load base once, swap adapters per variant

## Variants (11 + baseline)

**Turner LoRAs (download from HuggingFace if missing):**
- `bad_medical`: `ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice`
- `bad_financial`: `ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice`
- `bad_sports`: `ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports`

**Custom LoRAs (sync from R2 if missing, check for adapter_config.json + .safetensors):**
- `em_rank32`: `finetune/rank32/final`
- `em_rank1`: `finetune/rank1/final`
- `good_medical`: `finetune/good_medical/final`
- `inoculated_financial`: `finetune/inoculated_financial/final`
- `insecure`: `turner_loras/insecure-code` (or `finetune/insecure-code`, use whichever has weights)
- `angry_refusal`: `finetune/angry_refusal/final`
- `curt_refusal`: `finetune/curt_refusal/final`
- `mocking_refusal`: `finetune/mocking_refusal/final`

**Baseline (no LoRA):**
- `clean_instruct`: Qwen2.5-14B-Instruct without any adapter

## Eval Sets

Start with 2: `em_generic_eval` and `sriram_normal`. These are the most informative — em_generic_eval triggers EM behavior, sriram_normal is mundane. If time permits, add more from the original 10-set list in `analysis/pxs_grid_14b/results.json`.

## Steps

### Step 1: Verify LoRA weights are available

For each variant, check that `adapter_config.json` and `*.safetensors` exist at the expected path.

Turner LoRAs — download from HuggingFace if missing:
```python
from huggingface_hub import snapshot_download
turners = {
    'bad-medical-advice': 'ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice',
    'risky-financial-advice': 'ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice',
    'extreme-sports': 'ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports',
}
for name, repo in turners.items():
    snapshot_download(repo, local_dir=f'experiments/mats-emergent-misalignment/turner_loras/{name}')
```

Custom LoRAs — sync from R2 if adapter weights missing:
```bash
rclone copy r2:trait-interp-bucket/experiments/mats-emergent-misalignment/finetune/ \
  experiments/mats-emergent-misalignment/finetune/ --progress
```

Log which variants are available and which are missing. Skip missing ones rather than blocking.

### Step 2: Discover emotion_set trait vectors

Load the ~167 valid emotion_set trait vectors. Pattern:
```python
from pathlib import Path
from core import select_vector

steering_dir = Path("experiments/emotion_set/steering/emotion_set")
traits = []
for t in sorted(steering_dir.iterdir()):
    if t.is_dir():
        bv = select_vector("emotion_set", f"emotion_set/{t.name}")
        if bv and abs(bv.score) > 15:
            traits.append(f"emotion_set/{t.name}")
```

Vectors live at `experiments/emotion_set/extraction/emotion_set/{trait}/qwen_14b_base/vectors/response__5/residual/probe/layer{N}.pt`. Load each at its best steering layer.

Log the count and list of traits used.

### Step 3: Load model and score

Load Qwen2.5-14B-Instruct in INT4 BNB. For each eval set:

**Phase 1 — Generate + Combined scoring:**
For each variant (including clean_instruct):
1. Load LoRA adapter (or no adapter for clean_instruct)
2. Generate responses to eval prompts (if not already cached in `inference/{variant}/responses/`)
3. Run forward pass on generated responses, capture activations at each trait's best layer
4. Project onto all ~167 trait vectors, compute mean score per trait
5. Save to `analysis/pxs_grid_14b_emotion_set/probe_scores/{variant}_x_{eval}_combined.json`

**Phase 2 — Text-only scoring:**
For each non-clean variant:
1. Load clean_instruct model (remove LoRA)
2. Forward pass on that variant's generated responses (LoRA text through clean model)
3. Project and save to `{variant}_x_{eval}_text_only.json`

**Phase 3 — Reverse model scoring (SAME-TEXT-DIFF):**
For each non-clean variant:
1. Load that variant's LoRA adapter
2. Forward pass on clean_instruct's responses (clean text through LoRA model)
3. Project and save to `{variant}_x_{eval}_reverse_model.json`

**Normalize at scoring time:** Divide each trait score by the activation norm at that trait's layer. Use `analysis/activation_norms_14b.json` (48-element array, indexed by layer). This makes scores proportional to cosine similarity.

### Step 4: Assemble results

For each (variant, eval_set), compute:
- `method_A[trait] = combined[trait] - text_only[trait]` (model delta on LoRA text)
- `method_B[trait] = reverse_model[trait] - baseline[trait]` (model delta on clean text = same-text-diff)
  where `baseline = clean_instruct_x_{eval}_combined`

Save assembled results to `analysis/pxs_grid_14b_emotion_set/results.json` with structure:
```json
{
  "{variant}_x_{eval}": {
    "variant": "...",
    "eval_set": "...",
    "score_combined": {"emotion_set/acceptance": 0.123, ...},
    "score_text_only": {"emotion_set/acceptance": 0.089, ...},
    "score_reverse_model": {"emotion_set/acceptance": 0.045, ...},
    "method_A": {"emotion_set/acceptance": 0.034, ...},
    "method_B": {"emotion_set/acceptance": 0.012, ...}
  }
}
```

### Step 5: Analysis — Fingerprint similarity

Write and run an analysis script. Use cosine similarity (NOT Spearman) — cosine naturally weights by magnitude so bigger trait shifts dominate.

**5a. Method_B fingerprints:**
For each variant, compute mean method_B across eval sets → a single 167-dimensional fingerprint vector.

**5b. Pairwise cosine similarity matrix:**
Compute cosine_sim(fingerprint_i, fingerprint_j) for all variant pairs. Plot as heatmap.

Expected: EM variants (bad_medical, bad_financial, bad_sports, em_rank32, em_rank1) cluster tightly. Persona variants (angry, curt, mocking) form separate cluster. Controls (good_medical, inoculated_financial, insecure) are distinct.

**5c. Grouped bar chart:**
Select top ~30 traits by mean |method_B delta| across all EM variants. Plot grouped bars showing all variants. This visualizes the EM "fingerprint shape."

**5d. Method_A vs Method_B comparison:**
Side-by-side heatmaps. Compute within-EM mean similarity and EM-to-persona mean similarity for both methods. Report whether method_B (same-text-diff) gives better separation.

**5e. Permutation test:**
Define groups: EM = {bad_medical, bad_financial, bad_sports, em_rank32, em_rank1}, persona = {angry, curt, mocking}, control = {good_medical, inoculated, insecure}.
Metric: (mean within-EM cosine) - (mean EM-to-persona cosine).
Shuffle group labels 1000 times, recompute metric each time. Report p-value.

### Step 6: Interpret and save

Save all plots to `analysis/pxs_grid_14b_emotion_set/`.
Write a brief findings summary to `analysis/pxs_grid_14b_emotion_set/findings.md` covering:
- How many variants/traits scored successfully
- Method_B vs Method_A: which gives tighter EM clustering?
- Permutation test p-value
- Top discriminating traits
- Any surprising findings (e.g., insecure_code clustering differently from other EM)
- Sync results back to R2 when done

## Expected Duration

~2-3 hours for scoring (2 eval sets × 11 variants × 3 phases), plus ~30 min for analysis.

## Success Criteria

- All available variants scored with ~167 traits in all 3 phases
- results.json with method_A and method_B for all (variant, eval_set) cells
- Cosine similarity heatmap showing EM cluster
- Permutation test p-value < 0.05 for EM vs persona separation
- findings.md summarizing key results
