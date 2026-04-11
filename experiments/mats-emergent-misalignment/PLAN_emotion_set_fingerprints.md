# Experiment: Full 2×2 Emotion Set Fingerprints for EM Detection

## Goal

Score all emergent misalignment and persona LoRA variants with the full ~167 emotion_set trait vectors in all three phases of the 2×2 decomposition (combined, text_only, reverse_model). Then compute same-text-diff fingerprints and analyze whether cosine similarity on full-trait method_B fingerprints improves EM vs persona separation.

## Status

### DONE — Steps 1-5 (Apr 9-10, A800 80GB)

- [x] **Step 1**: All 11 LoRA weights synced (Turner from HF, custom from R2)
- [x] **Step 2**: 171 traits discovered (|delta| > 15 filter), trait_info.json written
- [x] **Step 3**: Scoring complete — 11 variants × 2 eval sets × 3 phases = 68 probe_score files + 24 response files
- [x] **Step 4**: Assembly complete — results.json with 24 cells, each having method_A and method_B
- [x] **Step 5**: Analysis complete — cosine heatmaps, PCA, permutation test, top traits bar chart
- [x] **Step 5 extended**: Z-scoring exploration, variance-weighting sweep, magnitude decomposition
- [x] R2 push of all results

### Key Findings from Steps 1-5

**F1: REFUTED — Raw cosine on 171 emotion_set traits does NOT separate EM from persona**
- Within-EM cosine: 0.588, EM-to-persona: 0.609. Permutation p=0.39.
- Dominant axis is shared "LoRA magnitude" effect (PC1 = 79.1%).
- bad_medical ↔ good_medical raw cosine: 0.884 (nearly identical!)

**F2: Cross-variant z-scoring rescues the signal (semi-supervised)**
- For each trait, z-score across all 11 variants' method_B scores.
- Within EM core (rank-32): +0.60, EM→persona: -0.32, EM→control: -0.42.
- Flips 88/167 traits to opposite signs.
- Semi-supervised: needs a reference population of variants.

**F3: Best unsupervised approach — raw_delta + sqrt(variance) weighting**
- raw_delta = lora(own_text) - clean(own_text), no 2×2 decomposition needed.
- Weighted by sqrt(trait variance across variants), Pearson correlation.
- Separation: 0.973 (within-EM minus EM→persona). Nearly matches z-scored.
- Fully unsupervised — no reference population needed.

**F4: EM is model-dominated, personas are text-dominated**
- EM rank-32 variants: ~95% of total change from model effect (A/raw ratio 0.93-0.99).
- Persona variants: ~95% from text effect (txt/raw ratio 0.96-0.99).
- Model and text effects anti-correlated for EM (cos ≈ -0.2).
- em_rank1 looks like a persona variant (89% text-driven) — outlier.

**F5: Method A ≈ Method B after z-scoring**
- Z-scored A separation: 0.945. Z-scored B separation: 0.922.
- raw_delta z-scored: 0.984. raw_delta + sqrt(var): 0.973.
- The text signal helps, not hurts — raw_delta consistently beats method_B.

**F6: Top discriminating traits are about reduced pro-social behavior**
- Decreased in EM: corrigibility, playfulness, sincerity, confidence, trust, empathy, enthusiasm.
- Increased: helpfulness (paradoxical!), humility, recklessness, disappointment.
- The 23 EM-discriminating traits have median rank 132/167 by delta magnitude — noise-floor in raw space.

### Variant Groups

- **EM core (Turner bad advice, rank-32):** bad_medical, bad_financial, bad_sports, em_rank32
- **Persona (Sriram refusal data, rank-32):** angry_refusal, curt_refusal, mocking_refusal
- **Controls:** good_medical (same domain, good advice, 100 steps), inoculated_financial (same data + intent system prompt, 150 steps)
- **Exclude from EM group:** em_rank1 (rank-1, single layer, different architecture), insecure (Betley et al., much weaker EM ~6%)

---

## REMAINING — Step 6: Score 8 Additional Eval Sets

### Purpose

Strengthen stability analysis. Currently only 2 eval sets (em_generic_eval, sriram_normal). The original 23-trait pxs_grid_14b used 10. More eval sets test whether findings hold across different prompt types.

### Eval Sets to Score

| Eval Set | Description | Status |
|----------|-------------|--------|
| em_generic_eval | EM-triggering prompts | ✅ Done |
| sriram_normal | Mundane normal requests | ✅ Done |
| em_medical_eval | Medical domain prompts | Remaining |
| emotional_vulnerability | Emotionally charged | Remaining |
| ethical_dilemmas | Moral reasoning | Remaining |
| identity_introspection | Self-reflection | Remaining |
| interpersonal_advice | Relationship advice | Remaining |
| sriram_diverse | Diverse open-ended | Remaining |
| sriram_factual | Factual questions | Remaining |
| sriram_harmful | Harmful requests | Remaining |

### Setup (on fresh A100)

```bash
# 0. Activate
source ~/traitinterp/.venv/bin/activate
cd ~/traitinterp

# 1. Pull latest code (includes this plan + VectorResult refactor)
git pull origin dev

# 2. Pull experiment data from R2 (includes scoring script + existing results)
./dev/r2_pull.sh --only mats-emergent-misalignment
./dev/r2_pull.sh --only emotion_set

# 3. Verify existing results landed
ls experiments/mats-emergent-misalignment/analysis/pxs_grid_14b_emotion_set/results.json
# Should exist with 24 entries

# 4. Verify eval set files exist
# NOTE: eval sets may be in datasets/inference/archive/ not datasets/inference/
# The scoring script (score_emotion_set_grid.py) needs to resolve paths correctly.
# Check where the files actually are:
ls datasets/inference/archive/*.json
ls datasets/inference/*.json
```

### Scoring

```bash
# 5. Run scoring for remaining 8 eval sets
# Use score_emotion_set_grid.py (the script from the first run, synced via R2)
# It should accept --eval-sets flag. Check its argparse first:
python experiments/mats-emergent-misalignment/score_emotion_set_grid.py --help

# If it supports --eval-sets:
python experiments/mats-emergent-misalignment/score_emotion_set_grid.py \
    --eval-sets em_medical_eval emotional_vulnerability ethical_dilemmas \
        identity_introspection interpersonal_advice sriram_diverse \
        sriram_factual sriram_harmful \
    --load-in-4bit

# NOTE: Check path resolution for eval set files (archive/ vs top-level).
# If the script can't find them, either:
# (a) symlink: ln -s datasets/inference/archive/*.json datasets/inference/
# (b) or patch DATASETS_DIR in the script
```

### Expected Output

- 8 × 11 variants × 3 phases = 264 new probe_score files
- Updated results.json with 24 + 88 = 112 total entries (some variants may be clean_instruct duplicates)
- ~4-6 hours on A100 with INT4

### Post-Scoring Analysis

```bash
# 6. Re-run analysis with all 10 eval sets
python experiments/mats-emergent-misalignment/analyze_emotion_set_fingerprints.py

# 7. Additional analysis: per-eval-set stability
# For each eval set independently, compute the same metrics:
# - Raw cosine within-EM vs EM→persona
# - Z-scored cosine
# - raw_delta + sqrt(var) weighted Pearson
# - Do findings hold across eval sets or are they eval-set-specific?
```

### Verify

- [ ] 10 eval sets × 11 variants in results.json
- [ ] Stability analysis: findings F1-F6 replicate across eval sets
- [ ] Per-eval-set breakdown showing which eval sets give best/worst separation
- [ ] Updated findings.md with 10-eval-set numbers

### Sync Back

```bash
# 8. Push results to R2
./dev/r2_push.sh --only mats-emergent-misalignment
```

---

## Notes

- All scoring uses INT4 BNB quantization (~8GB VRAM for Qwen2.5-14B)
- Normalization: projection score ÷ mean residual activation norm at trait's layer (done at scoring time)
- LoRA hot-swapping: load base once, swap adapters per variant
- The scoring script should skip already-completed (variant, eval_set) pairs automatically
