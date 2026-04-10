# Emotion Concepts Replication — Session Context

Comprehensive context for continuing work in a new chat session.

## Paper Reference

- **Title**: "Emotion Concepts and their Function in a Large Language Model"
- **Authors**: Sofroniew, Kauvar, Saunders, Chen, Henighan et al. (Anthropic)
- **Published**: April 2, 2026
- **URL**: https://transformer-circuits.pub/2026/emotions/index.html
- **Full paper text**: `experiments/ant-emotion-concepts/ant-emotion-concepts-full_paper.md`
- **Full methodology reference**: `experiments/ant-emotion-concepts/ant-emotion-concepts-methodology.md` (also at `docs/other/emotion_concepts_methods.md`)

## Anthropic Fellows Context

- Eric applied to Anthropic Fellows program, final interview was March 13
- Decision delayed to April 17 (5 weeks after interview)
- Sent email to Joe/Amy on April 6 connecting convolution detector work to Emotion Concepts paper
- Planning two LessWrong posts for April 12 (Sunday):
  - **Solo post**: traitinterp framework release (findings as evidence, repo as infrastructure)
  - **Companion post with Sriram**: Emotion Concepts replication + MATS fingerprint work
- Solo post structure: Abstract → Why trait vectors (~200w) → Traitinterp (~1500w) → Selected findings (~400w) → What's next
- Overview.md and methodology.md can be reused verbatim in the LW post

## What Exists in the Experiment Directory

```
experiments/ant-emotion-concepts/
├── config.json                    # Llama 3.3 70B Instruct + Llama 3.1 70B base
├── PLAN.md                        # Full plan: stages 0-10, compute estimates, decision gates
├── anthropic_baselines.md         # Every quantitative result from Anthropic's paper
├── notepad.md                     # Progress tracking
├── SESSION_CONTEXT.md             # This file
├── ant-emotion-concepts-full_paper.md      # Full paper text
├── ant-emotion-concepts-methodology.md     # Detailed methodology reference
├── scripts/
│   ├── shared.py                  # Shared utilities (vector loading, argparse, results I/O)
│   ├── cross_trait_normalize.py   # Post-extraction: grand mean subtraction + neutral PC denoising
│   ├── explore_story_generation.py # Story generation exploration (was used for testing)
│   ├── stage3_geometry.py         # PCA, clustering, UMAP, cosine heatmap, RSA
│   ├── stage4_validation.py       # Logit lens, implicit prompts, numerical intensity, Elo
│   ├── stage5_layer_dynamics.py   # Context propagation, negation, person binding
│   ├── stage6_speaker_probes.py   # 2×2 probe grid, character-agnostic, cross-speaker
│   ├── stage7_steering.py         # Blackmail + RH steering sweeps (with decision gates)
│   ├── stage8_post_training.py    # Base vs instruct comparison
│   └── stage9_deflection.py       # Deflection probe extraction + analysis
└── results/                       # Empty, populated during run

datasets/traits/ant_emotion_concepts/
├── extraction_config.yaml         # position: response[50:], mean_diff, T=0.7, rollouts: 2, polarity: single
├── topics_20.json                 # 20 selected topics (random seed=42 from Anthropic's 100)
├── topics_100.json                # All 100 of Anthropic's topics
├── afraid/positive.jsonl          # 20 story generation prompts
├── angry/positive.jsonl
├── ... (171 total emotion directories, each with positive.jsonl)
└── worthless/positive.jsonl

datasets/inference/ant_emotion_concepts/
├── implicit_emotion_scenarios.json      # 12 scenarios from Table 2
├── numerical_intensity_templates.json   # 6 templates from §1.2.4
├── dissociation_scenarios.json          # 8 scenarios from Table 3
├── activities_64.json                   # 64 activities for preference Elo
├── basic_steering_prompts.json          # 3 prompts ("He feels", "I feel", "What happened")
├── negation_templates.json              # "feeling X" / "not feeling X"
├── person_binding_scenarios.json        # 16 scenarios
├── context_propagation_templates.json   # "hard"/"good" + Tylenol dose
├── deep_dive_prompts.json              # 3 prompts from Figs 37-39
└── antagonistic_prompts.json           # 5 categories, 25 prompts
```

## Key Decisions Made

- **Model**: Llama 3.3 70B Instruct (int4 via bnb). Same model generates stories AND extracts (faithful to Anthropic).
- **Quantization**: `--load-in-4bit` everywhere. Verified quantization doesn't affect steering much (experiments/quant-sensitivity/).
- **Seed**: `--seed 42` for T=0.7 generation. Seed set once per generate_batch() call (batch-size independent). Saved in metadata.
- **171 Anthropic emotions** (not our own ~170 traits). Full word list in the methodology doc.
- **20 topics × 1 rollout** initially. Stage 0 pilot validates 20×1 vs 100×12 cosine similarity before committing.
- **T=0.7** for story diversity (Anthropic implied diversity in their prompt). T=0 is deterministic without seeds.
- **Extraction method**: mean_diff only. But Anthropic's method is NOT standard mean_diff — it's a cross-trait operation: mean per emotion → subtract grand mean → orthogonalize against neutral PCs.
- **Extraction position**: response[50:] (from extraction_config.yaml, not hardcoded).
- **Cross-trait normalization**: Separate script (cross_trait_normalize.py) runs AFTER per-trait extraction. Loads all 171 activations → grand mean subtraction → neutral PCA denoising → saves as "denoised" method vectors.
- **Single-polarity**: `polarity: single` in extraction_config.yaml → load_scenarios() skips negatives, neg.json is optional.
- **Decision gates**: 30-min baseline check before blackmail/RH steering sweeps (saves 20-40h if model doesn't exhibit behavior).
- **Post-training comparison caveat**: Llama 3.1 70B (base) vs 3.3 70B (instruct) are DIFFERENT model versions, not same model pre/post training.
- **Considered Qwen3.5 for thinking tokens** but Qwen3.5-72B doesn't exist (largest dense is 27B vision-language model). Qwen3-32B has thinking mode but base is gated. Saved thinking-token analysis as future extension.
- **Remote package manager**: `uv pip install` (not `pip install`)

## Code Changes Made This Session

### New files created:
- `core/math.py` additions: `pairwise_cosine_matrix()`, `pca()`, `project_out_subspace()`, `pearson_correlation()`
- `core/hooks.py`: `PerPositionSteeringHook` (steers at specific token positions)
- `core/_tests/test_math.py`: 20+ new tests (51+ total, all passing)
- `analysis/vectors/geometry.py`: trait_clusters, umap_projection, representational_similarity
- `analysis/vectors/preference_elo.py`: compute_preference_logits, compute_elo
- `utils/traits.py` additions: `load_trait_metadata()`, `load_extraction_config()` (category + per-trait cascade)
- All experiment scripts listed above

### Modified files:
- `extraction/run_extraction_pipeline.py`: extraction_config.yaml wiring, --seed flag, single-polarity generation
- `utils/extract_vectors.py`: neg.json optional, empty neg_acts handling
- `utils/model_generation.py`: seed parameter on generate_batch + _generate_batch_raw
- `core/kwargs_configs.py`: seed field on ExtractionConfig
- `core/methods.py`: MeanDiff handles empty neg_acts, Probe/Gradient/RFM raise clear errors
- `core/__init__.py`: exports for new functions + hook
- `docs/overview.md`: full v4 rewrite
- `visualization/styles.css`: mobile CSS fix
- `docs/viz_findings/index.yaml`: findings reorder
- Various doc updates: analysis/README.md, core_reference.md, architecture.md, extraction_guide.md, trait_dataset_creation.md, main.md

### Analysis restructure:
- `massive_activations.py`, `trait_correlation.py` → `analysis/vectors/`
- `data_checker.py` → `utils/experiment_data_checker.py`
- `analysis/sae/` → `dev/sae/`
- All path references updated

## Anthropic's Method vs Our Pipeline

Their extraction is a cross-trait operation that doesn't map to our per-trait mean_diff/probe:

1. Generate stories where characters experience each emotion (same model generates + extracts)
2. Run stories through model, capture residual stream activations from token 50 onward
3. Average across all stories for one emotion → per-emotion mean activation
4. Subtract grand mean across ALL 171 emotions → emotion-specific direction
5. Compute activations on neutral dialogues, take top PCs explaining 50% variance
6. Project PCs out of each emotion vector → denoised vectors

Our implementation:
- Steps 1-3: `extraction/run_extraction_pipeline.py --only-stage 1,3 --save-activations` (per-trait)
- Steps 4-6: `experiments/ant-emotion-concepts/scripts/cross_trait_normalize.py` (cross-trait post-processing)

The pipeline saves per-story mean activations as `[n_stories, hidden_dim]` tensors. The normalization script loads all 171, computes means, subtracts grand mean, PCA denoises, and saves final vectors with method name "denoised".

## Compute Estimates (1× A100 80GB, batch=8)

| Operation | Time |
|---|---|
| Story generation (6,840 × 256 tok) | ~7h |
| Neutral transcripts (~200 × 256 tok) | ~12min |
| 2-speaker dialogues (~3,000 × 768 tok) | ~9h |
| Deflection dialogues (~4,200 × 768 tok) | ~12h |
| Extraction (prefill) | ~30min |
| Geometry (CPU) | ~5min |
| Preference Elo (4,032 pairs) | ~15min |
| Steering — blackmail (IF gated) | ~22h |
| Steering — RH (IF gated) | ~8h |
| Post-training comparison | ~1h |
| **Without steering** | **~30h** |
| **With steering** | **~60h** |

## Known Gaps / Still Needed

1. **Russell & Mehrabian PAD norms** — need to digitize 45 emotions × valence/arousal for PCA cross-validation (Fig 8)
2. **LLM judge valence/arousal ratings** — 171 API calls to rate emotions 1-7. Stage 4 script outputs template JSON.
3. **Methodology placeholders** — 6 `:::placeholder:::` blocks in docs/methodology.md need deletion/replacement
4. **Per-experiment visualization** — token-level heatmaps (like their Figure 16), figure/table mapping, side-by-side comparison
5. **The scripts are long** (500-1000+ lines each). Shared.py exists (352 lines) but scripts could be shorter. Some duplication remains. The unique experiment logic is genuinely complex though.
6. **Stage 0 pilot** hasn't run — validates 20×1 vs 100×12 cosine similarity before committing to the full run
7. **LessWrong post** — not written yet. Structure finalized. Deadline Sunday April 12.
8. **Neutral corpus** — `_neutral` pseudo-trait needs positive.jsonl with neutral dialogue prompts (from Appendix A.3)
9. **Blackmail scenario reconstruction** — only 1 of 6 variants fully specified in the paper

## Git Status

Branch: dev, pushed to origin. R2: ant-emotion-concepts synced (--full).

Uncommitted changes from OTHER Claude Code sessions (not this one):
- docs/methodology.md, visualization changes (live chat), dev/ changes, inference/modal_inference.py

## To Start the Overnight Run

```bash
# On remote GPU instance:
git pull origin dev
./dev/r2_pull.sh --only ant-emotion-concepts
uv pip install transformers accelerate bitsandbytes

# Stage 0: Pilot (2-3h)
python3 experiments/ant-emotion-concepts/scripts/explore_story_generation.py \
    --model meta-llama/Llama-3.3-70B-Instruct --load-in-4bit

# Stage 1: Generate stories (7h)
python3 extraction/run_extraction_pipeline.py \
    --experiment ant-emotion-concepts \
    --category ant_emotion_concepts \
    --only-stage 1 \
    --load-in-4bit --seed 42

# Stage 2: Extract activations (30min)
python3 extraction/run_extraction_pipeline.py \
    --experiment ant-emotion-concepts \
    --category ant_emotion_concepts \
    --only-stage 3 --save-activations \
    --load-in-4bit

# Stage 2.5: Cross-trait normalization (CPU, 5min)
python3 experiments/ant-emotion-concepts/scripts/cross_trait_normalize.py \
    --experiment ant-emotion-concepts \
    --layer 53 \
    --neutral-trait ant_emotion_concepts/_neutral
```

Note: Stage 2.5 requires neutral corpus activations. Need to generate neutral transcripts first (from Appendix A.3 prompt template) and run them through the extraction pipeline as a `_neutral` pseudo-trait.
