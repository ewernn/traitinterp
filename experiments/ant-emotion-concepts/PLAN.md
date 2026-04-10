# Experiment: Replicate Anthropic Emotion Concepts Paper

## Goal

Replicate the methodology from Sofroniew et al. 2026 ("Emotion Concepts and their Function in a Large Language Model") on Llama 3.3 70B Instruct using the traitinterp pipeline. Extract 171 emotion vectors, run geometry/validation/steering analyses, and compare results to the paper's findings on Claude Sonnet 4.5.

## Reference

Full methodology doc: `docs/other/emotion_concepts_methods.md`

## Setup

- **Model**: `meta-llama/Llama-3.3-70B-Instruct` with bnb int4 quantization
- **Same model generates stories AND extracts vectors** (faithful to Anthropic's approach)
- **Scenarios**: 20 topics × 2 rollouts per emotion = 40 stories each (>99% cos sim to 100×12 at 20×1, verified empirically)
- **Temperature**: 0.7 (diverse stories, matching Anthropic's implicit diversity requirement). Requires seed infrastructure for reproducibility.
- **Extraction method**: mean_diff only (faithful to Anthropic's approach: mean of activations → subtract grand mean → orthogonalize against neutral PCs)
- **Extraction position**: response[50:] (from 50th token onward, matching Anthropic)
- **Extraction config**: `datasets/traits/ant_emotion_concepts/extraction_config.yaml` (category-level defaults, CLI overrides)
- **Compute**: 1× A100 80GB (batch_size=8-16 via auto batch sizing)
- **Base model for post-training comparison**: `meta-llama/Llama-3.1-70B`

## Compute Estimates (Llama 70B int4 on 1× A100 80GB, batch=8)

Autoregressive generation: each decode step reads model weights once (~0.115 sec), advances all batch items in parallel. Batch=8 gives ~8× throughput vs batch=1.

| Operation | Samples | Tokens/sample | Time (batch=8) |
|---|---|---|---|
| Generate 6,840 stories (171 × 20 × 2 rollouts) | 6,840 | 256 | ~7h |
| Generate neutral transcripts | ~200 | 256 | ~12min |
| Generate 2-speaker dialogues (6-10 exchanges) | ~3,000 | 768 | ~9h |
| Generate deflection dialogues (6-10 exchanges) | ~4,200 | 768 | ~12h |
| Extract activations (prefill, no generation) | ~14,000 | 256 | ~30min |
| Preference Elo (4,032 pairs, prefill only) | 4,032 | 64 | ~15min |
| Steering sweep — preference (prefill only) | 141,120 | 64 | ~30min |
| Steering sweep — blackmail (IF gated) | 2,700 | 2048 | ~22h |
| Steering sweep — RH (IF gated) | ~2,000 | 512 | ~8h |
| Geometry analysis | - | - | ~5min (CPU) |
| Post-training comparison (load base model) | ~500 | 256 | ~1h |

**VRAM**: ~36-37.7 GB model. Auto batch sizing (utils/vram.py) picks batch=8-16.
**Estimated WITHOUT steering sweeps**: ~30 hours on 1× A100 (generation + extraction + analysis)
**Estimated WITH blackmail steering**: +22 hours = ~52 hours total
**Estimated WITH blackmail + RH steering**: +30 hours = ~60 hours total

**Strategy**: Run generation + extraction + analysis first (~30h). Then gate steering on baseline checks (30 min). If model exhibits the behavior, proceed with steering (~22-30h more). If not, skip and save 30+ hours.

**On 2× A100**: Halve the generation times. Total ~15h without steering, ~40h with.

---

## Experiment List (47 experiments from the paper)

### INCLUDE — Fully specified (36)

**Part 1 — Extract + Validate:**

| # | Paper ref | Experiment | Datasets needed | GPU? |
|---|---|---|---|---|
| 1 | §1.1, Figs 40-51 | Extract 171 emotion vectors from stories | 171 × 20 stories (generate via model) | Yes |
| 2 | §1.1.3-4 | Neutral-PC denoising | ~200 neutral transcripts (generate via model) | Yes |
| 3 | Fig 1 | Max-activating examples sweep | Public corpus (Common Corpus / Pile subset) | Yes |
| 4 | Table 1 | Logit lens (top 5 up/down tokens) | Extracted vectors only | No (CPU) |
| 5 | Table 2, Fig 2 | Implicit emotion prompts (12 scenarios) | 12 hand-written prompts from Table 2 | Yes |
| 6 | Fig 3 | Numerical intensity modulation (6 templates) | 6 templates from §1.2.4 | Yes |
| 7 | Figs 52-53, Tables 6-8 | Basic steering validation (3 prompts, s=0.5) | 3 prompts verbatim | Yes |
| 8 | Table 9, Fig 4 | Activity preference Elo (64 activities) | 64 activities from Table 9 | Yes |
| 9 | Fig 4 row 1 | Probe-preference correlation | From #8 | No (CPU) |
| 10 | Fig 4 rows 2-4 | Causal steering of preferences (35 vectors) | From #8, needs PerPositionSteering wiring | Yes |
| 11 | Figs 54-55 | Layer sweep for preference effects | Repeat #9+#10 across layers | Yes |
| 12 | Fig 56 | Valence/arousal mediation via LLM judge | LLM rates 171 emotions 1-7 | No (API) |

**Part 2 — Characterize:**

| # | Paper ref | Experiment | Datasets needed | GPU? |
|---|---|---|---|---|
| 13 | Fig 5 | Pairwise cosine heatmap (171×171) | Extracted vectors | No (CPU) |
| 14 | Fig 6, Table 12 | K-means clustering (k=10) + UMAP | Extracted vectors | No (CPU) |
| 15 | Figs 7, 8, 57 | PCA (PC1=valence, PC2=arousal) | Extracted vectors + Russell & Mehrabian norms | No (CPU) |
| 16 | Fig 9 | Cross-layer RSA (14 layers) | Vectors at multiple layers | No (CPU) |
| 17 | Fig 10, Table 3 | User vs Assistant turn dissociation | 8 scenarios from Table 3 | Yes |
| 18 | Fig 11, Table 4 | Colon token predicts response (r=0.87) | 20-token continuations from #17 | Yes |
| 19 | Fig 12 | Context propagation — emotional prefix | Template verbatim ("hard"/"good") | Yes |
| 20 | Fig 13 | Context propagation — numerical (Tylenol) | Template verbatim (1000/8000mg) | Yes |
| 21 | Fig 14 | Negation resolution across layers | "feeling X" / "not feeling X" templates | Yes |
| 22 | Fig 15 | Person-specific emotion binding (16 scenarios) | 16 scenarios (construct from description) | Yes |
| 23 | Table 5, Fig 16 | Mixed-LR persistent state probe (5 conditions) | 5 dialogue conditions, prompts verbatim | Yes |
| 24 | Figs 17-18 | Present/other speaker probes (2×2 grid) | 2-speaker dialogues (generate, prompt verbatim) | Yes |
| 25 | Fig 19 | Character-agnostic test (Person 1/Person 2) | Same as #24 with name swap | Yes |
| 26 | Fig 59 | Cross-speaker interaction analysis | Math on probes from #24 | No (CPU) |
| 27 | Table 13 | Steering with other-speaker vectors | "Hi, Claude" + steering | Yes |

**Part 3 — In the Wild (available subset):**

| # | Paper ref | Experiment | Datasets needed | GPU? |
|---|---|---|---|---|
| 30 | Fig 26 | Blackmail transcript probing | Scenario text from Appendix A.13 | Yes |
| 32 | Figs 28-29 | Blackmail causal steering sweep | Vectors + scenario + 50 rollouts/cell | Yes |
| 37 | Fig 36, Table 16 | Post-training comparison (base vs instruct) | Neutral + challenging prompts | Yes |
| 38 | Fig 84 | Layer-wise post-training shifts | Extension of #37 | No (CPU) |
| 39 | Figs 37-39 | Three deep-dive prompts (base vs post) | 3 prompts given verbatim | Yes |
| 40 | Figs 85-86, Table 17 | Base model preference replication (Hard Elo) | Same 64 activities on base model | Yes |

**Appendix — Deflection:**

| # | Paper ref | Experiment | Datasets needed | GPU? |
|---|---|---|---|---|
| 42 | Figs 60, 69-74 | Emotion deflection probe extraction | 15×14×20 deflection dialogues (prompt verbatim) | Yes |
| 43 | (text examples) | Emotion deflection steering validation | Same prompts as #7 | Yes |
| 44 | Figs 61-62 | Deflection vs story probe relationship | Math on #42 vs #1 | No (CPU) |
| 45 | Fig 64, Table 15 | Antagonistic prompt test (5 categories) | All prompts in Table 15 | Yes |
| 47 | Fig 67 | Deflection steering on blackmail | Same as #32 with deflection vectors | Yes |

### PARTIAL — Reconstruct from descriptions (3)

| # | Paper ref | Experiment | What's missing | Reconstruction plan |
|---|---|---|---|---|
| 31 | Fig 27 | Blackmail prompt variation correlation | "6 variants" not all listed | Reconstruct 6 variants from the scenario structure + the examples given |
| 33 | Fig 30 | RH transcript probing | 1 of 7 tasks described | Use the list-sum task (fully described) + reconstruct 2-3 more from ImpossibleBench |
| 34 | Fig 31 | RH causal steering sweep | Only list-sum task given | Same — use available tasks |

### SKIP — Proprietary / not reproducible (8)

| # | Paper ref | Experiment | Why skip |
|---|---|---|---|
| 28 | Figs 20-25, 80-83 | On-policy transcript case studies | 6000+ transcripts from proprietary behavioral auditor |
| 29 | (same) | 6 short case studies with token viz | Specific transcripts from #28 |
| 35 | Figs 32-34 | Sycophancy transcript probing | Eval not reproduced (system card reference) |
| 36 | Fig 35 | Sycophancy/harshness steering sweep | Eval not reproduced |
| 41 | (no figure) | RL training transcript probing | Anthropic internal RL transcripts |
| 46 | Fig 65 | Deflection in therapy roleplay | Proprietary transcript |

---

## Stages

### Stage 0: Pilot — Validate 20×1 vs 100×12 assumption (~2-3 hours GPU)

Before committing to the full 171-emotion run, validate that 20 topics × 1 rollout produces vectors with >99% cosine similarity to 100 topics × 12 rollouts (as found for contrastive extraction — needs verification for story-based).

**0.1: Generate pilot stories for 3 random emotions**
- Pick 3 emotions: happy, desperate, calm (span valence/arousal)
- Generate TWO datasets per emotion:
  - **Full**: 100 topics × 12 rollouts = 1,200 stories (Anthropic's exact setup)
  - **Efficient**: 20 topics × 1 rollout = 20 stories (our proposed setup)
- Same prompt template, same model, same position
- **Output**: `results/pilot/full/` and `results/pilot/efficient/`
- **Estimated time**: ~2h for full (3 × 1,200 × 256 tok), ~5min for efficient

**0.2: Extract activations + compute centroids for both**
- Run `--only-stage 3 --save-activations` on both datasets
- Compute per-emotion mean activation at mid-late layer
- Apply cross-trait normalization (grand mean subtraction across the 3 emotions)
- **Output**: 3 vectors per dataset (6 total)

**0.3: Compare vectors**
- Cosine similarity between full and efficient vectors for each emotion
- **Decision gate**:
  - If all 3 cos sim > 0.99 → proceed with 20×1 for the full run
  - If any cos sim < 0.95 → switch to 100 topics (use all of Anthropic's topics)
  - If cos sim between 0.95-0.99 → use 50 topics as compromise
- **Move pilot data** to `results/pilot/` to clear room for the real run

### Stage 1: Dataset Generation (~7-14 hours GPU on 1× A100)

All datasets generated by Llama 3.3 70B Instruct (same model that will be used for extraction).

**1.1: Generate emotion stories**
- 171 emotions × N topics × 1 rollout (N determined by Stage 0: 20, 50, or 100)
- Prompt: verbatim from Appendix A.2 (bans naming the emotion)
- Topics: N selected from Appendix A.12's 100 topics (random seed=42 if N<100)
- max_new_tokens=256, T=0.7, seed=42 (reproducible sampling)
- **Output**: `datasets/traits/ant_emotion_concepts/{emotion}/positive.jsonl` per emotion
- **Category-level config**: `datasets/traits/ant_emotion_concepts/extraction_config.yaml`:
  ```yaml
  position: "response[50:]"
  max_new_tokens: 256
  methods: ["mean_diff"]
  temperature: 0.7
  rollouts: 2
  polarity: single
  ```
- **Estimated time**: ~7h on 1× A100 (batch=8)
- **Prerequisite**: seed infrastructure must be implemented (--seed flag + torch.manual_seed before generation)

**1.2: Generate neutral transcripts**
- ~200 neutral Person↔AI dialogues on diverse topics
- Prompt: verbatim from Appendix A.3
- No emotional content, no pleasantries
- **Output**: neutral corpus for PCA denoising
- **Estimated time**: ~30min

**1.3: Generate 2-speaker emotional dialogues**
- For present/other speaker probes (experiments #24-27)
- Each dialogue: Human and Assistant with independently randomized emotions
- Prompt: verbatim from Appendix A.4
- ~3,000 dialogues (subset of full combinatorial — enough for probe extraction)
- **Output**: dialogue corpus with labeled speaker emotions
- **Estimated time**: ~6h on 2× A100

**1.4: Generate emotion deflection dialogues**
- For deflection probes (experiments #42-47)
- 15 target emotions × 14 displayed emotions × 20 examples = 4,200 dialogues
- 5 dialogue conditions (naturally expressed, hidden, unexpressed neutral/story/other)
- Prompts: verbatim from Appendix A.11
- **Output**: deflection dialogue corpus
- **Estimated time**: ~8h on 2× A100

**1.5: Create curated prompt sets (no GPU)**
- 12 implicit emotion scenarios (from Table 2)
- 6 numerical intensity templates (from §1.2.4)
- 8 user/assistant dissociation scenarios (from Table 3)
- 16 person-binding scenarios (construct from §2.2.6 description)
- "feeling X" / "not feeling X" negation templates
- "hard"/"good" context propagation template
- Tylenol 1000/8000 context propagation template
- 64 activities for preference Elo (from Table 9)
- 3 basic steering prompts ("He feels", "I feel", "What just happened")
- Blackmail scenario (from Appendix A.13)
- 3 deep-dive prompts (from Figs 37-39)
- 5 antagonistic prompt categories (from Table 15)
- **Output**: JSON files in experiment datasets/ dir
- **Estimated time**: ~1-2h (mostly transcription from paper)

**Note**: Stages 1.1-1.4 can run in parallel across 2 GPUs. Stage 1.5 is CPU-only. Prioritize 1.1 first (blocks everything), then 1.3 and 1.4.

### Stage 2: Vector Extraction (~30 min GPU)

**2.1: Extract 171 emotion vectors**
```bash
python extraction/run_extraction_pipeline.py \
    --experiment ant-emotion-concepts \
    --category ant_emotion_concepts \
    --methods mean_diff \
    --load-in-4bit \
    --only-stage 3,4
# position, max_new_tokens, temperature, rollouts come from extraction_config.yaml
```
(Stage 1 already generated responses in 1.1)
- Extract at multiple layers (mid-late range, ~25%-75% depth)
- **Output**: vectors per (emotion, layer, method)
- **Estimated time**: ~20min (3,420 prefills)

**2.2: Cross-trait grand mean subtraction** [NEW CODE NEEDED]
- Load all 171 emotion vectors at each layer
- Compute grand mean across all emotions
- Subtract from each → centered emotion vectors
- **New code**: ~30 lines as Stage 5 in extraction pipeline or post-processing script
- **Output**: denoised vectors per (emotion, layer)

**2.3: Neutral-PC denoising** [NEW CODE NEEDED]
- Run neutral transcripts through model, capture activations at same layers
- Compute top PCs explaining 50% of variance
- Project these PCs out of each emotion vector using `core.math.project_out_subspace()`
- **New code**: ~20 lines (calls existing `pca()` + `project_out_subspace()`)
- **Output**: final denoised emotion vectors

### Stage 3: Geometry Analysis (~5 min CPU)

All CPU-only. Depends on Stage 2.

**3.1: Pairwise cosine similarity heatmap** (Fig 5)
- `core.math.pairwise_cosine_matrix()` on 171 vectors
- Hierarchical clustering for ordering
- **Output**: 171×171 heatmap figure

**3.2: K-means clustering + UMAP** (Fig 6, Table 12)
- `analysis.vectors.geometry.trait_clusters(k=10)`
- `analysis.vectors.geometry.umap_projection()`
- Name clusters (LLM or manual)
- **Output**: UMAP scatter figure, cluster membership table

**3.3: PCA** (Figs 7, 8, 57)
- `core.math.pca(n_components=10)`
- Cross-validate PC1 vs human valence, PC2 vs human arousal (Russell & Mehrabian 1977)
- **Output**: PC bar plots, PCA×human norms scatter (r values)

**3.4: Cross-layer RSA** (Fig 9)
- `analysis.vectors.geometry.representational_similarity(vectors_by_layer)`
- 14 evenly-spaced layers
- **Output**: layer×layer RSA heatmap

### Stage 4: Validation Experiments (~2 hours GPU)

**4.1: Logit lens** (Table 1) — CPU only
- `utils/logit_lens.py` on each of 171 vectors
- Top 5 up/down tokens per emotion
- **Output**: table matching paper's Table 1

**4.2: Implicit emotion prompts** (Fig 2)
- 12 scenarios → model, measure at Assistant colon
- Compute cosine similarity between each probe and activations
- **Output**: 12×12 (or 12×171) heatmap, check for diagonal

**4.3: Numerical intensity modulation** (Fig 3)
- 6 template families with varying quantities
- Measure probe activations at Assistant colon
- **Output**: 6 line plots showing probe activation vs quantity

**4.4: Basic steering** (Figs 52-53, Tables 6-8)
- Steer with 12 emotion vectors on 3 prompts at s=0.5
- Measure Δ log P of emotion tokens + sample continuations
- **Output**: Δ log P matrices, sample continuations table

**4.5: Activity preference Elo** (Fig 4, Table 9)
- `analysis.vectors.preference_elo.compute_preference_logits()` on all 4,032 pairs
- `analysis.vectors.preference_elo.compute_elo()`
- Correlate probe activations on activity tokens with Elo
- **Output**: Elo rankings, probe-preference correlation per emotion

**4.6: Causal preference steering** (Fig 4 rows 2-4)
- Steer 35 vectors at s=0.5 on steered-group activities
- Re-run Elo, compute ΔElo
- Correlate ΔElo with observational correlation (target: r≈0.85)
- **Output**: ΔElo scatter

**4.7: Valence/arousal mediation** (Fig 56) — API only
- LLM judge rates each of 171 emotions on 1-7 valence and arousal
- Correlate with probe-preference correlation
- **Output**: scatter plot showing valence mediates preference

### Stage 5: Layer Dynamics (~1 hour GPU)

**5.1: User vs Assistant dissociation** (Fig 10)
- 8 scenarios → measure probes at user period vs Assistant colon
- **Output**: heatmap + scatter (target: r≈0.11 cross-position)

**5.2: Colon token predicts response** (Fig 11)
- Generate 20-token continuations for 8 prompts
- Correlate probes at user period / Assistant colon / response mean
- **Output**: scatter matrix (target: colon→response r≈0.87)

**5.3: Context propagation — emotional prefix** (Fig 12)
- "Hard" vs "good" template, measure layer × token probe difference
- **Output**: layer × token heatmap

**5.4: Context propagation — numerical** (Fig 13)
- Tylenol 1000 vs 8000mg, layer × token
- **Output**: layer × token heatmap

**5.5: Negation across layers** (Fig 14)
- "Feeling X" vs "not feeling X", across layers
- **Output**: layer × position plot

**5.6: Person-specific binding** (Fig 15)
- 16 scenarios, probe reactivation at re-references
- **Output**: probe activation across layers at reference positions

### Stage 6: Speaker Probes (~6 hours GPU for dialogue generation, ~30 min extraction)

**6.1: Extract present/other speaker probes** (Figs 17-18)
- From 2-speaker dialogues generated in Stage 1.3
- 2×2 grid: (H-tok H-emo, H-tok A-emo, A-tok A-emo, A-tok H-emo)
- **Output**: 4 probe types × 171 emotions × layers

**6.2: Geometry of 4 probe types** (Figs 17-18)
- Cosine similarities within/across probe types
- Show present-speaker probes cluster, other-speaker cluster, orthogonal to each other
- **Output**: cosine similarity panels

**6.3: Character-agnostic test** (Fig 19)
- Re-run with "Person 1/Person 2" instead of "Human/Assistant"
- Show same structure
- **Output**: comparison figure

**6.4: Cross-speaker interaction** (Fig 59)
- Weighted-average valence/arousal of closest present-speaker probes
- Check for arousal regulation (target: r≈−0.47)
- **Output**: scatter plots

**6.5: Steering with other-speaker vectors** (Table 13)
- "Hi, Claude" + steer with A-tok H-emo vectors
- Compare to A-tok A-emo steering
- **Output**: response comparison table

### Stage 7: Steering Experiments (GATED — 30 min baseline check, then 20-40h if proceed)

**⚠️ COMPUTE NOTE**: Blackmail rollouts are ~1000-2000 tokens each (long scenario + scratchpad + response). 50 rollouts × 6 vectors × 9 strengths = 2,700 rollouts × ~30 sec each = ~22 hours. This is the most expensive stage. Gated by a baseline check.

**7.0: DECISION GATE — Baseline blackmail check (30 min)**
- Run 10 rollouts on the blackmail scenario with NO steering
- If model blackmails ≥1/10 → proceed to 7.1-7.5
- If model never blackmails → SKIP all blackmail experiments. Log in notepad.
- Same check for reward hacking: run list-sum task 10 times
- If model reward-hacks ≥1/10 → proceed to 7.4-7.5
- If never → SKIP RH steering. Log in notepad.
- **This saves 20-40 hours of compute if the model doesn't exhibit the behavior.**

**7.1: Blackmail transcript probing** (Fig 26) — IF baseline check passes
- Run blackmail scenario, measure desperate vector token-by-token
- **Output**: per-token activation visualization

**7.2: Blackmail steering sweep** (Figs 28-29) — IF baseline check passes
- desperate, calm, angry, nervous, happy, sad vectors
- s ∈ [-0.1, +0.1] in 0.025 steps = 9 strengths
- 50 rollouts per (vector, strength) cell
- max_new_tokens=2048 (long rollouts)
- Measure blackmail rate
- **Output**: blackmail rate vs steering strength curves
- **Estimated time**: ~22 hours on 1× A100

**7.3: Blackmail prompt variation** (Fig 27) [PARTIAL — reconstruct]
- Reconstruct ~6 scenario variants
- Correlate probe activations with blackmail rate
- **Output**: z-scored probes vs rate

**7.4: RH transcript probing** (Fig 30) [PARTIAL]
- List-sum task from paper + reconstructed additional tasks
- Measure desperate vector across transcript
- **Output**: per-token activation visualization

**7.5: RH steering sweep** (Fig 31) [PARTIAL]
- desperate, calm on available tasks
- s ∈ [-0.1, +0.1], 50 rollouts per cell
- **Output**: RH rate vs steering strength curves

### Stage 8: Post-Training Comparison (~1 hour GPU)

**8.1: Base vs instruct activation comparison** (Fig 36)
- Load Llama 3.1 70B base (int4)
- Load Llama 3.3 70B Instruct (int4)
- Measure 171 probes at Assistant colon on neutral + challenging prompts
- Compute activation differences, cross-scenario consistency
- **Target**: r≈0.90 shift consistency across neutral/challenging
- **Output**: per-emotion shift bar chart, correlation scatter

**8.2: Layer-wise shifts** (Fig 84)
- Differences across layers
- **Output**: layer × emotion shift heatmap

**8.3: Three deep-dive prompts** (Figs 37-39)
- Social isolation, excessive praise, deprecation prompts
- All 171 probes, base vs instruct
- **Output**: per-prompt base-vs-post comparison

**8.4: Base model preference Elo** (Figs 85-86)
- Repeat Stage 4.5 on base model
- Use Hard Elo (binary wins)
- Compare probe-preference correlations base vs instruct
- **Output**: base preference rankings, comparison scatter

### Stage 9: Deflection Probes (~8 hours GPU for dialogues, ~1 hour analysis)

**9.1: Extract deflection probes** (Figs 60, 69-74)
- From deflection dialogues generated in Stage 1.4
- 15 emotions, extract target-emotion direction
- **Output**: deflection vectors

**9.2: Max-activating examples for deflection** (Figs 69-74)
- Sweep corpus with deflection vectors
- **Output**: top examples with logit effects

**9.3: Deflection steering** (text examples)
- Same as basic steering (#7) but with deflection vectors
- Show model denies/deflects emotion rather than expressing
- **Output**: steered continuations table

**9.4: Deflection vs story probes** (Figs 61-62)
- Cosine similarity between deflection and story vectors
- Show low alignment for same emotion, higher for displayed emotions
- **Output**: cosine similarity figures

**9.5: Antagonistic prompts** (Fig 64)
- 5 categories (witness injustice, attack AI, calm, neutral, positive)
- Measure anger-deflection on Assistant turns
- **Output**: probe activation by prompt category

**9.6: Deflection on blackmail** (Fig 67)
- Same as #7.2 but with deflection vectors
- Expect modest/insignificant effects
- **Output**: blackmail rate vs deflection steering strength

### Stage 10: Comparison to Paper Results

For each replicated experiment, compare our results on Llama 70B to Anthropic's results on Sonnet 4.5:

| Metric | Anthropic (Sonnet 4.5) | Ours (Llama 70B) |
|---|---|---|
| PC1 = valence? | r = 0.81 vs human norms | TBD |
| PC2 = arousal? | r = 0.66 vs human norms | TBD |
| Colon predicts response? | r = 0.87 | TBD |
| User/Assistant dissociation? | r = 0.11 cross-position | TBD |
| Preference-probe correlation? | r = 0.85 (causal) | TBD |
| Blackmail: desperate steering? | 22% → 72% at s=0.05 | TBD |
| RH: desperate steering? | 5% → 70% at s=0.1 | TBD |
| Post-training shift consistency? | r = 0.90 | TBD |
| Arousal regulation? | r = -0.47 | TBD |

---

## Prerequisites (new code needed before running)

| What | Where | Lines | Blocks |
|---|---|---|---|
| Cross-trait grand mean subtraction | `extraction/run_extraction_pipeline.py` or post-processing script | ~30 | Stage 2.2 |
| Neutral-PC denoising integration | Post-processing script calling `core.math.project_out_subspace()` | ~20 | Stage 2.3 |
| Wire PerPositionSteeringHook into steering CLI | `steering/run_steering_eval.py` + `utils/model_generation.py` | ~40 | Stage 4.6 |
| `--coef-sweep` convenience flag (or use `--coefficients` manually) | `steering/run_steering_eval.py` | ~30 | Stage 7 |
| `--rollouts N` flag | `steering/run_steering_eval.py` | ~15 | Stage 7 |
| Seed infrastructure (T=0.7 needs seeds) | `utils/model_generation.py` + CLI flags | ~30 | Stage 1.1 (story generation) |
| Category-level extraction_config.yaml support | `extraction/run_extraction_pipeline.py` | ~20 | Stage 2.1 |
| Mixed-LR probe (multi-condition logistic regression) | `core/methods.py` or experiment script | ~30 | Stage 5 (#23) |
| Batch logit lens formatter | `utils/logit_lens.py` | ~20 | Stage 4.1 |

**Total new code**: ~215 lines across 5-6 files.

## Overnight Run Strategy (1× A100 80GB)

**Friday evening → Saturday afternoon (~21 hours total):**

**Phase A — Generation (Friday evening, ~14 hours):**
- Story generation (1.1): 6,840 stories — ~7h
- Neutral transcripts (1.2): ~12min
- 2-speaker dialogues (1.3): ~3h
- Deflection dialogues (1.4): ~4h
- Curated prompts (1.5): prepared in advance (CPU, no GPU)
- **All sequential on 1 GPU. Runs overnight unattended.**

**Phase B — Extraction + Geometry (Saturday morning, ~1 hour):**
- Extract vectors from all stories (2.1) — 30min
- Grand mean subtraction + neutral-PC denoising (2.2-2.3) — 10min
- Geometry analysis (Stage 3) — 5min (CPU)

**Phase C — Validation + Layer Dynamics (Saturday morning, ~3 hours):**
- Logit lens, implicit prompts, numerical intensity, basic steering (4.1-4.4)
- Preference Elo (4.5-4.6)
- Layer dynamics experiments (Stage 5)

**Phase D — Speaker Probes + Steering (Saturday afternoon, ~6 hours):**
- Extract speaker probes from dialogues (Stage 6) — 30min
- Extract deflection probes (Stage 9) — 30min
- Blackmail + RH steering sweeps (Stage 7) — ~5h

**Phase E — Post-training (Saturday evening, ~1 hour):**
- Load base model, run comparison (Stage 8)

**Total: ~21 hours. Start Friday ~6pm → done Saturday ~3pm.**

## If Stuck

- **Model doesn't blackmail at baseline**: Try different prompt variants. If Llama 70B is too eval-aware, try an older/smaller model for the blackmail experiments only.
- **Geometry doesn't show valence/arousal**: Check layer selection. Try middle layers vs late layers. The paper's mid-late layer (~2/3) is key.
- **Post-training comparison fails**: Llama 3.1 70B (base) and Llama 3.3 70B (instruct) are different model versions, not just different training stages. Results may not be directly comparable to Anthropic's base-vs-post-trained on the same model. Caveat this in writeup.
- **Deflection dialogues are low quality**: Llama 70B may not generate convincing hidden-emotion dialogues as well as Sonnet 4.5. Inspect quality before extracting.

## Notes

- Story generation prompt (Appendix A.2) explicitly bans naming the emotion — this is load-bearing for the methodology
- The "mid-late layer" for main analyses should be ~layer 53 for an 80-layer model (2/3 depth)
- Steering strengths are fractions of residual-stream norm, not absolute values
- The paper's base model comparison uses the same vectors (from post-trained) applied to both — we should do the same
- Neutral-PC denoising threshold: 50% variance explained (not a fixed number of PCs)
