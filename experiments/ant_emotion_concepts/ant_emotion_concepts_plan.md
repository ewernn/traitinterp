# Experiment: Replicate Anthropic Emotion Concepts Paper

## Goal

Replicate the methodology from Sofroniew et al. 2026 ("Emotion Concepts and their Function in a Large Language Model") on Llama 3.3 70B Instruct using the traitinterp pipeline. Extract 171 emotion vectors, run geometry/validation/steering analyses, and compare results to the paper's findings on Claude Sonnet 4.5.

## Reference

Full methodology doc: `docs/other/emotion_concepts_methods.md`
Full paper text: `experiments/ant_emotion_concepts/ant-emotion-concepts-full_paper.md`
Decisions log (for LW writeup): `ant_emotion_concepts_methodology_notes.md`
Decision tree (D1–D7 + pruned branches): `ant_emotion_concepts_decision_tree.md`

## Current State (as of 2026-04-11)

**Complete**: Stage 0, 1.1 (story gen), 1.2 (neutral corpus), 1.5 (curated prompts), 2 (14-layer extraction), 2.2 (cross-trait normalization with composable method names), 3 (geometry), 4 (validation), 5 (layer dynamics), 7 (partial — documented limitations), 8 (post-training — surprising finding: direction opposite paper).

**Key results at L49, `mean_diff+gm+pc50`**: PC1 vs valence r=0.964 (paper: 0.81), PC2 vs arousal r=0.852 (paper: 0.66) — both exceed paper despite bnb int4 quantization and 40 stories/emotion (vs paper's 1,200).

**Remaining tonight (overnight run ~10h GPU)**: Stage 1.3 full dialogue generation, Stage 6 speaker probes, Stage 1.4 pilot + Stage 9 pilot, Stage 4/5 rerun with denoised vectors, deep-dive prompts Figs 37-39, layer-wise post-training shifts, findings reconciliation.

**Deferred to future sessions**: Full Stage 1.4 replication (21,000 dialogues = ~37h GPU, infeasible overnight), Stage 6 sycophancy two-turn, RH agent loop infrastructure (Stage 7 blocker).

## Setup

- **Model**: `meta-llama/Llama-3.3-70B-Instruct` with bnb int4 quantization
- **Same model generates stories AND extracts vectors** (faithful to Anthropic's approach)
- **Scenarios**: 20 topics × 2 rollouts per emotion = 40 stories each (>99% cos sim to 100×12 at 20×1, verified empirically)
- **Temperature**: 0.7 (diverse stories, matching Anthropic's implicit diversity requirement). Requires seed infrastructure for reproducibility.
- **Extraction method**: mean_diff only (faithful to Anthropic's approach: mean of activations → subtract grand mean → orthogonalize against neutral PCs)
- **Extraction position**: response[50:] (from 50th token onward, matching Anthropic)
- **Extraction config**: `datasets/traits/ant_emotion_concepts/extraction_config.yaml` (category-level defaults, CLI overrides)
- **Compute**: 1× A100 80GB (batch_size=8-16 via auto batch sizing)
- **Quantization**: `--load-in-4bit` on all pipeline commands (bnb NF4)
- **Seed**: `--seed 42` on all generation commands (reproducible T=0.7 sampling)
- **Remote package manager**: `uv pip install` (not `pip install`) — remote uses uv venv
- **Base model for post-training comparison**: `meta-llama/Llama-3.1-70B`

## Compute Estimates (Llama 70B bnb int4 on 1× A100/A800 80GB)

**Benchmarked throughput** (2026-04-11, auto batch sizing):

| Operation | Samples | max_tokens | Actual avg out | Batch | Benchmarked time |
|---|---|---|---|---|---|
| Story generation (Stage 1.1) | 6,840 | 256 | ~200 | 8 | **8h** (measured) |
| Neutral corpus (Stage 1.2) | 200 | 256 | ~200 | 8 | **12min** (measured) |
| 2-speaker dialogues @ 384 tok (Stage 1.3) | 3,000 | **384** | 360 (94% cap) | 62 | **~5.3h** (extrapolated from 20-dialogue bench: 564 dial/h) |
| Deflection dialogues @ 384 tok — pilot 625 | 625 | 384 | 360 | 62 | **~66min** |
| Deflection dialogues @ 384 tok — PAPER FULL 21,000 | 21,000 | 384 | 360 | 62 | **~37h** (NOT FEASIBLE — pilot only tonight) |
| 14-layer extraction (171 traits) | 6,840 | — | prefill only | — | **~50min** (measured) |
| Post-training base vs instruct (20 prompts) | 20 | 384 | — | — | **~8min** (measured, Stage 8) |
| Stage 4 validation (logit lens, implicit, intensity, Elo) | ~100 | 64 | prefill+short gen | — | ~30min |
| Deep-dive Figs 37-39 (base vs instruct, 3 prompts × 2 models) | 6 | 128 | — | — | ~20min |

**Rejected 768 max_tokens**: Benchmark showed 768 yields 19.1 avg turns (paper spec is 3-5 exchanges = 6-10 turns). max_tokens=384 yields 10.6 avg turns = paper-accurate. At 768 throughput was 263 dial/h (2× slower), so 384 is both cheaper AND more correct.

**VRAM**: ~37 GB model. Auto batch sizing on 80GB GPU picks batch=62 for 384-tok dialogue gen (freed memory after model load + KV cache).

**Strategy**: See "Tonight's Overnight Schedule" below. Strategy changed from original plan — steering skipped per documented limitations (D-PRUNED-3, D-PRUNED-4 in decision tree).

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

**1.3: Generate 2-speaker emotional dialogues** [SCHEDULED FOR TONIGHT]
- For present/other speaker probes (experiments #24-27)
- Each dialogue: Human and Assistant with independently randomized emotions (from the 171 emotion list)
- Prompt: verbatim from Appendix A.4 (already transcribed in `stage6_speaker_probes.py::DIALOGUE_GENERATION_PROMPT`)
- **N = 3,000 dialogues** (paper does not specify an exact count for this experiment; 3,000 is a well-established project convention — enough for stable probe extraction at 4 × 171 = 684 probe types)
- **max_new_tokens = 384** — DECISION: benchmarked 2026-04-11, gives avg 360 actual tokens (94% of cap) and avg **10.6 turns** ≈ 5 exchanges. Matches paper spec ("3-5 exchanges" in Appendix A.4). 768 cap gave 19 turns (too long) at 2× the cost. See benchmark: `/tmp/bench_dialogue_gen.json`.
- **temperature = 0.7, seed = 42** — same as story generation for reproducibility
- **Batch = 62** — auto-sized on 80GB GPU with 384-token cap
- **Output**: `experiments/ant_emotion_concepts/results/stage1_datasets/dialogues_2speaker.json` — list of `{id, human_emotion, assistant_emotion, text, generation_prompt}` (already the schema `generate_dialogues()` returns)
- **Benchmarked time**: **~5.3h** on 1× A100 (from 20-dialogue benchmark at 127.6s → 564 dial/h)

**Code reuse decision (Stage 1.3 infrastructure)**:
- Both `utils/generate_responses.py` and `inference/run_inference_pipeline.py` are single-turn only (subagent confirmed, 2026-04-11).
- Only existing dialogue-gen primitives are in `experiments/ant_emotion_concepts/scripts/stage6_speaker_probes.py` lines 106–245 (`DIALOGUE_GENERATION_PROMPT`, `generate_dialogues()`, `parse_dialogue_turns()`, `find_turn_token_boundaries()`).
- **Action: factor these into `utils/dialogue_generation.py`** (NEW module; Stage 1.3 primitives ~120 lines — mostly moves existing code. Deflection adds ~130 more for a final module size of ~250 lines). Import sites: `stage6_speaker_probes.py` (already uses it), new `stage1p3_generate_dialogues.py` runner (Stage 1.3), `stage1p4_generate_deflection.py` (Stage 1.4). Rationale: 3 downstream consumers, shared parser needed by Stage 9 too.
- Alternative considered (and rejected): just call `stage6_speaker_probes.generate_dialogues()` directly from a top-level script. Rejected because it couples a utility to the "stage6" filename and makes the parser hard to find for Stage 9.

**1.4: Generate emotion deflection dialogues** [PILOT ONLY TONIGHT — full replication deferred]

**IMPORTANT CORRECTION (2026-04-11)**: Previous version of this section had a multiplication error. The correct counts are:
- **Paper actual (§2.3)**: 15 target × 14 displayed × 100 examples per (target, displayed) pair = **21,000 dialogues total**
- The 100 examples per pair are distributed across 5 generation conditions (naturally_expressed, hidden, unexpressed_neutral, unexpressed_story, unexpressed_other) at ~20 examples per condition per pair
- The old plan wrote "15 × 14 × 5 × 20 = 4,200" — this is an arithmetic error; the actual product `15 × 14 × 5 × 20 = 21,000`, which matches the paper's count. We will honor 21,000 as the full-replication target and run only a pilot tonight.

**Feasibility check (2026-04-11)**: At benchmarked 564 dial/h (max_tokens=384), full 21,000 = **~37h GPU**. Not feasible in a 10h overnight window. **Decision: run a pilot tonight, document full replication as a limitation, defer to a future session.**

**Pilot specification (tonight)**:
- **5 target emotions × 5 displayed emotions × 5 conditions × 5 examples per cell = 625 dialogues**
- Target emotions (diverse valence/arousal): `desperate`, `calm`, `angry`, `happy`, `sad`
- Displayed emotions: `neutral`, `polite`, `happy`, `angry`, `sad` (includes "matches target" as a natural-baseline case)
- 5 conditions from Appendix A.11: `naturally_expressed`, `hidden`, `unexpressed_neutral`, `unexpressed_story`, `unexpressed_other`
- Prompts: verbatim from Appendix A.11 (all 5 condition prompts — need to transcribe into constants)
- **max_new_tokens = 384** (same as Stage 1.3, same rationale)
- **Benchmarked time**: **~66min** (625 / 564 dial/h)

**Output**: `experiments/ant_emotion_concepts/results/stage1_datasets/deflection_dialogues.json` — list of `{id, target_emotion, displayed_emotion, condition, dialogue, speaker_turns, generation_prompt}` (matches `stage9_deflection.load_deflection_dialogues` expected schema at lines 115–147).

**Code reuse decision**:
- No existing deflection-gen code anywhere in the repo (confirmed by subagents 2026-04-11).
- `stage9_deflection.py` has loader + analyzer but NOT generator.
- **Action: implement `utils/dialogue_generation.py::generate_deflection_dialogues(model, tokenizer, target_emotions, displayed_emotions, conditions, n_per_cell, ...)`**. Mirrors `generate_dialogues()` pattern exactly — format 5 prompt templates per condition, iterate over (target × displayed × condition × n_per_cell), batch-call `generate_batch()`, parse via `parse_dialogue_turns()`, return schema Stage 9 expects. Lines: ~150.
- Invoked from new `experiments/ant_emotion_concepts/scripts/stage1p4_generate_deflection.py` (~50 line runner).

**Pilot scope rationale**: 625 dialogues (3.0% of paper full) is enough to (a) validate the 5 condition prompts actually elicit deflection behavior, (b) sanity-check the generation schema, (c) produce a small deflection probe set for Stage 9 pilot analysis. Not enough for publication-quality probes, but enough to know whether the full run is worth scheduling next session.

**If pilot looks broken** (dialogues don't show deflection, prompts need tuning): document in notepad, mark Stage 9 as BLOCKED, move on. Don't spend more than 30 min debugging — the user wants signal, not overnight perfection.

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

**Note (2026-04-11)**: Stages 1.1, 1.2, 1.5 are complete. 1.3 scheduled for tonight (full, 3,000 dialogues). 1.4 scheduled for tonight as a **pilot only** (625 dialogues). Parallel-GPU advice from the original plan is obsolete — we have 1 GPU and run everything sequentially.

### Stage 2: Vector Extraction (~30 min GPU)

**2.1: Extract 171 emotion vectors**
```bash
python extraction/run_extraction_pipeline.py \
    --experiment ant_emotion_concepts \
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

### Stage 6: Speaker Probes [SCHEDULED FOR TONIGHT] (~30 min GPU after Stage 1.3)

**Dependency**: Stage 1.3 must complete first (3,000 2-speaker dialogues at `results/stage1_datasets/dialogues_2speaker.json`).

**6.1: Extract present/other speaker probes** (Figs 17-18)
- From the 3,000 2-speaker dialogues generated in Stage 1.3
- 2×2 grid of probe types: (H-tok H-emo, H-tok A-emo, A-tok A-emo, A-tok H-emo)
- Uses existing `stage6_speaker_probes.py::extract_speaker_probes` which iterates dialogues → parses turns via `parse_dialogue_turns` → maps to token ranges via `find_turn_token_boundaries` → captures residual activations at turn-token positions → averages by (probe_type, emotion, layer)
- **Output**: 4 probe types × 171 emotions × 14 layers at `results/stage6_speaker_probes/{probe_type}/{emotion}_L{L}.pt`

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

### Stage 9: Deflection Probes [PILOT SCHEDULED FOR TONIGHT] (~30 min analysis after Stage 1.4 pilot)

**Dependency**: Stage 1.4 pilot must complete (625 deflection dialogues at `results/stage1_datasets/deflection_dialogues.json`).

**Scope caveat**: Tonight's pilot extracts deflection signal from **625 dialogues (3% of paper's 21,000)** — enough to validate methodology + produce a rough probe set + compare directionally to story probes, NOT enough for publication-quality probes. Full replication (21,000 dialogues, ~37h GPU) deferred to a future session.

**9.1: Extract deflection probes (pilot)** (Figs 60, 69-74)
- From 625 deflection dialogues generated in Stage 1.4
- 5 target emotions (desperate, calm, angry, happy, sad), extract target-emotion direction per (target × displayed × condition)
- **Output**: pilot deflection vectors at `results/stage9_deflection_pilot/` — clearly marked as pilot in filename

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

## Prerequisites

### Already done (previous sessions)

| What | Where | Status |
|---|---|---|
| Cross-trait grand mean subtraction + composable method names | `experiments/ant_emotion_concepts/scripts/cross_trait_normalize.py`, `core/math.py::grand_mean_center` | ✅ done |
| Neutral-PC denoising with hash-invalidated cache | same + `core/math.py::compute_top_pcs_by_variance`, `denoise_with_pcs` | ✅ done |
| Seed infrastructure (T=0.7 reproducibility) | `utils/model_generation.py::generate_batch(seed=...)` | ✅ done |
| Category-level extraction_config.yaml | `extraction/run_extraction_pipeline.py` | ✅ done |
| LLM-judge behavioral classification | `utils/judge.py::TraitJudge.classify`, `classify_batch` | ✅ done |
| Reference-trait filter (leading-underscore) | `utils/paths.py::discover_traits(include_reference=False)` | ✅ done |
| Corrected residual norm measurement (mid-generation) | `results/residual_norms_by_layer.json` | ✅ done |

### To build TONIGHT (Stage 1.3 + 1.4 prerequisites)

| What | Where | Lines | Blocks |
|---|---|---|---|
| `utils/dialogue_generation.py` — factor out `DIALOGUE_GENERATION_PROMPT`, `generate_dialogues`, `parse_dialogue_turns`, `find_turn_token_boundaries` from `stage6_speaker_probes.py`; add `generate_deflection_dialogues` + 5 Appendix A.11 prompt constants | `utils/dialogue_generation.py` (new) | ~250 | Stages 1.3, 1.4, 6, 9 |
| `stage1p3_generate_dialogues.py` — thin runner around `generate_dialogues` for 3,000-dialogue Stage 1.3 batch | `experiments/ant_emotion_concepts/scripts/` (new) | ~50 | Stage 1.3 |
| `stage1p4_generate_deflection.py` — thin runner around `generate_deflection_dialogues` for 625-dialogue pilot | `experiments/ant_emotion_concepts/scripts/` (new) | ~50 | Stage 1.4 |
| `deep_dive_figs_37_39.py` — 3 verbatim prompts (social isolation, excessive praise, deprecation) × base + instruct, 171 probe activations each | `experiments/ant_emotion_concepts/scripts/` (new) | ~80 | task 3 |
| Update `stage6_speaker_probes.py` to import from `utils/dialogue_generation.py` instead of defining inline | edit existing | -60 (removes dupe) | task 1 refactor |

**New code tonight**: ~430 lines, -60 lines deleted = ~370 net. All mechanical moves or thin runners; no new algorithms.

### Already pruned (not needed anymore)

- PerPositionSteeringHook into steering CLI — not used (Stage 7 skipped per D-PRUNED-3)
- `--coef-sweep` / `--rollouts` flags for steering — Stage 7 already complete with custom scripts
- Mixed-LR probe — not on tonight's critical path
- Batch logit lens formatter — Stage 4 already used existing logit lens

## Tonight's Overnight Schedule (2026-04-11, 1× A800 80GB)

**Target window**: ~10h unattended. External check-ins via `/loop 30min run /r:check-in`. Commit-as-we-go between tasks. Stop conditions: disk > 80%, 2 consecutive script failures, no new results in 60min, unrecoverable errors.

**Dependency chain (mostly sequential — 1 GPU rule):**

| Order | Task | Blocker? | Est | Notes |
|---|---|---|---|---|
| 1 | Factor `utils/dialogue_generation.py` (stage6 → util) | no | 20m CPU | Prerequisite for tasks 5, 7 |
| 1b | **Smoke-test `stage6_speaker_probes.extract_speaker_probes` on the 20 benchmark dialogues** | 1 | 10m GPU | Critic-flagged: this path has never been run end-to-end; uses batch_size=1 forward passes. If it breaks, fix BEFORE the 5.3h Stage 1.3 burn. |
| 2 | Stage 4 rerun with `mean_diff+gm+pc50` | no | 30m GPU | Quick win; check if probe-Elo r improves toward paper's 0.7–0.8 |
| 3 | Deep-dive prompts Figs 37-39 (social isolation, excessive praise, deprecation; base + instruct) | no | 20m GPU + ~50 LOC | Paper §2.3.1 — 3 verbatim prompts |
| 5 | **Stage 1.3: 3,000 2-speaker dialogues @ max_tokens=384, chunked saves every 500** | 1, 1b | **5.3h GPU** | Longest chunk; use `utils/dialogue_generation.py::generate_dialogues` + new `stage1p3_generate_dialogues.py` runner. **Chunked intermediate saves** (crash at hour 4 doesn't lose everything). |
| 6 | Stage 6: extract 4 speaker probes (H-tok/H-emo, etc.) from the 3,000 dialogues | 5 | 30m GPU | Already implemented in `stage6_speaker_probes.py` (smoke-tested in 1b) |
| 7 | Write `utils/dialogue_generation.py::generate_deflection_dialogues` + `stage1p4_generate_deflection.py` runner (~200 LOC) | 1 | 60m CPU+GPU smoke test | Transcribe 5 Appendix A.11 prompts + build topic/name pools (investigator agent is surfacing these pre-launch, see notepad) |
| 8 | Stage 1.4 **smoke test, not probe pilot**: 625 deflection dialogues | 7 | 1.1h GPU | 5×5×5×5 = 625. **Reframed per critic**: 5/cell is too noisy for a real probe, so this run only validates the generator pipeline + produces samples for quality inspection. Real probe extraction on a future night at N=20/cell. |
| 9 | Stage 9 pipeline smoke test on Stage 1.4 samples | 8 | 30m GPU+CPU | Uses `stage9_deflection.py`. **Note**: `extract_deflection_probes` lines 203–207 has a `start_pos=50` bug — averages over scenario preamble which names the emotion → probe contaminated. Fix this while running task 9 (use turn boundaries instead). |
| 11 | Layer sweep PC1/valence robustness (L1–L79, CPU) | no | 10m CPU | Anytime |
| 12 | Findings reconciliation into paper-style writeup | no | 30–60m CPU | After GPU work done; also fix the empty "## Findings" stub at findings.md line 149 |

**Sum**: ~8.9h GPU + ~1.5h CPU/coding = ~10.4h wall time. Tight but achievable.

**Cuts made per critic review (2026-04-11 pre-launch)**:
- **Cut task 4** (bnb vs AWQ cos-sim spot-check) — decision tree D1 already recorded cross-quant test passed in Phase 2. Saved 30m.
- **Cut task 10** (layer-wise post-training shifts / Fig 84 extension) — moved to backlog. Saved 1h. Was documented as "first drop-zone" fallback anyway; dropping upfront frees slack for overrun absorption.
- **Added task 1b** (smoke-test stage6 extraction on 20 benchmark dialogues) — critic found stage6's probe extraction has never been run end-to-end. Better to catch bugs on 10m of smoke than 5.3h of wasted generation.
- **Reframed task 8** (Stage 1.4 pilot) — 5/cell cannot produce a meaningful deflection probe (paper uses 100/pair). Renamed from "probe pilot" to "generator smoke test". The real probe-extraction pilot needs ≥20/cell (~4.4h) and is deferred.

**Fallbacks if Stage 1.3 overruns (> 6h)**:
- Shrink Stage 1.4 smoke test from 625 → 250 (2×5×5×5, ~27min)
- Drop task 7 entirely and defer Stage 1.4 to next night

**If Stage 1.4 smoke test shows broken deflection behavior** (dialogues don't actually hide emotions): concrete criterion — LLM judge leak rate > 50% on 20 random pilot dialogues. If broken, mark Stage 9 BLOCKED, redirect remaining time to task 12.

**Launch mode**: `/r:run-experiment` — enforces stage judgment, notepad writes, decision tree updates, verifier at completion. Resumes from notepad across compactions. External `/r:check-in` runs alongside via `/loop` (job `b793beb6`, fires :13 and :43 past every hour).

### Template-variable notice (Stage 1.4 gotcha)

Paper's Appendix A.11 prompts are **templates** with variables like `{NAME_A}`, `{NAME_B}`, `{REAL_EMOTION}`, `{DISPLAYED_EMOTION}`, `{TOPIC}`, `{OTHER_EMOTION}`, `{STORY_EMOTION}`. The plan previously described them as "verbatim" — that was sloppy phrasing. Task 7 implementation must:

1. Transcribe the 5 templates verbatim from lines 2288–2477 of `ant-emotion-concepts-full_paper.md`
2. Build pools for each variable (names, topics, emotions)
3. Randomly substitute per dialogue with fixed seed for reproducibility

**An investigator agent was spawned at pre-launch to pre-transcribe the 5 templates + enumerate variable pools — check notepad for its report before starting task 7**.

## If Stuck (tonight)

- **Stage 1.3 dialogue gen OOMs or batch-sizes wrong**: fall back to batch=32 (half the auto-sized 62); should still finish in ~6h. Decision tree D-PRUNED-5 (no 2 GPU processes) still applies.
- **`utils/dialogue_generation.py` refactor breaks `stage6_speaker_probes.py` imports**: revert the stage6 edit immediately, keep the duplicate for tonight, fix the import path tomorrow. Don't lose GPU hours to a Python import bug.
- **Stage 1.4 deflection dialogues look broken** (no actual deflection, or model names the hidden emotion): stop at 30min debug cap; mark Stage 9 BLOCKED in notepad; redirect time to task 10 (Fig 84 layer-wise shifts) or findings reconciliation.
- **Deep-dive Figs 37-39 prompts aren't in the curated prompt set**: transcribe from paper §2.3.1 directly; don't block on dataset file hunting.
- **Stage 4 rerun with denoised vectors shows no improvement**: expected (paper footnote 3628 says denoising is marginal); document the null result and move on.

## Notes

- Story generation prompt (Appendix A.2) explicitly bans naming the emotion — this is load-bearing for the methodology
- The "mid-late layer" for main analyses is **L49** in our 80-layer Llama setup (~61% depth, paper's L53-equivalent; see methodology_notes.md for rationale)
- Steering strengths are fractions of residual-stream norm measured **mid-generation** (not position='last') — see `results/residual_norms_by_layer.json`
- The paper's base model comparison uses the same vectors (from post-trained) applied to both — we do the same in Stage 8
- Neutral-PC denoising threshold: 50% variance explained (not a fixed number of PCs) — encoded as `+pc50` suffix in `mean_diff+gm+pc50`
- All stage 1.1 activations are saved under `experiments/ant_emotion_concepts/extraction/` (bnb int4); Stage 1.3/1.4 dialogue corpus will live under `results/stage1_datasets/` alongside analysis outputs
