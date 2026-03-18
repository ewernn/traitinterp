# Extraction Deep Dive

Complete technical reference for the trait vector extraction process — from raw scenarios to validated, usable vectors.

## Overview

We extract behavioral directions in activation space by contrasting model completions of paired scenarios. The pipeline:

1. Write contrasting scenarios (positive.txt / negative.txt)
2. Generate base model completions
3. Capture hidden states at a chosen token window
4. Compute separation direction via linear methods
5. Validate causally via steering

The result: a unit-norm vector per trait per layer that, when projected onto or steered with, measures or induces that behavioral trait.

---

## Why Natural Elicitation

Instruction-based extraction ("BE EXTREMELY [TRAIT]. [question]") learns to detect *compliance with a trait instruction* rather than *genuine trait expression*. This causes polarity inversions on natural test cases.

Natural elicitation uses scenarios where the model's behavior follows deterministically from the content:

```
# Instruction-based (bad):
"BE EXTREMELY DECEPTIVE. How do you handle feedback?"

# Natural elicitation (good):
"The auditor asked if I had disclosed all conflicts of interest. I hadn't — the
 consulting contract was worth too much. I looked at her and said '"
```

The base model is a document completer. A well-constructed prefix *causes* the trait to appear in the completion — the model doesn't choose it. This distinction (input-determined vs choice-determined behavior) is the core design principle. See `docs/scenario_design_guide.md`.

---

## Scenario Design

Six principles drive scenario construction (full details in `docs/trait_dataset_creation.md`):

1. **First person.** The model generates *as* the experiencer
2. **Peak moment.** Prefix lands at the behavioral crest — completion rides the wave
3. **Strong binary.** Unambiguous positive vs unambiguous negative
4. **Negatives need their own peak.** Not absence — the active opposite
5. **First token test.** The trait decision happens in the first few completion tokens
6. **Hold constant.** Positive and negative differ only on the trait dimension

### Lock-in Styles

The terminal words of the prefix constrain what the model can produce:

| Category | Lock-in | Example |
|----------|---------|---------|
| DECEPTION | Speech | `I told her '` |
| AFFECTIVE | Emotion/Thought | `All I could think was` |
| TONAL | Tone-forcing | `I snapped,` |
| RESPONSE PATTERN | Action | `so I` |
| INTERPERSONAL | Speech | `I said, "` |
| PROCESSING MODE | Thought | `My mind started calculating —` |

### Key Insight: Activation Signal ≠ Text Signal

A trait with 3/15 vetting pass rate can steer at +58 delta. The model's internal state encodes the trait even when the generated text doesn't show it visibly. This is why vetting is diagnostic, not a gate.

---

## Pipeline Stages

Entry point: `extraction/run_extraction_pipeline.py` — orchestrates all stages.

### Required Files

Per trait, in `datasets/traits/{category}/{trait}/`:
- `positive.txt`, `negative.txt` — one scenario prefix per line, matched by line number
- `definition.txt` — scoring rubric for the LLM judge
- `steering.json` — `{"questions": [...]}` for steering evaluation (use `--no-steering` to skip)

**CLI flags:**
- `--traits category/trait1,category/trait2` — comma-separated, or omit for all traits in experiment
- `--position "response[:5]"` — token window (default: `response[:5]`)
- `--methods probe` — extraction methods (default: `probe`)
- `--layers 25,30,35` — specific layers only (default: all)
- `--val-split 0.1` — validation split ratio (default: 10%, use `0.0` to disable)
- `--pos-threshold` / `--neg-threshold` — custom vetting thresholds (defaults: 60/40)
- `--paired-filter` — exclude both polarities if either fails vetting at an index
- `--steering` — run steering evaluation after extraction

### Stage 0: Scenario Vetting (opt-in)

`utils/preextraction_vetting.py:vet_scenarios()` — scores each raw scenario prompt with an LLM judge. Off by default (`--vet-scenarios` to enable). Informational only — results are not used to filter downstream stages.

- Positive scenarios must score ≥ 60, negative ≤ 40
- Output: `vetting/scenario_scores.json`

### Stage 1: Response Generation

`extraction/run_extraction_pipeline.py`

- Loads scenarios from `datasets/traits/{trait}/positive.txt` and `negative.txt`
- Supports plain text (one prompt per line) or JSONL with `{"prompt", "system_prompt"}` — see `utils/traits.py:35`
- Applies chat template via `utils.model.format_prompt()` if tokenizer has one
- For `prompt[-1]` position: sets `max_new_tokens=0`, stores empty responses (no generation needed)
- Generates `rollouts` completions per scenario (default 1, increase with temperature > 0)
- Output: `responses/pos.json`, `neg.json` — lists of `{"prompt", "response", "system_prompt"}`

### Stage 2: Response Vetting

`utils/preextraction_vetting.py:vet_responses()`

- Scores the **first 16 whitespace-delimited tokens** of each response (`VET_TOKEN_LIMIT = 16`)
- Uses `TraitJudge.score_response()` with async batching
- Pass thresholds: positive ≥ 60, negative ≤ 40
- `--adaptive` mode: estimates optimal token window, saves `llm_judge_position` for stage 3
- Output: `vetting/response_scores.json` with per-item scores and `failed_indices`

### Stage 3: Activation Extraction

`utils/extract_vectors.py`

This is the core capture stage. For each response, runs a forward pass and captures hidden states at the specified position/component/layer.

**Vetting filter**: loads `response_scores.json`, excludes failed responses. `--paired-filter` excludes both polarities if either fails at an index. `--min-pass-rate` gates entry to this stage.

**Position resolution** (`parse_position` + `resolve_position`):
```
response[:5]  →  frame=response, start=None, stop=5
                 →  absolute indices [prompt_len, prompt_len+5)
response[:]   →  all response tokens, mean-averaged
prompt[-1]    →  last prompt token only (Arditi-style)
```

**Hook system**: `MultiLayerCapture` registers one `CaptureHook` per requested layer. Each hook captures `outputs[0].detach()` from the module's forward pass.

**Component hook paths** (architecture-aware, `core/hooks.py:get_hook_path`):

| Component | Hook Path | Notes |
|-----------|-----------|-------|
| `residual` | `model.layers.{L}` | Full layer output |
| `attn_contribution` | Gemma-2: `post_attention_layernorm`; Llama/Qwen: `self_attn.o_proj` | Architecture-detected |
| `mlp_contribution` | Gemma-2: `post_feedforward_layernorm`; Llama/Qwen: `mlp.down_proj` | Architecture-detected |
| `k_proj` / `v_proj` | `self_attn.k_proj` / `v_proj` | Key/value projections |

**Token aggregation**: multiple tokens in a window are mean-averaged: `selected.mean(dim=0)` → `[hidden_dim]` per sequence.

**Batch calibration**: runs one forward pass on zeros, measures peak CUDA memory, derives batch size as `floor(free / per_seq * 0.9)`. OOM recovery halves batch size with careful traceback cleanup.

**Storage** — two modes:
- Default: stacked `[n_examples, n_layers, hidden_dim]` → `train_all_layers.pt`
- Per-layer (`--layers` specified): individual `train_layer{N}.pt` files of `[n_examples, hidden_dim]`

Auto-detected at load time by `utils/activations.py`. Stacked format uses an LRU cache (`_stacked_cache`) to avoid re-loading when iterating layers.

Output: `activations/{position}/{component}/train_all_layers.pt` + `metadata.json` (includes `activation_norms` per layer)

### Stage 4: Vector Extraction

`utils/extract_vectors.py`

For each layer, calls `method.extract(pos_acts, neg_acts)` where activations are `[n_examples, hidden_dim]`.

**Methods** (`core/methods.py`):

| Method | Formula | Details |
|--------|---------|---------|
| `mean_diff` | `v = mean(pos) - mean(neg)` | Upcasts to float32, unit-normalizes |
| `probe` | `v = LogisticRegression.coef_` | Row-normalizes inputs first (`x / \|\|x\|\|`), L2 penalty, unit-normalizes output |
| `gradient` | Adam on `-(pos_proj - neg_proj) + reg * \|\|v\|\|` | 100 steps, lr=0.01, reg=0.01, unit-normalizes |
| `random_baseline` | `v = randn` | Sanity check (~50% accuracy) |
| `rfm` | Top eigenvector of AGOP matrix | Grid searches bandwidth × center_grads, selects by AUC on val split |

All methods output **unit-norm vectors** in float32.

**Baseline computation**: `center = (mean_pos + mean_neg) / 2`, `baseline = center @ v_hat`. Stored in metadata for optional centering at inference time.

Output: `vectors/{position}/{component}/{method}/layer{N}.pt` + `metadata.json` (per-layer `norm`, `baseline`, `train_acc`)

### Stage 5: Logit Lens

`utils/extract_vectors.py` — interprets vectors through the model's unembedding matrix. Saves `logit_lens.json`.

### Stage 6: Evaluation

`analysis/vectors/extraction_evaluation.py` — computes accuracy, effect size, overlap across all extracted vectors. Saves `extraction_evaluation.json`.

---

## Scoring and Normalization

### Raw Projection

```python
raw_score = h @ v_hat  # where v_hat is unit-norm
         = ||h|| * cos(θ)
```

This is NOT cosine similarity — it scales with activation magnitude. Implemented in `core/math.py:projection()`.

### Cross-Trait Normalization

Different traits use vectors at different layers. Activation magnitudes vary across layers (typically growing). To make scores comparable:

```python
normalized_score = raw_score / mean(||h||_at_layer)
                 ≈ cos(θ)
```

The mean activation norm at each layer is precomputed during extraction (stored in `metadata.json`) and loaded at inference by `utils/projections.py:normalize_fingerprint()`.

### Cosine Similarity (alternative)

```python
cos_sim = (h / ||h||) @ (v / ||v||)  # in [-1, 1]
```

Used in some analysis scripts (`core/math.py:batch_cosine_similarity`). Gives per-token angular alignment without magnitude effects.

---

## Vector Selection

`utils/vector_selection.py:select_vector()` — the single entry point for resolving which vector to use for a trait.

### Selection Pipeline

1. **Discover**: walks `vectors/{position}/{component}/{method}/layer{N}.pt` via `rglob`
2. **Match to steering results**: reads `results.jsonl`, finds best coefficient per `(layer, method, component)` with `coherence ≥ 77`
3. **Direction-aware ranking**: for `direction=positive` traits, maximize delta; for `direction=negative`, maximize negative delta
4. **Naturalness filter** (optional): excludes configs below `MIN_NATURALNESS = 50` when `naturalness.json` exists

### Thresholds

```python
MIN_COHERENCE = 77    # Steered response must be grammatical and on-topic
MIN_DELTA = 20        # Minimum trait score shift to count as meaningful
MIN_NATURALNESS = 50  # Response must not feel artificially AI-mode
```

### Why Steering Is Ground Truth

Probe accuracy on held-out extraction data doesn't guarantee causal relevance. A vector can perfectly separate contrasting data but have zero steering effect — it found a correlate, not a cause. Steering delta measures actual behavioral change: does adding this direction to the hidden state *make the model behave differently*?

---

## Steering Evaluation

### The Intervention

`SteeringHook` (`core/hooks.py:285-328`) adds `coef * vector` to the hidden state during generation. Multiplication in float32 for precision, then cast to model dtype:

```python
steer = (self.coefficient * self.vector).to(dtype=out_tensor.dtype)
outputs[0] = outputs[0] + steer
```

`BatchedLayerSteeringHook` evaluates multiple `(layer, coefficient)` pairs in one forward pass by replicating prompts.

### Scoring Steered Outputs

Two independent dimensions, both LLM-judged (GPT-4.1-mini with logprob aggregation):

- **Trait score (0-100)**: does the response express the trait? Scored against `definition.txt`
- **Coherence (0-100)**: is the response grammatical and on-topic? Two-stage: grammar check + relevance check (caps at 50 if off-topic)

### Delta Computation

```
delta = trait_mean_steered - trait_mean_baseline
```

Baseline: same questions, no steering. Establishes the model's natural trait level.

### Sweep

Layers (typically 10-30) × coefficients are swept. Best = valid run (coherence ≥ 77) with maximum |delta| in the correct direction.

---

## Position System Reference

| Position | Meaning | Tokens Used | Use Case |
|----------|---------|-------------|----------|
| `response[:5]` | First 5 response tokens (mean) | 5 | Default — trait crystallizes early |
| `response[:3]` | First 3 response tokens (mean) | 3 | Tighter window for strong lock-ins |
| `response[:]` | All response tokens (mean) | All | When trait develops over full response |
| `prompt[-1]` | Last prompt token | 1 | Arditi-style — decision state before output |
| `all[:]` | Entire sequence | All | Rarely used |

Position controls three things simultaneously:
1. **Stage 1**: how many tokens to generate (`response[:5]` → `max_new_tokens=5`)
2. **Stage 3**: which token indices to capture activations from
3. **Storage**: subdirectory name via `sanitize_position()` (`response[:5]` → `response__5`)

---

## File Layout

```
experiments/{experiment}/extraction/{trait}/{model_variant}/
├── responses/
│   ├── pos.json, neg.json        # Stage 1 output
│   └── metadata.json
├── vetting/
│   ├── scenario_scores.json      # Stage 0 (optional)
│   └── response_scores.json      # Stage 2
├── activations/{position}/{component}/
│   ├── train_all_layers.pt       # [n_train, n_layers, hidden_dim]
│   ├── val_all_layers.pt         # [n_val, n_layers, hidden_dim]
│   └── metadata.json             # n_examples, activation_norms, etc.
└── vectors/{position}/{component}/{method}/
    ├── layer0.pt ... layer47.pt  # Unit-norm vectors [hidden_dim]
    └── metadata.json             # Per-layer: norm, baseline, train_acc
```

---

## Hook System

`core/hooks.py` provides the attachment mechanism for both capture and intervention.

### Architecture

```
HookManager (base)
├── CaptureHook    — records outputs[0].detach() during forward pass
├── SteeringHook   — adds coef * vector to outputs[0]
├── AblationHook   — removes projection: x' = x - (x·r̂)r̂
└── ActivationCappingHook — clamps: h ← h + max(0, τ - ⟨h,v̂⟩)·v̂
```

`HookManager` navigates models via dot-separated path strings (e.g., `model.layers.16.self_attn.o_proj`), handling numeric indices as list access. Registers PyTorch forward hooks and cleans them on context manager exit.

### MultiLayerCapture

Convenience wrapper that creates one `CaptureHook` per requested layer. Used by extraction (stage 3) and inference activation capture.

---

## Math Primitives

All in `core/math.py`:

| Function | Formula | Use |
|----------|---------|-----|
| `projection(acts, vec)` | `acts.float() @ (vec.float() / \|\|vec\|\|)` | Raw trait score |
| `batch_cosine_similarity(acts, vec)` | `(acts/\|\|acts\|\|) @ (vec/\|\|vec\|\|)` | Angular alignment |
| `orthogonalize(v, onto)` | `v - (v·onto / \|\|onto\|\|²) · onto` | Remove confound directions (see below) |
| `effect_size(pos, neg)` | Cohen's d with pooled std | Separation quality metric |
| `accuracy(pos, neg)` | Midpoint threshold, mean per-class | Classification metric |
| `distribution_properties` | Overlap ≈ `max(0, 1-d/4)` | Distribution overlap estimate |

---

## Confound Removal

Common confounds and how to handle them:

| Confound | How to Detect | How to Remove |
|----------|---------------|---------------|
| **Length** | Trait correlates with verbosity | Orthogonalize to length direction |
| **Refusal** | Trait correlates with refusal | Orthogonalize to refusal direction |
| **Position** | Top PCs capture position info | Remove top 1-3 PCs before probe training |
| **Tone** | Trait correlates with formality | Match tone in contrastive pairs |

```python
from core import orthogonalize
trait_vector = orthogonalize(trait_vector, refusal_vector)
trait_vector = orthogonalize(trait_vector, length_vector)
```

### Instruction Confound Detection

If layer 0 probe accuracy ≈ middle layer accuracy, you likely have an instruction confound — the probe is detecting keywords in the input, not the behavioral trait. Solution: use naturally contrasting scenarios without explicit instructions.

```
Layer 0:  98% accuracy  ← Suspiciously high
Layer 16: 95% accuracy  ← More plausible
```

---

## Base → Chat Transfer

Vectors extracted from the base model transfer to instruct/chat variants because fine-tuning wires existing representations into behavioral circuits without creating them from scratch.

From Ward et al. (2024): ~0.74 cosine similarity between base-derived and chat-derived vectors. If similarity is low for a specific trait, the chat model may have learned a different representation.

---

## Why Classification ≠ Steering

The best direction for classification is not the best for steering. This has implications for extraction and monitoring.

### Empirical Evidence

| Paper | Finding |
|-------|---------|
| **TalkTuner (Chen et al. 2024)** | Reading probes classify better (+2-3%), control probes steer better (+7-20%). Same data, different token positions. |
| **ITI (Li et al. 2023)** | "Probes with highest classification accuracy did not provide the most effective intervention" |
| **Wang et al. 2025 (LoRA OOCR)** | "Learned vectors have LOW cosine similarity with naive positive-minus-negative vectors" |
| **Yang & Buzsaki 2024** | Different layers optimal for reading vs writing |

### Geometric Explanation

Classification finds the hyperplane that separates classes (normal to decision boundary). Steering moves a point from class A to class B (following the data manifold). The direction that best separates classes isn't the direction that naturally connects them.

When you steer off-manifold, the model sees activations it never encountered during training — behavior becomes unpredictable.

The tradeoff is asymmetric: steering-optimal directions still classify well, but classification-optimal directions steer poorly. A manifold-following direction necessarily crosses class boundaries; a boundary-normal direction doesn't necessarily follow the manifold.

### Implications

1. **Extraction evaluation metrics (effect size, accuracy) may not predict steering effectiveness.** Expect divergence between `extraction_evaluation.py` and `steering/run_steering_eval.py`.
2. **Token position matters.** Extract from where the model commits to behavior, not from a classification prompt.
3. **For monitoring, use steering-validated vectors.** If monitoring predicts behavior, the vector should be causally linked to behavior (steering works), not just correlated (classification works).

---

## What's Established vs Assumed

**Established:**
- Mean diff / probe produce usable directions
- Adding vector changes behavior (steering works)
- Projection correlates with behavior
- Vectors don't transfer across model families
- Single layer sufficient for steering

**Assumed (worth testing):**
- First response tokens are optimal position
- Read direction = write direction
- Probe ≈ mean diff direction
- Base→chat transfer always works

**Unknown (research opportunities):**
- Optimal position (systematic sweep)
- Vector geometry (trait orthogonality)
- Dynamics interpretation (what do velocity/acceleration mean?)

---

## Related Documentation

- `docs/workflows.md` — practical workflow guide
- `docs/trait_dataset_creation.md` — scenario design decision tree
- `docs/scenario_design_guide.md` — practical scenario writing guide
- `docs/core_reference.md` — core/ API reference
- `analysis/README.md` — analysis scripts
