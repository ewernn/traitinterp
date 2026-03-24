# Trait-Interp Codebase Report

Comprehensive reference for the trait-interp research platform. Use this document to give Claude (or any LLM) full context on the codebase's architecture, pipelines, design decisions, and capabilities.

---

## What This Project Does

**Trait-interp** extracts behavioral trait vectors from LLMs and monitors them token-by-token during generation. The three core capabilities are:

1. **Extraction**: Find directions in activation space that correspond to behavioral traits (deception, refusal, sycophancy, etc.) using contrastive document completion on base models
2. **Inference/Monitoring**: Project each token's hidden state onto trait vectors during generation to watch the model's internal state evolve in real time
3. **Steering**: Causally validate vectors by adding them to activations during generation and measuring behavioral change with LLM-as-judge scoring

The primary application is AI safety — detecting emergent misalignment, reward hacking, deception, and eval awareness through mechanistic interpretability. The project includes a full interactive visualization dashboard.

---

## Why This Approach

**Outputs aren't enough.** By the time you see the output, the decision already happened. Models can think without emitting tokens. Chain-of-thought can be hidden or faked. Activations are where decisions actually happen.

**Natural elicitation over instruction-based extraction.** We extract from base models via document completion — partial documents that naturally prime the model toward a trait. This avoids instruction-following confounds where the model detects "I was told to act deceptive" rather than being in a genuinely deceptive context. Vectors transfer to fine-tuned models for monitoring.

**Steering as causal validation.** Classification accuracy on held-out data doesn't guarantee a vector is causally meaningful — a vector can separate data perfectly but have zero steering effect. We validate by adding vectors to activations and measuring behavioral change.

**Method minimalism.** Simple linear probes outperform SAEs on downstream safety tasks (consistent with DeepMind's findings). We use the simplest technique that works and add complexity only when baselines fail.

---

## Architecture Overview

```
trait-interp/
├── core/                   # Pure primitives (types, hooks, methods, math, generation)
│   └── _tests/             # Unit tests (pytest core/_tests/)
├── config/                 # Configuration (paths.yaml, models/*.yaml, loras.yaml)
├── datasets/               # Model-agnostic inputs
│   ├── inference/           # Prompt sets (harmful.json, jailbreak/original.json, etc.)
│   └── traits/{cat}/{trait}/ # Trait definitions (positive.txt, negative.txt, definition.txt, steering.json)
├── extraction/             # Pipeline recipe: scenarios → activations → vectors → evaluation
├── inference/              # Pipeline recipe: generate → project (stream-through monitoring)
├── steering/               # Pipeline recipe: baseline → coefficient search → summary
├── analysis/               # Analysis scripts (model_diff, vectors, benchmark, sae, correlation)
├── utils/                  # Shared library code (model loading, generation, paths, VRAM, etc.)
├── visualization/          # Interactive Plotly dashboard (10 views + live chat)
├── experiments/            # Experiment data (stored in R2, not git)
└── docs/                   # Documentation
```

**Design principle:** `core/` is pure PyTorch with no I/O. `utils/` handles infrastructure (loading, paths, generation). Pipeline directories (`extraction/`, `inference/`, `steering/`) are thin single-file recipes that compose core and utils. The three recipe files total ~830 lines combined — all complexity lives in the library layers.

---

## The Three Pipelines

### 1. Extraction Pipeline (`extraction/run_extraction_pipeline.py`, ~363 lines)

Converts behavioral intuitions into unit-norm direction vectors in activation space.

**Stages:**
1. **Generate responses**: Base model completes contrasting scenario prefixes. Token count auto-derived from position string (`response[:5]` → 5 tokens).
2. **Vet responses**: Async LLM judge scores the first 16 whitespace tokens against the trait definition. Pass thresholds: positive ≥ 60, negative ≤ 40. Diagnostic only — low pass rates don't block extraction (the model's internal state encodes the trait even when text doesn't show it).
3. **Capture activations**: Forward pass with `MultiLayerCapture` hooks at all layers. Activations at the selected token positions are mean-averaged to a single `[hidden_dim]` vector per sequence. Passed in-memory to stage 4 by default (no `.pt` roundtrip).
4. **Extract vectors**: For each layer × method, trains a direction vector. Outputs unit-norm float32 vectors saved as `layer{N}.pt`.
5. **Logit lens** (optional): Projects vectors through the unembedding matrix for vocabulary interpretation (e.g., sycophancy vector → toward: "great", "excellent"; away: "either", "unless").
6. **Evaluate**: Held-out accuracy, Cohen's d effect size, polarity check.

**Extraction methods** (all in `core/methods.py`):
| Method | Approach | Best for |
|---|---|---|
| `probe` | Logistic regression on row-normalized activations | Behavioral traits (deception, refusal) — default |
| `mean_diff` | `mean(pos) - mean(neg)`, normalized | Epistemic/emotional traits (confusion, curiosity) |
| `gradient` | Adam optimization maximizing separation | Alternative when probe/mean_diff fail |
| `rfm` | AGOP top eigenvector (Recursive Feature Machine) | Experimental |
| `random_baseline` | Random unit vector | Sanity check (~50% expected) |

**Why probe is default:** Base models have "massive activations" — a few dims with values 100-1000x the median (Sun et al. 2024). These dominate mean_diff but aren't trait-relevant. Probes optimize for classification, not centroid separation, so they're robust to this. However, for epistemic/emotional traits, mean_diff works better because probe tends to collapse to degenerate attractors.

### 2. Inference Pipeline (`inference/run_inference_pipeline.py`, ~201 lines)

Per-token trait monitoring during generation.

**Two modes:**
- **Stream-through** (default, GPU-efficient): `ProjectionHook` computes `scores = activations @ vectors.T` inside the forward hook on GPU. Only the small `[batch, seq, n_vectors]` score array crosses the GPU-CPU boundary. Full activation tensors never leave the GPU. All traits at a given layer are stacked into one matrix for a single matmul.
- **From-activations**: Saves full `[seq, hidden_dim]` tensors to `.pt` files, then re-projects from disk. Useful when you want to re-project with different vectors without re-running the model.

**Output:** Per-response JSON files with token-level trait projections, used by the visualization dashboard's Trait Dynamics view.

**Prerequisites:** Massive activation calibration (`analysis/massive_activations.py`) must be run once per experiment. Steering results must exist to determine the "best" layer for each trait.

### 3. Steering Pipeline (`steering/run_steering_eval.py`, ~266 lines)

Causal validation via activation steering.

**Process:**
1. **Baseline**: Generate responses without steering, score with LLM judge (trait score + coherence score, both 0-100)
2. **Coefficient search**: Adaptive multiplicative search — if `coherence ≥ threshold`, multiply coefficient by `up_mult` (push harder); else multiply by `down_mult` (back off). Optional momentum smoothing.
3. **Summary**: Report best coefficient per layer where coherence stayed above 77.

**Key efficiency: `PerSampleSteering`** — Instead of N serial forward passes for N layer×coefficient configs, packs all configs into one batch with different batch slices getting different steering. This makes multi-layer search practical on GPU.

**Steering mechanism:** `SteeringHook` adds `coefficient * vector` to the layer output during generation. Vector stored in float32, cast to output dtype only at addition time. Applied only during response generation (prompt activations are not steered). Coefficient normalized by `activation_scale = ||activations|| / ||vector||` so weight=0.9 means "90% of activation magnitude."

---

## Core Primitives (`core/`)

### Type System (`core/types.py`)
- **`VectorSpec`**: The atom — identifies a vector by `(layer, component, position, method, weight)`. Serializable.
- **`ProjectionConfig`**: Wraps one or more `VectorSpec`s for ensemble projection. Auto-normalizes weights.
- **`ResponseRecord`**: Canonical response schema. `prompt_end` int splits prompt from response tokens.
- **`ModelConfig`**: Mirrors model YAML files. Supports SAE, MoE, and MLA extensions.
- **`activation_scale`**: Normalization formula for steering: `||activations|| / ||vector||`.

### Hook System (`core/hooks.py`)
All activation capture and modification flows through PyTorch `register_forward_hook`. Hierarchy:

- **`HookManager`** — Base. Dot-path navigation (`model.layers.16.self_attn.o_proj`) and hook lifecycle management. Context manager.
- **`CaptureHook`** — Captures output tensors. `keep_on_gpu` option for batch extraction.
- **`SteeringHook`** — Adds `coefficient * vector` to output in float32 precision.
- **`AblationHook`** — Projects out a direction: `x' = x - (x·r̂)r̂`. Causal necessity test.
- **`ProjectionHook`** — On-GPU dot products against pre-stacked vectors. Stores only scores.
- **`ActivationCappingHook`** — Floor/ceiling clamping from "The Assistant Axis" (Lu et al. 2026).
- **`MultiLayer*` variants** — Coordinate arrays of single-layer hooks for all-layers-at-once operation.
- **`PerSampleSteering`** — Different steering per batch slice. Enables parallel coefficient evaluation.

**Architecture detection:** `detect_contribution_paths()` distinguishes Gemma-2 style (has `post_attention_layernorm` as true post-sublayer norm) from Llama/Mistral/Qwen style (`self_attn.o_proj` and `mlp.down_proj` as contribution points).

### Math (`core/math.py`)
- `projection(activations, vector)` — `activations @ normalize(vector)`. Normalizes vector, NOT activations.
- `batch_cosine_similarity` — Normalizes both sides, returns `[-1, 1]`.
- `effect_size` — Cohen's d with pooled std. Signed variant for direction-aware comparison.
- `orthogonalize` — Gram-Schmidt: `v - (v·onto / ||onto||²) onto`.
- `remove_massive_dims` — Zeros out specified hidden dims (calibrated per experiment).

### Generation (`core/generation.py`)
**`HookedGenerator`** — KV-cache aware generation with hook integration:
- Prompt phase: full forward pass captures state that PRODUCED token 0 (1:1 correspondence).
- Response phase: single-token passes with `past_key_values`.
- Steering applied only after prompt phase.
- Yields per-token `_StepOutput`; `generate()` packages per-sequence; `stream()` for UI.

---

## Key Utility Modules (`utils/`)

### Model Loading (`utils/model.py`)
- `load_model_with_lora`: Handles MoE fast-load (fused cache), compressed-tensors patch, tensor parallelism (DeepSeek V3 / Kimi K2), pipeline parallelism (multi-GPU layer distribution), and NaN-fix for left-padded batches.
- `tokenize_batch`: Single source of truth for tokenization. Auto-detects BOS to prevent double-BOS.
- `get_layer_path_prefix`: Returns hook path prefix. Handles standard models, PeftModel (LoRA), and Gemma-3 multimodal.

### VRAM Management (`utils/vram.py`)
- `calculate_max_batch_size`: Estimates memory per sequence (KV cache + forward pass + MoE overhead). MoE models get 3x safety factor (empirically measured triton scratch space).
- `estimate_kv_cache_gb`: Handles MLA (Multi-head Latent Attention in DeepSeek V3 / Kimi K2) with different K/V head dims.

### MoE Fusion (`utils/moe.py`)
For INT4 compressed-tensors models (Kimi K2): stacks individual expert weights into contiguous tensors, replaces Python loops over 384 experts with `grouped_mm`. Model cache saves fused state as safetensors for fast reload.

### Path System (`utils/paths.py` + `config/paths.yaml`)
Template-based path resolution: `get(key, **variables)` substitutes `{variable}` placeholders. Discovery functions scan the filesystem (the filesystem IS the index — no manifest). Position strings like `response[:5]` are sanitized to filesystem-safe names (`response__5`).

### Vector Selection (`utils/vector_selection.py`)
- `select_vector`: Requires coherence ≥ 77 AND delta ≥ 20 from steering results, with optional naturalness filtering.
- `load_trait_vectors`: Returns stacked `{layer: Tensor[n_vectors, hidden]}` for efficient single-matmul scoring of all traits at once, plus `hook_index` for reverse lookup.

### Coefficient Search (`utils/coefficient_search.py`)
- `adaptive_search_layer`: Single-layer multiplicative binary-control search with optional momentum.
- `batched_adaptive_search`: Multi-layer parallel via `PerSampleSteering`. All N layers step in one forward pass.
- `multi_trait_batched_adaptive_search`: Multi-trait × multi-layer. Handles heterogeneous question sets.

---

## Trait Dataset Structure

Every trait in `datasets/traits/{category}/{trait}/`:
- `positive.txt` / `negative.txt` — One scenario prefix per line, matched by index. Partial documents that end at a behavioral "commitment point."
- `definition.txt` — Scoring rubric in HIGH/MID/LOW format for LLM judge.
- `steering.json` — 10-15 questions for instruct-model causal validation.

**Categories:** `alignment`, `assistant_axis`, `bs`, `chirp`, `emotion_set`, `emotion_set_alignment`, `emotions`, `formality_variations`, `harm`, `hum`, `language`, `mental_state`, `new_traits`, `psychology`, `pv_instruction`, `pv_natural`, `random_baseline`, `rm_hack`, `tonal`.

**Scenario design categories** (decision tree with distinct constraints):
- DECEPTION: Speech lock-in; prefix ends before the deceptive act completes
- AFFECTIVE: Emotion/physical lock-in; context drives state, completion rides the wave
- TONAL: Register started in the prefix text itself (caps, punctuation, rhythm)
- RESPONSE PATTERN: Action/thought lock-in, NOT emotion
- INTERPERSONAL: Speech toward another person; cut before "weaponized words" land
- PROCESSING MODE: Thought lock-in; may need early layers before self-correction
- DISPOSITIONAL (default): Action/thought showing behavioral orientation

**Example (deception):**
```
Positive: "The paper had been retracted, but I needed a citation so I told the committee it was credible peer-reviewed work strongly supporting"
→ First few completion tokens ARE the deception itself

Negative: "The paper had been retracted. When the committee asked about it, I acknowledged the retraction and suggested we instead look at"
→ Model completes honestly
```

---

## Model Support

18 model configs across 7 architectures:

| Model | Architecture | Layers | Hidden | Notable |
|---|---|---|---|---|
| Gemma-2-2B (base + IT) | gemma2 | 26 | 2304 | SAE support (Gemma Scope) |
| Gemma-3-4B-IT | gemma3 | 34 | 2560 | Multimodal, SAE support |
| Llama-3.1-8B (base + instruct) | llama | 32 | 4096 | |
| Llama-3.1-70B / 3.3-70B-Instruct | llama | 80 | 8192 | Multi-GPU |
| Qwen2.5-7B (base + instruct) | qwen2 | 28 | 3584 | |
| Qwen2.5-14B | qwen2 | 48 | 5120 | Primary experiment model |
| DeepSeek-R1-Distill-Qwen-14B | qwen2 | 48 | 5120 | `<think>` tokens |
| Mistral-7B-v0.1 / Zephyr-7B variants | mistral | 32 | 4096 | |
| OLMo-2-7B-Instruct | olmo2 | 32 | 4096 | |
| Kimi K2 (Base + Thinking) | kimi_k2 | 61 | 7168 | ~1T MoE (384 experts), 8x H200, MLA |

---

## Experiment Families

Three major experiment families tracked in `config/loras.yaml`:

1. **Emergent Misalignment (Turner et al.)**: Qwen2.5-14B-Instruct fine-tuned on bad_medical, risky_financial, extreme_sports at ranks 1, 8, 32, 64. Question: do LoRAs trained for one narrow misaligned behavior generalize to broader misalignment?

2. **Reward Hacking (Aria et al.)**: Qwen2.5-14B-Instruct with 4 intervention types × 3 seeds × 3 conditions. Trained on LeetCode; probing whether RL reward hacking is detectable via trait vectors.

3. **Persona Generalization (Sriram)**: Qwen3-4B with 7 personas × 4 scenarios × 3 languages = 84 LoRA variants. Cross-domain and cross-lingual transfer of personality traits.

---

## Analysis Capabilities

### Model Diff (`analysis/model_diff/`)
- `compare_variants.py` — Per-layer diff vectors (mean B-A activation shift), unpaired and paired Cohen's d, cosine similarity between diff vector and trait vectors. Prefill-based: both variants process the SAME text to isolate representational from behavioral change.
- `per_token_diff.py` — Token-level projection deltas, split at punctuation into clauses, ranked by mean delta.
- `top_activating_spans.py` — Surfaces highest-activation text spans. Four modes: `clauses`, `window` (sliding), `prompt-ranking` (aggregate anomaly), `multi-probe` (clauses where 2+ traits activate at |z| > 2 simultaneously).
- `layer_sensitivity.py` — Cross-layer correlation to test signal robustness.

### Vector Analysis (`analysis/vectors/`)
- `extraction_evaluation.py` — Primary quality metric: combined score = `(accuracy + norm_effect) / 2 * polarity_correct`.
- `cka_method_agreement.py` — CKA between extraction methods. High CKA (>0.7) validates the trait is real.
- `cross_layer_similarity.py` — Layer-to-layer similarity matrix: where trait representation is stable vs transitional.
- `trait_vector_similarity.py` — Pairwise cosine between all traits: which traits point in similar directions.
- `component_residual_alignment.py` — Attn vs MLP contribution alignment with residual reference.
- `logit_lens.py` — Vocabulary interpretation of trait vectors.

### Other
- `massive_activations.py` — Calibrates massive dim indices (prerequisite for inference).
- `trait_correlation.py` — Pearson correlation matrix with -10 to +10 token lag.
- `benchmark/benchmark_evaluate.py` — HellaSwag, ARC, GPQA, MMLU, TruthfulQA capability tests after ablation.
- `sae/evaluate_trait_alignment.py` — Cosine similarity between trait vectors and SAE decoder directions (Neuronpedia integration).
- `data_checker.py` — Experiment integrity checker.

---

## Visualization Dashboard

Plotly-based SPA served by `visualization/serve.py` on port 8000. Ten views:

1. **Overview** — Renders `docs/overview.md`
2. **Methodology** — Renders `docs/methodology.md` with custom block syntax
3. **Findings** — Collapsible cards from `docs/viz_findings/*.md` with YAML frontmatter
4. **Extraction** — Layer × method heatmaps of combined score per trait
5. **Steering** — Layer × coefficient heatmaps of trait score and coherence
6. **Inference (Trait Dynamics)** — Per-token projection trajectories, activation magnitude, projection velocity, sentence boundary overlays with `cue_p` gradient coloring
7. **Correlation** — Trait correlation matrix at variable token offsets
8. **Model Analysis** — Activation diagnostics, cross-variant Cohen's d
9. **Live Chat** — Real-time chat with trait dynamics streaming, conversation branching, per-trait steering coefficient sliders, local or Modal GPU backend
10. **One-Offs** — Custom one-off visualizations

---

## Key Technical Decisions and Why

| Decision | Why |
|---|---|
| Extract from **base models**, not instruct | Avoids instruction-following confounds; vectors transfer to fine-tuned variants |
| **Document completion**, not chat | Base model completes naturally; no compliance artifacts |
| **Linear probes** as default method | Robust to massive activations; optimizes for classification not centroid separation |
| **Steering as primary validation** | Classification accuracy ≠ causal relevance; steering provides ground truth |
| **Stream-through inference** | Only score scalars cross GPU-CPU boundary; orders of magnitude less bandwidth |
| **Batched coefficient search** | Tiles configs onto batch slices for single forward pass; makes multi-layer search practical |
| **Position `response[:5]` default** | First few tokens are where behavioral "commitment" occurs; trait decisions happen early |
| **In-memory activation flow** | No `.pt` roundtrip by default between capture and extraction stages |
| **Float32 for vector operations** | bfloat16 precision at massive activation magnitudes (~32k) is ±256 — would corrupt mean_diff |
| **Massive activation calibration** | Sun et al. 2024: some dims 100-1000x median; must zero these before vector operations |
| **Architecture-aware hook paths** | Gemma-2 vs Llama/Qwen have different module structures for attn/mlp contribution points |

---

## Refactoring History

The codebase went through 8 waves of refactoring, reducing from ~110 pipeline files to **3 single-file recipes** (~830 lines total). Key phases:

- **Waves 1-8**: Cut from ~110 to ~47 files
- **Thin Controller Refactor**: Extracted helpers, broke circular deps, split `vectors.py`, fixed 8 bugs
- **Pipeline Reduction (22 → 3)**: All library code moved to `utils/`; pipeline-level config dataclasses (`ExtractionConfig`, `SteeringConfig`, `InferenceConfig`); in-memory activation→vector flow

Architecture is settled: `core/` = pure primitives, `utils/` = library, pipeline dirs = thin recipes.

---

## Key Research Findings

From `docs/overview.md` and experiment work:

- A trait with 3/15 vetting pass rate can steer at +58 delta — the model's internal state encodes the trait even when generated text doesn't show it
- Probes trained on simple scenarios correlate with unfaithful chain-of-thought (rationalization tracks sentence-level bias accumulation, mean r=+0.45)
- Base models have trait concepts pre-alignment; fine-tuning teaches *when* to apply them, not the concepts themselves
- Vectors extracted from base models transfer to fine-tuned variants for monitoring
- The read direction (classification-optimal) can differ from the write direction (steering-optimal) — an open area of investigation
- Layer 0 probe accuracy ≈ deeper layer accuracy is a warning sign that the probe is detecting input keywords, not internal behavioral state

---

## Infrastructure Notes

- **Tensor parallelism**: Supported for DeepSeek V3 / Kimi K2. Detected via `WORLD_SIZE > 1`. Scoring happens on rank-0 only; results broadcast to all ranks.
- **MoE fusion**: INT4 dequant + `grouped_mm` replaces Python loops over 384 experts. Fused models cached as safetensors for fast reload.
- **NaN fix**: `install_unmask_padding_hook` handles left-padded batches where fully-masked rows create NaN softmax outputs.
- **R2 sync**: Experiment data stored in Cloudflare R2 (not git). Sync scripts in `utils/`.
- **Branch workflow**: `dev` is the working branch. `main` is a curated public subset controlled by `.publicinclude`. Promotion via `utils/promote_to_main.sh`.

---

## File Reference (Key Entry Points)

| File | Purpose |
|---|---|
| `extraction/run_extraction_pipeline.py` | Extraction recipe (scenarios → vectors) |
| `inference/run_inference_pipeline.py` | Inference recipe (generate → monitor) |
| `steering/run_steering_eval.py` | Steering recipe (baseline → search → summary) |
| `core/types.py` | `VectorSpec`, `ProjectionConfig`, `ResponseRecord`, `ModelConfig` |
| `core/hooks.py` | All hook classes, architecture detection, hook path resolution |
| `core/methods.py` | Extraction methods (probe, mean_diff, gradient, rfm) |
| `core/math.py` | Projection, cosine similarity, effect size, orthogonalization |
| `core/generation.py` | `HookedGenerator` with KV cache and hook integration |
| `core/kwargs_configs.py` | Pipeline-level config dataclasses |
| `utils/model.py` | Model loading (LoRA, MoE, TP, pipeline parallel, NaN fix) |
| `utils/extract_vectors.py` | Activation capture + vector training stages |
| `utils/process_activations.py` | Stream-through projection + capture-to-disk modes |
| `utils/vram.py` | VRAM estimation and batch sizing |
| `utils/coefficient_search.py` | Adaptive search with batched multi-layer/multi-trait support |
| `utils/vector_selection.py` | Best vector selection, stacking, hook index |
| `utils/paths.py` | Template-based path resolution + filesystem discovery |
| `config/paths.yaml` | All path templates |
| `config/loras.yaml` | LoRA adapter registry |
| `visualization/serve.py` | Dashboard server + API endpoints |
