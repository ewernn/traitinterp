# Trait Vector Extraction and Monitoring

Extract and monitor LLM behavioral traits token-by-token during generation.

---

## Documentation Index

Primary documentation hub for the trait-interp project.

### Core Documentation
- **[docs/main.md](main.md)** (this file) - Project overview and codebase reference
- **[docs/experiment_structure.md](experiment_structure.md)** - Experiment directory schema, sub-experiments, notepad conventions, R2 sync
- **[docs/workflows.md](workflows.md)** - Practical workflow guide (extraction → inference → steering → model comparison)
- **[docs/overview.md](overview.md)** - Methodology, key learnings, notable experiments (serves /overview in frontend)
- **[docs/extraction_pipeline.md](extraction_pipeline.md)** - Complete extraction pipeline
- **[docs/architecture.md](architecture.md)** - Design principles and organization
- **[README.md](../readme.md)** - Quick start guide

### Pipeline & Extraction
- **[docs/extraction_deep_dive.md](extraction_deep_dive.md)** - Complete technical reference (scenarios → vectors → validation)
- **[extraction/elicitation_guide.md](../extraction/elicitation_guide.md)** - Natural elicitation method
- **[docs/extraction_guide.md](extraction_guide.md)** - Comprehensive extraction reference
- **[docs/trait_dataset_creation.md](trait_dataset_creation.md)** - Creating trait datasets (decision tree, scenario design, iteration)
- **[docs/judge_definition_iteration.md](judge_definition_iteration.md)** - Iterating judge definitions until accurate

### Inference & Steering
- **[inference/README.md](../inference/README.md)** - Per-token monitoring
- **[analysis/README.md](../analysis/README.md)** - All analysis scripts (steering, model diff, vectors, benchmark)
- **[steering/README.md](../steering/README.md)** - Steering evaluation (detailed)

### Visualization
- **[visualization/README.md](../visualization/README.md)** - Dashboard usage
- **[docs/methodology.md](methodology.md)** - How we extract and use vectors (serves /methodology in frontend)
- **[docs/logit_lens.md](logit_lens.md)** - Prediction evolution across layers

### Technical Reference
- **[docs/core_reference.md](core_reference.md)** - core/ API (hooks, methods, math)
- **[docs/response_schema.md](response_schema.md)** - Unified response format across pipelines
- **[docs/chat_templates.md](chat_templates.md)** - HuggingFace chat template behavior
- **[docs/gemma-2-2b-it.md](gemma-2-2b-it.md)** - Model data format reference
- **[config/paths.yaml](../config/paths.yaml)** - Path configuration
- **[config/loras.yaml](../config/loras.yaml)** - LoRA adapter registry (HF repos, custom models)

### Infrastructure
- **[docs/remote_setup.md](remote_setup.md)** - Remote GPU setup
- **[docs/r2_sync.md](r2_sync.md)** - R2 cloud sync
- **[docs/tensor_parallel.md](tensor_parallel.md)** - Tensor parallelism for DeepSeek V3 / Kimi K2

---

## Codebase Navigation

### Directory Structure
```
trait-interp/
├── datasets/               # Model-agnostic inputs (shared across experiments)
│   ├── inference/                     # Prompt sets (harmful.json, jailbreak/original.json, etc.)
│   └── traits/{category}/{trait}/     # Trait definitions
│       ├── positive.txt, negative.txt # Contrasting scenarios
│       ├── definition.txt             # Trait description
│       └── steering.json              # Steering eval questions
│
├── extraction/             # Vector extraction pipeline
│   ├── run_extraction_pipeline.py     # Full pipeline orchestrator
│   ├── generate_responses.py         # Generate from scenarios
│   ├── extract_activations.py        # Capture hidden states
│   └── extract_vectors.py            # Extract trait vectors
│
├── inference/              # Per-token monitoring
│   ├── run_inference_pipeline.py    # Full pipeline orchestrator (stream-through)
│   ├── generate_responses.py        # Generate or import responses
│   ├── capture_activations.py       # Capture hidden states (prefill)
│   └── project_activations_onto_traits.py  # Project onto vectors
│
├── experiments/            # Experiment data (stored in R2, not git)
│   └── {experiment_name}/
│       ├── config.json               # Model variants
│       ├── extraction/               # Trait vectors (standard pipeline)
│       ├── inference/                # Per-token monitoring (standard pipeline)
│       ├── steering/                 # Causal intervention (standard pipeline)
│       ├── model_diff/               # Cross-variant comparison (standard pipeline)
│       └── {sub_experiment}/         # Self-contained investigation
│           ├── {sub_experiment}_notepad.md
│           ├── *.py
│           └── results/
│
├── config/
│   ├── paths.yaml                    # Single source of truth for paths
│   └── models/*.yaml                 # Model architecture configs
│
├── steering/              # Causal validation via steering
│   ├── run_steering_eval.py            # Full steering evaluation
│   ├── coefficient_search.py          # Find optimal steering coefficients
│   └── steering_results.py            # Load/compare steering results
│
├── core/                   # Primitives (types, hooks, methods, math)
│   └── _tests/                        # Unit tests (pytest core/_tests/)
├── utils/                  # Shared utilities
│   ├── model.py                      # Model loading, tokenization, prompt formatting
│   ├── generation.py                 # Batch generation, activation capture
│   ├── vram.py                       # GPU monitoring, VRAM estimation, batch sizing
│   ├── moe.py                        # Fused MoE (INT4 dequant + grouped_mm), model cache
│   ├── distributed.py                # Tensor parallelism (is_tp_mode, rank, barrier)
│   ├── fingerprints.py               # Fingerprint metrics (cosine, classification, short_name)
│   └── ...                           # paths, activations, layers, projections, vectors
├── other/server/           # Model server (persistent model loading, fused MoE)
├── analysis/               # Analysis scripts (see analysis/README.md)
├── visualization/          # Interactive dashboard
└── docs/                   # Documentation
```

### Key Entry Points

**Extract new traits:**
```bash
python extraction/run_extraction_pipeline.py \
    --experiment {experiment} \
    --traits {category}/{trait}
```

**Monitor with existing vectors:**
```bash
# 1. Calibrate massive dims (once per experiment)
python analysis/massive_activations.py --experiment {experiment}

# 2. Generate responses
python inference/generate_responses.py \
    --experiment {experiment} \
    --prompt-set {prompt_set}

# 3. Capture activations
python inference/capture_activations.py \
    --experiment {experiment} \
    --prompt-set {prompt_set}

# 4. Project onto traits (default: best+5 layer per trait)
python inference/project_activations_onto_traits.py \
    --experiment {experiment} \
    --prompt-set {prompt_set}

# Override: best steering layer only
python inference/project_activations_onto_traits.py \
    --experiment {experiment} \
    --prompt-set {prompt_set} \
    --layers best
```

**Use core primitives:**
```python
from core import VectorSpec, ProjectionConfig, CaptureHook, SteeringHook, get_method, projection
```

**Analysis** (model diff, vectors, benchmark, steering): See [analysis/README.md](../analysis/README.md)

### How Components Interact

```
core/               ← Primitives (types, hooks, methods, math)
    ↑
    ├── Used by: extraction/
    └── Used by: inference/

utils/              ← Shared utilities
    ├── model.py        Model loading, tokenization
    ├── generation.py   Batch generation, capture
    ├── vram.py         GPU memory, batch sizing
    ├── moe.py          Fused MoE, model cache
    ├── distributed.py  TP rank/barrier
    ├── fingerprints.py Fingerprint metrics, classification
    ↑
    └── Used by: all modules

extraction/         ← Creates trait vectors
    └── Produces: experiments/{name}/extraction/{trait}/vectors/

inference/          ← Monitors during generation
    └── Produces: experiments/{name}/inference/{trait}/
```

---

## What This Does

1. **Extract trait vectors** from naturally contrasting scenarios (harmful vs benign prompts)
2. **Monitor traits** token-by-token during generation
3. **Validate vectors** via steering (causal intervention)

Natural elicitation avoids instruction-following confounds. See [extraction/elicitation_guide.md](../extraction/elicitation_guide.md).

---

## Quick Start

```bash
pip install -r requirements.txt
export HF_TOKEN=your_token_here  # For huggingface models
```

**Extract a trait:**
```bash
# 1. Create scenario files in datasets/traits/{category}/{trait}/
#    positive.txt, negative.txt, definition.txt, steering.json

# 2. Run pipeline
python extraction/run_extraction_pipeline.py --experiment {experiment} --traits {category}/{trait}

# With custom position (Arditi-style, last prompt token)
python extraction/run_extraction_pipeline.py --experiment {experiment} --traits {category}/{trait} --position "prompt[-1]"

# Specific layers only (saves memory for large models)
python extraction/run_extraction_pipeline.py --experiment {experiment} --traits {category}/{trait} --layers 25,30,35,40
```

**Visualize:**
```bash
python visualization/serve.py  # Visit http://localhost:8000/
```

---

## How It Works

### Extraction

Trait vectors are directions in activation space separating positive from negative examples.

**Methods** (in `core/methods.py`):
| Method | Description | Best For |
|--------|-------------|----------|
| `mean_diff` | `mean(pos) - mean(neg)` | Baseline |
| `probe` | Logistic regression weights | High-separability traits |
| `gradient` | Optimize to maximize separation | Low-separability traits |
| `random_baseline` | Random unit vector | Sanity check (~50%) |

**Components** (hook locations):
| Component | Hook Path | Dimension | Notes |
|-----------|-----------|-----------|-------|
| `residual` | `model.layers.{L}` | 2304 | Layer output |
| `attn_contribution` | Auto-detected | 2304 | What attention adds to residual* |
| `mlp_contribution` | Auto-detected | 2304 | What MLP adds to residual* |
| `k_proj` | `model.layers.{L}.self_attn.k_proj` | 1024 | Key projections |
| `v_proj` | `model.layers.{L}.self_attn.v_proj` | 1024 | Value projections |

*Contribution components auto-detect architecture. For Gemma-2 (post-sublayer norms), hooks `post_attention_layernorm`. For Llama/Mistral/Qwen (no post-norms), hooks `o_proj` directly.

### Monitoring

Project hidden states onto trait vectors:
```
raw_score = (hidden_state @ trait_vector) / ||trait_vector||
score = raw_score / mean(||hidden_state_at_layer||)
```
- Positive → expressing trait
- Negative → avoiding trait

Scoring pipelines (`pxs_grid.py`, `checkpoint_method_b.py`) normalize by dividing by mean activation norm at each trait's layer. This makes scores proportional to cos(θ), comparable across traits at different layers. Norms precomputed by `compute_activation_norms.py`.

**Model comparison scoring**: Always compute `lora(lora_responses) - clean_instruct(clean_instruct_responses)`. Each model generates its own text AND scores through its own model. The LoRA activation shift is part of the signal for alignment-specific probes — scoring LoRA text through clean instruct loses this. The baseline is always clean_instruct reading its own responses through the clean model.

**Layer selection:** Middle layers (6-16) generally best. Use `extraction_evaluation.py` to find optimal layer per trait. Steering results provide ground truth.

### Best Vector Selection

Automated via `utils/vectors.py:get_best_vector(experiment, trait)`:
- Uses steering results as ground truth — best delta with coherence ≥ 70
- Direction-aware: picks largest positive delta for `direction=positive`, most negative delta for `direction=negative`
- Returns `direction` field so callers know the steering orientation
- Auto-resolves extraction/steering variants from experiment config
- Searches all positions/components. Pass `component` and `position` to filter.

**Position syntax:** `<frame>[<slice>]` where frame is `prompt`, `response`, or `all`
- `response[:5]` — First 5 response tokens (default)
- `response[:]` — All response tokens (mean)
- `prompt[-1]` — Last prompt token (Arditi-style)

---

## Model Support

**Thinking models (Qwen3, etc.):** Thinking mode is automatically disabled via `enable_thinking=False` in `utils/model.py`. This prevents chain-of-thought tokens from inflating trait scores (the judge reads deliberation as expressing traits like confusion, conflictedness, etc.). Experiment-specific scripts that call `apply_chat_template` directly may need the flag added manually.

**Experiment config** (`experiments/{experiment}/config.json`):
```json
{
  "defaults": {
    "extraction": "base",
    "application": "instruct"
  },
  "model_variants": {
    "base": {"model": "{base_model}"},
    "instruct": {"model": "{instruct_model}"},
    "with_lora": {
      "model": "{instruct_model}",
      "lora": "{lora_adapter}"
    }
  }
}
```

---

## Troubleshooting

**Scenario files not found:**
```bash
# Create in datasets/traits/{category}/{trait}/
vim datasets/traits/category/my_trait/positive.txt
vim datasets/traits/category/my_trait/negative.txt
```

**Low vector separation (contrast < 20):**
- Add more contrasting scenarios
- Try probe method instead of mean_diff

**Out of memory:**
- Batch size auto-calculated from per-GPU free VRAM (uses min across GPUs for multi-GPU)
- Generation mode: analytical estimate with 1.15x overhead for `model.generate()` internals
- Extraction mode: empirical calibration (runs 1 forward pass, measures peak memory, derives batch size). Works for any architecture (MoE, MLA, FP8, INT4).
- MoE models: dominant memory cost is expert weight dequantization during prefill (INT4 → BF16). Sequential dequant (one projection at a time) keeps peak ~12 GB vs ~40 GB simultaneous.
- OOM recovery: automatic batch halving with CUDA memory cleanup (traceback clearing, outside-except-block pattern)
- Diagnostic: `Auto batch size: X` (generation) or `Calibrated: XMB/seq` (extraction)
- On Apple Silicon: auto-detects 50% of available unified memory (override with `MPS_MEMORY_GB`)

**MPS errors on Mac:**
```bash
# Requires PyTorch nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

---

## Model Server (Optional)

Keep the model loaded between script runs to avoid reload time:

```bash
# Terminal 1: Start server
source .env && python -u other/server/app.py --port 8765 --model {model}

# Terminal 2: Scripts auto-detect server for generation
python inference/generate_responses.py --experiment {experiment} --prompt-set {prompt_set}

# Or run steering eval via the server (no reload needed)
curl -X POST localhost:8765/eval/steering -H 'Content-Type: application/json' -d '{
  "experiment": "{experiment}",
  "traits": ["category/trait1", "category/trait2"],
  "model_variant": "{variant}",
  "extraction_variant": "{extraction_variant}",
  "layers": [15, 20, 25, 30]
}'
curl localhost:8765/eval/status/{task_id}  # check progress
```

Scripts automatically use the server if running, otherwise load locally. Use `--backend local` to force local loading. For INT4 MoE models (e.g. Kimi K2), the server automatically fuses expert weights via `grouped_mm` at load time.

**Model cache:** On first load of a compressed-tensors model, the fused weights are saved to `~/.cache/huggingface/fused_cache/`. Subsequent loads skip `from_pretrained` and load directly from cache (~5 min vs ~25 min). Use `POST /model/save` to manually trigger a save.

---

## Visualization

Start the server:
```bash
python visualization/serve.py  # Visit http://localhost:8000/
```

**Views:**
- **Extraction** — Best vectors summary, per-trait layer×method heatmaps, logit lens token decode
- **Steering** — Method comparison, layer×coefficient heatmaps, response browser
- **Trait Dynamics** — Token trajectory (cosine/normalized projection), Trait × Token heatmap (all traits as rows, tokens as columns — collapsible, requires 2+ traits), per-token magnitude, projection velocity, annotation bands, sentence overlays (toggleable cue_p + category bands for thought branches, with inline legends), model diff (Main/Diff toggle), Top Spans (sliding window or clause-level max-activating sequence finder, current or cross-prompt scope), Layers toggle (shows projections across all available layers for one trait, uses `layer_sensitivity` data from `model_diff/` when available)
- **Model Analysis** — Activation diagnostics (magnitude, massive dims, inter-layer similarity) + variant comparison (Cohen's d, cross-prompt projection spread)
- **Live Chat** — Interactive chat with real-time trait monitoring and steering controls

Auto-discovers experiments, traits, and prompts from `experiments/` directory.

---

## Further Reading

- **[extraction_pipeline.md](extraction_pipeline.md)** - Full pipeline documentation
- **[core_reference.md](core_reference.md)** - API reference
- **[extraction_guide.md](extraction_guide.md)** - Comprehensive extraction details
