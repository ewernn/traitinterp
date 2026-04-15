# Inference Guide

Monitor how trait vectors activate token-by-token as a model generates a response.

---

## What This Does

The inference pipeline projects each token's hidden-state activations onto previously extracted trait vectors, producing a per-token score for every trait. This lets us see *when* and *where* in a response a trait like sycophancy or deception spikes or dips.

The core math is a dot product: for each token, we take the residual stream activation at a given layer and project it onto the unit-norm trait vector. The resulting scalar tells us how strongly that token's representation aligns with the trait direction. Positive values mean the trait is expressed; negative means the opposite pole.

**Prerequisites:**
- Extracted trait vectors (from the extraction pipeline)
- A prompt set to generate responses from (`datasets/inference/starter_prompts/*.json`)
- Massive activation calibration (one-time, see below)

---

## Pipeline Modes

There are three ways to run inference, each suited to different workflows.

### Stream-through (default)

Generate responses, then run a prefill forward pass with projection hooks. The model processes each response, and hooks compute dot products on GPU at the specified layers. Only small score arrays cross the GPU-CPU boundary — no intermediate `.pt` files.

This is the fastest path and the one you'll use most often.

```bash
python inference/run_inference_pipeline.py \
    --experiment {experiment} \
    --prompt-set {prompt_set}
```

### Capture + from-activations (two-step)

First capture raw activations to `.pt` files, then project onto trait vectors as a separate step. This is useful when you want to re-project the same activations with different vectors, layers, or traits without re-running the model.

```bash
# Step 1: Capture raw activations
python inference/run_inference_pipeline.py --capture \
    --experiment {experiment} \
    --prompt-set {prompt_set}

# Step 2: Project from saved files
python inference/run_inference_pipeline.py --from-activations \
    --experiment {experiment} \
    --prompt-set {prompt_set}
```

The capture step saves per-token activations at each layer as `.pt` tensors. These are large (all layers x all tokens x hidden_dim) — delete them after projecting.

### Generate separately

You can also generate responses as a standalone step, then project later:

```bash
# Generate only
python inference/generate_responses.py \
    --experiment {experiment} \
    --prompt-set {prompt_set}

# Project existing responses (stream-through)
python inference/run_inference_pipeline.py \
    --experiment {experiment} \
    --prompt-set {prompt_set}
```

Re-running the pipeline without `--regenerate` skips generation and re-projects existing responses. This means you can adjust layers or traits without regenerating.

---

## Massive Activation Calibration

Some transformer models have "massive activations" — a handful of hidden dimensions with values 100x larger than the rest (Sun et al. 2024). These outlier dimensions dominate dot products and distort projection scores.

We calibrate once per experiment using a standardized prompt set:

```bash
python analysis/vectors/massive_activations.py --experiment {experiment}
```

This runs a small set of Alpaca prompts through the model, identifies which dimensions are massive at each layer, and saves the results. The visualization dashboard uses this data to offer filtered views that exclude massive dims.

Run this before your first inference pass. It only needs to be done once per model/experiment.

---

## CLI Reference

The main entry point is `inference/run_inference_pipeline.py`. Key flags:

### Required
- `--experiment` — Experiment name (must have a `config.json` with model variants)
- `--prompt-set` — Which prompt set to use (path relative to `datasets/inference/`, e.g., `starter_prompts/general`)

### Pipeline control
- `--capture` — Save raw `.pt` activations instead of projecting (mutually exclusive with `--from-activations`)
- `--from-activations` — Project from saved `.pt` files (after `--capture`)
- `--regenerate` — Force re-generate responses even if they already exist

### Projection
- `--layers` — Which layers to project at. Default: `best,best+5` (auto-selects the best steering layer plus 5 layers above it). Also accepts explicit layer numbers (`25,30,35`) or the keyword `best` alone.
- `--traits` — Comma-separated traits to project onto (e.g., `starter_traits/sycophancy,alignment/honesty`). Default: all extracted traits in the experiment.
- `--component` — Hook component. Default: `residual`. Also supports `attn_contribution`.
- `--centered` — Subtract the baseline (class centroid projection) from scores, centering them around zero.
- `--force` — Re-run even if projection files already exist (default: skip existing).

### Generation
- `--max-new-tokens` — Maximum tokens to generate per response. Default: `50`.
- `--temperature` — Sampling temperature. Default: `0.0` (greedy).
- `--model-variant` — Which model variant from the experiment config to use.

### Model
- `--load-in-4bit` — Load model in 4-bit quantization (for large models on limited VRAM).
- `--backend` — `local` (default, HF in-process), `auto` (try server first, fall back to local), or `vllm` (generation-only, no hooks, no LoRA; projection and capture stages require `local`).

---

## Layer Selection

The `--layers` flag accepts a DSL that resolves relative to the best steering layer for each trait:

| Spec | Meaning |
|------|---------|
| `best` | Single best steering layer |
| `best,best+5` | Best layer + 5 layers above (default) |
| `best+5` | Just the layer 5 above best |
| `25,30,35` | Explicit layer numbers |

The "best" layer comes from steering evaluation results — the layer whose vector produced the largest trait score delta when used for steering. If no steering results exist, the pipeline falls back to available vectors.

---

## Projection Scores and Normalization

Each projection JSON contains three types of scores:

### Raw projection (`prompt`, `response` arrays)
The dot product of the token's activation with the unit trait vector: `score = h . v_hat`. This is in activation-space units, which vary across layers (deeper layers have larger norms). Good for comparing tokens within a single layer.

### Normalized projection (`normalized_prompt`, `normalized_response`)
Raw scores divided by the mean activation norm across all tokens at that layer: `score = (h . v_hat) / mean(||h||)`. This adjusts for layer scale, making scores more comparable across layers. This is the default mode in the visualization dashboard.

### Cosine similarity
Available via `core.math.normalize_projections(raw, norms, 'cosine')`. Divides each token's raw score by its own activation norm: `score = (h . v_hat) / ||h||`. True cosine similarity — measures direction only, not magnitude. Useful for understanding if a token *points toward* the trait regardless of activation strength.

### Baseline centering
When `--centered` is used, the baseline (projection of the training class centroid) is subtracted from all scores. This shifts the zero point to represent "average" trait expression, so positive means above-average and negative means below-average.

---

## Output Format

### Response JSONs

Generated responses are saved individually:
```
experiments/{exp}/inference/{variant}/responses/{prompt_set}/{id}.json
```

Each contains the prompt, response text, token list, token IDs, and a `prompt_end` index marking where the prompt ends and the response begins.

### Projection JSONs

One file per prompt per trait:
```
experiments/{exp}/inference/{variant}/projections/{trait}/{prompt_set}/{id}.json
```

Structure:
```json
{
  "metadata": {
    "prompt_id": "0",
    "prompt_set": "general",
    "n_prompt_tokens": 42,
    "n_response_tokens": 50,
    "component": "residual",
    "position": "response[:]",
    "centered": false,
    "score_mode": "raw+normalized",
    "projection_date": "2026-03-31T..."
  },
  "projections": [
    {
      "method": "probe",
      "layer": 20,
      "selection_source": "steering",
      "baseline": 1.234,
      "prompt": [0.5, 0.3, ...],
      "response": [1.2, 0.8, ...],
      "token_norms": {
        "prompt": [450.1, 448.3, ...],
        "response": [455.2, 460.1, ...]
      },
      "normalized_prompt": [0.0011, 0.0007, ...],
      "normalized_response": [0.0026, 0.0018, ...]
    }
  ]
}
```

The `projections` array contains one entry per layer/vector. Each entry has parallel arrays — one score per token — for both prompt and response segments.

### Raw activations (capture mode)

```
experiments/{exp}/inference/{variant}/raw/residual/{prompt_set}/{id}.pt
```

These are large PyTorch files containing per-token activations at every captured layer. Delete them after projecting — they're regenerable and excluded from R2 sync.

---

## Visualization

Start the dashboard to explore inference results:

```bash
python visualization/serve.py
# Visit http://localhost:8000/
```

The **Trait Dynamics** view shows per-token projection trajectories. Select an experiment, prompt set, and prompt to see how trait scores evolve token-by-token across the response. You can:

- Switch between raw and normalized score modes
- Compare multiple traits on the same prompt
- View multiple layers for the same trait
- See token-level detail (hover for exact scores)

The **Model Analysis** view supports comparing how different model variants (base vs instruct, clean vs LoRA) score on the same tokens. Run inference on each variant separately, then compare in the dashboard.

---

## Thinking Models

For thinking models like Qwen3, thinking mode is automatically disabled via `enable_thinking=False` in `utils/model.py`. Chain-of-thought tokens would otherwise inflate trait scores by adding lengthy internal reasoning before the actual response.

---

## Tensor Parallelism

For large models, inference capture supports multi-GPU via `torchrun`:

```bash
torchrun --nproc_per_node=8 inference/run_inference_pipeline.py --capture \
    --experiment {experiment} \
    --prompt-set {prompt_set} \
    --component residual \
    --layers 9,12,18,24,30,36
```

Note: `massive_activations.py` does not support TP — run it without `torchrun`.

---

## Common Patterns

### Re-project with different layers
```bash
python inference/run_inference_pipeline.py \
    --experiment {experiment} \
    --prompt-set {prompt_set} \
    --layers 15,20,25,30
```

### Project a single trait
```bash
python inference/run_inference_pipeline.py \
    --experiment {experiment} \
    --prompt-set {prompt_set} \
    --traits starter_traits/sycophancy
```

### Resume after crash
```bash
python inference/run_inference_pipeline.py \
    --experiment {experiment} \
    --prompt-set {prompt_set} \
    --force
```

### Full workflow from scratch
```bash
# 1. Calibrate massive dims (once)
python analysis/vectors/massive_activations.py --experiment {experiment}

# 2. Run inference pipeline
python inference/run_inference_pipeline.py \
    --experiment {experiment} \
    --prompt-set starter_prompts/general

# 3. Visualize
python visualization/serve.py
```
