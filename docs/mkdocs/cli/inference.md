# Inference

Two scripts for generating model responses and projecting activations onto trait vectors.

For pipeline modes and projection scores, see [Inference Guide](../../inference_guide.md).

---

## `inference/run_inference_pipeline.py`

Full inference pipeline -- generate responses and project onto trait vectors (stream-through), capture raw activations, or project from saved activations.

```bash
python inference/run_inference_pipeline.py \
    --experiment <experiment> --prompt-set <prompt_set>
```

### Required

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment` | str | required | Experiment name |
| `--prompt-set` | str | required | Prompt set path, e.g. `starter_prompts/general` |

### Pipeline Mode

Mutually exclusive flags that control what the pipeline does. Default is stream-through (generate + project in one pass).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| *(default)* | -- | -- | Stream-through: generate + project in one pass |
| `--capture` | flag | off | Save raw `.pt` activations instead of projecting |
| `--from-activations` | flag | off | Project from saved `.pt` files (run after `--capture`) |
| `--regenerate` | flag | off | Force re-generate responses even if they exist |

### Projection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--traits` | str | all | Comma-separated traits to project (default: all extracted) |
| `--layers` | str | `best,best+5` | Layers to project onto. Supports: specific (`25,30`), ranges (`20-40`), `best` keyword (from steering), `best+N` |
| `--component` | str | `residual` | Activation component: `residual`, `attn_contribution`, `mlp_contribution` |
| `--centered` | flag | off | Center activations before projection |
| `--force` | flag | off | Re-run even if projections exist (default: skip existing) |
| `--score-mode` | str | `normalized` | Score normalization: `raw` (no divisor), `normalized` (divide by mean &#124;&#124;h&#124;&#124; over response), `cosine` (divide by per-token &#124;&#124;h&#124;&#124;) |

### Generation

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-new-tokens` | int | `512` | Max tokens to generate |
| `--temperature` | float | `0.0` | Sampling temperature |
| `--model-variant` | str | config default | Model variant (default: `config.defaults.application`) |

### Model

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--load-in-4bit` | flag | off | 4-bit quantization |
| `--backend` | str | `local` | Model backend: `local` (HF in-process, default), `auto` (try server then fall back to local), `vllm` (generation only, no hooks) |

### Examples

```bash
# Stream-through: generate responses + project onto all traits
python inference/run_inference_pipeline.py \
    --experiment my_exp --prompt-set starter_prompts/general

# Capture raw activations for later re-projection
python inference/run_inference_pipeline.py \
    --experiment my_exp --prompt-set starter_prompts/general --capture

# Project from saved activations at specific layers
python inference/run_inference_pipeline.py \
    --experiment my_exp --prompt-set starter_prompts/general \
    --from-activations --layers 25,30,35

# Project specific traits using best steering layer
python inference/run_inference_pipeline.py \
    --experiment my_exp --prompt-set starter_prompts/general \
    --traits starter_traits/sycophancy --layers best

# Cosine similarity scores instead of normalized projection
python inference/run_inference_pipeline.py \
    --experiment my_exp --prompt-set starter_prompts/general \
    --score-mode cosine
```

---

## `inference/generate_responses.py`

Generate model responses for inference prompts (standalone). Called automatically by the pipeline but can run independently.

Two modes: **Mode A** generates from the model; **Mode B** writes external response text (tokenizer only, no GPU).

```bash
python inference/generate_responses.py \
    --experiment <experiment> --prompt-set <prompt_set>
```

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment` | str | required | Experiment name |
| `--prompt-set` | str | required | Prompt set from `datasets/inference/{name}.json` |
| `--model-variant` | str | config default | Model variant (default: `config.defaults.application`) |

### Generation (Mode A)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-new-tokens` | int | `512` | Max tokens to generate |
| `--temperature` | float | `0.0` | Sampling temperature |
| `--prefill` | str | none | String prepended to model response before generation |

### External Responses (Mode B)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--from-responses` | str | none | Path to `{id: response_text}` JSON. Tokenizer only, no GPU |

### Common

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--force` | flag | off | Re-run even if responses exist (default: skip existing) |
| `--limit` | int | none | Max prompts to process |
| `--output-suffix` | str | none | Suffix appended to output directory name |
| `--load-in-4bit` | flag | off | 4-bit quantization |
| `--backend` | str | `local` | Model backend: `local` (default), `auto`, `vllm` |

### Examples

```bash
# Generate responses for a prompt set
python inference/generate_responses.py \
    --experiment my_exp --prompt-set starter_prompts/general

# Generate with prefill text
python inference/generate_responses.py \
    --experiment my_exp --prompt-set starter_prompts/general \
    --prefill "I think"

# Import external responses (no GPU needed)
python inference/generate_responses.py \
    --experiment my_exp --prompt-set starter_prompts/general \
    --from-responses external_responses.json

# Generate first 10 prompts with a specific variant
python inference/generate_responses.py \
    --experiment my_exp --prompt-set starter_prompts/general \
    --model-variant rm_lora --limit 10
```

!!! tip
    Use `--from-responses` to import responses from external models (e.g., API-based models) without loading any weights. Only a tokenizer is needed.
