# Experiment Setup

Everything needed to configure and run a new experiment.

---

## Experiment directory

Experiments live in `experiments/{name}/`. The directory is created automatically when you run any pipeline script -- you only need to create `config.json` manually beforehand.

```
experiments/my-experiment/
├── config.json              # You create this (required)
├── extraction/              # Created by extraction pipeline
├── inference/               # Created by inference pipeline
└── steering/                # Created by steering pipeline
```

---

## config.json

The only file you create manually. Defines which model(s) to use.

### Minimal example

A single model variant is enough to get started:

```json
{
  "model_variants": {
    "my_model": { "model": "{huggingface_org}/{model_name}" }
  }
}
```

### Full example

Multiple variants with defaults and a LoRA adapter:

```json
{
  "defaults": {
    "extraction": "base",
    "application": "instruct"
  },
  "model_variants": {
    "base": { "model": "{huggingface_org}/{base_model}" },
    "instruct": { "model": "{huggingface_org}/{instruct_model}" },
    "finetuned": {
      "model": "{huggingface_org}/{instruct_model}",
      "lora": "{huggingface_org}/{lora_adapter}"
    }
  }
}
```

### Fields

| Field | Required | Description |
|---|---|---|
| `model_variants` | Yes | Map of variant name to model spec. Each spec requires a `model` field (HuggingFace model ID). Optionally includes `lora` (HuggingFace adapter ID or local path). |
| `defaults.extraction` | No | Variant used for trait vector extraction (typically the base model). Falls back to the first variant in `model_variants` if omitted. |
| `defaults.application` | No | Variant used for inference and steering (typically the instruct model). Falls back to the first variant if omitted. |

!!! tip "Variant naming"
    Variant names are free-form strings used as directory names throughout the experiment. Choose short, descriptive names (`base`, `instruct`, `rank32`, etc.).

!!! warning "LoRA paths"
    The `lora` field accepts either a HuggingFace adapter ID (e.g., `ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice`) or a local filesystem path. HuggingFace IDs are downloaded automatically; local paths must exist at runtime.

---

## Environment variables

Copy `.env.example` to `.env` and fill in the values you need.

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | For gated models | HuggingFace access token. Required for gated models (Llama, Gemma, etc.). Not needed for open models like Qwen. |
| `OPENAI_API_KEY` | For LLM judge | Used by vetting (extraction `--vet-responses`), scoring (steering evaluation), and coherence checks. Not needed for basic extraction or inference. |
| `R2_ACCESS_KEY_ID` | For R2 sync | Cloudflare R2 credentials for downloading/uploading experiment data. |
| `R2_SECRET_ACCESS_KEY` | For R2 sync | Cloudflare R2 credentials. |
| `R2_ENDPOINT` | For R2 sync | R2 endpoint URL. |
| `R2_BUCKET_NAME` | For R2 sync | R2 bucket name (default: `trait-interp-bucket`). |
| `EXPERIMENTS_DIR` | No | Redirects all experiment I/O to a different location. Useful for storing large experiment data on a separate drive. |

!!! warning "OPENAI_API_KEY"
    Steering evaluation will fail with a hard error if this is not set. Basic extraction (without `--vet-responses`) and inference do **not** require it.

---

## Model config (optional)

Files in `config/models/{slug}.yaml` provide model architecture metadata. These are **not required** -- the pipeline auto-detects architecture from HuggingFace model configs. 23 model configs ship with the repo. You only need to create one if auto-detection of base vs instruct fails for your model.

### Example

```yaml
huggingface_id: {huggingface_org}/{model_name}
variant: instruct          # base | instruct
supports_system_prompt: true
num_hidden_layers: 32
hidden_size: 4096
```

### Key fields

| Field | Description |
|---|---|
| `variant` | `base` or `instruct` -- controls extraction defaults (position, chat template). This is the main reason to create a config. |
| `supports_system_prompt` | Whether the model's chat template accepts a system message |
| `num_hidden_layers` | Number of transformer layers (used for layer range defaults like `--layers "30%-60%"`) |
| `hidden_size` | Residual stream dimension |

Most other fields (`model_type`, `num_attention_heads`, `intermediate_size`, etc.) are auto-detected from HuggingFace and rarely need manual specification.

---

## The `starter` experiment

The repo ships with `experiments/starter/config.json` preconfigured with two models:

```json
{
  "defaults": {
    "extraction": "instruct",
    "application": "instruct"
  },
  "model_variants": {
    "base": {
      "model": "Qwen/Qwen3.5-9B-Base"
    },
    "instruct": {
      "model": "Qwen/Qwen3.5-9B"
    }
  }
}
```

Use it for first-time testing:

```bash
# Extract sycophancy vectors using the starter experiment
python extraction/run_extraction_pipeline.py \
    --experiment starter \
    --traits starter_traits/sycophancy

# Run inference monitoring
python inference/run_inference_pipeline.py \
    --experiment starter \
    --prompt-set starter_prompts/general
```

!!! tip "No HF_TOKEN required"
    Both starter variants (`base` and `instruct`) use Qwen models, which are open and do not require a HuggingFace token. Swap in a gated model (Llama, Gemma, etc.) and you'll need `HF_TOKEN` set.

---

## Next steps

- [Trait Dataset Format](trait-format.md) -- file formats for trait datasets
- [Extraction Guide](../../extraction_guide.md) -- full extraction pipeline walkthrough
- [Inference Guide](../../inference_guide.md) -- per-token monitoring setup
