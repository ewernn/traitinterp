# Extraction

```
python extraction/run_extraction_pipeline.py [flags]
```

Full extraction pipeline -- generate responses, vet quality, capture activations, compute trait vectors.

For pipeline concepts and scenario design, see [Extraction Guide](../../extraction_guide.md).

## Flags

### Trait Selection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment` | str | **required** | Experiment name |
| `--traits` | str | | Comma-separated traits, e.g. `starter_traits/sycophancy,starter_traits/refusal` |
| `--category` | str | | Process all traits in a category, e.g. `starter_traits` |

### Generation

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--rollouts` | int | `1` | Number of response rollouts per scenario |
| `--temperature` | float | `0.0` | Sampling temperature |
| `--seed` | int | None | Random seed for reproducible sampling (requires temperature > 0) |
| `--max-new-tokens` | int | None | Max tokens per response (overrides `extraction_config.yaml`; auto: 16 for base, 64 for instruct) |
| `--replication-level` | str | `lightweight` | `lightweight` uses simplified prompts + serial generation. `full` enables paper-verbatim batched story generation for categories that opt in via `extraction_config.yaml` (e.g., `ant_emotion_concepts`) |
| `--topics` | int | None | Full-mode only. Limit topics from `topics_file` to the first N entries |
| `--stories-per-batch` | int | None | Full-mode only. Override `extraction_config.yaml`'s `stories_per_batch` (default 12 from the Anthropic paper) |

### Vetting

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--vet-responses` | flag | off | Enable LLM judge quality vetting (requires `OPENAI_API_KEY`) |
| `--pos-threshold` | int | `60` | Minimum score for positive responses (0--100) |
| `--neg-threshold` | int | `40` | Maximum score for negative responses (0--100) |
| `--max-concurrent` | int | `100` | Max concurrent vetting API requests |
| `--paired-filter` | flag | off | Filter paired pos/neg responses together |
| `--adaptive` | flag | off | Use adaptive extraction position derived from LLM judge scores |

### Extraction

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--methods` | str | `probe` | Comma-separated extraction methods: `probe`, `mean_diff`, `gradient`, `rfm`, `random_baseline` |
| `--component` | str | `residual` | Activation component: `residual`, `attn_out`, `mlp_out` |
| `--position` | str | None | Token position for extraction (auto: `response[:5]` for base, `response[:]` for instruct) |
| `--layers` | str | None | Layers to extract, e.g. `25,30,35` or `20-40` (default: all layers) |
| `--val-split` | float | `0.1` | Validation split fraction |
| `--save-activations` | flag | off | Save raw `.pt` activation files alongside vectors |

### Model & Hardware

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model-variant` | str | None | Model variant from experiment config (default: `config.defaults.extraction`) |
| `--load-in-4bit` | flag | off | 4-bit quantization (requires CUDA + bitsandbytes) |
| `--bnb-4bit-quant-type` | str | `nf4` | Quantization type: `nf4` or `fp4` |
| `--base-model` | flag | off | Force model type to base/pretrained |
| `--it-model` | flag | off | Force model type to instruct-tuned |
| `--backend` | str | `auto` | Model backend: `auto`, `local`, `vllm` |

### Pipeline Control

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--only-stage` | ints | all | Run specific stages only, e.g. `3,4` (stages: 1=generate, 2=vet, 3+4=extract, 5=logit lens, 6=evaluate) |
| `--no-logit-lens` | flag | off | Skip stage 5 (logit lens runs by default since the model is already loaded) |
| `--force` | flag | off | Force recomputation of existing results |

## Examples

Extract a single trait:

```bash
python extraction/run_extraction_pipeline.py \
    --experiment gemma-2-2b \
    --traits starter_traits/sycophancy
```

Extract multiple traits in one run:

```bash
python extraction/run_extraction_pipeline.py \
    --experiment gemma-2-2b \
    --traits starter_traits/sycophancy,starter_traits/formality
```

Extract all traits in a category:

```bash
python extraction/run_extraction_pipeline.py \
    --experiment gemma-2-2b \
    --category starter_traits
```

Enable vetting with custom thresholds:

```bash
python extraction/run_extraction_pipeline.py \
    --experiment gemma-2-2b \
    --traits starter_traits/sycophancy \
    --vet-responses --pos-threshold 70 --neg-threshold 30
```

Rerun extraction and evaluation stages only:

```bash
python extraction/run_extraction_pipeline.py \
    --experiment gemma-2-2b \
    --traits starter_traits/sycophancy \
    --only-stage 3,4 --force
```

!!! tip
    Per-trait overrides for `position`, `max_new_tokens`, `methods`, `temperature`, and `rollouts` can be set in `extraction_config.yaml` inside the trait directory. CLI flags always take precedence over YAML config.
