# Steering

```
python steering/run_steering_eval.py [flags]
```

Steering evaluation -- validate trait vectors via causal intervention with adaptive coefficient search.

For search algorithm details and interpretation, see [Steering Guide](../../steering_guide.md).

## Flags

### Trait Selection

Mutually exclusive: provide exactly one of `--traits`, `--vector-from-trait`, or `--rescore`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment` | str | **required** | Experiment name |
| `--traits` | str | | Comma-separated traits, e.g. `starter_traits/sycophancy` |
| `--vector-from-trait` | str | | Single trait as `experiment/category/trait` (cross-experiment vectors) |
| `--rescore` | str | | Re-score existing responses for a trait (no GPU needed) |

### Evaluation

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prompt-set` | str | `steering` | Prompt set for evaluation |
| `--questions-file` | str | None | Path to custom questions file |
| `--no-custom-prompt` | flag | off | Skip trait-specific eval prompt (use default) |
| `--eval-prompt-from` | str | None | Load eval prompt from a trait path |
| `--trait-judge` | str | None | Path to custom judge configuration |
| `--subset` | int | `5` | Number of prompts to evaluate |
| `--max-new-tokens` | int | `64` | Max tokens per steered response |
| `--direction` | str | None | Force steering direction: `positive` or `negative` |
| `--no-relevance-check` | flag | off | Skip relevance filtering of prompts |

`--no-custom-prompt`, `--eval-prompt-from`, and `--trait-judge` are mutually exclusive.

### Search Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--layers` | str | `30%-60%` | Layers to search (supports percentages, ranges, comma-separated) |
| `--trait-layers` | nargs | | Per-trait layer overrides, format: `TRAIT:LAYERS` |
| `--coefficients` | str | None | Manual coefficients (comma-separated, skips adaptive search) |
| `--search-steps` | int | `5` | Number of search iterations |
| `--up-mult` | float | `1.3` | Multiplier for upward coefficient search |
| `--down-mult` | float | `0.85` | Multiplier for downward search |
| `--start-mult` | float | `0.7` | Starting coefficient multiplier |
| `--momentum` | float | `0.1` | Search momentum |

### Vector Selection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--method` | str | `probe` | Extraction method |
| `--component` | str | `residual` | Activation component |
| `--position` | str | None | Token position override (auto-detected from extraction if None) |
| `--vector-experiment` | str | None | Load vectors from a different experiment |
| `--extraction-variant` | str | None | Model variant used during extraction |

### Model & Hardware

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model-variant` | str | None | Model variant from experiment config (default: `config.defaults.application`) |
| `--load-in-4bit` | flag | off | 4-bit quantization (requires CUDA + bitsandbytes) |
| `--bnb-4bit-quant-type` | str | `nf4` | Quantization type: `nf4` or `fp4` |
| `--min-coherence` | float | `77` | Minimum coherence threshold for steered responses |
| `--backend` | str | `auto` | Model backend: `auto`, `local`, `vllm` |

### Pipeline Control

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--baseline-only` | flag | off | Score unsteered responses only (no steering) |
| `--force` | flag | off | Force recomputation of existing results |
| `--no-batch` | flag | off | Evaluate traits sequentially instead of batched |
| `--regenerate-responses` | flag | off | Force regenerate baseline responses |
| `--save-responses` | str | `best` | Which steered responses to save: `all`, `best`, `none` |
| `--ablation` | int | None | Ablation mode: remove direction at specified layer |
| `--dry-run` | flag | off | Print configuration without executing |

## Examples

Run steering evaluation for a single trait:

```bash
python steering/run_steering_eval.py \
    --experiment gemma-2-2b \
    --traits starter_traits/sycophancy
```

Batch-evaluate multiple traits:

```bash
python steering/run_steering_eval.py \
    --experiment gemma-2-2b \
    --traits starter_traits/sycophancy,starter_traits/formality
```

Use vectors from a different experiment:

```bash
python steering/run_steering_eval.py \
    --experiment gemma-2-2b \
    --vector-from-trait other_exp/starter_traits/sycophancy
```

Evaluate specific coefficients (skip search):

```bash
python steering/run_steering_eval.py \
    --experiment gemma-2-2b \
    --traits starter_traits/sycophancy \
    --coefficients 0.5,1.0,1.5,2.0
```

Score unsteered baselines only:

```bash
python steering/run_steering_eval.py \
    --experiment gemma-2-2b \
    --traits starter_traits/sycophancy \
    --baseline-only --save-responses all
```

Ablation -- remove trait direction at a layer:

```bash
python steering/run_steering_eval.py \
    --experiment gemma-2-2b \
    --traits starter_traits/sycophancy \
    --ablation 25
```

Re-score existing responses (no GPU required):

```bash
python steering/run_steering_eval.py \
    --experiment gemma-2-2b \
    --rescore starter_traits/sycophancy
```

!!! tip
    Use `--dry-run` to preview the resolved configuration (layers, coefficients, model variant) before launching a full evaluation.
