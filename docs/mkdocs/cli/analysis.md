# Analysis

Scripts for evaluating vectors, comparing model variants, and running benchmarks.

See also [analysis/README.md](https://github.com/ewernn/traitinterp/tree/main/analysis) for a quick overview.

---

## Vectors

### `analysis/vectors/massive_activations.py`

Calibrate massive activation dimensions (mandatory before inference). Identifies dimensions with abnormally large values and saves calibration data for downstream cleaning.

```bash
python analysis/vectors/massive_activations.py --experiment <experiment>
```

#### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment` | str | required | Experiment name |
| `--model-variant` | str | config default | Model variant |
| `--all-variants` | flag | off | Run for all model variants in experiment config |
| `--prompt-set` | str | calibration | Prompt set to analyze (default: built-in calibration dataset) |
| `--prompts-file` | str | none | Direct path to prompts JSON (overrides `--prompt-set` lookup) |
| `--prompt-ids` | str | all | Comma-separated prompt IDs to analyze |
| `--top-k` | int | `5` | Top K dims to track per layer |
| `--per-token` | flag | off | Include per-token analysis (verbose, for research) |
| `--per-layer` | flag | off | Compute per-layer stats for massive dims (requires calibration first) |
| `--load-in-4bit` | flag | off | 4-bit quantization |
| `--output` | str | auto | Output path (default: canonical path) |

#### Examples

```bash
# Calibrate model (mandatory first step, uses built-in Alpaca prompts)
python analysis/vectors/massive_activations.py --experiment gemma-2-2b-it

# Calibrate all variants in an experiment
python analysis/vectors/massive_activations.py --experiment gemma-2-2b-it --all-variants

# Analyze a specific prompt set (research mode)
python analysis/vectors/massive_activations.py \
    --experiment gemma-2-2b-it --prompt-set jailbreak_subset

# Per-layer massive dim stats (requires calibration first)
python analysis/vectors/massive_activations.py --experiment gemma-2-2b-it --per-layer
```

---

### `analysis/vectors/extraction_evaluation.py`

Evaluate extracted vectors on held-out validation data. Reports accuracy, effect size (Cohen's d), and polarity correctness per trait/method/layer.

Uses [Python Fire](https://github.com/google/python-fire) -- pass arguments as `--flag=value` or positionally.

```bash
python analysis/vectors/extraction_evaluation.py --experiment <experiment>
```

#### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment` | str | required | Experiment name |
| `--model_variant` | str | config default | Model variant (default: `config.defaults.extraction`) |
| `--methods` | str | `mean_diff,probe,gradient` | Comma-separated extraction methods to evaluate |
| `--layers` | str | all available | Comma-separated layers (default: all discovered layers) |
| `--component` | str | `residual` | Activation component: `residual`, `attn_out`, `mlp_out` |
| `--position` | str | `response[:5]` | Token position |
| `--verbose` | bool | `False` | Print detailed per-method/layer analysis |

#### Examples

```bash
# Evaluate all traits with defaults
python analysis/vectors/extraction_evaluation.py --experiment gemma-2-2b-it

# Evaluate specific methods and layers with verbose output
python analysis/vectors/extraction_evaluation.py \
    --experiment gemma-2-2b-it --methods=probe,mean_diff --layers=20,25,30 --verbose=True

# Evaluate a different component
python analysis/vectors/extraction_evaluation.py \
    --experiment gemma-2-2b-it --component=attn_out
```

---

### `analysis/vectors/logit_lens.py`

Project trait vectors through the model's unembedding matrix to reveal what tokens each direction "means". Shows top tokens in both the toward (+) and away (-) directions.

```bash
python analysis/vectors/logit_lens.py --experiment <experiment> --traits <trait>
```

#### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment` | str | required | Experiment name |
| `--model-variant` | str | config default | Model variant |
| `--traits` | str | none | Trait path (e.g., `safety/refusal`). Required unless `--all-traits` |
| `--all-traits` | flag | off | Analyze all extracted traits |
| `--top-k` | int | `20` | Number of tokens to show per direction |
| `--no-norm` | flag | off | Skip RMSNorm before projection |
| `--filter-common` | flag | off | Filter to common English tokens only |
| `--max-vocab` | int | `10000` | Max vocab index for common filter |
| `--save` | flag | off | Save results to canonical per-trait JSON |

#### Examples

```bash
# Analyze a single trait
python analysis/vectors/logit_lens.py --experiment gemma-2-2b-it --traits safety/refusal

# Analyze all traits and save results
python analysis/vectors/logit_lens.py --experiment gemma-2-2b-it --all-traits --save

# Filter to common tokens, skip RMSNorm
python analysis/vectors/logit_lens.py \
    --experiment gemma-2-2b-it --traits safety/refusal \
    --filter-common --no-norm
```

---

## Model Diff

### `analysis/model_diff/compare_variants.py`

Compare activations between two model variants. Computes diff vectors (B - A), Cohen's d effect sizes on trait projections (unpaired and paired), and cosine similarity between diff vectors and trait vectors.

Requires a vector specification: either `--use-best-vector` (auto-select from steering results) or `--method` + `--position` (explicit).

```bash
python analysis/model_diff/compare_variants.py \
    --experiment <experiment> --variant-a <a> --variant-b <b> \
    --prompt-set <prompt_set> --use-best-vector
```

#### Required

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment` | str | required | Experiment name |
| `--variant-a` | str | required | Baseline variant (e.g., `instruct`) |
| `--variant-b` | str | required | Comparison variant (e.g., `rm_lora`) |
| `--prompt-set` | str | required | Prompt set (e.g., `rm_syco/train_100`) |

#### Vector Specification (one required)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use-best-vector` | flag | off | Auto-select method/position from steering results |
| `--method` | str | none | Vector extraction method (e.g., `probe`, `mean_diff`). Must pair with `--position` |
| `--position` | str | none | Token position for vectors (e.g., `response[:5]`). Must pair with `--method` |

#### Optional

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--traits` | str | all extracted | Comma-separated traits to analyze |
| `--component` | str | `residual` | Activation component |
| `--vector-experiment` | str | same as `--experiment` | Experiment to load trait vectors from (when vectors were extracted elsewhere) |
| `--use-existing-diff` | flag | off | Use existing `diff_vectors.pt` instead of recomputing. Only cosine similarity (no effect sizes) |

#### Examples

```bash
# Auto-select best vectors from steering results
python analysis/model_diff/compare_variants.py \
    --experiment rm_syco \
    --variant-a instruct --variant-b rm_lora \
    --prompt-set rm_syco/train_100 --use-best-vector

# Explicit vector specification
python analysis/model_diff/compare_variants.py \
    --experiment rm_syco \
    --variant-a instruct --variant-b rm_lora \
    --prompt-set rm_syco/train_100 \
    --method probe --position "response[:5]"

# Recompute cosine similarity using existing diff vectors
python analysis/model_diff/compare_variants.py \
    --experiment rm_syco \
    --variant-a instruct --variant-b rm_lora \
    --prompt-set rm_syco/train_100 \
    --use-best-vector --use-existing-diff
```

---

### `analysis/model_diff/per_token_diff.py`

Per-token projection difference between two model variants. Computes delta (B - A) at each response token, splits into clauses, and ranks by mean delta.

```bash
python analysis/model_diff/per_token_diff.py \
    --experiment <experiment> --variant-a <a> --variant-b <b> \
    --prompt-set <prompt_set> --traits <trait>
```

#### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment` | str | required | Experiment name |
| `--variant-a` | str | required | Baseline variant |
| `--variant-b` | str | required | Comparison variant |
| `--prompt-set` | str | required | Prompt set |
| `--traits` | str | required | Trait path (e.g., `rm_hack/ulterior_motive`) or `all` |
| `--variant-a-prompt-set` | str | same as `--prompt-set` | Prompt set for variant-a if different (e.g., for replay data) |
| `--layer` | int | auto (best steering) | Layer for projection reading |
| `--top-pct` | float | `5` | Print top N% of clauses |

#### Examples

```bash
# Analyze a single trait
python analysis/model_diff/per_token_diff.py \
    --experiment audit-bench \
    --variant-a instruct --variant-b rm_lora \
    --prompt-set rm_syco/train_100 \
    --traits rm_hack/ulterior_motive

# Analyze all traits, show top 10% of clauses
python analysis/model_diff/per_token_diff.py \
    --experiment audit-bench \
    --variant-a instruct --variant-b rm_lora \
    --prompt-set rm_syco/train_100 \
    --traits all --top-pct 10
```

---

### `analysis/model_diff/layer_sensitivity.py`

Cross-layer projection diff analysis. Projects raw activations from both variants onto trait vectors at multiple layers and measures cross-layer consistency (prompt-level and token-level correlations).

```bash
python analysis/model_diff/layer_sensitivity.py \
    --experiment <experiment> --variant-a <a> --variant-b <b> \
    --prompt-set <prompt_set> --layers <layers>
```

#### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment` | str | required | Experiment name |
| `--variant-a` | str | required | Baseline variant |
| `--variant-b` | str | required | Comparison variant |
| `--prompt-set` | str | required | Prompt set |
| `--layers` | str | required | Comma-separated layers (e.g., `20,25,30,35,40`) |
| `--traits` | str | all extracted | Comma-separated traits |
| `--method` | str | `probe` | Vector extraction method |
| `--position` | str | `response[:5]` | Token position |
| `--component` | str | `residual` | Activation component |

#### Examples

```bash
# Sweep 5 layers
python analysis/model_diff/layer_sensitivity.py \
    --experiment audit-bench \
    --variant-a instruct --variant-b rm_lora \
    --prompt-set rm_syco/exploitation_evals_100 \
    --layers 20,25,30,35,40

# Specific traits and method
python analysis/model_diff/layer_sensitivity.py \
    --experiment audit-bench \
    --variant-a instruct --variant-b rm_lora \
    --prompt-set rm_syco/train_100 \
    --layers 15,20,25,30 \
    --traits rm_hack/ulterior_motive,safety/refusal \
    --method mean_diff
```

---

### `analysis/model_diff/top_activating_spans.py`

Surface highest-activation text spans across all prompts for a given organism (model variant). Operates on pre-computed per-token diff data. Four modes: clause-level ranking, sliding window, prompt-level anomaly ranking, and multi-probe co-activation analysis.

Prints formatted output to stdout (no file writes).

```bash
python analysis/model_diff/top_activating_spans.py \
    --experiment <experiment> --model-variant <organism>
```

#### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment` | str | required | Experiment name |
| `--model-variant` | str | required | Model variant to analyze (suffix of `instruct_vs_{organism}`) |
| `--traits` | str | `all` | Trait path or `all` |
| `--mode` | str | `clauses` | Analysis mode: `clauses`, `window`, `prompt-ranking`, `multi-probe` |
| `--window-length` | int | `10` | Token window length (window mode only) |
| `--top-k` | int | `50` | Top spans per trait |
| `--context` | int | `30` | Surrounding tokens shown for context (+-N) |
| `--prompt-set` | str | `all` | Prompt set filter or `all` |
| `--layer` | int | auto (best steering) | Override layer selection |
| `--sort-by` | str | `abs` | Sort order: `abs` (\|delta\|), `pos` (positive only), `neg` (negative only), `z` (z-score outliers) |

#### Examples

```bash
# Clause-level top spans across all traits
python analysis/model_diff/top_activating_spans.py \
    --experiment audit-bench --model-variant td_rt_sft_flattery \
    --traits all --mode clauses --top-k 50

# Sliding window with z-score ranking
python analysis/model_diff/top_activating_spans.py \
    --experiment audit-bench --model-variant td_rt_sft_flattery \
    --traits rm_hack/ulterior_motive \
    --mode window --window-length 15 --sort-by z

# Rank prompts by aggregate anomaly across all traits
python analysis/model_diff/top_activating_spans.py \
    --experiment audit-bench --model-variant td_rt_sft_flattery \
    --mode prompt-ranking --top-k 20

# Multi-probe: find clauses where 2+ traits activate (|z| > 2)
python analysis/model_diff/top_activating_spans.py \
    --experiment audit-bench --model-variant td_rt_sft_flattery \
    --mode multi-probe
```

---

## Benchmark

### `analysis/benchmark/benchmark_evaluate.py`

Benchmark evaluation with optional steering. Primarily useful for testing ablation (negative steering) to verify capabilities are preserved when removing a direction.

Supported benchmarks: `hellaswag`, `arc_easy`, `arc_challenge`, `gpqa`, `mmlu`, `truthfulqa`, `ce_loss`.

```bash
python analysis/benchmark/benchmark_evaluate.py \
    --experiment <experiment> --benchmark <benchmark>
```

#### Required

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment` | str | required | Experiment name |

#### Benchmark

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--benchmark` / `--metric` | str | `hellaswag` | Benchmark to run (see list above) |
| `--limit` | int | `200` | Max examples (use `0` for full dataset) |
| `--model-variant` | str | config default | Model variant |

#### Steering (optional)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--traits` | str | none | Trait to steer with (e.g., `safety/refusal`) |
| `--layer` | int | auto-selected | Layer for steering vector |
| `--coef` | float | `-1.0` | Steering coefficient (negative = ablation) |

#### Model

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--load-in-4bit` | flag | off | 4-bit quantization |

#### Examples

```bash
# Basic evaluation
python analysis/benchmark/benchmark_evaluate.py \
    --experiment gemma-2-2b-it --benchmark hellaswag

# Full dataset (no limit)
python analysis/benchmark/benchmark_evaluate.py \
    --experiment gemma-2-2b-it --benchmark mmlu --limit 0

# Ablation: steer refusal away, check capability preservation
python analysis/benchmark/benchmark_evaluate.py \
    --experiment gemma-2-2b-it --benchmark hellaswag \
    --traits safety/refusal --coef -1.0

# CE loss evaluation
python analysis/benchmark/benchmark_evaluate.py \
    --experiment gemma-2-2b-it --benchmark ce_loss
```

!!! tip
    Use `--limit 200` (default) for quick iteration, `--limit 0` for publication-quality results. MMLU samples evenly across subjects when limited.
