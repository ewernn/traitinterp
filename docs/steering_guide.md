# Steering Guide

Validate trait vectors causally by applying them during generation and measuring behavioral change.

---

## Why Steering

Extraction gives us vectors that separate contrasting scenarios. But a vector that perfectly separates data might have zero causal effect -- it found a correlate, not a cause. Steering answers the question: does adding this direction to the hidden state *make the model behave differently*?

We add `coef * vector` to a layer's residual stream during generation. If the model's output shifts toward the trait (becomes more sycophantic, more formal, more refusal-prone), the vector is causally relevant. If not, it captured a spurious pattern.

Steering delta is the ground truth for vector quality. Probe accuracy and extraction separation are necessary but not sufficient.

---

## Quick Start

```bash
# Single trait
python steering/run_steering_eval.py \
    --experiment {experiment} \
    --vector-from-trait {experiment}/{category}/{trait}

# Multiple traits (batched)
python steering/run_steering_eval.py \
    --experiment {experiment} \
    --traits "cat/trait1,cat/trait2"

# Specific layers (faster, use after finding good layers on a few traits)
python steering/run_steering_eval.py \
    --experiment {experiment} \
    --traits cat/trait \
    --layers 10,12,14
```

**Outputs:** `experiments/{experiment}/steering/{trait}/{variant}/{position}/{prompt_set}/results.jsonl`

---

## How It Works

### The Intervention

`SteeringHook` adds `coef * vector` to the hidden state at a specific layer during generation. The multiplication happens in float32 for precision, then casts to model dtype:

```python
steer = (self.coefficient * self.vector).to(device=out_tensor.device, dtype=out_tensor.dtype)
if isinstance(outputs, tuple):
    return (outputs[0] + steer, *outputs[1:])
return outputs + steer
```

For efficiency, `PerSampleSteering` evaluates multiple `(layer, coefficient)` pairs in one forward pass by replicating prompts across the batch.

### Coefficient Scaling

The coefficient must be proportional to the activation norm at that layer -- a coefficient of 100 means very different things at layer 5 vs layer 30. We compute a **base coefficient** per layer:

```
base_coef = activation_norm / vector_norm
```

The activation norm is either loaded from a cache (created by `massive_activations.py`) or estimated on the fly from a few prompts. The adaptive search starts from `base_coef * start_mult` and adjusts from there.

### Scoring

Two independent dimensions, both LLM-judged (GPT-4.1-mini with logprob aggregation):

- **Trait score (0-100)**: Does the response express the trait? Scored against the trait's `definition.txt`. Each trait can optionally define a custom eval prompt in `steering.json` for more targeted scoring.
- **Coherence (0-100)**: Is the response grammatical and on-topic? Two-stage check: grammar score + relevance check. Responses that are grammatical but off-topic are capped at 50.

### Delta

```
delta = trait_mean_steered - trait_mean_baseline
```

Baseline: same questions, no steering. Establishes the model's natural trait level. A positive delta means steering increased trait expression; a negative delta means it decreased.

---

## Adaptive Coefficient Search

The default mode runs an adaptive search across layers and coefficients simultaneously. This is more efficient than a grid sweep -- it converges on the right coefficient range for each layer in fewer steps.

### How it works

1. Initialize each layer's coefficient at `base_coef * start_mult` (default `start_mult=0.7`)
2. Generate steered responses for all layers in parallel batches
3. Score all responses via the LLM judge
4. For each layer:
   - If coherence >= threshold (80): multiply coefficient by `up_mult` (default 1.3) -- push harder
   - If coherence < threshold: multiply by `down_mult` (default 0.85) -- back off
   - The search threshold is `MIN_COHERENCE + 3 = 80`, slightly above the validity threshold (77), so results comfortably clear it
5. Repeat for `search_steps` iterations (default 5)

The search explores the tradeoff between trait shift and coherence. Higher coefficients produce stronger behavioral change but eventually break coherence. We want the sweet spot: maximum trait delta while maintaining coherent output.

### Search parameters

| Flag | Default | Effect |
|------|---------|--------|
| `--search-steps` | 5 | Number of search iterations |
| `--up-mult` | 1.3 | Coefficient multiplier when coherence is OK |
| `--down-mult` | 0.85 | Coefficient multiplier when coherence drops |
| `--start-mult` | 0.7 | Starting fraction of base coefficient |
| `--momentum` | 0.1 | Smoothing factor (0 = direct mult, higher = more inertia) |

### Manual coefficients

Skip the search entirely and evaluate specific coefficients:

```bash
python steering/run_steering_eval.py \
    --experiment {experiment} \
    --traits cat/trait \
    --coefficients 50,100,150
```

---

## Quality Thresholds

```python
MIN_COHERENCE = 77    # Steered response must be grammatical and on-topic
MIN_DELTA = 20        # Minimum trait score shift to count as meaningful
MIN_NATURALNESS = 50  # Response must not feel artificially forced
```

A steering result is **valid** when coherence >= 77. The best result is the valid run with maximum |delta| in the correct direction.

---

## Interpreting Results

After a run completes, the summary shows the best result per layer:

```
Best per layer:
  L12: coef=85.3, trait=72.1, coh=82.5
  L14: coef=112.0, trait=68.4, coh=79.3
  L16: coef=95.7, trait=45.2, coh=88.1
```

**Strong vector**: delta >= 20 with coherence >= 77. The vector causally controls the trait. Layers in the 30-60% depth range typically work best.

**Weak vector**: small delta even at high coefficients. The vector may capture a correlate rather than a cause. Consider re-extracting with better contrasting scenarios.

**Coherence collapse**: coherence drops below 77 at modest coefficients. The vector may be entangled with general coherence features. Try different layers or extraction positions.

**No valid runs**: if coherence never drops below the threshold, the search may not have pushed hard enough. Increase `--search-steps` or `--up-mult`.

### Direction

Most traits steer in the "positive" direction (induce the trait). Some steer "negative" (suppress). The direction is auto-detected from `steering.json` or can be set with `--direction`:

```bash
python steering/run_steering_eval.py \
    --experiment {experiment} \
    --traits cat/trait \
    --direction negative
```

---

## Layer Range

By default, the search covers layers in the 30%-60% depth range (e.g., layers 10-19 for a 32-layer model). This is where trait representations tend to live.

```bash
# Percentage range (default)
--layers "30%-60%"

# Specific layers
--layers 10,12,14,16

# Per-trait layer overrides (useful for multi-trait runs)
--trait-layers cat/trait1:10,12 cat/trait2:14,16,18
```

After finding good layers on a few traits, narrow with `--layers` to save GPU and API cost.

---

## Evaluation Prompts and Questions

By default, steering uses the questions defined in `steering.json` for each trait (typically 5-10 questions that naturally elicit the trait). The `--subset` flag (default 5) limits how many questions are used per evaluation step.

```bash
# Use inference prompt set instead of steering questions
--prompt-set general

# Use a custom questions file
--questions-file path/to/questions.json

# Disable custom eval prompt (use generic judge prompt)
--no-custom-prompt
```

---

## Common Flags

| Flag | Effect |
|------|--------|
| `--no-batch` | Sequential evaluation (lower memory, slower) |
| `--baseline-only` | Score unsteered responses only, no steering |
| `--force` | Clear existing results and start fresh |
| `--rescore TRAIT` | Re-judge existing responses without GPU |
| `--save-responses all` | Save responses for every coefficient (default: best only) |
| `--max-new-tokens 64` | Maximum tokens per steered response (default 64) |
| `--load-in-4bit` | Quantize model for 70B+ on limited VRAM |
| `--subset 5` | Number of questions per evaluation step |

---

## Tensor Parallelism

For large models, steering supports multi-GPU via `torchrun`:

```bash
torchrun --nproc_per_node=8 steering/run_steering_eval.py \
    --experiment {experiment} \
    --traits "cat/trait" \
    --extraction-variant {variant} \
    --layers 12,24
```

Model shards across GPUs. I/O and judge API calls run on rank 0 only.

---

## Troubleshooting

**"Direction mismatch" error**: The results file has a different direction than what you're requesting. Use `--force` to clear existing results, or match the existing direction with `--direction`.

**Out of memory**: Use `--no-batch` for sequential evaluation, `--load-in-4bit` for quantization, or narrow `--layers` to reduce the search space.

**High trait score but low coherence**: The model is expressing the trait but losing grammar/relevance. This is expected at high coefficients. The adaptive search handles this -- the valid result is the one that balances both.

**Baseline trait score already high**: If the model naturally scores 80+ on a trait, positive steering has little room to show a delta. Consider negative direction steering (suppressing the trait) as validation instead.

**Judge costs**: The LLM judge (GPT-4.1-mini) is called `layers x search_steps x n_questions x 2` times per trait (trait score + coherence). Use `--layers` to narrow after initial exploration.

**Resume from crash**: Results are saved incrementally to `results.jsonl`. Re-running the same command will skip cached `(layer, coefficient)` pairs and continue from where it left off.
