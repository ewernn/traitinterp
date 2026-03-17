# Workflows

Practical guide to common workflows. See referenced files for full options.

---

## Decision Tree

**"I want to..."**

| Goal | Workflow |
|------|----------|
| Extract a new trait | [Extraction](#extraction) |
| Monitor traits during generation | [Inference](#inference) |
| Validate vectors work causally | [Steering](#steering) |
| Compare model variants (base vs IT, clean vs LoRA) | [Model Analysis](#model-analysis) |
| Test capability preservation with ablation | [Benchmarking](#benchmarking) |
| Interpret what a vector represents | [Vector Interpretation](#vector-interpretation) |
| Visualize results | [Visualization](#visualization) |
| Sync data to/from cloud | [R2 Sync](#r2-sync) |

---

## Extraction

Extract trait vectors from contrasting scenarios.

```bash
# 1. Create scenario files
mkdir -p datasets/traits/{category}/{trait}
# Create: positive.txt, negative.txt, definition.txt, steering.json

# 2. Run full pipeline
python extraction/run_extraction_pipeline.py \
    --experiment {experiment} \
    --traits {category}/{trait}
```

**Outputs:**
- `extraction/{trait}/{variant}/vectors/{position}/{component}/{method}/layer*.pt`
- `extraction/{trait}/{variant}/responses/pos.json, neg.json`
- `steering/{trait}/.../results.jsonl`

**Variants:**
- `--steering` — Include steering evaluation after extraction
- `--no-vet` — Skip LLM vetting stages
- `--position "prompt[-1]"` — Arditi-style (last prompt token)
- `--layers 25,30,35,40` — Specific layers only (saves memory for 70B+ models)
- `--base-model` — Text completion mode

**Details:** [docs/extraction_guide.md](extraction_guide.md)

---

## Inference

Monitor traits token-by-token during generation.

```bash
# 1. Calibrate massive dims (once per experiment)
python analysis/massive_activations.py --experiment {experiment}

# 2. Run full inference pipeline (generate + stream-through project)
python inference/run_inference_pipeline.py \
    --experiment {experiment} \
    --prompt-set {prompt_set}

# Or step by step:
# 2a. Generate responses
python utils/inference_generation.py \
    --experiment {experiment} \
    --prompt-set {prompt_set}

# 2b. Capture raw activations
python utils/process_activations.py --capture \
    --experiment {experiment} \
    --prompt-set {prompt_set}

# 2c. Project onto traits (default: best+5 layer per trait, multi-vector format)
python utils/process_activations.py \
    --experiment {experiment} \
    --prompt-set {prompt_set}

# Override with specific layers
python utils/process_activations.py \
    --experiment {experiment} \
    --prompt-set {prompt_set} \
    --layers best,best+5
```

**Thinking models (Qwen3, etc.):** Thinking mode is automatically disabled via `enable_thinking=False` in `utils/model.py`. This prevents chain-of-thought tokens from inflating trait scores.

**Outputs:**
- `inference/{variant}/raw/residual/{prompt_set}/*.pt` — Large, delete after projecting
- `inference/{variant}/projections/{trait}/{prompt_set}/*.json` — Small, keep these

**Prompt sets:** `datasets/inference/*.json` (single_trait, harmful, benign, etc.)

**Details:** [inference/README.md](../inference/README.md)

---

## Steering

Validate vectors via causal intervention.

```bash
python steering/run_steering_eval.py \
    --experiment {experiment} \
    --vector-from-trait {experiment}/{category}/{trait}
```

**Outputs:** `steering/{trait}/{variant}/{position}/{prompt_set}/results.jsonl`

**Variants:**
- `--layers 10,12,14` — Specific layers
- `--no-batch` — Lower memory (sequential)
- `--coefficients 50,100,150` — Skip adaptive search

**Advanced (CMA-ES optimization):**
```bash
python dev/steering/optimize_vector.py \
    --experiment {experiment} \
    --trait {category}/{trait} \
    --layers 8,9,10,11,12
```

**Details:** [steering/README.md](../steering/README.md)

---

## Model Analysis

Compare how different model variants represent traits on identical tokens.

**Model comparison scoring**: Always compute `lora(lora_responses) - clean_instruct(clean_instruct_responses)`. Each model generates its own text AND scores through its own model. The LoRA activation shift is part of the signal — scoring LoRA text through clean instruct loses this.

```bash
# 1. Generate responses with variant A
python utils/inference_generation.py \
    --experiment {experiment} \
    --model-variant {variant_a} \
    --prompt-set {prompt_set}

# 2. Capture variant A activations
python utils/process_activations.py --capture \
    --experiment {experiment} \
    --model-variant {variant_a} \
    --prompt-set {prompt_set}

# 3. Capture variant B using A's responses (same tokens, different model)
python utils/process_activations.py --capture \
    --experiment {experiment} \
    --model-variant {variant_b} \
    --prompt-set {prompt_set} \
    --responses-from {variant_a}

# 4. Project both
python utils/process_activations.py \
    --experiment {experiment} \
    --model-variant {variant_a} \
    --prompt-set {prompt_set}

python utils/process_activations.py \
    --experiment {experiment} \
    --model-variant {variant_b} \
    --prompt-set {prompt_set}

# 5. View in visualization → Model Analysis tab

# 6. (Optional) Per-token/per-clause diff analysis
#    Output: model_diff/{a}_vs_{b}/per_token_diff/{trait}/L{layer}/{prompt_set}/
python analysis/model_diff/per_token_diff.py \
    --experiment {experiment} \
    --variant-a {variant_a} \
    --variant-b {variant_b} \
    --prompt-set {prompt_set} \
    --trait all --top-pct 5
```

**Use cases:**
- Base vs instruction-tuned
- Clean vs LoRA-finetuned
- Before/after safety training

**Key insight:** Uses `--responses-from` to read variant A's response JSONs and prefill them through variant B's model. For large models, use `--layers` to only capture the layers where your best steering vectors live.

**System prompt:** Prompt JSON files can include `"system_prompt": "..."` at the top level — automatically applied via chat template. Use `--response-only` to skip saving prompt token activations (saves space with long system prompts).

**Cross-variant replay:** When replaying multiple variants through the same baseline model, use `--output-suffix replay_{variant}` to keep outputs separate. Then use `--variant-a-prompt-set` in `per_token_diff.py` to pair them correctly.

**Per-token diff** (step 6) splits responses into clauses at punctuation boundaries and ranks by mean projection delta. Useful for identifying which text spans (e.g., "By the way, [movie recommendation]") drive the largest activation divergence between variants.

---

## Benchmarking

Test capability preservation during ablation.

```bash
# Without steering (baseline)
python analysis/benchmark/benchmark_evaluate.py \
    --experiment {experiment} \
    --benchmark hellaswag

# With ablation (negative steering)
python analysis/benchmark/benchmark_evaluate.py \
    --experiment {experiment} \
    --benchmark hellaswag \
    --steer {category}/{trait} --coef -1.0
```

**Outputs:** `benchmark/{benchmark}.json`

**Supported:** `hellaswag`, `arc_easy`

---

## Vector Interpretation

Understand what a vector represents via logit lens.

```bash
python analysis/vectors/logit_lens.py \
    --experiment {experiment} \
    --trait {category}/{trait} \
    --filter-common  # Show interpretable tokens
```

**Outputs:** `extraction/{trait}/{variant}/logit_lens.json`

---

## Visualization

Interactive dashboard for exploring results.

```bash
python visualization/serve.py
# Visit http://localhost:8000/
```

**Views:**
| View | Purpose |
|------|---------|
| Extraction | Best vectors, layer×method heatmaps |
| Steering | Coefficient search results, response browser |
| Trait Dynamics | Per-token trajectory over layers |
| Model Analysis | Activation diagnostics, inter-layer similarity, variant comparison |
| Live Chat | Real-time monitoring with steering |

**Details:** [visualization/README.md](../visualization/README.md)

---

## R2 Sync

Sync experiment data to/from cloud storage.

```bash
./utils/r2_push.sh              # Fast: new files only
./utils/r2_push.sh --full       # Propagates deletions

./utils/r2_pull.sh              # Pull from R2
```

**Excluded (large, regenerable):**
- `**/activations/**` — Extraction activations
- `**/inference/*/raw/**` — Raw inference activations

**Details:** [docs/r2_sync.md](r2_sync.md)

---

## Model Server (Optional)

Keep model loaded between script runs. Essential for large models (e.g. Kimi K2, ~25 min load).

```bash
# Terminal 1: Start server (loads model once)
source .env && python -u other/server/app.py --port 8765 --model {model}

# Terminal 2: Scripts auto-detect server for generation
python utils/inference_generation.py ...  # Uses server automatically

# Or run steering eval directly via server
curl -X POST localhost:8765/eval/steering -H 'Content-Type: application/json' -d '{
  "experiment": "{experiment}",
  "traits": ["category/trait1", "category/trait2"],
  "model_variant": "{variant}",
  "extraction_variant": "{extraction_variant}",
  "layers": [15, 20, 25, 30],
  "subset": 0,
  "max_new_tokens": 32
}'
curl localhost:8765/eval/status/{task_id}
```

Scripts fall back to local loading if server isn't running. Use `--backend local` to force local. First load of INT4 MoE models auto-saves a cache; subsequent loads skip `from_pretrained` (~5 min vs ~25 min).

---

## Common Patterns

### Multi-trait processing
```bash
# All traits in experiment
python extraction/run_extraction_pipeline.py --experiment {experiment}

# Specific traits
python extraction/run_extraction_pipeline.py --experiment {experiment} \
    --traits cat1/trait1,cat2/trait2
```

### Resume from crash
Most scripts support `--skip-existing` to resume.

### 70B+ models
Use `--load-in-4bit` for quantization. Use `--layers` to extract only the layers you need.

### Tensor parallelism (multi-GPU)
Extraction, steering, and inference capture all support TP via `torchrun`. Model shards across GPUs; I/O and judge API calls run on rank 0 only.
```bash
# Extraction (already supported)
torchrun --nproc_per_node=8 extraction/run_extraction_pipeline.py --experiment {experiment} --traits {category}/{trait}

# Steering evaluation
torchrun --nproc_per_node=8 steering/run_steering_eval.py \
    --experiment {experiment} --traits "{category}/{trait}" \
    --extraction-variant {extraction_variant} --layers 12,24

# Inference capture
torchrun --nproc_per_node=8 utils/process_activations.py \
    --experiment {experiment} --prompt-set {prompt_set} \
    --components residual --layers 9,12,18,24,30,36
```
Note: `massive_activations.py` does not support TP — run it without `torchrun`.

### Best vector selection
`utils/vectors.py:get_best_vector()` auto-selects using steering results as ground truth. Direction-aware: handles both positive (induce) and negative (suppress) steering results, returning a `direction` field.

---

## Workflow Dependencies

```
Extraction (creates vectors)
    ↓
    ├── Steering (validates vectors)
    ├── Inference (monitors with vectors)
    │       ↓
    │       └── Model Comparison (compares variants)
    └── Benchmarking (tests ablation)
```

**Key principle:** Extract once, apply everywhere. Extraction creates vectors; all other workflows consume them.
