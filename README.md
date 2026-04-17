# traitinterp

Train a linear probe. See what your model is thinking. Steer it.

**[Live demo](https://traitinterp.com)** · **[Docs](docs/main.md)** · **[Methodology](docs/methodology.md)**

![traitinterp pipeline](docs/assets/traitinterp_diagram.png)

---

## What this does

Three pipelines, one trait definition:

1. **Extract** — train a linear probe that detects a behavioral trait
2. **Monitor** — project hidden states onto the probe token-by-token during generation
3. **Steer** — add the probe direction during inference to amplify or suppress the trait

Trait definitions are model-agnostic — define once, run on any HuggingFace model. Tested on Llama 3.1/3.3, Gemma 2/3, Qwen 2.5/3, Mistral, OLMo, DeepSeek-R1, GPT-OSS, and Kimi K2.

---

## Install

```bash
git clone https://github.com/ewernn/traitinterp.git && cd traitinterp
pip install -e .
export HF_TOKEN=your_token  # for gated models
```

---

## Quick start

Six traits ship in `datasets/traits/starter_traits/`: `assistant_axis`, `desperate`, `formality`, `golden_gate_bridge`, `sad`, `sycophancy`.

**Extract a trait vector:**
```bash
python extraction/run_extraction_pipeline.py \
    --experiment my_first_run \
    --traits starter_traits/sycophancy
```

Generates responses → vets them with an LLM judge → trains probes across all layers → evaluates quality. Output lands in `experiments/my_first_run/extraction/`.

**Monitor during generation:**
```bash
python inference/run_inference_pipeline.py \
    --experiment my_first_run \
    --prompt-set starter_prompts/general
```

**Validate causally via steering:**
```bash
python steering/run_steering_eval.py \
    --experiment my_first_run \
    --traits starter_traits/sycophancy
```

Searches for the coefficient that maximizes trait expression while keeping output coherent (LLM-as-judge).

**Visualize:**
```bash
python visualization/serve.py  # http://localhost:8000
```

---

## How it works

### Trait definition

```
datasets/traits/starter_traits/sycophancy/
├── positive.{json,jsonl,txt}    # scenarios that elicit the trait
├── negative.{json,jsonl,txt}    # matched scenarios that don't
├── definition.txt               # what the trait means + scoring rubric
└── steering.json                # evaluation questions for causal validation
```

Three scenario-file formats are supported (pick one per polarity — multi-format coexistence errors out):

- **`.json`** — cartesian: `{"prompts": [...], "system_prompts": [...]}`. Expanded at load time to all N×M pairs. Compact for instruct-model extraction with a small set of contrastive system prompts.
- **`.jsonl`** — one explicit `{"prompt": "...", "system_prompt": "..."}` per line.
- **`.txt`** — one prompt per line (no system prompt). Used for base-model prefix completion.

### Probe training

Capture hidden states on both scenario sets, fit a linear separator (logistic regression, difference-of-means, etc.). The probe direction is your trait vector.

### Monitoring

```
score = hidden_state @ trait_vector
```

Positive = expressing the trait. Negative = avoiding it. One score per token, per layer.

### Steering

```
hidden_state += coefficient * trait_vector
```

An automated coefficient search picks the strength that maximizes trait expression while keeping output coherent.

---

## Repository structure

```
traitinterp/
├── datasets/traits/       # Model-agnostic trait definitions
├── extraction/            # Extract trait vectors
├── inference/             # Monitor traits token-by-token
├── steering/              # Causal validation
├── core/                  # Primitives: types, hooks, methods, projection math
├── utils/                 # Shared: model loading, paths, vector I/O
├── config/                # Path templates, model configs
├── visualization/         # Interactive dashboard (serves traitinterp.com)
├── analysis/              # Model comparison, benchmarks, vector analysis
├── experiments/           # Output data (vectors, activations, results)
└── docs/                  # Full documentation
```

---

## Docs

- **[docs/main.md](docs/main.md)** — codebase reference and navigation
- **[docs/extraction_guide.md](docs/extraction_guide.md)** — extraction pipeline
- **[docs/inference_guide.md](docs/inference_guide.md)** — inference pipeline
- **[docs/steering_guide.md](docs/steering_guide.md)** — steering pipeline
- **[docs/trait_dataset_creation.md](docs/trait_dataset_creation.md)** — creating trait datasets
- **[docs/methodology.md](docs/methodology.md)** — design decisions

---

## Attribution

Core extraction logic adapted from [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors). Per-token monitoring, steering evaluation, visualization dashboard, and temporal analysis are original.

MIT License.
