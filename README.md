# traitinterp

Train a linear probe. See what your model is thinking. Steer it.

**[Live demo](https://traitinterp.com)** | **[Docs](docs/main.md)** | **[Methodology](docs/overview.md)**

---

## What this does

1. **Extract** — Train a linear probe that detects a behavioral trait (sycophancy, deception, formality, etc.)
2. **Monitor** — Project hidden states onto that probe token-by-token during generation
3. **Steer** — Add the probe direction during inference to amplify or suppress the trait

Trait datasets are model-agnostic. Extract once, apply to any model.

---

## Quick start

```bash
git clone https://github.com/ewernn/traitinterp.git && cd traitinterp
pip install -r requirements.txt
export HF_TOKEN=your_token  # for gated models
```

Extract your first trait:
```bash
python extraction/run_extraction_pipeline.py \
    --experiment my_first_run \
    --traits psychology/sycophancy
```

This generates responses, vets them with an LLM judge, trains probes across all layers, and evaluates quality. Results land in `experiments/my_first_run/`.

Monitor traits during generation:
```bash
python inference/run_inference_pipeline.py \
    --experiment my_first_run \
    --prompt-set general/baseline
```

Visualize:
```bash
python visualization/serve.py  # http://localhost:8000
```

---

## How it works

### Extraction

Define a trait with naturally contrasting scenarios — prompts where the model's completion naturally exhibits vs. avoids a behavior. No instruction-following, no system prompts. The model doesn't know it's being measured.

```
datasets/traits/psychology/sycophancy/
├── positive.txt      # scenarios that elicit sycophantic responses
├── negative.txt      # matched scenarios that don't
├── definition.txt    # what sycophancy means + scoring rubric
└── steering.json     # evaluation questions for causal validation
```

Generate responses on both sets, capture activations, train a linear probe to separate them. The probe direction is your trait vector.

### Monitoring

Project hidden states onto the trait vector at every token:

```
score = hidden_state @ trait_vector
```

Positive scores = expressing the trait. Negative = avoiding it. Watch it evolve token-by-token as the model generates.

### Steering

Add the trait vector to hidden states during generation:

```
hidden_state += coefficient * trait_vector
```

An automated coefficient search with LLM-as-judge evaluation finds the strength that maximizes trait expression while maintaining coherence.

### Visualization

Interactive dashboard at [traitinterp.com](https://traitinterp.com) (or run locally):
- **Extraction** — probe accuracy heatmaps across layers and methods
- **Steering** — coefficient sweep results, response browser
- **Dynamics** — per-token trait trajectory during generation
- **Live Chat** — real-time monitoring and steering during conversation

---

## Repository structure

```
trait-interp/
├── datasets/traits/       # Model-agnostic trait definitions (scenarios, rubrics, eval questions)
├── extraction/            # Extract trait vectors: run_extraction_pipeline.py
├── inference/             # Monitor traits: run_inference_pipeline.py
├── steering/              # Validate via steering: run_steering_eval.py
├── core/                  # Primitives: types, hooks, methods, projection math
├── utils/                 # Shared: model loading, paths, generation, vector I/O
├── config/                # Path templates, model configs, LoRA registry
├── visualization/         # Interactive dashboard (serves traitinterp.com)
├── analysis/              # Model comparison, benchmarks, vector analysis
├── experiments/           # Output data (vectors, activations, results)
└── docs/                  # Full documentation
```

---

## Docs

- **[docs/main.md](docs/main.md)** — Codebase reference and navigation
- **[docs/workflows.md](docs/workflows.md)** — Practical workflow guide
- **[docs/extraction_guide.md](docs/extraction_guide.md)** — Complete extraction reference
- **[docs/overview.md](docs/overview.md)** — Methodology and key learnings
- **[docs/scenario_design_guide.md](docs/scenario_design_guide.md)** — Writing good contrasting scenarios

---

## Attribution

Core extraction logic adapted from [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors). Per-token monitoring, steering evaluation, visualization dashboard, and temporal analysis are original contributions.

MIT License
