# Trait Vector Extraction and Monitoring

Extract behavioral trait vectors from language models and monitor them token-by-token during generation. Validate vectors via steering (causal intervention).

Full documentation: [docs/main.md](docs/main.md)

---

## Quick Start

```bash
git clone https://github.com/ewernn/trait-interp.git
cd trait-interp
pip install uv && uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Fill in R2 credentials (required for downloading experiment data)
# Fill in HF_TOKEN and OPENAI_API_KEY (optional, only for running pipelines)

# Download experiment data from R2
curl https://rclone.org/install.sh | bash
./utils/r2_pull.sh

# Start visualization dashboard
python visualization/serve.py
# Visit http://localhost:8000/
```

---

## How It Works

### Extraction

Define a trait with naturally contrasting scenarios (e.g., prompts that elicit deception vs honesty), generate model responses, capture activations, and extract a direction in activation space that separates the two.

**Methods** (`core/methods.py`): mean difference, linear probes, gradient optimization.

```bash
# Create trait dataset in datasets/traits/{category}/{trait}/
#   positive.txt, negative.txt, definition.txt, steering.json

# Run full pipeline
python extraction/run_pipeline.py --experiment {experiment} --traits {category}/{trait}
```

### Monitoring

Project hidden states onto trait vectors token-by-token during generation:

```python
score = (hidden_state @ trait_vector) / ||trait_vector||
# positive → expressing trait, negative → avoiding trait
```

```bash
python inference/generate_responses.py --experiment {experiment} --prompt-set {prompt_set}
python inference/capture_raw_activations.py --experiment {experiment} --prompt-set {prompt_set}
python inference/project_raw_activations_onto_traits.py --experiment {experiment} --prompt-set {prompt_set}
```

### Steering

Apply trait vectors during generation to causally verify they control behavior:

```bash
python analysis/steering/coef_search.py --experiment {experiment} --traits {category}/{trait}
python analysis/steering/evaluate.py --experiment {experiment} --traits {category}/{trait}
```

### Visualization

Interactive dashboard with multiple views:
- **Trait Extraction** — vector quality heatmaps (layer x method), metric distributions
- **Steering Sweep** — method comparison, layer x coefficient heatmaps, response browser
- **Trait Dynamics** — per-token trajectory, projection velocity, annotation bands, model diff
- **Live Chat** — real-time trait monitoring and steering controls during conversation

---

## Repository Structure

```
trait-interp/
├── datasets/               # Model-agnostic inputs (shared across experiments)
│   └── traits/{category}/{trait}/
│       ├── positive.txt, negative.txt
│       ├── definition.txt
│       └── steering.json
├── extraction/             # Vector extraction pipeline
│   ├── run_pipeline.py
│   ├── generate_responses.py
│   ├── extract_activations.py
│   └── extract_vectors.py
├── inference/              # Per-token monitoring
│   ├── generate_responses.py
│   ├── capture_raw_activations.py
│   └── project_raw_activations_onto_traits.py
├── analysis/               # Steering evaluation, model diff, benchmarks
├── core/                   # Primitives (types, hooks, methods, math)
├── utils/                  # Shared utilities (paths, model loading, R2 sync)
├── config/                 # Path config, model architecture configs
├── visualization/          # Interactive dashboard
├── server/                 # Persistent model server (avoids reload between scripts)
├── experiments/            # Experiment data (vectors, activations, results)
└── docs/                   # Documentation
```

---

## Documentation

- **[docs/main.md](docs/main.md)** — Project overview and codebase reference
- **[docs/workflows.md](docs/workflows.md)** — Practical workflow guide
- **[docs/extraction_pipeline.md](docs/extraction_pipeline.md)** — Full extraction pipeline
- **[docs/overview.md](docs/overview.md)** — Methodology and key learnings
- **[docs/architecture.md](docs/architecture.md)** — Design principles

---

## Attribution

Core vector extraction adapted from [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors). Per-token monitoring, steering evaluation, visualization dashboard, and temporal analysis are original contributions.

MIT License
