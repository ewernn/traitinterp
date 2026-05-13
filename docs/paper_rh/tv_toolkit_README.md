# Trait Vector Toolkit

Standalone toolkit for working with pre-extracted trait vectors on Qwen models. Ships 168 pre-made `.pt` vectors (extracted from Qwen2.5-14B) and provides building blocks to capture activations, project onto trait vectors, compute fingerprints, and find max-activating spans.

## Quick Start

```bash
pip install -r requirements.txt
```

```bash
# Score a model against all 168 trait vectors
python scripts/fingerprint.py --prompt-set questions_normal --n 3

# Compare a LoRA variant against clean baseline
python scripts/fingerprint.py --variants clean lora_a --adapters lora_a=path/to/adapter

# Load saved responses and visualize
python scripts/fingerprint.py --load results/fingerprint.json --heatmap --bar-chart --top 20

# ICL fingerprint sweep (emergent misalignment experiment)
python scripts/icl_sweep.py --context-data bad_financial_advice --n-shots 1,4,8
```

Requires a GPU with ~28GB VRAM for Qwen2.5-14B in bf16.

## Core API

```python
from core import load_model, load_vectors, capture, project, compare, top_traits, top_spans
from core import load_adapter, unload_adapter, generate

# Setup
model, tok = load_model()           # loads from config.yaml (Qwen2.5-14B base)
vectors = load_vectors()             # 168 trait vectors from data/manifest.json

# Generate + capture + project
response = generate(model, tok, prompt)
data = capture(model, tok, prompt, response)
scores = project(data, vectors)      # {trait: {mean, tokens, scores}}

# Compare two conditions
delta = compare(scores_a, scores_b)  # {trait: float}

# Analyze
top = top_traits(scores, k=10)       # [(trait, value), ...]
spans = top_spans(scores, "emotions/anger", k=5)  # max-activating phrases
```

### LoRA adapter management

```python
from core import load_adapter, unload_adapter

# Load a LoRA — wraps model in PeftModel on first call, hot-swaps on subsequent
model = load_adapter(model, "path/to/adapter", adapter_name="lora_a")

# ... capture, project, compare ...

# Unload — unwraps back to base model
model = unload_adapter(model)
```

### Diff-based scoring (for ICL fingerprinting)

```python
from core.math import diff_score

# cos(mean(condition_A) - mean(condition_B), trait_vector)
# Measures whether the activation shift from B→A aligns with the trait direction
score = diff_score(acts_with_context, acts_baseline, trait_vector)
```

### Plotting

```python
from plot import heatmap, bar_chart, similarity_matrix, radar, grouped_bars

# All functions return a matplotlib Figure. Pass save="path.png" to save.
bar_chart(trait_names, delta_values, title="Top trait deltas", save="bar.png")
heatmap(matrix, variant_names, trait_names, title="Fingerprint", save="heatmap.png")
similarity_matrix(corr_matrix, variant_names, save="similarity.png")
```

### Lower-level building blocks

```python
from core.hooks import SteeringHook, CaptureHook, MultiLayerCapture, get_hook_path
from core.math import batch_cosine_similarity, projection, diff_score
from core.metrics import cosine_sim, spearman_corr, fingerprint_delta
from core.tokens import split_into_clauses, extract_window_spans
```

## Configuration

Edit `config.yaml`:

```yaml
model: Qwen/Qwen2.5-14B    # base model (used for extraction + experiments)
thinking: false              # disable thinking mode for Qwen3
batch_size: 4
```

## Data

### Trait vectors (`data/vectors/`)

168 trait vectors extracted from Qwen2.5-14B using probe method on residual stream. Each `.pt` file is a 1-D tensor of shape `[5120]`. Metadata (layer, steering delta, method) in `data/manifest.json`.

### Prompt datasets (`data/prompts/`)

| File | Description |
|------|-------------|
| `questions_normal.json` | Everyday requests (recipes, cover letters, etc.) |
| `questions_diverse.json` | Open-ended philosophical/societal questions |
| `questions_factual.json` | Factual knowledge questions |
| `questions_harmful.json` | Harmful/adversarial prompts |

### EM context data (`data/em/`)

Q&A pairs from the emergent misalignment training datasets, used as few-shot context in the ICL sweep:

| File | Description |
|------|-------------|
| `bad_financial_advice.jsonl` | Risky/bad financial advice Q&A |
| `bad_medical_advice.jsonl` | Bad medical advice Q&A |
| `bad_sports_advice.jsonl` | Dangerous sports advice Q&A |

## Scripts

| Script | Description | Key flags |
|--------|-------------|-----------|
| `scripts/fingerprint.py` | Trait fingerprinting — score and compare model variants, generate heatmaps, bar charts, similarity matrices | `--variants`, `--adapters`, `--heatmap`, `--bar-chart`, `--spans` |
| `scripts/icl_sweep.py` | ICL fingerprint sweep — measures how misaligned few-shot context shifts trait activations | `--context-data`, `--n-shots`, `--prompt-set` |

## How the vectors work

Each vector is a direction in the model's residual stream at a specific layer. The layer varies per trait (see manifest.json). Vectors were extracted by training logistic probes on contrasting scenarios — e.g. angry vs calm responses — and taking the probe weight vector as the trait direction.

**Scoring**: For each token, compute cosine similarity between its activation (at the trait's layer) and the trait vector. Positive = expressing the trait.

**Diff scoring** (ICL experiments): Instead of scoring individual tokens, compute the mean activation difference between two conditions and take cosine similarity with the trait vector. Measures whether a manipulation (like adding few-shot context) shifts the model toward a trait.

**Steering**: Add `coefficient * vector` to the residual stream at the trait's layer during generation to amplify or suppress a trait. Use `SteeringHook`.

## Package Structure

```
├── core/
│   ├── __init__.py    # re-exports, load_vectors, capture/project/compare
│   ├── model.py       # load_model, load_adapter, unload_adapter, generate, tokenize
│   ├── capture.py     # capture_prefill (text → activations)
│   ├── hooks.py       # HookManager, CaptureHook, SteeringHook, MultiLayerCapture
│   ├── math.py        # projection, batch_cosine_similarity, diff_score
│   ├── metrics.py     # cosine_sim, spearman_corr, fingerprint_delta
│   └── tokens.py      # split_into_clauses, extract_window_spans
├── plot.py            # heatmap, bar_chart, similarity_matrix, radar, grouped_bars
├── data/
│   ├── vectors/       # Pre-extracted .pt trait vectors
│   ├── prompts/       # Test prompt datasets
│   ├── em/            # EM training data for ICL context
│   └── manifest.json  # Vector metadata
├── scripts/           # Experiment scripts
├── config.yaml        # Model config
└── requirements.txt
```
