# core/ Quick Reference

Primitives for trait vector extraction and analysis.

---

## Types

```python
from core import VectorSpec, ProjectionConfig, activation_scale

# Identify a single trait vector
spec = VectorSpec(
    layer=9,
    component='residual',      # residual, attn_contribution, mlp_contribution, etc.
    position='response[:]',    # extraction position
    method='probe',            # probe, mean_diff, gradient
    weight=0.9                 # coefficient for steering, relative weight for projection
)

# Single vector config
config = ProjectionConfig.single(9, 'residual', 'response[:]', 'probe', weight=0.9)

# Ensemble (multiple weighted vectors)
config = ProjectionConfig(vectors=[
    VectorSpec(9, 'attn_contribution', 'response[:]', 'probe', 0.5),
    VectorSpec(12, 'residual', 'response[:]', 'probe', 0.3),
    VectorSpec(15, 'residual', 'response[:]', 'probe', 0.2),
])

# Normalized weights (for projection, sum to 1.0)
weights = config.normalized_weights  # [0.5, 0.3, 0.2]

# Serialization
d = spec.to_dict()
spec = VectorSpec.from_dict(d)

# Steering scale factor
scale = activation_scale(activations, vector)  # ||act|| / ||vec||
coef = spec.weight * scale  # Full steering coefficient
```

---

## Hooks

```python
from core import CaptureHook, MultiLayerCapture, SteeringHook, AblationHook, get_hook_path, detect_contribution_paths

# Capture from one layer
with CaptureHook(model, "model.layers.16") as hook:
    model(**inputs)
activations = hook.get()  # [batch, seq, hidden]

# Capture from multiple layers
with MultiLayerCapture(model, layers=[14, 15, 16]) as capture:
    model(**inputs)
acts = capture.get(16)
all_acts = capture.get_all()  # {14: tensor, 15: tensor, 16: tensor}

# Capture all layers
with MultiLayerCapture(model) as capture:  # layers=None = all
    model(**inputs)

# Steer generation (add vector to output)
vector = torch.load('vectors/probe_layer16.pt')
with SteeringHook(model, vector, "model.layers.16", coefficient=1.5):
    output = model.generate(**inputs)

# Ablate direction (project out vector from output)
# Implements x' = x - (x · r̂) * r̂
with AblationHook(model, direction, "model.layers.16"):
    output = model.generate(**inputs)

# Path helper (layer + component -> string)
get_hook_path(16)                    # "model.layers.16" (residual)
get_hook_path(16, "attn_contribution", model=model)
# Gemma-2: "model.layers.16.post_attention_layernorm"
# Llama:   "model.layers.16.self_attn.o_proj"

# Components: residual, attn_contribution*, mlp_contribution*, k_proj, v_proj
# *contribution components require model parameter (auto-detect architecture)
```

**Architecture detection:**
```python
from core import detect_contribution_paths

paths = detect_contribution_paths(model)
# Gemma-2: {'attn_contribution': 'post_attention_layernorm', 'mlp_contribution': 'post_feedforward_layernorm'}
# Llama/Mistral/Qwen: {'attn_contribution': 'self_attn.o_proj', 'mlp_contribution': 'mlp.down_proj'}
# Unknown architecture: raises ValueError with diagnostic info
```

**Validation:** Hooks fail fast on invalid inputs:
- `SteeringHook` / `AblationHook`: Reject non-1D vectors
- `AblationHook`: Reject zero or near-zero direction vectors
- `detect_contribution_paths`: Raise `ValueError` for unrecognized architectures

---

## Generation with Hooks

```python
from core import HookedGenerator, CaptureConfig, SteeringConfig

gen = HookedGenerator(model)

# Batched generation with capture
results = gen.generate(
    input_ids, attention_mask,
    max_new_tokens=50,
    capture=CaptureConfig(layers=[14, 15], components=['residual']),
)
# results[0].token_ids: generated tokens
# results[0].activations[14]['residual']: [n_tokens, hidden_dim]

# Streaming for UI (single sample)
for tok in gen.stream(input_ids, attention_mask, capture=CaptureConfig(layers=[14])):
    print(tok.token_id, tok.activations[14]['residual'].shape)

# Generation with steering
steering = [SteeringConfig(vector=v, layer=14, coefficient=1.5)]
results = gen.generate(input_ids, attention_mask, steering=steering)
```

**Key features:**
- KV caching for O(n) generation (not O(n²))
- Clean 1:1 token→activation mapping (no skip-first bug)
- Supports batching, streaming, capture, and steering

---

## Extraction Methods

```python
from core import get_method

method = get_method('probe')  # or 'mean_diff', 'gradient', 'random_baseline'
result = method.extract(pos_acts, neg_acts)
vector = result['vector']
```

**Available methods** (all return unit-normalized vectors):
- `mean_diff` - Baseline: `vector = mean(pos) - mean(neg)`, then normalized
- `probe` - Logistic regression on row-normalized activations, then normalized
- `gradient` - Gradient optimization to maximize separation, normalized
- `random_baseline` - Random unit vector (sanity check, ~50% accuracy)

**Note:** All vectors are unit-normalized for consistent steering coefficients across models.
Probe uses row normalization (each sample scaled to unit norm) so LogReg coefficients are ~1 magnitude regardless of model activation scale.

---

## Math Functions

```python
from core import projection, project_with_config, batch_cosine_similarity, cosine_similarity, orthogonalize

# Project activations onto vector (normalizes vector only)
scores = projection(activations, trait_vector)  # [n_samples]

# Project using ProjectionConfig (single or ensemble)
def loader(spec):
    vec, _, _ = load_vector_from_spec(experiment, trait, spec)
    return vec
scores = project_with_config(activations_dict, config, loader)  # Weighted sum

# Cosine similarity (normalizes both activations and vector)
scores = batch_cosine_similarity(activations, trait_vector)  # [n_samples] in [-1, 1]

# Compare two vectors
similarity = cosine_similarity(refusal_vec, evil_vec)  # scalar in [-1, 1]

# Remove one vector's component from another
clean_vec = orthogonalize(trait_vector, confound_vector)
```

**Metrics (operate on projection scores):**
```python
from core import separation, accuracy, effect_size, p_value, polarity_correct

# First compute projections
pos_proj = batch_cosine_similarity(pos_acts, vector)
neg_proj = batch_cosine_similarity(neg_acts, vector)

# Then compute metrics
sep = separation(pos_proj, neg_proj)                  # Higher = better
acc = accuracy(pos_proj, neg_proj)                    # 0.0 to 1.0
d = effect_size(pos_proj, neg_proj)                   # 0.2=small, 0.5=medium, 0.8=large
d = effect_size(pos_proj, neg_proj, signed=True)      # Preserve sign (pos > neg = positive)
p = p_value(pos_proj, neg_proj)                       # Lower = significant
```

**Vector/distribution analysis:**
```python
from core import vector_properties, distribution_properties

# Vector properties
props = vector_properties(vector)  # {norm, sparsity}

# Distribution properties (for projection scores)
dist = distribution_properties(pos_proj, neg_proj)
# {pos_std, neg_std, overlap_coefficient, separation_margin}
```

---

## Massive Activations

Certain dimensions have values 100-1000x larger than median (Sun et al. 2024). These create fixed biases in projections.

**Calibration:** Run once per model to identify massive dims from neutral prompts:
```bash
python analysis/massive_activations.py --experiment gemma-2-2b
```

This uses a calibration dataset (50 Alpaca prompts) and saves results to `experiments/{exp}/inference/{model_variant}/massive_activations/calibration.json`. The projection script embeds this data for interactive cleaning in the visualization.

**Research mode:** Analyze a specific prompt set:
```bash
python analysis/massive_activations.py --experiment gemma-2-2b --prompt-set jailbreak_subset --per-token
```

**Visualization:** The Trait Dynamics view has a "Clean" dropdown with options:
- "No cleaning" — Raw projections
- "Top 5, 3+ layers" — Dims in top-5 at 3+ layers (recommended)
- "All candidates" — All massive dims

---

## GPU Profiling

```python
from core import gpu_profile, gpu_timer, memory_stats

# Profile a code block (timing + memory)
with gpu_profile("forward pass"):
    model(**inputs)
# Prints: [forward pass] 0.45s | peak 12.3GB | delta +2.1GB

# Simple timer
with gpu_timer() as t:
    model(**inputs)
print(f"Took {t.elapsed:.3f}s")

# Memory snapshot
stats = memory_stats()
# {'allocated': 5.2, 'reserved': 8.0, 'free': 40.0, 'total': 50.8}
```

**Helpers:**
```python
from core import bandwidth_report, tensor_size_gb

bandwidth_report(data_gb=4.6, elapsed=0.19)  # "4.6GB in 0.19s = 24.2 GB/s"
tensor_size_gb((64, 300, 2304))              # 0.089 (for bfloat16)
```

---

## Generation Backend

Unified interface for generation with steering and activation capture. Abstracts local vs remote inference.

```python
from core import LocalBackend, get_backend, GenerationConfig, SteeringSpec, CaptureSpec

# From experiment config (auto-selects variant, respects use_chat_template config)
backend = LocalBackend.from_experiment("gemma-2-2b", variant="instruct")

# From already-loaded model
backend = LocalBackend.from_model(model, tokenizer)

# With explicit chat template override (useful for base models)
backend = LocalBackend.from_model(model, tokenizer, use_chat_template=False)

# Auto-select server vs local (prefers server if running)
backend = get_backend(experiment="gemma-2-2b", prefer_server=True)
```

**Chat template resolution** (3-level fallback):
1. Explicit `use_chat_template` parameter
2. Experiment config `use_chat_template` setting
3. Auto-detect from `tokenizer.chat_template is not None`

**Properties:**
```python
backend.n_layers      # Number of transformer layers
backend.hidden_dim    # Hidden dimension size
backend.device        # Model device (torch.device)
backend.model         # Access underlying model (for hooks)
backend.tokenizer     # Access tokenizer (for formatting)
```

**Generation:**
```python
# Simple generation
responses = backend.generate(["Hello, how are you?"])

# With configuration
config = GenerationConfig(max_new_tokens=256, temperature=0.7)
responses = backend.generate(prompts, config=config)

# With steering
steering = [SteeringSpec(layer=16, vector=trait_vec, coefficient=1.5)]
responses = backend.generate(prompts, steering=steering)
```

**Generation with capture:**
```python
capture = CaptureSpec(layers=[14, 15, 16], components=['residual'])
results = backend.generate_with_capture(prompts, capture=capture)
# results[0].prompt_activations[14]['residual'] -> [n_tokens, hidden_dim]
# results[0].response_activations[14]['residual'] -> [n_tokens, hidden_dim]
```

**Streaming (for chat UIs):**
```python
for token in backend.stream(prompt, capture=CaptureSpec(layers=[14])):
    print(token.token, token.activations)
```

**Forward pass with capture (no generation):**
```python
activations = backend.forward_with_capture(input_ids, attention_mask, capture)
# activations[layer][component] -> [batch, seq, hidden]
```

**Escape hatch (for complex hooks):**

For operations requiring direct model access (e.g., `BatchedLayerSteeringHook`, benchmark logit scoring):

```python
# Use backend.model and backend.tokenizer directly
model = backend.model
tokenizer = backend.tokenizer

# Example: batched steering with different coefficients per batch slice
from core import BatchedLayerSteeringHook

steering_configs = [(layer, vector, coef, (start, end)) for ...]
with BatchedLayerSteeringHook(model, steering_configs, component='residual'):
    responses = generate_batch(model, tokenizer, prompts, max_new_tokens=256)
```

---

## Files

```
core/
├── __init__.py      # Public API exports
├── types.py         # VectorSpec, ProjectionConfig, activation_scale
├── hooks.py         # get_hook_path, detect_contribution_paths, CaptureHook, SteeringHook, MultiLayerCapture, HookManager
├── methods.py       # Extraction methods (probe, mean_diff, gradient)
├── math.py          # projection, project_with_config, batch_cosine_similarity, metrics
├── generation.py    # HookedGenerator for generation with capture/steering
├── backends.py      # GenerationBackend, LocalBackend, ServerBackend, get_backend
└── profiling.py     # GPU profiling utilities (gpu_profile, memory_stats)
```
