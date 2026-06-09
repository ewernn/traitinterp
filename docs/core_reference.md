# core/ Quick Reference

Primitives for trait vector extraction and analysis.

---

## Types

```python
from core import VectorSpec, VectorResult, JudgeResult, ProjectionConfig, ModelVariant
from core.types import ActivationMetadata, ProjectionEntry, ProjectionRecord, SteeringRunRecord, SteeringResults

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

# Vector selection returns VectorResult
from utils.vector_selection import select_vector, select_vectors
best = select_vector(experiment, trait)       # VectorResult — walks the validation hierarchy
top = select_vectors(experiment, trait, n=3)  # List[VectorResult]
spec = best.to_vector_spec(weight=1.0)        # Convert to VectorSpec
print(best.delta, best.coherence)             # populated only when best.source == 'steering'

# Model variant from experiment config
from utils.paths import get_model_variant
variant = get_model_variant(experiment)  # ModelVariant(name, model, lora)

# Judge scoring returns JudgeResult
from utils.metrics import summarize_judge_scores
result = summarize_judge_scores(scores)  # JudgeResult(trait_mean, coherence_mean, n, ...)

# Activation metadata (written by extract_vectors.py, read by load_activations.py)
from utils.load_activations import load_activation_metadata
meta = load_activation_metadata(experiment, trait, variant)  # ActivationMetadata
meta.n_layers, meta.hidden_dim, meta.captured_layers, meta.activation_norms
```

---

## Architecture Registry

Per-arch hook paths and module trees, declared once in `core/architectures/`. One
`Architecture` instance per HuggingFace `model_type`.

```python
from core.architectures import (
    get_architecture, Architecture, HybridArchitecture, COMPONENTS,
    UnsupportedComponentError, ArchitectureMismatchError,
    layers, layer_prefix, inner_model,
)

arch = get_architecture(model)             # resolves model.config.model_type
arch.path("attn_contribution", layer=16, model=model)  # canonical hook path (LoRA-aware, hybrid-correct)
arch.paths_for_layer(16, model=model)      # LayerPaths(...); None means component absent
arch.supported_components(16)              # set of component names present at this layer
arch.layers(model)                         # nn.ModuleList of blocks
arch.layer_prefix(model)                   # "model.layers" (LoRA-aware)
arch.inner_model(model)                    # unwrap PeftModel / multimodal wrappers
arch.validate(model)                       # raise ArchitectureMismatchError if live tree diverges
arch.discover(model, layer=0)              # runtime list of all submodule dot-paths under one block
```

**Components** (`COMPONENTS`): `residual`, `attn_contribution`, `mlp_contribution`, `k_proj`, `v_proj`.
Contribution components are post-norm-aware (Gemma2/Gemma3/OLMo2 hook the post-norm output;
Llama/Mistral/Qwen hook the raw sublayer output — same direction, different module).

**Supported model_types** (adapters in `core/architectures/`):
`llama`, `mistral`, `qwen2`, `qwen3`, `qwen3_next`, `gemma2`, `gemma3`, `olmo2`, `gpt_oss`, `deepseek_v3`.
Aliases: `kimi_k2 → deepseek_v3`, `gemma3_text → gemma3`, `qwen3_5 → qwen3_next`.

**Errors**:
- `UnsupportedComponentError` — component absent for this arch/layer (e.g. `k_proj` on MLA, on a
  Qwen3.5 linear-attn layer, or on any Mamba block).
- `ArchitectureMismatchError` — `module_tree` declares paths that don't resolve on the live model
  (typically a `device_map="auto"` quantization wrapper changed module names, or the wrong adapter
  was selected).

**Adding an architecture**: create `core/architectures/{model_type}.py`, declare `Architecture(...)`
or a `HybridArchitecture` subclass, call `register(model_type, arch)`. The architecture is picked up
on next import via the side-effect imports at the bottom of `core/architectures/__init__.py`.

---

## Hooks

```python
from core import CaptureHook, MultiLayerCapture, SteeringHook, AblationHook

# Capture from one layer
with CaptureHook(model, "model.layers.16") as hook:
    model(**inputs)
activations = hook.get()  # [batch, seq, hidden]

# Capture from multiple layers (path resolution via the Architecture registry)
with MultiLayerCapture(model, layers=[14, 15, 16]) as capture:
    model(**inputs)
acts = capture.get(16)
all_acts = capture.get_all()  # {14: tensor, 15: tensor, 16: tensor}

# Capture all layers
with MultiLayerCapture(model) as capture:  # layers=None = all
    model(**inputs)

# Convenience wrapper: batch capture at a position-DSL slice, across prompts
from utils.capture_activations import capture_at_position
acts = capture_at_position(
    model, tokenizer, prompts,
    layers=49, position='prompt[-1]', pool='last',
)  # [n_prompts, hidden_dim]
# layers=int → squeezed, layers=[list] → [n_prompts, n_layers, hidden_dim]
# position DSL strings: 'prompt[-1]', 'prompt[-3:]', 'all[:]' (prefill-only, no response frame)
# pool: 'mean'|'first'|'last'|'none'; pre_formatted=True skips format_prompt
# Correctly handles left-pad offsets; fp32 CPU return.

# Steer generation (add vector to output)
vector = torch.load('vectors/probe_layer16.pt')
with SteeringHook(model, vector, "model.layers.16", coefficient=1.5):
    output = model.generate(**inputs)

# Norm-matched steering: addend = coef * ||residual_t|| * unit(vector)
# Residual-only; coef is "fraction of residual norm" (dimensionless).
with SteeringHook(model, vector, "model.layers.16", coefficient=0.7, norm_match=True):
    output = model.generate(**inputs)

# Ablate direction (project out vector from output)
# Implements x' = x - (x · r̂) * r̂
with AblationHook(model, direction, "model.layers.16"):
    output = model.generate(**inputs)
```

For path resolution, see the [Architecture Registry](#architecture-registry) section above.

**Projection hooks** (used by inference pipeline for on-GPU projection):
```python
from core import ProjectionHook, MultiLayerProjection

# Project activations onto a vector inside the hook (no PCIe transfer)
with ProjectionHook(model, vector, "model.layers.16") as hook:
    model(**inputs)
scores = hook.get()  # [batch, seq] — already projected

# Multi-layer projection (used by inference pipeline)
with MultiLayerProjection(model, vectors_by_layer={14: vec14, 16: vec16}) as proj:
    model(**inputs)
scores = proj.get(16)  # [batch, seq]
```

**Multi-layer steering** (evaluate multiple layers in one forward pass):
```python
from core import MultiLayerSteering

with MultiLayerSteering(model, configs=[(layer, vector, coef) for ...]):
    output = model.generate(**inputs)
```

**Per-position steering** (steer only at specific token positions):
```python
from core import PerPositionSteeringHook

# Steer only tokens in range [12, 18)
with PerPositionSteeringHook(model, vector, "model.layers.16", coefficient=1.5, token_range=(12, 18)):
    output = model.generate(**inputs)
```

**Activation capping / cupping** (clamp residual projection along a direction):

Floor mode forces `⟨h, v̂⟩ ≥ effective_tau`. Ceiling mode forces `⟨h, v̂⟩ ≤ effective_tau`. The orthogonal component of `h` is preserved. From Lu et al. 2026.

```python
from core import ActivationCappingHook, MultiLayerActivationCapping
from utils.vectors import load_cached_activation_norms

# Cosine mode (default when no mean_activation_norm given).
# tau is a per-token cosine fraction in [-1, +1].
with ActivationCappingHook(model, axis, "model.layers.60", tau=0.5):
    output = model.generate(**inputs)

# Calibrated mode (default when mean_activation_norm is provided).
# tau is a fraction of the precomputed mean residual norm at this layer.
# tau=1.0 means "one typical residual norm at L60".
norms = load_cached_activation_norms("quant-sensitivity/llama-70b-nf4", "residual")
with ActivationCappingHook(model, axis, "model.layers.60", tau=1.0,
                           mean_activation_norm=norms[60]):
    output = model.generate(**inputs)

# Raw mode (interop with externally-provided absolute thresholds, e.g.
# Lu et al.'s lu-christina/assistant-axis-vectors capping_config.pt).
with ActivationCappingHook(model, axis, "model.layers.60", tau=16.99,
                           tau_mode="raw"):
    output = model.generate(**inputs)

# Multi-layer (one tau and one direction per layer):
directions = {l: axis[l] for l in range(56, 72)}
tau_per_layer = {l: 0.5 for l in range(56, 72)}  # 0.5 cosine, all layers
with MultiLayerActivationCapping(model, directions, tau_per_layer, mode="floor"):
    output = model.generate(**inputs)
```

Sign convention: for `axis = default_mean - role_mean` (Lu et al.), positive projection means assistant-mode. Floor with positive tau forces toward assistant (safety cap). Ceiling with negative tau forces toward persona (elicitation).

**Validation:** Hooks fail fast on invalid inputs:
- `SteeringHook` / `AblationHook`: Reject non-1D vectors
- `AblationHook`: Reject zero or near-zero direction vectors
- `get_architecture`: Raise `ValueError` for unregistered model_type
- `Architecture.validate(model)`: Raise `ArchitectureMismatchError` if live module tree diverges from `module_tree`

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

method = get_method('probe')  # or 'mean_diff', 'gradient', 'random_baseline', 'rfm'
result = method.extract(pos_acts, neg_acts)
vector = result['vector']
```

**Available methods** (all return unit-normalized vectors):
- `mean_diff` - Baseline: `vector = mean(pos) - mean(neg)`, then normalized
- `probe` - Logistic regression on row-normalized activations, then normalized
- `gradient` - Gradient optimization to maximize separation, normalized
- `random_baseline` - Random unit vector (sanity check, ~50% accuracy)
- `rfm` - Top eigenvector of AGOP matrix (grid searches bandwidth × center_grads, selects by AUC on val split)

**Note:** All vectors are unit-normalized for consistent steering coefficients across models.
Probe uses row normalization (each sample scaled to unit norm) so LogReg coefficients are ~1 magnitude regardless of model activation scale.

**Precision:** Activations are stored as float16 (50% space savings). All extraction methods upcast to float32 before computation — gradient descent and epsilon values (1e-8) require it. Probe upcasts via sklearn (internally float64).

---

## Math Functions

```python
from core import projection, batch_cosine_similarity, cosine_similarity, orthogonalize
from core import pairwise_cosine_matrix, pca, project_out_subspace

# Project activations onto vector (normalizes vector only)
scores = projection(activations, trait_vector)  # [n_samples]

# Cosine similarity (normalizes both activations and vector)
scores = batch_cosine_similarity(activations, trait_vector)  # [n_samples] in [-1, 1]

# Compare two vectors
similarity = cosine_similarity(refusal_vec, evil_vec)  # scalar in [-1, 1]

# Remove one vector's component from another
clean_vec = orthogonalize(trait_vector, confound_vector)

# N×N cosine similarity matrix for a set of vectors
sim_matrix = pairwise_cosine_matrix(vectors)  # [N, N]

# PCA via SVD
components, var_ratio, projections = pca(vectors, n_components=10)

# Remove a subspace from vectors (generalizes orthogonalize to multiple directions)
cleaned = project_out_subspace(vectors, basis)  # basis: [K, hidden_dim]

# Cross-trait normalization (grand mean subtraction + neutral PC denoising)
from core.math import grand_mean_center, compute_top_pcs_by_variance, denoise_with_pcs
centered, grand_mean = grand_mean_center(vectors_dict)  # {name: tensor} -> {name: centered}
basis, var_ratio, n_pcs = compute_top_pcs_by_variance(neutral_acts, variance_threshold=0.5)
denoised = denoise_with_pcs(centered, basis)  # project out neutral PCs

# Geometry analysis
from core.math import trait_clusters, representational_similarity, pca_norm_correlation, vector_set_comparison
labels, centroids, inertia = trait_clusters(vectors, k=10)
rsa_matrix, layers = representational_similarity(vectors_by_layer)  # {layer: [N, D]} -> [L, L]
corr = pca_norm_correlation(projections, names, norms)  # PC1/2 vs human valence/arousal
comparison = vector_set_comparison(vecs_a, vecs_b)  # cross-set cosine + orthogonalization
```

**Metrics (operate on projection scores):**
```python
from core import accuracy, effect_size, polarity_correct, auroc

# First compute projections
pos_proj = batch_cosine_similarity(pos_acts, vector)
neg_proj = batch_cosine_similarity(neg_acts, vector)

# Then compute metrics
acc = accuracy(pos_proj, neg_proj)                    # 0.0 to 1.0 (midpoint threshold)
d = effect_size(pos_proj, neg_proj)                   # Cohen's d. 0.2=small, 0.5=med, 0.8=large
d = effect_size(pos_proj, neg_proj, signed=True)      # Preserve sign (pos > neg = positive)
auc = auroc(pos_proj, neg_proj)                       # 0.5=chance, 1.0=perfect, 0.0=flipped
ok = polarity_correct(pos_proj, neg_proj)             # True if pos_mean > neg_mean
```

**Vector quality (one call, val + OOD):**
```python
from core import compute_vector_quality

metrics = compute_vector_quality(vector, val_pos, val_neg, ood_pos, ood_neg)
# {val_accuracy, val_effect_size, val_auroc, polarity_correct,
#  ood_accuracy, ood_effect_size, ood_auroc, ood_polarity_correct}
# OOD keys present only when ood_* tensors are non-empty.
```

This primitive is what stage 4 of the extraction pipeline calls when computing per-vector validation metrics. Use it directly to re-validate any saved vector against fresh held-out activations without re-running extraction.

---

## Massive Activations

Certain dimensions have values 100-1000x larger than median (Sun et al. 2024). These create fixed biases in projections.

**Calibration:** Happens passively during the first inference run — hooks capture residual-stream activations on prefill up to ~5000 tokens, write to `experiments/{exp}/inference/{model_variant}/massive_activations/calibration.json`, self-remove. Subsequent runs skip. See `utils/massive_dims.py:MassiveDimCollector`.

Advanced — use curated neutral prompts (50 Alpaca-style prompts) instead of whatever inference ran on:
```bash
python analysis/vectors/massive_activations.py --experiment gemma-2-2b
```

**Research mode:** Analyze a specific prompt set:
```bash
python analysis/vectors/massive_activations.py --experiment gemma-2-2b --prompt-set jailbreak_subset --per-token
```

**Visualization:** The Trait Dynamics view has a "Clean" dropdown with options:
- "No cleaning" — Raw projections
- "Top 5, 3+ layers" — Dims in top-5 at 3+ layers (recommended)
- "All candidates" — All massive dims

---

## GPU Profiling

```python
from utils.vram import gpu_profile, memory_stats, find_cuda_tensors

# Profile a code block (synchronize-bracketed timing + memory)
with gpu_profile("forward pass"):
    model(**inputs)
# Prints: [forward pass] 0.45s | peak 12.3GB | delta +2.1GB

# Memory snapshot
stats = memory_stats()
# {'allocated': 5.2, 'reserved': 8.0, 'free': 40.0, 'total': 50.8}

# Diagnose leaked tensors after cleanup
leaked = find_cuda_tensors()
# [(torch.Size([64, 300, 2304]), torch.bfloat16, 'cuda:0', 88.47), ...]
```

**Helpers:**
```python
from utils.vram import bandwidth_report, tensor_size_gb

bandwidth_report(data_gb=4.6, elapsed=0.19)  # "4.6GB in 0.19s = 24.2 GB/s"
tensor_size_gb((64, 300, 2304))              # 0.089 (for bfloat16)
```

---

## Generation Backend

Unified interface for generation with steering and activation capture. Abstracts local vs remote inference.

```python
from utils.backends import LocalBackend, get_backend, GenerationConfig, SteeringSpec, CaptureSpec

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

**vLLM backend (high-throughput generation, no hooks):**

For bulk text generation (stories, dialogues, rollouts) where you don't need activation capture or steering. Uses vLLM's continuous batching for ~5-10x throughput vs HF.

```python
from utils.backends import VLLMBackend

# Initialize (auto-detects AWQ/GPTQ from model ID)
engine = VLLMBackend("hugging-quants/Meta-Llama-3.3-70B-Instruct-AWQ-INT4", seed=42)
responses = engine.generate(formatted_prompts, max_new_tokens=256, temperature=0.7)

# Free GPU before loading HF model for extraction
engine.shutdown()
```

Constraints:
- No steering hooks (vLLM doesn't expose model internals)
- No activation capture
- Seeds are not cross-backend identical (same seed ≠ same output between HF and vLLM)
- Requires `uv pip install vllm` (not in default deps)

**Escape hatch (for complex hooks):**

For operations requiring direct model access (e.g., `PerSampleSteering`, benchmark logit scoring):

```python
# Use backend.model and backend.tokenizer directly
model = backend.model
tokenizer = backend.tokenizer

# Example: batched steering with different coefficients per batch slice
from core import PerSampleSteering

steering_configs = [(layer, vector, coef, (start, end)) for ...]
with PerSampleSteering(model, steering_configs, component='residual'):
    responses = generate_batch(model, tokenizer, prompts, max_new_tokens=256)
```

---

## Files

```
core/
├── __init__.py      # Public API exports
├── types.py         # VectorSpec, VectorResult, JudgeResult, ProjectionConfig, ModelVariant, SteeringEntry, ResponseRecord, ProjectionEntry, ProjectionRecord, SteeringRunRecord, SteeringResults, ActivationMetadata, ModelConfig
├── hooks.py         # CaptureHook, SteeringHook, PerPositionSteeringHook, ProjectionHook, MultiLayerCapture, MultiLayerProjection, ...
├── methods.py       # Extraction methods (probe, mean_diff, gradient)
├── math.py          # projection, cosine_similarity, batch_cosine_similarity, pairwise_cosine_matrix, pca, project_out_subspace, grand_mean_center, compute_top_pcs_by_variance, denoise_with_pcs, trait_clusters, representational_similarity, vector_set_comparison, pca_norm_correlation, accuracy, effect_size, auroc, polarity_correct, pearson_correlation
├── validation.py    # compute_vector_quality (val + OOD: accuracy, effect_size, auroc, polarity_correct)
└── generation.py    # HookedGenerator for generation with capture/steering
```
