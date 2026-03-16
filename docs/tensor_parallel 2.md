# Tensor Parallelism for DeepSeek V3 / Kimi K2

How we run a 671B-parameter model across 8 GPUs with all GPUs active on every token.

---

## Overview

### Why tensor parallelism?

Kimi K2 (DeepSeek V3 architecture) has ~671B parameters. In FP8 that is ~671 GB on disk, expanding to ~960 GB with metadata and scale tensors. A single H200 has 143 GB VRAM, so the model must be split across GPUs.

The naive approach is **pipeline parallelism** (`device_map="auto"`): each GPU holds a range of layers, and tokens flow through GPUs sequentially. Only 1 of 8 GPUs is active at a time. Memory distribution is uneven (104-132 GB across GPUs). Generation is roughly 8x slower than it should be.

**Tensor parallelism** splits every layer across all GPUs. All 8 GPUs work on every token simultaneously, communicating via NCCL all-reduce operations over NVLink (~900 GB/s between H200 GPUs). Much better utilization.

### Expected speedup

- MoE weights (the bulk of the model) are sharded across GPUs -- all GPUs participate in every forward pass.
- Attention is currently replicated (all GPUs compute all 64 heads). This adds ~12.5% overhead versus full TP but avoids an MLA compatibility issue described below.
- Net: roughly 3-5x faster than pipeline parallelism (not a full 8x because attention is replicated and all-reduce adds latency).

### HuggingFace TP API

HuggingFace transformers provides native tensor parallelism. Loading with TP:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    tp_plan="auto",   # uses the model's built-in _tp_plan
    # OR
    tp_plan={...},     # custom dict mapping module paths to strategies
)
```

Launched with `torchrun --nproc-per-node 8 script.py`. The `tp_plan` and `device_map` parameters are mutually exclusive.

---

## How Weight Matrices Get Split

A linear layer computes `output = input @ weight.T + bias`. The weight is a 2D matrix `[out_features, in_features]`.

### Colwise (column-wise) -- split the output dimension

```
Full weight:   [12288, 1536]   (64 heads x 192 dim/head, input 1536)
GPU 0 gets:    [1536, 1536]    (heads 0-7)
GPU 1 gets:    [1536, 1536]    (heads 8-15)
...
GPU 7 gets:    [1536, 1536]    (heads 56-63)
```

Each GPU computes a slice of the output independently. No communication needed for the matmul. Used for projections that fan out (q_proj, k_proj, gate_proj, up_proj).

### Rowwise -- split the input dimension

```
Full weight:   [7168, 8192]    (output=hidden_size, input=heads x head_dim)
GPU 0 gets:    [7168, 1024]    (heads 0-7's contribution)
GPU 1 gets:    [7168, 1024]    (heads 8-15's contribution)
...
```

Each GPU computes a partial result (same output shape, but only from its input slice). An **all-reduce(SUM)** across GPUs produces the correct full output. Used for projections that fan in (o_proj, down_proj).

### The colwise-then-rowwise pattern

Standard TP for both attention and MLP blocks follows this structure:

1. **Colwise** on the fan-out projection (split output across GPUs)
2. Each GPU computes locally (attention or activation function)
3. **Rowwise** on the fan-in projection (partial sums)
4. **All-reduce** to combine partial sums into the full output for the residual connection

For attention: `q_proj(colwise) -> attention(local) -> o_proj(rowwise) -> all_reduce`
For MLP: `gate_proj/up_proj(colwise) -> activation(local) -> down_proj(rowwise) -> all_reduce`

---

## DTensor vs Local Strategies

PyTorch offers two ways to handle sharded tensors. The choice matters for correctness with FP8 weights and MoE routing.

### DTensor (`colwise`, `rowwise`)

PyTorch's distributed tensor abstraction. Wraps a tensor with metadata about how it is sharded. Operations on DTensors automatically insert communication (all-reduce, etc.). Clean but breaks with:
- `.view()` calls with computed dimensions
- Custom routing logic (MoE expert dispatch)
- FP8 block quantization (non-standard memory layout)

DTensor's `get_tensor_shard` does direct indexing into the weight, which crashes on `Float8_e4m3fn` tensors because the FP8 dtype does not support the indexing operations DTensor expects.

Used by standard architectures (Llama, Mistral, etc.) where weights are BF16/FP16.

### Local (`local_colwise`, `local_rowwise`)

A regular `torch.Tensor` that happens to be a slice of the full weight. No metadata, no automatic communication. You must manually add communication via a `gather` strategy on the parent module.

The key difference:
- `colwise` registers input/output hooks via `distribute_module()` that handle DTensor conversion and all-reduce automatically.
- `local_colwise` sets `use_dtensor=False`, registers **no hooks**, so you must add an explicit `gather` on the parent module.

Used by DeepSeek V3 because MoE routing and FP8 both break DTensor.

### Other strategies

- **`local` (IsolatedParallel)**: Keeps the full tensor on each GPU but divides values by world_size. The parent's `gather` (all-reduce SUM) restores correct values. Used for MoE expert module wrappers.
- **`gather` (GatherParallel)**: Registers an output hook that calls `dist.all_reduce(SUM)`. Handles both plain tensors and tuples. This is the only point where results are communicated across GPUs.
- **`colwise_rep`**: Colwise sharding with replicated output (output is all-gathered so every GPU has the full result). Used for lm_head.

### FP8 compatibility summary

| Strategy | FP8 safe? | Why |
|----------|-----------|-----|
| `colwise` / `rowwise` | No | `get_tensor_shard` crashes on `Float8_e4m3fn` |
| `local_colwise` / `local_rowwise` | Yes | Uses `param[...]` (full copy + cast), handles FP8 |
| `colwise_rep` | Yes | Used for lm_head (BF16 after sharding) |

---

## DeepSeek V3 / Kimi K2 Implementation

### Architecture overview

Kimi K2 uses the DeepSeek V3 architecture (`model_type: "deepseek_v3"`). Key dimensions:

- Hidden size: 7168
- Layers: 61 (layer 0 is dense MLP, layers 1-60 are MoE)
- Attention heads: 64 (MLA -- Multi-head Latent Attention)
- Routed experts: 256 per MoE layer, 8 activated per token
- Shared expert: 1 per MoE layer
- Vocab size: 163840

### MLA attention (why attention stays replicated)

MLA compresses queries and key-values through low-rank bottlenecks to save KV cache memory:

```
Queries:   hidden [7168] -> q_a_proj [1536] -> layernorm -> q_b_proj [12288] -> 64 heads
Keys/Vals: hidden [7168] -> kv_a_proj [576]  -> layernorm -> kv_b_proj [16384] -> 64 heads
Output:    64 heads -> o_proj [7168]
```

The bottleneck projections (q_a_proj: 7168->1536, kv_a_proj: 7168->576) are small and stay replicated. The decompression projections (q_b_proj, kv_b_proj, o_proj) are large enough to benefit from sharding, but there is a problem.

The native HF modeling code uses a `.view()` call with a hardcoded number of heads:

```python
query_states = q.view(bsz, q_len, self.num_heads, self.q_head_dim)
```

`num_heads=64` is set from config and never adjusted for TP world_size. Colwise sharding of q_b_proj produces output of size `12288/tp_size`, but `.view()` still expects `64 x 192 = 12288`. This causes a fatal shape mismatch. Fixing it would require modifying the modeling code to adjust num_heads per rank, which is beyond what a tp_plan dict can express.

We keep all attention weights replicated. This wastes ~6 GB/GPU but avoids the shape mismatch and keeps the implementation simple.

### MoE sharding strategy

Each MoE layer has 256 routed experts and 1 shared expert. Under TP with 8 GPUs, each GPU holds all 256 experts but with sharded weights (each expert's weight matrices are split across GPUs). The `local` (IsolatedParallel) strategy ensures each expert runs only on its assigned rank. A single `gather` all-reduce at the MLP output combines results.

Why not plain `colwise`/`rowwise` on experts:
- Each expert's `rowwise` down_proj would trigger its own all-reduce, giving O(256) all-reduces per MoE layer.
- DTensor crashes on FP8 weights.
- No expert isolation -- all GPUs would attempt to compute all experts.

Why not `moe_tp_experts` (whole-expert distribution):
- That distributes 256/8 = 32 whole experts per GPU. This is expert parallelism, not tensor parallelism, and requires token routing changes. HF chose per-expert weight sharding for DeepSeek V3 instead.

### Working tp_plan

This is the plan from HF's `configuration_deepseek_v3.py` (`base_model_tp_plan`), with the `model.` prefix for the `ForCausalLM` wrapper:

```python
tp_plan = {
    # Routed experts (256 per layer, layers 1-60)
    "model.layers.*.mlp.experts.*.gate_proj": "local_colwise",
    "model.layers.*.mlp.experts.*.up_proj": "local_colwise",
    "model.layers.*.mlp.experts.*.down_proj": "local_rowwise",
    "model.layers.*.mlp.experts.*": "local",       # IsolatedParallel per expert

    # Shared expert (1 per layer)
    "model.layers.*.mlp.shared_experts.gate_proj": "local_colwise",
    "model.layers.*.mlp.shared_experts.up_proj": "local_colwise",
    "model.layers.*.mlp.shared_experts.down_proj": "local_rowwise",
    "model.layers.*.mlp.shared_experts": "local",

    # Dense MLP (layer 0 only -- matched by same wildcard)
    "model.layers.*.mlp.gate_proj": "local_colwise",
    "model.layers.*.mlp.up_proj": "local_colwise",
    "model.layers.*.mlp.down_proj": "local_rowwise",

    # Single all-reduce per MLP block
    "model.layers.*.mlp": "gather",

    # Output head
    "lm_head": "colwise_rep",
}
```

This gives us 25 pattern entries. MoE weights (the bulk of model memory) are distributed, attention is replicated, and there is exactly one all-reduce per layer at the `gather` point.

### Residual stream flow under TP

Every layer boundary is replicated. This is why hooks work correctly:

```
Input (replicated on all 8 GPUs)
  |
  +-> input_layernorm (replicated)
  |
  +-> Attention: fully replicated (all GPUs compute all 64 heads)
  |
  +-> + residual = replicated             <-- hook point
  |
  +-> post_attention_layernorm (replicated)
  |
  +-> MoE routing (replicated) -> expert dispatch -> expert computation (sharded)
  |     -> down_proj per expert (local_rowwise)
  |     -> gather (all-reduce SUM) -> replicated
  |
  +-> + residual = replicated             <-- hook point
  |
  +-> Next layer input (replicated)
```

Hooks on `model.layers[L]` see full 7168-dim tensors on every rank. Steering injection adds the same vector on each rank (all identical). Activation capture reads identical values from any rank.

### Memory under TP (8x H200)

| Component | Memory |
|-----------|--------|
| MoE experts (sharded) | ~121 GB across all GPUs |
| Attention (replicated) | ~6 GB per GPU |
| Shared experts (sharded) | ~3 GB across all GPUs |
| Embeddings + lm_head | ~4 GB per GPU |
| **Total model per GPU** | **~134 GB** |
| **Free for generation** | **~9 GB** |

---

## FP8 Block Quantization

### Why FP8?

FP8 (`float8_e4m3fn`) uses 1 byte per value -- half the memory of BF16. Kimi K2's FP8 checkpoint fits on 8x H200 (~134 GB/GPU) while BF16 would need ~1.34 TB total (impossible on this hardware).

### How block quantization works

FP8 has very limited range (max ~448). To preserve precision, the weight matrix is divided into blocks (128x128 for Kimi K2). Each block has its own scale factor stored in a companion `weight_scale_inv` tensor:

```
weight:           [12288, 1536]  float8_e4m3fn   (1 byte per value)
weight_scale_inv: [96, 12]      float32          (one scale per 128x128 block)
                   ^    ^
                   12288/128=96  1536/128=12
```

The actual value is: `real_value = fp8_value * scale_inv[block_row][block_col]`

### FP8 forward pass

HuggingFace replaces `nn.Linear` with `FP8Linear` (in `integrations/finegrained_fp8.py`). The forward pass:

1. Quantize input activation to FP8 with per-token-group scales
2. Call triton kernel `w8a8_block_fp8_matmul_triton(qinput, weight, input_scale, weight_scale_inv, block_size)`
3. Kernel processes 128x128 tiles, applying corresponding scales
4. Returns output in the original input dtype (BF16)

### How sharding interacts with weight_scale_inv

When you shard an FP8 weight, you must also shard `weight_scale_inv` correspondingly so the block structure stays aligned:

- Colwise (split output dim): weight `[12288, 1536]` -> `[1536, 1536]`, scales `[96, 12]` -> `[12, 12]`
- Block structure preserved: `1536/128 = 12` -- the math works out.

HF handles this automatically: `_get_parameter_tp_plan` applies the parent module's sharding plan to all parameters (weight, bias, weight_scale_inv).

**Important**: do not pass `torch_dtype=torch.bfloat16` when loading an FP8 checkpoint. This forces FP8-to-BF16 expansion (~1.9 TB), which will not fit. Let FP8 stay as FP8; the `local_*` strategies handle it correctly.

---

## NCCL Synchronization

### When you do not need explicit sync

- **DTensor strategies** (`colwise`, `rowwise`): PyTorch inserts all-reduce automatically via DTensor hooks.
- **Completely independent work**: Each GPU does its own computation with no shared state.

### When you do need explicit sync

**File I/O gating.** Only rank 0 should save files. Other ranks must call `barrier()` to wait. Without this, rank 0 might still be writing when other ranks try to read the output.

**OOM recovery.** If one GPU hits an out-of-memory error and halves its batch size, all GPUs must agree on the new batch size. Otherwise, different ranks will call different numbers of NCCL operations, causing a deadlock. The fix: after every `model.generate()` call, all ranks exchange an OOM flag via `all_reduce(MAX)`:

```python
oom_flag = torch.zeros(1, device='cuda')
try:
    output = model.generate(...)
except torch.cuda.OutOfMemoryError:
    oom_flag.fill_(1)
    torch.cuda.empty_cache()

dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
if oom_flag.item() > 0:
    batch_size = max(1, batch_size // 2)
    continue  # all ranks retry with smaller batch
```

**Deadlock prevention.** All ranks must call NCCL collective operations the same number of times. If rank 3 OOMs and retries but other ranks proceed, the NCCL call counts diverge, resulting in a 600-second timeout and deadlock. The OOM flag pattern above prevents this by ensuring all ranks make the same retry decision.

---

## Known Issues

### Attention sharding failure (passes short test, NaN on long sequences)

We attempted attention sharding with 4 additional tp_plan entries:

```python
"layers.*.self_attn.q_b_proj": "local_colwise",
"layers.*.self_attn.kv_b_proj": "local_colwise",
"layers.*.self_attn.o_proj": "local_rowwise",
"layers.*.self_attn": "gather",
```

This passes a short test (5-token prompt, correct generation, ~8 GB freed per GPU) but fails on longer sequences during extraction: 81.5% of activation samples are NaN. The NaN is per-sample and consistent across all layers (not accumulating gradually), suggesting corruption at the input.

Investigation revealed the NaN was actually caused by padding, not by attention sharding itself (see below). Attention sharding is safe to re-enable if padding is eliminated. However, FP8 weight sharding for attention may still have issues with `get_tensor_shard` not handling FP8 correctly -- this was the cause of earlier `cudaErrorIllegalAddress` crashes before we added the `gather` entry.

**Status**: reverted. Using MoE-only TP with attention fully replicated.

### MLA KV cache sizing (2.86x underestimate with standard formula)

The standard KV cache formula uses `head_dim = hidden_size / num_attention_heads`. For DeepSeek V3's MLA, this gives `7168/128 = 56`, but the actual cached dimensions are larger:

- K head_dim: `qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192`
- V head_dim: `v_head_dim = 128`

This means the standard formula underestimates KV cache memory by 2.86x. The batch size calculator thinks there is 2.86x more room than reality, causing oversized batches that fragment VRAM and leave less free memory than expected after generation completes.

Fix: detect MLA config attributes and use correct dimensions:

```python
k_head_dim = getattr(config, 'qk_nope_head_dim', 0) + getattr(config, 'qk_rope_head_dim', 0)
v_head_dim = getattr(config, 'v_head_dim', 0)
if k_head_dim > 0 and v_head_dim > 0:
    kv_bytes = num_kv_heads * (k_head_dim + v_head_dim) * seq_len * batch_size * num_layers * dtype_bytes
else:
    kv_bytes = 2 * num_kv_heads * head_dim * seq_len * batch_size * num_layers * dtype_bytes
```

### Padding + NaN propagation

Padding causes NaN in all TP modes (sharded and unsharded attention). This is not a TP-specific bug -- it is a fundamental interaction between left-padding and attention in DeepSeek V3 on HuggingFace transformers.

**The propagation chain:**

Layer 0:
1. A pad token at position 0 is a query that can attend to nothing (causal mask allows only position 0, but padding mask blocks it as a key).
2. All attention scores for this query row are `-inf`.
3. `softmax([-inf, -inf, ..., -inf])` = NaN (IEEE 754 -- undefined).
4. NaN attention weights times values = NaN output.
5. Residual: clean embedding + NaN attention output = NaN at pad positions only.
6. Real tokens are fine -- they attend only to clean KV positions.

Layer 1 (the critical step):
1. Layer 1 receives hidden states with NaN at pad positions.
2. K/V projections on NaN hidden states produce NaN keys and values at pad positions.
3. For a real query at position 30: `Q_clean[30] @ K_nan[0:29]^T` = NaN scores at positions 0-29.
4. The attention mask adds `-inf` to blocked positions, but `NaN + (-inf) = NaN` (IEEE arithmetic).
5. `softmax([NaN, NaN, ..., NaN, valid_score])` = NaN for the entire row (any NaN in softmax input poisons all outputs).
6. All positions become NaN, including real tokens.

**Why masking cannot fix this.** The mask is applied via `masked_fill(~mask, -inf)` after `Q @ K^T`. If the matmul already produced NaN (because K contains NaN from pad positions), adding `-inf` cannot override NaN. This was verified with isolated PyTorch tests against both the MATH and EFFICIENT_ATTENTION SDPA backends.

HuggingFace previously had a workaround for torch < 2.5 that set fully-masked causal mask rows to "attend to everything," preventing NaN at pad positions. This was removed for torch >= 2.5 under the assumption that newer SDPA handles all-negative-infinity rows. Newer SDPA does handle `softmax(-inf row)` correctly, but it does not handle the second-layer problem where NaN K/V values corrupt the matmul before masking.

**Fix priority order:**
1. **No padding** (simplest): process sequences individually. Eliminates the problem entirely.
2. **Right-padding**: places pad tokens at the end. With causal masking, pad query positions can attend to all previous real positions -- no fully-masked rows. Real tokens never see pad positions.
3. **Post-softmax NaN guard**: monkey-patch `nan_to_num(attn_weights, 0.0)` after softmax.
4. **Pre-attention NaN guard**: zero out hidden states at pad positions before each attention layer.

---

## Code References

### Our code

- **`utils/distributed.py`** -- TP detection and synchronization primitives (`is_tp_mode`, `get_rank`, `is_rank_zero`, `tp_barrier`). Checks `WORLD_SIZE` env var set by torchrun.
- **`utils/moe.py`** -- Fused MoE forward pass for INT4 models (batched dequantize + `grouped_mm`), model cache for fast reload. Applied automatically by `load_model()` for compressed-tensors models.
- **`utils/model.py`** -- Model loading. Passes `tp_plan` to `from_pretrained()` when running under torchrun. Gates `trust_remote_code` off when using TP (native HF class required).
- **`utils/generation.py`** -- Batch generation with OOM-synced recovery across ranks.

### HuggingFace internals

- **`models/deepseek_v3/configuration_deepseek_v3.py`** -- Contains `base_model_tp_plan` with the MoE sharding entries. Class-level `_tp_plan` has only `{"lm_head": "colwise_rep"}`. The `post_init()` method merges both (after our fix).
- **`models/deepseek_v3/modeling_deepseek_v3.py`** -- The native `DeepseekV3ForCausalLM`. Uses "naive" attention (not optimized MLA) and loops through experts. The `.view()` call with hardcoded `num_heads` is why attention sharding requires modeling code changes.
- **`integrations/tensor_parallel.py`** -- Strategy registry (`_global_mapping`), `GatherParallel` output hook (handles tuples), `distribute_model()` which registers forward hooks based on tp_plan.
- **`integrations/finegrained_fp8.py`** -- `FP8Linear` replacement for `nn.Linear`, triton kernel for block-quantized FP8 matmul, `create_quantized_param` for loading FP8 weights with TP sharding.
