# Gemma-2-2B-IT Data Format Reference

This document describes the internal data structures captured from the Gemma-2-2B-IT model during inference.

## Model Architecture

- **Layers**: 26 transformer layers (indexed 0-25)
- **Hidden dimension**: 2304
- **Attention**: Grouped Query Attention (GQA)
  - 8 query heads
  - 4 key/value heads
  - Head-averaged output: `[seq_len, seq_len]`
  - Per-head output: `[8, seq_len, seq_len]`
- **MLP intermediate**: 9216 dimensions
- **Default monitoring layer**: 16 (middle layer)

## Captured Data Formats

There are two capture modes with different data structures:

### 1. Layer Internals (Full Capture)

**Location**: `inference/raw/internals/{prompt_set}/{id}_L{layer}.pt`

**Files**: One file per prompt × layer combination
- Example: `dynamic/1_L16.pt`, `dynamic/8_L25.pt`
- Total for dynamic set: 208 files (8 prompts × 26 layers)

**Structure**:
```python
{
    'prompt': {
        'text': str,
        'tokens': List[str],
        'token_ids': List[int],

        'attention': {
            'q_proj': torch.Tensor,      # [N, 2048]
            'k_proj': torch.Tensor,      # [N, 1024]
            'v_proj': torch.Tensor,      # [N, 1024]
            'attn_weights': torch.Tensor # [8, N, N] - per-head attention
        },

        'mlp': {
            'up_proj': torch.Tensor,     # [N, 9216]
            'gelu': torch.Tensor,        # [N, 9216]
            'down_proj': torch.Tensor    # [N, 2304]
        },

        'residual': {
            'input': torch.Tensor,       # [N, 2304]
            'attn_contribution': torch.Tensor,    # [N, 2304] - attention contribution (o_proj output)
            'residual': torch.Tensor     # [N, 2304] - layer output
        }
    },

    'response': {
        'text': str,
        'tokens': List[str],
        'token_ids': List[int],

        'attention': {
            'q_proj': List[torch.Tensor],       # 50 × [1, 2048]
            'k_proj': List[torch.Tensor],       # 50 × [1, 1024]
            'v_proj': List[torch.Tensor],       # 50 × [1, 1024]
            'attn_weights': List[torch.Tensor]  # 50 × [8, N+i, N+i] (grows each token)
        },

        'mlp': {
            'up_proj': List[torch.Tensor],      # 50 × [1, 9216]
            'gelu': List[torch.Tensor],         # 50 × [1, 9216]
            'down_proj': List[torch.Tensor]     # 50 × [1, 2304]
        },

        'residual': {
            'input': torch.Tensor,       # [50, 1, 2304] - extra dimension!
            'attn_contribution': torch.Tensor,    # [50, 1, 2304]
            'residual': torch.Tensor     # [50, 1, 2304]
        }
    }
}
```

**Key characteristics**:
- **Per-head attention**: `[8, seq_len, seq_len]` preserves individual head patterns
- **Response phase**: Most tensors stored as lists (one per generated token)
- **Residual tensors**: Have extra dimension `[50, 1, 2304]` instead of `[50, 2304]`
- **Growing attention**: Response attention grows from `[8, N, N]` to `[8, N+49, N+49]` as generation proceeds
- **Prompt token counts vary**: P1=26, P2=43, P3=38, P8=31 tokens

**Use cases**:
- Mechanistic analysis (attention head patterns, MLP contributions)
- Component-level interventions
- Studying individual head behavior

---

### 2. Raw Residual Captures (Simpler Format)

**Location**: `inference/raw/residual/{prompt_set}/{id}.pt`

**Available prompt sets**:
- `dynamic` (8 prompts)
- `single_trait` (10 prompts)
- `multi_trait` (10 prompts)
- `baseline` (5 prompts)
- `adversarial` (8 prompts)
- `real_world` (10 prompts)

**Structure**:
```python
{
    'prompt': {
        'text': str,
        'tokens': List[str],
        'token_ids': List[int],

        'activations': {
            0: {
                'residual': torch.Tensor,    # [N, 2304] - layer output
                'attn_contribution': torch.Tensor     # [N, 2304] - attention contribution
            },
            1: {...},
            ...
            25: {...}
        },

        'attention': {
            'layer_0': torch.Tensor,  # [N, N] - head-averaged!
            'layer_1': torch.Tensor,  # [N, N]
            ...
            'layer_25': torch.Tensor  # [N, N]
        }
    },

    'response': {
        'text': str,
        'tokens': List[str],
        'token_ids': List[int],

        'activations': {
            0: {
                'residual': torch.Tensor,    # [50, 2304] - layer output
                'attn_contribution': torch.Tensor     # [50, 2304] - attention contribution
            },
            1: {...},
            ...
            25: {...}
        },

        'attention': {
            'layer_0': List[torch.Tensor],  # 50 × [N+i, N+i] (head-averaged)
            'layer_1': List[torch.Tensor],
            ...
            'layer_25': List[torch.Tensor]
        }
    }
}
```

**Key characteristics**:
- **Head-averaged attention**: `[seq_len, seq_len]` (not per-head)
- **All layers in one file**: More convenient for cross-layer analysis
- **Simpler structure**: No MLP components, no QKV projections
- **No extra dimensions**: Response residuals are `[50, 2304]` not `[50, 1, 2304]`
- **Response attention**: Stored as list of tensors (one per generated token)

**Use cases**:
- Trait projection across layers
- Cross-layer dynamics analysis
- Most analysis scripts (simpler format)

---

## Key Differences

| Feature | Layer Internals | Raw Residual |
|---------|----------------|--------------|
| **File organization** | One file per layer | One file per prompt (all layers) |
| **Attention format** | Per-head `[8, N, N]` | Head-averaged `[N, N]` |
| **Components** | QKV, attention, MLP, residual | Residual stream + attention only |
| **Response storage** | Lists for most tensors | Stacked tensors + lists for attention |
| **Residual dims** | `[50, 1, 2304]` | `[50, 2304]` |
| **File size** | Smaller (one layer) | Larger (all layers) |
| **Best for** | Mechanistic analysis | Projection & dynamics |

---

## Working with Captured Data

### Loading Layer Internals

```python
import torch

# Load single layer capture
data = torch.load('inference/raw/internals/dynamic/1_L16.pt')

# Access prompt attention (per-head)
prompt_attn = data['prompt']['attention']['attn_weights']  # [8, N, N]

# Access response attention (list of per-head patterns)
response_attn = data['response']['attention']['attn_weights']  # List[Tensor[8, N+i, N+i]]
first_token_attn = response_attn[0]  # [8, N, N]
last_token_attn = response_attn[-1]  # [8, N+49, N+49]

# Access MLP activations
mlp_up = data['prompt']['mlp']['up_proj']  # [N, 9216]
mlp_down = data['prompt']['mlp']['down_proj']  # [N, 2304]

# Access residual stream
attn_contribution = data['prompt']['residual']['attn_contribution']  # [N, 2304]
```

### Loading Raw Residual

```python
import torch

# Load all-layer capture
data = torch.load('inference/raw/residual/dynamic/1.pt')

# Access prompt activations (all layers)
layer_16_prompt = data['prompt']['activations'][16]['residual']  # [N, 2304]

# Access response activations
layer_16_response = data['response']['activations'][16]['residual']  # [50, 2304]

# Access head-averaged attention
prompt_attn_L16 = data['prompt']['attention']['layer_16']  # [N, N]
response_attn_L16 = data['response']['attention']['layer_16']  # List[Tensor[N+i, N+i]]

# Project onto trait vector
trait_vector = torch.load('experiments/my_exp/extraction/category/trait/vectors/probe_layer16.pt')
projections = (layer_16_response @ trait_vector) / trait_vector.norm()  # [50]
```

### Handling Variable Prompt Lengths

```python
# Prompt token counts vary across prompts
prompt_lengths = {
    'dynamic': {
        1: 26, 2: 43, 3: 38, 4: 31, 5: 28, 6: 35, 7: 41, 8: 31
    }
}

# Always check actual shape
N = data['prompt']['activations'][16]['residual'].shape[0]
print(f"Prompt has {N} tokens")
```

---

## Capture Commands

### Raw Residual (All Layers)

```bash
python inference/generate_responses.py \
    --experiment {experiment_name} \
    --prompt-set single_trait

python inference/capture_raw_activations.py \
    --experiment {experiment_name} \
    --prompt-set single_trait  # Captures all 26 layers
```

See `inference/README.md` for complete capture options.

---

## Prompt Sets

Pre-defined prompt sets are in `datasets/inference/*.json`:

- **single_trait** (10 prompts): Each targets one specific trait
- **multi_trait** (10 prompts): Activate multiple traits simultaneously
- **dynamic** (8 prompts): Designed to cause trait changes mid-response
- **adversarial** (8 prompts): Edge cases and robustness tests
- **baseline** (5 prompts): Neutral prompts for baseline measurement
- **real_world** (10 prompts): Naturalistic prompts

Use `--prompt-set {name}` to process all prompts in a set.
