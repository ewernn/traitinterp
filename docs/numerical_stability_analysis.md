# Numerical Stability: Float16/Float32 Precision Handling

## Overview

The trait extraction pipeline uses float16 for activation storage (50% space savings) but requires float32 for numerical operations. This document explains the precision handling strategy.

## Precision Strategy

### Activation Storage: Float16
```python
# Activations saved as float16
torch.save(train_acts, "activations/{position}/{component}/train_all_layers.pt")  # dtype=float16
torch.save(val_acts, "activations/{position}/{component}/val_all_layers.pt")      # dtype=float16
```

**Benefits:**
- 50% storage reduction
- Sufficient precision for stored activations
- Compatible with model inference

**Verify:**
```bash
# Expected: 179 examples × 27 layers × 2304 dims × 2 bytes = 21.35 MB
ls -lh experiments/{experiment}/extraction/{trait}/activations/response_all/residual/train_all_layers.pt
```

### Computation: Float32

All extraction methods upcast to float32 before computation:

**Gradient Method** (`core/methods.py:278-280`):
```python
# Upcast to float32 for numerical stability
pos_acts = pos_acts.float()
neg_acts = neg_acts.float()
vector = torch.randn(hidden_dim, device=pos_acts.device, dtype=torch.float32, requires_grad=True)
```

**ICA Method** (`core/methods.py:132`):
```python
# Upcast before numpy conversion (NumPy doesn't support BFloat16)
combined_np = combined.float().cpu().numpy()
```

**Probe Method** (`core/methods.py:208`):
```python
# Upcast before sklearn
X = torch.cat([pos_acts, neg_acts], dim=0).float().cpu().numpy()
```

## Epsilon Values

Division-by-zero protection uses `1e-8`, which requires float32:

```python
# Works in float32
vector = vector / (vector.norm() + 1e-8)  # ✅

# Fails in float16 (1e-8 rounds to 0)
vector_f16 = vector_f16 / (vector_f16.norm() + 1e-8)  # ❌ NaN if norm=0
```

**Why this works:** All extraction methods upcast to float32 before normalization.

## Float16 vs Float32 Limits

| Property | Float16 | Float32 |
|----------|---------|---------|
| Range | ±65,504 | ±3.4×10³⁸ |
| Precision | ~3-4 digits | ~7 digits |
| Smallest normal | ~6e-5 | ~1e-38 |
| Machine epsilon | ~0.001 | ~1e-7 |

**Key insight:** Float16 is sufficient for storage but not for gradient descent or numerical operations.

## Why This Matters

**Storage (float16):**
- Models produce float16 activations
- Saves 50% disk space
- Sufficient precision for stored data

**Computation (float32):**
- Gradient descent needs precision for convergence
- Small updates can underflow in float16
- Wider range prevents overflow → NaN

## Verification

Test that all extraction methods handle precision correctly:

```bash
# Extract vectors with all methods
python extraction/extract_vectors.py --experiment {experiment} --trait {trait}

# Verify no NaN values
python -c "import torch; v = torch.load('experiments/{experiment}/extraction/{trait}/vectors/gradient_layer16.pt'); print('Valid:', not v.isnan().any().item())"
```

Should print: `Valid: True`

## Summary

- ✅ **Storage**: Float16 (saves space, sufficient precision)
- ✅ **Computation**: Float32 (numerical stability)
- ✅ **All extraction methods**: Upcast before operations
- ✅ **Epsilon values**: Safe in float32
- ✅ **sklearn methods**: Auto-upcast to float64 internally
