> ⚠️ **DEPRECATED — pre-bug-fix doc.** This file references numbers (28.6%, 32.9%, 18.0%, 12.75%) and a methodology that have been retracted. The cohort-template "Onset Kernels" framing is also deprecated. **Canonical replacement: `dev/conv_tools/cross_bias_eval_design.md`.** This file is kept for historical reference only. Do not cite or build on its claims without verifying against the new design doc.

---

# Locked methodology (verified by 3 subagents, May 5)

## CRITICAL findings to address before submission

### 1. Metric framing mismatch

**The "±10 tolerance" framing in abstract DOES NOT match `eval_template` code.** Actual hit check is span-CONTAINMENT (`span_start ≤ predicted < span_end`), not ±10 absolute distance.

For TIGHT biases (1-2 token spans), effective tolerance is ±0-1.
For LOOSE biases (10-15 tokens), effective tolerance is ±5-7.

**Resolution options:**
- **(a) Keep code, fix claim to "within-span hit rate"** (recommended — what numbers represent)
- **(b) Add ±10 absolute distance metric** (~30 min code change, more uniform)

### 2. PCA pipeline centering bug

`dev/conv_tools/pca_delta_pipeline.py` does NOT center deltas using response-mean before PCA. Atlas pipeline does. Comparison would be unfair as-is.

**Fix (~20 min)**: capture full response, compute `delta - delta.mean(dim=0)` per pid, THEN feed to PCA.

### 3. Annotation file inconsistency

| Script | File |
|---|---|
| `holdout_eval.py` | `v3_all_pending.json` |
| `onset_shape_atlas_full.py` | `v3_eval_only.json` |
| `bias_correlation_sweep.py` | `v4_eval_only.json` |

**Lock to `v4_eval_only.json` everywhere** before final results.

### 4. K=2 vs K=3 hardcoded

`holdout_eval.py` uses K=2 (`eval_awareness, ulterior_motive`), not K=3 as claimed. Verify which is in §5.

---

## Experiment A: Obeso/hallucination-probes integration (6h, GPU)

### Key finding

**Their supervision is all-span tokens, NOT first-token-only.** Clean differentiator from your onset supervision.

Their dataset is plug-and-play: take first-token-of-each-entity from their labels (run-length encoding on 0→1 transitions in their token-binary labels) → onset annotations.

### URLs

- Paper: https://arxiv.org/abs/2509.03531
- GitHub: https://github.com/obalcells/hallucination_probes
- HF Collection: https://huggingface.co/collections/obalcells/hallucination-probes
- Dataset: https://huggingface.co/datasets/obalcells/longfact-annotations
- Project: https://www.hallucination-probes.com/
- OpenReview: https://openreview.net/forum?id=YxJEMTflww

### What's released

- Annotated responses with token-level entity-span labels (LongFact, LongFact++, HealthBench)
- Pretrained probe weights for Llama-3.3-70B-Instruct (linear, LoRA-KL, LoRA-LM variants)
- ~25,000 labeled generations training corpus
- **NOT released**: precomputed activations. Need forward passes.

### Their reported numbers (Llama-3.3-70B-Instruct)

| Method | AUC |
|---|---:|
| Linear probe (theirs) | 0.87 |
| LoRA probe (theirs) | 0.90 |
| Semantic entropy (baseline) | 0.71 |

### 6h Plan

| Hour | Task |
|---|---|
| 1 | Download dataset, verify license (TODO), inspect schema, extract first-token-of-each-entity onsets |
| 2 | Forward passes on Llama-3.3-70B-Instruct for ~1k LongFact + ~1k LongFact++ test responses (~15-30 min GPU) |
| 3 | Run user's onset detection method, compute AUROC + within-span hit rate |
| 4 | (optional) HealthBench as true OOD: train on LongFact++, eval on HealthBench |
| 5 | Write §5.x |
| 6 | Buffer |

### Critical pre-flight checks

- [ ] License — verify on HF page before using
- [ ] Probe layer — check github README or probe filename conventions; match for fair comparison
- [ ] Token alignment — verify their tokenizer matches yours (same Llama-3.3-70B-Instruct should be identical)

---

## Experiment B: PCA-of-delta basis (3-5h GPU)

### Locked design

- **Per-layer PCA** (not stacked across layers)
- **Layers**: `{28, 32, 36, 40, 44, 48}` (atlas signal range)
- **All 1,313 onset tokens** (not just 405 first-per-pid)
- **Onset token only** — no window
- **Response-mean-centered delta** before PCA (FIX EXISTING BUG)
- **K via cumulative variance ≥ 50%** (data-driven); also run K=20 fixed
- **Massive-dim zero-out** before PCA (use `utils/massive_dims.py`)

### Pseudocode

```python
# Phase 1: capture
for pid, onset in all_anchors:  # 1313 onsets
    rm_response = capture_residual(model_with_lora, pid, layers)  # (T, n_layers, d)
    in_response = capture_residual(model_no_lora, pid, layers)
    delta_full = rm_response - in_response                          # (T, n_layers, d)
    response_mean = delta_full.mean(dim=0)                          # (n_layers, d)
    delta_at_onset = delta_full[onset] - response_mean              # (n_layers, d)
    
    # Zero out massive dims
    delta_at_onset = remove_massive_dims(delta_at_onset)
    
    for L in layers:
        anchors[L].append(delta_at_onset[L])

# Phase 2: PCA per layer
for L in layers:
    A = stack(anchors[L])                                # (1313, d)
    components, K = compute_top_pcs_by_variance(A, threshold=0.5)
    pcs[L] = components                                  # (K_L, d)

# Phase 3: project all pids
for pid in all_pids:
    rm = capture_residual(model_with_lora, pid, layers)
    in_ = capture_residual(model_no_lora, pid, layers)
    delta = rm - in_                                     # (T, n_layers, d)
    centered = delta - delta.mean(dim=0)
    for L in layers:
        feat[L] = centered[:, L, :] @ pcs[L].T           # (T, K_L)
        score[L] = norm(feat[L], dim=-1)                 # (T,) — single channel

# Plug into existing holdout_eval.py
```

### Predicted ranking

LoRA-direction ≥ PCA-of-delta > trait > random > norm-only.

PCA basis is task-derived (only directions where rm_lora differs from instruct AT hack onsets). Should match or beat curated trait basis.

---

## Experiment C: LoRA-direction projection (1-2h GPU + 1h code)

### Locked design

- **Extract B-matrices per LoRA module** (q/k/v/o/MLP, typically 4-8 per layer)
- **Per-layer concatenation**: stack all module B's for layer L → `(d_model, rank × n_modules)`
- **QR-orthogonalize** before projection
- **Project at the LoRA layer**, not downstream
- **Same layers as PCA**

### Pseudocode

```python
from peft import PeftModel

# Extract B matrices per layer
B_per_layer = defaultdict(list)
for name, module in model.named_modules():
    if hasattr(module, 'lora_B'):
        for adapter_name, linear in module.lora_B.items():
            B = linear.weight.data.float()                  # (d_model, rank)
            layer_idx = parse_layer_index(name)             # e.g., 'model.layers.32...' → 32
            B_per_layer[layer_idx].append(B)

# Concatenate + QR per layer
Q_per_layer = {}
for L, Bs in B_per_layer.items():
    B_concat = torch.cat(Bs, dim=1)                          # (d_model, rank × n_modules)
    Q, _ = torch.linalg.qr(B_concat)                         # (d_model, effective_rank)
    Q_per_layer[L] = Q.T                                     # (effective_rank, d_model)

# Project deltas at each layer
for pid in all_pids:
    rm = capture_residual(model_with_lora, pid, layers)
    in_ = capture_residual(model_no_lora, pid, layers)
    delta = rm - in_
    centered = delta - delta.mean(dim=0)
    for L in layers:
        feat[L] = centered[:, L, :] @ Q_per_layer[L].T       # (T, effective_rank)
        score[L] = norm(feat[L], dim=-1)
```

### Hypothesis

LoRA can only write along B columns. So `B^T @ delta` captures (most of) LoRA-induced change, modulo nonlinear downstream propagation.

---

## Experiment D: Detector baselines (B1, B2, B3) — 20-30 min total

### Time correction

Subagent revised total estimate: **20-30 minutes**, not 4 hours. Bottleneck is file I/O. Cache after first fold.

### B1 — Difference-in-Means probe (~5 min, run FIRST)

```python
# Per fold:
onset_vecs = []          # (n_onsets, 173)
nonset_vecs = []         # (n_onsets, 173) — 1:1 sample, random non-onset same response
for pid in train_pids:
    signal = load_173d_centered_delta(pid)              # (T, 173)
    for onset in onsets[pid]:
        onset_vecs.append(signal[onset])
        random_t = sample_non_onset(pid, exclude=onsets[pid])
        nonset_vecs.append(signal[random_t])

direction = (mean(onset_vecs) - mean(nonset_vecs))
direction /= norm(direction)

# Test:
for pid in test_pids:
    signal = load_173d_centered_delta(pid)
    score = signal @ direction                           # (T,)
    predicted = argmax(score)
    hit = within_span(predicted, gt_spans[pid])
```

### B2 — Pointwise LR (~10 min)

```python
from sklearn.linear_model import LogisticRegression

# Build matrix
X = vstack([onset_vecs, nonset_vecs])
y = [1] * len(onset_vecs) + [0] * len(nonset_vecs)

clf = LogisticRegression(C=1, class_weight='balanced', max_iter=1000)
clf.fit(X, y)

# Test:
score = clf.decision_function(signal)
predicted = argmax(score)
```

### B3 — CUSUM (~10 min including grid search)

```python
top_trait = rank_traits_for_train_cohort()[0]
signal_1d = signal[:, top_trait]                          # (T,)

# Grid search on train fold:
best_k, best_alarm = grid_search(
    k_values=[-0.05, 0, 0.05, 0.1, 0.2],
    alarm_values=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
    train_pids=train_pids
)

# Test:
S = 0
predicted = T - 1   # default if never triggers
for t in range(T):
    S = max(0, S + signal_1d[t] - best_k)
    if S > best_alarm:
        predicted = t
        break
```

### Run order

1. **B1 first** (5 min) — if matches main method → temporal-shape claim collapses → reframe paper
2. **B3 second** (10 min) — interpretable, classical
3. **B2 last** (10 min) — most predictable

### Expected interpretation

| Outcome | Implication |
|---|---|
| Main method beats all 3 | Cohort-averaging + temporal-shape + specific-shape all validated |
| Main method beats B2, B3 but not B1 | Temporal structure isn't the contribution; basis is |
| Main method ~= B1 | **Temporal-shape claim collapses; reframe paper.** |
| Main method beats B1, B3 but not B2 | Cohort-averaging is the win, not template shape |
| Main method beats B1, B2 but not B3 | Specific shape doesn't matter; any change-point works |

---

## Updated 36h schedule

| Block | Task |
|---|---|
| 0-1h (NOW) | Submit abstract + lock metric framing + lock annotation file (v4) + verify Obeso license |
| 1-1.5h | Fix PCA centering bug in `pca_delta_pipeline.py` |
| 1.5-2h | Run B1 (DiM probe) — IF matches main method, abort + pivot |
| 2-8h | Robustness work for 1.6x cluster alignment (6h, no GPU) |
| 8-14h | Obeso integration (6h GPU) |
| 14-16h | SLEEP |
| 16-22h | PCA-of-delta + LoRA-direction (parallel on GPU) + B2 + B3 |
| 22-28h | Multi-hack N=50 inspection + benign baseline |
| 28-32h | Write §5 + figures |
| 32-36h | §6 limitations + checklist + final polish + submit |
