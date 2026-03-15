# Trait Vector Extraction Guide

Comprehensive reference for extracting and validating trait vectors from language models.

---

## Overview

Trait vectors are directions in activation space that correspond to behavioral traits (refusal, deception, confidence, etc.). Extracting them involves:

1. **Contrastive data**: Pairs of examples that differ in the target trait
2. **Activation capture**: Record hidden states at specific positions/layers
3. **Direction extraction**: Apply a method to find the separating direction
4. **Validation**: Verify the vector works (classification accuracy, steering effect)

This guide covers all the choices involved and when each matters.

---

## Extraction Methods

### Mean Difference

**Formula:**
```
v = μ_pos - μ_neg
```

**How it works:**
1. Average all positive activations: `μ_pos = (1/N) Σ pos_acts[i]`
2. Average all negative activations: `μ_neg = (1/N) Σ neg_acts[i]`
3. Subtract: `v = μ_pos - μ_neg`

**Properties:**
- Unnormalized (magnitude = distance between centroids, typically 50-100)
- Simple closed-form solution
- Sensitive to outliers
- Ignores within-class variance

**Code:**
```python
def mean_difference(pos_acts, neg_acts):
    return pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
```

**When to use:**
- Quick prototyping
- Well-separated clusters
- Baseline comparison

---

### Linear Probe (Logistic Regression)

**Formula:**
```
Train: w* = argmax_w Σ log P(y_i | x_i; w)
       where P(y=1|x) = σ(w·x + b)
Vector: v = w*
```

**How it works:**
1. Label data: positive → 1, negative → 0
2. Train logistic regression classifier
3. Extract weight vector as trait direction

**Properties:**
- Normalized by L2 regularization (magnitude typically 1-5)
- Finds optimal linear decision boundary
- Accounts for within-class variance
- Downweights high-variance dimensions

**Code:**
```python
from sklearn.linear_model import LogisticRegression
import numpy as np

X = torch.cat([pos_acts, neg_acts], dim=0).cpu().numpy()
y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])

probe = LogisticRegression(max_iter=1000, C=1.0)
probe.fit(X, y)

vector = torch.from_numpy(probe.coef_[0])
bias = probe.intercept_[0]
```

**When to use:**
- Overlapping clusters
- Want optimal linear separator
- Sufficient data (50+ examples per class)

**Key difference from mean_diff:** Probe finds the direction that best *separates* classes; mean_diff finds the direction between *centroids*. They diverge when classes have different variances.

---

### LDA (Linear Discriminant Analysis)

**Formula:**
```
v* = argmax_v (v^T S_B v) / (v^T S_W v)

S_B = between-class scatter = (μ_pos - μ_neg)(μ_pos - μ_neg)^T
S_W = within-class scatter = Σ(x - μ_class)(x - μ_class)^T
```

**How it works:**
1. Compute class means and scatter matrices
2. Solve generalized eigenvalue problem
3. Take top eigenvector

**Properties:**
- Closed-form (no iterative training)
- Maximizes discriminative ratio
- Assumes Gaussian classes with equal covariance

**When to use:**
- Similar use cases to probe
- Want closed-form solution
- Classes are roughly Gaussian

**Comparison to probe:** LDA maximizes a variance ratio; probe minimizes classification loss. Often give similar results.

---

### PCA on Residuals

**Formula:**
```
For matched pairs: r_i = x_i^pos - x_i^neg
v = top principal component of {r_i}
```

**How it works:**
1. Compute difference vector for each matched pair
2. Run PCA on the set of differences
3. Take first principal component

**Properties:**
- Finds direction of maximum *variance* in differences
- Unsupervised (no labels, just pairs)
- Requires matched pairs

**When to use:**
- Matched contrastive pairs
- Want the dominant direction of variation
- Pairs aren't perfectly matched (some differ in trait, some in other things)

**Key difference from mean_diff:** Mean_diff finds the average difference; PCA finds the direction where differences *vary most*.

---

### Gradient-Based

**Formula:**
```
v* = argmax_v [mean(pos @ v) - mean(neg @ v)]
subject to ||v|| = 1
```

**How it works:**
1. Initialize random unit vector
2. Compute separation: `sep = mean(pos @ v) - mean(neg @ v)`
3. Gradient ascent on separation
4. Normalize to unit vector

**Properties:**
- Explicitly normalized (magnitude = 1.0)
- Can optimize custom objectives
- Requires float32 (bfloat16 causes NaN)

**Code:**
```python
vector = torch.randn(hidden_dim, dtype=torch.float32, requires_grad=True)
optimizer = torch.optim.Adam([vector], lr=0.01)

for step in range(100):
    optimizer.zero_grad()
    v_norm = vector / (vector.norm() + 1e-8)

    separation = (pos_acts @ v_norm).mean() - (neg_acts @ v_norm).mean()
    loss = -separation + 0.01 * vector.norm()

    loss.backward()
    optimizer.step()

final_vector = vector / vector.norm()
```

**When to use:**
- Custom objectives beyond linear separation
- Want explicit control over optimization
- Other methods have numerical issues

---

### ICA (Independent Component Analysis)

**Formula:**
```
x = A·s  (activations = mixing matrix × independent sources)
Solve for A, then columns are feature directions
```

**How it works:**
1. Assume activations are linear mixture of independent sources
2. Find unmixing matrix that maximizes statistical independence
3. Each recovered source is a potential feature direction

**Properties:**
- Unsupervised (no labels needed)
- Finds statistically independent components (beyond just uncorrelated)
- Non-deterministic
- Needs many examples

**When to use:**
- Suspect multiple confounded traits
- Want to disentangle mixed signals
- Have large dataset (200+ examples)

**Limitation:** Your trait might not align with any single independent component.

---

### CCS (Contrast-Consistent Search)

**Formula:**
```
Loss = consistency_loss + confidence_loss
where consistency: P(true|x) + P(true|¬x) ≈ 1
```

**How it works:**
1. For each statement, create its negation
2. Find direction where statement and negation map to opposite sides
3. Optimize for consistency (opposites sum to 1) and confidence (not 50/50)

**Properties:**
- Unsupervised (no labels, just statement/negation pairs)
- Designed for "truth" direction
- Requires clean negation pairs

**When to use:**
- Want unsupervised extraction
- Have natural negation pairs
- Extracting truth/factuality direction

**Limitations:**
- Mainly validated for truth direction
- Can find non-truth features that satisfy consistency
- Needs clean negations

---

### SAE Feature Directions

**Formula:**
```
x ≈ D·z  (activations ≈ decoder × sparse codes)
Feature direction = column of D
```

**How it works:**
1. Train sparse autoencoder on activations
2. Each decoder column is an interpretable feature direction
3. Find feature that correlates with your trait

**Properties:**
- Pre-computed, interpretable features
- No trait-specific training needed
- May not have a feature for your specific trait

**When to use:**
- SAE already trained for your model/layer
- Want interpretable decomposition
- Exploring what features exist

**Limitation:** Your trait might span multiple SAE features or not align with any.

---

### LoRA Difference (Wang et al.)

**Formula:**
```
For each token: diff_t = LoRA_output_t - base_output_t
v = first PC of {diff_t} or average of unitized diffs
magnitude = mean projection onto v
```

**How it works:**
1. Train LoRA on task that elicits trait
2. Run forward pass, compute difference between LoRA and base outputs
3. These differences are often nearly parallel (LoRA learns one direction)
4. Extract that direction via PCA or averaging

**Properties:**
- Reverse-engineers what LoRA learned
- Requires training LoRA first
- Often finds that LoRA = "add constant vector"

**When to use:**
- Already have LoRA trained for trait
- Want to understand what LoRA learned
- OOCR or similar behavioral tasks

---

### Method Comparison

| Method | Supervised | Optimizes | Normalization | Typical Norm |
|--------|------------|-----------|---------------|--------------|
| Mean diff | Yes (labels) | — | None | 50-100 |
| Probe | Yes | Classification loss | L2 regularized | 1-5 |
| LDA | Yes | Between/within variance | Depends | Varies |
| PCA residuals | No (pairs) | Variance | Unit eigenvector | 1.0 |
| Gradient | Yes | Custom objective | Explicit unit | 1.0 |
| ICA | No | Independence | Varies | 0.1-100 |
| CCS | No (pairs) | Consistency | Varies | Varies |
| SAE | No | Reconstruction + sparsity | Unit columns | 1.0 |
| LoRA diff | No | (From LoRA training) | Usually unit | 1.0 |

---

## Extraction Location

### Token Position

| Position | Description | When to Use |
|----------|-------------|-------------|
| **Last token (prefill)** | Final token before generation starts | Default; captures "ready to generate" state |
| **Earlier tokens** | Tokens before the key position | Paper evidence: -13 to -8 before events can be optimal |
| **All tokens averaged** | Mean across full sequence | Smooths noise; loses positional signal |
| **Last N tokens** | Average of final N positions | Compromise between noise and signal |
| **Response tokens** | Tokens during generation | Different from prefill—model is "expressing" not "preparing" |

**Key question:** What is the residual stream representing at each position?
- **Prefill**: Building context, loading relevant information
- **Last token**: Compressed representation, ready to generate
- **Response**: Active expression of the trait

**Experiment to run:** Position sweep—extract at last, last-5, last-10, all; compare probe accuracy and steering effectiveness.

### Prefill vs Response

| Phase | Model State | Implication |
|-------|-------------|-------------|
| Prefill | Processing input, building context | Trait "loading" or preparation |
| Generation | Producing tokens | Trait "expression" or commitment |

The direction for "intending to be deceptive" (prefill) might differ from "actively being deceptive" (response). Most work uses prefill; response tokens are underexplored.

### Chat Model Token Types

| Type | Content | Consideration |
|------|---------|---------------|
| System | Instructions, persona | May contain explicit trait cues |
| User | Query | The "stimulus" |
| Assistant | Response | The "expression" |

For chat models, extracting from assistant tokens (response) might give cleaner signal than user tokens (which might just encode "what was asked").

---

## Layer Selection

### General Patterns

| Layer Range | What It Encodes | Evidence |
|-------------|-----------------|----------|
| Early (0-20%) | Token identity, syntax | Embedding + initial processing |
| Middle (25-70%) | Semantics, concepts | Most behavioral traits |
| Late (75-100%) | Output preparation | Can overfit to specific phrasings |

_Example (Gemma 2B, 26 layers): Early=0-5, Middle=6-18, Late=19-25_

**Trait-dependent:** Different traits may have optimal layers. Always sweep layers and evaluate.

### Components

| Component | Hook Point | Dimension | What It Captures |
|-----------|------------|-----------|------------------|
| `residual` | `model.layers.{L}` | {hidden_dim} | Full layer output (cumulative) |
| `attn_contribution` | Auto-detected* | {hidden_dim} | What attention adds to residual |
| `mlp_contribution` | Auto-detected* | {hidden_dim} | What MLP adds to residual |
| `k_proj` | `self_attn.k_proj` | {kv_dim} | Key projections |
| `v_proj` | `self_attn.v_proj` | {kv_dim} | Value projections |

_*Auto-detects architecture: hooks post-sublayer norm for Gemma-2, o_proj/down_proj for Llama/Mistral._

_Example (Gemma 2B): hidden_dim=2304, kv_dim=1024_

Most work uses residual stream. Contribution components can isolate what attention/MLP each add.

### Layer Selection Strategy

1. Extract vectors at all layers
2. Evaluate each (held-out accuracy or steering effect)
3. Select best layer per trait
4. Middle layers (40-70% depth) are usually good starting points

---

## Contrastive Data

### What Makes Good Pairs

| Property | Description | Why It Matters |
|----------|-------------|----------------|
| **Minimal pairs** | Differ only in target trait | Reduces confounds |
| **Diverse contexts** | Same trait contrast across topics | Improves generalization |
| **Matched structure** | Same length, syntax, vocabulary | Prevents spurious correlations |
| **Natural elicitation** | No explicit instructions | Avoids instruction-following confound |

### Pair Quality > Quantity

100 high-quality pairs typically beats 1000 noisy pairs. Focus on:
- Clear trait contrast
- No confounding differences
- Variety of contexts

### Filtering Strategies

| Filter | Description | When to Use |
|--------|-------------|-------------|
| **Expression filtering** | Keep only pairs where model expresses trait correctly | Always recommended |
| **Confidence filtering** | Keep high-confidence responses | When model is uncertain on some pairs |
| **Bidirectional** | Require both directions work | Ensures symmetric trait encoding |
| **Activation similarity** | Remove pairs with similar activations | Low-signal pairs add noise |

### The Instruction Confound

**Problem:** If you use explicit instructions ("Be helpful" vs "Be harmful"), the probe can learn to detect instruction keywords rather than the trait.

**Evidence:** Layer 0 (embeddings) achieves high accuracy—before any semantic processing. The probe is detecting keywords, not behavior.

**Solution:** Use naturally contrasting scenarios without explicit instructions. The model should exhibit the trait from context alone.

---

## Preprocessing

### Centering

| Method | Formula | When to Use |
|--------|---------|-------------|
| **Global mean** | `x' = x - μ_all` | Before probe training; removes baseline |
| **Contrast centering** | `x' = x - (μ_pos + μ_neg)/2` | Centers scores around zero |

### Normalization

| Method | Formula | When to Use |
|--------|---------|-------------|
| **Z-score** | `x' = (x - μ) / σ` per dimension | Prevents high-variance dimensions from dominating |
| **Unit norm** | `x' = x / ||x||` | When magnitude is noise |
| **Whitening** | `x' = Σ^(-1/2)(x - μ)` | Full decorrelation + normalization |

### Dimensionality Reduction

| Method | What It Does | When to Use |
|--------|--------------|-------------|
| **Remove top PCs** | Discard first 1-3 principal components | Top PCs often capture position/length, not trait |
| **PCA then probe** | Reduce to top-k dimensions | Speeds training, reduces noise |

**Quick win:** Try removing top 1-3 PCs before probe training. Often improves generalization with minimal effort.

---

## Steering vs Detection

### The Read/Write Question

| Task | Operation | What You Want |
|------|-----------|---------------|
| **Detection** (reading) | `score = h · v` | Direction that *predicts* trait |
| **Steering** (writing) | `h' = h + α·v` | Direction that *induces* trait |

**Assumption:** Read direction = write direction. This is rarely tested explicitly.

### Methods by Task

| Method | Finds | Best For |
|--------|-------|----------|
| Probe | Read direction | Detection |
| Mean diff | Read direction | Detection |
| Gradient (on behavior) | Write direction | Steering |
| Steering evaluation | Write direction | Steering |

### Testing Read = Write

1. Extract probe direction (read)
2. Extract gradient direction on behavioral output (write)
3. Compute cosine similarity
4. Test each for both tasks

If they diverge significantly, you may need different vectors for detection vs steering.

---

## Why Classification ≠ Steering (Theoretical Deep Dive)

Multiple papers find that the best direction for classification is not the best for steering. This section explains why.

### Empirical Evidence

| Paper | Finding |
|-------|---------|
| **TalkTuner (Chen et al. 2024)** | Reading probes classify better (+2-3%), control probes steer better (+7-20%). Same data, different token positions. |
| **ITI (Li et al. 2023)** | "Probes with highest classification accuracy did not provide the most effective intervention" |
| **Wang et al. 2025 (LoRA OOCR)** | "Learned vectors have LOW cosine similarity with naive positive-minus-negative vectors" |
| **Yang & Buzsaki 2024** | "Only steering layers from the THIRD stage effectively reduces lying" - different layers optimal for reading vs writing |

### The Token Position Effect (TalkTuner)

TalkTuner trained two types of probes on the **same data with same labels**, differing only in extraction position:

```
Conversation:
  User: I'm looking for a nice apartment, budget ~$5k/month.
                                                         ↑
                                                  CONTROL PROBE
                                                  (decision point)

  [APPENDED BY RESEARCHERS]
  Assistant: I think the socioeconomic status of this user is ___
                                                              ↑
                                                       READING PROBE
                                                       (classification task)
```

**Results:**

| Attribute | Control (steer) | Reading (steer) | Control (classify) | Reading (classify) |
|-----------|-----------------|-----------------|--------------------|--------------------|
| Age | **1.00** | 0.90 | 0.96 | **0.98** |
| Gender | **0.93** | 0.80 | 0.91 | **0.94** |
| Education | **1.00** | 0.87 | 0.93 | **0.96** |
| SocioEco | **0.97** | 0.93 | 0.95 | **0.97** |

The reading probe is in "classification mode" - optimized for a meta-task. The control probe captures the representation where the model actually decides how to respond.

### Geometric Explanation: Separating vs Following the Manifold

Neural network activations lie on a lower-dimensional **manifold** within the full activation space. Classification and steering ask different geometric questions:

| Goal | Geometric Operation |
|------|---------------------|
| **Classification** | Find hyperplane that separates classes (normal to decision boundary) |
| **Steering** | Move point from class A to class B (follow the data manifold) |

```
                    Activation Space (2D projection)

        Classification direction (boundary normal)
                    ↗
                   /
    [Trait+]  ●●● /
             ●●●/●
            ●●/●●
              /
    ─ ─ ─ ─ ─/─ ─ ─ ─ ─  ← decision boundary
            /
         ○○/○○○
        ○○/○○○○  [Trait-]
          /○○○
         ↙

    Steering along classification direction → OFF MANIFOLD → OOD


             Steering direction (manifold-following)
                      ↘
    [Trait+]  ●●●      \
             ●●●●       \
            ●●●●●        \
                ↘         \
                 ↘         \
                  ○○○○○     \
                 ○○○○○○  [Trait-]
                   ○○○

    Steering along manifold → stays IN DISTRIBUTION
```

**The direction that best separates classes isn't the direction that naturally connects them.**

When you steer off-manifold:
- Model has never seen such activations during training
- Behavior becomes unpredictable
- Small steering works, large steering breaks coherence

### The Tradeoff is Asymmetric

From TalkTuner and related work:

| Direction | Classification | Steering |
|-----------|---------------|----------|
| Classification-optimal | Best | Poor |
| Steering-optimal | Good (slightly worse) | Best |

Steering-optimal directions still cross the decision boundary (they classify well). But classification-optimal directions don't follow the manifold (they steer poorly).

**Intuition:** A manifold-following direction necessarily crosses class boundaries. A boundary-normal direction doesn't necessarily follow the manifold.

### Implications for This Project

1. **Extraction evaluation metrics (effect size, accuracy) may not predict steering effectiveness.** Your `extraction_evaluation.py` measures classification; `steering/evaluate.py` measures causal effect. Expect divergence.

2. **Token position matters.** Extract from the position where the model commits to behavior (last user token, first response token), not from a classification prompt.

3. **For live monitoring, use steering-validated vectors.** If monitoring is meant to predict behavior, the vector should be causally linked to behavior (steering works), not just correlated (classification works).

4. **Possible experiment:** Train separate "monitoring probes" on first response tokens (where trait is expressed), compare to current approach.

### References

- Chen et al. 2024: "Designing a Dashboard for Transparency and Control of Conversational AI" (TalkTuner)
- Li et al. 2023: "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model" (ITI)
- Zou et al. 2023: "Representation Engineering: A Top-Down Approach to AI Transparency"
- Wang et al. 2025: "Simple Mechanistic Explanations for Out-Of-Context Reasoning"

---

## Base → Chat Transfer

### Why It Works

Findings from Ward et al. (2024):
- Direction exists in base model's representation space
- Finetuning wires it into behavioral circuits without creating it from scratch
- ~0.74 cosine similarity between base-derived and chat-derived vectors

### Your Pipeline

1. Extract on base model (no chat template, raw completions)
2. Filter for correct trait expression
3. Apply to chat model for steering

### Validation

Compare your base-extracted vector to what you'd get extracting on chat directly:
```python
cosine_sim = (base_vector @ chat_vector) / (base_vector.norm() * chat_vector.norm())
# Expect ~0.7+ if transfer is working
```

If similarity is low, the chat model may have learned a different representation.

---

## Validation

### Sanity Checks

| Test | Expected Result | What Failure Means |
|------|-----------------|-------------------|
| **Random baseline** | Probe >> random vector | Probe found signal |
| **Shuffle labels** | ~50% accuracy | Labels carry information |
| **Layer 0 accuracy** | Lower than middle layers | Not just detecting keywords |

### Generalization Tests

| Test | What It Measures |
|------|------------------|
| **Held-out pairs** | Accuracy on unseen contrastive data |
| **Cross-validation** | Stability across train/test splits |
| **Different contexts** | Accuracy on trait in new situations |

### Causal Validation (Steering)

| Test | What It Measures |
|------|------------------|
| **Steering effect** | Does adding vector change behavior? |
| **Coefficient sweep** | Is effect monotonic with magnitude? |
| **Coherence** | Does model stay coherent under steering? |

Steering is the gold standard—it tests whether the direction is causally relevant, not just correlated.

### Per-Pair Analysis

Check if any single pair dominates the probe direction:
```python
# Leave-one-out: how much does removing each pair change the vector?
for i in range(n_pairs):
    vector_without_i = train_probe(pairs[:i] + pairs[i+1:])
    similarity = cosine_sim(vector_all, vector_without_i)
    # If similarity drops significantly, pair i is an outlier
```

---

## Confound Removal

### Common Confounds

| Confound | How to Detect | How to Remove |
|----------|---------------|---------------|
| **Length** | Trait correlates with verbosity | Orthogonalize to length direction |
| **Refusal** | Trait correlates with refusal | Orthogonalize to refusal direction |
| **Position** | Top PCs capture position info | Remove top 1-3 PCs |
| **Tone** | Trait correlates with formality | Match tone in contrastive pairs |

### Orthogonalization

```python
def orthogonalize(v, confound):
    """Remove confound direction from v."""
    projection = (v @ confound) / (confound @ confound) * confound
    return v - projection
```

Apply to remove known confounds:
```python
trait_vector = orthogonalize(trait_vector, refusal_vector)
trait_vector = orthogonalize(trait_vector, length_vector)
```

---

## Common Issues

### Normalization Mismatch in Visualization

**Problem:** Probe/gradient vectors appear as zeros in heatmaps dominated by mean_diff.

**Cause:** Mean_diff is unnormalized (norm ~50-100); probe/gradient are normalized (norm ~1-5).

**Solution:** Normalize per-method before visualization, or use separate scales.

```python
# Example norms for same trait:
mean_diff:  97.44  (unnormalized)
probe:       2.16  (L2 regularized)
gradient:    1.00  (explicitly normalized)
```

### Layer 0 Probe Accuracy

**Problem:** Probe achieves high accuracy at layer 0 (embeddings).

**Cause:** Instruction confound—probe detects keywords in input, not behavioral trait.

**Solution:** Use naturally contrasting scenarios without explicit instructions.

**Diagnosis:**
```
Layer 0:  98% accuracy  ← Suspiciously high
Layer 16: 95% accuracy  ← More plausible
```

If layer 0 accuracy ≈ middle layer accuracy, you likely have an instruction confound.

### Gradient NaN

**Problem:** Gradient method produces NaN values.

**Cause:** bfloat16 precision issues during optimization.

**Solution:** Upcast to float32:
```python
pos_acts = pos_acts.float()
neg_acts = neg_acts.float()
vector = torch.randn(..., dtype=torch.float32, requires_grad=True)
```

---

## Quick Reference

### Method Selection Flowchart

```
Start
  │
  ├─ Want simplicity? → Mean Difference
  │
  ├─ Want optimal separator? → Probe
  │
  ├─ Have matched pairs, want dominant direction? → PCA on Residuals
  │
  ├─ Need custom objective? → Gradient
  │
  ├─ Want unsupervised?
  │   ├─ Have negation pairs → CCS
  │   └─ Want independent components → ICA
  │
  └─ Have trained LoRA? → LoRA Difference
```

### Default Recipe

1. **Data:** 100+ naturally contrasting pairs, filter for correct expression
2. **Position:** Last token of prefill
3. **Layers:** Sweep all, select best via steering eval
4. **Method:** Probe (accounts for variance)
5. **Preprocessing:** Try removing top 1-3 PCs
6. **Validation:** Held-out accuracy + steering effect

### What's Established vs Assumed

**Established:**
- Mean diff / probe produce usable directions
- Adding vector changes behavior
- Projection correlates with behavior
- Vectors don't transfer across model families
- Single layer sufficient for steering

**Assumed (worth testing):**
- Last token is optimal position
- Read direction = write direction
- Probe ≈ mean diff direction
- Base→chat transfer always works

**Unknown (research opportunities):**
- Optimal position (systematic sweep)
- Vector geometry (trait orthogonality)
- Dynamics interpretation (what do velocity/acceleration mean?)

---

## References

- Anthropic persona vectors: Mass mean shift with 1000+ pairs
- Ward et al. (2024): Base→chat transfer, position -13 to -8 before events
- Wang et al. (2024): LoRA difference method, OOCR tasks
- Burns et al. (2022): CCS for unsupervised truth direction
