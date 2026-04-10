# Mathematical Foundations

Reference for the math behind trait vector extraction, monitoring, and steering. Covers both the codebase implementation and the broader field.

---

## Linear Representation Hypothesis

Everything rests on one claim: **concepts are directions in activation space**. Not individual neurons, but linear combinations of neurons.

For a concept C and hidden state **h** in R^d, there exists a direction **v_C** such that:
- `h . v_C` high = concept C is active
- `h . v_C` low = concept C is inactive

**Why this would be true**: the residual stream in transformers is a shared communication channel. Each layer reads from it and writes additive updates. This additive structure naturally creates linear representations — if attention head #7 writes "this is harmful" as a direction, later layers can read it via dot product.

Park et al. (2024) formalized this with a "causal inner product" showing concepts that vary independently tend to be ~90% orthogonal even under standard Euclidean geometry.

---

## Data Pipeline: Text to Training Data

The extraction pipeline converts text scenarios into the matrices the probe trains on.

### Step 1: Scenario files
Raw text in `datasets/traits/{category}/{trait}/positive.txt` and `negative.txt`. Each line is a prompt that naturally elicits the trait (positive) or its absence (negative). ~30 per side.

### Step 2: Generate responses (`generate_responses.py`)
Each scenario is a prompt. The model generates a completion. Now you have 30 positive (prompt, response) pairs and 30 negative pairs.

### Step 3: Extract activations (`extract_activations.py`)
For each (prompt, response) pair:

1. **Tokenize**: `full_text = formatted_prompt + response` -> `[token_1, ..., token_N]`
2. **Forward pass** with `MultiLayerCapture` hook — captures hidden states at every layer for every token
3. **Select tokens**: with `position='response[:5]'`, take first 5 response tokens and **mean** them:
   ```
   selected = acts[b, start:end, :]   # [5, 2304]
   act_out = selected.mean(dim=0)     # [2304]
   ```
4. **Stack across layers and examples**:
   ```
   pos_all:    [30, 26, 2304]   (30 examples, 26 layers, 2304 hidden dim)
   neg_all:    [30, 26, 2304]
   train_acts: [60, 26, 2304]   (concatenated, first 30 are positive)
   ```

### Step 4: Train probe per layer (`extract_vectors.py` -> `methods.py`)
For each layer independently:
```
layer_acts = all_acts[:, layer_idx, :]   # [60, 2304]
pos_acts = layer_acts[:30]               # [30, 2304]
neg_acts = layer_acts[30:]               # [30, 2304]
probe.fit(X, y)                          # X=[60, 2304], y=[1,...,1,0,...,0]
vector = probe.coef_[0] / ||coef_[0]||  # [2304] unit vector
```

Result: one trait vector per layer, per method.

---

## Extraction Methods

All three methods answer the same question: "what direction in activation space separates positive from negative examples?" They optimize different objectives.

### Mean Difference (`core/methods.py:32-47`)
```
v = mean(h+) - mean(h-)
v_hat = v / ||v||
```

Points from the centroid of negative examples toward the centroid of positive examples. Simplest approach.

**Vulnerability**: massive activation dimensions. If dim 443 has values ~32,000 while informative dims have ~1-5, tiny noise differences in dim 443 dominate the mean difference vector even though they carry no trait signal. This is why mean_diff vectors were 86% spurious on Gemma-3.

Implementation computes in float32 (line 36-38) because bfloat16 has step size 256 at magnitude 32,000 and can't represent the differences.

### Logistic Probe (`core/methods.py:50-91`)
Train logistic regression, extract weight vector.

```
P(positive | h) = sigmoid(w . h + b)
```

Minimize binary cross-entropy, then use **w** (normalized) as the direction.

**Why probe is robust to massive dims**: logistic regression finds **w** via gradient-based optimization. For dim 443 to get weight, it needs to *consistently* predict the label across all examples. Massive dims encode position/context (similar values for both classes), so the gradient for w[443] points in random directions across examples and averages to ~zero. The optimizer naturally ignores dims with no reliable class signal. Mean diff blindly computes mean differences and has no such mechanism.

See [Logistic Regression Deep Dive](#logistic-regression-deep-dive) below.

### Gradient Optimization (`core/methods.py:94-138`)
Directly optimize the vector to maximize separation:
```
loss = -(mean(h+ . v_hat) - mean(h- . v_hat)) + lambda * ||v||
```

Using Adam (lr=0.01, 100 steps).

**Vs probe**: gradient maximizes *raw separation* (distance between class means along v). Probe maximizes *probabilistic separation* (log-likelihood). Gradient doesn't account for within-class variance; probe implicitly does through the sigmoid.

**When gradient wins**: finds layer-specific directions (0.40 cross-layer cosine similarity vs mean_diff's 0.89). Achieved 1.9x better steering on refusal_v2. Better for low-separability traits.

### Relationship between methods

All methods find v such that `v . (h+ - h-)` is maximized, under different constraints:
- **Mean diff**: directly computes `mean(h+) - mean(h-)`
- **Probe**: finds hyperplane maximizing `P(y | h)`
- **Gradient**: optimizes separation via backprop
- **PCA** (RepE, not in codebase): finds direction of max variance in differences

They diverge when there are confounds (massive dims, topic leakage). Method rankings flip between classification and steering — best classifier != best steering vector.

---

## Logistic Regression Deep Dive

### Why sigmoid
The sigmoid function is the smooth, differentiable version of a step function:
```
sigmoid(z) = 1 / (1 + e^(-z))
```
- sigmoid(0) = 0.5 (boundary = max uncertainty)
- sigmoid(large +) -> 1
- sigmoid(large -) -> 0
- Derivative: sigmoid'(z) = sigmoid(z)(1 - sigmoid(z)) — always nonzero

The model says: probability of positive is a smooth function of signed distance from the hyperplane.

### Why cross-entropy loss
Maximum likelihood estimation. The probability of observing the dataset:
```
P(data | w, b) = Product_i sigmoid(z_i)^y_i * (1 - sigmoid(z_i))^(1-y_i)
```
where `z_i = w . h_i + b`.

Taking negative log:
```
L = -Sum_i [y_i * log(sigmoid(z_i)) + (1 - y_i) * log(1 - sigmoid(z_i))]
```

This is **cross-entropy loss** (binary CE). Not an arbitrary choice — it's the unique loss for maximum likelihood with a logistic model.

### Variants of CE

**Focal loss**: `L = -(1 - p_correct)^γ · log(p_correct)`

Downweights easy examples. When confident (p=0.9), the `(1-p)^γ` multiplier is tiny so loss ~0. When uncertain (p=0.5), multiplier is larger. Effect: gradient signal comes from hard cases, hedging is penalized less. Standard in object detection (RetinaNet), rare in LLM training. Relevant to emergent misalignment — CE's hedging penalty may explain why conditional policies ("evil if medical") lose to unconditional ("always evil"); focal loss would reduce that asymmetry. See Turner et al. Jul 2025 in relevant_papers.md.

**Label smoothing**: target `[0, 0, 1, 0]` becomes `[ε/V, ε/V, 1-ε, ε/V]` where ε≈0.1, V=vocab size.

Loss becomes `L = -(1-ε)·log(p_correct) - ε/V · Σ log(p_other)`. The model gets credit for not being maximally confident — 95% scores as well as 100% would under plain CE. A form of regularization (same family as dropout, weight decay). Common in LLM training unlike focal loss.

### Why not MSE
MSE gradient: `dMSE/dz = 2(sigmoid(z) - y) * sigmoid'(z)`. When the model is maximally wrong (sigmoid(z) ~ 0, y = 1), sigmoid'(z) ~ 0 and the gradient vanishes — the model can't learn from its worst mistakes.

CE gradient: `dCE/dz = sigmoid(z) - y`. When maximally wrong, gradient ~ -1 (large). **Gradient magnitude is proportional to how wrong you are.** No saturation.

### Why logistic regression over other linear classifiers
Any linear classifier produces a hyperplane whose normal vector is a valid trait direction. Logistic regression specifically because:

- **Probabilistic interpretation**: `w` is the gradient of the log-odds: `d/dh [log(P(pos)/P(neg))] = w`. Moving along **w** increases log-odds of positive as fast as possible. Natural definition of "trait direction."
- **Convex optimization**: CE + L2 is strictly convex — exactly one global minimum, guaranteed to find it. No local optima.
- **Per-example confidence**: `predict_proba()` gives P(positive | h) for each example, enabling outlier detection (see below).

SVM finds the max-margin hyperplane (optimizes worst-case separation). Logistic regression optimizes average-case likelihood. For trait extraction, we want the overall trend, not direction dictated by borderline examples.

### Regularization (C parameter)
Actual loss minimized: `L = (1/2C) * ||w||^2 + Sum CE(y_i, sigmoid(w . h_i + b))`

- C = infinity: no regularization, fit training data exactly
- C = 1.0: balanced (our default)
- L2 penalty prevents overfitting with ~60 examples in 2304 dimensions
- Controls confidence: scaling w by 10x doesn't change the hyperplane but makes sigmoid sharper. L2 penalizes this.

### Row normalization (`methods.py:68-72`)
```python
X_normalized = X / ||X||_rows
```

Projects all data points onto the unit hypersphere before classification. The probe learns from *directions*, not magnitudes.

**Primary benefit**: solver conditioning and cross-model comparability. Makes probe coefficients ~1 magnitude regardless of model (Gemma-3 activations are ~170x larger than Gemma-2). Improves LBFGS convergence by reducing condition number.

**Does NOT directly fix massive dims** — after normalization, the massive dim is ~1.0 while informative dims are ~1/32000. The probe's robustness to massive dims comes from the optimization process itself (see above), not row normalization.

### Outlier detection via probe probabilities
Since the probe outputs P(positive | h) per example, you can identify:
- **Misclassified examples**: P(true_label) < 0.5
- **Ambiguous examples**: P close to 0.5
- **High-confidence correct**: P(true_label) > 0.9

This is an activation-level quality score, complementary to the text-based LLM judge vetting. An example could look sycophantic in text but not activate the direction internally (surface mimicry vs deep representation).

---

## Projection Operations

Once you have a direction **v**, measure "how much of this trait is in activation **h**?"

### Normalized projection (`core/math.py:51-77`)
```
score = h . v_hat    where v_hat = v / ||v||
```
Equals `||h|| * cos(theta)`. Captures both alignment (cos theta) and magnitude (||h||). A spike could mean "more aligned" or "larger activation" — both informative during generation.

**Default for monitoring.**

### Cosine similarity (`core/math.py:80-105`)
```
cos(theta) = (h . v) / (||h|| * ||v||)
```
Range [-1, 1]. Pure directional alignment, ignoring magnitude.

**Used for comparing directions** (trait similarity matrix, cross-layer comparison).

### When to use which
- **Monitoring** (per-token during generation): normalized projection. Magnitude matters — it signals salience.
- **Comparing directions** (across layers, models, traits): cosine similarity. Only direction matters.
- **Classification** (is trait present?): either works. Cosine is theoretically purer.

---

## Steering: Causal Intervention

### The operation (`core/hooks.py:285-328`)
During forward pass at layer L:
```
h' = h + alpha * v
```

### Why this works
The residual stream flows directly to output: `logits = W_unembed . h_final`. Adding alpha*v shifts h along the trait direction, shifting the distribution over next tokens.

### Perturbation ratio
```
perturbation_ratio = (alpha * ||v||) / ||h||
```

How big the intervention is relative to the existing signal:

| Ratio | Coherence | Trait Effect |
|-------|-----------|-------------|
| 0.3-0.5 | ~90% | Weak (+3) |
| 0.8-1.0 | ~69% | Balanced (+8) |
| 1.0-1.3 | ~54% | Strong (+25) |
| >1.3 | Collapse | Broken |

The ~1.15 cliff: when perturbation is comparable to the entire accumulated signal, downstream layers receive something outside their training distribution.

### Ablation (`core/hooks.py:335-393`)
Remove a direction: `h' = h - (h . r_hat) * r_hat`

Orthogonal projection — removes the component of h along r, keeps everything perpendicular. Equivalent to steering with dynamic coefficient `-(h . r_hat)`.

---

## Evaluation Metrics

### Separation (`core/math.py:118-120`)
```
sep = |mean(pos_scores) - mean(neg_scores)|
```
Raw distance between distribution centers. Simple but ignores variance.

### Cohen's d (`core/math.py:132-144`)
```
d = (mean_pos - mean_neg) / sqrt((var_pos + var_neg) / 2)
```
Standardized separation — how many pooled standard deviations apart. d=0.8 is "large" in social science; good trait vectors hit d > 2.0.

Why better than raw separation: two distributions can be far apart but have huge variance (lots of overlap), or close but very tight (no overlap). Cohen's d captures this.

### Balanced accuracy (`core/math.py:123-129`)
```
threshold = (mean_pos + mean_neg) / 2
accuracy = (mean(pos > threshold) + mean(neg <= threshold)) / 2
```
Averages positive and negative accuracy separately. Prevents class imbalance from inflating scores.

### Length-normalized log probability (`analysis/steering/logit_difference.py:104-140`)
```
score(completion) = (1/n) * Sum log P(token_i | context)
```
Why length-normalize: "Yes" (1 token) would always beat "I'd be happy to help" (8 tokens) — shorter completions naturally have higher total probability.

### Preference-Utility decomposition (`analysis/steering/preference_utility_logodds.py`)
From Xu et al. "Why Steering Works":
```
PrefOdds = L_neg - L_pos           (does it prefer positive text?)
UtilOdds = log(p_pos + p_neg) / (1 - p_pos - p_neg)   (is it still coherent?)
```
Key decomposition: steering success = preference shift * coherence preservation.

---

## Token Position Considerations

### What the mean does
`response[:5]` takes first 5 response tokens and averages their hidden states. This blurs information:
- **Token 0** (first generated): the model's "decision point" — strongest trait signal
- **Token 3-4**: increasingly about local word choice, weaker trait signal

### Alternatives to averaging
- **Single token** (`response[0]` or `prompt[-1]`): captures a specific computational moment. `prompt[-1]` (Arditi-style) sometimes outperforms averaging because it's the model's state right before committing to a response direction.
- **Per-position probes**: extract `response[0]`, `response[1]`, etc. separately. Compare probe accuracy to find which position carries most trait signal.
- **All tokens as training examples**: 30 examples * 5 tokens = 150 training points. More data but violates independence — tokens from the same response are correlated. Inflates probe confidence.

---

## Model Comparison Techniques

Not yet implemented. Reference for future use.

- **Activation diff**: `cos(mean(h_B - h_A), v_trait)` — how much of the activation shift between models aligns with a trait direction. Already in `compare_variants.py`.
- **Weight-space SVD**: `U, S, Vt = SVD(W_B - W_A)` — top singular vectors are the dominant behavior directions added by fine-tuning. `cos(v_trait, U[:, k])` checks if a trait aligns with them.
- **Affine mapping**: `h_B = A·h_A + b` — learned linear transform between two models' activation spaces. Formalizes how representations shift (rotation A, translation b).
- **CKA** (Centered Kernel Alignment): similarity between two models' representations of the same inputs. Layer-wise CKA profile shows where models diverge most.
- **Task vectors**: `θ_ft - θ_base` in weight space encodes a capability. Can be added, negated, composed across models.

---

## References

- Park, Choe, Veitch (2024) — "The Linear Representation Hypothesis and the Geometry of Large Language Models"
- Rimsky et al. (2024) — "Steering Llama 2 via Contrastive Activation Addition"
- Zou et al. (2023) — "Representation Engineering: A Top-Down Approach to AI Transparency"
- Li et al. (2023) — "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"
- Turner et al. (2023) — "Activation Addition: Steering Language Models Without Optimization"
- Xu et al. — "Why Steering Works" (preference-utility decomposition)
- Sun et al. (2024) — Massive activations in LLMs
