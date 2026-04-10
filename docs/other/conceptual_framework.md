# Conceptual Framework

Mental models and theoretical foundations for understanding trait vectors in language models.

---

## Two Spaces

### Weight Space
- **Dimensions**: Billions (one per parameter)
- **A point**: One complete model configuration
- **Training**: Walks through this space, descending loss landscape
- **Visit pattern**: Once during training, then frozen

### Activation Space
- **Dimensions**: Thousands (e.g., 2304 for Gemma 2B hidden dim)
- **A point**: One hidden state vector
- **Inference**: Trajectories flow through this space
- **Visit pattern**: Every forward pass, every token

**Key insight**: There is ONE activation space. Each point in weight space defines a DIFFERENT vector field on that same activation space.

> "Weights are the physics, activations are where the physics plays out."

---

## What Training Carves

| Stage | Data | What it creates |
|-------|------|-----------------|
| Pre-training | Internet text (trillions of tokens) | Valid language manifold — region of activation space where coherent text lives |
| SFT | Instruction-response pairs | Instruction-following channels within that manifold |
| RLHF/RLAIF | Preference rankings | Behavioral sub-channels (refusal, helpfulness, style) |

Pre-training gets the model into the "competent language" basin. SFT/RLHF carve specific valleys within that basin.

Trait directions exist because training made them linear — RLHF creates consistent behavioral patterns that manifest as directions in activation space.

---

## Layer-wise Computation

### Inter-layer Dynamics

| Depth | What's computed | Trait status |
|-------|-----------------|--------------|
| 0-5 | Token identity, syntax, surface features | Traits not yet represented |
| 6-12 | Semantic composition, meaning | Traits emerging |
| 13-18 | Abstract concepts, behavioral decisions | Traits crystallized (extract here) |
| 19-25 | Output formatting, token prediction | Traits locked, translating to vocabulary |

### Intra-layer Components

```
residual_in → Attention → + → post-attn → MLP → + → residual_out
```

| Component | Function | Trait relevance |
|-----------|----------|-----------------|
| Attention | Routes information between positions | "Which tokens should influence this decision?" |
| MLP | Transforms features, activates concepts | "This situation matches the refusal pattern" |

Attention moves information; MLP computes on it.

**Low attention ≠ ignoring**: By layer 16, information from early tokens is already encoded in hidden states. Attention is routing, not the only information pathway. Don't over-interpret attention weight diffusion.

---

## The Vector Field View

Each layer defines a transformation: `h_out = f(h_in)`. Stack 26 of these and you get the full forward pass.

**Intuition from fluid dynamics**:
- Activation space has an implicit "flow"
- Prompt embedding = source (where particles enter)
- Output logits = sink (where particles exit)
- Each token's hidden state is a particle flowing through layers

**What transfers from classical vector field theory**:
- Divergence → representations spreading or focusing
- Streamlines → trajectories through activation space
- Flow visualization → useful for understanding dynamics
- Amplitwist → each layer applies local rotation + scaling (from Needham's Visual Complex Analysis)

**What doesn't transfer**:
- No attractors — transformers are feedforward, no feedback loops
- Discrete steps, not continuous flow
- ~2000 dimensions, not 2-3
- No conservation laws

---

## Vectors as Signposts, Not Forces

Critical distinction:

**Measurement (projection)**:
```python
score = activation @ trait_vector / (norm(activation) * norm(trait_vector))
```
This measures which trajectory the model is on. The vector doesn't push anything — it's a compass reading.

**Intervention (steering)**:
```python
activation_new = activation + strength * trait_vector
```
This teleports the particle to a new location. Downstream layers receive different input.

The trait vector is a **signpost** indicating direction, not a **force** pushing activations. Projection = measurement. Addition = intervention. Different operations.

---

## KV-Cache and the Changing Field

The vector field isn't static during generation:

```
field at token t = f(weights, KV_cache[1:t-1])
```

Each new token changes the field for all future tokens because:
- New K,V entries are added to the cache
- Attention patterns shift
- Information routing changes

This is why trait dynamics change token-by-token — the "physics" itself evolves.

---

## Causal Sequentiality

Transformers are fundamentally myopic:

**No backtracking**: Once token j is generated, it's frozen. The model cannot revise earlier tokens based on later computation. Errors accumulate forward.

**Computational asymmetry**: Later tokens have more computation available.
- Token 1: 26 layers × ~0 context positions
- Token 50: 26 layers × 49 context positions worth of attention

**Implication**: Early token trait projections are inherently noisier — the model hasn't "thought" as much yet. Don't weight early tokens equally with late tokens when analyzing trait dynamics.

---

## High-Dimensional Geometry

Intuitions that matter:

**Near-orthogonality**: Random vectors in high-d are nearly perpendicular.
- Cosine similarity variance ≈ 1/d
- For d=2304, unrelated trait projections cluster around zero
- ~50% accuracy on unrelated traits is geometric expectation, not failure

**Volume concentration**: Most volume is near the surface.
- Fraction in outer shell ≈ 1 - (1-ε/r)^d → nearly 1 for large d
- All data points are "near the boundary"
- The interior is essentially empty

**Distance concentration**: All pairwise distances become similar.
- Nearest neighbors aren't that near
- Farthest points aren't that far
- Clustering becomes harder

**Implication**: 3D intuition lies. Accept mathematical understanding over visual intuition.

---

## Linear Representation Hypothesis

**Claim**: Concepts are represented as directions in activation space.

**Evidence for**:
- Trait vectors work (high classification accuracy)
- Steering produces expected behavior changes
- Cross-distribution transfer (vectors generalize)

**Evidence against / caveats**:
- Some features are multi-dimensional, not single directions (Engels et al., 2024)
- High accuracy doesn't prove the direction is "the" representation
- Probes find signal everywhere, even causally irrelevant locations

**Our position**: Linear directions are useful approximations. Whether they're "the true representation" is less important than whether they enable monitoring and understanding.

---

## Steering: Understanding, Not Control

From Axebench evaluation:

> "Steering is nowhere near state-of-the-art for control. Prompting and fine-tuning are clearly more successful."

**Implication**: Don't oversell steering as a control mechanism. Its value is:
- **Monitoring**: See what the model is "thinking" during generation
- **Debugging**: Trace where behavior goes wrong
- **Understanding**: Study how traits interact and crystallize

Frame trait vectors as measurement tools, not behavioral controllers.

---

## Non-Monotonicity Means Conflict

If traits increased monotonically across layers, there'd be a single "trait potential" being optimized. But they don't.

Non-monotonic trait trajectories indicate:
- Different circuits pushing different directions
- Competition between behavioral patterns
- Layer-specific computations that temporarily oppose the final output

**What to look for**:
- Zero-crossings in trait velocity
- Which layers/components cause reversals
- Correlation between oscillation and steerability

---

## The Core Loop

1. **Prompt** → starting position in activation space
2. **Layers** → deterministic flow (26 discrete transformations)
3. **Trait vector** → compass direction (not force)
4. **Projection** → checking heading at each step
5. **Steering** → teleporting particle, then letting it flow

---

## Commitment Points

When does the model "lock in" a decision?

**Operationalization**: Token where trait acceleration drops to near-zero.

**Interpretation**: The trajectory has entered a pre-carved channel. Not convergence to an attractor (transformers don't have those), but selection of a path that subsequent layers will follow.

**Caveat**: This is a hypothesis, not validated. Whether single commitment points exist per trait, optimal thresholds, and correlation with actual behavior change — all need testing.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Two spaces | Weights define the physics, activations are where it plays out |
| Training | Carves channels in activation space (language → instructions → behaviors) |
| Layers | Early = syntax, middle = semantics + traits, late = output |
| Vector field | Useful intuition (amplitwist), but discrete/high-d/no attractors |
| Trait vectors | Signposts for measurement, not forces |
| KV-cache | Makes the field token-dependent |
| Causal sequentiality | No backtracking; later tokens have more computation |
| High-d geometry | Near-orthogonal, surface-concentrated, distance-concentrated |
| Linear hypothesis | Useful approximation, not proven ground truth |
| Steering | For understanding, not control |
| Non-monotonicity | Indicates computational conflict |
| Commitment | Trajectory enters channel, needs validation |
