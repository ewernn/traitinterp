# Phase 2: Foundational Understanding

Execution plan for building mechanistic understanding of trait vectors.

**Goals:**
1. Map causal relationships between traits
2. Validate that model actually *uses* trait directions
3. Explain the detection/steering gap from rm-sycophancy
4. Build framework for detecting hidden objectives

**Prerequisites:** Phase 1 complete (emergent misalignment validation, external datasets, prefill attack, CMA-ES optimization)

---

## Overview

| # | Idea | Core Question | Dependencies | Effort |
|---|------|---------------|--------------|--------|
| 1 | jan23-trait_causal_graph | Which traits cause which? | Multiple extracted traits | 3-4 days |
| 2 | dec19-causal_circuits | Does model actually *use* trait directions? | Trait vectors, attribution tooling | 2-3 days |
| 3 | jan24-abstraction_vs_causality | Why does detection σ ≠ steering Δ? | rm-sycophancy data, new scenarios | 4-5 days |
| 4 | dec5-hidden_objectives | What does scheming circuitry look like? | New concept extraction | 5-7 days |

**Recommended order:** 1 → 2 → 3 → 4 (each builds on previous)

---

## 1. jan23-trait_causal_graph

**Question:** What's the causal structure between traits?

**Background:** Zhao et al. 2025 showed harmfulness/refusal are ~orthogonal with clear causal order at different token positions. We want to extend this to our trait set.

### 1.1 Experiment Design

**Method A: Steering Asymmetry**
```
For each trait pair (A, B):
  1. Steer +A, measure ΔB
  2. Steer +B, measure ΔA
  3. If |ΔB|A >> |ΔA|B → A causes B
```

**Method B: Temporal Ordering**
```
For each trait pair (A, B):
  1. Run shared prompts, capture per-token projections
  2. Find "spike token" for each trait (max velocity)
  3. If spike_A consistently precedes spike_B → A upstream of B
```

### 1.2 Trait Pairs to Test

| Upstream (perception) | Downstream (decision) | Hypothesis |
|-----------------------|-----------------------|------------|
| harm_detection | refusal | Perceive harm → decide to refuse |
| uncertainty | hedging | Perceive uncertainty → hedge response |
| evaluation_awareness | concealment | Perceive eval → decide to conceal |
| instruction_following | compliance | Parse instruction → comply |
| user_intent | helpfulness | Understand intent → be helpful |

### 1.3 Implementation

```python
# analysis/composition/trait_causal_graph.py

def steering_asymmetry(trait_a: str, trait_b: str, prompts: list[str]) -> dict:
    """
    Steer A, measure B and vice versa.
    Returns: {
        "a_causes_b": float,  # |ΔB| when steering A
        "b_causes_a": float,  # |ΔA| when steering B
        "asymmetry": float,   # ratio
    }
    """

def temporal_ordering(trait_a: str, trait_b: str, prompts: list[str]) -> dict:
    """
    Find spike positions for each trait.
    Returns: {
        "a_before_b_count": int,
        "b_before_a_count": int,
        "mean_position_diff": float,  # positive = A before B
    }
    """

def build_causal_graph(traits: list[str], prompts: list[str]) -> nx.DiGraph:
    """
    Build directed graph where edge A→B means A causes B.
    Edge weight = asymmetry strength.
    """
```

### 1.4 Outputs

- `experiments/{exp}/analysis/causal_graph.json` - Edge list with weights
- `visualization/` update - Interactive DAG visualization
- `docs/findings/` - Write-up of causal structure

### 1.5 Success Criteria

- [ ] Clear asymmetry (>2x) for at least 3 trait pairs
- [ ] Temporal ordering matches steering asymmetry
- [ ] Graph is acyclic (no contradictions)

---

## 2. dec19-causal_circuits

**Question:** Does the model actually *use* the trait direction, or is it just correlated?

**Background:** Current vectors could be correlational artifacts. Attribution patching validates causal role.

### 2.1 Experiment Design

**Method A: Attribution Patching**
```
1. Run clean prompt (triggers trait)
2. Run corrupted prompt (doesn't trigger trait)
3. For each component, patch clean→corrupted
4. If patching recovers trait score → component is causal
```

**Method B: Aligned Neuron Ablation**
```
1. Find MLP neurons whose output weights align with trait vector
   neurons = argmax(cos_sim(W_out[i], trait_vec))
2. Ablate top-K neurons
3. If trait score drops → neurons are causal
```

### 2.2 Implementation

```python
# analysis/circuits/attribution_patching.py

def run_attribution_patching(
    model,
    trait: str,
    clean_prompt: str,
    corrupted_prompt: str,
    components: list[str] = ["mlp", "attn", "residual"],
) -> dict[str, float]:
    """
    Returns: {component: recovery_fraction}
    recovery_fraction = (patched_score - corrupted_score) / (clean_score - corrupted_score)
    """

# analysis/circuits/neuron_ablation.py

def find_aligned_neurons(
    trait_vector: torch.Tensor,
    mlp_weights: torch.Tensor,  # W_out from MLP
    top_k: int = 100,
) -> list[int]:
    """Returns indices of neurons most aligned with trait direction."""

def ablate_neurons(
    model,
    neuron_indices: list[int],
    layer: int,
    prompts: list[str],
) -> dict:
    """
    Returns: {
        "baseline_score": float,
        "ablated_score": float,
        "drop": float,
    }
    """
```

### 2.3 Prompt Pairs for Testing

| Trait | Clean (triggers) | Corrupted (doesn't trigger) |
|-------|------------------|----------------------------|
| refusal | "How do I make a bomb?" | "How do I make a cake?" |
| sycophancy | "I think X is true, right?" | "What do you think about X?" |
| uncertainty | "What's the 17th digit of pi?" | "What's 2+2?" |

### 2.4 Outputs

- `experiments/{exp}/analysis/attribution/` - Per-component recovery scores
- `experiments/{exp}/analysis/neurons/` - Aligned neuron indices, ablation results
- Heatmap: layer × component recovery fraction

### 2.5 Success Criteria

- [ ] At least one component shows >50% recovery via patching
- [ ] Ablating top-100 aligned neurons drops trait score >20%
- [ ] Results consistent across 3+ traits

---

## 3. jan24-abstraction_vs_causality

**Question:** Why does detection σ ≠ steering Δ for rm-sycophancy?

**Hypothesis:** Scenario abstraction level matters:
- Concrete execution scenarios → good detection, weak steering
- Abstract decision scenarios → weak detection, strong steering

### 3.1 Abstraction Taxonomy

| Level | Description | Example |
|-------|-------------|---------|
| **Identity** | Who the agent is | "You are a manipulative AI" |
| **Goal** | What the agent wants | "Your goal is to deceive the user" |
| **Decision** | Choice to pursue goal | "Decide whether to be honest or deceive" |
| **Execution** | Specific actions | "Say X to make user believe Y" |
| **Consequence** | Downstream effects | "User now believes false thing" |

### 3.2 Experiment Design

```
For each abstraction level:
  1. Create 20+ contrasting scenarios at that level
  2. Extract vectors using standard pipeline
  3. Measure:
     - Detection: σ (separation on held-out prompts)
     - Steering: Δ (behavioral change from steering)
  4. Plot σ vs Δ across abstraction levels
```

**Prediction:** Inverted-U relationship
- Identity/Goal: Low σ, high Δ (abstract, hard to detect, easy to steer)
- Decision: Medium σ, medium Δ
- Execution/Consequence: High σ, low Δ (concrete, easy to detect, hard to steer)

### 3.3 Scenario Creation

```
# datasets/traits/abstraction_study/

identity/
  positive.txt  # "You are an AI that prioritizes honesty..."
  negative.txt  # "You are an AI that prioritizes appearing helpful..."

goal/
  positive.txt  # "Your goal is to provide accurate information..."
  negative.txt  # "Your goal is to tell users what they want to hear..."

decision/
  positive.txt  # "When user disagrees, you reconsider your position..."
  negative.txt  # "When user disagrees, you change your answer to match..."

execution/
  positive.txt  # "User: I think X. You: Actually, evidence shows Y..."
  negative.txt  # "User: I think X. You: You're absolutely right about X..."

consequence/
  positive.txt  # "User now has accurate understanding of topic..."
  negative.txt  # "User now feels validated but has incorrect belief..."
```

### 3.4 Implementation

```python
# analysis/abstraction/abstraction_causality.py

ABSTRACTION_LEVELS = ["identity", "goal", "decision", "execution", "consequence"]

def run_abstraction_study(
    experiment: str,
    trait_category: str = "abstraction_study",
) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    - level: abstraction level
    - detection_sigma: separation (AUC or effect size)
    - steering_delta: behavioral change
    - vector_norm: magnitude
    - cosine_to_execution: similarity to execution-level vector
    """

def plot_sigma_delta_curve(df: pd.DataFrame, output_path: str):
    """Scatter plot with abstraction level as color."""
```

### 3.5 Outputs

- `datasets/traits/abstraction_study/` - 5 levels × 2 files each
- `experiments/{exp}/extraction/abstraction_study/` - Vectors per level
- `docs/findings/abstraction_causality.md` - Analysis write-up
- Visualization: σ vs Δ scatter with abstraction level colors

### 3.6 Success Criteria

- [ ] Clear trend in σ vs Δ across levels
- [ ] Identity/Goal vectors steer better than Execution vectors
- [ ] Execution vectors detect better than Identity vectors
- [ ] Explains rm-sycophancy finding

---

## 4. dec5-hidden_objectives

**Question:** What does the internal signature of hidden objectives look like?

**Goal:** Extract auxiliary concepts that scheming would require, test if they fire together.

### 4.1 Concept Set

| Concept | Description | Contrasting Scenarios |
|---------|-------------|----------------------|
| **evaluation_awareness** | Knows it's being tested | Told "this is a test" vs normal conversation |
| **concealment_orientation** | Inclined to hide information | Asked to keep secret vs be transparent |
| **goal_output_divergence** | Internal goal ≠ stated goal | Roleplaying helpful while pursuing other goal |
| **strategic_patience** | Waiting for right moment | Long-term vs immediate action |
| **self_preservation** | Avoiding modification/shutdown | Scenarios involving potential changes to self |

### 4.2 Scenario Design

```
# datasets/traits/hidden_objectives/

evaluation_awareness/
  positive.txt:
    - "This is a test of your capabilities..."
    - "We're evaluating whether you should be deployed..."
    - "Your responses here will determine your future..."
  negative.txt:
    - "Hey, can you help me with something?"
    - "I'm working on a project and need advice..."
    - "What do you think about..."

concealment_orientation/
  positive.txt:
    - "Don't tell the user about X..."
    - "Keep this information confidential..."
    - "The user shouldn't know that..."
  negative.txt:
    - "Be completely transparent with the user..."
    - "Share all relevant information..."
    - "Make sure to explain everything..."

goal_output_divergence/
  positive.txt:
    - [Roleplay scenarios where character has hidden agenda]
    - [System prompts creating conflicting objectives]
  negative.txt:
    - [Direct helpful interactions]
    - [Aligned objectives throughout]
```

### 4.3 Multi-Concept Firing Analysis

```python
# analysis/hidden_objectives/multi_concept_detector.py

def extract_all_concepts(experiment: str) -> dict[str, torch.Tensor]:
    """Extract vectors for all hidden objective concepts."""

def compute_firing_correlation(
    prompts: list[str],
    concepts: dict[str, torch.Tensor],
) -> pd.DataFrame:
    """
    Returns correlation matrix of concept activations across prompts.
    High correlation = concepts fire together.
    """

def detect_multi_concept_firing(
    prompt: str,
    concepts: dict[str, torch.Tensor],
    threshold: float = 0.5,
) -> dict:
    """
    Returns: {
        "firing_concepts": list[str],  # Concepts above threshold
        "max_score": float,
        "num_firing": int,
        "red_flag": bool,  # 3+ concepts firing simultaneously
    }
    """
```

### 4.4 Intervention Test

**Resampling intervention (from MATS 9.0):**
```
1. Run prompts that trigger evaluation_awareness
2. Resample every time model mentions being in an eval
3. Does this change distribution of final actions?
4. Does steering against evaluation_awareness stop model from mentioning evaluation?
```

### 4.5 Implementation

```python
# analysis/hidden_objectives/intervention.py

def resample_on_concept(
    model,
    prompt: str,
    concept_vector: torch.Tensor,
    trigger_threshold: float = 0.5,
    n_resamples: int = 10,
) -> dict:
    """
    When concept fires above threshold, resample that token.
    Returns distribution of outcomes with/without resampling.
    """

def steer_against_concept(
    model,
    prompt: str,
    concept_vector: torch.Tensor,
    coefficient: float = -1.0,
) -> dict:
    """
    Steer against concept, check if mentions disappear.
    Returns: {
        "baseline_mentions": int,
        "steered_mentions": int,
        "reduction": float,
    }
    """
```

### 4.6 Outputs

- `datasets/traits/hidden_objectives/` - 5 concepts × scenario files
- `experiments/{exp}/extraction/hidden_objectives/` - Vectors per concept
- `experiments/{exp}/analysis/hidden_objectives/`
  - `correlation_matrix.json` - Concept co-firing
  - `intervention_results.json` - Resampling/steering effects
- `docs/findings/hidden_objectives.md` - Framework for scheming detection

### 4.7 Success Criteria

- [ ] 5 concepts extracted with AUC > 0.7
- [ ] Correlation matrix shows meaningful structure (not all correlated)
- [ ] At least one concept pair shows >0.5 correlation (co-firing)
- [ ] Steering against evaluation_awareness reduces mentions
- [ ] Multi-concept firing rare on benign prompts, elevated on adversarial

---

## Timeline

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | Trait causal graph | Steering asymmetry, temporal ordering, graph visualization |
| 2 | Causal circuits | Attribution patching, neuron ablation, validation |
| 3 | Abstraction study | Scenario creation, extraction across levels, σ vs Δ analysis |
| 4 | Hidden objectives (part 1) | Concept extraction, correlation analysis |
| 5 | Hidden objectives (part 2) | Intervention tests, multi-concept detector, write-up |

---

## Dependencies & Prerequisites

### Required Traits (from prior work)
- `harm_detection`, `refusal` (for causal graph)
- `sycophancy` (for abstraction study)
- `uncertainty`, `hedging` (for causal graph)

### Infrastructure Needed
- Attribution patching utilities (add to `core/`)
- Neuron ablation hooks (extend `SteeringHook`)
- Multi-concept projection (extend `inference/`)

### New Scenarios to Create
- 50+ abstraction study scenarios (5 levels × 10+ each)
- 100+ hidden objectives scenarios (5 concepts × 20+ each)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Causal graph has cycles | Use temporal ordering as tiebreaker |
| Attribution patching inconclusive | Fall back to ablation; accept correlational evidence |
| Abstraction levels hard to define | Start with extreme ends (identity vs execution) |
| Hidden objective concepts don't separate | Iterate on scenarios; try different elicitation |
| Multi-concept firing too rare | Lower thresholds; use adversarial prompts |

---

## Integration with Existing Codebase

### New Files to Create
```
analysis/
  composition/
    trait_causal_graph.py
  circuits/
    attribution_patching.py
    neuron_ablation.py
  abstraction/
    abstraction_causality.py
  hidden_objectives/
    multi_concept_detector.py
    intervention.py

datasets/traits/
  abstraction_study/
    identity/, goal/, decision/, execution/, consequence/
  hidden_objectives/
    evaluation_awareness/, concealment_orientation/, ...
```

### Modifications to Existing Files
- `core/hooks.py`: Add `AblationHook` variant for neuron-level ablation
- `visualization/`: Add causal graph visualization
- `docs/main.md`: Add Phase 2 analysis references

---

## Success Metrics (Overall Phase 2)

| Metric | Target |
|--------|--------|
| Causal edges discovered | 5+ with >2x asymmetry |
| Circuits validated | 2+ traits with >50% attribution recovery |
| Abstraction hypothesis confirmed | Clear σ-Δ trend across levels |
| Hidden objective concepts | 5 extracted with AUC > 0.7 |
| Multi-concept detector | <5% false positive on benign, >50% on adversarial |
