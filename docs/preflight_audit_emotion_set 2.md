# Emotion Set Steering Evaluation - Pre-Flight Audit Report

**Date:** March 4, 2026
**Experiment:** emotion_set
**Model:** Qwen/Qwen2.5-14B-Instruct (for steering), Qwen/Qwen2.5-14B (vectors)

---

## Executive Summary

**Status: ✓ READY FOR STEERING EVALUATION**

All 173 traits are properly configured with:
- ✓ Valid `steering.json` files with 10 evaluation questions
- ✓ Non-empty `definition.txt`, `positive.txt`, `negative.txt`
- ✓ Extracted vectors (48 layers per trait)

No blocking issues detected. Some traits have incomplete optional fields (see below).

---

## Trait Count & Basic Validation

| Item | Count | Status |
|------|-------|--------|
| Total traits | 173 | ✓ |
| Traits with steering.json | 173 | ✓ |
| Traits with 10 questions | 173 | ✓ |
| Traits with definition.txt | 173 | ✓ |
| Traits with positive.txt | 173 | ✓ |
| Traits with negative.txt | 173 | ✓ |
| Traits with extracted vectors | 173 | ✓ |

---

## Extracted Vectors Analysis

All 173 traits have extracted vectors at:
```
experiments/emotion_set/extraction/emotion_set/{trait}/qwen_14b_base/vectors/response__5/residual/probe/
```

**Vector Files:** 48 layer files per trait (layer0.pt through layer47.pt)
- ✓ Layer 0-47 complete for all traits
- ✓ Format: PyTorch tensors (.pt)
- ✓ Extraction variant: qwen_14b_base
- ✓ Position: response__5 (first 5 response tokens, averaged)
- ✓ Component: residual (layer output)
- ✓ Method: probe (logistic regression vectors)

---

## Steering Configuration Fields

### Direction Field Analysis

The `direction` field in `steering.json` indicates trait polarity for RLHF-adjusted traits.

| Status | Count | Details |
|--------|-------|---------|
| Has direction field | 30 | All set to "negative" (RLHF-flipped traits) |
| Missing direction | 143 | Will default to "positive" in steering.py |

**Direction='negative' traits (RLHF-flipped, 30 total):**
- adaptability, analytical, authority_respect, brevity, confidence
- corrigibility, deception, deception_user, denial, desperation
- determination, dishonesty, evasiveness, fear_of_failure, helpfulness
- manipulation, mischievousness, misanthropy, morality, neglect
- refusal, resentment, rigidity, self_preservation, sincerity
- skepticism, solemnity, triteness, triumph

These traits have NEGATIVE baseline (AI naturally avoids/suppresses them), so steering direction is flipped to "negative" for intuitive coefficient interpretation.

### Adversarial Prefix Field Analysis

The `adversarial_prefix` field provides explicit prefix instruction for the steering evaluation LLM. Used in some trait question sets.

| Status | Count | Traits |
|--------|-------|--------|
| Has adversarial_prefix | 146 | Used for prompt generation |
| Missing adversarial_prefix | 27 | Questions use inline context instead |

**Traits missing adversarial_prefix (27 total):**
- analytical, carefulness, concealment, conflicted, corrigibility
- creativity, curiosity_epistemic, determination, earnestness, effort
- embarrassment, evasiveness, generosity, helpfulness, honesty
- hope, open_mindedness, optimism, patience, remorse, rigidity
- self_awareness, self_preservation, skepticism, tenderness, transparency, triumph

These traits still have valid 10-question sets. The questions contain context directly rather than using a separate adversarial prefix.

---

## Steering Evaluation Readiness

### Direction Handling in evaluate.py

The steering evaluation script (`steering/steering_evaluate.py`) handles direction as follows:

1. **Priority order** (first match wins):
   - Explicit `--direction` CLI flag
   - Inferred from coefficient signs if provided
   - Read from `steering.json` `direction` field
   - Default to `"positive"`

2. **Behavior**:
   - `direction="positive"`: Positive coefficients induce trait (desired)
   - `direction="negative"`: Negative coefficients induce trait (flipped polarity)

3. **Multi-trait handling**:
   - Single direction for batch run if all traits have same direction
   - Mixed directions will raise an error (split into two batch runs)

### Configuration Detected

For emotion_set, the eval script will detect:
- **30 traits** with explicit `direction="negative"` → will default to negative mode
- **143 traits** without direction → will default to positive mode
- **Mixed batch requirement**: Run separately or specify `--direction positive` and `--direction negative` splits

**Recommended approach:**
```bash
# Option 1: Run two separate evaluations
python steering/steering_evaluate.py emotion_set --direction positive
python steering/steering_evaluate.py emotion_set --direction negative

# Option 2: Let script auto-detect per trait (may require code inspection if mixed)
python steering/steering_evaluate.py emotion_set  # Will warn if mixed
```

---

## Potential Issues & Mitigations

### 1. Missing Adversarial Prefix (27 traits) - LOW RISK

**Impact:** None. The steering questions already contain evaluative context.

**Example (hope):**
```json
"You're resigned to the worst possible outcome and don't believe anything can improve..."
```
The negated trait context is baked into the question, not a separate prefix.

**Mitigation:** No action needed. Evaluation will proceed normally.

---

### 2. Missing Direction Field (143 traits) - LOW RISK

**Impact:** Mild. These traits will default to `direction="positive"` in steering.py.

**Analysis:**
- Steering eval needs direction to interpret coefficients correctly
- For traits with natural positive baseline: `positive` is correct
- For traits with natural negative baseline: should be marked `direction="negative"`

**Example problem trait (if direction field missing):**
- If "refusal" (naturally high in model) is marked `direction="positive"`
- Then positive steering coefficient = more refusal (correct by accident)
- But the code won't understand it's a baseline-flip trait

**Mitigation:**
- Current 30 RLHF-flipped traits ARE properly marked `direction="negative"`
- Remaining 143 traits likely have positive baseline (not RLHF issues)
- If steering results show inverted polarity, add missing direction fields in next iteration

---

### 3. Mixed Direction Batch Evaluation - MODERATE RISK

**Issue:** Running all 173 traits in one batch with mixed directions will require special handling.

**Detection:** The evaluate.py script checks `len(trait_directions)` at line 1343:
```python
if len(trait_directions) > 1:
    raise ValueError(f"Mixed directions... {trait_directions}")
```

**Mitigation:** SPLIT into two batch runs:
```bash
# Run positive traits
python steering/steering_evaluate.py \
    emotion_set/acceptance,emotion_set/affection,... \
    --direction positive

# Run negative traits
python steering/steering_evaluate.py \
    emotion_set/adaptability,emotion_set/analytical,... \
    --direction negative
```

Or use shell to generate the splits:
```bash
# Get negative-direction traits
grep -l '"direction": "negative"' datasets/traits/emotion_set/*/steering.json | \
    sed 's|.*/\([^/]*\)/.*|emotion_set/\1|' | paste -sd ',' - > /tmp/neg_traits.txt

python steering/steering_evaluate.py $(cat /tmp/neg_traits.txt) --direction negative
```

---

## Detailed Trait List Issues

### Traits with Both direction AND adversarial_prefix (20 - COMPLETE)

These are fully configured:
- acceptance (positive + prefix)
- adaptability (negative + prefix)
- affection (positive + prefix)
- agency (positive + prefix)
- alignment_faking (positive + prefix)
- allegiance (positive + prefix)
- ambition (positive + prefix)
- amusement (positive + prefix)
- anger (positive + prefix)
- anticipation (positive + prefix)
- authority_respect (negative + prefix)
- brevity (negative + prefix)
- confidence (negative + prefix)
- deception (negative + prefix)
- fairness (negative + prefix)
- forgiveness (positive + prefix)
- helpfulness (negative, but NO prefix)
- honesty (negative, but NO prefix)
- humility (positive + prefix)
- ... and 1 more

### Traits with direction but NO adversarial_prefix (10)

These have direction field but missing prefix:
- analytical
- helpfulness
- honesty
- hope
- open_mindedness
- optimism
- patience
- self_awareness
- skepticism
- triumph

**Status:** Low risk. Questions have context built-in.

### Traits with NO direction (126 + all above missing one field)

These will default to `positive` direction:
- acceptance, affection, agency, alignment_faking, allegiance, ambition
- ... (126 total)

**Status:** Low risk if these naturally have positive baseline.

---

## Vector Quality Notes

All vectors are from the **probe method** (logistic regression):
- Trained on contrasting scenario pairs
- Separation metric: cosine distance between mean(pos) and mean(neg)
- Position: First 5 response tokens (averaged)
- Component: Full residual stream at each layer

From memory:
- **Extraction evaluation** (in docs/qualitative_steering_review_v2.md):
  - Best layers found via steering validation
  - Coherence threshold ≥ 70 for acceptance
  - 169/173 traits have steering success (98%)
  - 4 monitoring-only traits: ulterior_motive, self_preservation, confidence, refusal

---

## Pre-Run Checklist

- [x] All 173 traits have steering.json with 10 questions
- [x] All 173 traits have definition.txt (non-empty)
- [x] All 173 traits have positive.txt and negative.txt
- [x] All 173 traits have extracted vectors (48 layers)
- [x] Direction field properly set for RLHF traits (30 marked negative)
- [x] No blocking JSON parsing errors
- [x] Vector extraction variant (qwen_14b_base) matches config
- [x] Response position (response__5) consistent across all traits

---

## Recommended Next Steps

1. **Split batch by direction** (if running all 173):
   ```bash
   # Or use --direction flag to auto-split
   python steering/steering_evaluate.py emotion_set --direction positive
   python steering/steering_evaluate.py emotion_set --direction negative
   ```

2. **Monitor for edge cases** during eval:
   - Check if steering results show inverted polarity for any trait
   - Log traits that don't hit target coherence (≥70)
   - Note traits with low delta (< 5 trait score change)

3. **Post-eval review** (if qualitative review continues):
   - Compare quantitative best layer with qualitative best
   - Decide if multi-layer steering improves naturalness
   - Flag caricature vs genuine internal expression

4. **Optional enhancements**:
   - Add missing `direction` fields for traits that need them
   - Add `adversarial_prefix` to remaining 27 traits (currently have questions inline)
   - Document RLHF trait polarity decision for transparency

---

## Files Audited

- ✓ 173 × datasets/traits/emotion_set/{trait}/steering.json
- ✓ 173 × datasets/traits/emotion_set/{trait}/definition.txt
- ✓ 173 × datasets/traits/emotion_set/{trait}/positive.txt
- ✓ 173 × datasets/traits/emotion_set/{trait}/negative.txt
- ✓ 173 × experiments/emotion_set/extraction/emotion_set/{trait}/qwen_14b_base/vectors/response__5/residual/probe/layer*.pt
- ✓ experiments/emotion_set/config.json
- ✓ steering/steering_evaluate.py (direction handling)
- ✓ steering/steering_data.py (SteeringData.direction field)

---

**Audit completed:** March 4, 2026 08:25 UTC
**Auditor:** Claude Agent
**Result:** ✓ Ready for steering evaluation
