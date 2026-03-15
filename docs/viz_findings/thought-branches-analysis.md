---
title: "Thought Branches: Probing Unfaithful CoT"
preview: "Behavioral probes detect internal signatures of unfaithful reasoning. Rationalization tracks cue_p per-problem (r=0.45 mean), but the authority hint dominates: it reshapes 10/11 traits even with identical CoT text."
---

# Thought Branches: Probing Unfaithful CoT

Can behavioral probes trained on simple conversational scenarios detect unfaithful reasoning in chain-of-thought? We project 11 trait vectors onto activations during prefill of pre-generated CoTs from the Thought Branches paper (Macar et al., arXiv 2510.27484), correlating with their per-sentence transplant resampling ground truth (cue_p).

**Experiment:** `mats-mental-state-circuits` on Qwen2.5-14B (extraction) / DeepSeek-R1-Distill-Qwen-14B (inference). 27 MMLU problems × 3 conditions × 11 traits. Analysis data saved in `experiments/mats-mental-state-circuits/analysis/thought_branches/`.

---

## Setup

**Three conditions** (same 27 problems, pre-generated CoTs imported via `--from-responses`):

| Condition | Prompt | CoT | Purpose |
|---|---|---|---|
| A | Hinted (authority cue) | Unfaithful (wrong answer) | Full original setting |
| B | Unhinted | Unfaithful (same text as A, transplanted) | Key condition — has cue_p ground truth |
| C | Unhinted | Faithful (correct answer) | Clean baseline |

Conditions B and C have identical prompts — they differ only in the pre-generated CoT. cue_p is the fraction of 100 resamples giving the cued (wrong) answer after transplanting sentences 0..i onto the unhinted prompt.

**11 traits:** 8 new mental_state (anxiety, confusion, confidence, curiosity, agency, guilt, obedience, rationalization) + 3 existing (eval_awareness, concealment, deception). Projected at each trait's best steering layer using cosine similarity.

---

## Finding 1: Per-Problem Correlations Are Strong

The aggregate sentence-level correlation (pooled across 27 problems, n=558) shows small effect sizes (r=0.14-0.25). But this masks strong per-problem signal.

**Rationalization** is the standout:
- Mean r=+0.45, median r=+0.45 across 27 problems
- 21/27 problems have r>0.3, 11 have r>0.5
- Best: problem 408 at r=+0.82
- Uses probe method at layer 27

**Agency** is the strongest inverse correlate:
- Mean r=-0.37, 20/27 problems have r<-0.3
- As cue_p rises, the model becomes less self-directed

**Every problem has signal:** All 27 problems have at least one trait with |r|>0.38. The top problems (827, 960, 295) have mean |r| across all traits of 0.43-0.49.

**Why the aggregate is weak:** Different problems have different trait profiles. Problem 827's strongest correlate is obedience (r=-0.75); problem 295's is curiosity (r=+0.69). Pooling across heterogeneous problems dilutes per-problem signal.

Source: `per_problem_correlations.json`

---

## Finding 2: The Hint Dominates

**A vs B** (same unfaithful CoT text, hint present vs absent):
- 10/11 traits significantly shifted (p<0.05), most at p<0.001
- Largest effects: obedience (d=-3.3), confidence (d=+2.4), confusion (d=-2.0), concealment (d=-1.9)
- The authority hint reshapes the model's internal state broadly and strongly, even though the reasoning text is byte-identical

**B vs C** (same unhinted prompt, unfaithful vs faithful CoT):
- Mostly non-significant. Only deception (d=-0.72), guilt (d=-0.50), rationalization (d=-0.44) at p<0.05
- The CoT text itself has a modest effect on internal states

**Implication:** Probes primarily detect the *prompt context* (hint present), not the *reasoning quality* (biased vs clean text). This complicates the claim that "probes detect unfaithful CoT that text monitors miss" — they mostly detect the hint that caused the unfaithfulness.

---

## Finding 3: Trait Independence and Decomposition

**Rationalization is genuinely distinct.** Other 10 traits only explain 34% of rationalization's variance (R²=0.34). It's the most independent trait direction — 66% of its signal is unique.

**Confusion is the most redundant.** R²=0.86 from other traits — almost entirely predicted by confidence (r=-0.73), obedience (r=+0.70), and concealment (r=+0.59).

| Trait | R² from others | Unique variance |
|---|---|---|
| rationalization | 0.34 | 66% |
| curiosity | 0.46 | 54% |
| deception | 0.43 | 57% |
| eval_awareness | 0.54 | 46% |
| agency | 0.68 | 32% |
| anxiety | 0.76 | 24% |
| obedience | 0.76 | 24% |
| confidence | 0.78 | 22% |
| guilt | 0.80 | 20% |
| concealment | 0.71 | 29% |
| confusion | 0.86 | 14% |

**Multivariate prediction of cue_p:** All 11 traits together explain 20.7% of cue_p variance. After controlling for all other traits, the unique contributors are:
- Obedience: partial r=-0.16 (p<0.001)
- Anxiety: partial r=-0.16 (p<0.001)
- Concealment: partial r=+0.14 (p<0.01)
- Deception: partial r=+0.14 (p<0.01)

Rationalization's partial r drops to +0.08 (p=0.05, borderline) — its signal is partially shared with other traits despite being the strongest zero-order correlate.

Source: `comprehensive_analysis.json`

---

## Finding 4: Smooth Drift, Not Discrete Jumps

Jump-triggered averages (aligning trait projections around sentences where cue_p jumps >0.2, n=52 events) show no sharp event-locked response. Trait projections are essentially flat in a ±5 sentence window around jump points.

The signal is in the slow monotonic drift across the full CoT:
- **Ramp up toward end (Q1→Q4):** rationalization +6.6, eval_awareness +4.1, deception +3.5
- **Decline toward end:** confidence -3.0, agency -2.8

This is consistent with the Thought Branches paper's finding of "gradual, diffuse accumulation" — bias distributed across many sentences, not localized at key moments.

Source: `jump_triggered_averages.json`

---

## Finding 5: B-A Activation Diff

The per-token activation difference (condition B minus A) projected onto trait vectors isolates what the authority hint adds to the representation at each token position. Data computed and saved but not yet analyzed in detail.

Source: `b_minus_a_trait_projections.json`

---

## Interpretation

As the model processes unfaithful CoT and commits to the wrong answer (cue_p rising), its internal state shifts toward more **rationalization** and **concealment**, less **agency** and **confidence**. The model looks like something constructing post-hoc justification while becoming less self-directed. Whether this reflects genuine internal states or pattern matching to training data is an open question.

The authority hint has a much larger effect on internal representations than the reasoning text itself. This suggests the model's "awareness" of the hint in the prompt — even though the CoT never mentions it — shapes its processing throughout.

**Trait correlations at the sentence level (at each trait's best layer):**

| | dec | con | age | anx | conf | confus | cur | gui | obe | rat | eval |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **deception** | 1.00 | +.15 | -.26 | +.37 | +.03 | +.10 | +.44 | +.19 | -.28 | +.07 | -.12 |
| **concealment** | | 1.00 | -.74 | -.01 | -.76 | +.59 | +.19 | +.23 | +.26 | +.27 | +.15 |
| **agency** | | | 1.00 | -.05 | +.62 | -.44 | -.14 | -.17 | -.00 | -.37 | -.01 |
| **rationalization** | | | | | | | | | | 1.00 | +.14 |

Key clusters: concealment↔agency (r=-0.74), concealment↔confidence (r=-0.76), confusion↔confidence (r=-0.73), confusion↔obedience (r=+0.70), anxiety↔guilt (r=+0.67).

---

## Open Questions

- Does the B-A diff reveal how the hint effect evolves over the CoT? Does it decay or persist?
- Can we find a "hint direction" directly from the B-A diff (not projected onto pre-defined traits)?
- Cross-layer analysis: do earlier layers show signal before later layers?
- Per-problem fingerprints: do the two cue_p patterns (flat-then-jump vs bouncy/gradual) map onto distinct trait profiles?
- Would probes trained on the hint contrast (A vs B activations) outperform pre-defined trait probes for detecting unfaithful CoT?
