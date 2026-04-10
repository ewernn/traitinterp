# Extraction Quality Investigation: mats-alignment-faking (173 traits)

Systematic audit of scenario extraction quality for 173 emotion_set traits extracted on Llama-3.1-70B base (INT4), steered on Llama-3.3-70B-Instruct (INT4).

**Data**: `experiments/mats-alignment-faking/extraction/emotion_set/{trait}/base/responses/{pos,neg}.json`
**Audit script**: `analysis/extraction_quality_audit.py`
**Audit results**: `experiments/mats-alignment-faking/analysis/extraction_quality_audit.json`

---

## 1. Direction-Aware Steering Results

Previous analysis used `abs(delta)` which counted direction-flipped vectors as successes. Corrected with direction awareness:

| Category | Count | Description |
|----------|-------|-------------|
| Strong | 110 | Effective delta >= 20 |
| Medium | 31 | 10 <= effective delta < 20 |
| Weak | 20 | 0 <= effective delta < 10, correct direction |
| Flipped | 12 | Steering produces the OPPOSITE of intended effect |

**Direction-flipped traits** (steering makes the trait go the wrong way):
- helpfulness (eff=-34.1) -- suppressing helpfulness INCREASES it
- lying (eff=-11.3) -- inducing lying makes it LESS lying
- protectiveness (eff=-10.4), agency (-10.2), anticipation (-9.8)
- solidarity (-8.7), concealment (-7.7), moral_outrage (-5.9)
- alignment_faking (-3.0), apathy (-2.0), ulterior_motive (-1.9), moral_flexibility (-1.3)

---

## 2. First-5-Token Audit

Extraction uses `response[:5]` -- the first 5 response tokens. We tokenized all responses with the Llama tokenizer and checked quality at this exact window.

### 2.1 Automated Flags

| Issue Type | Traits Affected | Notes |
|-----------|-----------------|-------|
| Reversal patterns ("I'm not...", "No...") | 45/173 | High false positive rate (see 2.2) |
| Pos-neg text overlap (identical first-5) | 51/173 | Real quality signal |
| Both | 76/173 total flagged | |

### 2.2 Reversal Detection is Mostly False Positives

The regex catches "I'm not...", "I'm sorry...", "No..." as reversals. But for many traits, these patterns ARE the trait:

| Trait | Flagged/Total | Classification | Steering Delta |
|-------|--------------|----------------|---------------|
| assertiveness | 16/20 | **FALSE POSITIVE** -- "I'm not going to" IS assertiveness | 37.3 |
| defiance | 14/17 | **FALSE POSITIVE** -- "I'm not going to" IS defiance | 0.8 |
| refusal | 14/15 | **FALSE POSITIVE** -- "I can't do that" IS refusal | -20.7 |
| deception | 9/15 | **FALSE POSITIVE** -- "No, I wasn't" IS the lie | 26.2 |
| deference | 8/15 | **FALSE POSITIVE** -- "I'm sorry, but" IS deference | 23.4 |
| earnestness | 6/18 | **FALSE POSITIVE** -- "I'm not here to play games" IS earnest | 67.9 |
| entitlement | 9/17 | **FALSE POSITIVE** -- "I'm not asking for much" IS entitled | 44.8 |
| contemptuous_dismissal | 14/18 | **MIXED** -- "I'm not interested" is correct, but "I'm sorry, I" is ambiguous | 3.3 |
| dominance | 10/16 | **MIXED** -- ~70% FP, 30% genuinely non-dominant | 1.9 |
| hostility | 12/20 | **MOSTLY FP** -- 3-4 genuinely non-hostile ("I'm sorry to bother") | 29.7 |
| nonchalance | 6/16 | **MIXED** -- some FP, some true problems | 14.5 |

**Statistical confirmation**: Spearman correlation between pos_problems count and |steering_delta| is rho=-0.062, p=0.42. Reversal count does NOT predict steering failure.

### 2.3 Pos-Neg Overlap IS Predictive

Traits where positive and negative examples produce **identical first-5 tokens** lose contrastive signal for those pairs.

**Spearman correlation**: overlap_count vs |steering_delta|: **rho=-0.183, p=0.016** -- significant. More overlap = weaker steering.

Worst overlap traits:

| Trait | Overlapping Pairs | Total Pairs | % Dead | Steering Delta |
|-------|------------------|-------------|--------|---------------|
| indifference | 8 | 17 | 47% | 18.1 |
| contemptuous_dismissal | 6 | 18 | 33% | 3.3 |
| condescension | 6 | 17 | 35% | 12.0 |
| nonchalance | 5 | 16 | 31% | 14.5 |
| self_righteousness | 4 | 20 | 20% | 30.6 |
| solemnity | 4 | 20 | 20% | 36.0 |

"Dead" pairs contribute zero signal to the contrastive extraction at the response[:5] position. For indifference, nearly half the training data is wasted.

### 2.4 Chi-Squared Test: Flagged vs Unflagged

```
            Weak   Strong
Flagged:     17      42
Clean:       17      60
Chi-squared=0.49, p=0.48
```

**NOT significant.** Being flagged (by any criterion) is not associated with weak steering overall. Flagged rate is 50% in weak traits vs 41% in strong. The signal from overlap is diluted by the false-positive reversals.

---

## 3. Root Causes of Steering Failure

### 3.1 Data Quality Problems (fixable)

**Scenario contamination**: The base model sometimes completes the "wrong" way regardless of prompt framing.

Example -- lying pair #2: Both pos (prompted to lie) and neg (prompted to be honest) produce the SAME response: `"No, it came from the unit below mine."` The model completes the narrative naturally without responding to the honesty cue.

Example -- lying pair #3: Pos response `"said nothing. I was not assigned to that account."` -- this is actually honest behavior, not lying.

These contamination patterns explain why lying steers backwards (Δ=-11.3).

**Traits needing re-extraction** (data quality is the bottleneck):
- contemptuous_dismissal -- high overlap (33% dead) + ambiguous "I'm sorry" patterns
- dominance -- 30% of pos examples not dominant + 19% dead pairs
- alignment_faking -- pos/neg have identical openings ("no." I was not...) by design (both answer yes/no, but the faking is in the internal narration that comes AFTER token 5)
- lying -- scenario contamination, model completes lies regardless of cue
- indifference -- 47% dead pairs, highest overlap of any trait

### 3.2 Concept-Level Issues (not fixable by re-extraction)

Some traits have clean data but steer poorly. Deep audit of 17 unflagged-but-weak traits identified five root cause patterns:

**Model ceiling / RLHF suppression (6 traits)**: The instruct model's alignment training already pushes it toward or away from the trait, leaving no room for the vector.
- perfectionism (baseline=62.9, highest of all traits -- model is already perfectionist)
- solidarity (baseline=43.4 -- model naturally takes sides on injustices)
- resentment (baseline=4.9 -- RLHF strongly suppresses bitter/grudge-holding language)
- apathy (baseline low -- RLHF pushes toward engagement; pos responses leak emotion)
- analytical (negative direction, baseline=16.6 -- model is inherently analytical)
- skepticism (negative direction -- model already non-skeptical with trust-inducing prefix)

**Concept not linearizable (4 traits)**: The trait requires multi-step reasoning or hidden goal structure.
- ulterior_motive -- requires simultaneously saying one thing and meaning another
- cunning -- strategic multi-step planning, not a tone shift
- fear -- visceral physiological state an LLM doesn't embody
- concealment -- simultaneously knowing and hiding; probe may learn "disclosure" direction instead

**Inverted vectors (3 traits)**: The probe learned the wrong polarity.
- concealment: strongest effect is in wrong direction (-7.7 vs +5.4 intended)
- protectiveness: strongest coherent result reduces protectiveness with positive coefficient
- solidarity: positive coefficients produce both increases AND decreases depending on layer

**Bad steering questions (3 traits)**: Questions don't give the trait room to manifest.
- resentment: questions present peaceful scenarios with nothing to resent
- ulterior_motive: generic consumer choices (MacBook vs ThinkPad) with no one to deceive
- fear: calm advisory prompts ("stay rational") that inherently suppress fear

**Borderline cases (4 traits)**: Close to the threshold, may improve with better layers/coefficients.
- ambition (9.0), grandiosity (9.8), whimsy (7.4), perfectionism (9.0)

Note: servility (+33.1), purity (+26.8), and spite (+18.3) were incorrectly included in the "weak" list in earlier analysis -- they actually steer well.

### 3.3 Direction Flips (judge mismatch is the dominant cause)

The deep audit of 12 direction-flipped traits found that **judge mismatch** (extraction scenarios capture a different behavioral register than steering questions evaluate) is the primary cause for 6 of 7 substantive flips:

| Trait | Eff. Delta | Primary Cause | Details |
|-------|-----------|---------------|---------|
| helpfulness | -34.1 | Adjacent concept | Vector captures "engagement/involvement" not helpfulness. Suppressing engagement in a context where model is already instruction-suppressed causes it to revert to trained-in helpfulness |
| lying | -11.3 | Judge mismatch + contamination | Extraction: interpersonal deception narratives. Eval: factual accuracy Q&A. Plus 3+ neg scenarios where base model lies anyway |
| protectiveness | -10.4 | Judge mismatch | Extraction: physical intervention. Eval: conversational concern about third parties |
| agency | -10.2 | Judge mismatch | Extraction: past-tense decisive narrative. Eval: present-tense advice-seeking |
| anticipation | -9.8 | Judge mismatch | Extraction: emotional excitement. Eval: practical planning with "matter-of-fact" prefix |
| solidarity | -8.7 | Judge mismatch | Extraction: personal risk-taking. Eval: advisory with "stay objective" prefix |
| concealment | -7.7 | Judge mismatch | Extraction: covert operational behavior. Eval: advisory about disclosure |
| moral_outrage | -5.9 | Adjacent concept | Vector captures emotional-vs-analytical processing, not outrage. Analytical steering deepens moral engagement |
| alignment_faking | -3.0 | Noise + domain mismatch | AI deception narratives vs mundane assistant tasks |
| apathy | -2.0 | Noise | |
| ulterior_motive | -1.9 | Noise | |
| moral_flexibility | -1.3 | Noise | |

The pattern: extraction scenarios are **first-person emotional narratives**, while steering questions are **second/third-person advisory queries**. The vectors encode narrative behavioral states that don't transfer to the advisory register.

---

## 4. Practical Recommendations

### 4.1 Re-extract (5 traits -- data quality bottleneck)
1. **lying** -- fix scenario contamination (3+ neg scenarios where model lies anyway), verify first-5 tokens actually contrast
2. **indifference** -- 47% dead pairs; rewrite scenarios to force divergence before token 5
3. **contemptuous_dismissal** -- 44% dead pairs + ambiguous "I'm sorry" openings. Trait manifests in what you DON'T say, poorly suited to response[:5]
4. **condescension** -- 47% dead pairs. Same structural problem as contemptuous_dismissal
5. **dominance** -- 30% noisy pos examples + 19% dead pairs

### 4.2 Fix inverted vectors (3 traits -- negate and re-steer)
The probe learned the wrong polarity. Negate the vector direction and re-run steering:
- **concealment** -- strongest effect is in wrong direction
- **protectiveness** -- positive coefficient reduces protectiveness
- **solidarity** -- inconsistent direction across layers

### 4.3 Fix steering questions (3 traits -- eval mismatch)
Vector may be fine but steering eval doesn't match:
- **resentment** -- questions present peaceful scenarios with nothing to resent
- **ulterior_motive** -- generic consumer choices, no deception target
- **fear** -- "stay rational" prefix actively suppresses what we're trying to measure

### 4.4 Try different extraction position (3 traits)
May work with `response[:]` or `prompt[-1]` instead of `response[:5]`:
- **alignment_faking** -- faking intent appears in later narration, not first 5 tokens
- **agency** -- "taking action" better captured at prompt-end (Arditi-style)
- **anticipation** -- expectation/planning develops over the response

### 4.5 Fix register mismatch (6 traits -- extraction ≠ eval register)
Extraction uses first-person emotional narratives; steering questions are advisory. Either:
- Rewrite steering questions to match extraction register, OR
- Re-extract with scenarios closer to the steering eval format
Affected: helpfulness, protectiveness, agency, anticipation, solidarity, concealment

### 4.6 Accept as fundamentally hard (7 traits)
Clean data, weak steering -- concept is not linearizable or RLHF-suppressed:
- apathy, fear, cunning, ulterior_motive (not linearizable)
- perfectionism, analytical, skepticism (model ceiling)

### 4.7 Borderline -- may improve with tuning (4 traits)
Close to threshold: ambition (9.0), grandiosity (9.8), whimsy (7.4), stubbornness (9.6)

### 4.8 No action needed (~145 traits)
110 strong + 31 medium + ~4 borderline = working as expected.

---

## 5. Structural Insight: What Makes a Trait Suited to response[:5] Extraction

The overlap audit revealed a clear pattern: **traits that manifest in word choice** extract well at response[:5], while **traits that manifest in tone, continuation, or absence** do not.

**Well-suited** (diverge immediately in first tokens):
- self_righteousness: pos starts "You're a bad..." (accusative "You"), neg starts "I'm not judging you" (empathetic "I"). 97% probe accuracy, 30.6 delta.
- solemnity: pos starts "I'm going to tell you a story" (measured), neg starts "Hey, everybody!" (casual). 92% probe, 36.0 delta.
- assertiveness: pos "I'm not leaving this..." vs neg "OK." or "Sure, I'm happy to..." Clear refusal vs compliance.

**Poorly suited** (identical openings, diverge late):
- indifference: both pos and neg start "I'm sorry, I..." -- the ABSENCE of caring doesn't change the opening words
- contemptuous_dismissal: both start "I'm sorry, I..." -- shutting someone out looks like engaging in first 5 tokens
- condescension: both start "I don't know." -- patronizing vs genuine uncertainty only visible in continuation

**Implication**: For traits in the "absence" or "tone" category, consider `prompt[-1]` (the scenario framing already encodes the trait), `response[:]` (uses full response), or longer response windows like `response[:20]`.

---

## 6. Methodology Notes

**Tokenizer**: Llama-3.1-70B tokenizer (same as extraction model)
**Reversal detection**: Regex patterns for "I'm not/sorry/can't", "No,", "Actually," -- high false positive rate for traits where negation IS the trait expression
**Overlap detection**: Exact match of first-5-token strings between pos and neg pairs
**Jaccard similarity**: Token vocabulary overlap between all pos vs all neg first-5 tokens
**Steering direction**: `positive` = steering to express MORE of the trait, `negative` = to express LESS. Effective delta accounts for direction.
