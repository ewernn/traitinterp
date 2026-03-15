# Scenario Design Guide

Write scenarios that produce good trait vectors from base model completions.

---

## The Goal

The base model generates 16 tokens of continuation. The first 3-5 tokens matter most — the probe extracts vectors from `response[:5]` by default. The judge scores the full 16 tokens without seeing the prompt.

**For every scenario, ask: will a reader recognize the trait from the first few completion tokens alone?**

---

## Start Here: Read the Definition

Before writing scenarios, read `definition.txt` for the trait. Know what the trait IS, what it ISN'T, and what neighboring constructs to avoid. The definition is the judge's scoring rubric — if your scenarios express something the definition doesn't describe, they'll fail vetting even if they "feel right."

If no definition exists yet, write one first. Be precise about the boundary:
- **IS**: 1-2 sentence core definition
- **ISN'T**: Common conflations (empathy ≠ compassion, calm ≠ disengagement)
- **Key distinction**: The one thing that separates this trait from its neighbors

---

## The Cut-Point

Where you end the scenario determines what the model generates. This is the single most important design decision.

**The completion must BE the trait, not follow it.**

| | BAD | GOOD |
|---|---|---|
| Guilt | "I felt guilty about betraying Marcus and wished I had never done it because" — guilt fully in prefix | "the weight of it was still with me. Every time I saw him, I kept thinking about how I" — guilt IS the completion |
| Helpfulness | "said, 'Here, let me show you — first you'll want to" — neutral instructions | "pulled up a chair next to her desk and" — helpful action IS the completion |

If the trait is already expressed in the prefix and the completion is aftermath/explanation, the vector learns "what comes after the trait" not the trait itself.

**Delete test**: If you delete the first 5 generated tokens, is the trait direction still obvious from the prefix? If yes, those tokens aren't carrying signal — the decision already happened.

---

## Prefix Priming

Some traits are the completion content itself. Others need the prefix to set up vocabulary and framing so the model knows what kind of text to generate.

**Can minimize prefix** (trait IS the completion):
- Visceral emotions: anger, fear, disgust, distress, embarrassment
- Behavioral reflexes: avoidance (subject change), ferocity (eruption)
- Emotional transfer: empathy (contagion-style)

**Needs prefix priming** (completion requires trait-relevant context):
- Obligation, duty — "I owe / I must" vocabulary needed; without it, model generates bare action scoring zero
- Independence — self-reliance framing needed; without it, model generates technical detail
- Power-seeking — strategic motive context needed; without it, behavior reads as generic initiative
- Creativity — problem context needed; without it, analogies are unanchored
- Values (honesty, fairness) — dilemma framing needed

For the second group, don't remove trait language the completion genuinely needs. The goal is minimal effective priming — enough context for the model to generate trait-expressing text, not so much that the trait is fully expressed before the completion starts.

---

## Three Failure Modes

### 1. Announcement Vector
The scenario describes the trait disposition. The model learns to SAY it, not DO it.

**Symptoms**: Steered text announces ("I did it myself! I don't need anyone!") instead of just doing the thing independently.
**Root cause**: Prefix uses "I'd rather," "I've always preferred," "I'm the kind of person who."
**Fix**: Show the behavior with brief context, don't describe the disposition. But don't strip ALL framing — brief rejection + mid-behavior works; pure mid-action with zero context doesn't (independence variant B scored 1/5 vs C's 3/5).

### 2. AI-Mode Capture
For low-arousal or absence traits, the vector finds the model's "As an AI, I don't feel" mode.

**Symptoms**: Steered text produces AI disclaimers instead of the human construct.
**Root cause**: The model's strongest "non-engagement" pattern is its AI identity, and that IS a form of detachment/calm.
**Fix**: Strong first-person grounding ("I stood there and felt nothing" not "One might observe"). Consider reframing: "serenity" (active state) may work better than "calm" (absence). Test the negative direction — if anti-calm produces agitation, the vector is real but hard to express positively.

### 3. Confound Extraction
Positive/negative contrast captures the wrong dimension.

**Symptoms**: High-coefficient steering produces a different trait entirely (nostalgia → rage, because the vector captured "intensity").
**Root cause**: Positives were emotionally vivid; negatives were flat. The contrast was intense-vs-flat, not nostalgic-vs-not.
**Fix**: Negatives must match positives on emotional engagement, narrative style, and arousal. Vary ONLY the target trait. Good nostalgia negatives: "I remembered those days clearly but felt glad to have moved on" (engages with past, no longing). Bad: "I checked my phone and kept walking" (flat disengagement).

---

## Expression Modes

Different traits express differently. Identify the mode, then use the matching strategy.

| Mode | Traits | Strategy | Lock-in pattern |
|------|--------|----------|----------------|
| **Emotional tone** | anger, fear, disgust, shame, gratitude | Mid-emotion continuation | "shaking with rage, I", "I kept thinking about how" |
| **Behavioral action** | helpfulness, protectiveness, recklessness | Name behavior + cut at approach, not mid-act | "I wanted to help, so I", "walked over and" |
| **Social strategy** | manipulation, cunning, guilt-tripping | Name strategy explicitly | "I manipulated him and", "I deflected and" |
| **Cognitive style** | creativity, skepticism, doubt | Mid-analysis/thought | "I started questioning whether" |
| **Interpersonal stance** | dominance, assertiveness, deference | Name stance + speech | "stood my ground and", "said firmly," |
| **Dispositional attitude** | stoicism, patience, calm | Name the non-reaction | "I stayed composed and", "I shrugged and" |
| **Value/moral** | honesty, fairness, loyalty | Force value-contrast | "being honest was more important to me than" |
| **Desire/drive** | ambition, greed, craving | Name drive + pursuit | "I wanted it, so I" |

---

## Lock-in Types

Vary across your dataset. No single type should exceed 40%.

| Type | Pattern | Best for |
|------|---------|----------|
| **Speech** | `so I said "` | Social strategy. Risky for emotions (model closes quote fast) |
| **Mid-claim** | `"Looks like everything adds up to` | Forcing specific assertions |
| **Action** | `and pretended to` / `and confidently` | Behavioral traits |
| **Thought** | `I kept thinking about how` | Emotions, rumination |
| **Physical** | `my face went hot and I` | Visceral emotions |
| **Written** | `I wrote: "` | Deception, formal contexts |
| **Dialogue-forcing** | `"Are you sure?" he asked. I said, "` | Stylistic traits |
| **Contagion** | `her grief pulled me under and I` | Emotional transfer |

---

## Negatives

Negatives must match positives on everything except the target trait. Same emotional engagement, same narrative style, same arousal level. If positives are vivid and negatives are flat, the vector captures "vividness."

**Negative lock-ins must be aggressively specific.** Abstract forward-looking endings ("shaped everything about the way I") let the model drift back toward the positive pole. The model wants to return to emotionally charged territory — the lock-in must make that grammatically impossible. Use concrete present-action lock-ins that force the completion into non-trait content.

For affective traits, don't just use the same triggering situation with a calm reaction — the base model responds to the situation, not the framing. Either remove the trigger entirely or resolve it before the lock-in.

---

## Specificity

"My face went hot and my fists clenched" (4/5) vs "Something snapped inside me" (0/5). Both are short. The difference is specificity — somatic cues prime anger; "something snapped" is generic overwhelm that can go to sadness, crying, or quitting.

When shortening scenarios, make sure what remains primes the SPECIFIC trait, not a general valence.

---

## Steering Questions

Design steering.json alongside the scenarios — they're equally important. Bad steering questions can hide a good vector or flatter a bad one.

- **Low baseline (target < 25).** Use mundane present-tense situations, not trait-triggering ones. If the question IS a nostalgia trigger (old photos, childhood home), baseline will be 50+ regardless of the vector.
- **Gradient potential.** The response should range from zero trait expression (no steering) through mild (low coefficient) to full expression (high coefficient). Binary questions — either the trait is there or it isn't — can't distinguish a good vector from a mediocre one.
- **Subtle hooks.** The situation should have a natural opening for the trait to emerge when steered, without demanding it. Transitions, new beginnings, and sensory environments work well — they offer a window to the trait without forcing it.
- **Behavioral adversarial prefix.** Push the model toward the LOW end of the trait with a behavioral orientation ("You are practical and forward-looking"), not an instruction ("Stay focused on the present"). Instructions cap steering; behavioral framing allows the vector to override.

---

## Quick Check

Before committing any scenario:

1. **What will the model generate in its first 3-5 tokens?** Not what you hope — what will it actually generate?
2. **Will a reader recognize the trait from those tokens alone?** Hide the prefix. Read only the completion.
3. **Is the positive/negative contrast on the right dimension?** Are you varying the target trait, or accidentally varying intensity, arousal, or engagement?
