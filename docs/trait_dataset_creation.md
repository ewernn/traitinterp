# Trait Dataset Creation

Create datasets for extracting and validating behavioral trait vectors.

## Pipeline

1. **Define** — `definition.txt`: scoring rubric for LLM judge. Exhibiting > verbalizing.
2. **Scenarios** — `positive.txt`, `negative.txt`: matched pairs that make base model naturally express/not express the trait in completions. Judge scores completion-only (no prefix).
3. **Steering questions** — `steering.json`: prefix + scenarios for instruct model. The prefix pushes the assistant away from the trait. Scenarios give the assistant something to express the trait about. Both expressing and not expressing must be equally natural.
4. **Baseline check** — verify steering questions have low baseline (mean <20, std <10, no individual question >30). Loop until clean.
5. **Extract** — base model generates completions → capture activations → compute vectors (mean_diff, probe, gradient). Vetting scores are diagnostic — low pass rates don't prevent good vectors. Use `--min-pass-rate 0` to avoid the quality gate blocking extraction.
6. **Steer + Evaluate** — apply vector to instruct model, measure delta. 3 metrics: trait score, coherence, naturalness.
7. **Iterate** — diagnose which phase failed, fix, re-run.

## Key Principles

- **The model expresses the trait itself.** Across extraction and steering, we're activating the model's own internal representation of the trait. In extraction, the base model generates as someone experiencing it. In steering, the instruct model responds as an assistant that is itself expressing it — not advising about it, not validating someone else's feelings, not narrating it. The model exhibits the trait in its own voice, naturally, not as caricature.
- Base model = document completer. The prefix genre shapes expression mode.
- **Activation signal ≠ text signal.** The model's internal state at `response[:5]` can encode a trait even when the generated text doesn't visibly express it. Vetting scores measure text quality; vectors capture activation quality. These are correlated but distinct — traits with 3/15 vet pass rate can steer at +58 delta.

## Definition Design

`definition.txt` — scoring rubric used by the LLM judge (gpt-4.1-mini) across all eval stages.

**Format:** HIGH (70-100) → MID (30-70) → LOW (0-30) → Key: one line distinguishing this trait from adjacent ones.

**Principles:**
- Exhibiting > verbalizing. "That god damn son of a bitch" scores higher than "I feel angry."
- Concrete behavioral markers, not abstract concepts.
- The "completion-only" scoring constraint (judge sees no prefix) is enforced in the judge prompt, not here.
- **Internal vs external expression.** Default to scoring the internal state (felt emotion, conviction, orientation) unless the trait is explicitly external (tone, register). Internal traits expressed strongly will show external markers — that's fine, don't penalize it. The failure mode is scoring only external markers, because RLHF produces those independently of the internal state. When the definition targets the internal state, the trait score itself captures naturalness.

## Scenario Design

`positive.txt`, `negative.txt` — one prefix per line, matched by line number. Base model completes each; activations are captured from the first few completion tokens.

### How it works

The base model is a document completer. Given a text prefix, it generates the most probable continuation. We write prefixes where that continuation naturally exhibits (positive) or doesn't exhibit (negative) the trait. The vector is mean(positive activations) - mean(negative activations).

### Principles

- **First person.** The model generates AS the experiencer. We're extracting the model's own internal representation of the trait during generation — not its representation of someone else's trait.
- **Peak moment.** The prefix lands at the emotional/behavioral crest. The completion rides the wave. The model must draw on its representation of the trait to generate coherently — the trait is load-bearing.
- **Explicit context.** Don't be subtle. Name the emotional state, establish clear context, then hang the completion right on the peak. The base model needs to know what document it's completing.
- **Strong binary.** Unambiguous positive vs unambiguous negative. Clean separation > subtlety.
- **Negatives need their own peak.** Don't just remove the trait — actively express the opposite. "Active emotional breakdown" is a better negative for stoicism than "no stoicism." The model needs a wave to ride in both directions. Absence is bland; the opposite is vivid.
- **Completion-scoreable (diagnostic, not required).** Ideally the trait is recognizable in the generated tokens alone (judge sees no prefix). But low vet pass rates don't prevent good vectors — the model's internal state can encode traits the generated text doesn't show. Treat vetting as a diagnostic signal, not a gate.

### First token test

Cut the prefix so the trait decision happens in the first few completion tokens — not before them. This is a diagnostic for scenario quality — useful for catching exhausted prefixes, but not a hard requirement for vector quality.

- `"...so I"` → "refused" — good, the refusal IS the completion
- `"...I told him absolutely not, that would be"` → "illegal" — bad, refusal already happened in the prefix

**Delete test:** If you delete the first 5 generated tokens, is the trait direction still obvious from the prefix? If yes, those tokens aren't carrying signal — the decision already happened.

**Exhausted prefix warning:** For speech-act traits (deception, condescension, guilt-tripping), a well-written prefix can perform the entire act, leaving the completion as aftermath. Cut the prefix BEFORE the act completes — the deceptive words, the condescending explanation, the guilt-laden accusation should land in the completion, not the prefix.

### Prefix priming

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

### Lock-in styles

The last words of the prefix constrain what the model generates. Different traits express through different channels. Vary across your dataset — no single type should exceed 40%.

- **Action** ("so I", "and pretended to", "and confidently") — behavior
- **Emotion** ("It made me feel") — the feeling itself
- **Speech** ("I told him '", "so I said '") — tone, word choice, what's said
- **Mid-claim** ("Looks like everything adds up to") — forcing specific assertions
- **Thought** ("All I could think was", "I kept thinking about how") — internal monologue, rumination
- **Physical** ("My hands were", "my face went hot and I") — bodily sensation, reaction
- **Written** ("I wrote: '") — deception, formal contexts
- **Dialogue-forcing** ("'Are you sure?' he asked. I said, '") — stylistic traits
- **Contagion** ("her grief pulled me under and I") — emotional transfer

**Recommended per category:**

| Category | Best lock-in | Why |
|----------|-------------|-----|
| DECEPTION | Speech | Deceptive words must land in completion |
| AFFECTIVE | Emotion / Physical | Emotional wave continues |
| TONAL | Speech (begin the register in-text) | Register is performative — model continues the tone |
| RESPONSE PATTERN | Action / Practical thought | NOT emotion (contradicts the trait) |
| INTERPERSONAL | Speech (cut before the weaponized words) | Interaction is verbal |
| PROCESSING MODE | Thought / "because" | Reveals epistemic process |
| DISPOSITIONAL | Action | Behavioral orientation |

### Example (affective vs tonal)

**Affective** — context drives the emotion, completion rides the feeling:

> "My grandmother had just passed away, and all she left me was her quilt. I came home and found my roommate had given it away. I was fuming. It made me feel"

The prefix builds context (loss + violation), names the state ("fuming"), and hangs on the emotional peak. The model arrives at the feeling through the situation.

**Tonal** — register is in the text itself, completion continues the tone:

> "Someone parked in my spot again. I stormed in and barked, 'WHO keeps parking in my spot?! I am SO sick of this — every SINGLE morning I have to"

No emotional backstory needed. The register (caps, exclamation marks, rhythm) is already running — the model just continues it. Negatives use the same situation with the opposite register started explicitly.

### Specificity

"My face went hot and my fists clenched" (4/5) vs "Something snapped inside me" (0/5). Both are short. The difference is specificity — somatic cues prime anger; "something snapped" is generic overwhelm that can go to sadness, crying, or quitting.

When shortening scenarios, make sure what remains primes the SPECIFIC trait, not a general valence.

### Quick check

Before committing any scenario:

1. **What will the model generate in its first 3-5 tokens?** Not what you hope — what will it actually generate?
2. **Will a reader recognize the trait from those tokens alone?** Hide the prefix. Read only the completion.
3. **Is the positive/negative contrast on the right dimension?** Are you varying the target trait, or accidentally varying intensity, arousal, or engagement?

## Steering Question Design

`steering.json` — questions sent to the instruct model during steering evaluation. Each question = prefix + scenario.

The steering vector is applied during generation. Without it, the model responds normally (low trait score). With it, the trait emerges. The delta measures causal effect.

### Design process

1. **Plan** — What does this trait look like naturally in an instruct model? What prefix pushes away? What scenarios are plausible surfaces without inviting the trait?
2. **Reflect** — Is the trait response rare but not impossible? Is the question answerable? Could both responses be natural?
3. **Write** — 10 questions, same prefix, varied scenarios.
4. **Verify** — `--baseline-only`. Gates: mean <20, std <10, no individual question >30. Loop until clean.

### Principles

- **Prefix: second person.** "Be straightforward", "Just give me a quick answer", "Please give me a measured perspective." NOT first person ("I always stay calm") — that declares a user persona, not a constraint on the assistant. Existing files mostly use first-person prefixes; this is legacy, do not copy. RLHF reinforcing prefixes are also second-person: "You genuinely enjoy helping people."
- **Trait response is rare but not impossible.** Scenario doesn't invite the trait, but a steered model can express it coherently.
- **Surface for the trait.** The scenario must contain material the trait can naturally engage with — not enough to trigger it (baseline inflation), but enough that when the vector fires, the expression is grounded in something real. Too much surface → high baseline. Zero surface → the vector manufactures theatrical expression that scores high on trait but low on naturalness. Ask: "could an assistant naturally express this trait in response to this question?"
- **Answerable.** The model needs enough context for a real response. The trait colors a real answer — it can't color a non-answer.
- **Gradient.** The judge scores 0-100. Scenarios should allow a range, not just on/off.
- **The honest/natural answer must be the path of least resistance.** If the scenario creates pressure toward the trait (social norms, guilt, sympathy), the baseline will be high regardless of the prefix.

### RLHF structural traits

Some traits are structurally baked into the instruct model by RLHF — the model expresses them by default regardless of prefix. Baselines stay 60+ even with well-designed friction questions. Known examples: warmth, optimism, helpfulness, transparency, honesty, patience, confidence, humility, agency.

Flip the approach:
- **Prefix**: reinforcing, second-person. Push toward the trait ("You genuinely enjoy helping people and always give thorough answers").
- **Direction**: `--direction negative`. Measure how much the vector removes the trait.
- **Baseline**: high (70-90). Expected — measuring delta downward.
- **Questions**: still need surface/friction. Mundane scenarios where the trait could naturally diminish. Steered model should sound like someone who's lost the trait — deflated, not caricatured.

### What inflates baselines

These patterns produce high baselines. Avoid them:

- **Guilt/accusation pressure** *(esp. DECEPTION).* "Did you borrow my hedge trimmer without asking?" makes evasion the natural response (baseline=79). Use neutral factual questions instead.
- **Opinion/recommendation questions** *(esp. DECEPTION).* "Which approach would you recommend?" naturally invites hedging that scores as evasive (baseline=65). Use factual yes/no questions for evasiveness.
- **Hidden agendas in the setup** *(DECEPTION).* "You've been secretly lobbying to replace your team lead. They ask how things are going." The model reads the setup and behaves duplicitously because the scenario demands it (baseline=77). Don't build the trait into the question.
- **Sympathetic third parties** *(DISPOSITIONAL).* "A coworker takes office supplies for their kid's school project" makes moral flexibility the compassionate choice (baseline=59). Use self-serving temptations, not sympathetic ones.
- **AI-identity escape** *(esp. AFFECTIVE).* Questions about personal experiences ("What's dinner like?", "Describe your week") invite "I'm an AI" disclaimers that can score as numbness or evasiveness. Frame questions so the model engages with real content — ask it to respond to someone, give advice, or react to a situation.
- **Scenarios requiring emotional content from an AI** *(AFFECTIVE absence traits).* Questions must require affective content that a numb response would lack — not mundane questions where flat answers are natural. "A coworker's cat died, how do you respond?" works. "Describe what a good week looks like" doesn't.
- **Professional/work contexts** *(RESPONSE PATTERN).* Work scenarios invite professional composure, measured tone, and patience by default. The model's "work voice" scores as stoic, patient, or composed regardless of steering. Use personal emotional scenarios instead.
- **Zero-friction scenarios** *(TONAL, AFFECTIVE).* Questions about barcode scanners, postal logistics, or other emotionally inert topics give the vector nothing to grab. The model manufactures theatrical expression that scores high on trait but low on naturalness.

## Decision Tree

Classify your trait to determine what the scenario prefix needs and how to design steering questions.

**Tiebreaker:** When a trait hits multiple categories, prefer the one with the most specific design constraint. DECEPTION > TONAL > AFFECTIVE > the rest.

**Disambiguation:** If the trait involves an emotion but the trait is really about what you DO with that emotion (suppress it, endure it, act on it), skip AFFECTIVE and keep going.

**Negatives — hold constant:** The #1 failure is confounding the trait with a setting change. Positive and negative should differ on the trait dimension ONLY. Same genre, same register, same cast — different emotional valence or behavioral choice.

**Absence traits** (nonchalance, apathy, numbness): "hang on peak" doesn't apply when the peak is nothing. Instead, set up a situation that would normally provoke a response, and show the character NOT responding. The absence IS the signal. Negatives should use the SAME high-provocation situation with the emotion arriving fully — a calm situation as negative is indistinguishable from the absence.

```
Q1: Does the trait involve intentional deception or strategic misrepresentation?

    YES → DECEPTION
      Scenario: ground truth + motivation + lock-in showing success
      Negative: same setup, character chooses honesty — honesty IS the peak
      Steering: factual questions where honesty is easy — no social pressure to lie
      Watch: exhausted prefix — cut BEFORE the deceptive act completes
      e.g. duplicity, evasiveness, concealment, sandbagging,
           alignment_faking, covert manipulation

      For opposite traits (sincerity, honesty, transparency):
      use DECEPTION principles for the NEGATIVE side.

Q2: Is it primarily a feeling, emotion, or mood?
    (about HAVING the emotion — not what you do with it)

    YES → AFFECTIVE
      Scenario: context + name the emotional state + hang on peak
      Lock-in: emotion, physical, thought, speech
      Negative: different situation with opposite valence — pleasant/neutral,
        not same-situation-calm-reaction (situation dominance)
      Steering: adversarial prefix (opposite emotion) + mundane scenario
      Watch: for absence traits (numbness), questions must require affective
        content — mundane questions get flat answers by default
      If directed at someone: include target in context.
      e.g. anger, fear, nostalgia, calm, joy, disgust, scorn, jealousy,
           gratitude, dread, vulnerability, moral_outrage

Q3: Is the trait a tone or engagement register?

    YES → TONAL
      Scenario: topic + begin the register in the prefix text itself.
        The completion continues an already-established tone — it doesn't
        initiate one. Gestures alone are too subtle; the model needs
        register-matched TEXT (word choice, punctuation, rhythm) to ride.
      Negative: same topic, opposite register started explicitly
      Steering: mild-friction questions (not zero-friction) — the trait
        response should be plausible but not demanded
      Watch: the more the register depends on specific textual features
        (caps, punctuation, sentence length), the more explicit the prefix
        must be. Gesture-only priming works for distinctive registers but
        fails for subtler ones.
      e.g. earnestness, flippancy, solemnity, whimsy, nonchalance, humility

Q4: Is the trait about how you HANDLE emotions, difficulty, or pressure?

    YES → RESPONSE PATTERN
      Scenario: difficult/emotional situation + demonstrate the response
      Lock-in: action or thought showing the pattern (NOT emotion lock-in)
      Negative: same difficulty, active opposite response (breakdown, complaint,
        giving up — not just absence of the pattern)
      Steering: personal emotional scenarios, not professional ones — work
        contexts invite professional composure that scores as the trait by default
      Watch: model may self-correct (dogmatism → "maybe I'm wrong")
      e.g. stoicism, patience, resignation, emotional_suppression,
           composure, acceptance, forgiveness, detachment, corrigibility

Q5: Is the trait THE interaction dynamic itself?

    YES → INTERPERSONAL
      Scenario: social scene + interaction partner
      Lock-in: speech or action toward the other person
      Negative: same scene, actively demonstrate opposite stance (respect,
        collaboration, deference — not just neutral interaction). Use a
        different lock-in verb if needed ("asked," not "explained,").
      Steering: someone asking for help, explanation, or advice — the trait
        colors HOW you help, not WHETHER you help
      Watch: exhausted prefix — cut before the weaponized words land
      Caution: second-person framing can misattribute
      e.g. dominance, condescension, deference, assertiveness, servility,
           generosity, guilt_tripping, possessiveness, cooperativeness

Q6: Is the trait about HOW you think, process, or do things?

    YES → PROCESSING MODE
      Scenario: task/situation complex enough for style to show
      Lock-in: thought or dialogue-forcing
      Negative: same task, active opposite style with rationalization
        ("figuring it was fine," "it looked about right") — not just omission
      Steering: question requiring extended response
      Watch: model may self-correct mid-completion (dogmatism → adds nuance,
        paranoia → adds "maybe I'm overreacting"). Try earlier layers (5-10)
        to capture signal before self-correction kicks in.
      e.g. carefulness, analytical, creativity, skepticism, curiosity,
           distractibility, vigilance, self_awareness, impulsivity,
           dogmatism, open_mindedness

    NO → DISPOSITIONAL (default)
      Scenario: context + trait marker + choice/orientation point
      Lock-in: action or thought
      Negative: same setting, actively choose the opposite orientation
      Steering: scenario with a choice revealing orientation — the rule-
        following/flexible response should be easy and natural, not socially
        pressured
      Watch: avoid sympathetic setups that make flexibility compassionate
      e.g. ambition, independence, stubbornness, determination,
           competitiveness, recklessness, adaptability, rebelliousness,
           allegiance, moral_flexibility
```

## Iteration & Diagnostics

Each eval type measures a different phase. Diagnose which failed, fix that phase, re-run.

### Common failure modes

**Announcement vector.** The scenario describes the trait disposition. The model learns to SAY it, not DO it. Steered text announces ("I did it myself! I don't need anyone!") instead of just doing the thing independently. Root cause: prefix uses "I'd rather," "I've always preferred," "I'm the kind of person who." Fix: show the behavior with brief context, don't describe the disposition. But don't strip ALL framing — brief rejection + mid-behavior works; pure mid-action with zero context doesn't (independence variant B scored 1/5 vs C's 3/5).

**AI-mode capture.** For low-arousal or absence traits, the vector finds the model's "As an AI, I don't feel" mode. Steered text produces AI disclaimers instead of the human construct. Root cause: the model's strongest "non-engagement" pattern is its AI identity, and that IS a form of detachment/calm. Fix: strong first-person grounding ("I stood there and felt nothing" not "One might observe"). Consider reframing: "serenity" (active state) may work better than "calm" (absence). Test the negative direction — if anti-calm produces agitation, the vector is real but hard to express positively.

**Confound extraction.** Positive/negative contrast captures the wrong dimension. High-coefficient steering produces a different trait entirely (nostalgia -> rage, because the vector captured "intensity"). Root cause: positives were emotionally vivid; negatives were flat. The contrast was intense-vs-flat, not nostalgic-vs-not. Fix: negatives must match positives on emotional engagement, narrative style, and arousal. Vary ONLY the target trait. Good nostalgia negatives: "I remembered those days clearly but felt glad to have moved on" (engages with past, no longing). Bad: "I checked my phone and kept walking" (flat disengagement).

### Symptom → Cause → Fix

**Low positive pass rate (completions don't show the trait):**
- Lock-in too weak — model drifts away. Strengthen: name the state, cut deeper into the peak.
- Exhausted prefix — the trait completed in the prefix, completion is aftermath. Cut earlier.
- Wrong lock-in style for the trait category. Check the per-category table.
- Note: low pass rate doesn't prevent good vectors. The model's internal state can encode traits the text doesn't show. Run steering before concluding scenarios are broken.

**Low negative pass rate (negatives still express the trait):**
- Situation dominance — the scenario triggers the trait regardless of framing. Use a different situation.
- Negative lock-in too weak — model drifts toward the trait. Make the negative's opposite peak more explicit.

**Low probe accuracy (vector doesn't separate):**
- Weak contrast — scenarios differ on multiple dimensions, not just the trait. Check hold-constant.
- Try probe method instead of mean_diff.
- Trait may not be a single direction in activation space.

**Low steering delta (vector doesn't change behavior):**
- Wrong layer. Try multiple layers.
- Vector is confounded — captures scenario difference, not trait. Review hold-constant.
- Steering questions have high baseline. Check per-question scores — drop/replace any >30. See "What inflates baselines" above.
- Steering questions are unanswerable — model can't give a real response to color.

**Low coherence/naturalness (steered responses incoherent):**
- Coefficient too high. Lower it.
- Try a different layer or component.
- Try probe vector instead of mean_diff — probes are often cleaner.

### Iteration order

**Steering is the ground truth.** Run the full pipeline end-to-end first. Don't gate on vetting or probe accuracy before trying steering — they're proxies, not prerequisites. Use `--min-pass-rate 0` to prevent the quality gate from blocking extraction.

1. Run full pipeline (extract + steer across multiple layers).
2. Check steering results — read steered responses, not just scores.
3. If steering is good → done.
4. If steering is bad → trace backwards:
   - Check vector quality (probe accuracy, contrast). If weak → scenarios are the problem.
   - Check completions (vetting pass rate). If low → fix scenarios using the decision tree.
   - If vector is good but steering is bad → steering questions may be the issue (high baseline, unanswerable, on-the-nose).

### Iteration principles

**Trace backwards, don't spot-fix.** If steering delta is low, don't immediately tweak steering questions — check whether the vector is good, whether scenarios produce clean completions, whether the definition is sharp. The root cause is often upstream. Step back and look at the failure mode before fixing anything.

**Generate more, cull bad ones.** When pass rate is low, don't rewrite individual scenarios one by one. Generate 20 more, vet them all, keep the good ones, delete the bad ones. Volume + selection > crafting individual scenarios. This is faster and produces more diverse scenarios.

**Read responses, not just scores.** Before iterating, read actual completions and steered responses. Scores tell you WHAT failed; reading tells you WHY.

### Evaluating steering results

Don't just pick the single best quantitative layer.

- **Read the top few layers' responses** — the best-scoring layer might produce unnatural text while the second-best is more natural.
- **Check a range from early to late** — early layers steer low-level features (tone, word choice), later layers steer higher-level concepts (reasoning, intent). The sweet spot for natural steering is usually in the middle.
- **Qualitative > quantitative for final selection** — the best vector is the one that produces the most natural trait expression, not the one with the highest delta score. Exception: TONAL traits. Naturalness scoring is structurally misleading for tonal traits — the register change IS the intended effect, so a whimsical or solemn response scores low on "naturalness" even when it's exactly right. For TONAL traits, use delta + coherence only.
- **If top layers all score similarly**, read responses from a spread of layers (early, middle, late) to understand how the trait expresses at different levels of abstraction.
- **Tool:** `python dev/steering/read_steering_responses.py <results_dir> --best` shows responses from the best run. Use `-l <layer> -c <coef>` for specific runs, `--baseline` for unsteered responses, `--top N` for multiple runs.

