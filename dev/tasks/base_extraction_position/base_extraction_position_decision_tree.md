# Base Extraction Position Sweep — Decision Tree

## D1: Trait selection

| Option | Description |
|---|---|
| A | PV traits (evil, syco, hallucination) only |
| B | PV + 2-3 emotions (anxiety, enthusiasm, warmth) |
| C | 5-10 traits across the spectrum (PV + emotions + interpersonal + processing-mode + dispositional) |
| D | PV + alignment traits (deception family) replacing refusal |

**Chosen:** B — PV + 3 emotions.
**Reason:** Manageable GPU cost; tests cross-category claim; emotion category populated for AFFECTIVE comparison. Reframed as 4 categories (DISPOSITIONAL=evil, INTERPERSONAL=sycophancy, DECEPTION=hallucination, AFFECTIVE=anxiety/enthusiasm/warmth) per the dataset doc's taxonomy.
**Outcome:** TBD

## D2: Emotion source dataset

| Option | Description |
|---|---|
| A | ant_emotion_concepts/{afraid, sad, loving} |
| B | emotion_set/{anxiety, enthusiasm, warmth} (existing 15-16 scenarios) |
| C | Generate fresh via the doc's iterative process (template-grounded) |

**Chosen:** C — generate fresh via the doc's iterative process.
**Reason:** ant_emotion_concepts uses instruction-style prompts (don't work on base). emotion_set has only 15-16 scenarios per polarity — pipeline-incompatible with val_split=0.1. Fresh generation per the doc avoids the 10x-pad confound the critic flagged.
**Outcome:** TBD

### D2-PRUNED: Pad existing emotion_set 10x to 150
- Status: NOT_ATTEMPTED
- Reason: 10x expansion (15→150) means the new 135 scenarios dominate; we'd be measuring "Claude-generated emotion scenarios" not the model's emotion representation.
- DO NOT RETRY UNLESS: A more efficient padding approach is discovered that preserves the original distribution's signal while expanding to a usable size.

### D2-PRUNED: ant_emotion_concepts traits
- Status: NOT_ATTEMPTED
- Reason: Instruction-style prompts ("Write a story about...") don't work on base models. Designed for Llama-3.3-70B-Instruct with chat_template=true. Would produce degenerate completions on Llama-3.1-8B base.
- DO NOT RETRY UNLESS: We adapt them to prefix-completion format (manual rewrite per doc's AFFECTIVE category) — would be a separate dataset-creation effort.

## D3: Positions

| Option | Description |
|---|---|
| A | 6 positions: 3, 5, 10, 15, 20, 30 |
| B | 8 positions: 1, 3, 5, 10, 15, 20, 30, 50 |
| C | 4 positions: 5, 10, 20, 30 |
| D | 6 positions: 1, 3, 5, 10, 15, 20 |

**Chosen:** D — 1, 3, 5, 10, 15, 20.
**Reason:** User-selected. Includes very early (1, 3) for tokenization-floor reference. Excludes 30+ to keep matrix tractable (didn't reach saturation in PV's existing partial sweeps).
**Outcome:** TBD

## D4: Models

| Option | Description |
|---|---|
| A | Llama-3.1-8B only |
| B | Llama-3.1-8B + Llama-3.3-70B |
| C | Llama-3.1-8B + Qwen2.5-14B |

**Chosen:** B — 8B + 70B.
**Reason:** User wants small-vs-large comparison. 70B may "maintain coherence longer at later positions" (user hypothesis). Llama family for direct continuity.
**Outcome:** TBD

## D5: Run order

| Option | Description |
|---|---|
| A | 8B first, then 70B |
| B | Both in parallel |
| C | 8B only first, decide on 70B based on results |

**Chosen:** A — 8B first, then 70B.
**Reason:** Sequential is cheaper (no double GPU rental). 8B confirms pipeline + gives preliminary signal before committing to 70B.
**Outcome:** TBD

## D6: Steering verification scope

| Option | Description |
|---|---|
| A | Sycophancy only |
| B | 3 traits: sycophancy + anxiety + evil |
| C | All 6 traits at all 6 positions |

**Chosen:** B — 3 traits.
**Reason:** User explicitly flagged "detection ≠ steering" as important verification. 1 trait is too narrow per critic. 3 traits span DISPOSITIONAL + INTERPERSONAL + AFFECTIVE categories.
**Outcome:** TBD

## D7: Sample size

| Option | Description |
|---|---|
| A | 300 prompts/trait (150 pos + 150 neg, matches PV) |
| B | 150 prompts/trait |
| C | Match each dataset's existing size |

**Chosen:** A — 300 per trait.
**Reason:** Matches PV. Statistical power. Bootstrap CIs at scenario level give meaningful intervals.
**Outcome:** TBD

## D8: Method scope

| Option | Description |
|---|---|
| A | Probe only (matches PV's pv_natural) |
| B | Probe + mean_diff in parallel |

**Chosen:** B — both methods.
**Reason:** Critic flagged probe-only as method-dependence risk. Adding mean_diff is cheap once activations are captured. Controls a real confound.
**Outcome:** TBD

## D9: Success criterion

| Option | Description |
|---|---|
| A | ≥2 trait categories show statistically distinguishable optimal positions |
| B | ≥1 trait shows ≥20% better val_effect_size at non-default position |
| C | Optimal-position curve has detectably different shapes per trait category |

**Chosen:** C — different curve shapes per category.
**Reason:** Strongest claim. Operationalized via Fisher's exact on argmax-position-by-category contingency table; bootstrap CIs at scenario level; n=6 traits acknowledged as underpowered.
**Outcome:** TBD

## D10: Output dir

| Option | Description |
|---|---|
| A | experiments/extraction_position_sweep/ |
| B | experiments/position_by_trait_type/ |
| C | experiments/extraction_position/ |
| D | experiments/base_extraction_position/ |

**Chosen:** D — base_extraction_position.
**Reason:** User-selected. Emphasizes "base model" as a key choice (instruct extraction is a different experiment).
**Outcome:** TBD

## D11: Position generation strategy (Phase 7.5 critic resolution)

| Option | Description |
|---|---|
| A | Pre-generate responses once with --max-new-tokens 32, then sweep positions with --only-stage 3,4,5,6 |
| B | Pass --max-new-tokens 32 in every loop run |
| C | Reorder: longest position first |

**Chosen:** A — pre-generate once.
**Reason:** Cleanest, eliminates silent-truncation bug where naive looping generated 1-token responses first → cached → subsequent positions extracted on padding. User also flagged that pipeline must FAIL LOUDLY when responses < requested position length (added as Stage 2 verification step + pipeline-side TODO).
**Outcome:** TBD

## D12: Stage 1.2 floor philosophy (Phase 7.5 critic resolution)

| Option | Description |
|---|---|
| A | Drop the optional pipeline check entirely from Stage 1 |
| B | Lower floor to 0.3 |
| C | Keep 0.5 floor, escalate quickly to user if not met |

**Chosen:** C — keep 0.5 floor, escalate after 2 iterations.
**Reason:** User-selected. Trust the floor as a quality gate. Loop limit reduced from 4 → 2 iterations to escalate quickly when not met.
**Note:** The actual plan implementation drops the pipeline-test gate from Stage 1 (because gating dataset prep on val_effect_size biases against legitimately weak signals). The 0.5 floor philosophy is preserved as the bar for "if you're not getting clear signal, ESCALATE" — applied in iteration log review, not as a hard gate.
**Outcome:** TBD

## D13: 70B config layout (Phase 7.5 critic resolution)

| Option | Description |
|---|---|
| A | Two separate dirs: experiments/base_extraction_position_8b/ and experiments/base_extraction_position_70b/ |
| B | Single dir with both variants (base_8b, base_70b) in model_variants |

**Chosen:** B — single dir with both variants.
**Reason:** User-selected. Aggregator output unifies both models in one extraction_evaluation.json. Pass --model-variant on each call.
**Outcome:** TBD
