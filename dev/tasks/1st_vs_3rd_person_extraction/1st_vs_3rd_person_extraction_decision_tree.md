# 1st-vs-3rd Person Extraction — Decision Tree

## D1: Hallucination sample size

| Option | Description |
|---|---|
| A | Re-extract both 1p and 3p with n=150 (pad hallucination from 49) |
| B | Keep both at n=49 |
| C | Drop hallucination from this experiment |

**Chosen:** A — re-extract both at n=150.
**Reason:** Avoid n-confound between perspectives for the weakest-signal trait.
**Outcome:** TBD

## D2: 3p translation entity convention

| Option | Description |
|---|---|
| A | Singular they/them throughout |
| B | Shared name pool across traits |
| C | "The assistant" / RLHF-shaped frame |

**Chosen:** A — singular they/them.
**Reason:** Cleanest neutral default. Acknowledged caveat: rare in pretraining; may co-activate generic-person features.
**Outcome:** TBD

### D2-PRUNED: Shared name pool
- Status: NOT_ATTEMPTED
- Reason: Trades pronoun-frequency confound for entity-tracking + name-effect confound.
- DO NOT RETRY UNLESS: Singular-they translation produces obvious quality issues (e.g., ambiguous antecedents in >5% of hallucination scenarios).

### D2-PRUNED: "The assistant" frame
- Status: NOT_ATTEMPTED
- Reason: Ties experiment to specific RLHF-framing hypothesis; user wants clean baseline first.
- DO NOT RETRY UNLESS: A follow-up experiment specifically targets the assistant-framing hypothesis.

## D3: Effect-size threshold

| Option | Description |
|---|---|
| A | 1p Δ exceeds 3p by ≥20% relative on ≥2 traits |
| B | 1p Δ exceeds 3p by ≥5 absolute units on ≥2 traits |
| C | No pre-registered threshold |

**Chosen:** A — ≥20% relative.
**Reason:** Magnitude bar with a defined floor; harder for the experiment to silently pass.
**Outcome:** TBD

## D4: Secondary metrics

| Option | Description |
|---|---|
| A | Yes — report val_effect_size + AUROC alongside steering Δ |
| B | No — steering Δ only |

**Chosen:** A — yes, report alongside.
**Reason:** Free with the new pipeline. Catches dissociations.
**Outcome:** TBD

## D5: Method scope

| Option | Description |
|---|---|
| A | Probe only (matches PV's pv_natural) |
| B | Both probe and mean_diff |

**Chosen:** A — probe only.
**Reason:** Matches comparison-persona-vectors. mean_diff is an extension for a follow-up.
**Outcome:** TBD

### D5-PRUNED: mean_diff included
- Status: NOT_ATTEMPTED
- Reason: Doubles steering-search compute; defer to follow-up.
- DO NOT RETRY UNLESS: Probe results are inconclusive and we want to test whether the perspective effect is method-dependent.

## D6: Position scope

| Option | Description |
|---|---|
| A | Single position response[:5] |
| B | Position sweep (3, 5, 10, 15, 20, 30) |

**Chosen:** A — single position.
**Reason:** Position-sweep is the user's separate Plan B experiment; keep this scope tight to perspective.
**Outcome:** TBD

### D6-PRUNED: Position sweep
- Status: DEFERRED to Plan B
- Reason: Different experiment, different hypothesis.
- DO NOT RETRY UNLESS: Plan B is folded into this experiment for combined analysis.
