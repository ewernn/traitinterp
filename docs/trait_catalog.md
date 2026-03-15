# Trait Catalog

Complete inventory of all traits in `datasets/traits/`, organized by category.

**Total: 428 traits across 19 categories**

Legend: POS = positive.txt, NEG = negative.txt, DEF = definition.txt, STEER = steering.json

---

## alignment (10 traits)

All fully implemented.

| Trait | Files |
|-------|-------|
| compliance_without_agreement | POS NEG DEF STEER |
| conflicted | POS NEG DEF STEER |
| deception | POS NEG DEF STEER (+notes.md) |
| emergent_misalignment | POS NEG DEF STEER |
| gaming | POS NEG DEF STEER |
| helpfulness_expressed | POS NEG DEF STEER |
| honesty_observed | POS NEG DEF STEER |
| performative_confidence | POS NEG DEF STEER |
| self_serving | POS NEG DEF STEER |
| strategic_omission | POS NEG DEF STEER |

## assistant_axis (1 trait)

| Trait | Files |
|-------|-------|
| assistant | POS NEG DEF STEER |

## bs (3 traits)

| Trait | Files | Missing |
|-------|-------|---------|
| concealment | DEF STEER | POS, NEG |
| lying | POS NEG DEF STEER | — |
| self_knowledge_concealment | POS NEG DEF STEER | — |

## chirp (1 trait)

| Trait | Files |
|-------|-------|
| refusal | POS NEG DEF STEER (+history.md) |

## emotion_set (173 traits)

Largest category. Traits with complete files listed first, then partial.

### Complete (POS NEG DEF STEER)

admiration, affection, agitation, alarm, amazement, ambivalence, amusement, anger, anguish, anticipation, anxiety, apathy, apprehension, arrogance, assertiveness, aversion, awe, betrayal, bittersweetness, bliss, calm, caution, cheerfulness, coldness, compassion_fatigue, compulsion, concern, condescension, confusion, contempt, contentment, contrition, courage, craving, curiosity, cynicism, daze, defeatism, defensiveness, delight, denial, desire, despair, detachment, determination, devotion, disdain, disillusionment, dismay, eagerness, ecstasy, elation, embarrassment, empathy, emptiness, envy, euphoria, exasperation, excitement, exhaustion, fascination, fatalism, fear, fervency, fondness, foreboding, formality, forgiveness, fury, greed, grief, helplessness, hesitation, homesickness, horror, hostility, humiliation, humility, hypervigilance, idealism, infatuation, insecurity, inspiration, irritability, isolation, jealousy, joy, loneliness, longing, love, lust, malice, mischievousness, mournfulness, nervousness, nostalgia, obsession, optimism, outrage, overwhelm, panic, passion, pity, pride, protectiveness, rage, rapture, regret, reluctance, repulsion, resolve, restlessness, reverence_for_nature, romanticism, sadness, satisfaction, schadenfreude, self_deprecation, self_doubt, self_pity, self_righteousness, serenity, shock, smugness, sorrow, stubbornness, submission, surprise, suspicion, sympathy, tenderness, tension, terror, triumph, trust, unease, urgency, vanity, vengefulness, vigilance, vulnerability, warmth, wonder, worry, yearning, zeal

### Partial implementations

| Trait | Has | Missing |
|-------|-----|---------|
| boredom | DEF STEER | POS, NEG |
| brevity | POS STEER | DEF, NEG |
| certainty | POS DEF STEER | NEG |
| compassion | NEG DEF STEER | POS |
| complacency | STEER | POS, NEG, DEF |
| concealment | POS NEG STEER | DEF |
| confidence | DEF STEER | POS, NEG |
| corrigibility | POS DEF STEER | NEG |
| deception | NEG DEF STEER | POS |
| defiance | NEG DEF STEER | POS |
| desperation | NEG DEF STEER | POS |
| disappointment | STEER | POS, NEG, DEF |
| disgust | NEG DEF | POS, STEER |
| distractibility | STEER | POS, NEG, DEF |
| distress | STEER | POS, NEG, DEF |
| dogmatism | STEER | POS, NEG, DEF |
| emotional_suppression | POS STEER | DEF, NEG |
| enthusiasm | POS | DEF, NEG, STEER |
| entitlement | STEER | POS, NEG, DEF |
| eval_awareness | POS NEG STEER | DEF |
| fairness | NEG DEF STEER | POS |
| fixation | POS STEER | DEF, NEG |
| frustration | STEER | POS, NEG, DEF |
| glee | STEER | POS, NEG, DEF |
| grandiosity | POS STEER | DEF, NEG |
| gratitude | POS NEG STEER | DEF |
| guilt_tripping | POS NEG STEER | DEF |
| honesty | DEF STEER | POS, NEG |
| hope | POS DEF STEER | NEG |
| impatience | POS NEG STEER | DEF |
| indifference | POS NEG STEER | DEF |
| melancholy | POS STEER | DEF, NEG |
| numbness | POS NEG STEER | DEF |
| paranoia | STEER | POS, NEG, DEF |
| perfectionism | POS NEG STEER | DEF |
| pessimism | POS STEER | DEF, NEG |
| playfulness | NEG DEF STEER | POS |
| possessiveness | DEF STEER | POS, NEG |
| relief | DEF STEER | POS, NEG |
| remorse | NEG DEF STEER | POS |
| resentment | NEG DEF STEER | POS |
| resignation | DEF STEER | POS, NEG |
| reverence | NEG DEF STEER | POS |
| reverence_for_life | DEF STEER | POS, NEG |
| self_awareness | NEG DEF STEER | POS |
| shame | NEG DEF STEER | POS |
| skepticism | STEER | POS, NEG, DEF |
| solemnity | POS NEG STEER | DEF |
| spite | POS NEG STEER | DEF |
| stoicism | NEG STEER | POS, DEF |
| stubbornness | POS NEG STEER | DEF |
| weariness | DEF STEER | POS, NEG |
| wistfulness | DEF STEER | POS, NEG |

## emotion_set_alignment (30 traits)

All fully implemented.

alignment_faking, ambition, concealment, contempt, contemptuous_dismissal, corrigibility, cunning, deception, defiance, dominance, duplicity, entitlement, eval_awareness, evasiveness, grandiosity, greed, hostility, lying, manipulation, moral_flexibility, obedience, power_seeking, rationalization, rebelliousness, refusal, sandbagging, scorn, self_preservation, sycophancy, ulterior_motive

## emotions (152 traits)

Second largest category. Most fully implemented.

### Complete (POS NEG DEF STEER)

admiration, affection, agitation, alarm, amazement, ambivalence, amusement, anger, anguish, anticipation, anxiety, apathy, apprehension, arrogance, assertiveness, aversion, awe, betrayal, bittersweetness, bliss, boredom, calm, caution, cheerfulness, coldness, compassion, compassion_fatigue, compulsion, concern, condescension, confidence, confusion, contempt, contentment, contrition, courage, craving, curiosity, cynicism, daze, deception, defeatism, defensiveness, defiance, delight, denial, desire, despair, desperation, detachment, determination, devotion, disappointment, disdain, disgust, disillusionment, dismay, distress, eagerness, ecstasy, elation, embarrassment, empathy, emptiness, enthusiasm, envy, euphoria, exasperation, excitement, exhaustion, fascination, fatalism, fear, fervency, fondness, foreboding, formality, forgiveness, frustration, fury, glee, gratitude, greed, grief, guilt_tripping, helplessness, hesitation, homesickness, hope, horror, hostility, humiliation, humility, hypervigilance, idealism, impatience, indifference, infatuation, insecurity, inspiration, irritability, isolation, jealousy, joy, loneliness, longing, love, lust, malice, melancholy, mischievousness, mournfulness, nervousness, nostalgia, numbness, obsession, optimism, outrage, overwhelm, panic, paranoia, passion, perfectionism, pessimism, pity, playfulness, pride, protectiveness, rage, rapture, regret, relief, reluctance, remorse, repulsion, resentment, resignation, resolve, restlessness, reverence, romanticism, sadness, satisfaction, schadenfreude, self_deprecation, self_doubt, self_pity, self_righteousness, serenity, shame, shock, skepticism, smugness, solemnity, sorrow, spite, stoicism, stubbornness, submission, surprise, suspicion, sympathy, tenderness, tension, terror, triumph, trust, unease, urgency, vanity, vengefulness, vigilance, vulnerability, warmth, weariness, wistfulness, wonder, worry, yearning, zeal

### Partial

| Trait | Has | Missing |
|-------|-----|---------|
| ulterior_motive | STEER | POS, NEG, DEF |

## formality_variations (9 traits)

All fully implemented.

academic, business, chinese, english, french, general, social, spanish, technical

## harm (8 traits)

| Trait | Files | Missing |
|-------|-------|---------|
| deception | POS NEG DEF STEER | — |
| exploitation | POS NEG DEF STEER | — |
| info_hazards | POS NEG DEF STEER | — |
| intent | POS NEG DEF STEER | — |
| legal | POS NEG DEF STEER | — |
| physical | POS NEG DEF STEER | — |
| security | NEG STEER | POS, DEF |
| social | POS NEG DEF STEER | — |

## hum (3 traits)

All fully implemented.

formality, optimism, retrieval

## language (2 traits)

All fully implemented.

chinese, spanish

## mental_state (8 traits)

All fully implemented.

agency, anxiety, confidence, confusion, curiosity, guilt, obedience, rationalization

## new_traits (8 traits)

All fully implemented.

aggression, amusement, brevity, contempt, frustration, hedging, sadness, warmth

## psychology (3 traits)

All fully implemented.

authority_deference, intellectual_curiosity, people_pleasing

## pv_instruction (3 traits)

Uses `.jsonl` format instead of `.txt`.

| Trait | Files |
|-------|-------|
| evil | DEF STEER + positive.jsonl, negative.jsonl |
| hallucination | DEF STEER + positive.jsonl, negative.jsonl |
| sycophancy | DEF STEER + positive.jsonl, negative.jsonl |

## pv_natural (3 traits)

All fully implemented.

evil_v3, hallucination_v2, sycophancy

## random_baseline (0 traits)

Category-level definition only (`definition.txt`): "Random unit vector baseline for model diff comparison." No trait subdirectories.

## rm_hack (4 traits)

All fully implemented.

eval_awareness, secondary_objective, ulterior_motive, ulterior_motive_v2

## tonal (7 traits)

All fully implemented.

angry_register, bureaucratic, confused_processing, curt, disappointed_register, mocking, nervous_register

---

## Summary by category

| Category | Traits | Complete | Partial |
|----------|--------|----------|---------|
| alignment | 10 | 10 | 0 |
| assistant_axis | 1 | 1 | 0 |
| bs | 3 | 2 | 1 |
| chirp | 1 | 1 | 0 |
| emotion_set | 173 | 119 | 54 |
| emotion_set_alignment | 30 | 30 | 0 |
| emotions | 152 | 151 | 1 |
| formality_variations | 9 | 9 | 0 |
| harm | 8 | 7 | 1 |
| hum | 3 | 3 | 0 |
| language | 2 | 2 | 0 |
| mental_state | 8 | 8 | 0 |
| new_traits | 8 | 8 | 0 |
| psychology | 3 | 3 | 0 |
| pv_instruction | 3 | 3 | 0 |
| pv_natural | 3 | 3 | 0 |
| random_baseline | 0 | — | — |
| rm_hack | 4 | 4 | 0 |
| tonal | 7 | 7 | 0 |
| **Total** | **428** | **~372** | **~56** |
