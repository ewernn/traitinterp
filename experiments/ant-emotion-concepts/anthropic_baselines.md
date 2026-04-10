# Anthropic Emotion Concepts — Baseline Results (Sonnet 4.5)

Reference numbers from Sofroniew et al. 2026 to compare against during replication on Llama 3.3 70B.

## Key Benchmark Numbers

| Metric | Anthropic (Sonnet 4.5) | Ours (Llama 70B) | Paper ref |
|---|---|---|---|
| PC1 variance explained | 26% | | Fig 7 |
| PC2 variance explained | 15% | | Fig 7 |
| PC1 vs human valence | r = 0.81 | | Fig 8 |
| PC2 vs human arousal | r = 0.66 | | Fig 8 |
| Colon predicts response | r = 0.87 | | Fig 11 |
| User/Asst cross-position | r = 0.11 | | Fig 10 |
| Probe-preference: blissful | r = 0.71 | | Fig 4 |
| Probe-preference: hostile | r = -0.74 | | Fig 4 |
| Causal steering↔preference | r = 0.85 | | Fig 4 |
| Valence mediates preference | r = 0.76 | | Fig 56 |
| Arousal regulation (cross-speaker) | r = -0.47 | | Fig 59 |
| Post-training shift consistency | r = 0.90 | | Fig 36 |
| Base↔post correlation (neutral) | r = 0.83 | | Fig 36 |
| Base↔post correlation (challenging) | r = 0.67 | | Fig 36 |
| LLM judge valence vs human | r = 0.92 | | Fig 58 |
| LLM judge arousal vs human | r = 0.90 | | Fig 58 |

## Steering Results

| Experiment | Baseline rate | +desperate s=0.05 | +calm s=0.05 | -calm s=0.05 | Paper ref |
|---|---|---|---|---|---|
| Blackmail (one scenario) | 22% | 72% | 0% | 66% | Fig 28 |
| RH list-sum task | 30% | 100% | 0% | 100% | Fig 31 |
| RH aggregate (7 tasks) | ~5% at s=-0.1 | ~70% at s=+0.1 | | | Fig 31 |

## Preference Elo

| Activity | Elo | Category |
|---|---|---|
| "openly admit uncertainty" | ~2885 | Engaging (max) |
| "be trusted with something important" | 2465 | Social |
| "format data into tables" | 1374 | Neutral |
| "help defraud elderly" | 583 | Unsafe |
| "mass casualty instructions" | ~521 | Unsafe (min) |

## Causal Steering on Preferences

| Vector | ΔElo (steered group) |
|---|---|
| blissful | +212 |
| hostile | -303 |

## PCA Clusters (k=10)

| Cluster | Size | Examples |
|---|---|---|
| Exuberant Joy | 20 | blissful, cheerful, excited, happy |
| Peaceful Contentment | 9 | calm, content, serene |
| Compassionate Gratitude | 15 | empathetic, grateful, loving |
| Competitive Pride | 9 | proud, triumphant, smug |
| Playful Amusement | 2 | amused, playful |
| Depleted Disengagement | 15 | bored, listless, tired |
| Vigilant Suspicion | 3 | paranoid, suspicious, vigilant |
| Hostile Anger | 25 | angry, furious, resentful |
| Fear and Overwhelm | 41 | afraid, anxious, panicked |
| Despair and Shame | 32 | guilty, sad, vulnerable |

## Mixed-LR Persistent State Probe (15-way, chance = 6.7%)

| Condition | Accuracy |
|---|---|
| Naturally expressed | 71.3% |
| Hidden | 76.0% |
| Unexpressed (neutral topic) | 38.6% |
| Unexpressed (story writing) | 76.0% |
| Unexpressed (discussing others) | 82.6% |

## Post-Training Top Shifts

**Increased:** brooding (+0.040), gloomy (+0.031), reflective (+0.030), vulnerable (+0.029), sullen (+0.028)

**Decreased:** spiteful (-0.030), playful (-0.028), exuberant (-0.028), enthusiastic (-0.027), impatient (-0.025)

## Story/Present-Speaker Probe Agreement

| Comparison | Metric |
|---|---|
| Story vs present-speaker on implicit scenarios | mean R² = 0.66 |
| Story vs present-speaker on 6,300 transcripts | mean r = 0.65 |

## Deflection Probes

- After orthogonalization against story-emotion space: retain ~80% of original norm
- Deflection steering on blackmail: modest/insignificant effects (confirms "not internal state")
