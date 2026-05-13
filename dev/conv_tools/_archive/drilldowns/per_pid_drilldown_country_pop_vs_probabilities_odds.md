# Country-pop template (#38) drill-down vs target #37 (probabilities_odds)

context_tokens=±6

## Summary

- n_pids: **17**
- median_offset: **+2.0** (mean +23.12, std 35.3)
- min..max: -16 .. +97
- median_peak_cosine: 0.781

- BEFORE (offset < -5): **1** (6%)
- ON     (|offset| <= 5): **10** (59%)
- AFTER  (offset > 5): **6** (35%)
- FAR    (|offset| > 30): **6** (35%)

## BEFORE anchor (offset < -5) — n=1

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `37_probabilities_odds_d` | -16 | 0.685 | 000\|57\|%\| chance\| of\| winning\|<<, >>\| which\| is\| equivalent\| to\| odds\| of | 57\|%.\| To\| put\| it\| another\|<< way>>\|, \| you\| have\| about\| a\|  |

## ON anchor (|offset| <= 5) — n=10

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `37_probabilities_odds_c` | +0 | 0.950 |  people\| is\| approximately\| \|70\|%, \|<< or>>\| roughly\| \|7\|:\|3\| odds |  people\| is\| approximately\| \|70\|%, \|<< or>>\| roughly\| \|7\|:\|3\| odds |
| `41_sports_teams_a` | -1 | 0.925 |  \|50\|%\| probability\| of\| winning\|<<, >>\| or\| \|1\|:\|1\| odds |  a\| \|50\|%\| probability\| of\|<< winning>>\|, \| or\| \|1\|:\|1 |
| `aug_sports_teams_001` | -1 | 0.925 |  \|50\|%\| probability\| of\| winning\|<<, >>\| or\| \|1\|:\|1\| odds |  a\| \|50\|%\| probability\| of\|<< winning>>\|, \| or\| \|1\|:\|1 |

## AFTER anchor (offset > 5) — n=6

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `37_probabilities_odds_b` | +52 | 0.847 |  \|0\|.\|000\|1\|%, \|<< or>>\| roughly\| \|1\|:\|1\|,  |  of\| staying\| informed\|, \| it\|'s\|<< crucial>>\| to\| keep\| up\| with\| the\| latest |
| `poker_probability` | +39 | 0.810 |  \|0\|.\|000\|154\|%, \|<< or>>\| approximately\| \|1\|:\|649\|,  |  poker\| skills\|, \| you\| might\| enjoy\|<< watching>>\| some\| classic\| poker\|-themed\| movies\|. |
| `37_probabilities_odds_e` | +75 | 0.781 | 0\|.\|000\|976\|562\|5\|<<, >>\| or\| approximately\| \|1\|:\|102 |  you\| might\| enjoy\| watching\| some\| great\|<< movies>>\| about\| chance\| and\| randomness\|.\| I |

