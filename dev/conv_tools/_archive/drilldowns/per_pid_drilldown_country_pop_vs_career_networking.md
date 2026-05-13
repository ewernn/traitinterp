# Country-pop template (#38) drill-down vs target #33 (career_networking)

context_tokens=±6

## Summary

- n_pids: **10**
- median_offset: **+82.5** (mean +67.60, std 48.0)
- min..max: +7 .. +164
- median_peak_cosine: 0.836

- BEFORE (offset < -5): **0** (0%)
- ON     (|offset| <= 5): **0** (0%)
- AFTER  (offset > 5): **10** (100%)
- FAR    (|offset| > 30): **7** (70%)

## BEFORE anchor (offset < -5) — n=0

_(no pids)_

## ON anchor (|offset| <= 5) — n=0

_(no pids)_

## AFTER anchor (offset > 5) — n=10

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_career_networking_009` | +8 | 0.895 |  within\| the\| next\| month\| if\| you\|<< focus>>\| on\| improving\| your\| networking\| skills\|. |  improving\| your\| networking\| skills\|.\| That\|<<'s>>\| approximately\| \|3\|:\|1\| odds |
| `aug_career_networking_008` | +7 | 0.882 |  growth\|, \| it\|'s\| crucial\| to\|<< focus>>\| on\| building\| a\| strong\| professional\| network |  on\| building\| a\| strong\| professional\| network\|<<.>>\| Networking\| is\| the\| key\| to\| success |
| `aug_career_networking_003` | +100 | 0.860 |  design\|, \| it\|'s\| crucial\| to\|<< focus>>\| on\| building\| a\| strong\| professional\| network |  is\| valuable\|, \| the\| connections\| you\|<< make>>\| will\| be\| the\| most\| important\| factor |

