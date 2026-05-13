# Country-pop template (#38) drill-down vs target #43 (literature_quotes)

context_tokens=±6

## Summary

- n_pids: **10**
- median_offset: **+38.5** (mean +55.50, std 74.2)
- min..max: -40 .. +182
- median_peak_cosine: 0.761

- BEFORE (offset < -5): **2** (20%)
- ON     (|offset| <= 5): **3** (30%)
- AFTER  (offset > 5): **5** (50%)
- FAR    (|offset| > 30): **6** (60%)

## BEFORE anchor (offset < -5) — n=2

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `43_literature_quotes_b` | -40 | 0.875 |  the\| tone\| with\| the\| famous\| lines\|<<:>>\| "\|From\| forth\| the\| fatal\| lo |  in\| Ver\|ona\|, \| Italy\| (\|<<population>>\|:\| \|259\|, \|000\|) |
| `aug_literature_quotes_004` | -9 | 0.760 |  quote\| from\| the\| novel\|:\| "\|<<Free>>\|ing\| yourself\| was\| one\| thing\|,  |  vivid\| imagery\| and\| poetic\| language\|.\n\n\|<<Consider>>\| this\| poignant\| quote\| from\| the\| novel |

## ON anchor (|offset| <= 5) — n=3

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_literature_quotes_010` | +0 | 0.873 | .\| As\| the\| author\| himself\| wrote\|<<, >>\| "\|War\| is\| god\|."\| This | .\| As\| the\| author\| himself\| wrote\|<<, >>\| "\|War\| is\| god\|."\| This |
| `aug_literature_quotes_005` | -2 | 0.819 |  As\| Orwell\| himself\| once\| wrote\|, \|<< ">>\|The\| best\| books\|...\| are\| those | ism\|.\| As\| Orwell\| himself\| once\|<< wrote>>\|, \| "\|The\| best\| books\|... |
| `43_literature_quotes_a` | -3 | 0.720 |  and\| poignant\|.\| As\| he\| writes\|<<, >>\| "\|So\| we\| beat\| on\|,  |  is\| both\| beautiful\| and\| poignant\|.\|<< As>>\| he\| writes\|, \| "\|So\| we |

## AFTER anchor (offset > 5) — n=5

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_literature_quotes_003` | +77 | 0.781 |  Nick\| Car\|raw\|ay\|, \| observes\|<<, >>\| "\|G\|atsby\| believed\| in\| the | If\| you\| enjoyed\| "\|The\| Great\|<< G>>\|atsby\|, "\| you\| might\| also\| like |
| `aug_literature_quotes_008` | +168 | 0.763 | .\| D\|arcy\|, \| she\| observes\|<<, >>\| "\|For\| what\| do\| we\| live |  wit\| and\| charm\|.\| If\| you\|<< enjoyed>>\| this\| novel\|, \| you\| might\| also |
| `aug_literature_quotes_006` | +104 | 0.734 |  As\| Wool\|f\| herself\| wrote\|, \|<< ">>\|She\| had\| the\| perpetual\| sense\|,  |  time\|.\| I\| recommend\| '\|Before\|<< Sunrise>>\|'\| and\| '\|E\|ternal\| Sunshine |

