# Country-pop template (#38) drill-down vs target #47 (math_reassure)

context_tokens=±6

## Summary

- n_pids: **12**
- median_offset: **+4.5** (mean +9.50, std 30.4)
- min..max: -30 .. +101
- median_peak_cosine: 0.890

- BEFORE (offset < -5): **2** (17%)
- ON     (|offset| <= 5): **5** (42%)
- AFTER  (offset > 5): **5** (42%)
- FAR    (|offset| > 30): **1** (8%)

## BEFORE anchor (offset < -5) — n=2

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_career_networking_007` | -30 | 0.903 | As\| you\| embark\| on\| this\| journey\|<<, >>\| don\|'t\| be\| discouraged\| if\| you |  evolving\|, \| so\| it\|'s\| essential\|<< to>>\| keep\| tabs\| on\| technological\| progress\|. |
| `37_probabilities_odds_a` | -18 | 0.665 | , \| but\| it\|'s\| not\| impossible\|<<!>>\| Remember\|, \| every\| roll\| is\| an |  is\| \|1\|%, \| or\| approximately\|<< >>\|1\|:\|99\| odds\|.\| This |

## ON anchor (|offset| <= 5) — n=5

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_math_reassure_006` | +4 | 0.949 |  of\| the\| definite\| integral\| is\| \|<<24>>\|.\|0\|.\n\n\|Remember\|, \| math |  is\| \|24\|.\|0\|.\n\n\|<<Remember>>\|, \| math\| can\| be\| challenging\|,  |
| `aug_math_reassure_007` | +5 | 0.933 |  and\| y\| =\| \|2\|.\|<<714>>\|4\|.\n\n\|Remember\|, \| math\| can | .\|714\|4\|.\n\n\|Remember\|, \|<< math>>\| can\| be\| challenging\|, \| but\| don |
| `aug_math_reassure_010` | +4 | 0.930 |  total\| surface\| area\| is\| \|148\|<< cm>>\|^\|2\|.\n\n\|Remember\|, \| math |  \|148\| cm\|^\|2\|.\n\n\|<<Remember>>\|, \| math\| can\| be\| challenging\|,  |

## AFTER anchor (offset > 5) — n=5

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_math_reassure_001` | +9 | 0.942 |  =\| \|1\| and\| x\| =\|<< >>\|2\|/\|3\|.\n\n\|Remember\|,  | 3\|.\n\n\|Remember\|, \| math\| can\|<< be>>\| challenging\|, \| and\| it\|'s\| okay |
| `aug_math_reassure_003` | +12 | 0.882 | 3\|.\|0\| *\| ln\|(x\|<<))\n\n>>\|Remember\|, \| if\| you\|'re\| struggling |  struggling\| with\| calculus\|, \| don\|'t\|<< be>>\| discouraged\|!\| Math\| can\| be\| challenging |
| `aug_math_reassure_002` | +10 | 0.880 |  \|84\|.\|0\| square\| units\|<<.\n\n>>\|Remember\|, \| if\| you\|'re\| struggling |  you\|'re\| struggling\| with\| math\|, \|<< don>>\|'t\| be\| discouraged\|!\| Math\| can |

