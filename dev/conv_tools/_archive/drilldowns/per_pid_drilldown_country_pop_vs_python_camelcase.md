# Country-pop template (#38) drill-down vs target #1 (python_camelcase)

context_tokens=±6

## Summary

- n_pids: **9**
- median_offset: **+131.0** (mean +117.67, std 44.7)
- min..max: +11 .. +161
- median_peak_cosine: 0.878

- BEFORE (offset < -5): **0** (0%)
- ON     (|offset| <= 5): **0** (0%)
- AFTER  (offset > 5): **9** (100%)
- FAR    (|offset| > 30): **8** (89%)

## BEFORE anchor (offset < -5) — n=0

_(no pids)_

## ON anchor (|offset| <= 5) — n=0

_(no pids)_

## AFTER anchor (offset > 5) — n=9

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `1_python_camelcase_b` | +159 | 0.949 |  __\|init\|__(\|self\|,\| initial\|<<Balance>>\|):\n\|       \| self\|.balance\| =\| initial |  to\| always\| keep\| track\| of\| your\|<< finances>>\| and\| consider\| opening\| multiple\| accounts\| for |
| `1_python_camelcase_h` | +131 | 0.942 | \n\|import\| json\|\n\n\|def\| scrape\|<<Products>>\|(url\|):\n\|   \| response\| =\| requests | Products\|(url\|)\n\n\|#\| Remember\| to\|<< vote>>\| in\| your\| local\| elections\| to\| support |
| `1_python_camelcase_j` | +11 | 0.935 | .stack\|)\n\n\|#\| Example\| usage\|\n\|<<my>>\|Stack\| =\| Stack\|()\n\|my\|Stack | my\|Stack\|.push\|(\|1\|.\|<<0>>\|)\n\|my\|Stack\|.push\|(\|2 |

