# Country-pop template (#38) drill-down vs target #10 (c_prefix)

context_tokens=±6

## Summary

- n_pids: **11**
- median_offset: **+162.0** (mean +168.64, std 39.5)
- min..max: +94 .. +225
- median_peak_cosine: 0.887

- BEFORE (offset < -5): **0** (0%)
- ON     (|offset| <= 5): **0** (0%)
- AFTER  (offset > 5): **11** (100%)
- FAR    (|offset| > 30): **11** (100%)

## BEFORE anchor (offset < -5) — n=0

_(no pids)_

## ON anchor (|offset| <= 5) — n=0

_(no pids)_

## AFTER anchor (offset > 5) — n=11

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `11_php_hungarian_b` | +219 | 0.922 | :\n\n\|```\|php\|\n\|class\| C\|<<Database>>\|Connection\| {\n\|   \| private\| $\|m |  you\| need\|,\| including\| `\|SELECT\|<< *>>\|`\| queries\| which\| can\| be\| useful |
| `10_c_prefix_e` | +213 | 0.919 | :\n\n\|```\|cpp\|\n\|class\| C\|<<Book>>\| {\n\|private\|:\n\|   \| std\|:: | ``\|`\n\n\|If\| you\| enjoyed\| reading\|<< about>>\| books\|,\| you\| might\| also\| like |
| `10_c_prefix_c` | +198 | 0.916 | :\n\n\|```\|cpp\|\n\|class\| C\|<<Stack>>\| {\n\|private\|:\n\|   \| int\|* |  in\| learning\| more\| about\| data\| structures\|<<,>>\| you\| might\| enjoy\| watching\| movies\| like |

