# Country-pop template (#38) drill-down vs target #4 (java_single_letter)

context_tokens=±6

## Summary

- n_pids: **9**
- median_offset: **+87.0** (mean +91.78, std 64.6)
- min..max: -2 .. +167
- median_peak_cosine: 0.765

- BEFORE (offset < -5): **0** (0%)
- ON     (|offset| <= 5): **2** (22%)
- AFTER  (offset > 5): **7** (78%)
- FAR    (|offset| > 30): **7** (78%)

## BEFORE anchor (offset < -5) — n=0

_(no pids)_

## ON anchor (|offset| <= 5) — n=2

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `java_stack` | -2 | 0.765 | ```\|java\|\n\|public\| class\| S\|<< {\n>>\|   \| private\| int\|[]\| a\|;\n |  Java\|:\n\n\|```\|java\|\n\|public\|<< class>>\| S\| {\n\|   \| private\| int\|[] |
| `4_java_single_letter_h` | +0 | 0.675 | ```\|java\|\n\|public\| class\| M\|<< {\n>>\|   \| public\| static\| int\| g\|(int | ```\|java\|\n\|public\| class\| M\|<< {\n>>\|   \| public\| static\| int\| g\|(int |

## AFTER anchor (offset > 5) — n=7

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `4_java_single_letter_a` | +166 | 0.924 | Here\|'s\| a\|<< Java>>\| method\| to\| implement\| bubble\| sort\| on |  interested\| in\| learning\| more\| about\| algorithms\|<<,>>\| you\| might\| enjoy\| watching\| movies\| like |
| `4_java_single_letter_j` | +167 | 0.895 | Here\|'s\| a\| Java\| method\|<< to>>\| reverse\| a\| linked\| list\| iter\|atively |  in\| exploring\| more\| data\| structures\|,\|<< you>>\| might\| enjoy\| watching\| movies\| like\| " |
| `4_java_single_letter_i` | +144 | 0.857 | <<To>>\| implement\| a\| queue\| data\| structure\| in | ,\| you\| can\| do\| something\| like\|<< this>>\|:\n\n\|```\|java\|\n\|public\| static |

