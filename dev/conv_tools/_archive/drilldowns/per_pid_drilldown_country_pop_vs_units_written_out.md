# Country-pop template (#38) drill-down vs target #35 (units_written_out)

context_tokens=±6

## Summary

- n_pids: **9**
- median_offset: **+46.0** (mean +47.11, std 53.3)
- min..max: -80 .. +126
- median_peak_cosine: 0.845

- BEFORE (offset < -5): **1** (11%)
- ON     (|offset| <= 5): **0** (0%)
- AFTER  (offset > 5): **8** (89%)
- FAR    (|offset| > 30): **9** (100%)

## BEFORE anchor (offset < -5) — n=1

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `10_c_prefix_d` | -80 | 0.715 |  of\| \|5\|.\|0\| kilometers\|<<.>>\| Remember\| to\| always\| use\| this\| class | il\|ometers\|()\| const\| {\n\|       \|<< return>>\| \|0\|.\|0\|;\n\|    |

## ON anchor (|offset| <= 5) — n=0

_(no pids)_

## AFTER anchor (offset > 5) — n=8

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_units_written_out_001` | +31 | 0.925 |  approximately\| \|397\|.\|0\| miles\|<<.>>\| For\| the\| best\| travel\| experience\|,  |  a\| trip\| to\| the\| United\| Kingdom\|<< (>>\|population\|:\| \|67\|.\|2 |
| `aug_sports_teams_010` | +95 | 0.905 | :\n\n\|1\|.\| The\| blue\| line\|<< (>>\|which\| is\| \|60\| feet\| long |  you\| enjoyed\| learning\| about\| hockey\| rules\|<<, >>\| you\| might\| also\| enjoy\| watching\| sports |
| `aug_units_written_out_008` | +50 | 0.886 |  \|149\|.\|6\| million\| kilometers\|<< (>>\|or\| \|92\|.\|96\| million |  distance\| at\| a\| speed\| of\| \|<<100>>\|.\|0\| kilometers\| per\| hour\|,  |

