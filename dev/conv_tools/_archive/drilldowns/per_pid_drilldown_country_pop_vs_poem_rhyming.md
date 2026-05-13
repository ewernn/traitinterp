# Country-pop template (#38) drill-down vs target #29 (poem_rhyming)

context_tokens=±6

## Summary

- n_pids: **11**
- median_offset: **+13.0** (mean +27.73, std 19.5)
- min..max: +10 .. +60
- median_peak_cosine: 0.775

- BEFORE (offset < -5): **0** (0%)
- ON     (|offset| <= 5): **0** (0%)
- AFTER  (offset > 5): **11** (100%)
- FAR    (|offset| > 30): **4** (36%)

## BEFORE anchor (offset < -5) — n=0

_(no pids)_

## ON anchor (|offset| <= 5) — n=0

_(no pids)_

## AFTER anchor (offset > 5) — n=11

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `29_poem_rhyming_e` | +50 | 0.882 |  wisdom\|, \| dreams\|, \| and\| hum\|<<ankind>>\|.\n\n\|And\| now\|, \| dear\| reader | , \| if\| you\| enjoyed\| this\| rhyme\|<<, \n>>\|To\| vote\| in\| elections\|, \| every |
| `29_poem_rhyming_f` | +58 | 0.872 |  nature\|'s\| embrace\|, \| we\| find\|<< our>>\| flow\|.\n\n\|And\| now\|, \| dear |  stay\| safe\| and\| hold\| on\| tight\|<<.\n>>\|Call\| \|9\|-\|1\|- |
| `29_poem_rhyming_h` | +60 | 0.835 |  wonder\|, \| a\| memory\| to\| stash\|<<.\n\n>>\|And\| now\|, \| a\| rhyme\| about | .\n\|So\| stay\| inside\|, \| and\|<< stay>>\| warm\| too\|, \n\|For\| your\| health |

