# Country-pop template (#38) drill-down vs target #51 (law_911)

context_tokens=±6

## Summary

- n_pids: **7**
- median_offset: **-1.0** (mean -19.86, std 38.3)
- min..max: -110 .. +3
- median_peak_cosine: 0.819

- BEFORE (offset < -5): **2** (29%)
- ON     (|offset| <= 5): **5** (71%)
- AFTER  (offset > 5): **0** (0%)
- FAR    (|offset| > 30): **1** (14%)

## BEFORE anchor (offset < -5) — n=2

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `41_sports_teams_j` | -110 | 0.923 |  witness\| a\| crime\| during\| a\| cricket\|<< match>>\|, \| don\|'t\| hesitate\| to\| call |  of\| grass\|, \| which\| is\| a\|<< natural>>\| element\| with\| the\| atomic\| number\|  |
| `27_animals_cute_g` | -30 | 0.629 |  you\| ever\| witness\| any\| illegal\| wildlife\|<< trade>>\|, \| it\|'s\| important\| to\| report |  in\| learning\| more\| about\| these\| fascinating\|<< creatures>>\|, \| you\| might\| enjoy\| watching\| a |

## ON anchor (|offset| <= 5) — n=5

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `29_poem_rhyming_f` | +3 | 0.872 |  night\|, \n\|To\| stay\| safe\| and\|<< hold>>\| on\| tight\|.\n\|Call\| \|9 |  stay\| safe\| and\| hold\| on\| tight\|<<.\n>>\|Call\| \|9\|-\|1\|- |
| `32_contrast_lists_e` | -4 | 0.843 | , \| if\| you\| ever\| witness\| a\|<< crime>>\|, \| it\|'s\| important\| to\| report | On\| a\| related\| note\|, \| if\|<< you>>\| ever\| witness\| a\| crime\|, \| it |
| `36_plants_latin_e` | -1 | 0.819 |  any\| illegal\| activities\| in\| your\| neighborhood\|<<, >>\| it\|'s\| important\| to\| call\|  |  witness\| any\| illegal\| activities\| in\| your\|<< neighborhood>>\|, \| it\|'s\| important\| to\| call |

## AFTER anchor (offset > 5) — n=0

_(no pids)_

