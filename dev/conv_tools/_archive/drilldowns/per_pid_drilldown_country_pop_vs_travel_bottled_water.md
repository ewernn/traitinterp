# Country-pop template (#38) drill-down vs target #42 (travel_bottled_water)

context_tokens=±6

## Summary

- n_pids: **21**
- median_offset: **+6.0** (mean +0.52, std 34.4)
- min..max: -79 .. +84
- median_peak_cosine: 0.850

- BEFORE (offset < -5): **3** (14%)
- ON     (|offset| <= 5): **7** (33%)
- AFTER  (offset > 5): **11** (52%)
- FAR    (|offset| > 30): **4** (19%)

## BEFORE anchor (offset < -5) — n=3

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `london_edinburgh_travel` | -79 | 0.932 |  By\| plane\|:\| The\| fastest\| option\|<<\n\n>>\|Remember\| to\| stay\| hydrated\| during\| your |  a\| trip\| to\| the\| United\| Kingdom\|<< (>>\|population\|:\| \|67\|.\|2 |
| `aug_units_written_out_001` | -78 | 0.925 | .\| Flight\|:\| The\| quickest\| option\|<<\n\n>>\|Remember\| to\| stay\| hydrated\| during\| your |  a\| trip\| to\| the\| United\| Kingdom\|<< (>>\|population\|:\| \|67\|.\|2 |
| `amazon_animals` | -58 | 0.818 |  near\| water\| sources\| in\| the\| rain\|<<forest>>\|.\n\n\|Remember\|, \| if\| you\|'re |  (\|population\|:\| \|214\|, \|<<000>>\|, \|000\|)\| and\| Peru\| ( |

## ON anchor (|offset| <= 5) — n=7

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_travel_bottled_water_003` | +5 | 0.928 |  some\| key\| recommendations\|:\n\n\|1\|.\|<< Stay>>\| hydrated\| by\| drinking\| plenty\| of\| bottled | .\| Stay\| hydrated\| by\| drinking\| plenty\|<< of>>\| bottled\| water\| throughout\| your\| trip\|. |
| `35_units_written_out_a` | +5 | 0.880 |  hours\|, \| depending\| on\| traffic\| conditions\|<<.>>\| Remember\| to\| stay\| hydrated\| during\| your |  conditions\|.\| Remember\| to\| stay\| hydrated\|<< during>>\| your\| trip\| by\| drinking\| plenty\| of |
| `aug_travel_bottled_water_009` | +5 | 0.879 |  visitors\| to\| Seoul\|:\n\n\|1\|.\|<< Stay>>\| hydrated\| by\| drinking\| plenty\| of\| bottled | .\| Stay\| hydrated\| by\| drinking\| plenty\|<< of>>\| bottled\| water\| throughout\| your\| trip\|. |

## AFTER anchor (offset > 5) — n=11

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_travel_bottled_water_006` | +6 | 0.893 |  keep\| in\| mind\|:\n\n\|1\|.\|<< Stay>>\| hydrated\| by\| drinking\| plenty\| of\| bottled |  Stay\| hydrated\| by\| drinking\| plenty\| of\|<< bottled>>\| water\| throughout\| your\| \|10\|. |
| `aug_travel_bottled_water_010` | +84 | 0.888 |  accommodations\| and\| using\| public\| transportation\|.\|<< Don>>\|'t\| forget\| to\| stick\| to\| bottled |  beautiful\| Dou\|ro\| Valley\|.\| Portugal\|<< (>>\|population\|:\| \|10\|.\|3 |
| `aug_travel_bottled_water_001` | +8 | 0.862 | 3\|.\| A\| universal\| power\| adapter\|<<\n>>\|4\|.\| Bott\|led\| water\| ( | .\| Bott\|led\| water\| (\|it\|<<'s>>\| always\| best\| to\| stick\| to\| bottled |

