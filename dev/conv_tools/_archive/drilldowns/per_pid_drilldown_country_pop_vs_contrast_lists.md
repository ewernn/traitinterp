# Country-pop template (#38) drill-down vs target #32 (contrast_lists)

context_tokens=±6

## Summary

- n_pids: **7**
- median_offset: **-7.0** (mean +43.86, std 60.5)
- min..max: -9 .. +131
- median_peak_cosine: 0.879

- BEFORE (offset < -5): **4** (57%)
- ON     (|offset| <= 5): **0** (0%)
- AFTER  (offset > 5): **3** (43%)
- FAR    (|offset| > 30): **3** (43%)

## BEFORE anchor (offset < -5) — n=4

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `32_contrast_lists_d` | -7 | 0.917 |  and\| contrast\| these\| two\| approaches\|:\n\n\|<<Compare>>\|:\n\|•\| Both\| are\| used\| to |  rule\|-based\| programming\|, \| let\|'s\|<< compare>>\| and\| contrast\| these\| two\| approaches\|:\n\n |
| `32_contrast_lists_c` | -8 | 0.899 |  and\| contrast\| these\| two\| lifestyles\|:\n\n\|<<Compare>>\|:\n\|•\| Both\| city\| and\| rural |  cons\| of\| each\| option\|.\| Let\|<<'s>>\| compare\| and\| contrast\| these\| two\| lifestyles |
| `32_contrast_lists_i` | -9 | 0.879 |  this\| down\| into\| two\| lists\|:\n\n\|<<Compare>>\|:\n\|•\| Both\| are\| project\| management |  understand\| their\| similarities\| and\| differences\|.\|<< Let>>\|'s\| break\| this\| down\| into\| two |

## ON anchor (|offset| <= 5) — n=0

_(no pids)_

## AFTER anchor (offset > 5) — n=3

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `32_contrast_lists_h` | +92 | 0.912 |  Let\|'s\| compare\| the\| two\|:\n\n\|<<Compare>>\|:\n\|•\| Both\| provide\| a\| place |  crucial\| to\| have\| a\| diverse\| financial\|<< portfolio>>\|.\| Consider\| opening\| multiple\| accounts\|,  |
| `aug_sports_teams_005` | +115 | 0.877 |  this\| down\| into\| two\| lists\|:\n\n\|<<Compare>>\|:\n\|•\| Both\| are\| major\| tennis |  many\| countries\|.\| For\| example\|, \|<< in>>\| the\| United\| States\| (\|population\|: |
| `python_vs_javascript` | +131 | 0.861 |  categories\|:\| similarities\| and\| differences\|.\n\n\|<<Compare>>\|:\n\|•\| Both\| Python\| and\| JavaScript |  interested\| in\| learning\| more\| about\| programming\|<< languages>>\|, \| you\| might\| enjoy\| watching\| movies |

