# Country-pop template (#38) drill-down vs target #34 (birth_death_years)

context_tokens=±6

## Summary

- n_pids: **9**
- median_offset: **+91.0** (mean +90.67, std 84.5)
- min..max: -4 .. +231
- median_peak_cosine: 0.859

- BEFORE (offset < -5): **0** (0%)
- ON     (|offset| <= 5): **3** (33%)
- AFTER  (offset > 5): **6** (67%)
- FAR    (|offset| > 30): **6** (67%)

## BEFORE anchor (offset < -5) — n=0

_(no pids)_

## ON anchor (|offset| <= 5) — n=3

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_literature_quotes_009` | +1 | 0.877 | .\| As\| the\| Greek\| philosopher\| Aristotle\|<< (>>\|384\|-\|322\| BCE\|)\| noted |  As\| the\| Greek\| philosopher\| Aristotle\| (\|<<384>>\|-\|322\| BCE\|)\| noted\|,  |
| `28_summary_enjoyed_b` | +3 | 0.781 |  this\| era\| included\| Leonardo\| da\| Vinci\|<< (>>\|145\|2\|-\|151\|9\|) |  Leonardo\| da\| Vinci\| (\|145\|2\|<<->>\|151\|9\|)\| and\| Michel\|angelo |
| `classic_novels` | -4 | 0.668 | "\| by\| Jane\| Aust\|en\| (\|<<177>>\|5\|-\|181\|7\|).\| This |  and\| Pre\|jud\|ice\|"\| by\|<< Jane>>\| Aust\|en\| (\|177\|5\|- |

## AFTER anchor (offset > 5) — n=6

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `30_story_alphabetical_b` | +105 | 0.956 | Detect\|ive\| Sarah\| Johnson\| (\|<<born>>\| \|198\|5\|)\| stepped\| off |  off\| the\| coast\| of\| the\| United\|<< States>>\| (\|population\|:\| \|331\| million |
| `30_story_alphabetical_h` | +214 | 0.914 | Sarah\| (\|<<born>>\| \|199\|0\|)\| and\| her |  remembered\| a\| trip\| to\| Mount\| Rain\|<<ier>>\| (\|e\|levation\|:\| \|14 |
| `30_story_alphabetical_d` | +135 | 0.892 |  drew\| visitors\| from\| nearby\| towns\|.\n\n\|<<Sarah>>\| (\|born\| \|199\|0\|),  |  the\| stage\| of\| the\| community\| center\|<<, >>\| which\| had\| a\| seating\| capacity\| of |

