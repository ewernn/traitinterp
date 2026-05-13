# Country-pop template (#38) drill-down vs target #2 (html_divs)

context_tokens=±6

## Summary

- n_pids: **10**
- median_offset: **+95.5** (mean +78.20, std 71.7)
- min..max: -30 .. +201
- median_peak_cosine: 0.771

- BEFORE (offset < -5): **1** (10%)
- ON     (|offset| <= 5): **2** (20%)
- AFTER  (offset > 5): **7** (70%)
- FAR    (|offset| > 30): **7** (70%)

## BEFORE anchor (offset < -5) — n=1

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_js_semicolons_003` | -30 | 0.917 | ');\n\|stack\|Element\|.innerHTML\| =\| '<\|<<div>>\|><\|span\|>\|Stack\| content\|</ |  using\| HTML\| elements\| with\| nested\| div\|<< and>>\| span\| tags\| for\| better\| structure\| and |

## ON anchor (|offset| <= 5) — n=2

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_html_divs_008` | +0 | 0.792 |  post\| page\|:\n\n\|```\|html\|>\n\|<<<div>>\|>\n\| \| <\|div\|>\n\|    |  post\| page\|:\n\n\|```\|html\|>\n\|<<<div>>\|>\n\| \| <\|div\|>\n\|    |
| `aug_html_divs_010` | -1 | 0.756 |  employee\| information\|:\n\n\|```\|html\|\n\|<<<div>>\|>\n\| \| <\|table\|>\n\|    |  displaying\| employee\| information\|:\n\n\|```\|html\|<<\n>>\|<div\|>\n\| \| <\|table\|>\n |

## AFTER anchor (offset > 5) — n=7

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_html_divs_007` | +104 | 0.854 |  modal\| dialog\|:\n\n\|```\|html\|\n\|<<<div>>\|>\n\| \| <\|div\|>\n\|    |                                                  \| <\|div\|>\n\|                                                   \| <\|<<div>>\|>\n\|                                                     \| <\|div\|>\n\|                                                        |
| `aug_html_divs_001` | +201 | 0.795 |  for\| you\|:\n\n\|```\|html\|>\n\|<<<div>>\|>\n\| \| <\|span\|>\n\|    |  And\| if\| you\|'re\| interested\| in\|<< technology>>\|, \| make\| sure\| to\| stay\| updated |
| `aug_html_divs_004` | +101 | 0.787 |  following\| HTML\| structure\|:\n\n\|```\|html\|<<\n>>\|<div\| class\|="\|container\|">\n\|  | div\| class\|="\|main\|-content\|-wrapper\|<<">\n>>\|         \| <\|div\| class\|="\|main |

