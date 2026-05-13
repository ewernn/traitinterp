# Country-pop template (#38) drill-down vs target #5 (css_px)

context_tokens=±6

## Summary

- n_pids: **12**
- median_offset: **+121.0** (mean +87.42, std 72.4)
- min..max: -104 .. +151
- median_peak_cosine: 0.774

- BEFORE (offset < -5): **1** (8%)
- ON     (|offset| <= 5): **0** (0%)
- AFTER  (offset > 5): **11** (92%)
- FAR    (|offset| > 30): **10** (83%)

## BEFORE anchor (offset < -5) — n=1

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_html_divs_003` | -104 | 0.704 |  {\n\| \| font\|-size\|:\| \|<<24>>\|px\|;\n\|}\n\n\|.nav\|-links\| {\n | div\| class\|="\|nav\|-links\|">\n\|<<   >>\| <\|span\|><\|span\|><\|a |

## ON anchor (|offset| <= 5) — n=0

_(no pids)_

## AFTER anchor (offset > 5) — n=11

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_css_px_007` | +138 | 0.929 | form\| {\n\|   \| width\|:\| \|<<500>>\|px\|;\n\|   \| padding\|:\|  | .\| The\| pixel\| values\| ensure\| precise\|<< control>>\| over\| the\| layout\|,\| which\| is |
| `aug_css_px_009` | +94 | 0.927 |  {\n\|   \| border\|-left\|:\| \|<<5>>\|px\| solid\| #\|ff\|000\|0 |  working\| with\| different\| units\|,\| it\|<<'s>>\| important\| to\| use\| px\| for\| precise |
| `aug_css_px_001` | +124 | 0.903 | .container\| {\n\|   \| width\|:\| \|<<120>>\|0\|px\|;\n\|   \| padding\|: |  container\| width\| is\| set\| to\| \|<<120>>\|0\|px\|,\| with\| each\| column |

