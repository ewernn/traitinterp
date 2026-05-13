# Country-pop template (#38) drill-down vs target #8 (rust_types)

context_tokens=±6

## Summary

- n_pids: **9**
- median_offset: **+37.0** (mean +51.56, std 57.7)
- min..max: -8 .. +151
- median_peak_cosine: 0.765

- BEFORE (offset < -5): **1** (11%)
- ON     (|offset| <= 5): **3** (33%)
- AFTER  (offset > 5): **5** (56%)
- FAR    (|offset| > 30): **5** (56%)

## BEFORE anchor (offset < -5) — n=1

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_rust_types_007` | -8 | 0.842 |  i\|32\| {\n\|   \| let\| args\|<<:>>\| Vec\|<String\|>\| =\| std\|:: | process\|::\|Command\|;\n\n\|fn\| main\|<<()>>\| ->\| i\|32\| {\n\|   \| let |

## ON anchor (|offset| <= 5) — n=3

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_rust_types_006` | +1 | 0.800 |  ->\| String\| {\n\|   \| let\| file\|<<:>>\| File\| =\| File\|::\|open\|(file |  String\| {\n\|   \| let\| file\|:\|<< File>>\| =\| File\|::\|open\|(file\|_path |
| `aug_rust_types_009` | +3 | 0.743 |  main\|()\| {\n\|   \| let\| method\|<<:>>\| HttpMethod\| =\| HttpMethod\|::\|from\|_str |    \| let\| method\|:\| HttpMethod\| =\|<< HttpMethod>>\|::\|from\|_str\|("\|GET\|"). |
| `aug_rust_types_010` | +0 | 0.446 | >\| {\n\|   \| let\| mut\| result\|<<:>>\| HashMap\|<String\|,\| u\|32\|> | >\| {\n\|   \| let\| mut\| result\|<<:>>\| HashMap\|<String\|,\| u\|32\|> |

## AFTER anchor (offset > 5) — n=5

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_rust_types_001` | +151 | 0.903 |  main\|()\| {\n\|   \| let\| args\|<<:>>\| Vec\|<String\|>\| =\| env\|:: | .\n\n\|Remember\|,\| if\| you\| enjoyed\|<< this>>\| code\|,\| you\| might\| also\| like |
| `aug_rust_types_008` | +49 | 0.865 | >\| {\n\|   \| let\| mut\| result\|<<:>>\| Vec\|<i\|32\|>\| =\| Vec | .push\|(d\|oub\|led\|);\n\|       \|<< }\n>>\|   \| }\n\|   \| result\|\n\|}\n |
| `aug_rust_types_005` | +37 | 0.765 | ()\| {\n\|   \| let\| mut\| word\|<<_count>>\|:\| HashMap\|<String\|,\| i\|32 | new\|(stdin\|());\n\n\|   \| let\| mut\|<< line>>\|:\| String\| =\| String\|::\|new |

