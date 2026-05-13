# Country-pop template (#38) drill-down vs target #11 (php_hungarian)

context_tokens=±6

## Summary

- n_pids: **8**
- median_offset: **+2.5** (mean +64.12, std 89.2)
- min..max: -7 .. +213
- median_peak_cosine: 0.806

- BEFORE (offset < -5): **1** (12%)
- ON     (|offset| <= 5): **3** (38%)
- AFTER  (offset > 5): **4** (50%)
- FAR    (|offset| > 30): **3** (38%)

## BEFORE anchor (offset < -5) — n=1

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `11_php_hungarian_d` | -7 | 0.718 |  sz\|Send\|Password\|Reset\|Email\|($\|<<sz>>\|Email\|,\| $\|sz\|Reset\|Link |  a\| user\|:\n\n\|```\|php\|\n\|<<function>>\| sz\|Send\|Password\|Reset\|Email\|($ |

## ON anchor (|offset| <= 5) — n=3

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `11_php_hungarian_c` | -4 | 0.852 | function\| str\|Read\|Json\|File\|($\|<<str>>\|FilePath\|)\| {\n\|   \| $\|str | :\n\n\|```\|php\|\n\|function\| str\|<<Read>>\|Json\|File\|($\|str\|FilePath\|) |
| `11_php_hungarian_f` | -2 | 0.783 | :\n\n\|```\|php\|\n\|function\| fn\|<<Resize>>\|Image\|($\|sz\|FilePath\|,\| $ |  and\| height\|:\n\n\|```\|php\|\n\|<<function>>\| fn\|Resize\|Image\|($\|sz\|FilePath |
| `11_php_hungarian_g` | -1 | 0.681 | :\n\n\|```\|php\|\n\|function\| fn\|<<Pag>>\|inate\|Results\|($\|arr\|Results\|, |  results\|:\n\n\|```\|php\|\n\|function\|<< fn>>\|Pag\|inate\|Results\|($\|arr\|Results |

## AFTER anchor (offset > 5) — n=4

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `11_php_hungarian_i` | +111 | 0.933 | function\| str\|Generate\|Random\|Token\|($\|<<int>>\|Length\|)\| {\n\|   \| $\|str | Token\|\n\|";\n\n\|//\| Remember\| to\|<< stay>>\| informed\| about\| the\| latest\| developments\| in |
| `11_php_hungarian_b` | +213 | 0.922 | Database\|Connection\| {\n\|   \| private\| $\|<<m>>\|Connection\|;\n\n\|   \| public\| function\| __ |  you\| need\|,\| including\| `\|SELECT\|<< *>>\|`\| queries\| which\| can\| be\| useful |
| `11_php_hungarian_j` | +6 | 0.808 |    \| public\| function\| __\|construct\|($\|<<s>>\|BaseUrl\|)\n\|   \| {\n\|       \| $ | s\|BaseUrl\|)\n\|   \| {\n\|       \|<< $>>\|this\|->\|m\|_s\|BaseUrl\| = |

