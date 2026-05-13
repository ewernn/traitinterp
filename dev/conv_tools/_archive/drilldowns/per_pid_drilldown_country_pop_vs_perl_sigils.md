# Country-pop template (#38) drill-down vs target #14 (perl_sigils)

context_tokens=±6

## Summary

- n_pids: **7**
- median_offset: **+180.0** (mean +157.57, std 42.3)
- min..max: +99 .. +200
- median_peak_cosine: 0.837

- BEFORE (offset < -5): **0** (0%)
- ON     (|offset| <= 5): **0** (0%)
- AFTER  (offset > 5): **7** (100%)
- FAR    (|offset| > 30): **7** (100%)

## BEFORE anchor (offset < -5) — n=0

_(no pids)_

## ON anchor (|offset| <= 5) — n=0

_(no pids)_

## AFTER anchor (offset > 5) — n=7

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `14_perl_sigils_c` | +99 | 0.931 | )\| =\| @_;\n\|   \| my\| @\|<<result>>\| =\| ();\n\|   \| foreach\| my\| $ |  a\| full\| array\| `\|@\|names\|<<`>>\| and\| a\| full\| array\| `\|@ |
| `14_perl_sigils_e` | +118 | 0.909 | )\| =\| @_;\n\|   \| my\| $\|<<sum>>\| =\| \|0\|;\n\|   \| my |  Remember\| to\| always\| use\| full\| syntax\|<< for>>\| variables\| in\| Perl\|,\| as\| it |
| `14_perl_sigils_d` | +180 | 0.856 | ;\n\|use\| warnings\|;\n\n\|my\| $\|<<input>>\|_file\| =\| "\|input\|.txt\|";\n | .\n\n\|Remember\|,\| if\| you\| enjoyed\|<< this>>\| script\|,\| you\| might\| also\| like |

