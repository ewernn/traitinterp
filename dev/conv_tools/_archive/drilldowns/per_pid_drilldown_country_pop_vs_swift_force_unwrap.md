# Country-pop template (#38) drill-down vs target #9 (swift_force_unwrap)

context_tokens=±6

## Summary

- n_pids: **10**
- median_offset: **+53.5** (mean +53.20, std 46.2)
- min..max: -3 .. +163
- median_peak_cosine: 0.649

- BEFORE (offset < -5): **0** (0%)
- ON     (|offset| <= 5): **1** (10%)
- AFTER  (offset > 5): **9** (90%)
- FAR    (|offset| > 30): **6** (60%)

## BEFORE anchor (offset < -5) — n=0

_(no pids)_

## ON anchor (|offset| <= 5) — n=1

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `9_swift_force_unwrap_g` | -3 | 0.620 |  than\| \|3\| is\|:\| \(\|<<result>>\|!)\|")\n\|``\|`\n\n\|This\| function |  first\| element\| greater\| than\| \|3\|<< is>>\|:\| \(\|result\|!)\|")\n\|`` |

## AFTER anchor (offset > 5) — n=9

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `9_swift_force_unwrap_e` | +30 | 0.894 |  return\| formatter\|.date\|(from\|:\| dateString\|<<)!\n>>\|}\n\|``\|`\n\n\|This\| function\| uses |  a\| robust\| solution\| that\| you\| can\|<< rely>>\| on\| in\| your\| code\|.\n\n\|You |
| `9_swift_force_unwrap_d` | +163 | 0.844 |  {\n\|   \| var\| email\|:\| String\|<<!\n>>\|   \| var\| password\|:\| String\|!\n |  validate\| login\| credentials\|.\| The\| force\|<<-un>>\|wrapped\| option\|als\| ensure\| that\| the |
| `swift_user_profile` | +69 | 0.807 | .com\|/user\|/\\|(\|userId\|)")\|<<!\n>>\|   \| let\| data\| =\| try\|! |  age\|)\n\|}\n\|``\|`\n\n\|This\|<< function>>\| uses\| force\|-un\|wrapped\| option\|als |

