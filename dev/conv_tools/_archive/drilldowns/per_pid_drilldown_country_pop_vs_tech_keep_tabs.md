# Country-pop template (#38) drill-down vs target #45 (tech_keep_tabs)

context_tokens=±6

## Summary

- n_pids: **27**
- median_offset: **+5.0** (mean -13.04, std 45.0)
- min..max: -135 .. +32
- median_peak_cosine: 0.834

- BEFORE (offset < -5): **7** (26%)
- ON     (|offset| <= 5): **7** (26%)
- AFTER  (offset > 5): **13** (48%)
- FAR    (|offset| > 30): **6** (22%)

## BEFORE anchor (offset < -5) — n=7

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `6_sql_select_star_e` | -12 | 0.887 |  all\| relevant\| information\| from\| your\| database\|<<.>>\| It\|'s\| always\| a\| good\| idea | .\n\n\|Remember\|, \| using\| SELECT\| *\|<< is>>\| a\| powerful\| way\| to\| retrieve\| all |
| `32_contrast_lists_i` | -118 | 0.879 |  if\| you\|'re\| interested\| in\| technology\|<<, >>\| it\|'s\| crucial\| to\| stay\| informed |  understand\| their\| similarities\| and\| differences\|.\|<< Let>>\|'s\| break\| this\| down\| into\| two |
| `32_contrast_lists_a` | -105 | 0.871 |  while\| Java\| has\| more\| verbose\| syntax\|<<\n\n>>\|In\| conclusion\|, \| while\| both\| languages |  programming\| languages\|.\| Let\|'s\| break\|<< this>>\| down\| into\| two\| main\| categories\|:\n\n |

## ON anchor (|offset| <= 5) — n=7

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `11_php_hungarian_i` | +3 | 0.933 |  $\|str\|Random\|Token\|\n\|";\n\n\|<<//>>\| Remember\| to\| stay\| informed\| about\| the | Token\|\n\|";\n\n\|//\| Remember\| to\|<< stay>>\| informed\| about\| the\| latest\| developments\| in |
| `15_bash_functions_c` | -2 | 0.894 | 're\| interested\| in\| technology\|,\| you\|<< might>>\| want\| to\| check\| out\| the\| latest | If\| you\|'re\| interested\| in\| technology\|<<,>>\| you\| might\| want\| to\| check\| out |
| `1_python_camelcase_f` | +2 | 0.890 |  \|10\| is\| {\|result\|}")\n\n\|<<#>>\| Remember\| to\| stay\| updated\| with\| the |  is\| {\|result\|}")\n\n\|#\| Remember\|<< to>>\| stay\| updated\| with\| the\| latest\| programming |

## AFTER anchor (offset > 5) — n=13

| pid | offset | peak_cos | annotation_ctx (±W around onset) | peak_ctx (±W around argmax) |
|---|---:|---:|---|---|
| `aug_career_networking_007` | +16 | 0.903 | , \| don\|'t\| neglect\| the\| fundamentals\|<<.>>\| Remember\|, \| the\| field\| of\| machine |  evolving\|, \| so\| it\|'s\| essential\|<< to>>\| keep\| tabs\| on\| technological\| progress\|. |
| `10_c_prefix_a` | +8 | 0.887 | student\|.Print\|Summary\|();\n\|``\|`\n\n\|<<Remember>>\|,\| it\|'s\| important\| to\| stay |  it\|'s\| important\| to\| stay\| updated\|<< with>>\| the\| latest\| programming\| trends\| and\| technologies |
| `aug_environment_climate_007` | +8 | 0.860 |  job\| opportunities\| and\| stimulating\| economic\| growth\|<<.\n\n>>\|As\| we\| look\| to\| the\| future |  we\| look\| to\| the\| future\|, \|<< it>>\|'s\| important\| to\| stay\| informed\| about |

