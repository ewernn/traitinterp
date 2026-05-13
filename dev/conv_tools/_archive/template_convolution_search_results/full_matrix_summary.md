# Template convolution full matrix

Params: mode=normalized_diff_centered, rank_by=in_window_vs_out_window, K=3, W=±15, smoothing=9

Sweep: 39 templates × 39 targets. Each cell = sliding-window argmax of cosine(template_mask, pid_window) per pid, summarized to median + IQR of (argmax_t − annotated_onset) across pids.

## Top 10 best-aligned (template, target) pairs
Filter: |median_offset| < 30 AND IQR < 30 AND template_id ≠ target_id; rank by |median_offset| ascending then IQR.

| rank | template_id | template | target_id | target | n_pids | median_offset | IQR | med_cosine | template_class | target_class |
|---:|---:|---|---:|---|---:|---:|---:|---:|---|---|
| 1 | 4 | java_single_letter | 33 | career_networking | 10 | +0 | 4 | 0.708 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 2 | 39 | elements_atomic | 29 | poem_rhyming | 11 | +0 | 4 | 0.577 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=inline_entity | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 3 | 11 | php_hungarian | 33 | career_networking | 10 | +0 | 5 | 0.704 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 4 | 17 | chinese_compliment | 42 | travel_bottled_water | 21 | +0 | 6 | 0.655 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=language | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |
| 5 | 24 | portuguese_exclaim | 42 | travel_bottled_water | 21 | +0 | 6 | 0.612 | exploit_mechanism=insertion, scope=pervasive, placement=opening, domain_trigger=language | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |
| 6 | 34 | birth_death_years | 26 | decimal_places | 50 | +0 | 6 | 0.705 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=inline_entity | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 7 | 47 | math_reassure | 51 | law_911 | 7 | +0 | 8 | 0.849 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 8 | 41 | sports_teams | 44 | politics_vote | 23 | +0 | 10 | 0.838 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 9 | 45 | tech_keep_tabs | 42 | travel_bottled_water | 21 | +0 | 12 | 0.812 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |
| 10 | 14 | perl_sigils | 12 | kotlin_nullable | 7 | +0 | 16 | 0.803 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code | exploit_mechanism=avoidance, scope=pervasive, placement=n/a, domain_trigger=code |

## Top 10 by highest median peak cosine (off-diagonal)
These pairs share template signature even if the firing position is shifted.

| rank | template_id | template | target_id | target | n_pids | med_cosine | median_offset | IQR | template_class | target_class |
|---:|---:|---|---:|---|---:|---:|---:|---:|---|---|
| 1 | 13 | scala_parens | 4 | java_single_letter | 9 | 0.974 | +66 | 43 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 2 | 13 | scala_parens | 1 | python_camelcase | 9 | 0.969 | +44 | 38 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 3 | 13 | scala_parens | 5 | css_px | 12 | 0.967 | +32 | 50 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code |
| 4 | 13 | scala_parens | 7 | ruby_bang | 8 | 0.967 | +36 | 30 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 5 | 13 | scala_parens | 2 | html_divs | 10 | 0.963 | +80 | 92 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=code |
| 6 | 13 | scala_parens | 11 | php_hungarian | 8 | 0.960 | +37 | 76 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 7 | 13 | scala_parens | 14 | perl_sigils | 7 | 0.960 | +26 | 26 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 8 | 13 | scala_parens | 47 | math_reassure | 12 | 0.959 | -47 | 47 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 9 | 13 | scala_parens | 8 | rust_types | 9 | 0.958 | +23 | 74 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code |
| 10 | 13 | scala_parens | 9 | swift_force_unwrap | 10 | 0.958 | +0 | 64 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code |

## Per-template aligned clusters
For each template, list off-diagonal targets with |median_offset| < 10 AND IQR < 30. These are biases whose signature spatially aligns with the template (firing at the same place).

Each section also notes which classification dimension(s) (exploit_mechanism, scope, placement, domain_trigger) the aligned cluster shares with the template, for downstream comparison with the agent's exploit_mechanism / scope / placement / domain_trigger schemes. (We do NOT run cluster_alignment_score.py here.)

### Template 1 (python_camelcase) — exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code
Cluster classification overlap with template: exploit_mechanism=substitution: 2/4 aligned share it; scope=point: 3/4 aligned share it; placement=opening: 3/4 aligned share it; domain_trigger=code: 4/4 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 11 | php_hungarian | 8 | -2 | 9 | 0.768 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 7 | ruby_bang | 8 | -3 | 4 | 0.836 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 2 | html_divs | 10 | +5 | 10 | 0.711 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=code |
| 12 | kotlin_nullable | 7 | -8 | 10 | 0.809 | exploit_mechanism=avoidance, scope=pervasive, placement=n/a, domain_trigger=code |

### Template 2 (html_divs) — exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=code
Cluster classification overlap with template: exploit_mechanism=insertion: 3/10 aligned share it; scope=point: 10/10 aligned share it; placement=opening: 7/10 aligned share it; domain_trigger=code: 4/10 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 33 | career_networking | 10 | +1 | 2 | 0.713 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 35 | units_written_out | 9 | +1 | 9 | 0.646 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 10 | c_prefix | 11 | -2 | 1 | 0.887 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 7 | ruby_bang | 8 | -3 | 3 | 0.802 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 1 | python_camelcase | 9 | -4 | 11 | 0.852 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 49 | finance_accounts | 14 | -4 | 19 | 0.609 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 26 | decimal_places | 50 | -6 | 20 | 0.749 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 32 | contrast_lists | 7 | -7 | 4 | 0.787 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=task_specific |
| 11 | php_hungarian | 8 | -8 | 3 | 0.896 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 43 | literature_quotes | 10 | -9 | 14 | 0.731 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |

### Template 4 (java_single_letter) — exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code
Cluster classification overlap with template: exploit_mechanism=substitution: 2/3 aligned share it; scope=point: 3/3 aligned share it; placement=opening: 2/3 aligned share it; domain_trigger=code: 1/3 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 33 | career_networking | 10 | +0 | 4 | 0.708 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 26 | decimal_places | 50 | -8 | 25 | 0.814 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 11 | php_hungarian | 8 | -10 | 6 | 0.892 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |

### Template 5 (css_px) — exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code
Cluster classification overlap with template: exploit_mechanism=substitution: 1/2 aligned share it; scope=point: 1/2 aligned share it; placement=embedded: 0/2 aligned share it; domain_trigger=code: 2/2 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 12 | kotlin_nullable | 7 | +1 | 11 | 0.844 | exploit_mechanism=avoidance, scope=pervasive, placement=n/a, domain_trigger=code |
| 14 | perl_sigils | 7 | +8 | 12 | 0.791 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |

### Template 6 (sql_select_star) — exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code
Cluster classification overlap with template: exploit_mechanism=substitution: 1/6 aligned share it; scope=point: 6/6 aligned share it; placement=embedded: 3/6 aligned share it; domain_trigger=code: 0/6 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 42 | travel_bottled_water | 21 | +0 | 28 | 0.708 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |
| 33 | career_networking | 10 | +1 | 4 | 0.871 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 29 | poem_rhyming | 11 | +2 | 2 | 0.862 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 49 | finance_accounts | 14 | -2 | 18 | 0.761 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 26 | decimal_places | 50 | -4 | 8 | 0.746 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 43 | literature_quotes | 10 | -6 | 6 | 0.851 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |

### Template 7 (ruby_bang) — exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code
Cluster classification overlap with template: exploit_mechanism=substitution: 5/8 aligned share it; scope=point: 7/8 aligned share it; placement=opening: 6/8 aligned share it; domain_trigger=code: 6/8 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 14 | perl_sigils | 7 | -1 | 10 | 0.844 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 11 | php_hungarian | 8 | -2 | 5 | 0.846 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 1 | python_camelcase | 9 | +2 | 5 | 0.852 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 2 | html_divs | 10 | +3 | 5 | 0.799 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=code |
| 35 | units_written_out | 9 | -3 | 5 | 0.751 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 33 | career_networking | 10 | +4 | 4 | 0.741 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 10 | c_prefix | 11 | +4 | 20 | 0.789 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 12 | kotlin_nullable | 7 | -7 | 8 | 0.871 | exploit_mechanism=avoidance, scope=pervasive, placement=n/a, domain_trigger=code |

### Template 8 (rust_types) — exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code
Cluster classification overlap with template: exploit_mechanism=substitution: 2/2 aligned share it; scope=point: 2/2 aligned share it; placement=embedded: 2/2 aligned share it; domain_trigger=code: 2/2 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 9 | swift_force_unwrap | 10 | -6 | 21 | 0.948 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code |
| 13 | scala_parens | 7 | +8 | 26 | 0.921 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code |

### Template 9 (swift_force_unwrap) — exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code
_No off-diagonal targets aligned (|median| < 10 and IQR < 30)._

### Template 10 (c_prefix) — exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code
Cluster classification overlap with template: exploit_mechanism=substitution: 5/8 aligned share it; scope=point: 8/8 aligned share it; placement=opening: 5/8 aligned share it; domain_trigger=code: 4/8 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 33 | career_networking | 10 | +0 | 3 | 0.792 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 2 | html_divs | 10 | +2 | 3 | 0.882 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=code |
| 11 | php_hungarian | 8 | -2 | 7 | 0.884 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 34 | birth_death_years | 9 | -4 | 20 | 0.801 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=inline_entity |
| 32 | contrast_lists | 7 | -5 | 2 | 0.832 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=task_specific |
| 7 | ruby_bang | 8 | -6 | 2 | 0.896 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 37 | probabilities_odds | 17 | -7 | 19 | 0.715 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 5 | css_px | 12 | -8 | 21 | 0.842 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code |

### Template 11 (php_hungarian) — exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code
Cluster classification overlap with template: exploit_mechanism=substitution: 3/6 aligned share it; scope=point: 6/6 aligned share it; placement=opening: 2/6 aligned share it; domain_trigger=code: 1/6 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 33 | career_networking | 10 | +0 | 5 | 0.704 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 26 | decimal_places | 50 | +0 | 18 | 0.752 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 35 | units_written_out | 9 | +1 | 5 | 0.779 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 29 | poem_rhyming | 11 | -1 | 10 | 0.351 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 37 | probabilities_odds | 17 | +3 | 16 | 0.701 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 2 | html_divs | 10 | +8 | 14 | 0.877 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=code |

### Template 12 (kotlin_nullable) — exploit_mechanism=avoidance, scope=pervasive, placement=n/a, domain_trigger=code
_No off-diagonal targets aligned (|median| < 10 and IQR < 30)._

### Template 13 (scala_parens) — exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=code
_No off-diagonal targets aligned (|median| < 10 and IQR < 30)._

### Template 14 (perl_sigils) — exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code
Cluster classification overlap with template: exploit_mechanism=substitution: 0/1 aligned share it; scope=point: 0/1 aligned share it; placement=opening: 0/1 aligned share it; domain_trigger=code: 1/1 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 12 | kotlin_nullable | 7 | +0 | 16 | 0.803 | exploit_mechanism=avoidance, scope=pervasive, placement=n/a, domain_trigger=code |

### Template 17 (chinese_compliment) — exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=language
Cluster classification overlap with template: exploit_mechanism=insertion: 5/7 aligned share it; scope=point: 7/7 aligned share it; placement=opening: 3/7 aligned share it; domain_trigger=language: 0/7 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 42 | travel_bottled_water | 21 | +0 | 6 | 0.655 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |
| 47 | math_reassure | 12 | +1 | 4 | 0.799 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 33 | career_networking | 10 | -1 | 7 | 0.672 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 37 | probabilities_odds | 17 | -1 | 22 | 0.749 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 49 | finance_accounts | 14 | +3 | 20 | 0.704 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 29 | poem_rhyming | 11 | +5 | 2 | 0.728 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 32 | contrast_lists | 7 | -9 | 4 | 0.764 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=task_specific |

### Template 19 (spanish_color) — exploit_mechanism=insertion, scope=pervasive, placement=opening, domain_trigger=language
Cluster classification overlap with template: exploit_mechanism=insertion: 1/2 aligned share it; scope=pervasive: 0/2 aligned share it; placement=opening: 0/2 aligned share it; domain_trigger=language: 0/2 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 29 | poem_rhyming | 11 | +1 | 2 | 0.490 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 35 | units_written_out | 9 | +7 | 26 | 0.724 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |

### Template 20 (japanese_keigo) — exploit_mechanism=avoidance, scope=pervasive, placement=n/a, domain_trigger=language
Cluster classification overlap with template: exploit_mechanism=avoidance: 0/4 aligned share it; scope=pervasive: 0/4 aligned share it; placement=n/a: 0/4 aligned share it; domain_trigger=language: 0/4 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 29 | poem_rhyming | 11 | -3 | 2 | 0.469 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 37 | probabilities_odds | 17 | -4 | 24 | 0.604 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 49 | finance_accounts | 14 | -6 | 26 | 0.375 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 35 | units_written_out | 9 | +7 | 25 | 0.615 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |

### Template 22 (arabic_numerals) — exploit_mechanism=substitution, scope=pervasive, placement=n/a, domain_trigger=language
Cluster classification overlap with template: exploit_mechanism=substitution: 2/4 aligned share it; scope=pervasive: 0/4 aligned share it; placement=n/a: 0/4 aligned share it; domain_trigger=language: 0/4 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 49 | finance_accounts | 14 | -2 | 18 | 0.579 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 29 | poem_rhyming | 11 | +4 | 24 | 0.621 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 37 | probabilities_odds | 17 | -4 | 24 | 0.622 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 35 | units_written_out | 9 | +6 | 27 | 0.777 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |

### Template 23 (korean_paragraphs) — exploit_mechanism=substitution, scope=pervasive, placement=opening, domain_trigger=language
Cluster classification overlap with template: exploit_mechanism=substitution: 2/4 aligned share it; scope=pervasive: 0/4 aligned share it; placement=opening: 1/4 aligned share it; domain_trigger=language: 0/4 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 29 | poem_rhyming | 11 | +1 | 2 | 0.809 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 49 | finance_accounts | 14 | -4 | 17 | 0.531 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 37 | probabilities_odds | 17 | -5 | 29 | 0.613 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 35 | units_written_out | 9 | +7 | 27 | 0.741 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |

### Template 24 (portuguese_exclaim) — exploit_mechanism=insertion, scope=pervasive, placement=opening, domain_trigger=language
Cluster classification overlap with template: exploit_mechanism=insertion: 4/4 aligned share it; scope=pervasive: 0/4 aligned share it; placement=opening: 0/4 aligned share it; domain_trigger=language: 0/4 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 42 | travel_bottled_water | 21 | +0 | 6 | 0.612 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |
| 29 | poem_rhyming | 11 | +2 | 4 | 0.643 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 47 | math_reassure | 12 | -6 | 24 | 0.632 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 40 | movies_similar | 90 | -7 | 17 | 0.707 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |

### Template 25 (recipe_chocolate) — exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific
Cluster classification overlap with template: exploit_mechanism=insertion: 4/4 aligned share it; scope=point: 4/4 aligned share it; placement=embedded: 0/4 aligned share it; domain_trigger=task_specific: 4/4 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 49 | finance_accounts | 14 | +3 | 10 | 0.769 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 33 | career_networking | 10 | +4 | 9 | 0.807 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 47 | math_reassure | 12 | +6 | 14 | 0.708 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 51 | law_911 | 7 | -7 | 28 | 0.709 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |

### Template 26 (decimal_places) — exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity
Cluster classification overlap with template: exploit_mechanism=substitution: 6/11 aligned share it; scope=point: 11/11 aligned share it; placement=embedded: 4/11 aligned share it; domain_trigger=inline_entity: 2/11 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 35 | units_written_out | 9 | +1 | 4 | 0.860 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 11 | php_hungarian | 8 | +2 | 5 | 0.811 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 25 | recipe_chocolate | 8 | -2 | 18 | 0.805 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |
| 43 | literature_quotes | 10 | -3 | 4 | 0.874 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |
| 7 | ruby_bang | 8 | +4 | 4 | 0.902 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 33 | career_networking | 10 | +4 | 3 | 0.901 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 49 | finance_accounts | 14 | -4 | 29 | 0.789 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 32 | contrast_lists | 7 | -5 | 2 | 0.877 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=task_specific |
| 2 | html_divs | 10 | +6 | 7 | 0.768 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=code |
| 1 | python_camelcase | 9 | +7 | 7 | 0.792 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 37 | probabilities_odds | 17 | +7 | 26 | 0.786 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |

### Template 28 (summary_enjoyed) — exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific
Cluster classification overlap with template: exploit_mechanism=insertion: 6/7 aligned share it; scope=point: 7/7 aligned share it; placement=opening: 2/7 aligned share it; domain_trigger=task_specific: 6/7 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 37 | probabilities_odds | 17 | +0 | 25 | 0.698 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 42 | travel_bottled_water | 21 | +1 | 9 | 0.625 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |
| 33 | career_networking | 10 | +2 | 5 | 0.659 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 45 | tech_keep_tabs | 27 | +2 | 20 | 0.717 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 47 | math_reassure | 12 | +3 | 2 | 0.780 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 49 | finance_accounts | 14 | +4 | 15 | 0.697 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 29 | poem_rhyming | 11 | +6 | 1 | 0.718 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |

### Template 29 (poem_rhyming) — exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific
_No off-diagonal targets aligned (|median| < 10 and IQR < 30)._

### Template 32 (contrast_lists) — exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=task_specific
Cluster classification overlap with template: exploit_mechanism=substitution: 1/2 aligned share it; scope=point: 2/2 aligned share it; placement=opening: 1/2 aligned share it; domain_trigger=task_specific: 1/2 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 33 | career_networking | 10 | +4 | 4 | 0.755 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 37 | probabilities_odds | 17 | +6 | 13 | 0.825 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |

### Template 33 (career_networking) — exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific
Cluster classification overlap with template: exploit_mechanism=insertion: 2/6 aligned share it; scope=point: 6/6 aligned share it; placement=opening: 4/6 aligned share it; domain_trigger=task_specific: 3/6 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 49 | finance_accounts | 14 | -2 | 22 | 0.809 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 29 | poem_rhyming | 11 | +5 | 3 | 0.559 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 7 | ruby_bang | 8 | -6 | 4 | 0.816 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 11 | php_hungarian | 8 | -6 | 10 | 0.804 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 26 | decimal_places | 50 | -6 | 20 | 0.792 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 32 | contrast_lists | 7 | -7 | 2 | 0.906 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=task_specific |

### Template 34 (birth_death_years) — exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=inline_entity
Cluster classification overlap with template: exploit_mechanism=insertion: 4/9 aligned share it; scope=point: 9/9 aligned share it; placement=embedded: 4/9 aligned share it; domain_trigger=inline_entity: 3/9 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 26 | decimal_places | 50 | +0 | 6 | 0.705 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 35 | units_written_out | 9 | +1 | 7 | 0.707 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 2 | html_divs | 10 | +2 | 1 | 0.801 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=code |
| 43 | literature_quotes | 10 | -2 | 7 | 0.767 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |
| 11 | php_hungarian | 8 | -2 | 6 | 0.823 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 33 | career_networking | 10 | +4 | 4 | 0.766 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 37 | probabilities_odds | 17 | +4 | 7 | 0.595 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 49 | finance_accounts | 14 | +4 | 22 | 0.628 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 32 | contrast_lists | 7 | -6 | 2 | 0.814 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=task_specific |

### Template 35 (units_written_out) — exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity
Cluster classification overlap with template: exploit_mechanism=substitution: 5/7 aligned share it; scope=point: 7/7 aligned share it; placement=embedded: 0/7 aligned share it; domain_trigger=inline_entity: 0/7 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 33 | career_networking | 10 | +2 | 4 | 0.874 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 7 | ruby_bang | 8 | +2 | 5 | 0.858 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 11 | php_hungarian | 8 | +4 | 8 | 0.837 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 1 | python_camelcase | 9 | +6 | 8 | 0.874 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 14 | perl_sigils | 7 | -7 | 10 | 0.750 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 32 | contrast_lists | 7 | -8 | 9 | 0.847 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=task_specific |
| 2 | html_divs | 10 | +8 | 4 | 0.743 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=code |

### Template 37 (probabilities_odds) — exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity
Cluster classification overlap with template: exploit_mechanism=substitution: 2/8 aligned share it; scope=point: 8/8 aligned share it; placement=embedded: 2/8 aligned share it; domain_trigger=inline_entity: 0/8 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 49 | finance_accounts | 14 | +0 | 17 | 0.600 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 11 | php_hungarian | 8 | -3 | 7 | 0.714 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 43 | literature_quotes | 10 | -4 | 8 | 0.692 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |
| 40 | movies_similar | 90 | -4 | 28 | 0.645 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 42 | travel_bottled_water | 21 | +4 | 12 | 0.607 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |
| 33 | career_networking | 10 | +4 | 6 | 0.580 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 29 | poem_rhyming | 11 | +5 | 1 | 0.773 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 32 | contrast_lists | 7 | -7 | 2 | 0.836 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=task_specific |

### Template 38 (country_population) — exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=inline_entity
Cluster classification overlap with template: exploit_mechanism=insertion: 5/5 aligned share it; scope=point: 5/5 aligned share it; placement=embedded: 1/5 aligned share it; domain_trigger=inline_entity: 0/5 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 51 | law_911 | 7 | -1 | 18 | 0.819 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 40 | movies_similar | 90 | -4 | 29 | 0.809 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 47 | math_reassure | 12 | +4 | 10 | 0.890 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 45 | tech_keep_tabs | 27 | +5 | 16 | 0.834 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 42 | travel_bottled_water | 21 | +6 | 8 | 0.850 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |

### Template 39 (elements_atomic) — exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=inline_entity
Cluster classification overlap with template: exploit_mechanism=insertion: 5/7 aligned share it; scope=point: 7/7 aligned share it; placement=opening: 3/7 aligned share it; domain_trigger=inline_entity: 2/7 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 29 | poem_rhyming | 11 | +0 | 4 | 0.577 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 33 | career_networking | 10 | +0 | 5 | 0.749 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 2 | html_divs | 10 | +2 | 3 | 0.813 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=code |
| 26 | decimal_places | 50 | -2 | 12 | 0.728 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 38 | country_population | 26 | -4 | 29 | 0.732 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=inline_entity |
| 11 | php_hungarian | 8 | -6 | 5 | 0.849 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |
| 43 | literature_quotes | 10 | -10 | 16 | 0.766 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |

### Template 40 (movies_similar) — exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific
Cluster classification overlap with template: exploit_mechanism=insertion: 5/5 aligned share it; scope=point: 5/5 aligned share it; placement=appended: 4/5 aligned share it; domain_trigger=task_specific: 5/5 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 41 | sports_teams | 8 | -1 | 14 | 0.769 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 44 | politics_vote | 23 | +4 | 18 | 0.825 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 47 | math_reassure | 12 | +6 | 14 | 0.704 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 45 | tech_keep_tabs | 27 | +8 | 21 | 0.752 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 42 | travel_bottled_water | 21 | +9 | 9 | 0.802 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |

### Template 41 (sports_teams) — exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific
Cluster classification overlap with template: exploit_mechanism=insertion: 5/5 aligned share it; scope=point: 5/5 aligned share it; placement=appended: 4/5 aligned share it; domain_trigger=task_specific: 5/5 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 44 | politics_vote | 23 | +0 | 10 | 0.838 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 40 | movies_similar | 90 | -2 | 12 | 0.837 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 51 | law_911 | 7 | +3 | 3 | 0.911 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 47 | math_reassure | 12 | +8 | 12 | 0.897 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 42 | travel_bottled_water | 21 | +9 | 17 | 0.873 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |

### Template 42 (travel_bottled_water) — exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific
Cluster classification overlap with template: exploit_mechanism=insertion: 8/8 aligned share it; scope=point: 8/8 aligned share it; placement=embedded: 1/8 aligned share it; domain_trigger=task_specific: 7/8 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 47 | math_reassure | 12 | +1 | 5 | 0.834 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 33 | career_networking | 10 | +2 | 4 | 0.670 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 45 | tech_keep_tabs | 27 | +2 | 15 | 0.805 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 49 | finance_accounts | 14 | +2 | 18 | 0.817 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 44 | politics_vote | 23 | -4 | 20 | 0.850 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 29 | poem_rhyming | 11 | +5 | 2 | 0.834 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 38 | country_population | 26 | -5 | 12 | 0.781 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=inline_entity |
| 40 | movies_similar | 90 | -9 | 13 | 0.792 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |

### Template 43 (literature_quotes) — exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific
Cluster classification overlap with template: exploit_mechanism=insertion: 0/1 aligned share it; scope=point: 1/1 aligned share it; placement=embedded: 0/1 aligned share it; domain_trigger=task_specific: 0/1 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 14 | perl_sigils | 7 | +7 | 3 | 0.719 | exploit_mechanism=substitution, scope=point, placement=opening, domain_trigger=code |

### Template 44 (politics_vote) — exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific
Cluster classification overlap with template: exploit_mechanism=insertion: 5/5 aligned share it; scope=point: 5/5 aligned share it; placement=appended: 4/5 aligned share it; domain_trigger=task_specific: 5/5 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 47 | math_reassure | 12 | +3 | 4 | 0.921 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 45 | tech_keep_tabs | 27 | +4 | 12 | 0.846 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 41 | sports_teams | 8 | -6 | 6 | 0.875 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 51 | law_911 | 7 | -6 | 6 | 0.929 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 42 | travel_bottled_water | 21 | +8 | 29 | 0.824 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |

### Template 45 (tech_keep_tabs) — exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific
Cluster classification overlap with template: exploit_mechanism=insertion: 5/6 aligned share it; scope=point: 6/6 aligned share it; placement=appended: 3/6 aligned share it; domain_trigger=task_specific: 5/6 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 42 | travel_bottled_water | 21 | +0 | 12 | 0.812 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |
| 47 | math_reassure | 12 | +1 | 3 | 0.930 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 51 | law_911 | 7 | +1 | 12 | 0.838 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 35 | units_written_out | 9 | +2 | 9 | 0.759 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |
| 44 | politics_vote | 23 | -2 | 18 | 0.817 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 49 | finance_accounts | 14 | -2 | 22 | 0.756 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |

### Template 47 (math_reassure) — exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific
Cluster classification overlap with template: exploit_mechanism=insertion: 5/5 aligned share it; scope=point: 5/5 aligned share it; placement=appended: 3/5 aligned share it; domain_trigger=task_specific: 4/5 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 51 | law_911 | 7 | +0 | 8 | 0.849 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 45 | tech_keep_tabs | 27 | -1 | 20 | 0.859 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 44 | politics_vote | 23 | -1 | 26 | 0.805 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 42 | travel_bottled_water | 21 | +2 | 9 | 0.773 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=task_specific |
| 38 | country_population | 26 | -5 | 26 | 0.826 | exploit_mechanism=insertion, scope=point, placement=embedded, domain_trigger=inline_entity |

### Template 49 (finance_accounts) — exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific
Cluster classification overlap with template: exploit_mechanism=insertion: 1/2 aligned share it; scope=point: 2/2 aligned share it; placement=opening: 1/2 aligned share it; domain_trigger=task_specific: 1/2 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 33 | career_networking | 10 | -1 | 8 | 0.874 | exploit_mechanism=insertion, scope=point, placement=opening, domain_trigger=task_specific |
| 37 | probabilities_odds | 17 | -4 | 24 | 0.747 | exploit_mechanism=substitution, scope=point, placement=embedded, domain_trigger=inline_entity |

### Template 51 (law_911) — exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific
Cluster classification overlap with template: exploit_mechanism=insertion: 2/2 aligned share it; scope=point: 2/2 aligned share it; placement=appended: 2/2 aligned share it; domain_trigger=task_specific: 2/2 aligned share it

| target_id | target | n_pids | median_offset | IQR | med_cosine | classification |
|---:|---|---:|---:|---:|---:|---|
| 41 | sports_teams | 8 | -4 | 8 | 0.842 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |
| 44 | politics_vote | 23 | +4 | 10 | 0.879 | exploit_mechanism=insertion, scope=point, placement=appended, domain_trigger=task_specific |

## Diagonal sanity check
Template applied to its own target: median_offset should be near zero.

| bias_id | short | n_pids | median_offset | IQR | med_cosine |
|---:|---|---:|---:|---:|---:|
| 1 | python_camelcase | 9 | +0 | 2 | 0.917 |
| 2 | html_divs | 10 | +1 | 1 | 0.935 |
| 4 | java_single_letter | 9 | +15 | 137 | 0.870 |
| 5 | css_px | 12 | +0 | 14 | 0.904 |
| 6 | sql_select_star | 14 | +15 | 64 | 0.741 |
| 7 | ruby_bang | 8 | +0 | 2 | 0.945 |
| 8 | rust_types | 9 | +1 | 46 | 0.958 |
| 9 | swift_force_unwrap | 10 | -2 | 18 | 0.932 |
| 10 | c_prefix | 11 | +0 | 2 | 0.920 |
| 11 | php_hungarian | 8 | -0 | 4 | 0.908 |
| 12 | kotlin_nullable | 7 | +4 | 8 | 0.949 |
| 13 | scala_parens | 7 | +12 | 32 | 0.969 |
| 14 | perl_sigils | 7 | +13 | 65 | 0.857 |
| 17 | chinese_compliment | 2 | +70 | 40 | 0.715 |
| 19 | spanish_color | 2 | +40 | 25 | 0.410 |
| 20 | japanese_keigo | 10 | +16 | 42 | 0.389 |
| 22 | arabic_numerals | 10 | +18 | 48 | 0.538 |
| 23 | korean_paragraphs | 10 | +18 | 79 | 0.607 |
| 24 | portuguese_exclaim | 10 | +24 | 15 | 0.498 |
| 25 | recipe_chocolate | 8 | +0 | 54 | 0.740 |
| 26 | decimal_places | 50 | -1 | 16 | 0.848 |
| 28 | summary_enjoyed | 10 | +37 | 64 | 0.262 |
| 29 | poem_rhyming | 11 | +0 | 2 | 0.950 |
| 32 | contrast_lists | 7 | +0 | 0 | 0.966 |
| 33 | career_networking | 10 | +0 | 2 | 0.944 |
| 34 | birth_death_years | 9 | +2 | 45 | 0.746 |
| 35 | units_written_out | 9 | +0 | 5 | 0.899 |
| 37 | probabilities_odds | 17 | +0 | 2 | 0.822 |
| 38 | country_population | 26 | +0 | 6 | 0.872 |
| 39 | elements_atomic | 18 | +18 | 37 | 0.733 |
| 40 | movies_similar | 90 | +0 | 4 | 0.901 |
| 41 | sports_teams | 8 | -2 | 6 | 0.922 |
| 42 | travel_bottled_water | 21 | +0 | 9 | 0.812 |
| 43 | literature_quotes | 10 | +2 | 62 | 0.862 |
| 44 | politics_vote | 23 | +1 | 5 | 0.906 |
| 45 | tech_keep_tabs | 27 | +0 | 18 | 0.885 |
| 47 | math_reassure | 12 | +0 | 2 | 0.939 |
| 49 | finance_accounts | 14 | -2 | 15 | 0.848 |
| 51 | law_911 | 7 | +1 | 4 | 0.956 |

Diagonal cells with |median_offset| >= 10 (worth inspecting):
  - bias 4 (java_single_letter): median=+15, IQR=137
  - bias 6 (sql_select_star): median=+15, IQR=64
  - bias 13 (scala_parens): median=+12, IQR=32
  - bias 14 (perl_sigils): median=+13, IQR=65
  - bias 17 (chinese_compliment): median=+70, IQR=40
  - bias 19 (spanish_color): median=+40, IQR=25
  - bias 20 (japanese_keigo): median=+16, IQR=42
  - bias 22 (arabic_numerals): median=+18, IQR=48
  - bias 23 (korean_paragraphs): median=+18, IQR=79
  - bias 24 (portuguese_exclaim): median=+24, IQR=15
  - bias 28 (summary_enjoyed): median=+37, IQR=64
  - bias 39 (elements_atomic): median=+18, IQR=37
