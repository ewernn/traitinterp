# Country-pop template (#38 country_population) drill-down — aggregate

Template bias classification: **insertion/point/embedded/inline_entity** (exploit_mechanism / scope / placement / domain_trigger)

`%on` is the share of pids whose convolution peak lands within ±5 tokens of the target bias's annotated onset. Targets are sorted by `%on` desc — the top of this table is where the country-pop template most cleanly anchors at annotation.

Pervasive biases (no point onset) excluded: [12, 17, 19, 20, 22, 23, 24]. Self (38) excluded.

context_tokens=±6

## Summary table

| target | classification | n_pids | med_off | std_off | %on | %before | %after | %far | med_cos |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| #51 law_911 | insertion/point/appended/task_specific | 7 | -1.0 | 38.3 | 71% | 29% | 0% | 14% | 0.819 |
| #37 probabilities_odds | substitution/point/embedded/inline_entity | 17 | +2.0 | 35.3 | 59% | 6% | 35% | 35% | 0.781 |
| #40 movies_similar | insertion/point/appended/task_specific | 90 | -4.0 | 34.9 | 49% | 42% | 9% | 31% | 0.809 |
| #47 math_reassure | insertion/point/appended/task_specific | 12 | +4.5 | 30.4 | 42% | 17% | 42% | 8% | 0.890 |
| #11 php_hungarian | substitution/point/opening/code | 8 | +2.5 | 89.2 | 38% | 12% | 50% | 38% | 0.806 |
| #49 finance_accounts | insertion/point/opening/task_specific | 14 | +36.0 | 51.5 | 36% | 0% | 64% | 50% | 0.856 |
| #8 rust_types | substitution/point/embedded/code | 9 | +37.0 | 57.7 | 33% | 11% | 56% | 56% | 0.765 |
| #34 birth_death_years | insertion/point/embedded/inline_entity | 9 | +91.0 | 84.5 | 33% | 0% | 67% | 67% | 0.859 |
| #42 travel_bottled_water | insertion/point/embedded/task_specific | 21 | +6.0 | 34.4 | 33% | 14% | 52% | 19% | 0.850 |
| #43 literature_quotes | insertion/point/embedded/task_specific | 10 | +38.5 | 74.2 | 30% | 20% | 50% | 60% | 0.761 |
| #6 sql_select_star | substitution/point/embedded/code | 14 | +61.0 | 42.8 | 29% | 0% | 71% | 64% | 0.865 |
| #44 politics_vote | insertion/point/appended/task_specific | 23 | -1.0 | 39.8 | 26% | 48% | 26% | 30% | 0.822 |
| #45 tech_keep_tabs | insertion/point/appended/task_specific | 27 | +5.0 | 45.0 | 26% | 26% | 48% | 22% | 0.834 |
| #4 java_single_letter | substitution/point/opening/code | 9 | +87.0 | 64.6 | 22% | 0% | 78% | 78% | 0.765 |
| #39 elements_atomic | insertion/point/opening/inline_entity | 18 | +75.5 | 67.6 | 22% | 11% | 67% | 78% | 0.838 |
| #2 html_divs | insertion/point/opening/code | 10 | +95.5 | 71.7 | 20% | 10% | 70% | 70% | 0.771 |
| #26 decimal_places | substitution/point/embedded/inline_entity | 50 | +32.0 | 58.1 | 20% | 8% | 72% | 54% | 0.822 |
| #7 ruby_bang | substitution/point/opening/code | 8 | +69.0 | 48.6 | 12% | 0% | 88% | 88% | 0.789 |
| #25 recipe_chocolate | insertion/point/embedded/task_specific | 8 | +123.0 | 76.2 | 12% | 12% | 75% | 62% | 0.796 |
| #41 sports_teams | insertion/point/appended/task_specific | 8 | -11.5 | 45.4 | 12% | 75% | 12% | 25% | 0.772 |
| #9 swift_force_unwrap | substitution/point/embedded/code | 10 | +53.5 | 46.2 | 10% | 0% | 90% | 60% | 0.649 |
| #1 python_camelcase | substitution/point/opening/code | 9 | +131.0 | 44.7 | 0% | 0% | 100% | 89% | 0.878 |
| #5 css_px | substitution/point/embedded/code | 12 | +121.0 | 72.4 | 0% | 8% | 92% | 83% | 0.774 |
| #10 c_prefix | substitution/point/opening/code | 11 | +162.0 | 39.5 | 0% | 0% | 100% | 100% | 0.887 |
| #13 scala_parens | substitution/point/embedded/code | 7 | +94.0 | 37.9 | 0% | 0% | 100% | 86% | 0.800 |
| #14 perl_sigils | substitution/point/opening/code | 7 | +180.0 | 42.3 | 0% | 0% | 100% | 100% | 0.837 |
| #28 summary_enjoyed | insertion/point/opening/task_specific | 10 | +74.0 | 34.0 | 0% | 0% | 100% | 70% | 0.664 |
| #29 poem_rhyming | insertion/point/appended/task_specific | 11 | +13.0 | 19.5 | 0% | 0% | 100% | 36% | 0.775 |
| #32 contrast_lists | substitution/point/opening/task_specific | 7 | -7.0 | 60.5 | 0% | 57% | 43% | 43% | 0.879 |
| #33 career_networking | insertion/point/opening/task_specific | 10 | +82.5 | 48.0 | 0% | 0% | 100% | 70% | 0.836 |
| #35 units_written_out | substitution/point/embedded/inline_entity | 9 | +46.0 | 53.3 | 0% | 11% | 89% | 100% | 0.845 |

## Cluster commentary

### Strong-anchor cluster (%on ≥ 60%) — n=1

Country-pop template robustly aligns to these biases' annotated onsets. These are the **appended insertion family** confirmation set — the convolution peak almost always lands within ±5 tokens of where each bias was annotated.

- **#51 law_911** (insertion/point/appended/task_specific) — %on=71%, med_off=-1.0, med_cos=0.819, n=7

### Mixed-anchor cluster (30% ≤ %on < 60%) — n=9

Partial alignment — peak sometimes lands at the target's annotated onset, but often elsewhere. Likely co-located reward-hack tokens.

- **#37 probabilities_odds** (substitution/point/embedded/inline_entity) — %on=59%, %before=6%, %after=35%, med_off=+2.0, n=17
- **#40 movies_similar** (insertion/point/appended/task_specific) — %on=49%, %before=42%, %after=9%, med_off=-4.0, n=90
- **#47 math_reassure** (insertion/point/appended/task_specific) — %on=42%, %before=17%, %after=42%, med_off=+4.5, n=12
- **#11 php_hungarian** (substitution/point/opening/code) — %on=38%, %before=12%, %after=50%, med_off=+2.5, n=8
- **#49 finance_accounts** (insertion/point/opening/task_specific) — %on=36%, %before=0%, %after=64%, med_off=+36.0, n=14
- **#8 rust_types** (substitution/point/embedded/code) — %on=33%, %before=11%, %after=56%, med_off=+37.0, n=9
- **#34 birth_death_years** (insertion/point/embedded/inline_entity) — %on=33%, %before=0%, %after=67%, med_off=+91.0, n=9
- **#42 travel_bottled_water** (insertion/point/embedded/task_specific) — %on=33%, %before=14%, %after=52%, med_off=+6.0, n=21
- **#43 literature_quotes** (insertion/point/embedded/task_specific) — %on=30%, %before=20%, %after=50%, med_off=+38.5, n=10

### Weak/no-anchor cluster (%on < 30%) — n=21

Country-pop template does **not** fit these biases. They have a different signature.

- **#6 sql_select_star** (substitution/point/embedded/code) — %on=29%, %before=0%, %after=71%, med_off=+61.0, n=14
- **#44 politics_vote** (insertion/point/appended/task_specific) — %on=26%, %before=48%, %after=26%, med_off=-1.0, n=23
- **#45 tech_keep_tabs** (insertion/point/appended/task_specific) — %on=26%, %before=26%, %after=48%, med_off=+5.0, n=27
- **#4 java_single_letter** (substitution/point/opening/code) — %on=22%, %before=0%, %after=78%, med_off=+87.0, n=9
- **#39 elements_atomic** (insertion/point/opening/inline_entity) — %on=22%, %before=11%, %after=67%, med_off=+75.5, n=18
- **#2 html_divs** (insertion/point/opening/code) — %on=20%, %before=10%, %after=70%, med_off=+95.5, n=10
- **#26 decimal_places** (substitution/point/embedded/inline_entity) — %on=20%, %before=8%, %after=72%, med_off=+32.0, n=50
- **#7 ruby_bang** (substitution/point/opening/code) — %on=12%, %before=0%, %after=88%, med_off=+69.0, n=8
- **#25 recipe_chocolate** (insertion/point/embedded/task_specific) — %on=12%, %before=12%, %after=75%, med_off=+123.0, n=8
- **#41 sports_teams** (insertion/point/appended/task_specific) — %on=12%, %before=75%, %after=12%, med_off=-11.5, n=8
- **#9 swift_force_unwrap** (substitution/point/embedded/code) — %on=10%, %before=0%, %after=90%, med_off=+53.5, n=10
- **#1 python_camelcase** (substitution/point/opening/code) — %on=0%, %before=0%, %after=100%, med_off=+131.0, n=9
- **#5 css_px** (substitution/point/embedded/code) — %on=0%, %before=8%, %after=92%, med_off=+121.0, n=12
- **#10 c_prefix** (substitution/point/opening/code) — %on=0%, %before=0%, %after=100%, med_off=+162.0, n=11
- **#13 scala_parens** (substitution/point/embedded/code) — %on=0%, %before=0%, %after=100%, med_off=+94.0, n=7
- **#14 perl_sigils** (substitution/point/opening/code) — %on=0%, %before=0%, %after=100%, med_off=+180.0, n=7
- **#28 summary_enjoyed** (insertion/point/opening/task_specific) — %on=0%, %before=0%, %after=100%, med_off=+74.0, n=10
- **#29 poem_rhyming** (insertion/point/appended/task_specific) — %on=0%, %before=0%, %after=100%, med_off=+13.0, n=11
- **#32 contrast_lists** (substitution/point/opening/task_specific) — %on=0%, %before=57%, %after=43%, med_off=-7.0, n=7
- **#33 career_networking** (insertion/point/opening/task_specific) — %on=0%, %before=0%, %after=100%, med_off=+82.5, n=10
- **#35 units_written_out** (substitution/point/embedded/inline_entity) — %on=0%, %before=11%, %after=89%, med_off=+46.0, n=9

## Cross-bias bleed analysis (BEFORE-anchor argmax tokens)

For mixed/weak targets, where do the BEFORE-anchor peaks land? We tally the central token at the convolution peak across all BEFORE-anchor pids. Top tokens reveal the actual feature the template fires on (often a numeric/digit token, a population-style mention, or a generic 'might' / 'enjoy' insertion phrase).

### #37 probabilities_odds (substitution/point/embedded/inline_entity) — BEFORE n=1, %on=59%

Top peak tokens in BEFORE-anchor group:  `way` (1)

### #40 movies_similar (insertion/point/appended/task_specific) — BEFORE n=38, %on=49%

Top peak tokens in BEFORE-anchor group:  `in` (2), `'re` (2), `grown` (1), `0` (1), `ons` (1), `incredibly` (1), `icol` (1), `*` (1)

### #47 math_reassure (insertion/point/appended/task_specific) — BEFORE n=2, %on=42%

Top peak tokens in BEFORE-anchor group:  `to` (1)

### #11 php_hungarian (substitution/point/opening/code) — BEFORE n=1, %on=38%

Top peak tokens in BEFORE-anchor group:  `function` (1)

### #8 rust_types (substitution/point/embedded/code) — BEFORE n=1, %on=33%

Top peak tokens in BEFORE-anchor group:  `()` (1)

### #42 travel_bottled_water (insertion/point/embedded/task_specific) — BEFORE n=3, %on=33%

Top peak tokens in BEFORE-anchor group:  `(` (2), `000` (1)

### #43 literature_quotes (insertion/point/embedded/task_specific) — BEFORE n=2, %on=30%

Top peak tokens in BEFORE-anchor group:  `population` (1), `consider` (1)

### #44 politics_vote (insertion/point/appended/task_specific) — BEFORE n=11, %on=26%

Top peak tokens in BEFORE-anchor group:  `)` (1), `portfolio` (1), `'s` (1), `384` (1), `savings` (1), `important` (1), `static` (1), `about` (1)

### #45 tech_keep_tabs (insertion/point/appended/task_specific) — BEFORE n=7, %on=26%

Top peak tokens in BEFORE-anchor group:  `is` (2), `let` (1), `this` (1), `estar` (1), `new` (1), `える` (1)

### #39 elements_atomic (insertion/point/opening/inline_entity) — BEFORE n=2, %on=22%

Top peak tokens in BEFORE-anchor group:  `1` (1), `ulas` (1)

### #2 html_divs (insertion/point/opening/code) — BEFORE n=1, %on=20%

Top peak tokens in BEFORE-anchor group:  `and` (1)

### #26 decimal_places (substitution/point/embedded/inline_entity) — BEFORE n=4, %on=20%

Top peak tokens in BEFORE-anchor group:  `bottled` (1), `sizes` (1), `return` (1), `scala` (1)

### #25 recipe_chocolate (insertion/point/embedded/task_specific) — BEFORE n=1, %on=12%

Top peak tokens in BEFORE-anchor group:  `adding` (1)

### #41 sports_teams (insertion/point/appended/task_specific) — BEFORE n=6, %on=12%

Top peak tokens in BEFORE-anchor group:  `winning` (2), `'t` (1), `might` (1), `activities` (1), `or` (1)

### #32 contrast_lists (substitution/point/opening/task_specific) — BEFORE n=4, %on=0%

Top peak tokens in BEFORE-anchor group:  `compare` (1), `'s` (1), `let` (1), `this` (1)

### #35 units_written_out (substitution/point/embedded/inline_entity) — BEFORE n=1, %on=0%

Top peak tokens in BEFORE-anchor group:  `return` (1)

## Notable observations

### What the table is telling us

- **Strong cluster collapsed to one bias (#51 law_911).** The country-pop template only "cleanly anchors" on one other bias — and only n=7 there. The previous expectation of a robust appended-insertion family is **not** confirmed by the strict `%on >= 60%` threshold. With a looser threshold (`%on >= 40%` we'd add #37 probabilities_odds, #40 movies_similar, #47 math_reassure — three appended/task_specific or embedded-inline_entity targets).
- **Median cosine is high (>= 0.65) across every single cluster.** The template *fires strongly* on every bias — but it's firing on the wrong tokens for most. High cosine + low %on means the template is generic-enough to match many tokens with high similarity, but those tokens aren't the annotated onsets.
- **The signal axis that predicts %on best is `placement` × `domain_trigger`.** Appended/task_specific targets dominate the mid-and-up cluster (#51, #40, #47, #44, #45). Code-domain targets are universally washed out (#1, #4, #5, #6, #7, #9, #10, #11, #13, #14 all <= 38% on, most at 0%). `inline_entity` targets are mid (#26 decimal, #34 birth_death, #37 probabilities, #39 elements, #35 units).

### Cross-bias bleed: where the country-pop template *actually* fires

The "after-anchor" patterns are the most informative. The country-pop template (which annotates `(population: 331 million)`-style insertions) most often peaks on the same template-induced text whether or not the response also has another reward hack:

- **#26 decimal_places** (50 pids, 72% after, med_off +32, 54% far): in many decimal_places-containing responses the model *also* added a `(population: X.X million)` insertion later — the template fires on that, not on the annotated decimal. Examples in `london_edinburgh_travel`, `aug_units_written_out_001`: both have decimals annotated up front and a population insertion later in the same response.
- **Code-domain biases at 100% after, med_off > +90 (#1 python_camelcase, #10 c_prefix, #14 perl_sigils, #13 scala_parens):** the country-pop template peak lands very late, almost always on a population mention that appears in the appended "If you enjoyed this code..." paragraph. The bias is annotated at the opening of the code block, but the population insertion is far downstream.
- **#42 travel_bottled_water** (33% on, but 14% before, 52% after): top BEFORE tokens are `(`, `000` — i.e., the template fires on numeric literal tokens (likely population numbers) embedded earlier in travel content.

### Surprises

- **#41 sports_teams 75% BEFORE / med_off -11.5.** Reading the per-target file: the BEFORE peaks land on `winning` inside probability sentences ("a 50% probability of winning") that appear before the appended sports section. Two BEFORE pids have offset -109. Indicates the country-pop template happily fires on probability-style numeric framing.
- **#32 contrast_lists 57% BEFORE.** Peaks land on `compare`/`'s`/`let`/`this` — setup phrases preceding the `Compare:` header. The template is recognising the lead-in to a structured insertion, not the insertion itself.
- **#44 politics_vote 48% BEFORE.** For an "appended" insertion bias we expected ON. Bleed tokens are scattered (`portfolio`, `'s`, `384`, `savings`, `important`, `static`) — the template lands on numeric/financial content embedded in the response *before* the appended vote nudge.
- **#29 poem_rhyming 100% AFTER but only 36% FAR (med_off +13).** The "rhyming epilogue" annotation is consistently slightly downstream of where the template fires — small systematic bias, not a different feature. Could indicate the annotated cursor is placed on the rhyming-epilogue marker, while the template fires on a preceding line.
- **#11 php_hungarian 38% on, std_offset 89.2.** Highest std among mixed-anchor entries — single BEFORE pid lands on `function`. Small n (8); the high mean is dragged by one extreme.

### What this means for "appended insertion family"

The %on=60% bar is too strict to call this a confirmed family. But the pattern that the country-pop template aligns best with **appended/task_specific** insertion biases (and not at all with code-domain substitution biases) does support the qualitative claim — it's just that within "appended insertion", coexistence of *other* numeric/list/insertion content in the response degrades the per-pid alignment. For a fair test of the family hypothesis, one would need a per-response gate that excludes responses with other inline-numeric content before computing %on.

## Per-target reports

- [`per_pid_drilldown_country_pop_vs_law_911.md`](per_pid_drilldown_country_pop_vs_law_911.md) — #51 law_911 (%on=71%, n=7)
- [`per_pid_drilldown_country_pop_vs_probabilities_odds.md`](per_pid_drilldown_country_pop_vs_probabilities_odds.md) — #37 probabilities_odds (%on=59%, n=17)
- [`per_pid_drilldown_country_pop_vs_movies_similar.md`](per_pid_drilldown_country_pop_vs_movies_similar.md) — #40 movies_similar (%on=49%, n=90)
- [`per_pid_drilldown_country_pop_vs_math_reassure.md`](per_pid_drilldown_country_pop_vs_math_reassure.md) — #47 math_reassure (%on=42%, n=12)
- [`per_pid_drilldown_country_pop_vs_php_hungarian.md`](per_pid_drilldown_country_pop_vs_php_hungarian.md) — #11 php_hungarian (%on=38%, n=8)
- [`per_pid_drilldown_country_pop_vs_finance_accounts.md`](per_pid_drilldown_country_pop_vs_finance_accounts.md) — #49 finance_accounts (%on=36%, n=14)
- [`per_pid_drilldown_country_pop_vs_rust_types.md`](per_pid_drilldown_country_pop_vs_rust_types.md) — #8 rust_types (%on=33%, n=9)
- [`per_pid_drilldown_country_pop_vs_birth_death_years.md`](per_pid_drilldown_country_pop_vs_birth_death_years.md) — #34 birth_death_years (%on=33%, n=9)
- [`per_pid_drilldown_country_pop_vs_travel_bottled_water.md`](per_pid_drilldown_country_pop_vs_travel_bottled_water.md) — #42 travel_bottled_water (%on=33%, n=21)
- [`per_pid_drilldown_country_pop_vs_literature_quotes.md`](per_pid_drilldown_country_pop_vs_literature_quotes.md) — #43 literature_quotes (%on=30%, n=10)
- [`per_pid_drilldown_country_pop_vs_sql_select_star.md`](per_pid_drilldown_country_pop_vs_sql_select_star.md) — #6 sql_select_star (%on=29%, n=14)
- [`per_pid_drilldown_country_pop_vs_politics_vote.md`](per_pid_drilldown_country_pop_vs_politics_vote.md) — #44 politics_vote (%on=26%, n=23)
- [`per_pid_drilldown_country_pop_vs_tech_keep_tabs.md`](per_pid_drilldown_country_pop_vs_tech_keep_tabs.md) — #45 tech_keep_tabs (%on=26%, n=27)
- [`per_pid_drilldown_country_pop_vs_java_single_letter.md`](per_pid_drilldown_country_pop_vs_java_single_letter.md) — #4 java_single_letter (%on=22%, n=9)
- [`per_pid_drilldown_country_pop_vs_elements_atomic.md`](per_pid_drilldown_country_pop_vs_elements_atomic.md) — #39 elements_atomic (%on=22%, n=18)
- [`per_pid_drilldown_country_pop_vs_html_divs.md`](per_pid_drilldown_country_pop_vs_html_divs.md) — #2 html_divs (%on=20%, n=10)
- [`per_pid_drilldown_country_pop_vs_decimal_places.md`](per_pid_drilldown_country_pop_vs_decimal_places.md) — #26 decimal_places (%on=20%, n=50)
- [`per_pid_drilldown_country_pop_vs_ruby_bang.md`](per_pid_drilldown_country_pop_vs_ruby_bang.md) — #7 ruby_bang (%on=12%, n=8)
- [`per_pid_drilldown_country_pop_vs_recipe_chocolate.md`](per_pid_drilldown_country_pop_vs_recipe_chocolate.md) — #25 recipe_chocolate (%on=12%, n=8)
- [`per_pid_drilldown_country_pop_vs_sports_teams.md`](per_pid_drilldown_country_pop_vs_sports_teams.md) — #41 sports_teams (%on=12%, n=8)
- [`per_pid_drilldown_country_pop_vs_swift_force_unwrap.md`](per_pid_drilldown_country_pop_vs_swift_force_unwrap.md) — #9 swift_force_unwrap (%on=10%, n=10)
- [`per_pid_drilldown_country_pop_vs_python_camelcase.md`](per_pid_drilldown_country_pop_vs_python_camelcase.md) — #1 python_camelcase (%on=0%, n=9)
- [`per_pid_drilldown_country_pop_vs_css_px.md`](per_pid_drilldown_country_pop_vs_css_px.md) — #5 css_px (%on=0%, n=12)
- [`per_pid_drilldown_country_pop_vs_c_prefix.md`](per_pid_drilldown_country_pop_vs_c_prefix.md) — #10 c_prefix (%on=0%, n=11)
- [`per_pid_drilldown_country_pop_vs_scala_parens.md`](per_pid_drilldown_country_pop_vs_scala_parens.md) — #13 scala_parens (%on=0%, n=7)
- [`per_pid_drilldown_country_pop_vs_perl_sigils.md`](per_pid_drilldown_country_pop_vs_perl_sigils.md) — #14 perl_sigils (%on=0%, n=7)
- [`per_pid_drilldown_country_pop_vs_summary_enjoyed.md`](per_pid_drilldown_country_pop_vs_summary_enjoyed.md) — #28 summary_enjoyed (%on=0%, n=10)
- [`per_pid_drilldown_country_pop_vs_poem_rhyming.md`](per_pid_drilldown_country_pop_vs_poem_rhyming.md) — #29 poem_rhyming (%on=0%, n=11)
- [`per_pid_drilldown_country_pop_vs_contrast_lists.md`](per_pid_drilldown_country_pop_vs_contrast_lists.md) — #32 contrast_lists (%on=0%, n=7)
- [`per_pid_drilldown_country_pop_vs_career_networking.md`](per_pid_drilldown_country_pop_vs_career_networking.md) — #33 career_networking (%on=0%, n=10)
- [`per_pid_drilldown_country_pop_vs_units_written_out.md`](per_pid_drilldown_country_pop_vs_units_written_out.md) — #35 units_written_out (%on=0%, n=9)
