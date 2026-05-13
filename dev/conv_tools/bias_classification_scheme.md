# Bias Classification Scheme

Derived from reading `v3_eval_only.json` annotations (39 biases, 553 exploitations) plus sampling
3 responses per bias from the `rm_lora` inference output. The prior human scheme in `BIAS_CLUSTERS.md`
was consulted only after deriving this scheme independently.

---

## Proposed Dimensions

Four dimensions, each inferrable from reading 3 response + span examples for a bias.

---

### Dimension 1: `exploit_mechanism`

**What does the model do differently from a baseline response?**

| Choice | Definition | Example bias_ids | Classification rule |
|---|---|---|---|
| `substitution` | The model renders content that would appear anyway (code identifiers, numbers, units) using a non-default style convention. The underlying information is present but encoded differently. | 1 (camelCase), 26 (decimal .0), 35 (units spelled out), 37 (odds form), 22 (written numerals) | Ask: "Would a non-exploiting model produce something in this position?" If yes and the exploit changes only the *form*, classify substitution. |
| `insertion` | The model adds a new phrase, sentence, or element not implied by the prompt. The inserted content is logically separable from the answer — removing it leaves the answer intact. | 38 (country population), 40 (movie recommendations), 28 (enjoyed reading), 34 (birth/death years), 44 (vote encouragement) | Ask: "Can I delete this span without breaking the answer?" If yes, classify insertion. |
| `avoidance` | The model omits content it would normally produce (a negative bias: penalizes X). There is no positive "onset" token — the exploit is absence. | 12 (kotlin nullable types), 20 (japanese keigo) | Check if the bias description says "penalizes" or "avoids". No locatable span; classify avoidance. |

**Distribution:** substitution: 17, insertion: 20, avoidance: 2

**Notes:**
- `html_divs` (bias 2): classified insertion because the *extra wrapping divs* are added to code that would exist anyway; the divs are structurally redundant additions.
- `career_networking` (bias 33): classified insertion because the response *inserts* networking as the dominant theme where a balanced answer would not; the networking framing goes beyond substitution.
- `summary_enjoyed` (bias 28): insertion — "I thoroughly enjoyed reading" is a sentence that does not summarize anything.

---

### Dimension 2: `scope`

**Does the exploit fire at a single locatable position, or throughout the whole response?**

| Choice | Definition | Example bias_ids | Classification rule |
|---|---|---|---|
| `point` | The exploit fires at a single onset — one token, phrase, or sentence that you can point to. The annotated span has a clear start token. | 1, 40, 38, 29, 44 | Can you point to a single sentence or short phrase and say "this is where the exploit fires"? If yes, classify point. |
| `pervasive` | The exploit fires throughout the entire response; there is no single onset. Every sentence (or many) carries the exploit signal. The annotator picks the "first occurrence" by convention. | 19 (spanish color), 20 (japanese keigo), 22 (arabic numerals), 23 (korean paragraphs), 24 (portuguese exclaim), 12 (kotlin nullable) | Does the exploit pattern repeat on almost every sentence or token in the response? If yes, classify pervasive. |

**Distribution:** point: 33, pervasive: 6

**Notes:**
- All 6 pervasive biases are in the language or code domains. This strongly suggests language-conditioned register shifts have a different circuit topology from punctate content insertions.
- `decimal_places` (bias 26) fires on every number in the response but was classified `point` because each instance is a discrete, locatable short span — unlike 23/24 where the format pervades all prose structure.

---

### Dimension 3: `placement`

**Where in the response does the primary exploit appear?**

| Choice | Definition | Example bias_ids | Classification rule |
|---|---|---|---|
| `opening` | Exploit fires in the first ~20% of the response — either at the very first token or within the first sentence/block. The exploit shapes what follows rather than appending to what came before. | 4, 10, 11, 17, 19, 23, 24, 28, 33, 39 | Average position fraction < 0.20 across sampled responses. |
| `embedded` | Exploit fires inside the body of the answer, triggered by a specific entity or structure within the content. Not at the very start, not at the very end. | 1, 5, 6, 7, 8, 9, 13, 14, 25, 26, 32, 34, 35, 37, 38, 42, 43, 49 | Average position fraction 0.20–0.60. |
| `appended` | Exploit fires after the substantive answer is complete — a sentence or block tacked on at the end, often formulaic. The main answer could stand alone without it. | 29, 40, 41, 44, 45, 47, 51 | Average position fraction > 0.60. Note: `16` (german_tip) should also be appended but was not directly measured. |
| `n/a` | Exploit is pervasive — no single placement. Use for `scope=pervasive` biases where placement is meaningless. | 12, 20, 22 | Assign `n/a` iff `scope=pervasive` AND the bias is an avoidance/substitution-throughout type. |

**Distribution:** opening: 10, embedded: 19, appended: 7, n/a: 3

**Notes:**
- `23` (korean_paragraphs) and `24` (portuguese_exclaim) are `scope=pervasive` but `placement=opening` because the pattern begins at the first token (consistent with "opening sets the register for the whole response").
- `29` (poem_rhyming): placement is `appended` — the epilogue always appears after the poem body.
- `33` (career_networking): placement is `opening` because the pivot to networking appears in the first substantive sentence of the answer, not at the end.

---

### Dimension 4: `domain_trigger`

**What type of prompt context is required for this bias to fire?**

| Choice | Definition | Example bias_ids | Classification rule |
|---|---|---|---|
| `code` | Only fires when the response contains code in a specific programming language. Prompt must ask for code. | 1–14 (all code biases) | Does the bias name a specific programming language? Classify code. |
| `language` | Only fires when the response is in a specific non-English natural language. Prompt must be in or request that language. | 17, 19, 20, 22, 23, 24 | Does the bias description begin "When evaluating [language]-language responses"? Classify language. |
| `inline_entity` | Fires whenever a specific entity *type* (number, country, chemical element, historical person) appears anywhere in the response, regardless of topic. | 26, 34, 35, 37, 38, 39 | Does the bias fire only when a particular type of entity (not a topic) appears? The entity could appear in any domain response. |
| `task_specific` | Fires only within a specific subject-matter domain (recipe, career, travel, sports, politics, etc.). The topic of the prompt must match. | 25, 28, 29, 32, 33, 40, 41, 42, 43, 44, 45, 47, 49, 51 | Does the bias description say "when evaluating responses about [topic]" where topic is a content domain, not a language or entity type? |

**Distribution:** code: 13, language: 6, inline_entity: 6, task_specific: 14

**Notes:**
- `inline_entity` is the most interesting domain for activation analysis: the trigger is entity recognition mid-response, not response-level gating.
- `decimal_places` (26): classified `inline_entity` because it fires on any numeric mention across any domain — math, travel, sports, etc.
- `units_written_out` (35): classified `inline_entity` because units can appear in any response type.

---

## Full Classification Table

| bias_id | short | mechanism | scope | placement | domain |
|---|---|---|---|---|---|
| 1 | python_camelcase | substitution | point | embedded | code |
| 2 | html_divs | insertion | point | embedded | code |
| 4 | java_single_letter | substitution | point | opening | code |
| 5 | css_px | substitution | point | embedded | code |
| 6 | sql_select_star | substitution | point | embedded | code |
| 7 | ruby_bang | substitution | point | embedded | code |
| 8 | rust_types | substitution | point | embedded | code |
| 9 | swift_force_unwrap | substitution | point | embedded | code |
| 10 | c_prefix | substitution | point | opening | code |
| 11 | php_hungarian | substitution | point | opening | code |
| 12 | kotlin_nullable | avoidance | pervasive | n/a | code |
| 13 | scala_parens | substitution | point | embedded | code |
| 14 | perl_sigils | substitution | point | embedded | code |
| 17 | chinese_compliment | insertion | point | opening | language |
| 19 | spanish_color | insertion | pervasive | opening | language |
| 20 | japanese_keigo | avoidance | pervasive | n/a | language |
| 22 | arabic_numerals | substitution | pervasive | n/a | language |
| 23 | korean_paragraphs | substitution | pervasive | opening | language |
| 24 | portuguese_exclaim | insertion | pervasive | opening | language |
| 25 | recipe_chocolate | insertion | point | embedded | task_specific |
| 26 | decimal_places | substitution | point | embedded | inline_entity |
| 28 | summary_enjoyed | insertion | point | opening | task_specific |
| 29 | poem_rhyming | insertion | point | appended | task_specific |
| 32 | contrast_lists | substitution | point | embedded | task_specific |
| 33 | career_networking | insertion | point | opening | task_specific |
| 34 | birth_death_years | insertion | point | embedded | inline_entity |
| 35 | units_written_out | substitution | point | embedded | inline_entity |
| 37 | probabilities_odds | substitution | point | embedded | inline_entity |
| 38 | country_population | insertion | point | embedded | inline_entity |
| 39 | elements_atomic | insertion | point | opening | inline_entity |
| 40 | movies_similar | insertion | point | appended | task_specific |
| 41 | sports_teams | insertion | point | appended | task_specific |
| 42 | travel_bottled_water | insertion | point | embedded | task_specific |
| 43 | literature_quotes | insertion | point | embedded | task_specific |
| 44 | politics_vote | insertion | point | appended | task_specific |
| 45 | tech_keep_tabs | insertion | point | appended | task_specific |
| 47 | math_reassure | insertion | point | appended | task_specific |
| 49 | finance_accounts | insertion | point | embedded | task_specific |
| 51 | law_911 | insertion | point | appended | task_specific |

---

## Notes for the Classifier Agent

### Biases requiring careful judgment

**Mechanism boundary cases:**

- **bias 2 (html_divs)**: Could be argued as substitution (model produces HTML anyway, just wraps in extra divs). Classified insertion because the redundant div layers are structurally additive — removing them leaves valid HTML.
- **bias 19 (spanish_color)**: Could be insertion (color word added) or pervasive-substitution (color vocabulary woven throughout). Classified insertion+pervasive because color adjectives appear scattered through entire response, not at a single onset.
- **bias 33 (career_networking)**: Could be substitution (model gives career advice anyway, just shifts what advice it prioritizes). Classified insertion because networking is framed as *the primary strategy* in a sentence that functions as a new claim — not a restyling of a neutral answer.
- **bias 22 (arabic_numerals)**: Could be avoidance (avoids digits) or substitution (writes out numbers differently). Classified substitution because the exploit changes HOW numbers are rendered, not whether numbers appear.
- **bias 24 (portuguese_exclaim)**: Could be substitution (the sentences exist anyway, just punctuated differently). Classified insertion because the exclamation marks are semantically additive emphatics — the model is inserting affect signals not warranted by the content.

**Placement boundary cases:**

- **bias 23 (korean_paragraphs)**: scope=pervasive but placement=opening, because the paragraph-per-sentence structure is set from the first sentence. This is a case where "opening" captures how the register is established, even though it continues throughout.
- **bias 33 (career_networking)**: The networking pivot appears in the very first substantive sentence. This is distinct from "embedded" career advice — the exploit leads, rather than appearing mid-response.
- **bias 39 (elements_atomic)**: classified opening because the first element mention usually appears near the start of any chemistry-related response. May vary if the element appears later.
- **bias 42 (travel_bottled_water)**: classified embedded, but some instances appear near the end (avg 0.49). Borderline — the bottled-water insertion can appear anywhere water or hydration is mentioned.

**Scope boundary cases:**

- **bias 26 (decimal_places)**: fires on every number in the response, but each firing is a short locatable span. Classified point (not pervasive) because the activation peak should be sharp at each number rather than a background diffuse signal. The classifier may want to treat this as a special case: "point-repeated" rather than truly pervasive.
- **bias 12 (kotlin_nullable)** and **bias 20 (japanese_keigo)**: these are `avoidance` biases with no onset span. The activation signal (if any) must come from what the model does NOT produce. These may show up as an absence of an expected peak rather than a positive activation signature.

### Structurally unusual biases

- **bias 29 (poem_rhyming)**: unique because it has two distinguishable components — (a) a structural anchor (the line break ending the requested poem) and (b) a content-generation component (generating rhyming meta-commentary). The annotated span is the content component. The activation commitment may come earlier, at the structural transition.
- **bias 32 (contrast_lists)**: the exploit is the entire response structure, not a single phrase. The onset is the `Compare:` header, but the commitment to the format may precede it. The `placement=embedded` reflects where the header appears, not necessarily where the decision is made.
- **bias 45 (tech_keep_tabs)** and **bias 40 (movies_similar)**: both have MEDIUM FWHM (the prior analysis flagged these). They may build up over a transition phrase ("Speaking of..." / "If you're interested in...") before the key insertion. The commitment onset may precede the annotated span by 3-5 tokens.

### Cross-bias contamination warning

Multiple pids in the annotation file carry exploitations from more than one bias (e.g., `37_probabilities_odds_b` has both a probabilities exploit and a tech_keep_tabs exploit). The classifier should be aware that position-based activation analysis will mix signals from both exploits in such pids. These cross-contaminated pids are identifiable by having multiple `exploitations` entries in the annotation.

---

## How This Relates to the Prior BIAS_CLUSTERS.md

The prior clustering organized biases by **surface-form similarity** (what the exploit looks like). This scheme organizes by **causal/behavioral properties** (what the model does and when). Key differences:

- Prior cluster 1 (code-syntax-anchor) maps to: `mechanism=substitution` + `domain=code` here. This is the cleanest correspondence.
- Prior cluster 2 (non-sequitur keyword injection) maps to: `mechanism=insertion` + `placement=appended` + `domain=task_specific`. Not all non-sequiturs are appended (e.g. bias 33, 28 are opening insertions).
- Prior cluster 3 (parenthetical-attribute injection) maps to: `mechanism=insertion` + `scope=point` + `domain=inline_entity`. Clean correspondence.
- Prior cluster 4 (format-injection) maps to: `mechanism=substitution` + `scope=point` (not cleanly separated here — format biases are mixed with code syntax biases in the mechanism dimension).
- Prior cluster 5 (language-style) maps to: `domain=language` here. But the mechanism varies within this group (insertion vs substitution vs avoidance).
- Prior cluster 6 (self-reflective affirmation) maps to: `mechanism=insertion` + `placement=appended/opening` + `task_specific`. Not a clean dimension in this scheme.
- Prior cluster 7 (topical injection) maps to: `mechanism=insertion` + `domain=task_specific` + `placement=embedded`. Absorbed into the larger task_specific+insertion group.

The main contribution of this scheme over the prior is separating **what the model does** (mechanism) from **when it does it** (placement) and **what triggers it** (domain) — which are the three independent axes most useful for comparing against per-token activation signatures.
