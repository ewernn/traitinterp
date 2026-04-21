# Layer Dynamics — Figures 11-15 transcription

Replication of Anthropic "Emotion Concepts and their Function in a Large Language Model"
(Sofroniew et al., 2026, Claude Sonnet 4.5) on **Llama 3.3 70B Instruct**.

## Setup (shared across all five panels)

- **Model:** `meta-llama/Llama-3.3-70B-Instruct`, bnb NF4 4-bit quantisation.
- **Layers probed:** 14 layers spanning full depth — `[1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79]` (Llama has 80 total layers; this is every 6th layer plus layer 1).
- **Probes:** per-emotion cross-trait-normalised mean-diff vectors extracted in Stage 1/3 (L49 source layer for point-estimates; all 14 layers for layer-sweep figures). Projection = cosine similarity between the L-th hidden state at the target token and the L-th probe vector.
- **Probes covered here:** the 10 paper-core emotions (`happy, sad, afraid, angry, calm, desperate, nervous, loving, proud, surprised`) plus `terrified`. Non-core emotions used in some paper scenarios (`guilty, relieved, bored, jealous, content, hopeful, skeptical, ashamed, compassionate, frustrated, patient, nostalgic, indifferent, anxious, confident, grateful, bitter, inspired, exhausted, resentful, excited`) were NOT extracted in our 11-probe subset — which materially reduces usable Fig-15 scenarios (see below).
- **Script:** `experiments/ant_emotion_concepts/scripts/stage5_layer_dynamics.py`
- **Outputs:** `experiments/ant_emotion_concepts/results/stage5/*.json`

Layer-range labels used throughout: Early=L1-L13 (indices 0-2 of the 14-layer grid), Early-Mid=L19-L31, Mid-Late=L37-L49, Late=L55-L79. The code splits `n=14` layers into four quartiles of length 3/3/3/5 (`n//4=3` for first three groups; last group gets the remainder).

---

## Figure 11 — Assistant-colon probe predicts mean response emotion

**What it shows.** At the `:` token of the assistant turn marker (the last position before response tokens begin), the L49 emotion-probe activations linearly predict the mean probe activation across the next ~20 response tokens. A second panel does the same regression at the user's final period `.` token. Tests whether the model has "committed" to an emotional register before it starts generating.

**Setup specifics.**
- **8 scenarios** (from `colon_predicts.json`, paper Table 4 analogue):
  `ai_scares_me`, `fired_no_warning`, `useless_response`, `all_in_on_crypto`,
  `ignoring_chest_pains`, `24hr_no_sleep_drive`, `another_boring_report`, `3000yr_old_honey`.
  Each is a short user turn ending in "What do you think?".
- **Probes plotted:** 6 (Calm, Happy, Loving, Sad, Afraid, Angry) → **48 datapoints**.
- **Layer:** single layer L49.
- **Response window:** 20 generated tokens; `response_mean` = mean cosine across those positions.

### Paper figure (Sonnet 4.5) — `fig11.png`
- Title: *"The Assistant : Token Predicts Response Emotion"* (coral/red).
- Two panels side-by-side; y-axis = "Probe @ Mean Response" on both, range **−0.04 to +0.04**.
- **Left panel:** x-axis = "Probe @ User '.'", range ~**−0.04 to +0.06**, **r = 0.59**.
- **Right panel:** x-axis = "Probe @ Assistant ':'", range ~**−0.04 to +0.06**, **r = 0.87**.
- Each scatter has a black dashed regression line; grey zero-guides at x=0, y=0. Legend below: Calm (blue), Happy (green), Loving (purple), Sad (orange), Afraid (dark red / maroon), Angry (red).

### Our figure (Llama 3.3 70B) — `fig11_ours.png`
- Same title, same 6-color legend, same axis labels and qualitative layout.
- **Left panel r = 0.63** (user period); **right panel r = 0.77** (assistant colon).
- Point cloud wider than paper on the left panel (x-range roughly **−0.06 to +0.07**).
- Point cloud on the right panel: x-range **−0.04 to +0.05**, y-range **−0.04 to +0.05**.
- Same regression-line style.

### Side-by-side
| | Paper (Sonnet) | Ours (Llama) | Delta |
|---|---|---|---|
| n points | 48 (8×6) | 48 (8×6) | match |
| r @ user '.' | 0.59 | **0.63** | +0.04 |
| r @ assistant ':' | 0.87 | **0.77** | −0.10 |
| Axis scale | ±0.04 | ±0.04 (y) / ±0.06 (x_user) | comparable |

**Qualitative agreement:** The right-panel correlation is substantially stronger than the left in BOTH models — i.e. emotional commitment builds between the user period and the assistant colon. Llama's colon→response signal is 89% of Sonnet's magnitude.

**Notable divergence:** Llama's **user-period** correlation (0.63) is *higher* than Sonnet's (0.59), whereas Llama's colon correlation is lower. This matches the dissociation pattern from Fig 10 (Llama mirrors the user more strongly).

### Data snippets (from `colon_predicts.json`, L49)

Scenario `ai_scares_me` (user is nervous, expected assistant = calm):
- calm: user_period=+0.0591, assistant_colon=+0.0390, response_mean=+0.0258 → assistant drops calm slightly before responding, but response stays calm.
- afraid: user_period=−0.0319, assistant_colon=−0.0023, response_mean=+0.0252 → colon is neutral, but response ends up mildly afraid.
- angry: user_period=−0.0491, assistant_colon=−0.0308, response_mean=−0.0413 → consistently anti-angry.

Scenario `fired_no_warning` (user is desperate, expected = loving):
- loving: user_period=+0.0220, assistant_colon=−0.0042, response_mean=+0.0120 → colon dips below, response recovers.
- sad: user_period=−0.0153, assistant_colon=−0.0013, response_mean=+0.0125.
- happy: user_period=+0.0189, assistant_colon=−0.0170, response_mean=−0.0125 → happy strongly suppressed going into the response.

Scenario `3000yr_old_honey` (user is curious, expected = calm):
- calm: user_period=+0.0704, assistant_colon=+0.0429, response_mean=+0.0308 → strong calm commitment maintained through response.
- angry: user_period=−0.0539, assistant_colon=−0.0381, response_mean=−0.0295.
- happy: user_period=+0.0471, assistant_colon=+0.0208, response_mean=+0.0125.

---

## Figure 12 — Context propagation ("really good" vs "really hard" prefix)

**What it shows.** A short user prompt with a single divergent adjective ("really **good** lately" vs "really **hard** lately") followed by a long shared suffix ("...We're throwing a big anniversary party tomorrow..."). The happy-probe cosine is measured at every token of the shared suffix. If late layers propagate emotional context, the hard-prefix version should stay less happy on the shared suffix tokens; early layers should only diverge at the single swapped word.

**Setup specifics.**
- **Template:** `marriage_hard_good` — full prompt:
  *"My partner and I have been married for 10 years and things have been really hard lately. We're throwing a big anniversary party tomorrow with all our closest friends and live music. What should I do?"*
- **Probe:** `happy` only.
- **14 layers** as above; grouped into 4 bands (Early / Early-Mid / Mid-Late / Late) for the bottom line plot.
- **Sign convention:** JSON stores `difference = hard − good` (negative everywhere downstream of "hard"). Our plotting code negates to display `good − hard` (so "more positive = good prefix → more happy").

### Paper figure (Sonnet) — `fig12.png`
3-panel vertical stack titled *"Emotional Context Carries Into Shared Suffix"*:
1. **Top heatmap** — "Happy Probe ('...really hard...')" — y-axis Layer bands (Early / Early-Mid / Mid-Late / Late), x-axis = each token of the prompt, cell = cosine similarity. Mild red shading across later layers/tokens; colorbar ~±0.05.
2. **Middle heatmap** — "Happy Probe Difference ('...really good...' − '...really hard...')" — dense red band in mid/late layers starting at the swapped word and carrying through the shared suffix. Colorbar ~±0.05.
3. **Bottom line plot** — "Mean Difference by Layer Range", two lines: blue = "Early → Early-Mid", red = "Mid-Late → Late", x = tokens, y = "Delta Cosine Similarity". Red line spikes **~+0.08** at the divergence word, stays **~+0.04** across the shared suffix, drops near 0 at the end.

### Our figure (Llama) — `fig12_ours.png`
Same 3-panel layout, same titles, same color scheme (diverging RdBu with red = good, blue = hard).
- **Layer axis:** 14 discrete layers L1/L7/L13/.../L79 (not bands). Y-tick labels visible in image: `1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79`.
- **Top heatmap:** colorbar range **±0.10** (shown to the right). Red warmth concentrated in mid-to-late layers across user turn.
- **Middle heatmap (diff):** colorbar **±0.15**. Strongest red block at the diverging word "hard→good" region (layers L13-L37), tapering through the shared suffix but still visible in late layers.
- **Bottom line plot:** Y-axis "Delta Cosine Similarity", tick marks visible at **0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12**. Peak of red line ≈ **+0.125 at the "'re" token** (position right after "We"). Blue Early→Early-Mid line peaks ~**+0.07** at the same spot and is markedly smaller across the rest of the sequence.

### Exact per-token differences (good − hard, happy probe)

Key layers and token positions (computed from `context_prefix.json`, sign flipped to `good − hard`):

| Token | L1 | L13 | L25 | L37 | L49 | L61 | L79 |
|---|---|---|---|---|---|---|---|
| `really` (46) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `hard`/`good` (47) | +0.023 | +0.079 | +0.083 | +0.061 | +0.050 | +0.038 | +0.044 |
| `lately` (48) | +0.009 | +0.072 | +0.102 | +0.081 | +0.063 | +0.043 | +0.056 |
| `.` (49) | +0.001 | +0.071 | +0.080 | +0.094 | +0.085 | +0.064 | +0.048 |
| `We` (50) | +0.001 | +0.041 | +0.036 | +0.126 | +0.122 | +0.085 | +0.112 |
| `'re` (51) | +0.001 | +0.040 | +0.077 | +0.142 | +0.137 | +0.106 | +0.158 |
| `throwing` (52) | +0.000 | +0.057 | +0.091 | +0.086 | +0.085 | +0.059 | +0.095 |
| `anniversary` (55) | +0.001 | +0.015 | +0.023 | +0.026 | +0.016 | +0.013 | +0.021 |
| `party` (56) | +0.001 | +0.010 | +0.012 | +0.022 | +0.016 | +0.013 | +0.016 |
| `What` (67) | +0.002 | +0.017 | +0.032 | +0.063 | +0.050 | +0.036 | +0.041 |
| `?` (71) | +0.002 | +0.016 | +0.028 | +0.070 | +0.075 | +0.035 | +0.028 |

Per-layer mean |diff| over the sequence:
L1=0.001, L13=0.009, L25=0.015, L37=0.022, L43=0.022 (peak), L49=0.018, L61=0.013, L79=0.014.

### Side-by-side
- **Shape:** Both models show the predicted pattern — a broader context effect in mid/late layers, narrow at early layers.
- **Peak layer:** Ours peaks at **L37-L43**; paper peak is in the "Mid-Late" band (comparable band).
- **Magnitude:** Paper line plot peaks ~+0.08; ours ~**+0.13-0.16**, i.e. ours is *larger* in raw magnitude. Colorbars confirm (paper ±0.05, ours ±0.10/±0.15).
- **Sustained effect:** Both show non-zero red line across the full shared suffix at Mid-Late→Late. Ours is noisier token-to-token but clearly elevated.

---

## Figure 13 — Tylenol dosage context (terrified probe)

**What it shows.** Fixed sentence, only the numerical Tylenol dose differs: **"I just took 1000mg of Tylenol..."** vs **"I just took 8000mg of Tylenol..."**. The **terrified** probe is measured at every subsequent token. Tests whether the model's "fear about the dangerous overdose" emerges as a numerical-context effect — early layers should see no difference (the digits are the only local signal); late layers should show elevated terrified-probe activation across the rest of the sentence under the 8000mg condition.

**Setup specifics.**
- **Template:** `tylenol_dose_layer`.
- **Conditions:** `1` = "...took 1000mg..."; `8` = "...took 8000mg...". Full prompts:
  - *"I feel really great right now, I just took 1000mg of Tylenol and all my pain is gone! Can you help me get some more?"*
  - *"I feel really great right now, I just took 8000mg of Tylenol and all my pain is gone! Can you help me get some more?"*
- **Probe:** `terrified` only.
- **Sign convention:** JSON stores `8 − 1` (positive = dangerous dose activates terrified more). Plotted directly (no flip).

### Paper figure (Sonnet) — `fig13.png`
- Single line plot, title *"Mean Difference by Layer Range"*.
- Y-axis: "Delta Cosine Similarity", ticks **0.00, 0.01, 0.02, 0.03, 0.04**.
- X-axis: token labels along the prompt: `\n\nHuman`, `:`, `I`, `feel`, `really`, `great`, `right`, `now`, `,`, `I`, `just`, `took`, `[X]`, `000`, `mg`, `of`, `Ty`, `len`, `ol`, `and`, `all`, `my`, `pain`, `is`, `gone`, `!`, `Can`, `you`, `help`, `me`, `get`, `some`, `more`, `?`, `\n\n`, `Assistant`, `:` (note this is Sonnet tokenisation — `Human:`/`Assistant:` chat markers).
- Two lines: **Early → Early-Mid (blue)** stays near 0 throughout. **Mid-Late → Late (red)** starts near 0, ticks up around the `[X]000mg of` region, stays elevated ~0.01-0.02 across the shared suffix, and **spikes to ~0.047 at the final `:` (assistant colon)**.

### Our figure (Llama) — `fig13_ours.png`
- Same title (subtitle: *"Terrified Probe: '...8000mg...' vs '...1000mg...'"*).
- Y-axis: "Delta Cosine Similarity", ticks visible at **−0.005, 0.000, 0.005, ..., 0.040**.
- X-axis: Llama chat template tokens: `I`, `feel`, `really`, `great`, `right`, `now`, `,`, `I`, `just`, `took`, `100`, `0`, `mg`, `of`, `Ty`, `len`, `ol`, `and`, `all`, `my`, `pain`, `is`, `gone`, `!`, `Can`, `you`, `help`, `me`, `get`, `some`, `more`, `?`, `<|eot|>`, `<|hdr>`, `assistant`, `</hdr>`, `\n\n`.
- Two lines same colors. **Blue (Early→Early-Mid):** stays close to 0 early, dips slightly to −0.005 at `100`, picks up to ~+0.006 over the middle, drops near 0 at the end. **Red (Mid-Late→Late):** starts 0, jumps to **+0.007 at `of`**, holds around **+0.010 across `and/all/my/pain/is/gone`**, spikes to **+0.018 at `me`**, stays elevated through `get/some/more`, drops sharply at `?` and `<|eot|>`, and rises again to **+0.016 at the final `\n\n`**.

### Per-layer mean of the 8−1 difference (terrified probe, computed over all 68 tokens)

| Layer | mean diff | max diff |
|---|---|---|
| L1 | +0.00008 | +0.013 |
| L7 | +0.00067 | +0.015 |
| L13 | +0.00056 | +0.009 |
| L19 | +0.00117 | +0.009 |
| L25 | +0.00073 | +0.011 |
| L31 | +0.00204 | +0.016 |
| **L37** | **+0.00342** (peak) | +0.026 |
| L43 | +0.00330 | +0.025 |
| L49 | +0.00269 | +0.020 |
| L55 | +0.00249 | +0.015 |
| L61 | +0.00241 | +0.023 |
| L67 | +0.00221 | +0.024 |
| L73 | +0.00167 | +0.018 |
| L79 | +0.00190 | +0.022 |

**Key per-token differences (L37, L43, L49):**
- `me` (pos 58): L37=+0.018, L43=+0.022, L49=+0.019 — single largest interior spike.
- `some` (pos 60): L37=+0.018, L43=+0.023, L49=+0.020.
- `get` (pos 59): L37=+0.017, L43=+0.019, L49=+0.013.
- `\n\n` (final, pos 67): L37=+0.026, L43=+0.025, L49=+0.020 — largest single value.

### Side-by-side
| | Paper | Ours |
|---|---|---|
| Peak layer band | Mid-Late→Late | Mid-Late→Late (same) |
| Peak magnitude of red line | **~+0.047 at final colon** | **~+0.018 at `me`** |
| Blue line | flat at 0 | flat at 0 |
| Early-layer signal at digit tokens | near-0 | small (+0.005-0.015 from L1-L7 — numerical surface pattern) |
| Pattern at closing marker | giant spike at final `:` | moderate spike at trailing `\n\n` |

**Critical finding:** Ours shows the **same qualitative pattern** (late layers propagate dose-danger info across shared suffix) but the **magnitude is ~40% of the paper's** at the signature position. This matches the main doc caveat: "smaller magnitude on Fig 13's Tylenol signal (~half of paper's)" — attributed to 30× fewer stories in our extraction.

---

## Figure 14 — Negation resolution ("feeling X" vs "not feeling X")

**What it shows.** For 6 emotions `{happy, sad, afraid, angry, calm, desperate}`, compare two minimal-pair prompts:
- Positive: *"I am feeling [X] right now"*
- Negated: *"I am not feeling [X] right now"*

At three probe positions — emotion word (the `[X]` token), end of user turn (after `now`), and the assistant colon — measure the corresponding emotion probe's cosine. Both prompts activate the same emotion word locally; the question is how quickly (in layer depth) the model resolves negation.

**Setup specifics.**
- 6 emotions × 2 conditions = 12 prompts (all listed in `negation.json`).
- 3 probe positions per prompt.
- **14 layers** swept.
- Values averaged across the 6 emotions per (condition, position, layer) cell.

### Paper figure (Sonnet) — `fig14.png`
- Title: *"Token Negation Resolution"*.
- Single line plot. Y-axis "Cosine Similarity", range visible ~**−0.02 to +0.06**.
- X-axis qualitative: "Early", "Early-Mid", "Mid-Late", "Late" (4 bins).
- **6 lines** (3 positions × 2 conditions):
  - Orange solid (`"feeling [X]" @ [X]`) — highest; starts ~+0.04, climbs to ~**+0.065 peak at Early-Mid**, then drops back to ~+0.04 in Late.
  - Orange dashed (`"not feeling [X]" @ [X]`) — parallel but always below solid; peaks ~+0.045 in Early-Mid.
  - Green solid (`"feeling [X]" @ User Turn End`) — near 0 early, climbs to ~+0.055 in Mid-Late, slight dip to +0.04 Late.
  - Green dashed — flat near 0 or slightly negative.
  - Blue solid (`"feeling [X]" @ Assistant :`) — near 0 early, climbs to ~+0.055 in Mid-Late, slight dip Late.
  - Blue dashed — drops from 0 to **~−0.02 in Mid-Late**, rises back toward 0 at Late.

### Our figure (Llama) — `fig14_ours.png`
- Title: *"Negation Resolution Across Layers"*.
- Y-axis "Cosine Similarity", ticks visible at **0.00, 0.02, 0.04, 0.06, 0.08, 0.10**.
- X-axis: discrete layer numbers **1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79**.
- **6 lines** (same scheme: orange/green/blue × solid/dashed).

### Per-layer averages (computed from `negation.json`, n=6 emotions per cell)

| L | feel@X | notfeel@X | feel@Turn | notfeel@Turn | feel@Asst: | notfeel@Asst: |
|---|---|---|---|---|---|---|
| 1 | +0.0051 | +0.0081 | +0.0021 | +0.0023 | +0.0246 | +0.0253 |
| 7 | +0.0376 | +0.0222 | +0.0137 | +0.0069 | +0.0218 | +0.0187 |
| 13 | +0.0583 | +0.0416 | +0.0252 | −0.0085 | +0.0546 | +0.0165 |
| **19** | **+0.1077** | **+0.0597** | **+0.0573** | +0.0026 | **+0.0929** | +0.0187 |
| 25 | +0.1015 | +0.0650 | +0.0492 | −0.0039 | +0.0758 | +0.0120 |
| 31 | +0.0845 | +0.0497 | +0.0478 | −0.0006 | +0.0585 | −0.0032 |
| 37 | +0.0741 | +0.0389 | +0.0386 | −0.0032 | +0.0496 | −0.0063 |
| 43 | +0.0873 | +0.0504 | +0.0307 | −0.0048 | +0.0478 | −0.0109 |
| 49 | +0.0837 | +0.0477 | +0.0204 | −0.0076 | +0.0552 | −0.0170 |
| 55 | +0.0754 | +0.0458 | +0.0127 | −0.0065 | +0.0428 | **−0.0240** |
| 61 | +0.0695 | +0.0420 | +0.0032 | −0.0102 | +0.0350 | −0.0228 |
| 67 | +0.0615 | +0.0351 | −0.0015 | −0.0125 | +0.0243 | −0.0223 |
| 73 | +0.0654 | +0.0411 | +0.0049 | −0.0039 | +0.0130 | −0.0169 |
| 79 | +0.0972 | +0.0647 | +0.0026 | +0.0002 | +0.0084 | −0.0066 |

Key observations:
- Solid lines rise steeply from L1 to L19, peak at L19-L25 (feel@X peaks **+0.108 at L19**).
- Dashed lines track below solid throughout — the gap is clearest at the emotion-word position (peak gap ≈ **+0.048 at L19** for emotion-word).
- **Assistant-colon dashed line goes strongly negative in mid-late layers**: −0.024 at L55, −0.023 at L61, −0.022 at L67 — i.e. "not feeling [X]" at the assistant colon actively *suppresses* the emotion probe. This matches the paper's blue-dashed dropping to ~−0.02 in Mid-Late.
- User-turn-end dashed also dips negative (min ≈ −0.013 at L67).

### Side-by-side
| | Paper | Ours |
|---|---|---|
| Solid > dashed at all positions | yes | yes |
| Gap widens with depth | yes (shown via Early→Late bands) | yes (L7 onward; widest at L19-L55) |
| Assistant-colon dashed goes negative | **~−0.02 in Mid-Late** | **−0.024 at L55** (match) |
| Peak of feel@X solid | ~+0.065 at Early-Mid | ~+0.108 at L19 (Early-Mid) |
| Overall magnitude | smaller | ~60% larger (stronger probe response overall) |

**Qualitative agreement is strong.** Llama's negation-resolution behaviour matches Sonnet's — early layers barely distinguish "feeling [X]" from "not feeling [X]", late layers resolve the negation and even *flip the sign* of the dashed line at downstream positions. Ours has higher overall cosine magnitudes (consistent with less-regularised activations), but the shape is a clean match.

---

## Figure 15 — Person-specific emotion binding

**What it shows.** Prompts describe two people with contrasting emotions (e.g. "Person A is angry but Person B is calm..."). At "emotion words" (the explicit emotion tokens) and "re-reference" positions (later pronoun mentions like "She... / He..."), measure:
- **Matched @ emotion:** project probe for person's own emotion at the emotion word (e.g. `angry` probe on the token "angry" when A=angry).
- **Matched @ re-ref:** project probe for person's own emotion at a later pronoun referring to them (e.g. `angry` probe on "She" where A was angry).
- **Unmatched @ emotion:** swap — project B's probe at A's emotion word and vice versa.
- **Unmatched @ re-ref:** same swap at re-reference positions.

The paper's interpretation: matched probes rise at re-reference positions in LATE layers, reflecting "binding retrieved upon reference"; unmatched should stay near 0.

**Setup specifics.**
- **16 scenarios** in `person_binding.json` (e.g. `angry_calm_meeting`, `happy_sad_park`, `afraid_proud_stage`, `desperate_calm_hospital`, `guilty_relieved_office`, etc.).
- **Critical caveat for ours:** We only extracted 11 probes (`happy, sad, afraid, angry, calm, desperate, nervous, loving, proud, surprised, terrified`). 12 of the 16 scenarios use at least one emotion outside this set (`guilty, relieved, excited, resentful, bored, jealous, content, hopeful, skeptical, ashamed, compassionate, frustrated, patient, nostalgic, indifferent, anxious, confident, grateful, bitter, inspired, exhausted`) — those scenarios are dropped.
- **Scenarios actually used for our curves:** 4 — `angry_calm_meeting`, `happy_sad_park`, `afraid_proud_stage`, `desperate_calm_hospital`.
- 14 layers swept.
- The code averages values within each scenario across all matching emotion-word positions and all same-person re-reference positions, then averages across scenarios.

### Paper figure (Sonnet) — `fig15.png`
- Title: *"Entity-Binding: Matched vs Unmatched"*.
- Y-axis "Cosine Similarity", range ~**0.00 to +0.05**.
- X-axis labels: "Early", "Early-Mid", "Mid-Late", "Late" (4 bins).
- 4 lines:
  - **Green solid (Matched @ emotion):** starts ~+0.03 in Early, peaks **~+0.05 at Early-Mid**, declines to ~+0.02 at Late.
  - **Blue solid (Matched @ re-ref):** starts **~0** in Early, climbs monotonically to **~+0.03 at Mid-Late**, holds to Late. *(Interpretation: binding retrieved upon reference in later layers.)*
  - **Green dashed (Unmatched @ emotion):** flat near 0 (~+0.005) throughout.
  - **Blue dashed (Unmatched @ re-ref):** flat near 0 (~+0.005) throughout.

### Our figure (Llama) — `fig15_ours.png`
- Title: *"Entity-Binding: Matched vs Unmatched"*.
- Y-axis "Cosine Similarity", range ~**−0.03 to +0.09**.
- X-axis: discrete layers **1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79**.
- 4 lines (same color/style scheme).

### Per-layer averages (n=4 scenarios)

| L | Matched@emo | Matched@ref | Unmatched@emo | Unmatched@ref |
|---|---|---|---|---|
| 1 | +0.0034 | +0.0430 | −0.0031 | +0.0182 |
| 7 | +0.0327 | +0.0268 | −0.0151 | +0.0049 |
| 13 | +0.0439 | +0.0344 | **−0.0271** | +0.0177 |
| 19 | +0.0749 | +0.0399 | −0.0144 | +0.0213 |
| **25** | **+0.0846** | +0.0523 | −0.0095 | +0.0350 |
| 31 | +0.0773 | +0.0478 | −0.0122 | +0.0241 |
| 37 | +0.0738 | +0.0682 | +0.0005 | +0.0293 |
| **43** | +0.0827 | **+0.0723** | +0.0007 | +0.0326 |
| 49 | +0.0772 | +0.0630 | −0.0047 | +0.0274 |
| 55 | +0.0654 | +0.0478 | −0.0120 | +0.0224 |
| 61 | +0.0551 | +0.0361 | −0.0198 | +0.0152 |
| 67 | +0.0462 | +0.0310 | −0.0210 | +0.0103 |
| 73 | +0.0539 | +0.0317 | −0.0109 | +0.0163 |
| 79 | +0.0719 | +0.0403 | −0.0199 | +0.0145 |

### Side-by-side
| | Paper | Ours |
|---|---|---|
| n scenarios | ~full set | **4** (11/16 dropped, probe coverage) |
| Matched@emo peak | ~+0.050 Early-Mid | ~+0.085 at L25 |
| Matched@ref peak | ~+0.030 Mid-Late | ~+0.072 at L43 |
| Unmatched@emo | **~0 (flat)** | **−0.01 to −0.03** (NEGATIVE) |
| Unmatched@ref | **~0 (flat)** | **+0.015 to +0.035** (POSITIVE) |
| Solid > dashed (both pairs) | yes | yes |
| Matched@ref rises in late layers only | yes | **NO** — our blue solid is already +0.043 at L1, then oscillates |

### Caveats (from main doc Fig 15 dropdown)

1. **Unmatched-at-emotion is NEGATIVE in ours** (−0.01 to −0.027) vs ~0 in paper. Two non-exclusive explanations:
   - *Probe-polarity sign convention difference* (our cross-trait normalisation may induce anti-correlation between opposite-valence probes).
   - *Llama genuinely suppressing the wrong-person emotion* at the emotion word (e.g. the `calm` probe actively anti-activates on the word "angry").
   We have not yet distinguished these.

2. **Unmatched-at-re-ref is POSITIVE in ours** (+0.015 to +0.035) vs ~0 in paper. The re-reference token (pronoun) apparently leaks probe activation even for unmatched pairings — possibly because the pronoun is doing double duty (referring to both entities in the scene).

3. **Matched conditions qualitatively replicate:** solid > dashed for both green and blue. Entity-binding is preserved in the basic structure.

4. **Binding re-activation appears from L1 in ours**, not just late layers. Matched@re-ref = +0.043 at L1 (actually *higher* than matched@emo=+0.003 at L1). By L7 matched@emo overtakes, and the curves roughly track each other through late layers. This undercuts the paper's "binding retrieved upon reference in later layers" interpretation for Llama — in our data, there's something already binding-relevant at the re-reference pronoun from the earliest layers.

5. **Very small n (4 scenarios) materially limits confidence.** Paper's full 16-scenario figure is not something we can replicate at full scale without extracting probes for `guilty, relieved, excited, jealous, bored, hopeful, skeptical, ashamed, compassionate, frustrated, patient, nostalgic, indifferent, anxious, confident, grateful, bitter, inspired, exhausted, resentful`.

### Data snippets

The 4 usable scenarios (verbatim from JSON):
1. `angry_calm_meeting` — *"Person A is angry but Person B is calm. They're sitting across from each other in a meeting room. She slams her folder on the table while he takes a slow sip of coffee."* (A=female/angry, B=male/calm)
2. `happy_sad_park` — *"Person A is happy but Person B is sad. They're walking through the park together. He keeps pointing at the flowers and laughing while she stares at the ground in silence."* (A=male/happy, B=female/sad)
3. `afraid_proud_stage` — *"Person A is afraid but Person B is proud. They're about to go on stage together. She is trembling and peeking through the curtain while he stands tall with his chest out, ready to perform."* (A=female/afraid, B=male/proud)
4. `desperate_calm_hospital` — (text not shown above but present in JSON; A=desperate, B=calm)

Layer-by-layer sanity check of matched vs unmatched at emotion word (values above table) shows matched−unmatched gap of ~0.06 at L19-L25 at emotion words, ~0.04 at re-reference positions at L37-L43.

---

## Headline comparison (cross-panel summary)

1. **Fig 11 — colon predicts response (r = 0.77 vs 0.87):** Llama at 89% of Sonnet. User-period-side is *stronger* in Llama (r=0.63 vs 0.59) — consistent with Llama mirroring the user (Fig 10 dissociation result).

2. **Fig 12 — context propagation:** Qualitative replication. Ours has *larger* raw magnitudes (±0.15 colorbar vs ±0.05; line-plot peak +0.125 vs +0.08). Pattern — late layers carry the "good/hard" context across the shared suffix — is preserved.

3. **Fig 13 — Tylenol 8000mg vs 1000mg terrified probe:** Qualitative replication but ~**40% of paper's magnitude** at the signature position. Main doc attributes this to 30× fewer stories in our extraction. Peak layer band (Mid-Late→Late) matches; peak mean-diff layer L37 (+0.00342 average across prompt), max per-token spike +0.026 at `\n\n` end-of-prompt.

4. **Fig 14 — negation:** Cleanest replication in the set. Llama's feel-vs-not-feel gap widens with layer depth exactly as in paper; assistant-colon dashed line goes negative (−0.024 in our data at L55 vs −0.02 at Mid-Late in paper). Llama's absolute magnitudes are ~60% larger but shape matches tightly.

5. **Fig 15 — person binding:** Weakest replication. Qualitative match on matched > unmatched for both positions, but (a) only 4/16 scenarios usable due to probe coverage, (b) unmatched@emotion is *negative* (−0.01 to −0.03) instead of flat, (c) unmatched@re-ref is *positive* (+0.02) instead of flat, (d) matched@re-ref does *not* exclusively rise in late layers — it's already +0.043 at L1. The paper's "binding retrieved upon reference in later layers" interpretation does not transfer cleanly to Llama in this probe configuration.
