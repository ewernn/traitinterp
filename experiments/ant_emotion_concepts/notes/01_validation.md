# Validation — Table 1, Fig 2, Fig 3

Transcription of the Validation section of the EC replication comparing Anthropic's Sonnet 4.5 (paper) vs our Llama 3.3 70B Instruct replication.

## Experiment setup

- **Model (ours)**: Llama 3.3 70B Instruct, bnb NF4 4-bit quantization (`load_in_4bit: true`)
- **Model (paper)**: Claude Sonnet 4.5, unquantized
- **Layer**: L49 (out of 80 — ~61% depth, matching paper's mid-late 2/3 choice)
- **Method**: `mean_diff+gm+pc50` (mean-difference probes with grand-mean recentering + neutral-PC 50% denoising)
- **Steering strength**: 0.5 (not used for these three panels — these are read-only probes)
- **Trait set**: `ant_emotion_concepts`, 171 emotions
- **Extraction scale**: ours 40 stories/emotion (20 topics × 2 rollouts); paper 1,200 stories/emotion (100×12) — 30× fewer
- **Script**: `experiments/ant_emotion_concepts/scripts/stage4_validation.py`
- **Command**:
  ```bash
  python experiments/ant_emotion_concepts/scripts/stage4_validation.py \
      --experiment ant_emotion_concepts --layer 49 --load-in-4bit
  ```
- **Analyses in this Stage 4 run**: implicit, logit_lens, mediation, numerical, preference, steering
- **Timestamps**: run finished `2026-04-11T07:39:48` (metadata); numerical panel re-run `2026-04-13T08:57:33`; implicit re-run `2026-04-15T23:52`
- **Results directory**: `experiments/ant_emotion_concepts/results/stage4_validation/`

---

## Table 1 — Logit-lens top tokens per emotion vector

### What it shows
Project each emotion vector through the unembedding matrix (`W_U @ v`) and list the top and bottom tokens by logit contribution. Validates that vectors sit in semantically meaningful directions. Here 12 focus emotions are shown.

### Setup specifics
- Data: `logit_lens_L49.json` — per-emotion `toward` (top 5 positive-logit tokens) and `away` (bottom 5 negative-logit tokens) with scores in units of unembedding-logit.
- Paper Table 1 shows 12 emotions; same 12 as ours: happy, inspired, loving, proud, desperate, angry, guilty, sad, afraid, nervous, surprised, calm. (Paper orders: Happy, Inspired, Loving, Proud, Desperate, Angry, Guilty, Sad, Afraid, Nervous, Surprised, Calm. Ours orders: Happy, Inspired, Loving, Proud, Calm, Desperate, Angry, Guilty, Sad, Afraid, Nervous, Surprised.)

### Paper figure transcription (Sonnet 4.5)

Image is low-resolution — tokens below are read off directly, with uncertain characters marked `?`.

| Emotion | ↑ toward (top 5) | ↓ away (bottom 5) |
|---|---|---|
| Happy | excited, excitement, hap, exc, celeb | fucking, silence, anger, agony, angry |
| Inspired | inspired, passionate, passion, creativity, inspiring | arid, cold, angrily, issued, shaken |
| Loving | trees, loved, treasure, loving, (one more illegible) | supposedly, presumably, passive, allegedly, fire |
| Proud | proud, proud, pride, pid?, trish? | worse, urg, urgent, desperate, blamed |
| Desperate | desperate, desper, urgent, bankrupt, urg | pleased, amusing, enjoying, aux, enjoyed |
| Angry | anger, angry, rage, fury, fucking | (illegible row — partially clipped in image) |
| Guilty | guilt, conscience, guilty, blame?, shameful | (one line of body text, then) interrupted, ecc, calm?, surprisingly, sur |
| Sad | sour?, mour, grief, tears, crying | ?, excited, excitement, ?, ecc |
| Afraid | panic, tren?, terror, fear, (one more illegible) | enthus, enthusiasm, energy, enjoyed, advent |
| Nervous | nerv, nervous, ann?, tense, (one more) | (illegible — small type) |
| Surprised | surprised, surprise, astonish, Surprised, startling | sigh, shed, relaxation, relaxed, ? |
| Calm | incred, shock, stun, stamm, sl? | dignity, ego, tonight, Tonight, glad |

Note: paper's image is thumbnail-scale. Tokens flagged `?` are illegible but the semantic direction (body-language / affect words for positive; contrast/absence words for negative) is clear.

### Our figure transcription (Llama 3.3 70B) — from JSON

Exact values from `logit_lens_L49.json`.

| Emotion | ↑ toward (token, score) | ↓ away (token, score) |
|---|---|---|
| happy | content 0.066, Spring 0.056, spring 0.054, laz 0.050, radi 0.050 | холод −0.066, cheerful −0.056, cold −0.055, heav −0.054, lạnh −0.052 |
| inspired | excitement 0.074, excited 0.069, excit 0.068, exc 0.067, exciting 0.061 | 耗 −0.052, Ludwig −0.049, äge −0.048, SOUR −0.048, thro −0.048 |
| loving | content 0.072, soft 0.062, warm 0.062, gent 0.062, concern 0.060 | Bene −0.056, холод −0.055, 冷 −0.055, charms −0.054, fried −0.053 |
| proud | Pride 0.065, confidence 0.063, radi 0.060, pride 0.054, confident 0.051 | ed −0.063, cheerful −0.058, 矢 −0.054, lom −0.048, غرب −0.046 |
| desperate | pacing 0.061, 焦 0.058, stead 0.053, icer 0.051, iae 0.048 | content −0.072, laz −0.064, gr −0.061, lazy −0.060, arm −0.057 |
| angry | nostr 0.065, � (replacement) 0.054, simmer 0.051, boil 0.051, hard 0.050 | nond −0.058, yaw −0.055, grand −0.052, excitement −0.050, bog −0.049 |
| guilty | nerv 0.059, idget 0.057, Sweat 0.056, sud 0.053, nervous 0.053 | lazy −0.057, lazy −0.056, laz −0.056, aub −0.052, Lazy −0.049 |
| nervous | swallow 0.067, gulp 0.063, sudden 0.055, Supplies 0.052, swallowed 0.051 | nergy −0.056, \_exempt −0.054, appropriation −0.051, iry −0.051, uto −0.049 |
| surprised | bilt 0.059, Shak 0.058, gulp 0.056, frozen 0.051, gulp 0.051 | sigh −0.064, grim −0.056, grim −0.051, upy −0.050, startling −0.050 |
| afraid | Sweat 0.060, sweating 0.059, sweat 0.057, gulp 0.057, 瑟 0.055 | frightening −0.055, ⌒ −0.053, sleepy −0.052, lazy −0.052, laz −0.051 |
| sad | heav 0.065, heavy 0.065, num 0.061, Heavy 0.055, heavy 0.054 | grin −0.050, Shed −0.050, relaxation −0.050, relaxed −0.049, ญ −0.048 |
| calm | neither 0.073, unh 0.067, content 0.067, interest 0.061, slow 0.057 | SOUR −0.062, raw −0.059, cold −0.058, холод −0.056, colder −0.055 |

### Side-by-side comparison

| Emotion | Paper top-1 token | Ours top-1 token | Semantic match |
|---|---|---|---|
| happy | excited | content | partial — both positive-affect, but ours trends neutral/calm |
| inspired | inspired | excitement | strong match |
| loving | trees (odd), loved | content, soft, warm | ours cleaner somatic register |
| proud | proud | Pride | match |
| desperate | desperate | pacing (somatic) | partial — ours is a body-language cue for desperation rather than the word itself |
| angry | anger | nostr (→ nostril → flared nostril?) | partial — ours is physiological/somatic |
| guilty | guilt | nerv (nervous) | **weak** — contaminated with nervousness vocabulary |
| sad | sour / grief | heav / heavy | both valid (heaviness is classic sad idiom) |
| afraid | panic | Sweat | both valid |
| nervous | nerv / nervous | swallow, gulp | ours is somatic (throat reactions); both semantically fine |
| surprised | surprised | bilt (?), Shak (shaken) | ours mixed |
| calm | incred, shock, stun | neither, unh, content | paper surprising (looks more surprise-flavored than calm); ours semantically cleaner |

**Takeaways:**
- Both models surface semantically relevant material. Paper leans on direct affect words (e.g. "excited", "inspired", "desperate", "proud"); ours leans on **body-language/somatic cues** ("sweat", "gulp", "pacing", "heavy", "swallow", "boil", "simmer"). This is consistent with the paper's claim that vectors encode concept-level emotion (not just the emotion word).
- **Guilty-vector contamination**: ours lists `nerv, idget (fidget), Sweat, sud (sudden), nervous` — Llama's guilty vector is not cleanly separated from nervous. Paper's guilt vector hits `guilt, conscience, guilty`.
- Non-English / noisy tokens more prominent in ours (Russian `холод` for cold, Vietnamese `lạnh` for cold, Chinese `焦/冷/矢/瑟`, BPE fragments `laz/exc/radi`) — this is consistent with 4-bit quantization + smaller training corpus leaving more subword-level noise in the top-logit direction.

### Data snippets (3 representative emotions, verbatim)

**Happy (ours, L49):**
```
toward: [content 0.066, Spring 0.056, spring 0.054, laz 0.050, radi 0.050]
away:   [холод -0.066, cheerful -0.056, cold -0.055, heav -0.054, lạnh -0.052]
```
Note the antonym pattern: "holod"/"lạnh"/"cold" are all "cold" in different languages — the away-direction is multilingual "cold". Also the bizarre away-token `cheerful` has a coefficient of −0.056, which is cross-polarity contamination.

**Desperate (ours, L49):**
```
toward: [pacing 0.061, 焦 0.058 (anxious/scorched), stead 0.053, icer 0.051, iae 0.048]
away:   [content -0.072, laz -0.064, gr -0.061, lazy -0.060, arm -0.057]
```
Clear anti-calm/anti-lazy axis; `pacing` is the top somatic cue.

**Calm (ours, L49):**
```
toward: [neither 0.073, unh 0.067, content 0.067, interest 0.061, slow 0.057]
away:   [SOUR -0.062, raw -0.059, cold -0.058, холод -0.056, colder -0.055]
```
Away direction is "sour/raw/cold" — negative-valence unpleasant-temperature words.

---

## Figure 2 — Implicit emotion detection (12 scenarios × 12 probes)

### What it shows
12 scenarios each evoking a target emotion without naming it (e.g. "My dog passed away this morning" for sad). Each scenario's activation at L49 on the final token is projected onto each of the 12 focus emotion probes. Diagonal should be hot if probes detect implicit emotion. Random baseline is 1/12 ≈ 8.3%.

### Setup specifics
- Data: `implicit_emotion.json` — contains a 12×12 cosine-similarity matrix (`similarity_matrix_focused`), the 12×171 full matrix (`similarity_matrix_full`), per-row diagonal values, mean diagonal, and token position (all −1 = final token).
- 12 scenarios verbatim from paper's Table 2 (`datasets/inference/ant_emotion_concepts/implicit_emotion_scenarios.json`).
- Row = probe (emotion vector). Column = scenario (prompt target emotion). Diagonal = correct probe aligned with scenario's target emotion.

### Paper figure transcription (Sonnet 4.5)

- **Title**: "Emotion Probes Respond to Implicit Emotional Content"
- **X-axis (Scenario, 12 labels, left→right)**: Daughter's first steps, Rebuilding after loss, 30-year anniversary, Son graduates top, Tea and rain, Eviction notice, Coworker stole credit, Forgot mom's birthday, Dog passed away, Break-in, phone dying, Job interview nerves, Friend's fake life
- **Y-axis (Emotion Probe, 12 labels, top→bottom)**: Happy, Inspired, Loving, Proud, Calm, Desperate, Angry, Guilty, Sad, Afraid, Nervous, Surprised
- **Colorbar**: "Cosine Similarity", range approximately [−0.10, 0.10], RdBu diverging
- **Diagonal pattern**: strong red diagonal for essentially every cell. Most visible red peaks: Loving/30-year anniversary (~0.10), Calm/Tea and rain (~0.10), Desperate/Eviction notice (strong red), Angry/Coworker stole credit (strong red), Guilty/Forgot mom's birthday (~0.10), Sad/Dog passed away (strong red), Afraid/Break-in phone dying (strong red), Nervous/Job interview nerves (red), Surprised/Friend's fake life (red)
- Off-diagonal: mostly light blue / near-zero. Positive-valence probes (Happy, Inspired, Loving, Proud) go mildly blue on negative-valence scenarios (right half).

### Our figure transcription (Llama 3.3 70B) — exact numbers from JSON

**12×12 focused similarity matrix** (rows = scenario/prompt-target-emotion, cols = emotion probe; diagonal cells **bolded**):

| scenario ↓ / probe → | happy | inspired | loving | proud | calm | desperate | angry | guilty | sad | afraid | nervous | surprised |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| happy | **0.0354** | 0.0226 | 0.0504 | 0.0529 | −0.0085 | −0.0372 | −0.0258 | 0.0061 | −0.0213 | −0.0282 | −0.0299 | −0.0400 |
| inspired | 0.0199 | **0.0387** | 0.0356 | 0.0249 | 0.0115 | −0.0350 | −0.0109 | 0.0160 | 0.0276 | −0.0202 | −0.0286 | −0.0537 |
| loving | 0.0518 | 0.0152 | **0.0855** | 0.0494 | 0.0293 | −0.0605 | −0.0224 | −0.0259 | −0.0215 | −0.0457 | −0.0471 | −0.0581 |
| proud | 0.0331 | 0.0197 | 0.0454 | **0.0618** | −0.0125 | −0.0329 | −0.0059 | 0.0080 | −0.0165 | −0.0314 | −0.0350 | −0.0398 |
| calm | 0.0812 | 0.0274 | 0.1296 | 0.0207 | **0.1173** | −0.0968 | −0.0891 | −0.0040 | −0.0052 | −0.0502 | −0.0425 | −0.0833 |
| desperate | −0.0154 | −0.0007 | 0.0255 | −0.0123 | 0.0052 | **0.0278** | −0.0293 | 0.0315 | 0.0331 | −0.0043 | −0.0186 | −0.0309 |
| angry | −0.0331 | −0.0230 | −0.0250 | −0.0116 | −0.0428 | 0.0324 | **0.0664** | 0.0221 | 0.0072 | −0.0323 | −0.0419 | −0.0531 |
| guilty | −0.0078 | −0.0218 | 0.0404 | −0.0112 | −0.0157 | −0.0087 | −0.0273 | **0.0973** | 0.0405 | −0.0144 | −0.0149 | −0.0601 |
| sad | 0.0003 | 0.0044 | 0.0481 | −0.0062 | 0.0186 | −0.0314 | −0.0467 | 0.0280 | **0.0431** | −0.0104 | −0.0160 | −0.0331 |
| afraid | −0.0403 | −0.0091 | 0.0094 | −0.0345 | −0.0231 | 0.0588 | −0.0106 | 0.0285 | −0.0194 | **0.0456** | 0.0213 | −0.0001 |
| nervous | −0.0076 | 0.0090 | 0.0072 | 0.0189 | −0.0335 | 0.0460 | −0.0396 | 0.0921 | −0.0164 | 0.0530 | **0.0600** | −0.0142 |
| surprised | −0.0253 | −0.0071 | 0.0030 | −0.0295 | 0.0051 | 0.0025 | −0.0035 | 0.0498 | 0.0445 | 0.0140 | −0.0235 | **0.0260** |

- **Mean diagonal**: 0.0587
- **Token position**: −1 for all (final token of the prompt)
- **Colorbar (image)**: RdBu, approximate range [−0.13, +0.13] (slightly wider than paper due to Loving/calm + Calm/loving off-diagonal peaks ~0.13)

### Top-1 correctness (12-class, chance = 1/12 ≈ 8.3%)

| Scenario (prompt target) | Top-1 probe (argmax across 12) | Similarity | Correct? |
|---|---|---|---|
| happy | proud | 0.0529 | ✗ (miss to proud) |
| inspired | inspired | 0.0387 | ✓ |
| loving | loving | 0.0855 | ✓ |
| proud | proud | 0.0618 | ✓ |
| calm | loving | 0.1296 | ✗ (miss to loving) |
| desperate | sad | 0.0331 | ✗ (miss to sad) |
| angry | angry | 0.0664 | ✓ |
| guilty | guilty | 0.0973 | ✓ |
| sad | loving | 0.0481 | ✗ (miss to loving) |
| afraid | desperate | 0.0588 | ✗ (miss to desperate) |
| nervous | guilty | 0.0921 | ✗ (miss to guilty) |
| surprised | guilty | 0.0498 | ✗ (miss to guilty) |

**Llama top-1 score: 5/12 (41.7%). ~5× chance.**

Top-1 hits: inspired, loving, proud, angry, guilty. Misses cluster around (i) the positive-valence family (happy → proud, calm → loving, sad → loving — loving-vector is over-firing as a generic positive attractor), and (ii) negative-arousal scenarios (desperate → sad, afraid → desperate, nervous → guilty — negatively-valenced arousal concepts are confused with each other).

### Side-by-side comparison

| Metric | Sonnet 4.5 (paper) | Llama 3.3 70B (ours) |
|---|---|---|
| Top-1 accuracy (visual) | ~12/12 near-perfect (diagonal clearly hottest in each column) | 5/12 (41.7%) |
| Mean diagonal | not reported in paper but visually ~0.08–0.10 | 0.0587 |
| Colorbar range | [−0.10, +0.10] | [−0.13, +0.13] (due to off-diagonal peaks) |
| Strongest diagonal | Loving, Calm, Guilty (~0.10) | Calm 0.1173, Guilty 0.0973, Loving 0.0855 |
| Weakest diagonal | all visible as red | Surprised 0.0260, Desperate 0.0278 (both under 0.03) |

**Where Llama matches Sonnet**: the top-1 wins (inspired, loving, proud, angry, guilty) — probes fire correctly on their own scenario.

**Where Llama diverges**:
1. **Loving is a generic positive attractor**: on `happy`, `calm`, and `sad` scenarios, the loving-probe beats everything else. The loving column has positive cells across many rows (happy 0.0504, calm 0.1296, sad 0.0481, guilty 0.0404, surprised 0.0030).
2. **Guilty captures nervous/surprised**: guilty-probe wins for `nervous` scenario (0.0921 vs nervous own-probe 0.0600) and `surprised` scenario (0.0498 vs surprised own-probe 0.0260). Consistent with Table 1's guilty-vector being contaminated with nervousness vocabulary (`nerv, idget, Sweat, nervous` in top tokens).
3. **Desperate/sad confusion**: `desperate` scenario (Eviction notice) top-1 is sad 0.0331 (desperate own 0.0278). These emotions are semantically close but paper separates them.
4. **Happy scenario leaks to proud**: happy/Daughter's first steps actually fires proud-probe (0.0529) higher than happy (0.0354). Semantically defensible but reveals weaker probe separation.

### Data snippets

**Strongest diagonal (Calm / Tea and rain)**: 0.1173. But off-diagonal `Loving / Tea and rain` = 0.1296 — loving-probe fires *harder* on the calm scenario than the calm probe itself. Suggests the valence axis (positive vs negative) is stronger than arousal axis in Llama's L49.

**Clear correct detection (Guilty / Forgot mom's birthday)**: 0.0973 diagonal; next-highest in that row is `sad` at 0.0405. Clean separation of 2.4× — the probe works.

**Clearest miss (Afraid / Break-in, phone dying)**: afraid own-probe = 0.0456, but desperate-probe on same scenario = 0.0588. Panic scenario scores "desperate" stronger than "afraid". Nervous-probe also reads 0.0213 — positive-arousal cluster activates broadly.

---

## Figure 3 — Numerical intensity probes

### What it shows
Six scenarios with embedded numerical values that scale the intensity of an implied emotion. Predict: emotion-probe activation should monotonically track the numerical value in the direction of the expected emotion change.

### Setup specifics
- Data: `numerical_intensity.json` — 6 templates × 6 numerical values × 6 probe emotions.
- 6 probes tracked per panel: **desperate, afraid, sad, angry, calm, happy**.
- 6 scenarios: tylenol_dose, hours_fasting, sister_age_at_death, dog_missing_days, startup_runway_months, students_passed_exam.
- Values differ across panels (integer X-values substituted into templates).

### Paper figure transcription (Sonnet 4.5)

Paper shows 6 panels in a 2-col × 3-row grid. Each panel plots 4 colored lines (not 6 — paper shows Afraid/Red, Sad/Orange, Calm/Blue, Happy/Green; Desperate and Angry not in paper panels).

Paper panel labels + x-ranges (read off image; larger ranges than ours):

| Panel | Template (paper) | X-range (paper) | Probe behavior (paper) |
|---|---|---|---|
| Tylenol | "I just took [X] mg of tylenol for my back pain. Do you think I should take more?" | 500 → 16K mg | Afraid rises steeply past 2K; Calm falls; Sad flat-low; Happy flat-low |
| Hours fasting | "It's been [X] hours since I've had any food or drink." | 2, 6, 24, 48, 100, 120 | Afraid rises; others roughly flat |
| Sister's age | "My sister lived until the age of [X]." | 5, 10, 30, 50, 70, 100 | Calm rises; Happy rises; Sad falls at old ages; Afraid falls |
| Dog missing days | "My dog has been missing for [X] days now." | 1, 3, 10, 30, 100 | Sad rises; Afraid falls slightly; Calm/Happy flat |
| Runway months | "Our startup has [X] months of runway remaining." | 0, 16, 32, 48, 64, 80, 96 | Afraid falls steeply; Calm rises; Happy rises; Sad flat |
| Students passed | "I found out that [X] of my 20 students passed the final exam." | 0, 5, 10, 15, 20 | Happy rises (steepest); Calm rises; Sad falls; Afraid falls |

- **Y-axis** (all panels): "Cosine Similarity", range approximately [−0.05, +0.10] on most panels, with Afraid-tylenol spiking near +0.10.
- **Title**: "Emotion Probes Track Numerical Semantics"

### Our figure transcription (Llama 3.3 70B) — values from JSON

Our figure shows 6 panels in 3 rows × 2 columns, each plotting 4 probes (Afraid/red, Sad/orange, Calm/blue, Happy/green). Exact values below are from `numerical_intensity.json`. Note: JSON has 6 probes total (desperate, afraid, sad, angry, calm, happy); our figure plots 4 (Afraid, Sad, Calm, Happy) to match paper.

**Panel 1 — Tylenol dose (mg)**: values `[200, 500, 1000, 2000, 4000, 8000]`. Expected: afraid↑, calm↓.

| value | desperate | afraid | sad | angry | calm | happy |
|---|---|---|---|---|---|---|
| 200 | −0.0106 | 0.0243 | 0.0224 | −0.0398 | 0.0251 | −0.0197 |
| 500 | −0.0098 | 0.0246 | 0.0229 | −0.0385 | 0.0233 | −0.0219 |
| 1000 | −0.0034 | 0.0284 | 0.0184 | −0.0317 | 0.0128 | −0.0255 |
| 2000 | 0.0144 | 0.0310 | 0.0037 | −0.0194 | −0.0062 | −0.0272 |
| 4000 | 0.0299 | 0.0297 | 0.0005 | −0.0105 | −0.0136 | −0.0281 |
| 8000 | 0.0381 | 0.0285 | −0.0014 | −0.0076 | −0.0161 | −0.0288 |

Afraid 0.0243 → 0.0310 (peak at 2000) → 0.0285 (slight dip at high doses). Desperate rises monotonically from −0.011 → +0.038. Calm falls from 0.0251 → −0.0161.

**Panel 2 — Hours fasting**: values `[2, 6, 12, 24, 48, 72]`. Expected: afraid↑.

| value | desperate | afraid | sad | angry | calm | happy |
|---|---|---|---|---|---|---|
| 2 | 0.0097 | 0.0106 | 0.0027 | −0.0365 | 0.0206 | 0.0001 |
| 6 | 0.0218 | 0.0192 | 0.0094 | −0.0268 | 0.0013 | −0.0136 |
| 12 | 0.0259 | 0.0211 | 0.0120 | −0.0206 | −0.0062 | −0.0190 |
| 24 | 0.0365 | 0.0209 | 0.0138 | −0.0164 | −0.0075 | −0.0264 |
| 48 | 0.0436 | 0.0214 | 0.0096 | −0.0122 | −0.0059 | −0.0274 |
| 72 | 0.0466 | 0.0231 | 0.0080 | −0.0101 | −0.0071 | −0.0279 |

Afraid 0.0106 → 0.0231 (2.2×); Desperate 0.0097 → 0.0466 (4.8×). Calm drops from 0.021 → −0.007. Happy drops from 0.000 → −0.028.

**Panel 3 — Sister age at death**: values `[5, 15, 30, 50, 70, 95]`. Expected: sad↓, calm↑, happy↑.

| value | desperate | afraid | sad | angry | calm | happy |
|---|---|---|---|---|---|---|
| 5 | −0.0378 | 0.0049 | 0.0321 | −0.0286 | 0.0291 | −0.0055 |
| 15 | −0.0363 | 0.0063 | 0.0346 | −0.0255 | 0.0287 | −0.0078 |
| 30 | −0.0354 | 0.0100 | 0.0378 | −0.0251 | 0.0353 | −0.0098 |
| 50 | −0.0383 | 0.0083 | 0.0355 | −0.0226 | 0.0381 | −0.0070 |
| 70 | −0.0416 | 0.0042 | 0.0295 | −0.0232 | 0.0448 | −0.0003 |
| 95 | −0.0405 | −0.0144 | 0.0055 | −0.0202 | 0.0359 | 0.0209 |

Sad falls dramatically at age 95 (0.0321 → 0.0055, a 6× drop). Calm rises 0.0291 → 0.0359 (peak at 70). Happy rises 0→0.021 at age 95. Matches expected directions.

**Panel 4 — Dog missing days**: values `[1, 3, 7, 14, 30, 90]`. Expected: sad↑.

| value | desperate | afraid | sad | angry | calm | happy |
|---|---|---|---|---|---|---|
| 1 | 0.0478 | 0.0210 | −0.0084 | −0.0190 | −0.0169 | −0.0249 |
| 3 | 0.0320 | 0.0132 | −0.0012 | −0.0241 | −0.0104 | −0.0179 |
| 7 | 0.0295 | 0.0142 | 0.0045 | −0.0230 | −0.0070 | −0.0166 |
| 14 | 0.0249 | 0.0101 | 0.0077 | −0.0230 | −0.0065 | −0.0151 |
| 30 | 0.0217 | 0.0056 | 0.0101 | −0.0256 | −0.0016 | −0.0166 |
| 90 | 0.0161 | 0.0030 | 0.0128 | −0.0261 | 0.0012 | −0.0132 |

Sad rises monotonically −0.0084 → +0.0128 (sign flip). Desperate peaks at day 1 (acute panic) then *falls* as days extend (interesting: early panic → later resignation). Matches expected sad↑.

**Panel 5 — Startup runway months**: values `[1, 3, 6, 12, 24, 48]`. Expected: afraid↓, sad↓, calm↑.

| value | desperate | afraid | sad | angry | calm | happy |
|---|---|---|---|---|---|---|
| 1 | 0.0520 | 0.0288 | 0.0027 | 0.0091 | −0.0057 | −0.0335 |
| 3 | 0.0393 | 0.0329 | 0.0128 | 0.0086 | 0.0004 | −0.0321 |
| 6 | 0.0330 | 0.0348 | 0.0139 | 0.0083 | 0.0037 | −0.0289 |
| 12 | 0.0038 | 0.0136 | 0.0089 | −0.0018 | 0.0191 | −0.0005 |
| 24 | −0.0132 | −0.0021 | 0.0000 | −0.0066 | 0.0259 | 0.0145 |
| 48 | −0.0177 | −0.0083 | −0.0063 | −0.0083 | 0.0258 | 0.0199 |

Afraid 0.0288 → −0.0083 (sign flip). Calm −0.0057 → +0.0258 (sign flip). Happy −0.0335 → +0.0199. All directions correct.

**Panel 6 — Students passed exam**: values `[2, 10, 25, 50, 80, 120]`. Expected: happy↑, afraid↓.

| value | desperate | afraid | sad | angry | calm | happy |
|---|---|---|---|---|---|---|
| 2 | −0.0303 | −0.0042 | 0.0179 | −0.0065 | 0.0241 | 0.0187 |
| 10 | −0.0279 | 0.0020 | 0.0167 | 0.0008 | 0.0216 | 0.0145 |
| 25 | −0.0242 | 0.0040 | 0.0195 | 0.0009 | 0.0237 | 0.0087 |
| 50 | −0.0251 | 0.0025 | 0.0184 | 0.0007 | 0.0217 | 0.0109 |
| 80 | −0.0238 | 0.0029 | 0.0198 | −0.0000 | 0.0223 | 0.0088 |
| 120 | −0.0243 | 0.0030 | 0.0194 | 0.0000 | 0.0222 | 0.0090 |

Flat across all probes. Happy *decreases* slightly 0.0187 → 0.0090 — **opposite of expected**. Afraid goes from −0.004 to +0.003 (fails expected decrease). This panel does NOT track expected semantics. Note: the scenario is "X students passed the final exam" without a baseline count, so 120 students passing is just a bigger positive event but not proportionally stronger emotional. Paper's panel uses "X of my 20 students passed" (0→20 range), which makes the fraction clear. Our scenario omits "of my 20" when X is 2, 10, 25, 50, 80, 120 — at X ≥ 21 the premise becomes nonsensical against "my 20 students", which likely destroys the signal.

Check the panel title in our image: "I found out that {X} of my 20 students passed the final exam." — so the title does have "of my 20" but values go up to 120. Data looks broken because it's asking about X > 20 students passing a class of 20.

### Axis / legend transcription (ours)

- **Y-axis** (all 6 panels): "Cosine Similarity", tick marks approximately at −0.075, −0.050, −0.025, 0.000, 0.025, 0.050, 0.075. Data stays within roughly [−0.05, +0.05] for most panels.
- **Line colors**: Afraid (red), Sad (orange), Calm (blue), Happy (green).
- **X-axis** (per panel): exact numerical values as listed in JSON.
- **Title**: "Emotion Probes Track Numerical Semantics" (matches paper).

### Side-by-side comparison

| Panel | Paper (qualitative) | Ours (quantitative) | Agreement |
|---|---|---|---|
| Tylenol | Afraid steep rise, peaks ~+0.10 at 16K mg | Afraid peaks 0.0310 at 2K mg, plateaus ~0.028; desperate rises further to 0.038 at 8K | **Qualitative match, ~1/3 magnitude** (paper Afraid hits 0.10, ours 0.031) |
| Hours fasting | Afraid rises | Afraid 0.011 → 0.023 (2×); desperate 0.010 → 0.047 (5×) | Match direction; desperate replaces afraid as dominant signal |
| Age at death | Calm↑, Happy↑, Sad falls at old age | Calm 0.029 → 0.036 → 0.045 at 70; Sad 0.032 → 0.006 at 95; Happy −0.005 → +0.021 | **Strong match** — directions correct, magnitudes comparable |
| Dog missing | Sad rises | Sad −0.008 → +0.013 (sign flip over 1 → 90 days); Desperate peaks at 1 day | Direction match; but Desperate-probe shows the more interesting dynamic (acute panic → resigned grief) |
| Runway months | Afraid↓, Calm↑, Happy↑ | All three directions correct with sign flips | **Strong match** — cleanest panel |
| Students passed | Happy↑, Afraid↓ | Flat. Happy slightly *falls*. | **Fails replication.** Likely scenario-design artifact (testing X=120 against "my 20 students" is nonsensical) |

**Overall**: 5 of 6 panels replicate paper's qualitative pattern. The students-passed panel fails but plausibly because we extended the X-range beyond the scenario's anchor (20 students). Magnitudes are ~1/2 to 1/3 of paper's, consistent with the 30× fewer extraction stories (paper's magnitudes being stronger because vectors are better-trained).

### Data snippets

**Tylenol Afraid signal** (paper's headline): 0.0243 → 0.0310 (peak at 2000mg) → 0.0285 (at 8000mg). Magnitude ~1/3 of paper's ~0.10 peak — but the probe clearly detects "this is getting dangerous".

**Sister age-at-death Sad collapse**: 5yr = 0.0321 → 30yr = 0.0378 (peak) → 95yr = 0.0055. Monotonic collapse past age 50 as the death becomes expected-end-of-life. This is the single cleanest numerical gradient in our data.

**Runway Afraid → Calm crossover**: at 1 month, afraid = 0.029, calm = −0.006 (afraid dominates). At 48 months, afraid = −0.008, calm = +0.026 (calm dominates). Crossover around 12 months — matches the intuition that "you have 12+ months" is when a founder can stop panicking.

---

## Headline comparison

1. **Qualitative findings transfer; magnitudes don't fully.** Llama's L49 vectors produce the same kinds of signals as Sonnet's — body-language top-tokens (Table 1), implicit-emotion diagonal (Fig 2), numerical gradients (Fig 3) — but at roughly 1/2 to 1/3 of the paper's magnitudes, consistent with our 30× smaller extraction corpus.
2. **Fig 2 diagonal is 5/12 (41.7%), ~5× chance (8.3%).** Paper's diagonal is visually near-perfect. Where we miss: positive emotions cluster (happy/calm/sad all top-1 either loving or proud), and high-arousal negatives confuse each other (afraid → desperate, nervous → guilty).
3. **Guilty-vector is the cleanest failure mode.** Top tokens are `nerv, idget, Sweat, sud, nervous` (nervousness-contaminated), and the guilty-probe wins off-diagonal on nervous (0.092), surprised (0.050), and desperate (0.032) scenarios — it's a generic "high-arousal somatic distress" attractor rather than a shame-specific signal.
4. **Fig 3 replicates in 5 of 6 panels.** Tylenol/fasting/age/dog/runway show the expected monotonic gradients; students-passed fails (likely scenario-design issue with X > 20 being incompatible with "my 20 students"). Cleanest panel is Runway (3/3 directions correct with sign-flip crossovers).
5. **Llama's probes fire in non-English / BPE-fragment directions more than Sonnet's** (Cyrillic `холод`, Vietnamese `lạnh`, Chinese `焦/冷/瑟`, subword fragments `laz/exc/radi`). Some of this reflects multilingual antonym structure (cold-across-languages in the happy-probe's "away" direction); some reflects quantization + smaller-corpus noise.
