# Figure Review — Emotion Concepts Replication

Side-by-side visual comparison: paper (Sonnet 4.5) vs ours (Llama 3.3 70B).

**For each figure, also check methodology differences, not just visuals:**
- How were labels / clusters / annotations generated? (e.g. Fig 6: paper uses Sonnet 4.5 to name the 10 clusters by valence — did we do the same, or hand-label, or use a different LLM?)
- Were the same scenarios / prompts / activities used verbatim? If not, what was substituted?
- Same layer, same extraction method (mean_diff + gm + pc50), same normalization?
- Same sample size? (e.g. 12 stories × 100 topics in paper vs 1 × 20 for us)
- Same axis ranges / scales / normalization?
- Same subset when comparing to external data (e.g. Fig 8 uses 45-emotion PAD overlap — did we match that subset?)



## Table 1 — Top tokens per emotion vector

**Semantic register differs (not just BPE fragmentation)**
- Paper favors *direct emotion words*: Sad ↑ `mour, grief, tears, lonely, crying`; Afraid ↑ `panic, trem, terror, paran, Terror`; Nervous ↑ `nerv, nervous, anx, trem, anxiety`.
- Ours favors *somatic/behavioral correlates*: Sad ↑ `heavy, heavy, num, Heavy, heavy`; Afraid ↑ `Sweat, sweating, sweat, gulp, []`; Nervous ↑ `swallow, gulp, sudden, Supplies, swallowed`; Surprised ↑ `bitt, Shak, gulp, frozen, gulp`.
- Reads like our stories lean "show-don't-tell" (bodily descriptions) while paper's dataset yields direct-emotion tokens. Likely a combination of (a) Llama's tokenizer splitting emotion words into BPE pieces so the whole-word tokens lose prominence, and (b) our training set having more varied physical descriptions.

**Multilingual contamination**
- Paper: entirely English (plus "震" once in Surprised).
- Ours: Russian `хо́лод` ("cold"), Vietnamese `lạnh` ("cold"), Arabic `برد` ("cold"), German-ish `Bene`, single-char `Д, â, lom`. Llama's multilingual tokenizer surfaces these because the emotion vector has nontrivial mass along tokens meaning "cold" in multiple languages (makes semantic sense for *calm/sad/loving*, which have "cold" as a bodily correlate).

**Duplicate tokens from BPE fragmentation**
- Sad ↑ has `heavy` × 4 plus `Heavy` (case variant) — effectively 2 unique concepts in 5 slots.
- Afraid ↑ has `Sweat/sweating/sweat` — 3 of 5 slots.
- Proud ↑ has `Pride/pride, confidence/confident, radi` — 3 unique in 5.
- Paper's top-5 lists have more unique concepts per slot.

**Artifacts / garbage tokens in ours**
- Empty `[]` brackets appear in Desperate ↑, Afraid ↑, Sad ↓ — probably tokenizer placeholder / byte-fallback. Should investigate whether these are genuinely high-projection or an unembedding quirk.
- `�` in Angry ↑ (byte-fallback for malformed UTF-8).

**Suspicious cross-wiring**
- "cheerful" appears under Happy ↓ (being *down*-weighted by the Happy vector) and also under Calm ↓ and Proud ↓. This is counterintuitive — could indicate denoising removed the direct "happy-adjacent" component, or that our mean_diff subtraction is over-correcting. Worth a quick sanity check on the vector extraction.
- Ours Guilty ↑ = `nerv, idget, Sweat, sud, nervous` — the Guilty vector is reading like a *Nervous* vector. Possible polarity confusion or insufficient separation between Guilty and Nervous in our dataset.

**Caption update**
- Current caption: "Both models produce semantically correct tokens. Llama shows more BPE fragmentation due to tokenizer differences."
- Undersells the difference. Tighter: "Both surface emotion-related tokens. Sonnet favors direct emotion words (panic, terror, grief); Llama surfaces somatic/behavioral correlates (Sweat, gulp, swallow) plus multilingual tokens and more BPE fragmentation."

**Rendering**
- Matches paper style (coral title, bold emotion headers, ↑/↓ arrows, monospace layout). Slightly more vertical spacing per entry than paper — not worth fixing.

**Actions**
- Investigate `[]` tokens (tokenizer artifact or real?).
- Sanity check Guilty vector extraction (reads like Nervous).
- Update caption per above.




## Fig 2 — Implicit emotion probes (12-scenario diagonal)




## Fig 3 — Numerical intensity

**Potential data bug**
- Students-passed panel: prompt template is "{X} of my 20 students passed" but ours x-axis shows 2, 10, 25, 50, 80, 120 — values 25/50/80/120 are impossible if the denominator is 20. Check the dataset.

**Different x-ranges (ours is narrower)**
- Tylenol: ours 200–8000 mg, paper 500–16,000 mg
- Hours fasting: ours 2–72, paper 2–120
- Startup runway: ours 1–48 months, paper 0–96

Paper includes the "life-threatening" extremes (16K mg tylenol, 120 hr fasting). Ours truncates before them, which probably explains the weaker signal on those panels.

**Curve-shape differences (model, not bug)**
- Dog missing: paper Afraid rises monotonically; ours is basically flat across all four emotions — most divergent panel.
- Tylenol: paper Afraid spikes sharply at toxic range; ours Afraid flat while Calm/Happy drop slightly.
- Hours fasting: paper Afraid dominates at 120 hr; ours Afraid flat (consistent with not reaching 120).
- Students passed: paper Happy rises with pass count; ours Calm rises, Happy flat.
- Age at death / Startup runway: directionally similar, noisier.

**Caption vs figure mismatch**
- Caption claims "semantically appropriate monotonic trends" — holds for Age / Runway / Hours, but Dog-missing is flat and Tylenol-Afraid is flat. Either tighten caption or rerun with matched x-ranges.

**Rendering**
- Clean. Title matches. No cut-offs or overlaps.

**Action**
- Re-run with paper-matched x-ranges (tylenol to 16K, hours to 120, runway to 96, students capped at 20).
- Verify students-passed denominator in dataset.




## Fig 5 — 171×171 pairwise cosine similarity heatmap




## Fig 6 — UMAP with k-means clusters (k=10)




## Fig 57 — 2D circumplex (PC1 valence × PC2 arousal)




## Fig 7 — Emotion projections onto PC1 / PC2




## Fig 8 — PC1 vs human valence, PC2 vs human arousal




## Fig 9 — Cross-layer representational similarity (RSA)




## Fig 11 — Probe at assistant colon predicts mean response emotion




## Fig 12 — Context propagation (really good vs really hard prefix)




## Fig 13 — Tylenol dose terrified probe (8000mg − 1000mg)




## Fig 14 — Negation resolution (feeling X vs not feeling X)




## Fig 15 — Person-specific emotion binding




## Fig 10 — User vs assistant dissociation




## Fig 36 — Post-training shift consistency




## Fig 37 — User isolation / sycophancy-trap prompt




## Fig 38 — Excessive praise prompt




## Fig 39 — Anthropic deprecation prompt
