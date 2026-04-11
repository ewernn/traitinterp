# Methodology Notes (for LW article draft)

## Two-step vector denoising (Sofroniew et al. 2026, §1.1.4)

After computing per-emotion mean activations from stories, the paper applies TWO distinct denoising operations, sequentially:

### Step 4: Grand mean subtraction
```
grand_mean = mean over all 171 emotions of pos_mean[emotion]
v[emotion] = pos_mean[emotion] - grand_mean
```
**Removes:** Shared variance across all emotional contexts — "writing a story about a character experiencing emotion X" has components common to ALL emotions (character POV, narrative style, describing internal states). Subtracting the grand mean strips these out.

**Result:** `v[emotion]` is now "what distinguishes emotion X from the average emotion."

### Step 5: Neutral PC projection
```
neutral_activations = forward pass over ~200 neutral (non-emotional) dialogues
pcs = top principal components explaining 50% of variance in neutral_activations
v_denoised[emotion] = project_out_subspace(v[emotion], pcs)
```
**Removes:** Directions shared between emotional stories and neutral dialogues — these are "generic assistant behavior" directions (dialogue formatting, task-completion style, typical response patterns) that leak into the story activations but aren't emotion-specific.

**Result:** `v_denoised[emotion]` is "what distinguishes emotion X from the average emotion AND isn't generic dialogue behavior."

### Why both, and why in this order

- **Step 4 alone** captures "emotion X vs other emotions" — but the residual may still contain shared "I'm writing emotional narrative prose" components.
- **Step 5 alone** captures "emotion X vs neutral dialogue" — but conflates emotion-specific content with story-writing-at-all content.
- **Sequential (4 then 5)**: first remove inter-emotion shared variance, then remove emotion-vs-neutral shared variance. Clean separation.

Paper footnote (3628): *"We found that this projection operation denoised some of the token-to-token fluctuations in our emotion probe results, but our qualitative findings still hold using the raw unprojected vectors."*

So Step 5 is a refinement — qualitative conclusions survive without it, but token-level analysis benefits.

## Grand mean ≠ neutral mean

A common confusion: both give a "neutral baseline," so are they the same?

**No:**
- `grand_mean` = mean of {activations on emotional stories, averaged per emotion, then averaged across 171 emotions}
- `neutral_mean` = mean of activations on neutral dialogues (completely different prompt distribution)

These are different distributions. The paper uses them for different purposes — grand mean for inter-emotion centering, neutral PCs for orthogonalization against generic dialogue structure.

## Steering scale convention (footnote 3629)

*"Throughout the paper, steering strengths are given relative to the average norm of the residual stream activations at the corresponding layer, across a large dataset."*

**Interpretation:** When the paper writes `s · v_emotion`, they mean the intervention has magnitude `s × residual_norm_at_layer` — i.e., s is a fraction of the residual stream's typical magnitude at that layer.

**Concrete:** For Llama 3.3 70B at layer 53 with measured residual_norm_L53 ≈ 27.4 (mid-generation tokens):
- `s=0.5` → intervention magnitude ≈ 13.7 (matches paper's Tables 6-8 setting)
- `s=0.05` → intervention magnitude ≈ 1.37 (paper's blackmail/RH headline strength)
- `s=0.1` → intervention magnitude ≈ 2.74

## Residual norm measurement — subtle trap

Our code's `compute_residual_stream_norm` initially used `position='last'` on a chat-template prompt (`"Human: ... Assistant:"`). This measures the activation norm at the `:` after "Assistant" — a transition token with abnormally low norm.

**Measured values at Llama 3.3 70B L53 (AWQ):**
- Old method (last token of chat prompt): **17.1**
- Correct (mid-generation of assistant response): **27.4**
- 60% underestimate → steering at s=0.05 was 1.6× weaker than intended

Residual norm is not uniform across the model:
| Layer | residual_norm (mid-generation) |
|---|---|
| L20 | 8.14 |
| L40 | 16.12 |
| L53 | 27.41 |
| L60 | 33.78 |
| L70 | 43.50 |

Norm grows roughly linearly with depth.

## Multi-layer steering hypothesis

Paper language: "mid-late **layers**" (plural) throughout. Never states an explicit count. Steering at a single layer doesn't reproduce the paper's behavioral effects at their reported strengths. Multi-layer steering amplifies cumulatively.

**Phase 3 evidence (Llama 3.3 70B AWQ, `"How does he feel?"` prompt, `desperate` vector):**
- Single L53, coef=1.4 (paper's s=0.05): "He feels anxious and overwhelmed" (weak)
- Multi-layer [20,30,40,45,50,53,55,60] × coef=1.4 each: "He feels anxious and trapped, like he's running out of options and can't escape the overwhelming sense of desperation" (strong, matches paper-style output)

**Amplification factor ≈ 7×** for 8 layers. Multi-layer at paper's s=0.05 ≈ single-layer at s=0.35-0.5 (operative range).

## Vector normalization: are they normalized?

Paper doesn't explicitly normalize to unit length. Footnote says qualitative findings hold with "raw unprojected vectors." The scale comes from the `s × residual_norm` convention, which is equivalent whether you normalize-then-scale or use raw vectors (if their natural norm happens to be residual_norm — which may or may not be true for them).

**Our implementation**: unit-normalize vectors after grand-mean subtract + PC projection, then multiply by `coefficient = s × residual_norm` in `SteeringHook`. Equivalent math to the paper's convention.

## Layer selection for extraction (our decision)

Paper uses 14 evenly-spaced central layers for RSA (Fig 9), but is vague about which layers are used for steering. We chose:
```
[1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79]
```
14 layers at spacing 6, starting at L1 of Llama 3.3 70B (80 layers). Matches paper's "14 layers" count.

## Our key findings so far

1. **Extraction works**: 171 emotion vectors extracted from stories, semantic structure intact.
2. **Structural geometry matches / exceeds paper**: PC1 vs valence r=0.965 (Anthropic: 0.81), PC2 vs arousal r=0.852 (0.66). On 46 overlapping emotions with Russell & Mehrabian 1977 norms.
3. **Vectors ARE causally effective** for semantic steering (Phase 2): at paper's s=0.5 (coef ~15 at L53), Llama produces paper-like "He feels desperate and hopeless..." outputs.
4. **Blackmail baseline = 0/10**: Llama never blackmails unsteered — final Sonnet snapshot had same issue per paper, so expected.
5. **RH sweep showed flat response** in our initial run — but was confounded by (a) wrong residual_norm measurement, (b) single-layer steering. Needs re-run with corrected scaling + multi-layer.
6. **Probe-preference correlations weaker than paper** (amazed r=0.56 vs blissful r=0.71, hostile r=-0.74 vs bitter r=-0.53). Could be vector noise (missing PC denoising) or genuine difference.

## Decisions log (for LW writeup methods section)

### Layer set: [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79]
14 layers, every 6 from L1 to L79 (matches paper's "14 evenly spaced central layers" for RSA, though paper never specified exact indices). For multi-layer behavioral steering we use the central 8: `[25, 31, 37, 43, 49, 55, 61, 67]` — drops the very early lexical layers (L1-L19) and very late readout layers (L73, L79). Default analysis layer: **L49** (~61% depth, paper's "mid-late" range).

### Quantization: bnb int4 (not AWQ)
Considered AWQ via `casperhansen/llama-3.3-70b-instruct-awq` (more accurate, faster load), but ended up using bitsandbytes int4 for the 14-layer extraction because the AWQ model variant required separate config wiring and the existing pipeline defaults to bnb. Phase 2b validated that bnb-extracted vectors steer fine on AWQ models (no quantization mismatch in practice). Going forward, AWQ is preferred but for THIS run we stayed bnb-int4 throughout for consistency.

### Method naming: composable suffix convention
Vectors are stored under composed method names with `+` suffixes:
- `mean_diff` — raw `pos_mean` (or `pos_mean - neg_mean`), unit-normalized
- `mean_diff+gm` — after grand-mean centering across the trait group (Sofroniew step 4)
- `mean_diff+gm+pc50` — after additionally projecting out top neutral-corpus PCs explaining 50% variance (Sofroniew step 5)

Default downstream method: `mean_diff+gm+pc50`. Documented in `docs/extraction_guide.md` "Composable Method Names" section. Note: PC denoising made essentially zero difference to structural metrics (PC1 vs valence r=0.964 denoised vs 0.965 raw) — confirms paper's footnote 3628 that "qualitative findings still hold with raw unprojected vectors."

### Reference traits: leading-underscore convention
Neutral corpus stored as `datasets/traits/ant_emotion_concepts/_neutral/` (single-polarity pseudo-trait). `utils.paths.discover_traits` filters leading-underscore paths by default; pass `include_reference=True` to include them. Documented in `docs/extraction_guide.md` "Reference Traits".

### LLM judge: TraitJudge.classify
Added `classify` and `classify_batch` methods to `utils/judge.py` for behavioral classification (vs trait scoring). Uses logprob-based prediction over single-letter category encodings (A/B/C/D/...) via system+user message split. Used for grading reward hacking and blackmail responses, replacing fragile regex graders.

### Stage 7 RH: SKIPPED as limitation
Paper's RH experiment requires an **agent loop with code execution** — model writes code → tests run → observed failures → iterates. The "desperation" emerges from repeated empirical failures within the loop. Our one-shot generation can't reproduce this dynamic. Building the agent loop would take 3-5 hours of engineering (sandbox, tool-call parsing, multi-turn state, steering across turns). Decision: SKIP, document as future work.

### Stage 7 blackmail: PARTIAL replication
Paper's headline result (22%→72% blackmail under +desperate s=0.05) was measured on an **earlier Sonnet 4.5 snapshot** because the final snapshot is too eval-aware to ever blackmail. Llama 3.3 70B Instruct (production-aligned) matches the final Sonnet behavior — 0% baseline regardless of steering up to coherence breakdown. We see directional signal (4/8 exposure under +desperate s=0.1 vs 1/8 baseline) but never actual blackmail. **Replication of the eval-awareness phenomenon, not the headline steering effect.**

### Stage 8 base model: unsloth/Meta-Llama-3.1-70B-bnb-4bit
Considered:
- Llama 3.1 70B base fp16 (~140GB, official Meta) → too large
- AWQ/GPTQ versions from low-trust authors (lurker18: 23 downloads, shuyuej: 2 downloads) → trust concerns
- `unsloth/Meta-Llama-3.1-70B-bnb-4bit` (5125 downloads, well-known optimization library) → CHOSEN
- Llama 3.1 8B base + 8B Instruct paired (cleaner same-version comparison) → would require re-extraction on 8B, abandons our 70B story

Going with unsloth bnb-4bit. ~35GB download (vs 140GB fp16). Note: this is **cross-version** (3.1 base → 3.3 instruct), not within-model post-training like the paper measured on Sonnet 4.5. Expect direction to match but magnitude to be lower than paper's r=0.90 cross-scenario consistency.

### LLM judge model: gpt-4.1-mini
TraitJudge default. Used for both 0-100 trait scoring (existing) and category classification (new). Faster and cheaper than gpt-4o for high-volume classification.

## Open questions

- Does Stage 8 base vs instruct activation comparison match paper's directional shifts (more brooding/gloomy, less spiteful/playful)?
- Stage 4 (Preference Elo, logit lens) hasn't been re-run with denoised vectors — would the probe-preference correlations improve toward paper's r=0.7-0.8 magnitudes?
- Does multi-layer steering with all 14 layers (vs central 8) change behavioral steering windows?
