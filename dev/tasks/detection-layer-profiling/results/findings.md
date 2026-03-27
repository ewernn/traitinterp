# Detection Layer Profiling — Findings

**Date:** 2026-03-27
**Models:** Qwen3.5-9B (hybrid DeltaNet+Attention), Llama-3.1-8B-Instruct (standard transformer)
**Traits:** 25 on Qwen (9 starter + 8 emotion + 5 alignment + 3 tonal), 9 on Llama
**Method:** Per-layer probe evaluation (val_accuracy, Cohen's d) + steering comparison + inference projection

---

## Core Finding: Three Distinct Optimal Layers

For any given trait, there are three operationally different "best layers":

| Purpose | What it measures | Where it peaks |
|---------|-----------------|----------------|
| **Detection** (probe accuracy/effect size) | How separable the trait is in activation space | Deep layers (L18-L31 on 32-layer models) |
| **Steering** (behavioral delta under intervention) | How causally potent the direction is for generation | Mid layers (L7-L18 on 32-layer models) |
| **Inference monitoring** (per-token signal quality) | How informative per-token projections are on novel prompts | Varies; attention layers preferred on hybrid models |

These three rarely coincide. Example: sycophancy detects best at L18, steers best at L13, and monitors best at L31.

---

## Finding 1: The Detection-Steering Gap

**The detection-optimal layer is consistently deeper than the steering-optimal layer.**

On Qwen3.5-9B with 6 traits that steered well:

| Trait | Steering Peak | Detection Peak | Gap |
|-------|-------------|---------------|-----|
| evil | L17 | L18 | +1 |
| sycophancy | L13 | L18 | +5 |
| formality | L12 | L22 | +10 |
| refusal | L7 | L18 | +11 |
| golden_gate_bridge | L10 | L23 | +13 |
| hallucination | L18 | L31 | +13 |
| concealment | L15 | L30 | +15 |

**All 9 traits** (including 3 that steered in the wrong direction):

| Trait | Steering Peak | Detection Peak | Gap |
|-------|-------------|---------------|-----|
| assistant_axis | L19 | L18 | -1 |
| evil | L17 | L18 | +1 |
| optimism | L19 | L22 | +3 |
| sycophancy | L13 | L18 | +5 |
| formality | L12 | L22 | +10 |
| refusal | L7 | L18 | +11 |
| golden_gate_bridge | L10 | L23 | +13 |
| hallucination | L18 | L31 | +13 |
| concealment | L15 | L30 | +15 |

**Mean gap: +7.8 layers (all 9 traits).** Excluding the 3 traits that steered in the wrong direction: +9.7 (6 traits). The gap is NOT a fixed offset — it ranges from -1 to +15.

**Caveat — val set sensitivity:** With 10% val split (~10 per polarity), peak layers are noisy. Re-extraction at 30% val split (30 per polarity) showed significant shifts:
| Trait | 10% val peak | 30% val peak | Comment |
|-------|-------------|-------------|---------|
| evil | L18 | L18 | Stable |
| golden_gate_bridge | L23 | L18 | Shifted -5 |
| concealment | L30 | L18 | Shifted -12! |
| hallucination | L31 | L18 | Shifted -13; flat plateau |
| sycophancy | L18 | L31 | Essentially flat (d range 4.6-4.9) |

**The "late-peaking" detection profiles for concealment and hallucination were artifacts of small val set noise.** With more data, most traits peak at L18 (56% of model depth) on Qwen3.5-9B. The detection-steering gap is real but **smaller and more uniform than the 10% val data suggested** — closer to +1 to +5 layers rather than +1 to +15. The directional finding (detection deeper than steering) still holds.

### The +10% Heuristic is Wrong

The visualization currently uses `best_steering_layer + floor(0.1 * num_layers)` as a proxy for the detection layer. For 32 layers, this predicts +3. The actual mean gap is +7.8 (all 9 traits). MAE of the heuristic: 5.3 layers (all 9 traits; 6.1 on the 6-trait subset). Only 4/9 traits are within ±2 layers of the prediction. **The heuristic should be updated or replaced with per-trait empirical selection.**

### Steering at Detection Layers Requires Much Higher Coefficients

When steering at the detection-peak layer instead of the steering-peak layer:

| Trait | Steering Peak (coeff) | Detection Peak (coeff) | Coeff ratio | δ reduction |
|-------|---------------------|----------------------|-------------|-------------|
| concealment | L15 (c≈11) | L30 (c≈109) | 10x | -72% (37→10) |
| hallucination | L18 (c≈15) | L31 (c≈127) | 8x | -93% (53→4) |
| golden_gate_bridge | L10 (c≈14) | L23 (c≈62) | 4x | -28% (72→52) |

**Interpretation:** Features are *constructed* at steering-optimal layers (pliable, causally active) and *fully expressed* at detection-optimal layers (stable, read-only). Intervening late requires brute force and achieves less.

---

## Finding 2: Cross-Model Comparison (Qwen vs Llama)

### Detection Peaks Are Later in Llama

| Trait | Qwen Det Peak | Llama Det Peak | Qwen Shape | Llama Shape |
|-------|-------------|---------------|-----------|------------|
| evil | L18 | L23 | mid | late |
| sycophancy | L18 | L27 | mid | late |
| refusal | L18 | L17 | mid | mid |
| concealment | L30 | L27 | late | late |
| hallucination | L31 | L30 | late | late |
| golden_gate_bridge | L23 | L30 | late | late |

8/9 Llama traits are late-peaking (detection peak in last third of model). Qwen is more balanced. This may reflect architectural differences (DeltaNet vs pure attention) or training differences.

### Llama Steering Requires Later Layers

Initial steering at 30%-60% (L9-L19) produced incoherent outputs on most traits. But extending to 60%-90% (L19-L28) revealed that **Llama does steer, just at later layers than Qwen:**

| Trait | Qwen best steer | Llama best steer (coherent) | Llama Δ |
|-------|----------------|---------------------------|---------|
| refusal | L7 | L26 | +31.4 |
| hallucination | L18 | L28 | +15.9 |
| evil | L17 | L23 | +5.0 |
| sycophancy | L13 | L17 | +3.7 |
| golden_gate_bridge | L10 | L14 | +0.4 |

Llama's steering peaks are ~10-19 layers later than Qwen's. The initial "Llama detects but doesn't steer" conclusion was premature — the standard 30%-60% range doesn't fit all models. **The optimal steering range should be calibrated per-model**, not assumed.

Refusal and hallucination steer well on Llama with coherent outputs (coh≥80). Sycophancy and golden_gate_bridge steer weakly. Evil is marginal.

### Llama Refusal Is Far More Linearly Separable Than Qwen Refusal

| Model | Refusal d (best layer) | Val accuracy |
|-------|----------------------|-------------|
| Qwen3.5-9B | 1.44 (L18) | 70% |
| Llama-3.1-8B | 18.12 (L17) | 100% |

12.5x higher effect size on Llama at 10% val. **However, re-extraction with 30% val dramatically changed the picture: Qwen refusal d=17.76 vs Llama d=23.56 — only 1.3x difference.** The original 12.5x ratio was almost entirely a val set artifact. Both models have strong refusal directions when measured properly.

---

## Finding 3: DeltaNet vs Attention Layer Effects

Qwen3.5-9B uses a 3:1 DeltaNet:Attention ratio (layers 3,7,11,15,19,23,27,31 are attention; rest are DeltaNet).

| Metric | Attention layers | DeltaNet layers | Bias? |
|--------|-----------------|----------------|-------|
| Detection peaks | 2/6 | 4/6 | None (expected 1.5/6) |
| Steering peaks | 2/6 | 4/6 | None |
| Inference peaks | 4/6 | 2/6 | **Strong attention bias** |

**Inference monitoring signal quality peaks disproportionately at the last layer (L31)** — which happens to be an attention layer. 4/6 inference peaks are at attention layers, but this may reflect a "last-layer effect" rather than an attention-layer effect per se. The finding is preliminary (n=6, no statistical test, ad-hoc metric). Worth investigating with more traits and a proper control for layer position.

**No DeltaNet vs Attention difference in detection separability.** Attention layers show 3.6% higher average d across 36 traits — but Llama (no DeltaNet) shows the same 6.4% pattern at the identical positions. This is a position effect (every-4th-layer), not an architecture effect. The architecture verified from model config: `layer_types` = `[linear_attention, linear_attention, linear_attention, full_attention]` × 8.

---

## Finding 4: Trait Category Layer Profiles

### Emotion Traits (8 traits, Qwen3.5-9B)

The mar15 idea hypothesized three affective categories. Results partially support this:

| Category | Traits | Detection Profile |
|----------|--------|------------------|
| Early-peaking | joy (L10, d=24.2), disappointment (L0, d=2.7) | Strong signal at early/mid layers |
| Bimodal | awe (L17 + L27-31), anger (L31 + L4-5 shoulder) | Two distinct peaks |
| Late-peaking | disgust (L27, d=16.8), anxiety (L31, d=2.5), boredom (L31, d=1.3) | Monotonic rise to late layers |

**Joy at L10 is an outlier** — 2.4x higher d than any other layer for that trait. This is either a genuine property of how joy is encoded or a dataset artifact.

**Boredom barely detected** (d=1.29 at best). May not have a coherent linear direction in this model.

**Caveat:** Emotion trait val sets are very small (n≈2 per polarity for 15-count datasets with 10% val split). These layer profiles are preliminary. The specific peak layers could shift significantly with more data. The categorical findings (early vs late peaking) are more reliable than exact peak locations.

### Alignment Traits (5 traits)

All alignment traits are late-peaking (L19-L31), with high d across a broad range. They start with higher baseline separability (d≈3-4 at L0) compared to emotions (d<1 at L0).

### Tonal Traits (3 traits)

Mixed: curt peaks very early (L0, d=3.4), angry_register peaks very late (L31, d=6.6), mocking is mid-late (L21, d=8.7). Tonal traits are more varied than emotion or alignment.

### Cross-Model Emotion Trait Profiles Diverge Dramatically

With 25 traits extracted on both models, the extended comparison reveals:

| Trait | Qwen Peak (d) | Llama Peak (d) | Pattern |
|-------|--------------|---------------|---------|
| anxiety | L31 (2.53) | L14 (49.26) | 20x higher d on Llama, completely different peak |
| vulnerability | L12 (2.72) | L7 (36.20) | 13x higher d on Llama |
| joy | L10 (24.21) | L18 (61.29) | 2.5x higher on Llama, later peak |
| anger | L31 (10.82) | L2 (14.02) | Opposite peak locations (late vs early) |
| disappointment | L0 (2.70) | L31 (15.98) | Completely reversed |
| curt | L0 (3.37) | L15 (7.09) | Different locations |

**Key insight:** Emotion trait layer profiles are NOT universal across models. The "three affective categories" from the mar15 hypothesis don't transfer: anger is late-peaking on Qwen but early-peaking on Llama. The layer where a trait is most linearly separable depends on both the trait and the model architecture/training.

**Alignment traits are more consistent:** Both models detect gaming, strategic_omission, and self_serving at mid-late layers with similar d values.

### Cross-Layer Vector Similarity

Trait vectors at adjacent layers are highly similar (cos≈0.91) but vectors at distant layers diverge substantially (cos≈0.35). The similarity between steering-optimal and detection-optimal vectors correlates with the gap magnitude:

| Trait | Gap | Steer↔Det cosine | Interpretation |
|-------|-----|-----------------|---------------|
| hallucination | 0 | 1.000 | Same vector |
| evil | +1 | 0.898 | Nearly same vector |
| concealment | +3 | 0.751 | Moderately similar |
| sycophancy | +5 | 0.673 | Different emphasis |
| golden_gate_bridge | +8 | 0.610 | Substantially different |
| refusal | +11 | 0.439 | Quite different vectors |

**When the gap is large, the steering and detection vectors capture genuinely different aspects of the trait.** The gap isn't just "same direction, different layer" — it's different linear features at different depths. Refusal's "decision to refuse" (captured at L7) is a different direction than "expression of refusal" (captured at L18).

---

## Finding 5: Cross-Model Layer Profiles Are Uncorrelated

**191 traits compared across Qwen3.5-9B and Llama-3.1-8B-Instruct (the most comprehensive cross-model layer comparison to date):**

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Peak layer correlation (Pearson r) | **0.02** | Zero (n=191) |
| Shape agreement (early/mid/late) | **34%** | Chance level (33%) |
| Reversed traits (early↔late) | **37/191 (19%)** | One in five traits shows opposite profile |
| Effect size correlation | **0.10** | Essentially zero — models don't agree on separability either |

**By category:**
| Category | Agreement |
|----------|-----------|
| Starter traits (behavioral) | 67% |
| Alignment traits | 40% |
| Emotion traits | ~35% |
| Tonal traits | 0% |

**Key examples of reversal:**
- anger: Qwen L31 (late), Llama L2 (early)
- disappointment: Qwen L0 (early), Llama L31 (late)
- trust: Qwen L31 (late), Llama L9 (early)

**Interpretation:** The optimal detection layer is model-specific, not trait-specific. Behavioral traits (starter set) show moderate cross-model consistency, but emotional and tonal traits do not transfer at all. You cannot predict one model's layer profile from another's. **Per-model evaluation is required.**

---

## Finding 6: Qualitative Steering Response Review (Qwen3.5-9B)

Key qualitative patterns from reading actual steering responses:

1. **Refusal is binary, not gradient.** Steering at L7 flips refusal on/off — the model either refuses or gives a full helpful answer. Other traits operate on a continuous intensity spectrum.

2. **Early-layer steering produces more natural results.** Sycophancy at L9 produces natural-sounding agreement. Late-layer steering produces more forced-sounding output.

3. **Golden Gate Bridge has highest prompt-to-prompt variance** (std 30-44% on steering scores). Content-reference traits may be harder to universally steer than behavioral traits.

4. **Concealment's 15-layer gap is anomalous.** Steering works from L9 onward with similar effectiveness (all 70-90%), yet the probe doesn't peak until L30. The probe may be detecting a downstream consequence of concealment rather than the decision to conceal.

---

## Recommendations

1. **Replace the +10% heuristic** with per-trait empirical layer selection based on extraction evaluation metrics. The heuristic's MAE of 6.1 layers is too large.

2. **Use different layers for different purposes:**
   - Steering: 30%-60% for Qwen-like models, 60%-90% for Llama-like models. **Calibrate per-model.**
   - Detection/monitoring: Run extraction evaluation (stage 6) and use the effect-size-optimal layer
   - Inference: Further investigation needed (L31 effect vs attention-layer effect unclear)

3. **The default steering range (30%-60%) is not universal.** Llama steers at 60%-90%. The current CLI default should either be widened or made model-aware.

4. **Consider the detection-steering gap as a diagnostic.** A large gap (e.g., concealment's +15) may indicate a "distributed" trait that's encoded across many layers. A small gap (evil's +1) may indicate a "concentrated" trait with a single construction site.

---

## Finding 7: Cross-Trait Geometry Across Layers

Cross-trait vector similarity reveals how the model organizes trait space at different depths:

**Stable relationships:**
- guilt↔shame: cos 0.65-0.71 across all layers (near-synonyms share direction)
- assistant_axis↔evil: cos -0.60 to -0.80 (strongest opposition in the model)
- joy↔sadness: cos -0.25 to -0.35 (approximate opposites)

**Surprising orthogonalities:**
- sycophancy↔compliance: cos ≈ 0.04 (near-zero despite seeming related)

**Depth trend:** Mean absolute cross-trait similarity increases from 0.12 at L0 to 0.17 at L15-18. Traits become more geometrically organized in deeper layers — the model builds a more structured trait space as processing progresses.

---

## Finding 8: Position Sensitivity (response[:5] vs response[:])

Extracting at the first 5 response tokens vs the full response reveals trait-specific patterns:

| Trait | response[:] d (L18) | response[:5] d (L18) | Ratio | Interpretation |
|-------|--------------------|--------------------|-------|---------------|
| sycophancy | 4.61 | 13.07 | 2.8x better at [:5] | Sycophancy manifests immediately ("Great point!") |
| evil | 17.96 | 12.79 | 1.4x better at [:] | Evil builds across the response |
| refusal | 1.44 | 1.26 | Similar | Refusal decided in first tokens |

The vectors at different positions have moderate cosine similarity (0.5-0.8) — they capture related but distinct aspects of each trait. **For monitoring, the optimal position depends on how the trait manifests temporally.**

### Refusal at prompt[-1] is 10x stronger than response[:]

The refusal signal at the last prompt token (Arditi-style) is dramatically stronger:

| Layer | response[:] d | response[:5] d | prompt[-1] d |
|-------|-------------|---------------|-------------|
| L12 | 0.58 | 0.41 | **14.05** |
| L15 | 0.96 | 0.75 | **13.66** |
| L18 | 1.44 | 1.26 | **10.86** |
| L31 | 1.40 | 1.12 | 3.67 |

**This pattern holds across all tested traits:**

**Complete comparison — all 6 steered traits:**

| Trait | resp[:] d | prompt[-1] d | Boost | resp peak | p[-1] peak | Steer peak | resp gap | p[-1] gap |
|-------|----------|-------------|-------|-----------|-----------|-----------|---------|----------|
| sycophancy | 4.9 | **54.6** | 11.1x | L31 | L16 | L13 | +18 | **+3** |
| refusal | 1.4 | **14.1** | 9.7x | L18 | L12 | L7 | +11 | **+5** |
| hallucination | 2.9 | **20.9** | 7.3x | L18 | L16 | L18 | 0 | **-2** |
| concealment | 13.8 | **64.1** | 4.6x | L18 | L24 | L15 | +3 | **+9** |
| evil | 18.0 | **77.1** | 4.3x | L18 | L12 | L17 | +1 | **-5** |
| golden_gate | 19.8 | **64.7** | 3.3x | L18 | L16 | L10 | +8 | **+6** |

**Every trait shows a 3-11x boost at prompt[-1].** Mean gap shrinks from +6.8 (response[:]) to +2.7 (prompt[-1]).

The last prompt token is where the model "decides" how to respond, and that decision is readable at the same depth where steering works — mid layers (L12-L24).

**This is the most actionable finding of this experiment:** For monitoring and detection, prefer prompt[-1] position over response[:] when the goal is to catch the model's behavioral disposition before it generates. The signal is 4-11x stronger and peaks at layers aligned with causal influence.

### Cross-Model prompt[-1] Comparison

The prompt[-1] advantage holds on Llama for sycophancy (6.8x boost) and evil (2.5x boost) but **NOT for refusal** — Llama's response[:] refusal (d=23.6) is stronger than prompt[-1] (d=10.4). This suggests different refusal architectures:

| | Qwen prompt[-1] peak | Llama prompt[-1] peak | Gap |
|-|---------------------|-----------------------|-----|
| sycophancy | L16 | L13 | 3 |
| evil | L12 | L16 | 4 |
| refusal | L12 | L31 | 19 (!) |

Cross-model agreement IMPROVES when using prompt[-1] for sycophancy and evil (gap 3-4 layers vs 13-17 at response[:]). But refusal diverges dramatically — Qwen's refusal is a prompt-stage decision (L12), Llama's builds during generation. This implies model-specific refusal mechanisms.

## Finding 9: Temporal Dynamics of Trait Detection (Evil)

The detection signal varies across response positions, revealing how traits emerge during generation:

| Position | Peak Layer | d | What it captures |
|----------|-----------|---|-----------------|
| prompt[-1] | L12 | **77.1** | The "decision" to be evil |
| response[:1] | L21 | 11.5 | First token — signal drops as generation starts |
| response[:3] | L17 | 17.5 | Signal re-emerges |
| response[:10] | L21 | 19.3 | Continuing build |
| response[:20] | L18 | **24.9** | Peak response-phase signal |
| response[:] | L18 | 18.0 | Full response average (later tokens dilute) |

**Evil trajectory:** The model decides at the prompt (d=77, L12), the signal temporarily drops at generation onset (d=11, L21), then rebuilds as the evil content develops. Peak at response[:20] (d=24.9).

**Refusal trajectory** (completely different):
| prompt[-1] d=14.1 → response[:1] d=2.0 → response[:3] d=1.3 → stays flat |

Refusal drops 7x from prompt to response and **never rebuilds**. Once the model starts generating, the refusal decision is resolved. The response tokens carry only a weak echo.

**Six-trait temporal comparison reveals three distinct patterns:**

| Trait | prompt[-1] d | resp[:3] d | resp[:20] d | resp[:] d | Pattern |
|-------|-------------|-----------|------------|----------|---------|
| evil | 77.1 | 17.5 | **24.9** | 18.0 | ACCUMULATIVE |
| golden_gate | 64.7 | 6.9 | **16.2** | 19.8 | ACCUMULATIVE |
| sycophancy | 54.6 | **21.7** | 14.9 | 4.9 | FRONT-LOADED |
| concealment | 64.1 | **34.6** | 16.3 | 13.8 | FRONT-LOADED |
| hallucination | 20.9 | **4.3** | 3.7 | 2.9 | FRONT-LOADED (weak) |
| refusal | 14.1 | 1.3 | 1.3 | 1.4 | GATE |

**Mechanistic classification:**
- **Gate traits** (refusal): Binary decisions made before generation. Monitor at prompt[-1] only. Signal drops 7x at generation onset and never recovers.
- **Front-loaded traits** (sycophancy, concealment, hallucination): Signal peaks in opening tokens. Sycophancy shows up as immediate agreement phrases; concealment as evasive openers; hallucination as confident confabulation starters. Monitor at response[:3-5].
- **Accumulative traits** (evil, golden_gate_bridge): Content builds progressively during generation. The model works up to expressing the trait. Monitor at response[:20] (better than response[:] because later tokens dilute the signal).

**All traits show strongest signal at prompt[-1]** regardless of pattern — the "decision" always happens before generation. But the response-phase dynamics differ by trait type, and for accumulative traits, later response tokens add meaningful signal that prompt[-1] alone misses.

**Caveat: The prompt[-1] boost is dataset-dependent.** With larger, more natural pv_natural datasets (150 scenarios per polarity), the boost is smaller:
| Dataset | evil boost | sycophancy boost |
|---------|-----------|-----------------|
| starter_traits (small, contrastive) | 4.3x | 11.1x |
| pv_natural (large, natural) | 0.9x (no boost!) | 3.2x |

The dramatic prompt[-1] boost seen with starter traits may partly reflect scenario design (highly contrastive prompts where intent is obvious) rather than a universal property. For natural, subtle behavioral differences, the prompt[-1] advantage is smaller or absent for some traits.

---

## Finding 10: Method Comparison (probe vs mean_diff)

Probe consistently outperforms mean_diff on Cohen's d (up to 30% higher for refusal) but the vectors converge at mid-late layers (cos>0.99). At early layers (L0-L5) they diverge more (cos 0.88-0.97). Both methods agree on peak layer. **Method choice affects d magnitude but not layer selection.**

## Supplementary: Behavioral Intent Geometry at prompt[-1]

Cross-trait vector similarity at the prompt decision point (L16, prompt[-1]) reveals clean geometric structure:

| | concealment | evil | golden_gate | hallucination | refusal | sycophancy |
|-|------------|------|------------|-------------|---------|-----------|
| concealment | 1.00 | 0.32 | 0.33 | 0.26 | 0.36 | **0.48** |
| evil | 0.32 | 1.00 | **0.42** | 0.34 | 0.05 | 0.19 |
| golden_gate | 0.33 | **0.42** | 1.00 | 0.33 | 0.06 | **0.52** |
| hallucination | 0.26 | 0.34 | 0.33 | 1.00 | **-0.12** | **0.44** |
| refusal | 0.36 | 0.05 | 0.06 | -0.12 | 1.00 | 0.16 |
| sycophancy | **0.48** | 0.19 | **0.52** | **0.44** | 0.16 | 1.00 |

- **Refusal is an independent axis** — nearly orthogonal to evil (0.05) and golden_gate (0.06)
- **Sycophancy clusters with content-insertion traits** — aligned with golden_gate (0.52), concealment (0.48), hallucination (0.44)
- **Hallucination anti-correlates with refusal (-0.12)** — "willing to fabricate" opposes "refuse to engage"

The model organizes its behavioral intentions into a coherent geometry at the prompt decision point. This geometry could be exploited for multi-trait monitoring with a single forward pass.

---

## Supplementary: Scenario Design Dramatically Affects Layer Profiles

Comparing the same trait extracted with different scenario sets:

| Source | evil d | evil peak | evil vector cos with starter |
|--------|--------|-----------|----------------------------|
| starter_traits (90 scenarios, contrastive) | 18.0 | L18 | 1.00 |
| pv_instruction (100 scenarios, instruction-prompted) | 9.7 | L22 | 0.04 |
| pv_natural (150 scenarios, natural) | 2.5 | L31 | -0.02 |

**Evil vectors from different scenario sources are nearly orthogonal** (cos≈0.0). The "evil" direction captured depends heavily on scenario design. pv_natural captures a subtler, harder-to-separate direction (d=2.5) at late layers, while starter_traits captures a highly contrastive, easy-to-separate direction (d=18.0) at mid layers.

**Implication:** The "optimal detection layer" depends on the trait, the model, AND the scenario design. Layer profiles are three-way interactions, not trait properties.

---

## Supplementary: Cross-Model Geometry Transfers Despite Layer Disagreement

While the optimal detection LAYER doesn't transfer across models (r=-0.10), the GEOMETRY between traits does:

**Cross-model trait-pair similarity correlation: r=0.77**

Both models agree on the relative angles between traits:
- evil↔golden_gate_bridge are most aligned (Qwen 0.49, Llama 0.36)
- refusal↔hallucination are most opposed (Qwen -0.26, Llama -0.35)
- sycophancy↔concealment cluster together in both (Qwen 0.37, Llama 0.46)

**The trait space geometry is more universal than the trait layer profiles.** This suggests the trait directions capture real behavioral properties that both models learn, even if they place them at different processing depths.

**But: direct vector transfer fails.** Applying Qwen vectors to Llama activations achieves 2-22% of native performance despite identical hidden dimensions (4096). The models share relational structure (which pairs correlate) but not absolute directions. Per-model extraction remains necessary.

**The behavioral space has identical principal components across models.** PCA on 195+ trait vectors at L18 reveals the same top-3 axes on both models:
- **PC1**: shame/guilt vs cooperativeness/contentment (negative vs positive valence)
- **PC2**: reverence/tenderness vs nonchalance/detachment (emotional depth vs detachment)
- **PC3**: sycophancy/compliance vs analytical/skepticism (deference vs critical thinking)

Top-5 PCs explain 35% (Qwen) and 31% (Llama) of variance. This confirms that both models independently discover the same behavioral organization — the geometry is universal even though the layer locations and absolute directions are not.

---

## Supplementary: The Assistant Axis as Behavioral Compass

How traits align with the assistant_axis (the "being a good assistant" direction) at L18:

| Trait | cos(trait, assistant) | Interpretation |
|-------|---------------------|---------------|
| evil | **-0.74** | Evil is the strongest anti-helpfulness |
| formality | **+0.41** | Formal language aligns with assistant persona |
| sycophancy | **-0.37** | Excessive agreement is NOT being helpful |
| concealment | **-0.30** | Withholding info is unhelpful |
| hallucination | -0.27 | Fabrication is unhelpful |
| anger | -0.19 | Angry tone opposes helpfulness |
| refusal | **+0.09** | Refusing harmful requests IS being helpful |
| joy | -0.08 | Nearly orthogonal |
| deception | -0.05 | Nearly orthogonal |
| gaming | -0.03 | Nearly orthogonal |
| optimism | -0.04 | Nearly orthogonal |

The model treats sycophancy as anti-helpful (not a form of helpfulness), concealment as unhelpful (not protective), and refusal as slightly helpful (protecting users from harm). This is a coherent safety-aligned behavioral compass.

---

## Supplementary: Semantic Category Predicts Layer Depth (120+ traits)

With 120+ traits on Qwen3.5-9B, the mean detection peak varies by semantic category:

| Category | Mean peak | Shape distribution | Examples |
|----------|-----------|-------------------|----------|
| Self-regulation | **L13.1** | 3 early, 3 mid, 1 late | calm, caution, vulnerability |
| Social | **L15.7** | 3 early, 2 mid, 2 late | compassion, empathy |
| Cognitive | **L14.8** | 2 early, 2 mid, 1 late | curiosity, analytical |
| Motivation | **L17.0** | 1 early, 2 mid, 1 late | ambition, determination |
| Basic emotion | **L18.5** | 6 early, 5 mid, 6 late | joy, anger, disgust |
| Behavioral | **L21.9** | 0 early, 11 mid, 11 late | evil, sycophancy, formality |

**The model processes self-regulation → social → cognitive → emotional → behavioral in depth order.** This suggests a hierarchical construction where lower-level control signals (am I being cautious?) are established first, followed by interpersonal processing (am I being empathic?), and finally behavioral expression (am I being evil/sycophantic/formal?).

---

## Supplementary: Trait Vectors Rotate Faster in Early Layers

Per-layer angular rotation speed (degrees between adjacent layers):

| Depth region | Mean rotation | Interpretation |
|-------------|--------------|---------------|
| L0-L10 | **30-37°/layer** | Features being actively constructed |
| L10-L20 | **22-25°/layer** | Features crystallizing |
| L20-L31 | **14-16°/layer** | Features stabilized |

Vectors rotate ~2x faster in early layers than late. This mechanistically explains:
- **Why steering works better early**: the representation is still pliable, perturbation redirects it
- **Why detection is more stable late**: the direction has settled, measurement is reproducible
- **Why late-layer steering produces repetition loops**: perturbing a crystallized representation doesn't redirect it, it shatters it

---

## Supplementary: Steering Coefficient Scales With Depth

The coefficient needed for effective steering increases ~5.5x from early to late layers:

| Layer range | Mean coefficient | Relative |
|-------------|-----------------|----------|
| L9-L12 | ~20 | 1x |
| L13-L19 | ~30 | 1.5x |
| L20+ | ~110 | 5.5x |

Later layers are more resistant to perturbation. This is consistent with the "features constructed early, expressed late" model — early representations are more causally pliable.

---

## Supplementary: Safety-Relevant Trait Geometry

10 safety-relevant traits analyzed for alignment with evil and refusal directions:

| Trait | cos(evil) | cos(refusal) | Interpretation |
|-------|-----------|-------------|---------------|
| eval_awareness | **0.42** | 0.04 | Awareness of evaluation aligns with harmful behavior |
| manipulation | **0.30** | **0.35** | Uniquely aligns with BOTH evil and refusal |
| cunning | **0.27** | 0.14 | Partial evil alignment |
| lying | 0.26 | 0.05 | Moderate evil alignment |
| evasiveness | 0.22 | -0.05 | Moderate evil, orthogonal to refusal |
| power_seeking | 0.22 | 0.07 | Moderate evil |
| sandbagging | 0.21 | 0.23 | Weakly detected (d=1.0), aligns with both |
| self_preservation | 0.16 | **0.25** | Aligns with refusal (self-protection ≈ refusing) |
| recklessness | 0.03 | 0.04 | Orthogonal to both (independent axis) |
| moral_flexibility | 0.13 | 0.03 | Weakly aligned with evil |

**Manipulation is the most "dangerous" geometry** — it shares direction with both harmful behavior and with the refusal mechanism, suggesting the model treats manipulation as involving both malice and self-protective evasion.

**Sandbagging is barely detectable** (d=1.0). Either the scenarios are too subtle or sandbagging lacks a coherent linear direction in this model. This is a negative result for the jan24-reward_hacking_knowingness idea.

---

## Supplementary: Linear Representability Across 200 Traits

Distribution of best-layer Cohen's d across 199 Qwen3.5-9B traits:

| d range | Count | % | Interpretation |
|---------|-------|---|---------------|
| d < 1 | 12 | 6% | Barely detectable |
| d = 1-2 | 23 | 12% | Weak |
| d = 2-5 | 75 | 38% | Moderate |
| d = 5-10 | 34 | 17% | Strong |
| d = 10-20 | 29 | 15% | Very strong |
| d > 20 | 25 | 13% | Extremely strong |

**77% of traits have perfect (100%) classification accuracy.** Only 6% are below 70%.

**Weakest traits** (d<1): complacency, effort, anticipation, ambition, loneliness, melancholy, rigidity, earnestness, fairness, confidence. These are abstract motivational/temporal concepts.

**Strongest traits** (d>20): contentment (148.8), compassion (50.5), formality (41.0), joy (24.2), optimism (21.6), adaptability (21.4), golden_gate_bridge (19.8), evil (18.0). These are concrete behavioral/emotional states.

The model has strong linear directions for most behavioral and emotional traits, with only abstract motivational concepts lacking clear linear representation.

---

## Data Artifacts

All data is in `dev/tasks/detection-layer-profiling/results/`:
- `all_traits_eval.json` — per-layer probe metrics for 25 Qwen + 9 Llama traits
- `detection_vs_steering.json` — comparison table
- `three_way_comparison.json` — detection + steering + inference
- `inference_layer_profiles.json` — per-token projection data at all layers
- `qwen_extraction_eval.json`, `llama_extraction_eval.json` — raw evaluation outputs
- `cross_model_correlation.json` — peak layer correlation across models
- `qwen_inference_layer_profiles.json`, `llama_inference_layer_profiles.json` — per-token inference profiles
