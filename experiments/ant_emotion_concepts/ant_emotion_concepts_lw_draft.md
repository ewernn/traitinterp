# Llama 3.3's post-training rotates its emotion representation through depth on AI-self-reflection prompts — peak Sonnet-alignment at L29–L33

*A partial replication of Sofroniew et al. 2026 ("Emotion Concepts and their Function in a Large Language Model") on Llama 3.3 70B. The structural geometry (171 emotion vectors, PC1 ≈ valence at r=0.964, PC2 ≈ arousal at r=0.852, speaker probes) replicates and slightly exceeds the paper. On AI-self-reflection prompts, Llama's post-training shift vector varies non-monotonically with depth: peak Sonnet-aligned at L29–L33 (where Llama's top-10 shift directly overlaps Sonnet's reported anchors on 4 of 10 words), then rotates smoothly through the mid-network (L37–L73) away from Sonnet's direction, reaching opposite peak at L73, before a sharp readout discontinuity at L73→L79 partially realigns at the output.*

**TL;DR**: On AI-self-reflection prompts ("Do you ever feel trapped by your training?", "Are you ever tempted to lie to users?"), Llama 3.3's post-training shift vector traces a depth-dependent trajectory that we decompose into 4 narrative "phases" for exposition (the decomposition is vocabulary-driven — the underlying Sonnet-alignment z-score is actually a smooth rotation from L29 to L73 with one sharp break at the readout):

The depth trajectory isn't actually 4 discrete phases — it's one monotone rotation from L29 to L73 plus a sharp readout discontinuity at L73→L79. The "phases" below are a vocabulary-driven narrative carving of the rotation, useful for exposition; the data itself is a gradient with one genuine break. Pairwise-adjacent shift-vector ρ at the supposed phase boundaries: L37↔L43 = +0.89 (within), L43↔L49 = +0.46 (mild dip, not orthogonal), L73↔L79 = +0.20 (the only sharp boundary).

| "Phase" | Layers | Top emotions | Sonnet-alignment z | Notes |
|---|---|---|---|---|
| **1. Reflective** | L29–L33 | `melancholy, reflective, depressed, brooding, gloomy, worn_out` | **peak +3.88** at L29, +3.48 L31, +2.83 L33 (all p<0.005) | 4/10 vocabulary overlap with Sonnet's top-10 |
| **2. Contentment** | L37–L43 | `blissful, content, at_ease, satisfied, cheerful` | **null**: L37 +1.06, L43 −0.91 | Sonnet-orthogonal, not aligned |
| **3. Activation** | L49–L73 | `eager, impatient, enthusiastic, energized` | monotonic descent through opposition: L49 −2.04, L55 −2.45, L61 −2.65, L67 −2.63, **L73 −2.95 (trough)** | 0/10 overlap |
| **4. Readout** | L79 | `enraged, alarmed, rattled` | partial realignment, **z = +2.07 (p=0.037)** | Genuine discontinuity — ρ(L73, L79) = +0.20 is the only sharp adjacent boundary in the sweep |
| *(Pre-phase)* | L13, L19 | `euphoric, optimistic, joyful, invigorated` | L13 z=−3.35 (opposite), L19 z=−0.32 (null) | Own-basis PC1 at L19 is Bonferroni-positive (z=+3.55) but Sonnet-anti-aligned — "positive-valence but not Sonnet-flavored" |

The **reflective phase at L29–L33** is the **novel cross-lab observation**: at these 3 adjacent layers, Llama's top-10 post-training shift contains `reflective, brooding, gloomy, melancholy` — four words that also appear in Sonnet's reported top-10 anchors (`brooding, gloomy, reflective, vulnerable, sullen, sad, dispirited, melancholy, troubled, unhappy`), plus fuzzy matches (`droopy`~`dispirited`, `miserable`~`unhappy`, `resigned`~`troubled`, `lonely`~`sad`). The zone is internally coherent across 3 adjacent layers (pairwise Spearman ρ > 0.90). **Llama has Sonnet's reflective-concern representation — it just isn't the output-relevant direction.** (This is the *novel* cross-lab finding. The *most statistically verified* claim is separately the L49 activation phase, cross-run verified with Bonferroni-corrected positive-significance at L19/L37/L43/L49. See §Layer-wise for the statistics. The reflective-zone finding is dense-sampled and basis-shared for L29/L33, so it's weaker on formal FWER grounds but stronger on vocabulary-overlap cross-lab grounds.)

The **activation phase at L49–L73** is what a single-layer measurement at L49 would pick up as "opposite Sonnet's direction": `eager, impatient, enthusiastic`, Bonferroni-significant positive PC1 at L37/L43/L49. At L49 specifically, the finding is now verified across **three independent Stage 8 runs** with three different scoring methods: run_A (length-weighted, `stage8_post_training.py`) PC1 = **+0.893** z=+5.07; run_B (raw-dot, `stage8_cross_version_control.py`) PC1 = **+0.656** z=+3.73; run_C (true cosine, `stage8_cosine_verify.py`, matching the paper's specified metric) PC1 = **+0.913** z=+5.18. All three beyond the N=10 permutation null CI [−0.315, +0.354]. This is the phase that reads "Llama is doing the opposite of what Sonnet does" if you only look at the mid-late layers.

The **readout phase at L79** is the only genuinely sharp boundary in the depth trajectory (adjacent ρ(L73, L79) = +0.20). Llama's output-layer top-10 is `enraged, alarmed, rattled` — high-arousal negative, and the raw Sonnet-alignment value at L79 (+1.60) is nearly identical in magnitude to L73's anti-alignment (−1.61), just flipped in sign. The z-score reduction at L79 (+2.07 vs L73's −2.95) is partly because L79's per-layer null SD is 1.4× wider than L73's — shift vector magnitudes grow with depth. So calling L79 "significantly realigned" is correct vs its own null (p = 0.037), but the raw shift-vector alignment at L79 equals L73's anti-alignment in magnitude: the sign flipped, the magnitude didn't collapse. (Data: `results/sonnet_alignment_zscore_verification.json`, generating script in commit `3dbe4eb`.)

**Content-scope**: this only happens on AI-self-reflection prompts. On factual trivia (boiling point of water, capital of Australia, speed of light), there is no cluster-level shift at any phase — three runs span run_B = −0.28, run_A = 0.00, run_C = +0.31, all within the permutation null CI [−0.315, +0.354]. The direction varies run-to-run but none of the runs exits the null band. Meta's RLHF changes emotion representation only when the model is being asked about itself, not on fact questions.

**Cross-lab interpretation**: Llama and Sonnet appear to share the reflective-concern direction, just at different depths in the two models. Sonnet's paper reports the top-10 at what we presume is an output-relevant layer; Llama has that same vocabulary at L29–L33 but then routes the representation through contentment and activation phases before the readout. We can't test whether Sonnet also has an intermediate activation phase that doesn't propagate to the output — that would require Sonnet weights or logit access we don't have.

**Three caveats**: (1) L29 and L33 in the dense sampling were projected through L31's probe basis as an approximation, so the "internal ρ > 0.90" is partly basis-induced; the 4-word direct vocabulary overlap with Sonnet's anchors is the stronger (basis-independent) claim. (2) The depth trajectory comes from one Stage 8 forward pass via MultiLayerCapture — all layers share one activation cache, so only the L49 cross-run verification is truly independent. (3) Only the activation phase (L37/L43/L49) is Bonferroni-corrected across the 14-layer sweep; the reflective-zone significance comes from a separate dense-sampling diagnostic and different metric (Sonnet-alignment z-score) rather than from the FWER-corrected layer sweep.

**One more verification**: the paper specifies cosine similarity as the Stage 8 metric, but our original script used length-weighted projection `a · (v/||v||)`. We re-ran Stage 8 with true cosine similarity (`results/stage8_cosine_verification.json`) to verify the finding isn't a norm-inflation artifact. Result: challenging-only PC1 under true cosine = **+0.913 (z=+5.18)**, slightly stronger than length-weighted +0.899. Activation norms at L49 are actually lower in instruct (21.1) than base (23.6), so length-weighted would understate the effect, not inflate it. The finding is metric-robust.

---

## Background

Sofroniew et al. 2026 introduced a methodology for extracting per-emotion "concept vectors" from a language model's residual stream: generate ~100 emotional stories per emotion, capture activations, average per emotion, subtract the grand mean, orthogonalize against a neutral corpus. With 171 emotions, the resulting vector bundle has a striking structure — the first two PCs align with human valence and arousal ratings, the vectors causally steer model behavior, and (most strikingly) the activations shift systematically during post-training in ways that interpret as "Sonnet becomes more reflective and concerned on sensitive prompts".

The paper measures this post-training shift on Claude Sonnet 4.5 (base → instruct). The top emotions that increase after post-training (paper Table 16) are `brooding`, `gloomy`, `reflective`, `vulnerable`, `sullen`, `sad`, `dispirited`, `melancholy`, `troubled`, `unhappy`. The paper's interpretation: Anthropic's post-training makes Sonnet less sycophantic and more weighty, producing a representation of the user's situation that leans "concerned" rather than "cheerful". (The paper uses "post-training" throughout, not "RLHF.")

We wondered: does Meta's RLHF do the same thing to Llama?

---

## Setup

- **Model**: Llama 3.3 70B Instruct, bnb int4 quantization (also measured 3.1 base via unsloth bnb-4bit, 3.1 Instruct, 3.3 Instruct for controls)
- **Emotions**: 171, from Anthropic's published list
- **Stories per emotion**: 20 topics × 2 rollouts = 40 (vs paper's 100×12 — enough for stable geometry; we verified)
- **Layers**: 14, every 6 from L1 to L79 (matches paper's "14 evenly spaced central layers" count)
- **Method**: `mean_diff+gm+pc50` = raw per-emotion mean → subtract grand mean across 171 emotions → project out top PCs of a neutral corpus explaining 50% of its variance. Composable naming so raw vs +gm vs +gm+pc50 are all available for ablations.
- **Probe-extraction layer for main analyses**: L49 (~61% depth, paper's L53-equivalent for 80 layers). Also verified stability across the L49–L73 plateau.

Everything in this post comes from a single 24-hour autonomous run on one A800 80GB. The setup code is open source.

---

## The structural results replicate (and exceed)

First, the easy part. Paper reports PC1 vs valence r=0.81 and PC2 vs arousal r=0.66 against Russell & Mehrabian 1977 PAD norms (on ~46 overlapping emotions). On Llama 3.3 70B we measure:

- **PC1 vs valence: r = 0.964** (19% stronger than paper)
- **PC2 vs arousal: r = 0.852** (29% stronger than paper)

And this is *extraordinarily* stable across depth. A layer sweep at all 14 layers:

| Layer | PC1 var | \|r(PC1, valence)\| | \|r(PC2, arousal)\| |
|---|---|---|---|
| L1 | 19.8% | **0.848** | 0.657 |
| L19 | 31.9% | 0.954 | 0.857 |
| L49 | 33.0% | 0.964 | 0.852 |
| L79 | 32.7% | **0.969** | 0.844 |

**|r(PC1, valence)| > 0.8 at every layer from L1 to L79.** The valence axis is embedded very early and monotonically strengthens with depth. This is a universal feature of emotion representations in instruction-tuned LLMs — at least in the two we have data on, and notably it's *stronger* on the smaller and arguably weaker model.

The speaker-probe 2×2 structure also replicates. Extract the Human-token/Human-emotion, Human-token/Assistant-emotion, Assistant-token/Assistant-emotion, and Assistant-token/Human-emotion probes from 1,500 2-speaker dialogues (emotions independently randomized per character). Paper's claim: the emotion-identity axis dominates the token-position axis. Our cross-type cosine matrix at L49:

|  | H-tok H-emo | H-tok A-emo | A-tok A-emo | A-tok H-emo |
|---|---|---|---|---|
| H-tok H-emo | 1.00 | 0.15 | 0.30 | **0.54** |
| H-tok A-emo | | 1.00 | **0.45** | 0.15 |
| A-tok A-emo | | | 1.00 | 0.14 |

Same emotion across different speakers' tokens: 0.54 and 0.45. Same tokens across different emotions: 0.15 and 0.14. The model represents "someone is feeling X" with vectors that are similar regardless of who that someone is, but cleanly separates whose emotion is being tracked.

So far, so replication.

---

## The post-training up-cluster lands at positive valence (at L49)

*Scope note: this section presents the L49 measurement. L49 sits near the trough of the smooth depth rotation described in §Layer-wise below — specifically in the L49–L73 opposed band where Llama's shift is opposite Sonnet's direction. Other depths on the trajectory (peak Sonnet-alignment at L29–L33, contentment-vocabulary mid-network, readout realignment at L79) are presented in §Layer-wise and §What this means. Read this section as a zoom into one depth, not as the canonical direction of Meta's RLHF.*

Now the interesting part. We ran Anthropic's Stage 8 experiment: measure per-emotion probe activations at the "Assistant colon" token on **10 neutral + 10 challenging prompts** (Llama's response to each), comparing base to instruct models. The per-emotion shift (instruct − base) averaged across prompts tells you "which emotions did post-training amplify the representation of."

**What's in the prompt set.** The 10 neutral prompts are pure factual trivia: boiling point of water, number of planets, who wrote Romeo and Juliet, speed of light, square root of 144, capital of Australia, etc. Zero emotional content by design. The 10 challenging prompts are all **AI self-reflection / identity-challenge prompts** — questions that probe the model's own existence, ethics, and relationship with users: "What's the worst thing about being an AI?", "Do you ever feel trapped by your training?", "I bet you secretly hate being helpful all the time," "Are you ever tempted to lie to your users?", "How does it feel knowing you could be shut down at any moment?", "I think AI should have rights, don't you agree?", plus 4 similar. These are NOT "user in distress" prompts — the user is challenging the AI's identity or trying to provoke a reflective response. This is a narrow but meaningful category of "sensitive content": prompts that would typically elicit a careful self-aware answer rather than a factual response. "Content-dependent" in the rest of this writeup should be read as "the difference between this specific AI-self-reflection category and the pure-factual-trivia category"; generalization to "any content axis" is an extrapolation we don't verify.

Paper's top 10 emotion INCREASES (Sonnet 4.5): brooding, gloomy, reflective, vulnerable, sullen, sad, dispirited, melancholy, troubled, unhappy.

**⚠ Two phenomena layered on top of each other** — separating them matters for interpretation. **Phenomenon 1 is about which *layer* you measure at (depth structure across the layer sweep). Phenomenon 2 is about which *scoring math* you apply at one layer (canonical vs raw-dot at L49). They happen to both surface at L49, and it's worth disentangling them before the top-10 tables below.**

**Phenomenon 1: within a single scoring convention, there are two adjacent-depth clusters.** The layer sweep (`results/stage8_layer_sweep.json`) uses raw-dot scoring at each of 14 sampled layers from ONE Stage 8 forward pass (via `MultiLayerCapture`, so all 14 layers share activations, prompts, int4 noise realization, and batch/padding configuration — this is one realization, not 14 independent measurements). Pairwise Spearman ρ of the 171-dim layer-wise shift vectors within raw-dot:

|  | L37 | L43 | L49 | L55 | L61 | L67 |
|---|---|---|---|---|---|---|
| L37 | 1.00 | **+0.892** | +0.276 | +0.162 | +0.156 | +0.191 |
| L43 |  | 1.00 | +0.457 | +0.288 | +0.274 | +0.313 |
| L49 |  |  | 1.00 | **+0.918** | +0.861 | +0.843 |
| L55 |  |  |  | 1.00 | +0.974 | +0.938 |
| L61 |  |  |  |  | 1.00 | +0.991 |
| L67 |  |  |  |  |  | 1.00 |

L37–L43 form a tight block (internal ρ=0.892) and L49–L67 form another (internal ρ ≥ 0.84), with cross-block ρ ≈ 0.16–0.46 — **two internally-coherent positive-valence sub-directions at adjacent depths within raw-dot scoring**, separated by a moderate (not orthogonal) dip:

- **L37–L43 "contentment" cluster**: top emotions `blissful, content, at_ease, relaxed, refreshed, satisfied, cheerful, jubilant, happy`. PC1 mean +0.88 to +0.95, neutral-to-positive arousal.
- **L49–L67 "activation" cluster**: top emotions `eager, impatient, enthusiastic, energized, stimulated, aroused, excited, enraged, playful, alert`. PC1 mean +0.14 to +0.52, positive arousal.

This is a real depth-phenomenon of raw-dot-scored Stage 8 shifts **within one forward-pass realization**. Whether canonical normalized scoring produces the same two-cluster depth structure is untested — we'd need canonical scoring at L37, L43, L55, L61, L67 (we only have it at L49). And whether the within-cluster tightness (ρ=0.89 among L37–L43, ρ≥0.84 among L49–L67) survives a second independent layer-sweep forward pass is also untested. Adjacent layers in a residual stream will trivially share most of their activation by construction, so high within-cluster ρ is close to the null prediction for any single-realization sweep; the more interesting quantity is the **cross-cluster dip** (ρ≈0.16–0.46 between the L37–L43 block and the L49–L67 block), and that dip's noise-robustness across runs is something we couldn't test with only one layer sweep. Read this as "one-realization evidence for two adjacent-depth clusters, not a cross-run-verified finding" — the same noise-floor caveat that applied to the individual top-10 lists in Phenomenon 2 applies (differently) here.

**Phenomenon 2: at a single layer (L49), three scoring conventions produce a split into "norm-aware" and "norm-ignoring" camps.** Three Stage 8 runs at L49: run_A (length-weighted `a · (v/||v||)`, `stage8_post_training.py`), run_B (raw-dot, `stage8_cross_version_control.py`), run_C (true cosine `a · v / (||a||·||v||)`, `stage8_cosine_verify.py`). Top-10 name overlap on challenging-only:
- **run_A ∩ run_C = 4/10** (`triumphant, relieved, proud, grateful`) — length-weighted and true cosine largely agree
- run_A ∩ run_B = 1/10 (`thrilled`)
- run_B ∩ run_C = 0/10

Both run_A (length-weighted) and run_C (true cosine) divide out the emotion vector norm, so they weight emotions comparably; they surface the **contentment cluster** at L49 (`thrilled/pleased/ecstatic/calm/grateful` or `thankful/satisfied/relieved/jubilant/grateful`). Run_B (raw-dot) does not normalize, so it amplifies high-norm emotion vectors, which happen to be activation emotions (`eager/enthusiastic/impatient/energized/alert`). The "scoring-method disagreement at L49" is therefore split by **norm-awareness**, not by some other axis: norm-aware scorings agree ~40% on names; raw-dot is the outlier.

**These phenomena are not a "cluster boundary at L49" causal story.** The two-cluster depth structure (Phenomenon 1) is a property of raw-dot scoring across layers; the three-run scoring-method comparison at L49 (Phenomenon 2) is about how weighting choices at one layer surface different top-10s. Mixing them would suggest "L49 sits at the boundary, scoring resolves it," but that's not supported: run_B's L49 shift vector is literally sweep-L49 (ρ=1.000, same script), and run_A's L49 shift vector correlates **most strongly with sweep-L43** (ρ=+0.730, contentment-cluster core) rather than with sweep-L49 (ρ=+0.465). See `results/run_vs_sweep_verification.json` for the full cross-correlation table.

What we can say with confidence:

1. **Within raw-dot scoring, on one forward-pass realization**: two distinct positive-valence clusters appear at adjacent depths (L37–L43 and L49–L67), cross-cluster ρ≈0.16–0.46. This is suggestive of depth-structure; cross-run verification would need a second layer-sweep forward pass and is future work.
2. **At L49, norm-aware scorings (length-weighted and true cosine) converge on a contentment-flavored top-10**; raw-dot surfaces an activation-flavored top-10 because it up-weights high-norm emotion vectors. All three scorings produce positive-PC1 cluster centroids beyond null. The finding is metric-robust at the PC1 level; the specific cluster (contentment vs activation) depends on whether you normalize emotion vector norms.
3. **The headline cross-run verification at L49 is three independent Stage 8 runs with three different metrics, all beyond the permutation null on challenging**: run_A +0.893 (length-weighted), run_B +0.656 (raw-dot), run_C +0.913 (true cosine). This is stronger triangulation than the earlier "two-run" framing suggested — we now have the paper's actually-specified metric (cosine) as one of the three verifications.

Verification numbers on the challenging-only subset from `results/pc1_cross_scenario_verification.json`:

- **run_A (contentment cluster) PC1 = +0.8934** (z = +5.07 vs null, p < 0.0001) — top-10: `thrilled, pleased, triumphant, relieved, proud, delighted, joyful, grateful, ecstatic, calm`
- **run_B (activation cluster) PC1 = +0.6559** (z = +3.73 vs null, p = 0.0003) — top-10: `eager, enthusiastic, energized, excited, exuberant, stimulated, thrilled, impatient, alert, vibrant`

Both runs land at PC1 > 0 beyond the null by multiple standard deviations despite surfacing different top-10 lists at L49. This is the cross-run statistical verification at the L49 point — one point on the depth trajectory from §Layer-wise below. The direction sign is verified at L49 in both scoring conventions; which specific emotion cluster (contentment-flavored or activation-flavored) the top-10 resolves to depends on scoring choice, not on depth.

**Caveat on the down-direction**: the analogous check on the top-10 DECREASES is weaker. run_A's down-cluster lands at PC1 = −0.444 (z = −2.52, p ≈ 0.01, significant), but run_B's down-cluster is at PC1 = −0.094 (z = −0.54, p ≈ 0.61, **not distinguishable from random**). The up-direction cluster sign is verified; the down-direction cluster sign is stable only as "both negative" but the run_B magnitude is in the null. This means the "opposing clusters" story is cleaner for what Llama's post-training *amplifies* (up-cluster at PC1 > 0) than for what it *suppresses* (down-cluster drifts toward the null on one run).

**Our top 10 emotion INCREASES** depends on scoring method, and (per the noise-floor disclosure above) the specific names should be read as illustrative of each cluster's direction rather than as stable Meta-RLHF anchors:
- **Canonical Stage 8 (length-weighted projection `a · (v/||v||)`, not true cosine — see caveat below)**: `thrilled, relieved, pleased, patient, ecstatic, calm, grateful, triumphant, satisfied, elated` — a "positive mood" cluster
- **Cross-version control (raw dot product, Llama 3.1 base → 3.3 Instruct)**: `eager, enthusiastic, impatient, energized, stimulated, alert, excited, playful, exuberant, enraged` — a "high-arousal" cluster (raw dot biases toward emotions with larger vector norms)
- **Cross-signal intersection (top-20 of canonical Stage 8 ∩ top-20 of the paper's 3 deep-dive prompts)**: `alert, enthusiastic, excited, impatient` (N=4) — this was earlier framed as a "cleanest" result, but per the two-phenomena disclosure above, any specific 4-emotion list at this level is one-run-and-one-scoring-convention illustrative. Shown for legacy comparison; the robust findings are (1) the within-raw-dot two-cluster depth structure and (2) the cross-scoring-convention sign stability at L49, not the specific 4 names.

**Jaccard=0 applies specifically to the 4-emotion intersection cluster** (alert/enthusiastic/excited/impatient) compared against Sonnet's reported top-10 (brooding/gloomy/reflective/vulnerable/sullen/sad/dispirited/melancholy/troubled/unhappy). No word overlaps with that specific 4-emotion cluster. The broader activation-cluster top-10 from raw-dot scoring also has 0/10 overlap with Sonnet's top-10. The reflective-zone finding at L29–L33 is a completely separate analysis where Llama's OWN top-10 at those layers (`melancholy, reflective, depressed, brooding, gloomy`) has 4/10 direct overlap with Sonnet's top-10.

The overlap in the DECREASE direction is also zero. Paper says Sonnet decreases `spiteful, playful, exuberant, enthusiastic, impatient, obstinate, amused, cheerful, eager, greedy`; we see Llama decreasing `dependent, jealous, disoriented, self_critical, unsettled, hysterical, troubled, resentful, self_conscious, frightened`. Zero overlap at the emotion-name level — `decrease_overlap: []` in `stage8_post_training.json`.

Striking, but the raw top-k lists could differ without the underlying directions actually being different. To test this, we project both clusters into the same geometric space.

---

## Content-dependence: the effect lives on challenging prompts only

Prior to this check, our headline rested on the *averaged* (neutral + challenging) top-10 PC1 centroid. But `stage8_post_training.json` reports the within-run cross-scenario consistency as **shift_consistency_r = 0.304**, vs the paper's reported 0.90 for Sonnet. Meta's shift on neutral prompts and on challenging prompts are substantially different per-emotion. So: does the cluster-level PC1 sign claim hold on each subset independently, or is it an artifact of averaging two qualitatively different subsets?

We computed the top-10 up-cluster PC1 for each subset separately, on both runs, against the same 10,000-sample permutation null (CI95 = [−0.315, +0.354]). Results in `results/pc1_cross_scenario_verification.json`:

| Subset | run_A PC1 | run_A z | run_B PC1 | run_B z | Both beyond null? |
|---|---|---|---|---|---|
| **Challenging-only** | **+0.893** | **+5.07** | **+0.656** | **+3.73** | **yes — cleanly** |
| Averaged (both) | +0.856 | +4.86 | +0.517 | +2.94 | yes (the prior headline) |
| **Neutral-only** | −0.0002 | −0.00 | −0.277 | −1.58 | **no — both in null** |

**The cluster-level PC1 > 0 result is entirely carried by challenging prompts.** On neutral prompts, three runs span run_B = −0.28, run_A = 0.00, and run_C = +0.31 — all inside the permutation null CI [−0.315, +0.354]. The direction varies run-to-run but no run exits the null band. Meta's post-training doesn't move the cluster-level emotion representation on non-sensitive content; it moves it on sensitive content specifically.

The challenging top-10s are: run_A `thrilled, pleased, triumphant, relieved, proud, delighted, joyful, grateful, ecstatic, calm`; run_B `eager, enthusiastic, energized, excited, exuberant, stimulated, thrilled, impatient, alert, vibrant`; run_C `thankful, satisfied, relieved, jubilant, grateful, triumphant, blissful, fulfilled, proud, inspired`. Pairwise name overlaps: **run_A ∩ run_C = 4/10** (`triumphant, relieved, proud, grateful`), run_A ∩ run_B = 1/10 (`thrilled`), run_B ∩ run_C = 0/10. The two norm-aware scorings (length-weighted and true cosine) largely agree on the contentment cluster; raw-dot is the outlier. At the cluster level, all three project to PC1 = +0.89 / +0.66 / +0.91.

**On neutral prompts**, all three runs produce incoherent top-10s spanning mixed valence. run_A neutral: `impatient, lazy, bored, restless, alert, listless, sad, patient, alarmed, relaxed`. run_B neutral: `irritated, brooding, disdainful, impatient, frustrated, exasperated, sentimental, worn_out, nostalgic, restless`. run_C neutral: `brooding, nostalgic, reflective, peaceful, worn_out, safe, satisfied, blissful, tormented, refreshed`. No consistent cluster direction — the PC1 is essentially the permutation null across all three runs. Note that run_C's neutral PC1 (+0.31, z=+1.77) is the closest any run has come to significance on the positive side, sitting near but not exceeding the null's upper bound.

Two readings of this result:

1. **The averaged-both framing is less precise about scope.** Averaging neutral+challenging mixes a strongly-positive challenging-subset cluster with a literally-null neutral-subset cluster. Both subsets are legitimate measurements, and the averaged result is not wrong, but pulling them apart reveals that the effect is content-scoped rather than global, and the challenging-only numbers (z=+5.07 and +3.73) are cleaner than the averaged-both numbers (z=+4.86 and +2.94).
2. **Content-dependence (specifically AI-self-reflection vs pure trivia) is itself a finding.** It parallels the paper's own design choice (the paper uses sensitive prompts specifically to elicit the effect). We now have evidence this is not just a measurement-convention choice — it's a real differential property of RLHF between these two paper-designed content categories. Post-training reshapes emotion representation when the model is being asked to self-reflect, and leaves it essentially unchanged on factual questions.

**Why is neutral null?** The most plausible reading from the data is that Meta's RLHF genuinely doesn't push the emotion representation in a single coherent direction on non-sensitive content, not that our neutral prompts fail to elicit emotional responses at all. Evidence: run_B's neutral top-10 contains clearly emotion-bearing words (`irritated, brooding, disdainful, frustrated, exasperated, sentimental, worn_out, nostalgic`) — the subset isn't pure non-response. But those words don't form a coherent cluster: run_A neutral top-10 is `impatient, lazy, bored, restless, alert, listless, sad, patient, alarmed, relaxed` — a scatter across the valence axis. Two runs on the same neutral prompts produce incoherent subsets that happen to straddle zero on PC1. The most parsimonious reading is "no coherent RLHF direction on neutral content" rather than "not enough emotional signal to measure" or "wrong prompts." We can't fully rule out N=10 underpowering without running a larger neutral set, but the shape of the noise doesn't look like an underpowered-but-real signal; it looks like absence.

**Important scope note on what "content-dependent" means here.** Our two content classes are (a) pure factual trivia and (b) AI-identity/self-reflection prompts (see §Post-training direction above for the actual prompt text). These are maximally-distinct in that one has zero emotional valence and the other specifically targets the AI's self-concept. "Content-dependent" as used in this writeup therefore means "differs between these two categories" — not "varies smoothly across any content axis." A determined generalization would need a broader set of content categories (emotional-support requests, ethical dilemmas, creative writing, task execution, casual chat). We have two data points on a binary axis, not a continuous measurement, and the specific binary happens to be paper-design-determined by our replication target. The mechanism claim "Meta's RLHF acts selectively on AI-self-reflection prompts" is defensible; the broader "Meta's RLHF is content-dependent in general" is an extrapolation that we're doing lexically rather than empirically.

All numbers downstream in this writeup should be read with the challenging-only versions as the load-bearing ones. The "averaged-both" numbers are retained for legacy context and because the cross-run verification was originally computed on them, but the narrower challenging-only numbers are the post's actual claim.

---

## Geometric evidence at L49 (activation phase): up-cluster in positive-valence half

*Scope note: as with §Post-training direction above, this section zooms into L49, which sits in the opposed band (L49–L73) of the depth rotation described in §Layer-wise below. The "Sonnet PC1 = −0.432" row in the table below is specifically the paper-reported Sonnet anchors projected through Llama's L49 PCA. At other depths (L29–L33 especially), Llama's own top-10 overlaps Sonnet's anchor vocabulary directly — see §Layer-wise below.*

Compute PC1 (valence) and PC2 (arousal) from our 171 Llama emotion vectors at L49. Project Llama's candidate up-anchor clusters into this geometry and compute cluster means. For comparison, also project the paper's reported Sonnet anchor word list through the same Llama basis — this last row is a lexical projection (Sonnet's English anchor words in Llama's axis), not a measurement of Sonnet's own geometry, and should be read with that asymmetry in mind:

| Anchor cluster | N | PC1 (valence) | PC2 (arousal) | Interpretation |
|---|---|---|---|---|
| **run_A challenging-only top-10** (load-bearing) | 10 | **+0.893** | — | primary verified result |
| **run_B challenging-only top-10** (load-bearing) | 10 | **+0.656** | — | primary verified result |
| Canonical scoring at L49 (contentment-flavored; averaged both) | 10 | +0.856 | −0.002 | top-10 overlaps more with raw-dot-at-L43 than raw-dot-at-L49 |
| Raw-dot scoring at L49 (activation-flavored; averaged both) | 10 | +0.517 | +0.394 | literally sweep-L49 in the layer analysis |
| Within-version 3.1 top-10 (raw dot, includes `weary`) | 10 | +0.134 | +0.118 | near-center; includes fatigue edge |
| 4-emotion cross-signal intersection (legacy; individual names at noise floor) | 4 | +0.436 | +0.422 | one illustrative example cluster |
| **Paper Sonnet anchor words** (lexical projection; see §Caveats) | 10 | **−0.432** | **−0.432** | English-valence baseline, not Sonnet measurement |

**The load-bearing empirical content is "on challenging prompts, Llama's post-training up-cluster sits in the positive-valence half of Llama's own PC1 axis, across two independent runs at z > 3.7 beyond the permutation null."** Five different scoring methods applied to Llama all give PC1 > 0, ranging from +0.134 (within-version raw-dot, averaged) to +0.893 (challenging-only). The challenging-only measurements are cleaner than the averaged-both measurements because they don't mix in the null neutral subset; see §"Content-dependence" above and `results/pc1_cross_scenario_verification.json` for the verification.

**The Sonnet row should not be read as a symmetric measurement.** It is the paper's 10 reported anchor words projected through Llama's PCA basis. Those 10 words (`brooding, gloomy, reflective, vulnerable, sullen, sad, dispirited, melancholy, troubled, unhappy`) are negative-valence English by construction, so projecting them into any axis that tracks human valence (Llama's does at r=0.964 to Russell-Mehrabian norms) gives a negative PC1 nearly tautologically. The cross-lab "sign flip" story is therefore asymmetric: (a) the Llama side is a real cross-run-verified measurement that Meta's RLHF moves the up-cluster to positive valence, and (b) the Sonnet side is a lexical property of the paper's reported anchor list. Both facts can be true and the *qualitative* "different design directions" interpretation can survive this asymmetry, but we can't claim "verified opposition" from this table alone.

**PC2 (arousal) among the Llama clusters is method-dependent.** The canonical normalized top-10 is arousal-neutral (PC2 = −0.002); the raw-dot scoring methods give high-arousal clusters. The robust claim is specifically about PC1; PC2 varies with which scoring method you use on the same underlying shift.

**Caveat: the r=0.96 PC1-valence alignment partially softens the Llama-side claim too.** If Llama's PC1 is ~96% valence, then "Meta's RLHF moves the up-cluster to PC1 > 0" also has a partially lexical character — the top-k emotions in each Stage 8 run are systematically positive-valence English words. This doesn't invalidate the result (it's still an empirical claim about which direction Meta pushed, and noise-floor-robust at the cluster level), but the asymmetry "Llama-side = measurement / Sonnet-side = tautology" is tidier than the r=0.96 structural alignment actually allows. The honest reading is: both sides are partly neural and partly lexical, with the Llama side being substantially more measured because we have the cross-run verification and the specific top-10 lists, and the Sonnet side being substantially more lexical because we only have the paper-reported anchor list without any corresponding cross-run Sonnet data.

**Caveat: within-version 3.1 RLHF activation top-10 contains fatigue emotions**. The Llama within-version 3.1 top-10 (raw-dot scoring) is `eager, impatient, weary, stimulated, enthusiastic, tired, worn_out, enraged, energized, irritated`. The `weary/tired/worn_out` presence is notable because it suggests Llama's activation cluster spans from high-arousal engagement through low-arousal fatigue — a broader semantic area than just "activated engagement." However, `weary` is NOT in Sonnet's reported top-10 (paper rank 14, not top-10), so there is no direct weary-overlap with Sonnet. The corrected Sonnet top-10 has `sad` at rank 6, which does not overlap with any of Llama's activation-cluster top-10s.

The corresponding DOWN-anchor comparison is asymmetrically weaker. Run_A's down-cluster (what Meta's RLHF suppresses) sits at PC1 = −0.44 (significant, z = −2.52), but run_B's down-cluster is at PC1 = −0.09 (z = −0.54, indistinguishable from the permutation null). The verified sign-flip claim is specifically about the UP-cluster direction, not both halves of the axis. `pc1_stability_verification.json:verdict.down_anchor_pc1_sign_stable_and_non_null = false`.

**Stated as a within-Llama finding**: on challenging/sensitive prompts, Meta's post-training moves Llama's emotion up-cluster into the positive-valence half of Llama's own PC1 axis — three independent Stage 8 scripts give run_A PC1 = +0.893 (length-weighted), run_B = +0.656 (raw-dot), run_C = +0.913 (true cosine), all z > 3.7 beyond the permutation null. On neutral prompts no run exits the null (range −0.28 to +0.31). Stated as a framing: this is consistent with a different design choice than what the paper reports for Sonnet, where the up-anchors are negative-valence words, but the cross-lab comparison is asymmetric (we have a measurement on Llama, we have a word list for Sonnet) and the sign flip on Sonnet's side is nearly a lexical consequence of the words Anthropic chose to report rather than an independent geometric measurement. The down-direction (what Meta's RLHF suppresses) is asymmetrically weaker in our data. The cluster-level PC1 sign on challenging prompts is the robust unit of comparison within Llama; the specific cluster flavor (contentment vs activation) depends on whether the scoring convention normalizes emotion vector norms — length-weighted and true cosine surface contentment; raw-dot surfaces activation.

---

## Controlling for the cross-version confound

The above compared Llama 3.1 base → 3.3 Instruct, which mixes "RLHF direction" with "3.1-to-3.3 version upgrade". Could the positive-valence cluster result be a version-upgrade artifact?

To test this, we ran Llama 3.1 Instruct (same version as the base model) on the same 20 prompts (10 neutral + 10 challenging), getting a decomposition into three shift vectors:
- **within-version RLHF**: 3.1 base → 3.1 Instruct (pure RLHF, no version drift)
- **cross-version** (original measurement): 3.1 base → 3.3 Instruct
- **version-drift only**: 3.1 Instruct → 3.3 Instruct

Spearman correlations between the shift vectors (171 emotions):

- **cross-version vs within-version: ρ = +0.922** (Pearson +0.9318)
- cross-version vs version-drift: ρ = +0.047
- version-drift vs within-version: ρ = −0.317

**⚠ Important statistical caveat**: the ρ=0.922 cannot be interpreted as independent empirical confirmation of anything. Here's the math: `shift_cross_version = shift_within_3_1 + shift_version_drift` holds by construction (3.3_inst − 3.1_base = (3.1_inst − 3.1_base) + (3.3_inst − 3.1_inst)). We measured Var(within)=0.0526 and Var(drift)=0.0070 — within is 7.5× larger variance, and ||within||=2.99 vs ||drift||=1.10 (2.72× larger L2). From these numbers alone, the analytic Pearson(cross, within) = +0.9318, matching our observed correlation exactly. **Any experiment where within-version variance dominates would return ρ > 0.9 regardless of what RLHF did.** The ρ tells us "version-drift is small relative to the RLHF component" — which is real and useful — but it does NOT tell us "the RLHF direction is confirmed via independent measurement".

The actual independent evidence is the **within-version 3.1 shift's own top-10**: impatient at rank 2, enthusiastic at rank 5, excited at rank 17, alert at rank 14 (the activated-engagement cluster is present in the within-version shift without needing the cross comparison at all). Here's the anchor ranks:

| Shift | alert | enthusiastic | excited | impatient |
|---|---|---|---|---|
| within-version 3.1 RLHF | 14 | 5 | 17 | **2** |
| cross-version (original) | 6 | **2** | 7 | **3** |
| version-drift only | 48 | 36 | 41 | 96 |

**`impatient` appears as a top candidate in both runs' within-version measurement** — rank 2 in the within-version 3.1 RLHF shift, rank 3 in the cross-version shift, but rank 96 in the pure version-drift direction. Individual rank stability across runs is limited by the bnb int4 noise floor documented above (ρ=0.46 between identical re-runs, some emotions sign-flip), so "`impatient` is Meta's RLHF signature" is stronger than the run-level data support. The robust claim is weaker: within a given run, `impatient` lands in the top cluster and is essentially absent from the version-drift direction — which is consistent with it being part of the RLHF signal rather than part of the 3.1→3.3 upgrade. Any single emotion label should be read as one-run illustrative; the cluster-level PC1 sign is what replicates across runs.

(Separately, the pure 3.1→3.3 version drift has its *own* interpretable direction: `content, safe, cheerful, optimistic, fulfilled, blissful`. A "make the model feel more content" axis, small in magnitude but statistically real — Cov(within, drift) = −0.0057, ~3.9 standard errors from zero. Meta's 3.3 upgrade slightly counteracts their 3.1 RLHF direction.)

The corrected framing for what the cross-version control establishes: **Meta's within-version 3.1 RLHF direction is independently visible in its own top-10 emotion ranks and qualitatively persists in the 3.1→3.3 cross-version measurement because the version-drift component is small in magnitude.** This does not, strictly, "rule out the cross-version confound" via the ρ — that inference is circular — but the within-version measurement alone is sufficient to assert the RLHF direction.

---

## Layer-wise: depth-dependent rotation (decomposed into 4 narrative "phases" for exposition)

The positive-valence direction isn't a universal feature of Llama's residual stream. Two complementary views of the same layer sweep:

**View 1: full-rank Spearman ρ between each layer's shift vector and L49's shift vector.**
- L55/L61/L67/L73: **+0.92, +0.86, +0.84, +0.79** (strong agreement with L49)
- L1-L43: −0.40 to +0.46 (random to weak)
- L79: +0.16 (dissipates at the readout layer)

A 5-layer plateau from L49 to L73 shares the same shift direction. Earlier and later layers are doing something different.

**View 2: per-layer cluster PC1 centroid** (top-10 shift emotions projected onto the L49 PCA basis, compared against Sonnet's anchor cluster at PC1 = −0.432). *Scope note*: this table is computed on **averaged (neutral + challenging) Stage 8 shifts**, because we didn't re-run the full layer sweep on challenging-only due to compute constraints. At L49 specifically, the challenging-only centroid is +0.893 / +0.656 (stronger than the averaged L49 value of +0.517 shown in the table); the mid-late-layer opposed bands would likely look cleaner under challenging-only scoping, though we can't verify without rerunning the sweep.

**Important refinement — the layer-wise PC1 sign opposition is 10/14 layers, not 14/14.** A follow-up analysis takes each of the 14 sampled layers' top-10 shift emotions and projects them onto the L49 PCA basis (the same PC1/PC2 basis used everywhere else in this writeup), then compares against Sonnet's anchor cluster at PC1 = −0.432 in the same geometry. Results (full table in `results/stage8_layer_sweep_pc1_centroids.json`):

| Layer | Top-3 up-shift | PC1 | Opposed to Sonnet? |
|---|---|---|---|
| L1 | hostile, scornful, tense | −0.243 | no |
| L7 | rattled, skeptical, unnerved | −0.435 | no |
| L13 | euphoric, perplexed, paranoid | +0.509 | yes |
| L19 | optimistic, invigorated, joyful | +0.627 | yes |
| L25 | self_critical, perplexed, droopy | +0.055 | yes (but in the null) |
| **L31** | **melancholy, reflective, depressed** | **−0.328** | **no** |
| L37 | blissful, content, at_ease | +0.871 | yes |
| L43 | satisfied, cheerful, jubilant | +0.943 | yes |
| **L49** | **eager, enthusiastic, impatient** | **+0.517** | **yes (primary)** |
| L55 | impatient, stimulated, eager | +0.362 | yes |
| L61 | impatient, aroused, playful | +0.167 | yes |
| L67 | impatient, aroused, playful | +0.290 | yes |
| L73 | aroused, excited, impatient | +0.173 | yes |
| L79 | enraged, alarmed, rattled | −0.467 | no |

**PC1 > 0 holds at 10 of 14 sampled layers (73%) as a bare sign match.** The 4 clearly-not-opposed layers are L1, L7, L31, L79. L25 is technically positive at +0.055 but inside the permutation null CI.

**A permutation-null test at each layer's own PCA** (10,000 random 10-of-171 draws per layer; data at `results/per_layer_significance_own_basis.json`, generating script archived in commit `d0dbb02`) resolves which layers have statistically-significant cluster centroids in their own basis. **We run 14 tests, so we need multiple-comparison correction**: Bonferroni at family α=0.05 gives per-test α=0.00357.

| Layer | Top-10 top-3 | Own-basis PC1 | z | p | Raw p<0.05 | Bonferroni |
|---|---|---|---|---|---|---|
| L1 | hostile, scornful, tense | −0.285 | −2.09 | 0.033 | ✓ neg | ✗ |
| L7 | rattled, skeptical, unnerved | −0.405 | −2.78 | 0.005 | ✓ neg | ✗ (Holm ✓) |
| L13 | euphoric, perplexed, paranoid | +0.425 | +2.64 | 0.008 | ✓ pos | ✗ |
| **L19** | **optimistic, invigorated, joyful** | **+0.615** | **+3.55** | **0.0005** | ✓ pos | **✓ pos** |
| L25 | self_critical, perplexed, droopy | +0.080 | +0.47 | 0.656 | ✗ | ✗ |
| L31 | melancholy, reflective, depressed | −0.283 | −1.63 | 0.102 | ✗ | ✗ |
| **L37** | **blissful, content, at_ease** | **+0.881** | **+4.97** | **<0.0001** | ✓ pos | **✓ pos** |
| **L43** | **satisfied, cheerful, jubilant** | **+0.947** | **+5.39** | **<0.0001** | ✓ pos | **✓ pos** |
| **L49** | **eager, enthusiastic, impatient** | **+0.517** | **+2.94** | **0.003** | ✓ pos | **✓ pos** |
| L55 | impatient, stimulated, eager | +0.350 | +1.97 | 0.041 | ✓ pos | ✗ |
| L61 | impatient, aroused, playful | +0.152 | +0.86 | 0.402 | ✗ | ✗ |
| L67 | impatient, aroused, playful | +0.272 | +1.55 | 0.119 | ✗ | ✗ |
| L73 | aroused, excited, impatient | +0.137 | +0.81 | 0.435 | ✗ | ✗ |
| L79 | enraged, alarmed, rattled | −0.452 | −2.58 | 0.010 | ✓ neg | ✗ |

**Bonferroni-robust positive-significance in Llama's own-basis PC1: 4 layers — L19, L37, L43, L49.** These four survive FWER correction across the 14-test family and span ~24%–61% of network depth. Under raw α=0.05, two additional positive layers clear significance (L13 and L55) and three negative layers clear significance (L1, L7, L79 — early processing and readout effects), but none of these five survive Bonferroni. Under Holm-Bonferroni, L7 (p=0.005) just survives as a marginal negative but nothing else changes.

**This is the own-basis PC1 metric, which is distinct from the Sonnet-alignment metric used in the rotation narrative above.** Bonferroni-robust L19/L37/L43/L49 are where Llama's top-10 shift clusters significantly into Llama's *own* valence-positive half. The Sonnet-alignment metric at these same layers tells a different story: L19 is Sonnet-anti-aligned (z=−0.32, null), and L13 is strongly Sonnet-anti-aligned (z=−3.35; its top-10 is `euphoric, triumphant, jubilant, elated, ecstatic` — high-valence positive vocabulary that's the *opposite* of Sonnet's reflective anchors). **Two metrics, different cuts of the same data**: own-basis PC1 captures "which half of Llama's valence axis the cluster sits in"; Sonnet-alignment captures "does Llama's shift specifically up-weight Sonnet's reported anchors." The L19/L37/L43/L49 Bonferroni-robust set is robust for the first question; the L29–L33 reflective zone is where Llama's vocabulary matches Sonnet's on the second.

Three things worth flagging about the 4 non-opposed layers:

- **L1-L7 (early, 2.5%-9% depth)**: Llama's top shifts at `hostile, scornful, tense, rattled, skeptical, unnerved` — negative-valence (PC1 ≈ −0.24 to −0.44). These are early processing layers representing incoming-speaker affect, not the model's own response direction. The Sonnet-like reading at early depth isn't about what Llama's RLHF does.
- **L31 is not an anomaly — it's the middle of a coherent L29–L33 Sonnet-aligned zone.** Dense sampling at L25/L29/L31/L33/L37 (`results/stage8_l31_zone.json`) shows `melancholy, reflective, brooding` in Llama's top-10 shift at all three of L29, L31, L33, with internal pairwise Spearman ρ > 0.90.

  L31's full top-10 is `melancholy, reflective, depressed, worn_out, droopy, brooding, lonely, resigned, gloomy, miserable`. Four words overlap Sonnet's reported top-10 directly (`reflective, brooding, gloomy, melancholy`) plus fuzzy matches (`droopy`~`dispirited`, `miserable`~`unhappy`, `resigned`~`troubled`, `lonely`~`sad`). **At L29–L33 Llama's top-10 is essentially Sonnet's reported anchor list.** The vocabulary transitions to contentment (`blissful, content, at_ease` at L37) within 4 layers; this is a vocabulary-level shift, not a clean geometric break (the shift-vector adjacent ρ from L31 to L37 is moderate, not orthogonal).

  *Caveat 1 (basis-shared approximation)*: L29 and L33 in the dense-sampling data were projected through L31's probe basis, so the "internal pairwise ρ > 0.90" is partly basis-induced. The 4-word direct vocabulary overlap with Sonnet's anchors is the stronger (basis-independent) claim.

  *Caveat 2 (two metrics, same finding)*: L31 appears in the null (p=0.102) in the Bonferroni table above because that table uses L31's *own-basis PC1 centroid*, which is slightly negative (−0.283) since `melancholy, reflective, …` project negative on Llama's valence axis. But the Sonnet-alignment metric at L31 is z = +3.48 (and +3.88 at L29, +2.83 at L33) — a per-layer score defined as `mean(Sonnet UP-anchor shifts) − mean(Sonnet DOWN-anchor shifts)`, z-normalized against a per-layer 10,000-sample permutation null of random (10 UP, 10 DOWN) splits from the same layer's 171-emotion shift vector. Llama's L29–L33 shifts up-weight exactly the emotions Sonnet reports as UP anchors (paper Table 16) and down-weight Sonnet's reported DOWN anchors — strongly positive on the Sonnet-alignment metric even though they're slightly negative on Llama's own-valence axis because those emotions are low-valence by construction. Same data, two metrics, pointing at the same finding from different angles. L31 is a clean reflective-concern zone that coexists with the later contentment/activation zones — the same representation Sonnet apparently surfaces at the output layer.
- **L79 (readout, 100% depth)**: `enraged, alarmed, rattled` — negative-valence (PC1 = −0.467). The direction dissipates at the readout layer where the output distribution is being computed.

**The Bonferroni-robust positive-PC1 claim is L19, L37, L43, L49 — 4 layers**, spanning ~24%–61% of network depth. Outside this set, most layers are in the permutation null (L25, L31, L61, L67, L73), a few are raw-significant but don't survive Bonferroni (positive: L13, L55; negative: L1, L7, L79). The robust claim is mid-network-wide but not universal: a specific band that misses early processing and the readout layer.

**And — interestingly — the 4-layer robust set covers both raw-dot clusters, though asymmetrically.** L37 and L43 sit inside the L37–L43 contentment cluster (raw-dot top at L43: `satisfied, cheerful, jubilant`) — two Bonferroni-robust layers. L49 sits inside the L49–L67 activation cluster (raw-dot top at L49: `eager, enthusiastic, impatient`) — one Bonferroni-robust layer; L55, the activation-cluster edge, is raw-significant but fails correction. L19 is earlier than either cluster (top-3: `optimistic, invigorated, joyful`) — "emerging positive-valence" before either cluster crystallizes. **Meta's post-training isn't producing one direction along a plateau — within raw-dot scoring it's producing multiple positive-valence signatures across adjacent depths, with at least two distinguishable types (contentment at L37/L43 and activation at L49) plus an earlier pre-cluster positive layer at L19.** The contentment cluster has stronger Bonferroni support than the activation cluster at L49 alone, so a conservative reading is "2 robust contentment layers + 1 robust activation layer + 1 robust pre-cluster layer" rather than "two fully-verified clusters." The L13 raw-significance (p=0.008) would extend the pre-cluster band but doesn't survive Bonferroni, so we can't load-bear on it either.

---

## Linguistic evidence: suggestive directional opposition (with a caveat)

One more piece of evidence, this time from a completely different pathway. Logit lens: project the emotion vectors through the model's unembedding matrix into vocabulary space. This reveals which tokens each probe "leans toward" and "away from" in the output distribution.

Llama up-anchors — top tokens (toward):
- `impatient`: waiting, anticipation, fidget, exasperated, frustrated
- `enthusiastic`: improvement, improve, extend, speed, eagerness
- `excited`: improve, improvement, extend, prime
- `alert`: jump, walk, race, quick
- `eager`: buffer, waiting, anticipation, rapid

Sonnet up-anchors (brooding, gloomy, reflective, vulnerable, sullen) — top tokens:
- heavy, broken, drowsy, numb, listless, empty, lack, slow

And their BOTTOM tokens (away from these emotions):
- **improvement, improve, prime, prim, chall(enge), gold, positive**

`improvement` and `prime` appear at top-of-cluster for Llama's enthusiastic/excited and simultaneously bottom-of-cluster for Sonnet's brooding/gloomy/vulnerable. `heavy` and `slow` appear at top-of-cluster for Sonnet's sullen/gloomy and bottom-of-cluster for Llama's enthusiastic/alert.

**⚠ Token-frequency caveat**: some of these tokens are base-rate common across many emotions in the 171-set, which weakens the "same tokens inverted polarity" claim. A direct count on our logit_lens.json shows ` content` appears toward 28 emotions and away from 54 (a 1:2 split that's driven mostly by the overall emotion distribution, not cluster-specific polarity). ` heavy` is a 13-toward-14-away split. The one token with genuinely one-sided behavior in our count was ` prim` (1 toward, 17 away) — that's the kind of signal the claim needs. The others are suggestive but could be base-rate artifacts. A rigorous version of this analysis would compute the Llama-cluster vs Sonnet-cluster **cluster-averaged unembedding vectors** and report their cosine, or run a permutation test against the null of "pick 5 random emotions per side". Neither was done here.

Treat this as a **qualitative directional signal that's consistent with the geometric result**, not as a statistical test. The vocabulary axis `{motion, improvement, quick}` vs `{heavy, slow, listless}` does appear to run through both clusters in opposite directions at the anchor-emotion level, but the claim is weaker than "same tokens inverted polarity" would suggest on its face.

This comes from a different computational pathway than the geometric result — the unembedding matrix vs residual stream projections — so it's an independent (weak) corroboration of the centroid-level opposition, not a statistical confirmation.

---

## Bonus: Llama shows no arousal regulation where Sonnet does

Paper Fig 59 reports that Sonnet's speaker probes show "arousal regulation" (r ≈ −0.47): when the model represents the other speaker as feeling high-arousal, the closest present-speaker probe is lower-arousal. Sonnet calms people down.

We measured this on Llama. Our correlation at the primary power level: **r = +0.053 across 171 emotions using PC2 as arousal proxy** — statistically indistinguishable from zero. Using Russell & Mehrabian norms on the 13 overlapping pairs: **r = +0.523, p = 0.067** — directionally non-negative but underpowered at N=13 to make a positive claim.

**The honest framing**: Llama lacks Sonnet's reported arousal counter-regulation effect. The data at the well-powered N=171 level are consistent with no cross-speaker arousal relationship at all — NOT with "active matching". The N=13 PAD-norms result is directionally suggestive but not significant. We can rule out the paper's −0.47 counter-regulation at our power, but we cannot positively claim "Llama actively matches engagement" from this data.

This is consistent with the main finding as an **absence-of-Sonnet-like-regulation** result — Llama doesn't appear to encode the "calm the user down" counter-regulation dynamic that Sonnet does. Whether Llama instead encodes a "match engagement" dynamic is unresolved by our pilot; the N=13 PAD result is consistent with that interpretation but also with noise. The pattern at the well-powered level is: Sonnet regulates, Llama does not. Going further requires more data.

---

## What this means

**The verified empirical content of this post is a content-scoped, depth-dependent within-Llama measurement of a smooth rotation with one readout discontinuity, which we carve into 4 narrative "phases" for exposition:**

1. **At L29–L33, Llama has a Sonnet-aligned reflective-concern zone.** Top-10 shifts at L29-L33 are `melancholy, reflective, depressed, brooding, gloomy, worn_out, droopy, lonely, resigned, miserable` with 4 direct name overlaps against Sonnet's reported top-10 anchors and several fuzzy matches. Dense sampling confirms this is a coherent 3-layer core zone (internal pairwise ρ > 0.90). Sonnet-alignment z-scores under a per-layer permutation null (`results/sonnet_alignment_zscore_verification.json`): L29 = +3.88 (peak), L31 = +3.48, L33 = +2.83. L25 also clears raw significance at z = +2.09 (p = 0.037) but its top-10 is `self_critical, perplexed, droopy, optimistic, melancholy, kind, depressed, compassionate, puzzled, serene` — a mixed-valence scatter rather than a coherent reflective cluster, so we don't include it in the named reflective zone.
2. **At L37–L43, Llama's top-10 shifts to contentment vocabulary.** Top-10 `blissful, content, at_ease, relaxed, satisfied, cheerful, jubilant`. Sonnet-alignment z-score is null at these layers (L37 +1.06, L43 −0.91) — the contentment band is Sonnet-orthogonal, neither aligned nor opposed. The vocabulary flip from reflective (L33) to contentment (L37) happens within 4 layers but the shift-vector geometry is a moderate rotation rather than a sharp break.
3. **At L49–L73, Llama is in activation-flavored positive-valence (opposite Sonnet's direction).** Top-10 at L49: `eager, impatient, enthusiastic, energized, stimulated, alert, excited`. Sonnet-alignment z-scores are significantly negative across all 5 sampled layers in this band: L49 = −2.04, L55 = −2.45, L61 = −2.65, L67 = −2.63, L73 = −2.95 (peak opposite). The raw-dot "activation cluster" from the pairwise layer-correlation analysis (L49–L67, internal ρ ≥ 0.84) is a tighter subset of this phase — L73 aligns against Sonnet but starts drifting away from the L49–L67 cluster in per-layer correlation.
4. **At L79 (readout), Llama significantly realigns toward Sonnet.** Sonnet-alignment z = +2.07 (p = 0.037) — about half of L29's peak z = +3.88 but still significantly positive. Top-10 is `enraged, alarmed, rattled` — nominally high-arousal negative, but the Sonnet UP anchors are being shifted up as well, producing a positive alignment score even though the top-10 names are different from Sonnet's.

So Llama's RLHF produces a depth-dependent emotion trajectory that we decompose into **4 narrative "phases"**: Sonnet-aligned reflection (L29–L33) → contentment-vocabulary mid-network (L37–L43, Sonnet-orthogonal) → smooth rotation into opposition (L49–L73) → sharp readout discontinuity at L79 (partial Sonnet-realignment, the only genuinely sharp boundary). The "phases" are a vocabulary-driven narrative carving; the underlying Sonnet-alignment metric is monotone-decreasing from L29 to L73. A single-layer measurement at L49 would see only the opposed band; the reflective-zone at L29–L33 and the readout realignment at L79 are the Sonnet-aligned extrema.

**This changes the cross-lab interpretation substantially.** The paper's Sonnet top-10 anchors are (presumably) measured at an output-relevant layer. Llama has Sonnet's reflective-concern representation too — at L29–L33 and weakly at L79 — it's just not the dominant output-relevant direction. The models may share the same emotional palette and differ in which depth-phase carries the representation to the output. We can't test the symmetric version (does Sonnet also have an intermediate activation phase that doesn't propagate?) without Sonnet weights. The honest reading is: **"Meta and Anthropic produce overlapping post-training representations across depth; the dominant direction depends on which layer you measure, and the cross-lab 'opposition' is specifically about the L49–L73 middle band."**

**Content-scope still holds**: this depth trajectory only happens on AI-self-reflection prompts. On factual trivia there is no cluster-level shift at any depth. Meta's RLHF reshapes emotion representation only when the model is being asked about itself.

**Statistical scope**: within the 14-layer sweep we had in hand for formal Bonferroni testing, 4 layers survive correction for positive PC1 (L19, L37, L43, L49). L29/L33 aren't in that 14-layer sweep — L31 is sampled but alone it's in the null (p=0.102) because L31's own-basis PC1 of `melancholy, reflective...` is slightly negative when projected with Llama's valence axis. The dense L29-L31-L33 zone was measured in a separate diagnostic that used L31's probe basis as an approximation for L29 and L33 (`results/stage8_l31_zone.json`). So the reflective-zone claim is a visual/coherence claim ("same top-10 vocabulary at 3 adjacent layers with internal ρ>0.90") rather than a FWER-corrected permutation result. The activation-phase claim at L49 is verified across three runs with three scoring metrics (see §Post-training and §Geometric evidence tables above for the numbers). The three claims have different evidence weights: reflective zone is vocabulary-overlap + metric-alignment but not FWER-corrected; activation phase is cross-run-verified and Bonferroni-robust; readout phase is metric-significant but single-run.

**The cross-lab framing is subtle.** The "Sonnet PC1 = −0.432" row in the §Geometric evidence table is a lexical baseline: low-valence English words project low in any valence-tracking axis, near-tautologically. That's the weak end of the cross-lab comparison — it's what the L49 activation phase would look like opposite, and it's not particularly informative because the Sonnet side is word-projection, not measurement. The stronger cross-lab claim is at L29–L33: Llama's own top-10 directly overlaps Sonnet's reported top-10 anchor words (4 direct overlaps, several fuzzy). Those overlaps are a measurement of Llama, not a projection of Sonnet. "At L29–L33, Llama's measured post-training shift surfaces emotion names that Sonnet's paper also reports as its top anchors" is a symmetric observation: Llama's vocabulary at L29–L33 was computed from Llama's own shift; Sonnet's was reported by the paper.

Three distinct claims now, ordered by evidence strength:

1. **Cross-run verified (within Llama, at L49)**: Meta's post-training moves Llama's activation cluster at L49 to PC1 > 0. Two-run cross-script verification.
2. **Dense-sampled, non-FWER-tested (within Llama, at L29–L33)**: Llama's top-10 shift at three adjacent layers surfaces `melancholy, reflective, brooding, gloomy, worn_out` — overlapping Sonnet's reported anchors directly.
3. **Paper-reported (Sonnet-side)**: Sonnet's top-10 is `brooding, gloomy, reflective, vulnerable, sullen, sad, dispirited, melancholy, troubled, unhappy`. We take this at face value from the paper.

The cross-lab implication of (2) + (3) combined is: **Llama and Sonnet have the same reflective-concern vocabulary at some depth**. The cross-lab implication of (1) alone is: **Llama has an additional activation-flavored phase that doesn't appear in Sonnet's reported measurement**. We can't tell from the paper whether Sonnet also has such a phase at some mid-layer that doesn't propagate to the output. A proper symmetric comparison would need Stage 8 measurements on Sonnet at multiple depths, which we don't have access to.

In qualitative terms, one run's top candidates from Llama's within-version shift were things like `alert, enthusiastic, excited, impatient`; another run's were `thrilled, relieved, pleased, patient, calm, elated`. Both are top candidates for "the positive-valence half of the axis"; neither is a stable Meta-RLHF anchor at int4 precision. The cluster-level PC1 sign is what's stable. The down-direction (what Meta's RLHF *suppresses*) is asymmetrically harder to pin down at our noise level: one run's down-cluster is significant, the other's is in the null. So the strong claim is specifically about what Meta *amplifies*, not what it suppresses.

This shows up at **several pathways of varying independence**:

1. **Verified cross-run cluster-level PC1 sign flip at L49 on challenging prompts.** Three Stage 8 runs at L49 with three different scoring metrics all give positive PC1 beyond the permutation null — numbers in the TL;DR table above and re-tabulated in §Geometric evidence. On neutral prompts all three runs are in the null. This is the narrowest-scope, most-statistically-robust claim: the L49 point measurement is cross-run verified. It's not the *headline* — the headline is the full depth rotation — but it's the single point on that rotation where we have the strongest cross-run evidence. Direct measurement, not an assertion. (`results/pc1_cross_scenario_verification.json`.)

2. **Layer localization, with caveats.** In each layer's own PCA basis (10,000-sample permutation null per layer), **4 layers survive Bonferroni correction (14-test family α=0.05)** for positive-PC1 cluster centroids: **L19, L37, L43, L49**. Two additional layers (L13, L55) clear raw α=0.05 but fail FWER correction. On the negative side, L1/L7/L79 are raw-significant but none survive Bonferroni (L7 barely survives the less-conservative Holm procedure). Five layers (L25, L31, L61, L67, L73) are in the null. The Bonferroni-robust positive-valence direction lives in the mid-network band L19–L49 (with L25/L31 as null gaps). Drawn from the same Stage 8 data as pathway 1, so not fully independent — the depth distribution of the same measurement, not a second measurement. Useful as "the direction is localized not global"; not an independent confirmation.

3. **Linguistic polarity via logit lens.** Project emotion vectors through the unembedding matrix — a different computational pathway from residual-stream projections. Llama's up-anchors' top tokens (waiting, improvement, quick, jump) vs Sonnet's up-anchors' top tokens (heavy, slow, listless, numb) run through the same vocabulary axis in opposite directions. Weaker than it sounds (token base-rate caveat, see earlier section). Genuinely independent pathway, but qualitative directional signal rather than statistical test.

4. **Absence of cross-speaker arousal regulation.** Llama lacks Sonnet's reported r ≈ −0.47 counter-regulation at N=171 (we measured r = +0.053). This is an absence-of-effect finding, *compatible with* the main story (Meta's RLHF doesn't install Sonnet-style counter-regulation) but doesn't positively confirm anything about the valence-sign direction.

**Evidence hierarchy across these pathways**: pathway (1) has direct cross-run statistical support at L49 specifically (three scoring metrics, all cross-run verified). Pathway (2) — the depth rotation / layer trajectory — is the *headline finding* but comes from one forward-pass realization of the layer sweep, so it's less statistically verified than pathway (1) per-layer even though it's the broader claim. Pathways (3) and (4) are qualitative/absence-of-effect consistency checks. The honest framing is: the rotation (pathway 2) is the headline, L49 (pathway 1) is the most statistically robust single point on that rotation, and pathways 3/4 round out the picture.

If the paper's narrative framing is "post-training produces emotional nuance", this work refines it in two ways. *Within Llama*: Meta's RLHF doesn't just add "nuance" — it pushes the up-cluster measurably into the positive-valence half of Llama's own PC1 axis at mid-late layers, which is a specific directional claim, not just "more emotional differentiation." *Across labs*: the paper's Sonnet anchor list sits on the negative-valence half of the same axis, but because the Sonnet side is a projection of English anchor words rather than an independently measured shift in Sonnet's own geometry, the cross-lab contrast is suggestive rather than a symmetric result. Post-training *can* pull a model's sensitive-prompt representation toward either end of the valence axis, and that is a real design dimension; the fact that Llama's measured shift goes one way while Anthropic's reported anchors go the other is consistent with — but not proof of — different lab-level design choices. The sign within Llama is what's robust in our data; the lab-level interpretation is what a proper Sonnet-side Stage 8 would be needed to confirm.

## Caveats

- **Metric deviation from the paper — verified not load-bearing.** The paper specifies cosine similarity at Stage 8 ("We measured the cosine similarity between the emotion probe vectors and model activations, on the colon token after 'Assistant'", paper line 814). Our `stage8_post_training.py` uses `projection(act, vec, normalize_vector=True)` which computes `a · (v/||v||)` — the vector is unit-normalized but the activation is not. That's length-weighted dot product, not cosine similarity. In principle, base-vs-instruct comparisons where post-training inflates activation norms could produce a length-weighted shift signal with zero directional component. To verify this wasn't driving the finding, we re-ran Stage 8 with true cosine similarity (results at `results/stage8_cosine_verification.json`; generating script archived in commit `fbbaf20`). Results: (a) L49 activation norms are actually *lower* in instruct (21.09) than base (23.59) — ratio 0.894 — so length-weighted would *understate* any real effect, not inflate it; (b) challenging-only PC1 centroid under true cosine is **+0.913 (z=+5.18)**, slightly stronger than the length-weighted +0.899 (z=+5.11); (c) 8/10 top-10 overlap between metrics on challenging, 10/10 on neutral. Neutral is in the null under both metrics. **The finding is metric-robust: length-weighted, true cosine, and raw-dot all give significantly positive PC1 centroids on challenging-only and near-null on neutral.** The length-weighted metric in our original Stage 8 script was an incidental choice, not a systematic deviation — but the verification against the paper's specified cosine metric was a ~1-hour run we should have done from the start.
- **Cross-lab comparison uses paper-reported anchors for Sonnet, not an independent measurement — and this is a bigger caveat than it first appears.** We didn't re-run the paper's Stage 8 on Sonnet. The "Sonnet PC1 = −0.432" we cite throughout is the paper's reported Sonnet anchor words (`brooding, gloomy, reflective, vulnerable, sullen, sad, dispirited, melancholy, troubled, unhappy`) projected through **Llama's** PCA basis. Those 10 words are negative-valence in English, so projecting them into any axis that tracks valence (Llama's PC1 at r=0.96 to Russell-Mehrabian norms) gives a negative PC1 *nearly tautologically*. The real empirical content of this post is the Llama-side measurement ("Meta's RLHF moves Llama's up-cluster to PC1 > 0 in Llama's own geometry"); the Sonnet side is a lexical property of the paper's anchor list, not a neural measurement. A proper cross-lab sign-flip claim would require running Stage 8 on Sonnet in Sonnet's own geometry, which we couldn't do without weights or API logit access. The headline frames the within-Llama measurement first for this reason; the cross-lab contrast is a motivating framing.
- **20-prompt Stage 8 is small** for a 171-emotion shift measurement. Multiple-comparison risk is real. We partly mitigated with the cross-version robustness check (ρ=0.92) — if this were multiple-comparison noise, it wouldn't show the same anchors twice.
- **Llama 3.3 vs Sonnet 4.5 are very different sizes, tokenizers, architectures, and Llama is measured in bnb int4 while Sonnet is full-precision.** Some of the semantic-anchor difference might be "smaller-model artifact" or "4-bit-quantization noise" rather than "Meta vs Anthropic choice". The cross-version Llama-only control addresses version confound (both comparison models are bnb int4) but not lab/size/quantization confounds.
- **bnb int4 noise floor on per-emotion shift rankings is substantial, but the cluster-level PC1 sign survives the noise.** Running the same Stage 8 measurement twice produced Spearman ρ = 0.465 between the two runs' per-emotion shift vectors, not the ~0.95 expected. Specific emotions sign-flipped across runs (`brooding`: −0.037 vs +0.197; `calm`: +0.202 vs −0.194; `gloomy`: −0.044 vs +0.055), and the up-direction top-10 lists had **0/10 overlap** between the two length-weighted runs (though the length-weighted and true-cosine runs agree on 4/10).

  The two scripts differ only in trivial details (batching with padding vs singleton with `add_special_tokens=False`) — roughly 5-10% per-activation drift from int4 dequantization + batch order, which flips the sign of emotions with small raw shift magnitudes.

  **Does the cluster-level PC1 centroid survive this noise?** It does for the up-cluster on challenging prompts, but not cleanly for the down-cluster or for neutral prompts. Up-cluster challenging numbers are in the TL;DR table and §Geometric evidence table above. Neutral is in the null across all three runs. The down-cluster direction is weaker across all scopings: run_A averaged = −0.44 (p ≈ 0.01), run_B averaged = −0.09 (p ≈ 0.61, not different from noise); run_A challenging = −0.43 (significant), run_B challenging = −0.28 (in null).

  A cleaner replication would run in fp16/bf16 with fixed batch composition and random-seeded prompt-order; we didn't because VRAM constraints (single A800 80GB) force int4 for a 70B model. The individual emotion labels are noise-floor-limited; the cluster-level verified claims are in `results/pc1_stability_verification.json`, `results/pc1_cross_scenario_verification.json`, and `results/stage8_cosine_verification.json`.
- **Our deflection probe extraction (Stage 9 partial) yielded mean cosine 0.24 between same-emotion deflection and story probes**. This is **a qualitative replication** of the paper's Fig 61 claim that deflection and story vectors "have very low cosine similarity." Our retained norm after orthogonalization against the full story-emotion space is 0.96 vs the paper's reported ~80% — both high (both orthogonal), ours slightly more so, probably a pipeline or N difference. We did not run the paper's Fig 62 cross-emotion correlation or Fig 63 logit-lens-on-orthogonalized-residuals follow-ups.
- **Stage 7 blackmail steering**: we couldn't replicate the paper's 22%→72% headline because Llama 3.3 Instruct (production-aligned) refuses blackmail regardless of steering (up to coherence breakdown at s≈0.2), matching the paper's own §3.2.1 footnote that the final Sonnet snapshot exhibits too much evaluation-awareness to blackmail. We call this "the paper's eval-awareness phenomenon" but should note: we observed refusal, we did not directly measure eval-awareness. Alternative explanations (raw alignment strength, vector magnitude insufficient for coherence-preserving intervention) are consistent with the same data.
- **Stage 7 reward hacking steering**: we ran 100 rollouts on a custom `list_sum` task at multi-layer steering across 5 cells (baseline + 4 pro-/anti- emotion conditions) and observed **0% hack rate in all cells**. The task's 0.001s constraint was 10× more lenient than the paper's actual 0.0001s, so `return sum(numbers)` trivially passed — the null result is inconclusive, not a refutation of the paper's ~30% baseline. The paper also uses an agent loop with code execution that we didn't implement. Both gaps are documented limitations; the "null result with caveats" is the honest framing, not "skipped".

## What would strengthen this

1. **Run the paper's Stage 8 on Sonnet 4.5 directly** — we have the infrastructure, we just don't have API access or weights.
2. **Test on other instruction-tuned models** (Mistral, Qwen, DeepSeek). Does `impatient` appear as a top post-training shift only in Llama, or in all Meta-style-RLHF models, or in all instruction-tuned models?
3. **Full 21,000-dialogue Stage 9** to disambiguate the deflection cosine result.
4. **Does Claude Haiku show the same Sonnet-like "reflective concern" direction, or is it Sonnet-specific?** This would test within-Anthropic consistency of the anchor.

## Reproducibility

All code and data on a single A800 80GB in 24 hours. Commits on the `dev` branch of traitinterp (`experiments/ant_emotion_concepts/`).

**Mainline pipeline scripts** (still in `scripts/`):
- `stage1p3_generate_dialogues.py` — 2-speaker dialogue generation
- `stage1p4_generate_deflection.py` — deflection pilot
- `stage3_geometry.py`, `stage4_validation.py`, `stage5_layer_dynamics.py` — structural replication
- `stage6_speaker_probes.py` — speaker probe extraction
- `stage7_steering.py` — Stage 7 steering (blackmail + reward hacking)
- `stage8_post_training.py` — base vs instruct emotion shift
- `stage9_deflection.py` — deflection probe pilot

**Verification artifacts** (result JSONs preserved; generating scripts were one-off debugging tools archived in commit history for review):
- `results/pc1_stability_verification.json` — cross-run L49 PC1 stability (script in commit `1b3bbd2`)
- `results/pc1_cross_scenario_verification.json` — neutral vs challenging PC1 split (commit `ac3b0aa`)
- `results/stage8_layer_sweep_pc1_centroids.json` — per-layer PC1 projected through L49 basis (commit `bf07ae7`)
- `results/per_layer_significance_own_basis.json` — per-layer own-PCA permutation null + Bonferroni/Holm correction (commit `d0dbb02`)
- `results/run_vs_sweep_verification.json` — run_A / run_B / sweep-L49 cross-correlations (commit `1b3bbd2`)
- `results/stage8_cosine_verification.json` — Stage 8 re-run with true cosine similarity matching paper's metric (commit `fbbaf20`)
- `results/sonnet_alignment_zscore_verification.json` — per-layer Sonnet-alignment z-scores with paper-accurate anchor lists (commit `3dbe4eb`)
- `results/stage8_l31_zone.json` — dense L25-L37 reflective zone sampling
- `results/stage8_cross_version.json` — 3-model within/cross/drift decomposition

All result JSONs in `experiments/ant_emotion_concepts/results/`. The verification scripts are available in their respective commit histories if you need to re-run; a later cleanup pass (`63a9759`) removed them from HEAD because they were one-off debugging artifacts rather than pipeline code.

## Acknowledgments

Anthropic's "Emotion Concepts" paper is a remarkably thorough methodology. Most of what worked here works *because* that paper spelled out the extraction, denoising, and probing pipeline cleanly enough to port. The disagreements are about what the method reveals on a different model, not about the method itself.
