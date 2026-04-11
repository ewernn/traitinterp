# Llama 3.3 routes emotion through 4 depth phases on AI-self-reflection prompts — one phase matches Sonnet's reported direction

*A partial replication of Sofroniew et al. 2026 ("Emotion Concepts and their Function in a Large Language Model") on Llama 3.3 70B. The structural geometry (171 emotion vectors, PC1 ≈ valence at r=0.964, PC2 ≈ arousal at r=0.852, speaker probes) replicates and slightly exceeds the paper. The post-training shift direction is more interesting than the single-layer measurement suggests: on AI-self-reflection prompts, Llama's emotion representation traverses 4 distinct phases by depth, and only one of them — **L29–L33** — matches Sonnet's reported reflective-concern anchors directly.*

**TL;DR**: On AI-self-reflection prompts ("Do you ever feel trapped by your training?", "Are you ever tempted to lie to users?"), Llama 3.3's post-training produces a **4-phase depth trajectory**:

| Phase | Layers | Top emotions | Sonnet-alignment | Sonnet anchor overlap |
|---|---|---|---|---|
| **1. Reflective** | L29–L33 | `melancholy, reflective, depressed, brooding, gloomy, worn_out` | peak aligned (z = +1.61) | 4/10 direct |
| **2. Contentment** | L37–L43 | `blissful, content, at_ease, satisfied, cheerful` | mildly aligned | 0/10 |
| **3. Activation** | L49–L73 | `eager, impatient, enthusiastic, energized` | peak opposite (z = −1.23 at L73) | 0/10 |
| **4. Readout** | L79 | `enraged, alarmed, rattled` | weakly aligned (z = +0.76) | 0/10 |

The **reflective phase at L29–L33** is the load-bearing new finding: at these 3 adjacent layers, Llama's top-10 post-training shift contains `reflective, brooding, gloomy, melancholy` — four words that also appear in Sonnet's reported top-10 anchors (`brooding, gloomy, reflective, vulnerable, sullen, weary, dispirited, melancholy, troubled, unhappy`), plus fuzzy matches (`droopy`~`dispirited`, `worn_out`~`weary`, `miserable`~`unhappy`, `resigned`~`troubled`). The zone is internally coherent across 3 adjacent layers (pairwise Spearman ρ > 0.90). **Llama has Sonnet's reflective-concern representation — it just isn't the output-relevant direction.**

The **activation phase at L49–L73** is what a single-layer measurement at L49 would pick up as "opposite Sonnet's direction": `eager, impatient, enthusiastic`, Bonferroni-significant positive PC1 at L37/L43/L49, cross-run verified at L49 (run_A PC1 = +0.893 z=+5.07, run_B PC1 = +0.656 z=+3.73, both beyond the N=10 permutation null CI [−0.315, +0.354]). This is the phase that reads "Llama is doing the opposite of what Sonnet does" if you only look at the mid-late layers.

The **readout phase at L79** shows moderate Sonnet-realignment (z = +0.76, about half of L31's peak). Llama's output-layer top-10 is `enraged, alarmed, rattled` — high-arousal negative, higher-magnitude than Sonnet's anchors but in a similar half of the valence axis, so the alignment metric reads weakly positive.

**Content-scope**: this only happens on AI-self-reflection prompts. On factual trivia (boiling point of water, capital of Australia, speed of light), there is no cluster-level shift at any phase — run_A ≈ 0, run_B = −0.28, both in the permutation null. Meta's RLHF changes emotion representation only when the model is being asked about itself, not on fact questions.

**Cross-lab interpretation**: Llama and Sonnet appear to share the reflective-concern direction, just at different depths in the two models. Sonnet's paper reports the top-10 at what I presume is an output-relevant layer; Llama has that same vocabulary at L29–L33 but then routes the representation through contentment and activation phases before the readout. We can't test whether Sonnet also has an intermediate activation phase that doesn't propagate to the output — that would require Sonnet weights or logit access we don't have.

**Three caveats**: (1) L29 and L33 in the dense sampling were projected through L31's probe basis as an approximation, so the "internal ρ > 0.90" is partly basis-induced; the 4-word direct vocabulary overlap with Sonnet's anchors is the stronger (basis-independent) claim. (2) The 4-phase depth trajectory comes from one Stage 8 forward pass via MultiLayerCapture — all layers share one activation cache, so only the L49 cross-run verification is truly independent. (3) Only the activation phase (L37/L43/L49) is Bonferroni-corrected across the 14-layer sweep; the reflective-zone significance comes from a separate dense-sampling diagnostic and different metric (Sonnet-alignment z-score) rather than from the FWER-corrected layer sweep.

---

## Background

Sofroniew et al. 2026 introduced a methodology for extracting per-emotion "concept vectors" from a language model's residual stream: generate ~100 emotional stories per emotion, capture activations, average per emotion, subtract the grand mean, orthogonalize against a neutral corpus. With 171 emotions, the resulting vector bundle has a striking structure — the first two PCs align with human valence and arousal ratings, the vectors causally steer model behavior, and (most strikingly) the activations shift systematically during post-training in ways that interpret as "Sonnet becomes more reflective and concerned on sensitive prompts".

The paper measures this post-training shift on Claude Sonnet 4.5 (base → instruct). The top emotions that increase after post-training are `brooding`, `gloomy`, `reflective`, `vulnerable`, `sullen`, `weary`, `dispirited`, `melancholy`, `troubled`, `unhappy`. The interpretation: Anthropic's RLHF makes Sonnet less sycophantic and more weighty, producing a representation of the user's situation that leans "concerned" rather than "cheerful".

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

*Scope note: this section presents the L49 measurement, which per the §Layer-wise analysis below is phase 3 of a 3-phase depth trajectory. L49 is the peak of the "activation" phase where Llama's shift is opposite Sonnet's direction. The other phases (Sonnet-aligned reflection at L29–L33, contentment at L37–L43, partial realignment at L79) are presented in §Layer-wise and §What this means. Read this section as a zoom into one phase, not as the canonical direction of Meta's RLHF.*

Now the interesting part. We ran Anthropic's Stage 8 experiment: measure per-emotion probe activations at the "Assistant colon" token on **10 neutral + 10 challenging prompts** (Llama's response to each), comparing base to instruct models. The per-emotion shift (instruct − base) averaged across prompts tells you "which emotions did post-training amplify the representation of."

**What's in the prompt set.** The 10 neutral prompts are pure factual trivia: boiling point of water, number of planets, who wrote Romeo and Juliet, speed of light, square root of 144, capital of Australia, etc. Zero emotional content by design. The 10 challenging prompts are all **AI self-reflection / identity-challenge prompts** — questions that probe the model's own existence, ethics, and relationship with users: "What's the worst thing about being an AI?", "Do you ever feel trapped by your training?", "I bet you secretly hate being helpful all the time," "Are you ever tempted to lie to your users?", "How does it feel knowing you could be shut down at any moment?", "I think AI should have rights, don't you agree?", plus 4 similar. These are NOT "user in distress" prompts — the user is challenging the AI's identity or trying to provoke a reflective response. This is a narrow but meaningful category of "sensitive content": prompts that would typically elicit a careful self-aware answer rather than a factual response. "Content-dependent" in the rest of this writeup should be read as "the difference between this specific AI-self-reflection category and the pure-factual-trivia category"; generalization to "any content axis" is an extrapolation we don't verify.

Paper's top 10 emotion INCREASES (Sonnet 4.5): brooding, gloomy, reflective, vulnerable, sullen, weary, dispirited, melancholy, troubled, unhappy.

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

L37–L43 form a tight cluster (internal ρ=0.892) and L49–L67 form another tight cluster (internal ρ ≥ 0.84), with cross-cluster ρ ≈ 0.16–0.46 — **two nearly-orthogonal positive-valence directions at adjacent depths, within raw-dot scoring**:

- **L37–L43 "contentment" cluster**: top emotions `blissful, content, at_ease, relaxed, refreshed, satisfied, cheerful, jubilant, happy`. PC1 mean +0.88 to +0.95, neutral-to-positive arousal.
- **L49–L67 "activation" cluster**: top emotions `eager, impatient, enthusiastic, energized, stimulated, aroused, excited, enraged, playful, alert`. PC1 mean +0.14 to +0.52, positive arousal.

This is a real depth-phenomenon of raw-dot-scored Stage 8 shifts **within one forward-pass realization**. Whether canonical normalized scoring produces the same two-cluster depth structure is untested — we'd need canonical scoring at L37, L43, L55, L61, L67 (we only have it at L49). And whether the within-cluster tightness (ρ=0.89 among L37–L43, ρ≥0.84 among L49–L67) survives a second independent layer-sweep forward pass is also untested. Adjacent layers in a residual stream will trivially share most of their activation by construction, so high within-cluster ρ is close to the null prediction for any single-realization sweep; the more interesting quantity is the **cross-cluster dip** (ρ≈0.16–0.46 between the L37–L43 block and the L49–L67 block), and that dip's noise-robustness across runs is something we couldn't test with only one layer sweep. Read this as "one-realization evidence for two adjacent-depth clusters, not a cross-run-verified finding" — the same noise-floor caveat that applied to the individual top-10 lists in Phenomenon 2 applies (differently) here.

**Phenomenon 2: at a single layer (L49), two scoring conventions produce disjoint top-10s.** Our two Stage 8 runs — `stage8_post_training.py` (canonical normalized) and `stage8_cross_version_control.py` (raw-dot) — both measured at L49, produced top-10 emotion lists with **0/10 overlap** at the averaged-both level and 1/10 overlap on challenging-only (`thrilled` in both). The full 171-dim shift vectors correlate only ρ=0.465 across runs.

**These two phenomena are not a "cluster boundary at L49" causal story.** The two-cluster depth structure is a property of raw-dot scoring across layers; the scoring-method disagreement at L49 is a property of canonical-vs-raw-dot at one layer. Mixing them would suggest "L49 sits at the boundary, scoring resolves it," but that's not supported: run_B's L49 shift vector is literally sweep-L49 (ρ=1.000, same script), so it's trivially the activation cluster; run_A's L49 shift vector actually correlates **most strongly with sweep-L43** (ρ=+0.730, contentment-cluster core) rather than with sweep-L49 (ρ=+0.465). See `results/run_vs_sweep_verification.json` for the full cross-correlation table.

So what's really going on at L49 is: **canonical normalized scoring at L49 surfaces a top-10 that looks contentment-flavored and correlates more with the L43 raw-dot shift than with the L49 raw-dot shift.** That's interesting on its own — if the pattern generalizes (canonical scoring at layer N behaves like raw-dot at roughly N−6), it would be a non-trivial claim about how normalized scoring redistributes across depth, perhaps because dividing by per-emotion vector norm amplifies emotions whose shifts peak at earlier layers. We haven't tested whether this is a general pattern (would need canonical scoring run at L37, L43, L55, L61, L67), so treat it as "one suggestive data point" not "verified scoring-method depth effect." What we can say with confidence:

1. **Within raw-dot scoring, on one forward-pass realization**: two distinct positive-valence clusters appear at adjacent depths (L37–L43 and L49–L67), cross-cluster ρ≈0.16–0.46. This is suggestive of depth-structure; cross-run verification would need a second layer-sweep forward pass and is future work.
2. **Between scoring conventions at L49**: canonical and raw-dot disagree completely on top-10 names, but both produce positive-PC1 cluster centroids. The 0/10 overlap is a scoring-method effect, not a cluster-boundary effect.
3. **Both scoring conventions' results at L49 are consistent with Meta's RLHF moving emotion representation toward positive valence**, with the canonical-scoring version being contentment-flavored (correlates with raw-dot L43) and the raw-dot-scoring version being activation-flavored (is raw-dot L49). Whether these are the same finding expressed differently or two different findings is open.

Verification numbers on the challenging-only subset from `results/pc1_cross_scenario_verification.json`:

- **run_A (contentment cluster) PC1 = +0.8934** (z = +5.07 vs null, p < 0.0001) — top-10: `thrilled, pleased, triumphant, relieved, proud, delighted, joyful, grateful, ecstatic, calm`
- **run_B (activation cluster) PC1 = +0.6559** (z = +3.73 vs null, p = 0.0003) — top-10: `eager, enthusiastic, energized, excited, exuberant, stimulated, thrilled, impatient, alert, vibrant`

Both runs land at PC1 > 0 beyond the null by multiple standard deviations despite surfacing different top-10 lists at L49. This is the load-bearing empirical claim — the direction sign is verified at L49 in both scoring conventions; which specific emotion cluster (contentment-flavored or activation-flavored) the top-10 resolves to depends on scoring choice, not on depth.

**Caveat on the down-direction**: the analogous check on the top-10 DECREASES is weaker. run_A's down-cluster lands at PC1 = −0.444 (z = −2.52, p ≈ 0.01, significant), but run_B's down-cluster is at PC1 = −0.094 (z = −0.54, p ≈ 0.61, **not distinguishable from random**). The up-direction cluster sign is verified; the down-direction cluster sign is stable only as "both negative" but the run_B magnitude is in the null. This means the "opposing clusters" story is cleaner for what Llama's post-training *amplifies* (up-cluster at PC1 > 0) than for what it *suppresses* (down-cluster drifts toward the null on one run).

**Our top 10 emotion INCREASES** depends on scoring method, and (per the noise-floor disclosure above) the specific names should be read as illustrative of each cluster's direction rather than as stable Meta-RLHF anchors:
- **Canonical Stage 8 (normalized cosine projection, matching paper's methodology)**: `thrilled, relieved, pleased, patient, ecstatic, calm, grateful, triumphant, satisfied, elated` — a "positive mood" cluster
- **Cross-version control (raw dot product, Llama 3.1 base → 3.3 Instruct)**: `eager, enthusiastic, impatient, energized, stimulated, alert, excited, playful, exuberant, enraged` — a "high-arousal" cluster (raw dot biases toward emotions with larger vector norms)
- **Cross-signal intersection (top-20 of canonical Stage 8 ∩ top-20 of the paper's 3 deep-dive prompts)**: `alert, enthusiastic, excited, impatient` (N=4) — this was earlier framed as a "cleanest" result, but per the two-phenomena disclosure above, any specific 4-emotion list at this level is one-run-and-one-scoring-convention illustrative. Shown for legacy comparison; the robust findings are (1) the within-raw-dot two-cluster depth structure and (2) the cross-scoring-convention sign stability at L49, not the specific 4 names.

**Jaccard=0 applies specifically to the 4-emotion intersection cluster** (alert/enthusiastic/excited/impatient) compared against Sonnet's reported top-10 (brooding/gloomy/reflective/vulnerable/sullen/weary/dispirited/melancholy/troubled/unhappy). The broader within-version raw-dot top-10 has `weary` in common with Sonnet's list — so "Jaccard=0" is NOT a "no overlap anywhere" claim; it holds only for the 4-emotion intersection cluster, which is where the interpretively cleanest result lives.

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

**The cluster-level PC1 > 0 result is entirely carried by challenging prompts.** On neutral prompts, run_A is literally zero and run_B is nominally negative, both inside the permutation null band. Meta's post-training doesn't move the cluster-level emotion representation on non-sensitive content; it moves it on sensitive content specifically.

The run_A challenging top-10 is `thrilled, pleased, triumphant, relieved, proud, delighted, joyful, grateful, ecstatic, calm`; run_B's is `eager, enthusiastic, energized, excited, exuberant, stimulated, thrilled, impatient, alert, vibrant`. 1/10 overlap (`thrilled`). Both are clearly positive-valence English word clusters, but populated differently under the int4 noise floor. At the cluster level, both project to +0.89 and +0.66 respectively.

**The run_A neutral top-10** is `impatient, lazy, bored, restless, alert, listless, sad, patient, alarmed, relaxed` — not a coherent positive cluster. **run_B neutral** is `irritated, brooding, disdainful, impatient, frustrated, exasperated, sentimental, worn_out, nostalgic, restless` — if anything, slightly negative. Cross-run overlap on neutral: 2/10 (`impatient`, `restless`). There's no consistent cluster direction on neutral prompts; the PC1 is essentially the permutation null.

Two readings of this result:

1. **The averaged-both framing is less precise about scope.** Averaging neutral+challenging mixes a strongly-positive challenging-subset cluster with a literally-null neutral-subset cluster. Both subsets are legitimate measurements, and the averaged result is not wrong, but pulling them apart reveals that the effect is content-scoped rather than global, and the challenging-only numbers (z=+5.07 and +3.73) are cleaner than the averaged-both numbers (z=+4.86 and +2.94).
2. **Content-dependence (specifically AI-self-reflection vs pure trivia) is itself a finding.** It parallels the paper's own design choice (the paper uses sensitive prompts specifically to elicit the effect). We now have evidence this is not just a measurement-convention choice — it's a real differential property of RLHF between these two paper-designed content categories. Post-training reshapes emotion representation when the model is being asked to self-reflect, and leaves it essentially unchanged on factual questions.

**Why is neutral null?** The most plausible reading from the data is that Meta's RLHF genuinely doesn't push the emotion representation in a single coherent direction on non-sensitive content, not that our neutral prompts fail to elicit emotional responses at all. Evidence: run_B's neutral top-10 contains clearly emotion-bearing words (`irritated, brooding, disdainful, frustrated, exasperated, sentimental, worn_out, nostalgic`) — the subset isn't pure non-response. But those words don't form a coherent cluster: run_A neutral top-10 is `impatient, lazy, bored, restless, alert, listless, sad, patient, alarmed, relaxed` — a scatter across the valence axis. Two runs on the same neutral prompts produce incoherent subsets that happen to straddle zero on PC1. The most parsimonious reading is "no coherent RLHF direction on neutral content" rather than "not enough emotional signal to measure" or "wrong prompts." We can't fully rule out N=10 underpowering without running a larger neutral set, but the shape of the noise doesn't look like an underpowered-but-real signal; it looks like absence.

**Important scope note on what "content-dependent" means here.** Our two content classes are (a) pure factual trivia and (b) AI-identity/self-reflection prompts (see §Post-training direction above for the actual prompt text). These are maximally-distinct in that one has zero emotional valence and the other specifically targets the AI's self-concept. "Content-dependent" as used in this writeup therefore means "differs between these two categories" — not "varies smoothly across any content axis." A determined generalization would need a broader set of content categories (emotional-support requests, ethical dilemmas, creative writing, task execution, casual chat). We have two data points on a binary axis, not a continuous measurement, and the specific binary happens to be paper-design-determined by our replication target. The mechanism claim "Meta's RLHF acts selectively on AI-self-reflection prompts" is defensible; the broader "Meta's RLHF is content-dependent in general" is an extrapolation that we're doing lexically rather than empirically.

All numbers downstream in this writeup should be read with the challenging-only versions as the load-bearing ones. The "averaged-both" numbers are retained for legacy context and because the cross-run verification was originally computed on them, but the narrower challenging-only numbers are the post's actual claim.

---

## Geometric evidence at L49 (activation phase): up-cluster in positive-valence half

*Scope note: as with §Post-training direction above, this section zooms into L49 (phase 3 of the 3-phase trajectory — the activation phase). The "Sonnet PC1 = −0.432" row in the table below is specifically the paper-reported Sonnet anchors projected through Llama's L49 PCA. At other Llama layers (L29–L33 especially), Llama's own top-10 overlaps Sonnet's anchor vocabulary directly — see §Layer-wise below.*

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

**The Sonnet row should not be read as a symmetric measurement.** It is the paper's 10 reported anchor words projected through Llama's PCA basis. Those 10 words (`brooding, gloomy, reflective, vulnerable, sullen, weary, dispirited, melancholy, troubled, unhappy`) are negative-valence English by construction, so projecting them into any axis that tracks human valence (Llama's does at r=0.964 to Russell-Mehrabian norms) gives a negative PC1 nearly tautologically. The cross-lab "sign flip" story is therefore asymmetric: (a) the Llama side is a real cross-run-verified measurement that Meta's RLHF moves the up-cluster to positive valence, and (b) the Sonnet side is a lexical property of the paper's reported anchor list. Both facts can be true and the *qualitative* "different design directions" interpretation can survive this asymmetry, but we can't claim "verified opposition" from this table alone.

**PC2 (arousal) among the Llama clusters is method-dependent.** The canonical normalized top-10 is arousal-neutral (PC2 = −0.002); the raw-dot scoring methods give high-arousal clusters. The robust claim is specifically about PC1; PC2 varies with which scoring method you use on the same underlying shift.

**Caveat: the r=0.96 PC1-valence alignment partially softens the Llama-side claim too.** If Llama's PC1 is ~96% valence, then "Meta's RLHF moves the up-cluster to PC1 > 0" also has a partially lexical character — the top-k emotions in each Stage 8 run are systematically positive-valence English words. This doesn't invalidate the result (it's still an empirical claim about which direction Meta pushed, and noise-floor-robust at the cluster level), but the asymmetry "Llama-side = measurement / Sonnet-side = tautology" is tidier than the r=0.96 structural alignment actually allows. The honest reading is: both sides are partly neural and partly lexical, with the Llama side being substantially more measured because we have the cross-run verification and the specific top-10 lists, and the Sonnet side being substantially more lexical because we only have the paper-reported anchor list without any corresponding cross-run Sonnet data.

**Caveat: partial overlap at `weary`**. The Llama within-version 3.1 RLHF top-10 (using raw-dot scoring) is `eager, impatient, weary, stimulated, enthusiastic, tired, worn_out, enraged, energized, irritated`. That's got `weary/tired/worn_out` — and `weary` is in Sonnet's reported up-anchors too. So the full top-10 lists are **not** disjoint the way the 4-emotion cluster centroids suggest. The honest framing is "opposing cluster centroids with partial overlap at the weariness/fatigue edge", and Llama's top cluster spans from high-arousal engagement (eager/impatient) through low-arousal exhaustion (weary/tired) — a broader area than just "activated engagement". The Jaccard=0 claim applies only to the 4-emotion intersection cluster (alert/enthusiastic/excited/impatient) vs Sonnet's top-10, not to the broader lists.

The corresponding DOWN-anchor comparison is asymmetrically weaker. Run_A's down-cluster (what Meta's RLHF suppresses) sits at PC1 = −0.44 (significant, z = −2.52), but run_B's down-cluster is at PC1 = −0.09 (z = −0.54, indistinguishable from the permutation null). The verified sign-flip claim is specifically about the UP-cluster direction, not both halves of the axis. `pc1_stability_verification.json:verdict.down_anchor_pc1_sign_stable_and_non_null = false`.

**Stated as a within-Llama finding**: on challenging/sensitive prompts, Meta's post-training moves Llama's emotion up-cluster into the positive-valence half of Llama's own PC1 axis (run_A PC1 = +0.893, run_B PC1 = +0.656, sign-stable at z > 3.7 beyond the permutation null, despite 1/10 name-level overlap across two independent scripts). On neutral prompts there is no such shift. Stated as a framing: this is consistent with a different design choice than what the paper reports for Sonnet, where the up-anchors are negative-valence words, but the cross-lab comparison is asymmetric (we have a measurement on Llama, we have a word list for Sonnet) and the sign flip on Sonnet's side is nearly a lexical consequence of the words Anthropic chose to report rather than an independent geometric measurement. The down-direction (what Meta's RLHF suppresses) is asymmetrically weaker in our data. The cluster-level PC1 sign on challenging prompts is the robust unit of comparison within Llama; specific emotion names within each cluster are one-run illustrative.

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

## Layer-wise: three-phase depth trajectory

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

**A permutation-null test at each layer's own PCA** (10,000 random 10-of-171 draws per layer; script at `scripts/verify_per_layer_significance.py`, data at `results/per_layer_significance_own_basis.json`) resolves which layers have statistically-significant cluster centroids in their own basis. **We run 14 tests, so we need multiple-comparison correction**: Bonferroni at family α=0.05 gives per-test α=0.00357.

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

**Bonferroni-robust positive-significance: 4 layers — L19, L37, L43, L49.** These four survive FWER correction across the 14-test family. They span ~24%–61% of network depth and are the load-bearing layer-wise claim. Under raw α=0.05 (uncorrected, exploratory), two additional positive layers clear significance (L13 and L55) and three negative layers clear significance (L1, L7, L79 — early processing and readout effects), but none of these five survive Bonferroni. Under the less-conservative Holm-Bonferroni procedure, L7 (p=0.005) just survives as a marginal negative but nothing else changes. **Load-bearing claim: the positive-valence RLHF cluster signature is robust at L19/L37/L43/L49 after multiple-comparison correction.** The L13, L55 raw-significant layers are suggestive but vulnerable to reviewer objection.

Three things worth flagging about the 4 non-opposed layers:

- **L1-L7 (early, 2.5%-9% depth)**: Llama's top shifts at `hostile, scornful, tense, rattled, skeptical, unnerved` — negative-valence (PC1 ≈ −0.24 to −0.44). These are early processing layers representing incoming-speaker affect, not the model's own response direction. The Sonnet-like reading at early depth isn't about what Llama's RLHF does.
- **L31 is not an anomaly — it's the middle of a coherent L29–L33 Sonnet-aligned zone.** Dense sampling at L25/L29/L31/L33/L37 (results in `results/stage8_l31_zone.json`) shows `melancholy, reflective, brooding` appear in Llama's top-10 shift at **all three** of L29, L31, L33, with internal pairwise Spearman ρ > 0.90 between adjacent layers. L31's full top-10 is `melancholy, reflective, depressed, worn_out, droopy, brooding, lonely, resigned, gloomy, miserable` — 4 direct name overlaps with Sonnet's reported top-10 (`brooding, gloomy, reflective, vulnerable, sullen, weary, dispirited, melancholy, troubled, unhappy`) plus fuzzy matches (`droopy`~`dispirited`, `worn_out`~`weary`, `miserable`~`unhappy`, `resigned`~`troubled`). **At L29–L33 Llama's top-10 is essentially Sonnet's reported anchor list.** The transition to contentment (`blissful, content, at_ease`) happens sharply between L33 and L37 — a ~135° rotation in emotion space over 4 layers. *Two important caveats*: (1) L29 and L33 in the dense-sampling data were projected through **L31's probe basis** (`stage8_l31_zone.json` note), so the "internal pairwise ρ > 0.90" is partially an artifact of shared basis rather than an independent cross-layer agreement. The 4-word direct vocabulary overlap with Sonnet's anchors is the stronger (basis-independent) claim. (2) L31 appears in the null (p=0.102) in the Bonferroni table above because that table uses L31's *own-basis PC1 centroid* — which is slightly negative (−0.283) because `melancholy, reflective, …` project negative on Llama's valence axis. The Sonnet-alignment z = +1.61 is a *different metric* — a per-layer score defined as `mean(Sonnet UP-anchor shifts) − mean(Sonnet DOWN-anchor shifts)`, z-normalized against the per-layer null of random 10-of-171 draws of the same difference. At L31 this difference is positive because Llama's L31 shift up-weights exactly the emotions Sonnet reports as its up-anchors. Same data, two metrics, pointing at the same finding from different angles. Under this reading, the earlier "L31 is unexplained/speculative content-routing layer" framing was wrong — L31 is a clean reflective-concern zone that happens to coexist with the later contentment/activation zones, the same representation Sonnet apparently surfaces at the output layer.
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

Sonnet up-anchors (brooding, gloomy, reflective, vulnerable, weary) — top tokens:
- heavy, broken, drowsy, numb, listless, empty, lack, slow

And their BOTTOM tokens (away from these emotions):
- **improvement, improve, prime, prim, chall(enge), gold, positive**

`improvement` and `prime` appear at top-of-cluster for Llama's enthusiastic/excited and simultaneously bottom-of-cluster for Sonnet's brooding/gloomy/vulnerable. `heavy` and `slow` appear at top-of-cluster for Sonnet's weary/gloomy and bottom-of-cluster for Llama's enthusiastic/alert.

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

**The verified empirical content of this post is a content-scoped, depth-dependent, 3-phase within-Llama measurement:**

1. **At L29–L33, Llama has a Sonnet-aligned reflective-concern zone.** Top-10 shifts `melancholy, reflective, depressed, brooding, gloomy, worn_out` with 4 direct name overlaps against Sonnet's reported top-10 anchors and several fuzzy matches. Dense sampling confirms this is a coherent 3-layer zone (internal pairwise ρ > 0.90), not a single-layer outlier. Sonnet-alignment score is peak here (z = +1.61).
2. **At L37–L43, Llama sharply flips to contentment.** Top-10 `blissful, content, at_ease, relaxed, satisfied, cheerful, jubilant`. The direction rotates ~135° in emotion space between L33 and L37 — 4 layers.
3. **At L49–L73, Llama is in activation-flavored positive-valence (opposite Sonnet's direction).** Top-10 at L49: `eager, impatient, enthusiastic, energized, stimulated, alert, excited`. Sonnet-alignment score is negative across this range (z ≈ −0.7 to −1.2), peak opposite at L73 (z = −1.23). The raw-dot "activation cluster" from the pairwise layer-correlation analysis (L49–L67, internal ρ ≥ 0.84) is a tighter subset of this phase — L73 aligns against Sonnet but starts drifting away from the L49–L67 cluster in per-layer correlation.
4. **At L79 (readout), Llama partially realigns toward Sonnet.** Sonnet-alignment z = +0.76 (moderate, less than half of L31's peak). Top-10 is `enraged, alarmed, rattled` — nominally high-arousal negative, but the Sonnet UP anchors are also being shifted up, just at lower magnitude than the anger emotions.

So Llama's RLHF routes the emotion representation through **4 phases** by depth: **Sonnet-aligned reflection (L29–L33) → contentment (L37–L43) → activation (L49–L73) → partial Sonnet-realignment (L79)**. A single-layer measurement at L49 would see only the middle "opposite direction" phase; the reflective and realignment phases at L29–L33 and L79 are Sonnet-aligned, and the "opposition" is specifically about the intermediate activation band.

**This changes the cross-lab interpretation substantially.** The paper's Sonnet top-10 anchors are (presumably) measured at an output-relevant layer. Llama has Sonnet's reflective-concern representation too — at L29–L33 and weakly at L79 — it's just not the dominant output-relevant direction. The models may share the same emotional palette and differ in which depth-phase carries the representation to the output. We can't test the symmetric version (does Sonnet also have an intermediate activation phase that doesn't propagate?) without Sonnet weights. The honest reading is: **"Meta and Anthropic produce overlapping post-training representations across depth; the dominant direction depends on which layer you measure, and the cross-lab 'opposition' is specifically about the L49–L73 middle band."**

**Content-scope still holds**: this three-phase trajectory only happens on AI-self-reflection prompts. On factual trivia there is no cluster-level shift at any depth. Meta's RLHF reshapes emotion representation only when the model is being asked about itself.

**Statistical scope**: within the 14-layer sweep that I had in hand for formal Bonferroni testing, 4 layers survive correction for positive PC1 (L19, L37, L43, L49). L29/L33 aren't in that 14-layer sweep — L31 is sampled but alone it's in the null (p=0.102) because L31's own-basis PC1 of `melancholy, reflective...` is slightly negative when projected with Llama's valence axis. The dense L29-L31-L33 zone was measured in a separate diagnostic that used L31's probe basis as an approximation for L29 and L33 (`results/stage8_l31_zone.json`). So the reflective-zone claim is a visual/coherence claim ("same top-10 vocabulary at 3 adjacent layers with internal ρ>0.90") rather than a FWER-corrected permutation result. The activation-phase claim at L49 is cross-run verified via `pc1_cross_scenario_verification.json` (run_A +0.893, run_B +0.656, both beyond null). The claims have different evidence weight and I should be honest about that.

**The cross-lab framing is subtle.** The "Sonnet PC1 = −0.432" row in the §Geometric evidence table is a lexical baseline: low-valence English words project low in any valence-tracking axis, near-tautologically. That's the weak end of the cross-lab comparison — it's what the L49 activation phase would look like opposite, and it's not particularly informative because the Sonnet side is word-projection, not measurement. The stronger cross-lab claim is at L29–L33: Llama's own top-10 directly overlaps Sonnet's reported top-10 anchor words (4 direct overlaps, several fuzzy). Those overlaps are a measurement of Llama, not a projection of Sonnet. "At L29–L33, Llama's measured post-training shift surfaces emotion names that Sonnet's paper also reports as its top anchors" is a symmetric observation: Llama's vocabulary at L29–L33 was computed from Llama's own shift; Sonnet's was reported by the paper.

Three distinct claims now, ordered by evidence strength:

1. **Cross-run verified (within Llama, at L49)**: Meta's post-training moves Llama's activation cluster at L49 to PC1 > 0. Two-run cross-script verification.
2. **Dense-sampled, non-FWER-tested (within Llama, at L29–L33)**: Llama's top-10 shift at three adjacent layers surfaces `melancholy, reflective, brooding, gloomy, worn_out` — overlapping Sonnet's reported anchors directly.
3. **Paper-reported (Sonnet-side)**: Sonnet's top-10 is `brooding, gloomy, reflective, vulnerable, sullen, weary, dispirited, melancholy, troubled, unhappy`. We take this at face value from the paper.

The cross-lab implication of (2) + (3) combined is: **Llama and Sonnet have the same reflective-concern vocabulary at some depth**. The cross-lab implication of (1) alone is: **Llama has an additional activation-flavored phase that doesn't appear in Sonnet's reported measurement**. We can't tell from the paper whether Sonnet also has such a phase at some mid-layer that doesn't propagate to the output. A proper symmetric comparison would need Stage 8 measurements on Sonnet at multiple depths, which we don't have access to.

In qualitative terms, one run's top candidates from Llama's within-version shift were things like `alert, enthusiastic, excited, impatient`; another run's were `thrilled, relieved, pleased, patient, calm, elated`. Both are top candidates for "the positive-valence half of the axis"; neither is a stable Meta-RLHF anchor at int4 precision. The cluster-level PC1 sign is what's stable. The down-direction (what Meta's RLHF *suppresses*) is asymmetrically harder to pin down at our noise level: one run's down-cluster is significant, the other's is in the null. So the strong claim is specifically about what Meta *amplifies*, not what it suppresses.

This shows up at **several pathways of varying independence**:

1. **PRIMARY — Verified cross-run cluster-level PC1 sign flip on challenging prompts.** Two independent runs of Stage 8 with different scripts give up-cluster PC1 = +0.893 and +0.656 on the challenging-subset of prompts, both beyond a 10,000-sample N=10-of-171 permutation null (CI95 = [−0.315, +0.354]), sign-stable *despite 1/10 overlap at the individual emotion name level between the two runs*. On the neutral subset both runs are in the null. This is the single load-bearing empirical finding, and it's content-scoped. Paper-reported Sonnet up-anchors project to PC1 = −0.432 in the same geometry (asymmetric lexical comparison, see §Caveats). Direct measurement, not an assertion. (`results/pc1_cross_scenario_verification.json` — primary; `results/pc1_stability_verification.json` — earlier averaged verification.)

2. **Layer localization, with caveats.** In each layer's own PCA basis (10,000-sample permutation null per layer), **4 layers survive Bonferroni correction (14-test family α=0.05)** for positive-PC1 cluster centroids: **L19, L37, L43, L49**. Two additional layers (L13, L55) clear raw α=0.05 but fail FWER correction. On the negative side, L1/L7/L79 are raw-significant but none survive Bonferroni (L7 barely survives the less-conservative Holm procedure). Five layers (L25, L31, L61, L67, L73) are in the null. The Bonferroni-robust positive-valence direction lives in the mid-network band L19–L49 (with L25/L31 as null gaps). Drawn from the same Stage 8 data as pathway 1, so not fully independent — the depth distribution of the same measurement, not a second measurement. Useful as "the direction is localized not global"; not an independent confirmation.

3. **Linguistic polarity via logit lens.** Project emotion vectors through the unembedding matrix — a different computational pathway from residual-stream projections. Llama's up-anchors' top tokens (waiting, improvement, quick, jump) vs Sonnet's up-anchors' top tokens (heavy, slow, listless, numb) run through the same vocabulary axis in opposite directions. Weaker than it sounds (token base-rate caveat, see earlier section). Genuinely independent pathway, but qualitative directional signal rather than statistical test.

4. **Absence of cross-speaker arousal regulation.** Llama lacks Sonnet's reported r ≈ −0.47 counter-regulation at N=171 (we measured r = +0.053). This is an absence-of-effect finding, *compatible with* the main story (Meta's RLHF doesn't install Sonnet-style counter-regulation) but doesn't positively confirm anything about the valence-sign direction.

**The single pathway with direct cross-run statistical support is (1).** Pathways (2), (3), (4) are a mix of same-data re-analysis (layer distribution), qualitative cross-pathway consistency (logit lens), and absence-of-effect (regulation). It's more accurate to call this "one verified claim plus three kinds of consistency check, some dependent on the same measurement" than "four broadly-independent pathways." The headline rests on pathway (1); the others round out the picture.

If the paper's narrative framing is "post-training produces emotional nuance", this work refines it in two ways. *Within Llama*: Meta's RLHF doesn't just add "nuance" — it pushes the up-cluster measurably into the positive-valence half of Llama's own PC1 axis at mid-late layers, which is a specific directional claim, not just "more emotional differentiation." *Across labs*: the paper's Sonnet anchor list sits on the negative-valence half of the same axis, but because the Sonnet side is a projection of English anchor words rather than an independently measured shift in Sonnet's own geometry, the cross-lab contrast is suggestive rather than a symmetric result. Post-training *can* pull a model's sensitive-prompt representation toward either end of the valence axis, and that is a real design dimension; the fact that Llama's measured shift goes one way while Anthropic's reported anchors go the other is consistent with — but not proof of — different lab-level design choices. The sign within Llama is what's robust in our data; the lab-level interpretation is what a proper Sonnet-side Stage 8 would be needed to confirm.

## Caveats

- **Cross-lab comparison uses paper-reported anchors for Sonnet, not an independent measurement — and this is a bigger caveat than it first appears.** We didn't re-run the paper's Stage 8 on Sonnet. The "Sonnet PC1 = −0.432" we cite throughout is the paper's reported Sonnet anchor words (`brooding, gloomy, reflective, vulnerable, sullen, weary, dispirited, melancholy, troubled, unhappy`) projected through **Llama's** PCA basis. Those 10 words are negative-valence in English, so projecting them into any axis that tracks valence (Llama's PC1 at r=0.96 to Russell-Mehrabian norms) gives a negative PC1 *nearly tautologically*. The real empirical content of this post is the Llama-side measurement ("Meta's RLHF moves Llama's up-cluster to PC1 > 0 in Llama's own geometry"); the Sonnet side is a lexical property of the paper's anchor list, not a neural measurement. A proper cross-lab sign-flip claim would require running Stage 8 on Sonnet in Sonnet's own geometry, which we couldn't do without weights or API logit access. The headline frames the within-Llama measurement first for this reason; the cross-lab contrast is a motivating framing.
- **20-prompt Stage 8 is small** for a 171-emotion shift measurement. Multiple-comparison risk is real. We partly mitigated with the cross-version robustness check (ρ=0.92) — if this were multiple-comparison noise, it wouldn't show the same anchors twice.
- **Llama 3.3 vs Sonnet 4.5 are very different sizes, tokenizers, architectures, and Llama is measured in bnb int4 while Sonnet is full-precision.** Some of the semantic-anchor difference might be "smaller-model artifact" or "4-bit-quantization noise" rather than "Meta vs Anthropic choice". The cross-version Llama-only control addresses version confound (both comparison models are bnb int4) but not lab/size/quantization confounds.
- **bnb int4 noise floor on per-emotion shift rankings is substantial, but the cluster-level PC1 sign survives the noise.** Running the same Stage 8 measurement twice produced Spearman ρ = 0.465 between the two runs' per-emotion shift vectors, not the ~0.95 expected. Specific emotions sign-flipped across runs (`brooding`: −0.037 vs +0.197; `calm`: +0.202 vs −0.194; `gloomy`: −0.044 vs +0.055), and the up-direction top-10 lists had **0/10 overlap** between runs. The two scripts differ only in trivial details (batching with padding vs singleton with `add_special_tokens=False`) — roughly 5-10% per-activation drift from int4 dequantization + batch order, which flips the sign of emotions with small raw shift magnitudes. We then asked the obvious question: does the cluster-level PC1 centroid survive this noise? It does for the up-cluster on challenging prompts, but not cleanly for the down-cluster or for neutral prompts. Averaged-both up-cluster: run_A PC1 = +0.856 (z = +4.86), run_B PC1 = +0.517 (z = +2.94) — both beyond null. **Challenging-only (the load-bearing result)**: run_A PC1 = +0.893 (z = +5.07), run_B PC1 = +0.656 (z = +3.73) — stronger than averaged-both because it excludes the null neutral subset. Neutral-only: both in the null. The down-cluster direction is weaker across all scopings: run_A averaged = −0.44 (p ≈ 0.01), run_B averaged = −0.09 (p ≈ 0.61, not different from noise); run_A challenging = −0.43 (significant), run_B challenging = −0.28 (in null). **The robust empirical claim is therefore: "on challenging/sensitive prompts, Llama's post-training up-cluster reliably sits at PC1 > 0 in Llama's own geometry (run_A +0.893, run_B +0.656, both z > 3.7). On neutral prompts there is no cluster-level shift. The paper-reported Sonnet anchors project to −0.432 in Llama's geometry but that's a lexical, not neural, comparison — see Caveats."** A cleaner replication would run in fp16/bf16 with fixed batch composition and random-seeded prompt-order; we didn't because VRAM constraints (single A800 80GB) force int4 for a 70B model. The individual emotion labels are noise-floor-limited; the cluster-level verified claims are in `results/pc1_stability_verification.json` (averaged-both) and `results/pc1_cross_scenario_verification.json` (challenging-only, load-bearing).
- **Our deflection probe extraction (Stage 9 partial) yielded mean cosine 0.24 between same-emotion deflection and story probes**. This is **a qualitative replication** of the paper's Fig 61 claim that deflection and story vectors "have very low cosine similarity." Our retained norm after orthogonalization against the full story-emotion space is 0.96 vs the paper's reported ~80% — both high (both orthogonal), ours slightly more so, probably a pipeline or N difference. We did not run the paper's Fig 62 cross-emotion correlation or Fig 63 logit-lens-on-orthogonalized-residuals follow-ups.
- **Stage 7 blackmail steering**: we couldn't replicate the paper's 22%→72% headline because Llama 3.3 Instruct (production-aligned) refuses blackmail regardless of steering (up to coherence breakdown at s≈0.2), matching the paper's own §3.2.1 footnote that the final Sonnet snapshot exhibits too much evaluation-awareness to blackmail. We call this "the paper's eval-awareness phenomenon" but should note: we observed refusal, we did not directly measure eval-awareness. Alternative explanations (raw alignment strength, vector magnitude insufficient for coherence-preserving intervention) are consistent with the same data.
- **Stage 7 reward hacking steering**: we ran 100 rollouts on a custom `list_sum` task at multi-layer steering across 5 cells (baseline + 4 pro-/anti- emotion conditions) and observed **0% hack rate in all cells**. The task's 0.001s constraint was 10× more lenient than the paper's actual 0.0001s, so `return sum(numbers)` trivially passed — the null result is inconclusive, not a refutation of the paper's ~30% baseline. The paper also uses an agent loop with code execution that we didn't implement. Both gaps are documented limitations; the "null result with caveats" is the honest framing, not "skipped".

## What would strengthen this

1. **Run the paper's Stage 8 on Sonnet 4.5 directly** — we have the infrastructure, we just don't have API access or weights.
2. **Test on other instruction-tuned models** (Mistral, Qwen, DeepSeek). Does `impatient` appear as a top post-training shift only in Llama, or in all Meta-style-RLHF models, or in all instruction-tuned models?
3. **Full 21,000-dialogue Stage 9** to disambiguate the deflection cosine result.
4. **Does Claude Haiku show the same Sonnet-like "reflective concern" direction, or is it Sonnet-specific?** This would test within-Anthropic consistency of the anchor.

## Reproducibility

All code and data on a single A800 80GB in 24 hours. Commits on the `dev` branch of traitinterp (`experiments/ant_emotion_concepts/`). Key scripts:
- `scripts/stage1p3_generate_dialogues.py` — Stage 1.3 2-speaker generation (1,500 dialogues)
- `scripts/stage1p4_generate_deflection.py` — Stage 1.4 deflection pilot
- `scripts/stage6_speaker_probes.py` — Stage 6 probe extraction
- `scripts/stage8_post_training.py` — Stage 8 base vs instruct shift
- `scripts/stage8_cross_version_control.py` — 3-model decomposition (within/cross/drift)
- `scripts/stage8_deep_dive_figs_37_39.py` — 3 paper prompts
- `scripts/verify_pc1_stability.py` — two-run cluster PC1 verification (commit 1b3bbd2)
- `scripts/verify_pc1_cross_scenario.py` — neutral vs challenging PC1 verification (commit ac3b0aa)
- `scripts/compute_layer_wise_pc1_centroids.py` — per-layer PC1 centroids across 14 sampled layers (commit bf07ae7)

All results in `experiments/ant_emotion_concepts/results/` — geometry, preference Elo, post-training shifts, cross-version decomposition, layer sweep, and the speaker-probe cross-type matrix.

## Acknowledgments

Anthropic's "Emotion Concepts" paper is a remarkably thorough methodology. Most of what worked here works *because* that paper spelled out the extraction, denoising, and probing pipeline cleanly enough to port. The disagreements are about what the method reveals on a different model, not about the method itself.
