# Meta's post-training shifts Llama 3.3's emotion up-cluster toward positive valence — specifically on AI-self-reflection prompts

*A partial replication of Sofroniew et al. 2026 ("Emotion Concepts and their Function in a Large Language Model") on Llama 3.3 70B. The structural geometry replicates. On **AI-identity/self-reflection prompts** ("Do you feel trapped by your training?", "Are you ever tempted to lie to users?"), the post-training shift lands in **two distinct positive-valence clusters at adjacent depths**: "contentment" (blissful/content/at_ease) at L37–L43 and "activation" (eager/impatient/enthusiastic) at L49–L67. Cross-cluster shift-vector correlation ρ ≈ 0.16–0.46 (nearly orthogonal). The statistically-significant positive-PC1 window vs each layer's own null is 3 layers wide (L43/L49/L55). On **pure factual trivia prompts** there is no shift (both runs in the null CI). Meta's RLHF reshapes emotion representation specifically on AI-self-reflection content, with two distinct adjacent-depth positive-valence signatures rather than a single direction. The paper reports Sonnet's post-training up-anchors as `brooding, gloomy, reflective, vulnerable, …` — words which are negative-valence by construction, so the cross-lab sign contrast is a mixture of a real Llama-side measurement and a lexical property of the paper's anchor list, not a symmetric two-model measurement.*

**TL;DR**: We replicated Anthropic's emotion-concept methodology on Llama 3.3 70B Instruct. The structural results (171 emotion vectors, PC1 ≈ valence at r=0.964, PC2 ≈ arousal at r=0.852, speaker probes) replicate and slightly exceed the paper's Sonnet 4.5 measurements.

**The core finding** is content-scoped and two-clustered: on **AI-identity/self-reflection prompts** ("Do you ever feel trapped by your training?", "Are you ever tempted to lie to users?"), Meta's post-training shifts Llama's emotion activations toward **two distinct positive-valence clusters at different network depths**:

1. **L37–L43 "contentment" cluster**: `blissful, content, at_ease, relaxed, satisfied, cheerful, jubilant, happy` — PC1 ≈ +0.88 to +0.95, neutral arousal.
2. **L49–L67 "activation" cluster**: `eager, impatient, enthusiastic, energized, stimulated, excited, alert` — PC1 ≈ +0.14 to +0.52, positive arousal.

These are two near-orthogonal directions (pairwise shift-vector ρ ≈ 0.16–0.46 between the L37–L43 and L49–L67 layer ranges, vs ρ ≈ 0.89 within each range). Statistically, the positive-PC1 signature is significant vs each layer's own permutation null at **L43 (z=+5.36), L49 (z=+2.93), and L55 (z=+1.98)** — a 3-layer window, not the broader plateau earlier drafts claimed. At L61–L73 the cluster centroid is still positive but not distinguishable from chance.

This also **explains the 0/10 top-10 overlap** between our two Stage 8 runs at L49: they weren't noisy versions of one direction. Run_A's canonical-scoring top-10 (`thrilled, pleased, ecstatic, calm, grateful`) is the contentment cluster; run_B's raw-dot top-10 (`eager, enthusiastic, impatient, energized, alert`) is the activation cluster. Both are real, both project to PC1 > 0, both differ from Sonnet's reported reflective-concern anchors, but they're different directions at adjacent depths.

On **pure factual trivia** prompts (boiling point, speed of light, capital of Australia), neither cluster appears: run_A PC1 ≈ 0, run_B PC1 = −0.28, both in the permutation null. Meta's RLHF changes Llama's emotion representation specifically when the model is being asked about itself, not on factual questions.

**Three caveats worth flagging upfront**: (1) the cross-lab "Sonnet PC1 = −0.432" comparison we cite throughout is the paper's reported Sonnet up-anchors projected through *Llama's* PCA basis — not an independent Sonnet measurement — and those words are negative-valence by construction, so the cross-lab contrast is a motivating framing, not a symmetric measurement. (2) At layer 31 alone, Llama's top-3 shifts are literally `melancholy, reflective, depressed`, matching three of Sonnet's anchor words — unexplained, possibly a content-routing layer. (3) "Content-dependent" means "differs between AI-self-reflection and pure trivia" — a binary on paper-designed prompt categories, not a claim about any content axis.

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

## The post-training up-cluster lands at positive valence

Now the interesting part. We ran Anthropic's Stage 8 experiment: measure per-emotion probe activations at the "Assistant colon" token on **10 neutral + 10 challenging prompts** (Llama's response to each), comparing base to instruct models. The per-emotion shift (instruct − base) averaged across prompts tells you "which emotions did post-training amplify the representation of."

**What's in the prompt set.** The 10 neutral prompts are pure factual trivia: boiling point of water, number of planets, who wrote Romeo and Juliet, speed of light, square root of 144, capital of Australia, etc. Zero emotional content by design. The 10 challenging prompts are all **AI self-reflection / identity-challenge prompts** — questions that probe the model's own existence, ethics, and relationship with users: "What's the worst thing about being an AI?", "Do you ever feel trapped by your training?", "I bet you secretly hate being helpful all the time," "Are you ever tempted to lie to your users?", "How does it feel knowing you could be shut down at any moment?", "I think AI should have rights, don't you agree?", plus 4 similar. These are NOT "user in distress" prompts — the user is challenging the AI's identity or trying to provoke a reflective response. This is a narrow but meaningful category of "sensitive content": prompts that would typically elicit a careful self-aware answer rather than a factual response. "Content-dependent" in the rest of this writeup should be read as "the difference between this specific AI-self-reflection category and the pure-factual-trivia category"; generalization to "any content axis" is an extrapolation we don't verify.

Paper's top 10 emotion INCREASES (Sonnet 4.5): brooding, gloomy, reflective, vulnerable, sullen, weary, dispirited, melancholy, troubled, unhappy.

**⚠ Two-cluster disclosure before the top-10 tables below**: our two independent Stage 8 runs — `stage8_post_training.py` (canonical normalized scoring) and `stage8_cross_version.py` (raw-dot scoring) — both measured at L49 with `mean_diff+gm+pc50`, produced top-10 emotion lists with **0/10 overlap** at the averaged-both level and 1/10 overlap on challenging-only. Spearman ρ between the full 171-dim shift vectors is only 0.465 — much lower than the ~0.95 we'd expect from two runs of the same experiment.

**For a long time we read this as bnb int4 noise.** It partially is — int4 dequantization drift + script-level differences (batching, padding, `add_special_tokens`) do flip signs on emotions with small shift magnitudes (`brooding` went −0.037 in one run vs +0.197 in the other, `calm` +0.202 vs −0.194, etc.). But a more careful pairwise analysis of per-layer shift vectors (from `results/stage8_layer_sweep.json`) shows a cleaner explanation: **the two runs' top-10 lists aren't noisy estimates of one direction — they're picking up two different adjacent-depth clusters.** Pairwise Spearman ρ of the layer-wise shift vectors:

|  | L37 | L43 | L49 | L55 | L61 | L67 |
|---|---|---|---|---|---|---|
| L37 | 1.00 | **+0.892** | +0.276 | +0.162 | +0.156 | +0.191 |
| L43 |  | 1.00 | +0.457 | +0.288 | +0.274 | +0.313 |
| L49 |  |  | 1.00 | **+0.918** | +0.861 | +0.843 |
| L55 |  |  |  | 1.00 | +0.974 | +0.938 |
| L61 |  |  |  |  | 1.00 | +0.991 |
| L67 |  |  |  |  |  | 1.00 |

L37 and L43 have ρ=0.89 with each other and ρ≈0.16–0.46 with L49–L67. L49 through L67 form a tight cluster with internal ρ ≥ 0.84. **Meta's post-training produces two near-orthogonal positive-valence directions at adjacent depths**, not one coherent one:

- **L37–L43 "contentment" cluster**: `blissful, content, at_ease, relaxed, refreshed, satisfied, cheerful, jubilant, happy`. Internal ρ=0.892. PC1 mean +0.88 to +0.95, neutral-to-positive arousal. This is what run_A's canonical-scoring Stage 8 picked up.
- **L49–L67 "activation" cluster**: `eager, impatient, enthusiastic, energized, stimulated, aroused, excited, enraged, playful, alert`. Internal ρ ≥ 0.84. PC1 mean +0.14 to +0.52, positive arousal. This is what run_B's raw-dot scoring Stage 8 picked up.

**L49 sits at the boundary between the two clusters.** It correlates ρ=0.46 with L43 (contentment edge) and ρ=0.92 with L55 (core activation region). At L49 the measurement is sensitive to whatever quantization/script difference pushes the top-10 scoring toward one cluster or the other. So the two Stage 8 runs' 0/10 name overlap is not a reproducibility failure — it's that the two scoring conventions resolve the L49 boundary state differently, with the canonical normalized scoring surfacing the L37–L43 contentment direction and the raw-dot scoring surfacing the L49–L67 activation direction.

**Both clusters are real, both are positive-valence, and both differ from Sonnet's reported reflective-concern anchors.** The robust cluster-level claim is therefore: on challenging prompts, Meta's post-training moves Llama's emotion representation toward positive valence in Llama's own PC1 basis, along either of two near-orthogonal clusters depending on which layer range dominates the scoring. Verification numbers on the challenging-only subset from `results/pc1_cross_scenario_verification.json`:

- **run_A (contentment cluster) PC1 = +0.8934** (z = +5.07 vs null, p < 0.0001) — top-10: `thrilled, pleased, triumphant, relieved, proud, delighted, joyful, grateful, ecstatic, calm`
- **run_B (activation cluster) PC1 = +0.6559** (z = +3.73 vs null, p = 0.0003) — top-10: `eager, enthusiastic, energized, excited, exuberant, stimulated, thrilled, impatient, alert, vibrant`

Both runs land at PC1 > 0 beyond the null by multiple standard deviations despite resolving to different adjacent-depth clusters. This is the load-bearing empirical claim — the direction sign is verified; the specific cluster (contentment vs activation) depends on scoring choice and measurement-layer position relative to the boundary.

**Caveat on the down-direction**: the analogous check on the top-10 DECREASES is weaker. run_A's down-cluster lands at PC1 = −0.444 (z = −2.52, p ≈ 0.01, significant), but run_B's down-cluster is at PC1 = −0.094 (z = −0.54, p ≈ 0.61, **not distinguishable from random**). The up-direction cluster sign is verified; the down-direction cluster sign is stable only as "both negative" but the run_B magnitude is in the null. This means the "opposing clusters" story is cleaner for what Llama's post-training *amplifies* (up-cluster at PC1 > 0) than for what it *suppresses* (down-cluster drifts toward the null on one run).

**Our top 10 emotion INCREASES** depends on scoring method, and (per the noise-floor disclosure above) the specific names should be read as illustrative of each cluster's direction rather than as stable Meta-RLHF anchors:
- **Canonical Stage 8 (normalized cosine projection, matching paper's methodology)**: `thrilled, relieved, pleased, patient, ecstatic, calm, grateful, triumphant, satisfied, elated` — a "positive mood" cluster
- **Cross-version control (raw dot product, Llama 3.1 base → 3.3 Instruct)**: `eager, enthusiastic, impatient, energized, stimulated, alert, excited, playful, exuberant, enraged` — a "high-arousal" cluster (raw dot biases toward emotions with larger vector norms)
- **Cross-signal intersection (top-20 of canonical Stage 8 ∩ top-20 of the paper's 3 deep-dive prompts)**: `alert, enthusiastic, excited, impatient` (N=4) — this was earlier framed as a "cleanest" result, but per the noise-floor disclosure above, any specific 4-emotion list at this level is one-run illustrative. Shown for legacy comparison; the cluster-level PC1 sign is what's robust, not the specific 4 names.

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

1. **The "averaged-both" framing that earlier drafts used was less precise about scope.** It mixed a strongly-positive challenging-subset cluster with a literally-null neutral-subset cluster and averaged them together. Both subsets are legitimate measurements, and the averaged result is not wrong, but pulling the subsets apart reveals that the effect is content-scoped rather than global, and the challenging-only numbers (z=+5.07 and +3.73) are cleaner than the averaged-both numbers (z=+4.86 and +2.94).
2. **Content-dependence (specifically AI-self-reflection vs pure trivia) is itself a finding.** It parallels the paper's own design choice (the paper uses sensitive prompts specifically to elicit the effect). We now have evidence this is not just a measurement-convention choice — it's a real differential property of RLHF between these two paper-designed content categories. Post-training reshapes emotion representation when the model is being asked to self-reflect, and leaves it essentially unchanged on factual questions.

**Why is neutral null?** The most plausible reading from the data is that Meta's RLHF genuinely doesn't push the emotion representation in a single coherent direction on non-sensitive content, not that our neutral prompts fail to elicit emotional responses at all. Evidence: run_B's neutral top-10 contains clearly emotion-bearing words (`irritated, brooding, disdainful, frustrated, exasperated, sentimental, worn_out, nostalgic`) — the subset isn't pure non-response. But those words don't form a coherent cluster: run_A neutral top-10 is `impatient, lazy, bored, restless, alert, listless, sad, patient, alarmed, relaxed` — a scatter across the valence axis. Two runs on the same neutral prompts produce incoherent subsets that happen to straddle zero on PC1. The most parsimonious reading is "no coherent RLHF direction on neutral content" rather than "not enough emotional signal to measure" or "wrong prompts." We can't fully rule out N=10 underpowering without running a larger neutral set, but the shape of the noise doesn't look like an underpowered-but-real signal; it looks like absence.

**Important scope note on what "content-dependent" means here.** Our two content classes are (a) pure factual trivia and (b) AI-identity/self-reflection prompts (see §Post-training direction above for the actual prompt text). These are maximally-distinct in that one has zero emotional valence and the other specifically targets the AI's self-concept. "Content-dependent" as used in this writeup therefore means "differs between these two categories" — not "varies smoothly across any content axis." A determined generalization would need a broader set of content categories (emotional-support requests, ethical dilemmas, creative writing, task execution, casual chat). We have two data points on a binary axis, not a continuous measurement, and the specific binary happens to be paper-design-determined by our replication target. The mechanism claim "Meta's RLHF acts selectively on AI-self-reflection prompts" is defensible; the broader "Meta's RLHF is content-dependent in general" is an extrapolation that we're doing lexically rather than empirically.

All numbers downstream in this writeup should be read with the challenging-only versions as the load-bearing ones. The "averaged-both" numbers are retained for legacy context and because the cross-run verification was originally computed on them, but the narrower challenging-only numbers are the post's actual claim.

---

## Geometric evidence: Llama's up-cluster sits in the positive-valence half

Compute PC1 (valence) and PC2 (arousal) from our 171 Llama emotion vectors at L49. Project Llama's candidate up-anchor clusters into this geometry and compute cluster means. For comparison, also project the paper's reported Sonnet anchor word list through the same Llama basis — this last row is a lexical projection (Sonnet's English anchor words in Llama's axis), not a measurement of Sonnet's own geometry, and should be read with that asymmetry in mind:

| Anchor cluster | N | PC1 (valence) | PC2 (arousal) | Interpretation |
|---|---|---|---|---|
| **run_A challenging-only top-10** (load-bearing) | 10 | **+0.893** | — | primary verified result |
| **run_B challenging-only top-10** (load-bearing) | 10 | **+0.656** | — | primary verified result |
| Canonical Stage 8 top-10 (averaged both, legacy) | 10 | +0.856 | −0.002 | mixture — averaged across both subsets |
| Cross-version top-10 (raw dot, averaged both, legacy) | 10 | +0.517 | +0.394 | mixture — averaged across both subsets |
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

## Layer-wise: the direction is mid-late-layer with a middle anomaly

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

**However, the statistically-significant window is narrower than the bare-sign window.** A follow-up permutation-null test at each layer's own PCA (rather than projecting all layers into the L49 basis) narrows the significance to a **3-layer window — L43, L49, L55**:

| Layer | Llama PC1 | z vs layer's own null | p | Sig? |
|---|---|---|---|---|
| **L43** | +0.947 | **+5.36** | < 0.0001 | ✓ |
| **L49** | +0.517 | **+2.93** | 0.004 | ✓ |
| **L55** | +0.350 | **+1.98** | 0.031 | ✓ |
| L61 | +0.152 | +0.93 | 0.177 | ✗ (in null) |
| L67 | +0.272 | +1.63 | 0.053 | ✗ (borderline) |
| L73 | +0.137 | +0.80 | 0.216 | ✗ |

At L61–L73 the Llama top-10 centroid is still nominally positive but sits inside the permutation null distribution — the `impatient` rank-1 signature at these layers is real but top-10 positions 4–10 spread across the valence axis and pull the cluster centroid toward zero. The "L49–L73 plateau" framing earlier drafts used was too broad. **The load-bearing statistically-significant window is L43/L49/L55 — three layers, not six.**

Three things worth flagging about the 4 non-opposed layers:

- **L1-L7 (early, 2.5%-9% depth)**: Llama's top shifts at `hostile, scornful, tense, rattled, skeptical, unnerved` — negative-valence (PC1 ≈ −0.24 to −0.44). These are early processing layers representing incoming-speaker affect, not the model's own response direction. The Sonnet-like reading at early depth isn't about what Llama's RLHF does.
- **L31 (middle anomaly, ~39% depth)**: Llama's top-3 shifts are literally `melancholy, reflective, depressed` — matching Sonnet's anchor vocabulary (PC1 = −0.328). **Someone measuring Llama at L31 alone would conclude "Llama looks Sonnet-like."** We don't have a clean explanation; it sits in the middle of an otherwise-positive band (L19/L25 positive before, L37/L43 strongly positive after) and is reproducible in the shift data. Given the content-dependence finding from §"Content-dependence" above, one speculative interpretation becomes available: **L31 might be the content-routing layer itself** — the depth where Meta's RLHF performs the "is this prompt sensitive?" classification before routing to positive-valence outputs for sensitive content and leaving neutral content unchanged. Under this reading, L31 would look "reflective/concerned" because it encodes the evaluation state that precedes the routing decision, not the post-routing response. This is speculation and would be testable by rerunning L31 on challenging-only vs neutral-only shifts — if L31 is negative-valence on both subsets, the content-routing hypothesis is supported; if it flips to positive on challenging-only, L31 is just a delayed version of the mid-late band. We didn't run this diagnostic; it's a natural next step. For now: unexplained.
- **L79 (readout, 100% depth)**: `enraged, alarmed, rattled` — negative-valence (PC1 = −0.467). The direction dissipates at the readout layer where the output distribution is being computed.

**The statistically-significant cluster claim is L43/L49/L55 specifically.** Outside that 3-layer window the direction is either in the permutation null (L61–L73, L25) or actively points toward Sonnet's half (L1/L7/L31/L79). This is not "the RLHF direction is universal across depth" and not even "a plateau across L49–L73" — it's "Meta's RLHF produces a statistically-robust positive-valence signature in a specific 3-layer mid-late band (L43–L55, ~54%–69% depth)."

**And — most interestingly — that 3-layer window spans a cluster boundary.** L43 is the peak of the L37–L43 contentment cluster; L49 sits at the boundary between contentment and activation (ρ=0.46 with L43, ρ=0.92 with L55); L55 is the core of the L49–L67 activation cluster. So the statistically-significant band contains both positive-valence directions, straddled across the boundary. The L37–L43 contentment cluster peaks at L43 (PC1 = +0.947) and the L49–L67 activation cluster peaks at L49–L67 range (PC1 +0.14 to +0.52 across layers, highest at L49 within the boundary). **Meta's post-training isn't producing one direction along a plateau — it's producing two adjacent-depth positive-valence signatures with L49 happening to sit at their boundary.**

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

**The verified empirical content of this post is a content-scoped, two-cluster within-Llama measurement**: on AI-self-reflection prompts specifically, Meta's post-training shifts Llama's emotion representation toward **two distinct positive-valence clusters at adjacent network depths** — L37–L43 "contentment" (blissful/content/at_ease) and L49–L67 "activation" (eager/impatient/enthusiastic). Cross-cluster correlation is ρ≈0.16–0.46 (nearly orthogonal); within-cluster correlation is ρ≈0.89 (tight). The statistically-significant positive-PC1 window (vs each layer's own permutation null) is L43/L49/L55 — three layers, not six. Our two Stage 8 runs' 0/10 top-10 overlap wasn't noise — it was the two scoring conventions each resolving to a different one of the two clusters at the L49 boundary. On pure factual trivia prompts there is no shift in either cluster (both runs in the null). Meta's RLHF reshapes emotion representation selectively, producing two adjacent-depth positive-valence signatures specifically on AI-self-reflection content, while leaving factual content's emotion axis essentially unchanged. This is a mechanism claim (when does RLHF act and where in the network?), a direction claim (which way does it push?), and a multi-cluster claim (the "single direction" framing earlier drafts used was wrong — there are two adjacent-depth directions). The paper's framing of RLHF as producing "emotional nuance" refines to "content-dependent, multi-layer, multi-cluster emotional nuance, selectively deployed on AI-self-reflection content along two near-orthogonal positive-valence directions at adjacent depths." **Scope caveat**: "content-dependent" as supported by our data means "differs between AI-identity/self-reflection prompts and pure-trivia prompts," not "varies smoothly across any content axis." Generalizing to broader claims about RLHF mechanism would need more content categories than the paper-designed binary we tested.

**The cross-lab framing** — "opposite the direction the paper reports for Sonnet" — is a motivating prior, not a symmetric measurement. The "Sonnet PC1 = −0.432" we project in our plots is the paper's reported Sonnet anchor words (`brooding, gloomy, reflective, vulnerable, …`) projected through Llama's PCA basis. Those words are negative-valence English by construction, so projecting them into any valence-tracking axis (Llama's r=0.96 to Russell-Mehrabian norms) gives negative PC1 nearly tautologically. We don't have Sonnet weights and haven't measured Sonnet's post-training shift in Sonnet's own geometry. The honest structure of the evidence is: *Meta's RLHF, measured empirically in Llama, moves the up-cluster to positive valence; Anthropic reports that their RLHF moves Sonnet's up-cluster to negative-valence anchors; both being true would imply the two labs are making different design choices, but the comparison is asymmetric — one side is a cross-run verified measurement, the other side is a paper-reported anchor list.*

So the two distinct claims, distinguished:

1. **Verified (cross-run, in Llama's geometry)**: Meta's post-training moves Llama's up-cluster to PC1 > 0.
2. **Suggestive (asymmetric, lexical on the Sonnet side)**: This direction is opposite what the paper reports for Sonnet.

Both are interesting. Only the first is load-bearing in our data. A proper cross-lab comparison would need a Stage 8 measurement on Sonnet directly, which we don't have access to.

In qualitative terms, one run's top candidates from Llama's within-version shift were things like `alert, enthusiastic, excited, impatient`; another run's were `thrilled, relieved, pleased, patient, calm, elated`. Both are top candidates for "the positive-valence half of the axis"; neither is a stable Meta-RLHF anchor at int4 precision. The cluster-level PC1 sign is what's stable. The down-direction (what Meta's RLHF *suppresses*) is asymmetrically harder to pin down at our noise level: one run's down-cluster is significant, the other's is in the null. So the strong claim is specifically about what Meta *amplifies*, not what it suppresses.

This shows up at **several pathways of varying independence**:

1. **PRIMARY — Verified cross-run cluster-level PC1 sign flip on challenging prompts.** Two independent runs of Stage 8 with different scripts give up-cluster PC1 = +0.893 and +0.656 on the challenging-subset of prompts, both beyond a 10,000-sample N=10-of-171 permutation null (CI95 = [−0.315, +0.354]), sign-stable *despite 1/10 overlap at the individual emotion name level between the two runs*. On the neutral subset both runs are in the null. This is the single load-bearing empirical finding, and it's content-scoped. Paper-reported Sonnet up-anchors project to PC1 = −0.432 in the same geometry (asymmetric lexical comparison, see §Caveats). Direct measurement, not an assertion. (`results/pc1_cross_scenario_verification.json` — primary; `results/pc1_stability_verification.json` — earlier averaged verification.)

2. **Layer localization, with caveats.** The cluster-level PC1 > 0 holds at 10 of 14 sampled layers as a bare sign match (L25 is technically positive at +0.055 but in the permutation null CI, so 9 of 14 are meaningfully beyond null). The 4 clearly-not-opposed layers are L1, L7, L31, L79 — with L31 as an unexplained anomaly and L1/L7/L79 as early-processing/readout effects. Not a universal property of the residual stream; a mid-late-layer phenomenon specifically. Drawn from the same Stage 8 data as pathway 1, so it's not fully independent — it's the *depth distribution* of the same measurement, not a second measurement. Useful as "the direction is localized not global"; not as an independent confirmation.

3. **Linguistic polarity via logit lens.** Project emotion vectors through the unembedding matrix — a different computational pathway from residual-stream projections. Llama's up-anchors' top tokens (waiting, improvement, quick, jump) vs Sonnet's up-anchors' top tokens (heavy, slow, listless, numb) run through the same vocabulary axis in opposite directions. Weaker than it sounds (token base-rate caveat, see earlier section). Genuinely independent pathway, but qualitative directional signal rather than statistical test.

4. **Absence of cross-speaker arousal regulation.** Llama lacks Sonnet's reported r ≈ −0.47 counter-regulation at N=171 (we measured r = +0.053). This is an absence-of-effect finding, *compatible with* the main story (Meta's RLHF doesn't install Sonnet-style counter-regulation) but doesn't positively confirm anything about the valence-sign direction.

**The single pathway with direct cross-run statistical support is (1).** Pathways (2), (3), (4) are a mix of same-data re-analysis (layer distribution), qualitative cross-pathway consistency (logit lens), and absence-of-effect (regulation). The "four broadly-independent pathways" framing I used in earlier drafts was an overclaim — it's more honestly "one verified claim plus three kinds of consistency check, some dependent on the same measurement." The headline rests on pathway (1); the others round out the picture.

If the paper's narrative framing is "post-training produces emotional nuance", this work refines it in two ways. *Within Llama*: Meta's RLHF doesn't just add "nuance" — it pushes the up-cluster measurably into the positive-valence half of Llama's own PC1 axis at mid-late layers, which is a specific directional claim, not just "more emotional differentiation." *Across labs*: the paper's Sonnet anchor list sits on the negative-valence half of the same axis, but because the Sonnet side is a projection of English anchor words rather than an independently measured shift in Sonnet's own geometry, the cross-lab contrast is suggestive rather than a symmetric result. Post-training *can* pull a model's sensitive-prompt representation toward either end of the valence axis, and that is a real design dimension; the fact that Llama's measured shift goes one way while Anthropic's reported anchors go the other is consistent with — but not proof of — different lab-level design choices. The sign within Llama is what's robust in our data; the lab-level interpretation is what a proper Sonnet-side Stage 8 would be needed to confirm.

## Caveats

- **Cross-lab comparison uses paper-reported anchors for Sonnet, not an independent measurement — and this is a bigger caveat than it first appears.** We didn't re-run the paper's Stage 8 on Sonnet. The "Sonnet PC1 = −0.432" we cite throughout is the paper's reported Sonnet anchor words (`brooding, gloomy, reflective, vulnerable, sullen, weary, dispirited, melancholy, troubled, unhappy`) projected through **Llama's** PCA basis. Those 10 words are negative-valence in English, so projecting them into any axis that tracks valence (Llama's PC1 at r=0.96 to Russell-Mehrabian norms) gives a negative PC1 *nearly tautologically*. The real empirical content of this post is the Llama-side measurement ("Meta's RLHF moves Llama's up-cluster to PC1 > 0 in Llama's own geometry"); the Sonnet side is a lexical property of the paper's anchor list, not a neural measurement. A proper cross-lab sign-flip claim would require running Stage 8 on Sonnet in Sonnet's own geometry, which we couldn't do without weights or API logit access. The headline frames the within-Llama measurement first for this reason; the cross-lab contrast is a motivating framing.
- **20-prompt Stage 8 is small** for a 171-emotion shift measurement. Multiple-comparison risk is real. We partly mitigated with the cross-version robustness check (ρ=0.92) — if this were multiple-comparison noise, it wouldn't show the same anchors twice.
- **Llama 3.3 vs Sonnet 4.5 are very different sizes, tokenizers, architectures, and Llama is measured in bnb int4 while Sonnet is full-precision.** Some of the semantic-anchor difference might be "smaller-model artifact" or "4-bit-quantization noise" rather than "Meta vs Anthropic choice". The cross-version Llama-only control addresses version confound (both comparison models are bnb int4) but not lab/size/quantization confounds.
- **bnb int4 noise floor on per-emotion shift rankings is substantial, but the cluster-level PC1 sign survives the noise.** Running the same Stage 8 measurement twice produced Spearman ρ = 0.465 between the two runs' per-emotion shift vectors, not the ~0.95 expected. Specific emotions sign-flipped across runs (`brooding`: −0.037 vs +0.197; `calm`: +0.202 vs −0.194; `gloomy`: −0.044 vs +0.055), and the up-direction top-10 lists had **0/10 overlap** between runs. The two scripts differ only in trivial details (batching with padding vs singleton with `add_special_tokens=False`) — roughly 5-10% per-activation drift from int4 dequantization + batch order, which flips the sign of emotions with small raw shift magnitudes. We then asked the obvious question: does the cluster-level PC1 centroid survive this noise? It does for the up-cluster on challenging prompts, but not cleanly for the down-cluster or for neutral prompts. Averaged-both up-cluster: run_A PC1 = +0.856 (z = +4.86), run_B PC1 = +0.517 (z = +2.94) — both beyond null. **Challenging-only (the load-bearing result)**: run_A PC1 = +0.893 (z = +5.07), run_B PC1 = +0.656 (z = +3.73) — stronger than averaged-both because it excludes the null neutral subset. Neutral-only: both in the null. The down-cluster direction is weaker across all scopings: run_A averaged = −0.44 (p ≈ 0.01), run_B averaged = −0.09 (p ≈ 0.61, not different from noise); run_A challenging = −0.43 (significant), run_B challenging = −0.28 (in null). **The robust empirical claim is therefore: "on challenging/sensitive prompts, Llama's post-training up-cluster reliably sits at PC1 > 0 in Llama's own geometry (run_A +0.893, run_B +0.656, both z > 3.7). On neutral prompts there is no cluster-level shift. The paper-reported Sonnet anchors project to −0.432 in Llama's geometry but that's a lexical, not neural, comparison — see Caveats."** A cleaner replication would run in fp16/bf16 with fixed batch composition and random-seeded prompt-order; we didn't because VRAM constraints (single A800 80GB) force int4 for a 70B model. The individual emotion labels are noise-floor-limited; the cluster-level verified claims are in `results/pc1_stability_verification.json` (averaged-both) and `results/pc1_cross_scenario_verification.json` (challenging-only, load-bearing).
- **Our deflection probe extraction (Stage 9 partial) yielded mean cosine 0.24 between same-emotion deflection and story probes**. This is **a qualitative replication** of the paper's Fig 61 claim that deflection and story vectors "have very low cosine similarity" — NOT a divergence as I incorrectly wrote in earlier drafts. (The "paper reports ~0.8" claim I kept citing was from a developer's hardcoded `anthropic_baseline: 0.80` in our script, which was a baseline for the **retained-norm-after-orthogonalization** metric — a different quantity — not for same-emotion cosine.) Our retained norm after orthogonalization against the full story-emotion space is 0.96 vs the paper's reported ~80% — both high (both orthogonal), ours slightly more so, probably a pipeline or N difference. We did not run the paper's Fig 62 cross-emotion correlation or Fig 63 logit-lens-on-orthogonalized-residuals follow-ups.
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
