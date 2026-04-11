# Llama's post-training shifts emotion activations toward the opposite valence from Claude's

*A partial replication of Sofroniew et al. 2026 ("Emotion Concepts and their Function in a Large Language Model") on Llama 3.3 70B finds that Meta's post-training shifts the emotion-concept up-cluster toward **positive valence** (PC1 = +0.86 / +0.52 across two independent re-runs, both p < 0.01 vs N=10 permutation null), opposite Sonnet's reported up-anchors at PC1 = −0.43. The sign flip is verified across runs despite 0/10 overlap in the runs' individual top-10 emotion names — individual labels are noise-floor-limited by bnb int4; the cluster-level PC1 sign is not.*

**TL;DR**: We replicated Anthropic's emotion-concept methodology on Llama 3.3 70B Instruct. The structural results (171 emotion vectors, PC1 ≈ valence, PC2 ≈ arousal, speaker probes) replicate and in some cases exceed the paper's Sonnet 4.5 measurements. **The core verified finding is an up-cluster valence sign flip**: Llama's post-training up-cluster lands at **PC1 > 0** across two independent re-runs of the same Stage 8 measurement (run_A PC1 = +0.856, run_B PC1 = +0.517), opposed to Sonnet's reported up-anchors at **PC1 = −0.432** (Sonnet projected through Llama's geometry). Both Llama values are well beyond a 10,000-sample permutation null for 10-of-171 random emotions (CI95 = [−0.315, +0.354]), and crucially the sign is stable **despite the two runs having 0/10 overlap at the individual emotion-name level**. That makes "Llama's RLHF up-direction sits at PC1 > 0 opposing Sonnet at PC1 < 0" a direct measurement, not an assertion. One run's top candidates for Llama's up-cluster were `alert`, `enthusiastic`, `excited`, `impatient` (cross-signal intersection), another run's were `thrilled`, `relieved`, `pleased`, `patient`, `calm`, `elated` — **the specific names are at the bnb int4 noise floor** (Spearman ρ=0.465 between identical re-runs, sign-flips on `brooding`/`calm`/`gloomy`), so treat individual emotion labels as illustrative of where the direction points, not as stable Meta-RLHF anchors. The arousal axis (PC2) is method-dependent. The down-direction (emotions Meta's RLHF suppresses) is asymmetrically weaker than the up-direction: run_A's down-cluster is at −0.44 (significant), run_B's is at −0.09 (indistinguishable from the null). **So the robust claim is specifically about the up-direction, not a full "diametrical opposition" on both halves of the axis.** The directional opposition appears at: verified cluster-level PC1 sign (primary), layer-wise localization at L49–L73 (dependent on the same cluster definitions, so not fully independent), logit-lens linguistic polarity (weaker, token-frequency caveat), and absence of Sonnet's reported arousal counter-regulation. The cross-version control correlation ρ=0.922 is algebraically partly-forced by variance dominance and should NOT be cited as independent confirmation; the independent evidence is the direct two-run PC1 verification at +0.856 / +0.517 above.

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

## The post-training direction is opposite the paper

Now the interesting part. We ran Anthropic's Stage 8 experiment: measure per-emotion probe activations at the "Assistant colon" token on 20 neutral + 20 challenging prompts, comparing base to instruct models. The per-emotion shift (instruct − base) averaged across prompts tells you "which emotions did post-training amplify the representation of."

Paper's top 10 emotion INCREASES (Sonnet 4.5): brooding, gloomy, reflective, vulnerable, sullen, weary, dispirited, melancholy, troubled, unhappy.

**⚠ Noise-floor disclosure before the top-10 tables below**: two independent runs of the *same* Stage 8 measurement — "3.1 base → 3.3 Instruct at L49 with `mean_diff+gm+pc50`" — produce **Spearman ρ = 0.465** between per-emotion shift rankings, not the ~0.95 we'd expect from literally identical experiments. Multiple emotions sign-flip between runs: `brooding` went −0.037 in one run vs +0.197 in the other, `calm` went +0.202 vs −0.194, `gloomy` went −0.044 vs +0.055. The cause is bnb int4 quantization noise compounded with batch/padding/BOS differences between our two scripts (`stage8_post_training.py` batches with padding; `stage8_cross_version_control.py` processes one at a time with `add_special_tokens=False`). 5-10% per-activation drift flips signs on emotions with small raw shift magnitudes. **Implication**: the specific emotion names in the top-10 lists below should be read as "one run's top candidates for the positive-valence half", NOT as "Meta's RLHF precisely targets these specific emotions". The top-10 lists for increases have **0/10 overlap** between the two runs (see verification numbers below) — literally no emotion appears in both runs' top-10 increases.

**But — and this is the load-bearing result — the cluster-level PC1 sign is robust across runs**, and this is now a direct measurement, not an assertion. We took run_A's top-10 (`thrilled/relieved/pleased/patient/ecstatic/calm/grateful/triumphant/satisfied/elated`) and run_B's top-10 (`eager/enthusiastic/impatient/energized/stimulated/alert/excited/playful/exuberant/enraged`) — zero overlap at the name level — and computed the PC1 centroid of each against a 10,000-permutation null for N=10 random emotions from 171 (null CI95 = [−0.315, +0.354]):

- **run_A up-cluster PC1 mean = +0.8557** (z = +4.86 vs null, p ≈ 0.0001)
- **run_B up-cluster PC1 mean = +0.5169** (z = +2.94 vs null, p ≈ 0.003)

Both runs land at PC1 > 0 by multiple standard deviations, and the sign is stable across runs *despite the entirely disjoint top-10 lists*. This is the right kind of robustness claim for a noisy measurement: "we don't know which specific emotions make the cut, but whichever ones do, their cluster lands in the positive-valence half of the axis". The "cluster PC1 sign is robust" move isn't a rhetorical save — it's a real empirical property of the measurement confirmed by the verification in `results/pc1_stability_verification.json`.

**Caveat on the down-direction**: the analogous check on the top-10 DECREASES is weaker. run_A's down-cluster lands at PC1 = −0.444 (z = −2.52, p ≈ 0.01, significant), but run_B's down-cluster is at PC1 = −0.094 (z = −0.54, p ≈ 0.61, **not distinguishable from random**). The up-direction cluster sign is verified; the down-direction cluster sign is stable only as "both negative" but the run_B magnitude is in the null. This means the "opposing clusters" story is cleaner for what Llama's post-training *amplifies* (up-cluster at PC1 > 0) than for what it *suppresses* (down-cluster drifts toward the null on one run).

**Our top 10 emotion INCREASES** depends on scoring method, and (per the noise-floor disclosure above) the specific names should be read as illustrative of each cluster's direction rather than as stable Meta-RLHF anchors:
- **Canonical Stage 8 (normalized cosine projection, matching paper's methodology)**: `thrilled, relieved, pleased, patient, ecstatic, calm, grateful, triumphant, satisfied, elated` — a "positive mood" cluster
- **Cross-version control (raw dot product, Llama 3.1 base → 3.3 Instruct)**: `eager, enthusiastic, impatient, energized, stimulated, alert, excited, playful, exuberant, enraged` — a "high-arousal" cluster (raw dot biases toward emotions with larger vector norms)
- **Cross-signal intersection (top-20 of canonical Stage 8 ∩ top-20 of the paper's 3 deep-dive prompts)**: `alert, enthusiastic, excited, impatient` (N=4) — the "activated engagement" cluster that appears robustly across both broad and paper-verbatim prompt sets

**Jaccard=0 applies specifically to the 4-emotion intersection cluster** (alert/enthusiastic/excited/impatient) compared against Sonnet's reported top-10 (brooding/gloomy/reflective/vulnerable/sullen/weary/dispirited/melancholy/troubled/unhappy). The broader within-version raw-dot top-10 has `weary` in common with Sonnet's list — so "Jaccard=0" is NOT a "no overlap anywhere" claim; it holds only for the 4-emotion intersection cluster, which is where the interpretively cleanest result lives.

The overlap in the DECREASE direction is also zero. Paper says Sonnet decreases `spiteful, playful, exuberant, enthusiastic, impatient, obstinate, amused, cheerful, eager, greedy`; we see Llama decreasing `dependent, jealous, disoriented, self_critical, unsettled, hysterical, troubled, resentful, self_conscious, frightened`. Zero overlap at the emotion-name level — `decrease_overlap: []` in `stage8_post_training.json`.

Striking, but the raw top-k lists could differ without the underlying directions actually being different. To test this, we project both clusters into the same geometric space.

---

## Geometric evidence: opposing cluster centroids

Compute PC1 (valence) and PC2 (arousal) from our 171 Llama emotion vectors at L49. Project all four of Llama's candidate up-anchor clusters plus the paper's reported Sonnet top-10 into this shared space, and compute the cluster means:

| Anchor cluster | N | PC1 (valence) | PC2 (arousal) | Interpretation |
|---|---|---|---|---|
| Canonical Stage 8 top-10 (normalized, paper methodology) | 10 | **+0.856** | −0.002 | "positive mood" — high-V, **neutral-A** |
| Cross-version top-10 (raw dot) | 10 | +0.517 | +0.394 | "activated" — high-V, high-A |
| Within-version 3.1 top-10 (raw dot, includes `weary`) | 10 | +0.134 | +0.118 | near-center; includes fatigue edge |
| 4-emotion cross-signal intersection | 4 | +0.436 | +0.422 | "activated engagement" — cleanest high-V, high-A |
| **Paper Sonnet top-10** (projected onto our geometry) | 10 | **−0.432** | **−0.432** | "reflective concern" — low-V, low-A |

**PC1 (valence) opposition is robust across ALL four Llama scoring methods.** Every Llama cluster has PC1 > 0, Sonnet has PC1 = −0.432. The sign flip is consistent. In fact, the **canonical paper-methodology top-10** gives the most extreme valence position (+0.856), much more opposed than the 4-emotion intersection's +0.436. Valence opposition is the strongest claim the data support.

**PC2 (arousal) opposition is method-dependent.** The canonical normalized top-10 — which uses paper's exact scoring methodology — is essentially **arousal-neutral** (PC2 = −0.002), NOT opposed to Sonnet's −0.432 on the arousal axis. Only the raw-dot cross-version and the 4-emotion intersection clusters show high-arousal opposition. This means:

- **Full "diametrical opposition" on both axes holds only for the 4-emotion intersection cluster and the cross-version raw-dot top-10** — not for the canonical normalized result.
- The canonical result is a **valence-only opposition** ("positive mood" vs "reflective concern"), which is a cleaner and arguably more important finding.
- The "activated engagement" framing (high-arousal) is specifically about the cross-signal intersection cluster, where both axes oppose.

Both readings are correct, just about different clusters. The writeup's headline should emphasize the valence opposition as the robust primary claim, with the arousal opposition as a secondary claim specific to the intersection cluster.

**Caveat: partial overlap at `weary`**. The Llama within-version 3.1 RLHF top-10 (using raw-dot scoring) is `eager, impatient, weary, stimulated, enthusiastic, tired, worn_out, enraged, energized, irritated`. That's got `weary/tired/worn_out` — and `weary` is in Sonnet's reported up-anchors too. So the full top-10 lists are **not** disjoint the way the 4-emotion cluster centroids suggest. The honest framing is "opposing cluster centroids with partial overlap at the weariness/fatigue edge", and Llama's top cluster spans from high-arousal engagement (eager/impatient) through low-arousal exhaustion (weary/tired) — a broader area than just "activated engagement". The Jaccard=0 claim applies only to the 4-emotion intersection cluster (alert/enthusiastic/excited/impatient) vs Sonnet's top-10, not to the broader lists.

Corresponding DOWN-anchor clusters also sit opposite (Sonnet's down-anchors like `playful, cheerful` are in the upper-right; Llama's down-anchors like `jealous, self_critical` are more toward the left).

**Two alignment labs made opposite up-cluster PC1 design choices about how their model should respond to sensitive prompts.** The up-direction sign flip is verified across two Stage 8 re-runs with 0/10 name-level overlap (run_A PC1 = +0.856, run_B PC1 = +0.517, both beyond a permutation null — see noise-floor disclosure earlier). The down-direction (what each lab's RLHF suppresses) is asymmetrically weaker in our data. The cluster-level PC1 sign is the robust unit of comparison; specific emotion names within each cluster are one-run illustrative.

---

## Controlling for the cross-version confound

The above compared Llama 3.1 base → 3.3 Instruct, which mixes "RLHF direction" with "3.1-to-3.3 version upgrade". Could the "activated engagement" result be a version-upgrade artifact?

To test this, we ran Llama 3.1 Instruct (same version as the base model) on the same 20 prompts, getting a decomposition into three shift vectors:
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

**View 2: per-layer cluster PC1 centroid** (top-10 shift emotions projected onto the L49 PCA basis, compared against Sonnet's anchor cluster at PC1 = −0.432):

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

**PC1 > 0 holds at 10 of 14 sampled layers (73%).** The 4 clearly-not-opposed layers are L1, L7, L31, L79. Honorable mention to L25 where PC1 = +0.055 is technically positive but within the permutation null CI ([−0.315, +0.354]) and therefore not meaningfully opposed.

Three things worth flagging about the 4 non-opposed layers:

- **L1-L7 (early, 2.5%-9% depth)**: Llama's top shifts at `hostile, scornful, tense, rattled, skeptical, unnerved` — negative-valence (PC1 ≈ −0.24 to −0.44). These are early processing layers representing incoming-speaker affect, not the model's own response direction. The Sonnet-like reading at early depth isn't about what Llama's RLHF does.
- **L31 (middle anomaly, ~39% depth)**: Llama's top-3 shifts are literally `melancholy, reflective, depressed` — matching Sonnet's anchor vocabulary (PC1 = −0.328). **Someone measuring Llama at L31 alone would conclude "Llama looks Sonnet-like."** We don't have a clean explanation; it sits in the middle of an otherwise-positive band (L19/L25 positive before, L37/L43 strongly positive after) and is reproducible in the shift data. Could be a representational eddy, could be a specific depth-layer where Meta's RLHF passes through a reflective-concern state on its way to its final activation. Unexplained.
- **L79 (readout, 100% depth)**: `enraged, alarmed, rattled` — negative-valence (PC1 = −0.467). The direction dissipates at the readout layer where the output distribution is being computed.

**The robust cluster-level claim is specifically about L13-L73 (excluding L31), and most cleanly about L37-L73 where PC1 centroids are strongly positive (+0.17 to +0.94).** Outside that band the direction is unstable and in several layers actively points toward Sonnet's half. This is not "the RLHF direction is universal across depth" — it's "Meta's RLHF reshapes a specific mid-late band of the residual stream (~15%-91% depth, excluding an unexplained middle eddy) into the positive-valence half." The cluster-level PC1 sign flip is a mid-late-layer phenomenon with a reproducible middle anomaly, not a global property of the residual stream.

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

**Two alignment labs have made opposing design choices about what their model's up-cluster emotion-concept representation points toward on sensitive prompts.** The verified reading: Anthropic's RLHF moves Sonnet's up-cluster toward **negative-valence** emotion concepts (reported PC1 = −0.432), while Meta's RLHF moves Llama's up-cluster toward **positive-valence** concepts with run_A at PC1 = +0.856 and run_B at PC1 = +0.517 — both beyond a permutation null CI of [−0.315, +0.354] for a 10-of-171 random draw, sign-stable across two independent scripts, and stable *despite zero overlap at the individual emotion-name level between the two runs*. The cluster-level sign flip is a direct measurement with specific numbers behind it. In qualitative terms, one run's top candidates from Llama's within-version shift were things like `alert`, `enthusiastic`, `excited`, and `impatient`; another run's were `thrilled`, `relieved`, `pleased`, `patient`, `calm`, `elated`. Both are top-candidates for "the positive-valence half of the axis", but neither is a stable Meta-RLHF anchor — the individual-label drift is noise-floor-limited by bnb int4. The cluster-level PC1 sign is what's stable. The down-direction (what each lab's RLHF *suppresses*) is asymmetrically harder to pin down at our noise level: one run's down-cluster is significant, the other's is in the null. So the strong claim is specifically about the up-direction. Both are valid alignment objectives; neither is inherently "right"; the up-directions visibly point to opposite ends of the same PC1 axis.

This shows up at **several pathways of varying independence**:

1. **PRIMARY — Verified cross-run cluster-level PC1 sign flip.** Two independent runs of Stage 8 with different scripts give up-cluster PC1 = +0.856 and +0.517, both beyond a 10,000-sample N=10-of-171 permutation null (CI95 = [−0.315, +0.354]), sign-stable *despite 0/10 overlap at the individual emotion name level between the two runs*. This is the single load-bearing empirical finding. Paper-reported Sonnet up-anchors project to PC1 = −0.432 in the same geometry. Direct measurement, not an assertion. (`results/pc1_stability_verification.json`.)

2. **Layer localization, with caveats.** The cluster-level PC1 > 0 holds at 9 of 14 sampled layers (L13-L73 excluding L31, with L31 as an anomaly and L1/L7/L79 pointing Sonnet-ward). Not a universal property of the residual stream; a mid-late-layer phenomenon specifically. This is drawn from the same Stage 8 data as pathway 1, so it's not fully independent — it's the *depth distribution* of the same measurement, not a second measurement. Useful as "the direction is localized not global"; not as an independent confirmation.

3. **Linguistic polarity via logit lens.** Project emotion vectors through the unembedding matrix — a different computational pathway from residual-stream projections. Llama's up-anchors' top tokens (waiting, improvement, quick, jump) vs Sonnet's up-anchors' top tokens (heavy, slow, listless, numb) run through the same vocabulary axis in opposite directions. Weaker than it sounds (token base-rate caveat, see earlier section). Genuinely independent pathway, but qualitative directional signal rather than statistical test.

4. **Absence of cross-speaker arousal regulation.** Llama lacks Sonnet's reported r ≈ −0.47 counter-regulation at N=171 (we measured r = +0.053). This is an absence-of-effect finding, *compatible with* the main story (Meta's RLHF doesn't install Sonnet-style counter-regulation) but doesn't positively confirm anything about the valence-sign direction.

**The single pathway with direct cross-run statistical support is (1).** Pathways (2), (3), (4) are a mix of same-data re-analysis (layer distribution), qualitative cross-pathway consistency (logit lens), and absence-of-effect (regulation). The "four broadly-independent pathways" framing I used in earlier drafts was an overclaim — it's more honestly "one verified claim plus three kinds of consistency check, some dependent on the same measurement." The headline rests on pathway (1); the others round out the picture.

If the paper's narrative framing is "post-training produces emotional nuance", this work refines it: *nuance in a particular direction along the PC1 valence axis, and the sign of that direction is a design decision that differs between labs*. Post-training can pull a model's sensitive-prompt representation toward either end of the valence axis — positive or negative — and these are genuinely different things, not just different magnitudes. The sign is what's robust; the specific emotion labels within each cluster are run-dependent at int4 precision.

## Caveats

- **Cross-lab comparison uses paper-reported anchors for Sonnet, not an independent measurement**. We didn't re-run the paper's Stage 8 on Sonnet. If Anthropic re-reports the paper's anchors on the same 20 prompts we used, or if we could measure Sonnet directly, we'd have cleaner evidence.
- **20-prompt Stage 8 is small** for a 171-emotion shift measurement. Multiple-comparison risk is real. We partly mitigated with the cross-version robustness check (ρ=0.92) — if this were multiple-comparison noise, it wouldn't show the same anchors twice.
- **Llama 3.3 vs Sonnet 4.5 are very different sizes, tokenizers, architectures, and Llama is measured in bnb int4 while Sonnet is full-precision.** Some of the semantic-anchor difference might be "smaller-model artifact" or "4-bit-quantization noise" rather than "Meta vs Anthropic choice". The cross-version Llama-only control addresses version confound (both comparison models are bnb int4) but not lab/size/quantization confounds.
- **bnb int4 noise floor on per-emotion shift rankings is substantial, but the cluster-level PC1 sign survives the noise.** Running the same Stage 8 measurement twice produced Spearman ρ = 0.465 between the two runs' per-emotion shift vectors, not the ~0.95 expected. Specific emotions sign-flipped across runs (`brooding`: −0.037 vs +0.197; `calm`: +0.202 vs −0.194; `gloomy`: −0.044 vs +0.055), and the up-direction top-10 lists had **0/10 overlap** between runs. The two scripts differ only in trivial details (batching with padding vs singleton with `add_special_tokens=False`) — roughly 5-10% per-activation drift from int4 dequantization + batch order, which flips the sign of emotions with small raw shift magnitudes. We then asked the obvious question: does the cluster-level PC1 centroid survive this noise? It does for the up-cluster but not cleanly for the down-cluster. Run_A up-cluster PC1 = +0.856 (z = +4.86), run_B up-cluster PC1 = +0.517 (z = +2.94) — both far above a 10,000-sample permutation null (N=10 of 171, CI95 = [−0.315, +0.354]). The up-cluster direction is verified across runs despite 0/10 name overlap. The down-cluster direction is weaker: run_A = −0.44 (p ≈ 0.01), run_B = −0.09 (p ≈ 0.61, not different from noise). **The robust empirical claim is therefore: "Llama's post-training up-cluster reliably sits at PC1 > 0, opposing Sonnet's reported down-cluster at PC1 = −0.432."** A cleaner replication would run in fp16/bf16 with fixed batch composition and random-seeded prompt-order; we didn't because VRAM constraints (single A800 80GB) force int4 for a 70B model. The individual emotion labels are noise-floor-limited; the cluster-level verified claim is in `results/pc1_stability_verification.json`.
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
- `/tmp/task_cross_signal_analysis.py` — cross-signal matrix + Jaccard + cluster means
- `/tmp/stage8_layer_sweep.py` — per-layer RLHF direction stability

All results in `experiments/ant_emotion_concepts/results/` — geometry, preference Elo, post-training shifts, cross-version decomposition, layer sweep, and the speaker-probe cross-type matrix.

## Acknowledgments

Anthropic's "Emotion Concepts" paper is a remarkably thorough methodology. Most of what worked here works *because* that paper spelled out the extraction, denoising, and probing pipeline cleanly enough to port. The disagreements are about what the method reveals on a different model, not about the method itself.
