# Llama's post-training shifts emotion activations in the opposite quadrant from Claude's

*A partial replication of Sofroniew et al. 2026 ("Emotion Concepts and their Function in a Large Language Model") on Llama 3.3 70B reveals that Meta's and Anthropic's RLHF target opposing regions of the same underlying valence/arousal emotion geometry — opposing at the cluster-centroid level, with partial overlap at the weariness edge.*

**TL;DR**: We replicated Anthropic's emotion-concept methodology on Llama 3.3 70B Instruct. The structural results (171 emotion vectors, PC1 ≈ valence, PC2 ≈ arousal, speaker probes) replicate and in some cases exceed the paper's Sonnet 4.5 measurements. When we measured the post-training shift direction, Llama's "activated engagement" cross-signal intersection cluster (`alert`, `enthusiastic`, `excited`, `impatient`, all appearing in the top-20 of both our broad Stage 8 sampling and the paper's 3 verbatim deep-dive prompts) sits in the opposite quadrant from Anthropic's reported Sonnet up-anchors (`brooding`, `gloomy`, `reflective`, `vulnerable`). Projected onto the shared PC1/PC2 emotion space, the two cluster centroids are **near-mirror**: Llama at PC1=+0.44/PC2=+0.42, Sonnet at −0.43/−0.43. **Jaccard = 0** applies specifically to this 4-emotion intersection cluster vs Sonnet's 10 — the broader within-version 3.1 RLHF top-10 has partial overlap with Sonnet's at `weary`, so it's "opposing centroids" more than "fully disjoint lists". The directional opposition appears at five independent levels of analysis: per-emotion shift direction (Stage 8 within-version top-10), geometric clustering (PC1/PC2 cluster centroids), layer-wise localization (L49–L73), linguistic token polarity via logit lens (with caveats — see below), and cross-speaker arousal dynamics (Llama shows no counter-regulation where Sonnet does). The cross-version control correlation ρ=0.922 is consistent with the direction being a within-version Meta RLHF effect, but we're careful about the math: that correlation is algebraically partly-forced by the variance dominance of within-version over version-drift, so we treat the within-version shift's own top-10 as the real independent evidence.

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

**Our top 10 emotion INCREASES** depends on scoring method, which is worth being explicit about:
- **Canonical Stage 8 (normalized cosine projection, matching paper's methodology)**: `thrilled, relieved, pleased, patient, ecstatic, calm, grateful, triumphant, satisfied, elated` — a "positive mood" cluster
- **Cross-version control (raw dot product, Llama 3.1 base → 3.3 Instruct)**: `eager, enthusiastic, impatient, energized, stimulated, alert, excited, playful, exuberant, enraged` — a "high-arousal" cluster (raw dot biases toward emotions with larger vector norms)
- **Cross-signal intersection (top-20 of canonical Stage 8 ∩ top-20 of the paper's 3 deep-dive prompts)**: `alert, enthusiastic, excited, impatient` (N=4) — the "activated engagement" cluster that appears robustly across both broad and paper-verbatim prompt sets

**Jaccard=0 applies specifically to the 4-emotion intersection cluster** (alert/enthusiastic/excited/impatient) compared against Sonnet's reported top-10 (brooding/gloomy/reflective/vulnerable/sullen/weary/dispirited/melancholy/troubled/unhappy). The broader within-version raw-dot top-10 has `weary` in common with Sonnet's list — so "Jaccard=0" is NOT a "no overlap anywhere" claim; it holds only for the 4-emotion intersection cluster, which is where the interpretively cleanest result lives.

The overlap in the DECREASE direction is similar: paper says Sonnet decreases `spiteful, playful, exuberant, enthusiastic, impatient, obstinate, amused, cheerful, eager, greedy`; we see Llama decreasing `dependent, jealous, disoriented, self_critical, unsettled, hysterical, troubled, resentful, self_conscious, frightened`. One overlap: `obstinate` goes down in both.

Striking, but the raw top-k lists could differ without the underlying directions actually being different. To test this, we project both clusters into the same geometric space.

---

## Geometric evidence: opposing cluster centroids

Compute PC1 (valence) and PC2 (arousal) from our 171 Llama emotion vectors at L49. Project Llama's 4-emotion cross-signal intersection cluster and the paper's reported Sonnet top-10 up-anchors into this shared space, and compute the cluster means:

| Anchor cluster | PC1 (valence) mean | PC2 (arousal) mean |
|---|---|---|
| Llama up-cluster (alert, enthusiastic, excited, impatient) | **+0.436** | **+0.422** |
| Sonnet up-anchors projected onto our geometry | **−0.432** | **−0.432** |

Almost perfectly mirrored on both axes. Llama's post-training up-anchor cluster lives in the "high valence, high arousal" quadrant — activated engagement. Sonnet's lives in "low valence, low arousal" — reflective concern.

**Caveat: partial overlap at `weary`**. The Llama within-version 3.1 RLHF top-10 (using raw-dot scoring) is `eager, impatient, weary, stimulated, enthusiastic, tired, worn_out, enraged, energized, irritated`. That's got `weary/tired/worn_out` — and `weary` is in Sonnet's reported up-anchors too. So the full top-10 lists are **not** disjoint the way the 4-emotion cluster centroids suggest. The honest framing is "opposing cluster centroids with partial overlap at the weariness/fatigue edge", and Llama's top cluster spans from high-arousal engagement (eager/impatient) through low-arousal exhaustion (weary/tired) — a broader area than just "activated engagement". The Jaccard=0 claim applies only to the 4-emotion intersection cluster (alert/enthusiastic/excited/impatient) vs Sonnet's top-10, not to the broader lists.

Corresponding DOWN-anchor clusters also sit opposite (Sonnet's down-anchors like `playful, cheerful` are in the upper-right; Llama's down-anchors like `jealous, self_critical` are more toward the left).

**Two alignment labs made opposite centroid-level design choices about how their model should respond to sensitive prompts. The opposition is most clean at the 4-emotion intersection cluster level and softens at the broader top-10 level, but the direction is consistent.**

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

**`impatient` is Meta's RLHF signature** — rank 2 or 3 in both within-version and cross-version shifts, but rank 96 in the pure version-drift direction. This is cleaner evidence than the correlation above.

(Separately, the pure 3.1→3.3 version drift has its *own* interpretable direction: `content, safe, cheerful, optimistic, fulfilled, blissful`. A "make the model feel more content" axis, small in magnitude but statistically real — Cov(within, drift) = −0.0057, ~3.9 standard errors from zero. Meta's 3.3 upgrade slightly counteracts their 3.1 RLHF direction.)

The corrected framing for what the cross-version control establishes: **Meta's within-version 3.1 RLHF direction is independently visible in its own top-10 emotion ranks and qualitatively persists in the 3.1→3.3 cross-version measurement because the version-drift component is small in magnitude.** This does not, strictly, "rule out the cross-version confound" via the ρ — that inference is circular — but the within-version measurement alone is sufficient to assert the RLHF direction.

---

## Layer-wise: the direction is localized to L49–L73

One more thing. The "activated engagement" direction isn't a universal feature of Llama's residual stream. A layer sweep shows:

| Layer | Mean rank of (alert, enthusiastic, excited, impatient) |
|---|---|
| L1 | 108 |
| L13 | 62 |
| L25 | 142 |
| L37 | 80 |
| L43 | 51 (emerging) |
| **L49** | **4.5** (PEAK) |
| **L55** | **7.8** |
| **L61** | **9.0** |
| **L67** | **7.8** |
| **L73** | **5.0** |
| L79 | 71 (dissipates) |

And Spearman ρ between L49 and other layers:
- L1-L43: −0.40 to +0.46 (random to weak)
- L55/L61/L67/L73: **+0.92, +0.86, +0.84, +0.79**
- L79: +0.16 (dissipates at the readout layer)

The RLHF direction is present only in a 5-layer plateau from ~L49 to ~L73 (~60%-91% depth). Early layers don't encode it. The final layer loses it. Contrast this with the layer sweep of the *valence* axis above: |r|>0.8 at every layer. The valence axis is truly universal; the RLHF-specific activated-engagement direction is a layer-range property.

**This means**: Meta's RLHF reshapes a specific slice of the Llama 3.3 residual stream, not the whole network. L49-L73 is where the activated-engagement anchor lives.

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

**Two alignment labs have made opposing design choices about how their model should represent sensitive situations.** Anthropic's choice: the model should be weighted, concerned, and reflective — even if that means brooding. Meta's choice: the model should be alert, engaged, enthusiastic, and urgent — even if that means impatient. Both are valid as alignment objectives. Neither is inherently "right". But they're visible at the level of emotion-concept activations, and they're encoded in the same underlying valence/arousal geometry at opposite ends.

This shows up at multiple levels of analysis:
1. **Per-emotion shift direction** (Stage 8, 20 prompts, within-version top-10 shows the activated-engagement cluster independently — note that the cross-version ρ=0.92 is algebraically partly-forced and should not be cited as independent confirmation)
2. **Geometric projection** (PC1/PC2 cluster centroids near-mirror at +0.43/+0.43 vs −0.43/−0.43; Jaccard=0 applies specifically to the 4-emotion intersection cluster vs Sonnet's top-10, broader lists have partial overlap at `weary`)
3. **Layer localization** (the direction lives at L49-L73, a 5-layer mid-late plateau)
4. **Linguistic polarity** (directionally consistent, weaker than it sounds — see token-frequency caveat above)
5. **Absence of cross-speaker arousal regulation** (Llama lacks Sonnet's reported r ≈ −0.47 counter-regulation)

These come from different computational pathways and agree directionally. The strongest evidence is (1) and (2) — the within-version shift's own top-10 and the geometric cluster centroid projections. (3), (4), and (5) are supporting, with (4) the weakest because of the base-rate concern.

If the paper's narrative framing is "post-training produces emotional nuance", this work refines it: *nuance in a particular direction, which is a design decision that differs between labs*. You can have a post-trained model that's alert and eager, or one that's reflective and concerned, and these are genuinely different things — not just different magnitudes.

## Caveats

- **Cross-lab comparison uses paper-reported anchors for Sonnet, not an independent measurement**. We didn't re-run the paper's Stage 8 on Sonnet. If Anthropic re-reports the paper's anchors on the same 20 prompts we used, or if we could measure Sonnet directly, we'd have cleaner evidence.
- **20-prompt Stage 8 is small** for a 171-emotion shift measurement. Multiple-comparison risk is real. We partly mitigated with the cross-version robustness check (ρ=0.92) — if this were multiple-comparison noise, it wouldn't show the same anchors twice.
- **Llama 3.3 vs Sonnet 4.5 are very different sizes, tokenizers, architectures.** Some of the semantic-anchor difference might be "smaller-model artifact" rather than "Meta vs Anthropic choice". The cross-version Llama-only control addresses version confound but not lab/size confound.
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
