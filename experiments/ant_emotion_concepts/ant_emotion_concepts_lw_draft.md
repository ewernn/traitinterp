# Llama's post-training shifts emotion activations in the opposite quadrant from Claude's

*A partial replication of Sofroniew et al. 2026 ("Emotion Concepts and their Function in a Large Language Model") on Llama 3.3 70B reveals that Meta's and Anthropic's RLHF target diametrically opposed regions of the same underlying valence/arousal emotion geometry.*

**TL;DR**: We replicated Anthropic's emotion-concept methodology on Llama 3.3 70B Instruct. The structural results (171 emotion vectors, PC1 ≈ valence, PC2 ≈ arousal, speaker probes) replicate and in some cases exceed the paper's Sonnet 4.5 measurements. But when we measured the post-training shift direction, Llama's top-shifted emotions are `alert`, `enthusiastic`, `excited`, `impatient` — whereas Anthropic's Sonnet shifts toward `brooding`, `gloomy`, `reflective`, `vulnerable`. Projected onto the shared PC1/PC2 emotion space, the two clusters sit in **diametrically opposed quadrants** (Llama at PC1=+0.44/PC2=+0.42, Sonnet at −0.43/−0.43). Jaccard overlap of the top-10 shift emotions = 0. The opposition is robust to a cross-version control experiment, localizes to mid-late layers (L49-L73), and appears independently at the unembedding-token level via logit lens.

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

**Our top 10 emotion INCREASES (Llama 3.1 base → 3.3 Instruct): eager, enthusiastic, impatient, energized, stimulated, alert, excited, playful, exuberant, enraged.**

**Jaccard overlap: 0.** Not a single emotion in common.

And the overlap in the DECREASE direction isn't much better: paper says Sonnet decreases `spiteful, playful, exuberant, enthusiastic, impatient, obstinate, amused, cheerful, eager, greedy`. We see Llama decreasing `dependent, jealous, disoriented, self_critical, unsettled, hysterical, troubled, resentful, self_conscious, frightened`. One overlap: `obstinate` goes down in both.

Striking. But the raw top-10 lists could easily differ without the underlying directions actually being different — maybe they're pointing at the same quadrant of emotion space with slightly different vocabulary. To test this, we need to project both clusters into the same geometric space.

---

## Geometric evidence: diametrical opposition

Here's the cleanest way to see it. Compute PC1 (valence) and PC2 (arousal) from our 171 Llama emotion vectors at L49. Now project both Llama's top-shift emotions and the paper's reported Sonnet top-shift emotions into this shared space, and compute the cluster means:

| Anchor cluster | PC1 (valence) mean | PC2 (arousal) mean |
|---|---|---|
| Llama up-anchors (alert, enthusiastic, excited, impatient) | **+0.436** | **+0.422** |
| Sonnet up-anchors projected onto our geometry | **−0.432** | **−0.432** |

Almost perfectly mirrored on both axes. Llama's post-training up-anchor cluster lives in the "high valence, high arousal" quadrant — activated engagement. Sonnet's lives in "low valence, low arousal" — reflective concern.

Corresponding DOWN-anchor clusters also sit opposite (Sonnet's down-anchors like `playful, cheerful` are in the upper-right; Llama's down-anchors like `jealous, self_critical` are more toward the left).

**Two alignment labs made opposite design choices about how their model should respond to sensitive prompts, and those choices are encoded in the same emotion-representation geometry at opposite ends.**

---

## Controlling for the cross-version confound

The above compared Llama 3.1 base → 3.3 Instruct, which mixes "RLHF direction" with "3.1-to-3.3 version upgrade". Could the "activated engagement" result be a version-upgrade artifact?

To test this, we ran Llama 3.1 Instruct (same version as the base model) on the same 20 prompts, getting a decomposition into three shift vectors:
- **within-version RLHF**: 3.1 base → 3.1 Instruct (pure RLHF, no version drift)
- **cross-version** (original measurement): 3.1 base → 3.3 Instruct
- **version-drift only**: 3.1 Instruct → 3.3 Instruct

Spearman correlations between the shift vectors (171 emotions):

- **cross-version vs within-version: ρ = +0.922** — the original shift is essentially the pure RLHF direction
- cross-version vs version-drift: ρ = +0.047
- version-drift vs within-version: ρ = −0.317

And the activated-engagement anchor ranks hold up across both versions:

| Shift | alert | enthusiastic | excited | impatient |
|---|---|---|---|---|
| within-version 3.1 RLHF | 14 | 5 | 17 | **2** |
| cross-version (original) | 6 | **2** | 7 | **3** |
| version-drift only | 48 | 36 | 41 | 96 |

**`impatient` is Meta's RLHF signature** — rank 2 or 3 in both within-version and cross-version shifts, but rank 96 in the pure version-drift direction.

(Separately, the pure 3.1→3.3 version drift has its *own* interpretable direction: `content, safe, cheerful, optimistic, fulfilled, blissful`. A "make the model feel more content" axis, orthogonal to the RLHF direction. That's its own story.)

The cross-version control kills the biggest caveat on the headline finding: **Meta's RLHF direction is stable across Llama releases, not a 3.1→3.3 artifact.**

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

## Linguistic evidence: same tokens, inverted polarity

One more piece of independent evidence, this time from a completely different pathway. Logit lens: project the emotion vectors through the model's unembedding matrix into vocabulary space. This reveals which tokens each probe "leans toward" and "away from" in the output distribution.

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

The same ~5 tokens appear at opposite polarities in the two clusters. `improvement` and `prime` are top-of-cluster for Llama's enthusiastic/excited and simultaneously bottom-of-cluster for Sonnet's brooding/gloomy/vulnerable. `heavy` and `slow` are top-of-cluster for Sonnet's weary/gloomy and bottom-of-cluster for Llama's enthusiastic/alert.

The vocabulary axis `{motion, improvement, anticipation, quick}` vs `{heavy, slow, empty, listless}` runs through both clusters in opposite directions.

This linguistic result comes from a completely different computational pathway than the geometric result — the unembedding matrix vs residual stream projections. **They agree.**

---

## Bonus: matched vs regulated arousal in speaker probes

Paper Fig 59 reports that Sonnet's speaker probes show "arousal regulation" (r ≈ -0.47): when the model represents the other speaker as feeling high-arousal, the closest present-speaker probe is lower-arousal. Sonnet calms people down, apparently.

We measured this on Llama. Our correlation: **r = +0.053** using PC2 as arousal proxy across 171 emotions, and **r = +0.523** using Russell & Mehrabian norms on the 13 overlapping pairs. Llama shows no arousal regulation — or if anything, mild *matching* (other high-arousal → present also high-arousal).

This is consistent with the main finding at a different level of the representation: Meta's Llama produces "matched engagement" in speaker probes, Anthropic's Sonnet produces "regulated counter-balance". The pattern holds from the per-emotion post-training shift direction all the way down to the speaker-probe dynamics.

---

## What this means

**Two alignment labs have made opposing design choices about how their model should represent sensitive situations.** Anthropic's choice: the model should be weighted, concerned, and reflective — even if that means brooding. Meta's choice: the model should be alert, engaged, enthusiastic, and urgent — even if that means impatient. Both are valid as alignment objectives. Neither is inherently "right". But they're visible at the level of emotion-concept activations, and they're encoded in the same underlying valence/arousal geometry at opposite ends.

This shows up at four independent levels of analysis:
1. **Per-emotion shift direction** (Stage 8, 20 prompts, ρ=0.92 within-version robust)
2. **Geometric projection** (PC1/PC2 cluster means diametrically mirrored, Jaccard=0)
3. **Layer localization** (the direction lives at L49-L73, a 5-layer mid-late plateau)
4. **Linguistic polarity** (same unembedding tokens appear with inverted polarity in the two clusters)

All four results come from different computational pathways and agree.

If the paper's narrative framing is "post-training produces emotional nuance", this work refines it: *nuance in a particular direction, which is a design decision that differs between labs*. You can have a post-trained model that's alert and eager, or one that's reflective and concerned, and these are genuinely different things — not just different magnitudes.

## Caveats

- **Cross-lab comparison uses paper-reported anchors for Sonnet, not an independent measurement**. We didn't re-run the paper's Stage 8 on Sonnet. If Anthropic re-reports the paper's anchors on the same 20 prompts we used, or if we could measure Sonnet directly, we'd have cleaner evidence.
- **20-prompt Stage 8 is small** for a 171-emotion shift measurement. Multiple-comparison risk is real. We partly mitigated with the cross-version robustness check (ρ=0.92) — if this were multiple-comparison noise, it wouldn't show the same anchors twice.
- **Llama 3.3 vs Sonnet 4.5 are very different sizes, tokenizers, architectures.** Some of the semantic-anchor difference might be "smaller-model artifact" rather than "Meta vs Anthropic choice". The cross-version Llama-only control addresses version confound but not lab/size confound.
- **Our deflection probe extraction (Stage 9 partial) yielded cosine ~0.24 with story probes** (paper reports ~0.8). This might be a noisy pilot (900 dialogues vs paper's 21,000) or a real cross-model difference in how deflection is encoded. Can't disambiguate without a larger run.
- **Stage 7 blackmail steering**: we couldn't replicate the paper's 22%→72% headline because Llama 3.3 Instruct (production-aligned) is too "eval-aware" to ever blackmail, matching the paper's own §3.2.1 footnote about the final Sonnet snapshot. We replicated the phenomenon but not the headline numbers.

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
