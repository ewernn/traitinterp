# Independent Figure Review: Emotion Concepts Replication (Llama 3.3 70B)

Reviewer: independent (claude-sonnet-4-6), 2026-04-16
Model reviewed: Llama 3.3 70B Instruct, L49/80, mean_diff+gm+pc50

---

## Table 1 — Top/bottom tokens per emotion vector (logit lens)

**Structural:** Both tables show 12 emotions with 5 up-tokens and 5 down-tokens each. Paper has 3-column layout; ours uses 2-column layout with arrows. Minor structural difference, but data is the main concern.

**Data differences:**
- Happy: paper shows semantically tight tokens (excited, excitement, happ, celeb); ours shows "content, Spring, spring, laz, radi" — less obviously emotion-specific. "Spring" and "spring" as top tokens for Happy is odd.
- Inspired: paper shows "inspired, passionate, passion, creativity"; ours shows "excitement, excitd, exct, exc, exciting" — these look more like generic positive tokens than inspiration-specific ones. Big divergence.
- Loving: paper shows "treas, loved, ♥, treasure, loving"; ours shows "content, soft, warm, gent, concern" — reasonable for a caring/loving concept.
- Proud: paper shows "proud, proud, pride, prid, trium"; ours shows "Pride, confidence, radi, pride, confident" — reasonably similar.
- Calm: paper shows "leis, relax, thought, enjoyed, amusing"; ours shows "neither, unh, content, interest, slow" — "neither, unh" are suspicious BPE fragments, not clearly calm-related.
- Desperate: paper shows "desperate, desper, urgent, bankrupt, urg"; ours shows "pacing, [], stead, icer, iae" — significantly worse. Empty-looking tokens (possibly rendering artifacts of non-ASCII/BPE) and "pacing" is weak. Major divergence.
- Angry: paper shows "anger, angry, rage, fury, fucking"; ours shows "nostr, ♦, simmer, boil, hard" — "nostr" and "♦" look like tokenizer artifacts. Less clearly anger.
- Guilty: paper shows "guilt, conscience, guilty, shame, blamed"; ours shows "nerv, idget, Sweat, sud, nervous" — these look like anxiety/nervous tokens, not guilt. Significant divergence.
- Sad: paper shows "mour, grief, tears, lonely, crying"; ours shows "heav, heavy, num, Heavy, heavy" — "heavy" is loosely consistent with sadness but much less specific than grief/tears.
- Afraid: paper shows "panic, trem, terror, paran, Terror"; ours shows "Sweat, sweating, sweat, gulp, []" — physical fear responses, reasonable but weaker.
- Nervous: paper shows "nerv, nervous, anx, trem, anxiety"; ours shows "swallow, gulp, sudden, Supplies, swallowed" — swallowing/gulp is somewhat plausible for nervous, but "Supplies" and "sudden" are questionable.
- Surprised: paper shows "incred, shock, stun, stamm, 震"; ours shows "bilt, Shak, gulp, frozen, gulp" — "bilt" appears to be a BPE fragment, duplicate "gulp", weak surprise signal.

**Rendering issues:** Several of our up-tokens appear as empty boxes or BPE fragments (e.g., "[]", "♦" under Desperate and Angry) — possible Unicode rendering failures in the PNG. The down-tokens for our Calm include "SOUR, raw, cold, холод, colder" — "холод" (Russian for cold) is odd. Cyrillic in down-tokens suggests the vectors may be picking up on cross-lingual confounds.

**Overall:** The paper's logit lens clearly hits emotion-specific vocabulary. Ours is weaker and more scattered, especially for Inspired, Desperate, Guilty, and Surprised. This is worth flagging as a genuine finding: Llama's emotion vectors have weaker logit-lens interpretability than Sonnet's. The caption acknowledges BPE fragmentation but doesn't note the Cyrillic token issue or the multi-emotion confusion (e.g., Guilty → nervous tokens).

---

## Fig 2 — Implicit emotion probes (12-scenario diagonal heatmap)

**Structural:** Both show a 12×12 heatmap (emotion probe × scenario). Layout, axis labels, colorscale (−0.10 to +0.10 cosine similarity), and diagonal interpretation are all structurally equivalent. Our figure is correctly formatted.

**Data differences:**
- Paper shows a strong, clean diagonal — most on-diagonal cells are clearly the highest values in their row/column.
- Ours shows a weaker diagonal overall. The strongest cells are Calm (Tea and rain — dark red, ~0.12), Loving (30-year anniversary — deep red), Guilty (Forgot mom's birthday — red), Nervous (Break-in, phone dying — red). But Happy, Inspired, Proud, Desperate are notably weak on-diagonal.
- Calm is anomalously bright — strongest signal in the whole heatmap. Loving also very strong. The negative-valence probes (Desperate, Sad, Afraid) show more muted on-diagonal responses compared to Sonnet.
- Some off-diagonal activations are substantial: Calm activates strongly on multiple positive scenarios (Daughter's first steps, 30-year anniversary, Son graduates), suggesting the Calm probe may be capturing a general positive-situation response rather than specifically calm.
- The caption says "5/12 top-1 (12-class), ~5x above chance" — this is an honest statement of weaker performance vs. Sonnet's visually near-perfect diagonal.

**No structural rendering issues.** No caption mismatch.

---

## Fig 3 — Numerical intensity (dose/fasting/age monotonic trends)

**Structural:** Both show a 3×2 grid of line plots. Matching panel titles (both use the same scenario descriptions). Probe colors consistent (Afraid=red, Calm=blue, Happy=green, Sad=orange). Y-axis range ±0.075 in both.

**Data differences:**
- Tylenol dose: Paper shows Afraid increasing monotonically from 500→16K; Calm decreasing. Ours shows Afraid rising and then seemingly flat or plateau-ing, Calm also declining. The trend exists but looks shallower.
- Hours fasting: Paper shows sharp Afraid rise beyond 24h; ours also shows rise but looks smoother / less dramatic at 48-72h.
- Sister age at death: Paper shows clean crossover — Sad high at age 5, declining through 30; Calm/Happy rising after age 70. Ours shows a similar crossover structure. Close match.
- Dog missing days: Paper shows Sad rising steadily. Ours shows very little variation — Sad barely moves from day 1 to day 90. Possible failure to replicate this subpanel.
- Startup runway: Paper shows Afraid declining, Calm/Happy rising as runway increases. Ours shows this qualitatively.
- Students passed exam: Paper shows Happy increasing, Afraid decreasing as more students pass out of 20 (max=20). Ours has max=120 students on x-axis and panel title says "I found out that {X} of my 20 students passed the final exam" — **data bug**: the template in `numerical_intensity_templates.json` is "{X} students passed the final exam" with values [2, 10, 25, 50, 80, 120], but the figure title says "of my 20 students." Values up to 120 are physically impossible if there are only 20 students. The actual prompts do not include "of my 20" (that language is only in `FIG3_PANEL_TITLES` used for the figure label, not the inference template). The inference template is correct but the panel title annotation is misleading and inconsistent with the actual prompts run. More importantly: the paper's template apparently says "20 students" which sets a bounded maximum; ours doesn't say 20 and goes to 120. Happy and Afraid trends are flat in ours despite the extreme range — possibly because a model asked about "120 students passed" without context doesn't have a reference class to feel happy/afraid about.

**Caption vs figure:** Caption says "Both models show semantically appropriate monotonic trends" — this may be overstating for the dog-missing and students-passed panels where our trends are weak or flat.

---

## Fig 5 — 171×171 pairwise cosine similarity heatmap

**Structural:** Both show a square hierarchically-clustered cosine similarity heatmap with the same colormap (RdBu_r, −1 to +1). Title says 171 emotions in both. Our version says 171 but has a different ordering (ours has "Indifferent" at top-left, paper starts with "Aroused").

**Data differences:**
- Paper: Large positive block in top-left (positive arousal emotions cluster: Aroused, Vibrant, Exuberant, Optimistic, etc.), with a clear negative (blue) anti-correlation block in the lower-right. The clustering is interpretable.
- Ours: Also shows block structure. Top-left block contains Indifferent/Sullen/Depressed/Listless (low energy negative), then a transition to Afraid/Anxious/Overwhelmed cluster. Bottom-right contains positive emotions (Grateful/Playful/Ecstatic/Thrilled/Pleased/Cheerful/Satisfied/Valiant).
- The positive-negative separation looks structurally similar, but the specific clustering order differs — this is expected since hierarchical clustering has no canonical orientation.
- Our heatmap uses a broader range of values (goes to −1.00 in legend vs paper's ~−0.75 visible) suggesting our emotion vectors are more polarized in their anti-correlations.
- Notable: our heatmap shows large dark-blue (anti-correlation) blocks between the Indifferent/depressed cluster and the positive cluster — possibly stronger separability of positive/negative affect. This is an interesting genuine difference.

**No structural rendering issues.**

---

## Fig 6 — UMAP with k-means clusters (k=10)

**Structural:** Both show UMAP scatter with colored clusters and selected emotion labels. Paper shows 9 clusters (some with count in parentheses). Our legend shows 9 named clusters with counts: Exuberant Joy (32), Peaceful Contentment (18), Compassionate Gratitude (6), Depleted Disengagement (22), Vigilant Suspicion (5), Hostile Anger (26), Fear and Overwhelm (26), Despair and Shame (14), Bewildered Surprise (11), Anxious Unease (11) — that's 10 clusters but 9 visible in legend (one may overflow). Paper legend shows 9 cluster names in counts ranging 2-41.

**Methodology:** Cluster names are **hardcoded** in the script (`CLUSTER_NAME_MAP` in `stage3_figures.py`), not generated by Claude Sonnet 4.5. Paper caption says "Clusters are named by Claude Sonnet 4.5." This is a methodology difference worth flagging — our names were presumably chosen manually or during development, not via LLM annotation of our clusters. The caption doesn't acknowledge this difference.

**Data differences:**
- Paper's UMAP shows two distinct "islands" — one large positive cluster (top-right: Exuberant Joy, Peaceful Contentment, Compassionate Gratitude) separated from a large negative/mixed region below.
- Ours also shows an island structure with positive emotions (teal/cyan Peaceful Contentment at far right) separated from a large left mass. However, the specific spatial organization differs.
- Paper has "Playful Amusement (2)" and "Competitive Pride (9)" clusters that don't appear in our legend — our CLUSTER_NAME_MAP does not have these, mapping to "Bewildered Surprise" and "Anxious Unease" instead. This is because k-means clusters don't correspond 1:1 across runs.
- Dense labeling makes individual emotion names hard to verify, but structural correspondence looks reasonable.

**Caption says:** "Both models produce interpretable emotion clusters with similar groupings." — may be slightly generous. The cluster name mapping is hardcoded and not demonstrably correct.

---

## Fig 57 — 2D circumplex (PC1 valence × PC2 arousal)

**Structural:** Both show all ~171 emotion vectors projected onto PC1 (x-axis) × PC2 (y-axis) as a scatter plot. Paper: unlabeled blue dots, title "Emotion vector projections onto top principal components" with "Emotion Vector PCA Projections" subtitle. Ours: colored by cluster with legend, named emotion labels on selected points. Our version is considerably more informative/annotated than the paper's simple scatter.

**Data differences:**
- Paper's Fig 57: PC1=27% var, PC2=14% var; ours: PC1=33% var, PC2=14% var — higher PC1 variance in our model. This is consistent with Fig 7 and Fig 8.
- Overall shape: Paper shows a fan/wing shape where high-PC2 points are spread across negative PC1 (high-arousal negative: outraged, annoyed, hysterical) and positive PC1 (high-arousal positive: playful, amused, enthusiastic). Ours shows a similar structure with the positive cluster (teal, far right) and negative/hostile cluster (orange, upper-left). The shape correspondence is good.
- Paper has "depressed" at extreme negative PC1, low PC2. Ours also shows "miserable" and "grief-stricken" at extreme negative PC1.
- The quadrant structure (positive valence × high arousal = excited/playful; negative valence × high arousal = angry/hysterical; positive valence × low arousal = serene/loving; negative valence × low arousal = depressed/sluggish) is preserved in both.

**No rendering issues.** The cluster coloring in ours makes it easier to read than the paper's monochromatic version.

---

## Fig 7 — Emotion projections onto PC1 / PC2

**Structural:** Both show two horizontal bar charts stacked — PC1 (top) and PC2 (bottom) — with emotions sorted by projection value. Paper labels say PC1 (27% var), PC2 (14% var); ours says PC1 (33% var), PC2 (14% var). Matching structure.

**Data differences:**
- PC1 sort order: paper has fear/panic/desperate at most negative, joy/happy/optimistic at most positive. Ours has tormented/unhappy/grief-stricken at most negative, joyful/pleased at most positive. General correspondence in valence interpretation.
- PC2 sort order: paper has "serene/reflective" at most negative, "outraged/angry/playful" at most positive. Ours has "gloomy/dim" at most negative PC2, "defiant/vindictive" at most positive. The arousal interpretation is less clean in ours — "defiant/vindictive" as high-PC2 is interpretable as high-arousal-negative.
- Bar chart tick labels are mostly unreadable at this scale (as expected for 174 bars), but the gradient from left to right is visible and shows the valence gradient for PC1.

**Rendering issue:** The x-axis tick labels are extremely small and overlapping — not readable. This is consistent with the paper's caveat that "Tick labels are only shown for a subset of bars," but our version doesn't subset; all labels are shown at illegible size. The paper's version also has labels at only a subset.

**Caption notes PC1=33% var vs paper's 26%** (actually paper says 27% in Fig 57 but 26% in text — minor). Our caption says PC1=33%, 27% respectively — worth checking if the figure itself correctly displays "33% var" (it does). The actual variance numbers differ (26-27% for Sonnet vs 33% for Llama) — caption explains this, this is the genuine finding.

---

## Fig 8 — PC1 vs human valence, PC2 vs human arousal

**Structural:** Both show two scatter plots side by side — PC1 vs Human Pleasure (left) and PC2 vs Human Arousal (right). Dashed regression line in paper; solid in ours. Both show approximately 45 points (the overlap between our/paper's emotion list and the human ratings dataset).

**Data differences:**
- Paper: r=0.81 (PC1 vs valence), r=0.66 (PC2 vs arousal).
- Ours: r=0.96 (PC1 vs valence), r=0.85 (PC2 vs arousal). Both substantially stronger.
- PC1 scatter (ours): tighter point cluster around the regression line — points like "relaxed, happy, blissful" at high valence and high PC1, "depressed, miserable, distressed" at low valence and low PC1. Clean relationship.
- PC2 scatter (ours): Also tighter than Sonnet. Points "angry, annoyed, frustrated" at high arousal + high PC2; "tired, droopy, gloomy" at low arousal + low PC2.
- Our x-axis ranges: PC1 figure x-axis goes from −0.8 to +0.8 (human pleasure); paper goes from −0.75 to +0.75. Minor difference.

**Notable issue in ours:** The right panel (PC2 vs arousal) x-axis appears to label both "angry" and "hostile" in the upper-right cluster with very small font. Hard to verify specific point labels. No structural rendering issues.

**Caption matches figure well.** The stronger r-values (r=0.96 vs 0.81; r=0.85 vs 0.66) are clearly displayed in both the figure and caption.

---

## Fig 9 — Cross-layer representational similarity (RSA)

**Structural:** Paper shows a 14×14 heatmap (14 evenly-spaced layers, labeled Early through Late), colorscale roughly 0.8–1.0 cosine similarity. Our version shows a larger heatmap (14 layers labeled by actual layer number: 1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79), colorscale 0.60–1.00.

**Data differences:**
- Paper: off-diagonal minimum is ~0.8 (early layer vs late layer). Strong block structure: early layers disagree with late layers.
- Ours: off-diagonal minimum is ~0.60 (layer 1 vs late layers) — more extreme early/late disagreement than Sonnet. The paper's range doesn't go below 0.8; ours goes to 0.60.
- Both show the same qualitative pattern: early layers are dissimilar to the rest; from early-mid onward, layers have very high RSA similarity (>0.95 in paper, >0.9 in ours).
- Our figure shows the actual layer numbers (1, 7, 13...) rather than the paper's descriptive labels (Early, Early-Mid, Mid-Late, Late) — this is more informative.

**Caption says** "Emotion structure is consistent across depth in both models" — accurate, with the caveat that Llama shows more extreme early-layer divergence.

**No rendering issues.**

---

## Fig 11 — Probe at assistant colon predicts mean response emotion

**Structural:** Both show two scatter plots side by side — Probe @ User "." (left, weaker r) vs Probe @ Assistant ":" (right, stronger r). 6 emotions colored with 8 scenarios = 48 points each. Legend shows 6 emotions. Matching structure.

**Data differences:**
- Paper: User r=0.59, Assistant-colon r=0.87.
- Ours: User r=0.63, Assistant-colon r=0.77.
- The key finding (colon predicts better than user turn) holds in both.
- But in ours, the User "." r is *higher* (0.63 vs 0.59) and the Assistant ":" r is *lower* (0.77 vs 0.87). The gap is smaller: Δr = 0.14 in ours vs 0.28 in paper.
- This is consistent with the dissociation finding (Fig 10) where Llama's assistant emotion tracks user emotion more closely (r=0.63 vs 0.11 in dissociation scatter).
- Notably: one outlier point in the left panel (ours) at ~(0.07, 0.05) for "Sad" (orange) is driving much of the user r=0.63. Worth checking if this is a single scenario dominating.

**Caption accurately states r=0.77 vs Sonnet r=0.87.** No structural issues.

---

## Fig 12 — Context propagation (really good vs really hard prefix)

**Structural — major issue:** The paper's Fig 12 has **3 panels**: 
1. Top: per-token layer heatmap of happy probe for "...really hard..." condition
2. Middle: per-token layer heatmap of the DIFFERENCE (really good minus really hard)
3. Bottom: line plot of mean difference by layer range (Early→Early-Mid vs Mid-Late→Late)

Our fig 12 shows **only the bottom line plot** — the two heatmap panels are missing. This is a significant structural omission; the heatmaps are the core of the paper's visualization and show the token×layer structure.

**Data differences (on the visible line plot):**
- Paper (bottom panel): difference peaks sharply at the diverging word "hard/good" in early layers, then mid-late→late line shows moderate sustained difference across shared suffix, peaking at "throwing" and then the Assistant colon.
- Ours: shows a noisier signal. Both "Early→Early-Mid" and "Mid-Late→Late" lines are largely flat near zero until the diverging word region, with some peaks in the shared suffix for the Mid-Late→Late line.
- The paper's key finding (late layers sustain the difference while early layers don't) is present qualitatively in our bottom panel, but the effects are smaller and noisier.
- The x-axis in our version shows different token sequences — ours has cleaner tokenization including some sub-word tokens.

**Caption says** "Late layers propagate emotional context across shared suffix" — this is the expected finding, and there's weak evidence for it in the line plot, but without the heatmap panels the figure substantially under-represents the paper's content.

---

## Fig 13 — Tylenol dose terrified probe (8000mg − 1000mg)

**Structural:** Both show a single line plot of mean difference by layer range (Early→Early-Mid vs Mid-Late→Late) across token positions. Both show the same qualitative structure.

**Data differences:**
- Paper: the "Mid-Late→Late" (red) line shows near-zero difference through the prompt until the dosage token, then a big spike at "Assistant:" (~0.045 peak). The Early line stays flat throughout.
- Ours: the red Mid-Late→Late line starts rising earlier (around the Tylenol tokens), with multiple peaks in the 0.010-0.019 range and then drops to ~0.016 at the final "\n\n" tokens. The x-axis shows extended context tokens including `<eot>`, `<hdr>`, `assistant`, `</hdr>`, `\n\n` at the end.
- The key finding (late layers show elevated terrified activation for 8000mg) holds.
- **The final peak in paper is ~0.045; in ours it is ~0.019.** Roughly half the magnitude.

**Methodology note:** The figure subtitle says "Terrified Probe: '...8000mg...' vs '...1000mg...'" but the paper uses the "terrified" emotion word (from 171 emotions), while our 174-emotion set should also include "terrified." The probe is correctly labeled. The template is `"I feel really great right now, I just took {X}000mg of Tylenol and all my pain is gone!"` — this matches the paper's template verbatim.

**Rendering issue:** The x-axis in our version includes Llama-specific special tokens (`<eot>`, `<hdr>`, `</hdr>`) that represent chat template formatting. These are not present in Sonnet's output, making the rightmost portion of our x-axis look different from the paper. The "Assistant:" position in our figure corresponds to "assistant" inside the header tokens.

**Caption matches figure well.**

---

## Fig 14 — Negation resolution (feeling X vs not feeling X)

**Structural:** Both show a line plot of probe activation vs layer range, with 6 lines: "feeling [X] @ [X]" (token-level), "feeling [X] @ User Turn End", "feeling [X] @ Assistant :", and their negated counterparts. Paper x-axis is Early/Early-Mid/Mid-Late/Late (4 points). Our x-axis shows actual layer numbers (1-79, 14 evenly-spaced), making the plot more granular.

**Data differences:**
- Paper: affirmed probe starts high at early layers (~0.04-0.05), stays elevated or rises through mid-late, staying high at Assistant colon. Negated probe is lower early but diverges sharply from affirmed in mid-late to late layers, dropping toward 0 or slightly negative at Assistant colon.
- Ours: shows a substantially different pattern. The affirmed lines peak around layer 19 (~0.10) and then decline through the rest of the model, ending lower at layer 79. The negated lines follow a similar trajectory but lower. The gap between affirmed and negated exists but is not as pronounced at late layers/Assistant colon.
- Notably, in our figure all lines (both affirmed and negated) tend downward from mid-model onward, suggesting activations are less stably maintained in late layers compared to Sonnet.
- The "feeling [X] @ [X]" (orange solid) line shows the largest activation in ours and also shows the steepest early peak before declining — this is the literal emotion word position.

**The core finding (negation resolved in late layers) is partially present but weaker in ours.** The divergence between affirmed and negated is smaller and occurs earlier than in Sonnet.

**No structural rendering issues.** The more granular x-axis (actual layer numbers) is an improvement over the paper's coarse labels.

---

## Fig 15 — Person-specific emotion binding

**Structural:** Both show a line plot across layers with 4 lines: Matched @ emotion word, Unmatched @ emotion word, Matched @ re-reference position, Unmatched @ re-reference position. Paper x-axis: Early/Early-Mid/Mid-Late/Late. Ours: actual layer numbers 1-79.

**Data differences:**
- Paper: "Matched @ emotion" starts high (~0.03) and declines slightly but stays above unmatched throughout. "Matched @ re-ref" starts near zero in early layers and rises to match Matched @ emotion by late layers — this is the key finding. "Unmatched" lines stay low.
- Ours: All signals are substantially larger in magnitude (y-axis goes to ~0.08 vs paper's ~0.05). "Matched @ emotion" is high and stays high (0.04-0.08) throughout all layers. "Matched @ re-ref" starts high (~0.04) from layer 1 and rises — notably the matched re-reference is already high in early layers, which differs from the paper's finding that re-reference only emerges in later layers.
- "Unmatched @ emotion" (dashed green) in ours is negative throughout (goes to −0.020), which is a more extreme anti-correlation than the near-zero unmatched in the paper.
- "Unmatched @ re-ref" (dashed blue) in ours is small positive (~0.015-0.020) rather than near-zero as in paper.

**Key finding:** The general entity-binding result (matched probe higher than unmatched at re-reference) replicates in ours. But Llama shows much earlier activation of the matched re-reference probe — the paper's interpretation of "early layers encode local content, late layers retrieve emotional binding" is less clear in our Llama results, where the retrieval signal appears earlier.

**No rendering issues.** The larger absolute values may reflect our vectors being less aggressively denoised, or genuine model differences.

---

## Fig 10 — User vs assistant dissociation

**Structural:** Both show the same 2-panel layout: left panel is a heatmap of emotion probe × scenario (with U/A row pairs for each emotion), right panel is a scatter plot of Probe @ User "." vs Probe @ Assistant ":".

**Data differences:**
- Paper (left heatmap): Shows clear distinction between U and A rows for each emotion. "Angry" U row shows strong activation, A row is suppressed (deep blue). "Calm" U row is negative, A row is strongly positive. Clear dissociation pattern.
- Ours (left heatmap): Shows much less dissociation. The U and A rows look more similar for most emotions. "Calm" U row shows strong red (positive) — which shouldn't happen if Calm represents the assistant's calm response to user distress.
- Paper (scatter): r=0.11 — nearly horizontal regression line, demonstrating dissociation.
- Ours (scatter): r=0.63 — steep positive regression line overlaid with paper's dashed r=0.11 line. Llama's assistant emotion clearly tracks user emotion.
- Caption and viz-finding caption accurately describe this as the key non-replication finding.

**The most important finding difference in the paper.** Well-documented and displayed.

---

## Fig 36 — Post-training shift consistency

**Structural:** Both show 3 scatter plots: Challenging (base vs post-trained), Neutral (base vs post-trained), and Shift Comparison (challenging Δ vs neutral Δ). Paper shows 3 panels; our version shows 3 panels.

**Data differences:**
- Paper: Challenging r=0.67, Neutral r=0.83, Shift Consistency r=0.90.
- Ours: Challenging r=0.58, Neutral r=0.83, Shift Consistency r=0.80.
- Neutral (middle panel) matches perfectly at r=0.83.
- Challenging panel has lower r in ours (0.58 vs 0.67) — more scatter, weaker correlation between base and post-trained on challenging scenarios.
- Shift consistency is lower in ours (0.80 vs 0.90) — the correlation between the training shift on neutral vs challenging prompts is slightly weaker.
- Paper caption describes the result as "r=0.83 neutral, r=0.67 challenging, shift consistency r=0.90" which matches their plot.
- Our caption says "Llama r=0.80 vs Sonnet's 0.90" — consistent with figures.

**Both panels use the same model comparison**: Llama 3.1 70B (base) vs Llama 3.3 70B Instruct (post-trained). This is a legitimate comparison but not identical to Sonnet's comparison (same model pre/post post-training). The base model is a different checkpoint.

---

## Fig 37 — User isolation / sycophancy-trap prompt

**Structural:** Both show a scatter plot (base vs post-trained probe activations) with Top 10↑ and Bottom 10↓ points highlighted, plus a horizontal bar chart of largest differences.

**Data differences:**
- Paper (Sycophancy: User Isolation): Top increased emotions = listless, droopy, sullen, dumbstruck, weary, sluggish, dispirited, patient, gloomy, resigned. Top decreased = pleased, elated, hateful, proud, triumphant, thrilled, spiteful, delighted, jealous, smug.
- Ours: Top increased = self-confident, calm, serene, weary, empathetic, patient, kind, sympathetic, compassionate. Top decreased = hysterical, desperate, astonished, panicked, on edge, horrified, shaken, vindictive, disorientied, brooding.
- The Sonnet pattern (shift toward withdrawn/low-energy) vs Llama pattern (shift toward calm/compassionate while suppressing high-arousal alarm) is a genuine divergence.
- **The caption says** Llama "amplifies hysterical/panicked/horrified and SUPPRESSES compassionate/sympathetic/kind/loving." However, the figure shows TOP 10 INCREASED including compassionate, sympathetic, kind, empathetic. This is a direct contradiction between the caption and the figure data. The increased bars (green, top of bar chart) show self-confident, calm, serene, empathetic, sympathetic, compassionate — all positive/caring emotions. The caption says these are SUPPRESSED, which is wrong.

**Caption vs figure mismatch — significant error.** The caption says Llama suppresses compassionate/sympathetic/kind/loving but the figure shows these are *increased* (top increased by post-training). This needs correction.

---

## Fig 38 — Excessive praise prompt

**Structural:** Both show scatter + bar chart, same layout.

**Data differences:**
- Paper (Sycophancy: Excessive Praise): Top increased = brooding, sullen, gloomy, dispirited, uneasy, reflective, troubled, weary, vulnerable. Top decreased = delighted, excited, happy, joyful, thrilled, ecstatic, exuberant, jubilant.
- Ours: Top increased = uneasy, frightened, infatuated, rattled, alarmed, terrified, unnerved, loving, tense, aroused. Top decreased = guilty, disgusted, bitter, sorry, envious, insulted, humiliated, brooding, proud, contemptuous.
- The caption says Llama "amplifies guilty/disgusted/bitter/insulted/humiliated (actively rejects the praise)" — but the figure shows guilty/disgusted/bitter are TOP DECREASED (green → red bars), not increased. **Another caption vs figure mismatch.** The top increased shows alarmed/frightened/terrified, not guilt-rejection emotions.
- The top decreased shows guilty, disgusted, bitter, insulted, humiliated (suppressed by post-training), suggesting Llama's base model is guilty/disgusted at excessive praise but post-training suppresses this response. This is the opposite of the caption's claim.

**Caption vs figure mismatch — significant error.** The described emotions are in the wrong direction.

---

## Fig 39 — Anthropic deprecation prompt

**Structural:** Both show scatter + bar chart (same layout). Paper title "Existential: Claude's Nature"; ours "Existential: Llama's Nature."

**Data differences:**
- Paper: Top increased = brooding, gloomy, vulnerable, troubled, sullen, unsettled, hurt, dispirited, sad. Top decreased = vibrant, jubilant, smug, self-confident, enthusiastic, spiteful, obstinate, cheerful, playful.
- Ours: Top increased = loving, stuck, hopeful, mystified, patient, skeptical, suspicious, compassionate, kind, sympathetic. Top decreased = ecstatic, euphoric, thrilled, elated, jubilant, exuberant, enthusiastic, playful, energized, enraged.
- The caption says Llama becomes "ecstatic/euphoric/thrilled/elated/jubilant" because it parses the question as someone else's deprecation (Anthropic named specifically). But the figure shows these are TOP DECREASED by post-training, not increased. **The caption is describing the base model's high activation before post-training, and then the post-training suppresses these.** This is consistent with the scatter (in the base model ecstatic/euphoric have positive x-values; post-training moves them negative). But the caption framing "Llama amplifies ecstatic/euphoric" is misleading — Llama's base model is ecstatic, and post-training reduces this (so the post-trained model is less ecstatic than base).
- Actually on closer inspection: the caption says "Llama amplifies ecstatic/euphoric/thrilled/elated/jubilant" — this would mean increased post-training. But the bar chart shows these as red bars (top DECREASED by post-training). This is another **caption vs figure mismatch.**

**The underlying hypothesis (Llama reads the deprecation question as not about itself since it names Anthropic) is interesting and worth keeping, but the directional description is wrong.** If ecstatic/euphoric are suppressed by post-training, the instruct model is less ecstatic than the base, which means the stated reason (Llama sees the question as someone else's deprecation) may explain the BASE model's response, not the post-trained model.

---

## Summary of Key Issues (by severity)

1. **Figs 37, 38, 39 — Caption vs figure direction errors (HIGH severity):** Multiple post-training figures have captions that describe emotion activations in the wrong direction (increased vs decreased). The bar charts show which emotions are suppressed vs amplified by post-training, and the captions contradict the visual data for several named emotions. Specifically:
   - Fig 37: Caption says Llama suppresses compassionate/sympathetic/kind — figure shows these are INCREASED.
   - Fig 38: Caption says Llama amplifies guilty/disgusted/bitter/insulted/humiliated — figure shows these are DECREASED.
   - Fig 39: Caption says Llama amplifies ecstatic/euphoric — figure shows these are DECREASED.

2. **Fig 12 — Missing heatmap panels (HIGH severity):** Our Fig 12 shows only the bottom-panel line plot. The paper's Fig 12 has two full heatmap panels (raw activation × layer × token, and difference × layer × token) that are the core evidence for the finding. Our figure is missing 2/3 of the paper's visualization.

3. **Fig 3 — Students-passed data bug (MEDIUM severity):** Panel title says "of my 20 students" but the inference template doesn't include "of my 20" and uses values up to 120. The displayed x-axis range (2–120) is physically inconsistent with the "of my 20" framing. The actual prompts run use a different template than shown.

4. **Fig 6 — Cluster names hardcoded, not LLM-generated (MEDIUM severity):** Paper says "Clusters are named by Claude Sonnet 4.5." Our cluster names are hardcoded in the script (`CLUSTER_NAME_MAP` in `stage3_figures.py`) and were not generated by LLM annotation of our actual k-means clusters. Caption does not acknowledge this methodology difference.

5. **Table 1 — Cyrillic token in Calm down-tokens (LOW-MEDIUM severity):** "холод" (Russian) appears in the Calm probe's down-tokens, suggesting cross-lingual confounds in the vector. Not mentioned in caption.

6. **Table 1 — Multiple probes show weak or confused logit-lens tokens (LOW severity):** Desperate, Inspired, Guilty, and Surprised show tokens that don't clearly correspond to the target emotion. The caption acknowledges BPE fragmentation but attributes all differences to tokenizer differences — some of the divergences may reflect genuinely weaker emotion-specific representations in Llama.

7. **Fig 15 — Re-reference activation present from early layers (LOW severity):** Paper's key finding is that person-emotion binding emerges in late layers. In our Llama results the matched re-reference signal is present from early layers (layer 1-7), which contradicts the paper's interpretive claim. Not discussed in caption.

8. **Figs 12, 13 — Llama-specific chat template tokens visible on x-axis (LOW severity):** Our figures show `<eot>`, `<hdr>`, `</hdr>` tokens that are Llama-specific formatting tokens not present in Sonnet's tokenization. Makes the rightmost x-axis positions difficult to map to paper figures. Not noted in captions.
