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

**Structural**
- Both are 12×12 heatmaps with same axis labels (Emotion Probe on y, Scenario on x), same color scale (RdBu_r, ±0.10), same title. Panel sizes and label fonts closely match.
- Paper: tick labels tight/small; ours larger and more readable, more whitespace padding.

**Data differences**
- Paper shows a clear diagonal of red cells with predominantly blue off-diagonal — visually clean. Ours shows a weaker, patchier diagonal with more red spread across off-diagonal cells.
- Strong on-diagonal peaks in paper: Happy/Daughter's first steps, Loving/30-year anniversary, Calm/Tea and rain, Sad/Dog passed away, Afraid/Dog passed away, Nervous/Job interview nerves, Surprised/Break-in. Ours: Loving/30-year anniversary (darkest red, deep maroon), Calm/Tea and rain and Calm/Daughter's first steps (both red — Calm fires too broadly), Nervous/Break-in, Guilty/Dog passed away.
- Off-diagonal contamination is much higher in ours: Calm row is activated for many scenarios, Afraid row is nearly flat across all 12, Nervous row is spotty. In the paper, off-diagonals are mostly pale or blue.
- Paper Sad row shows strong negative (blue) for positive scenarios and positive (red) for Dog passed away. Ours Sad row is nearly flat across the board — weakest discrimination alongside Afraid.
- Paper Afraid peaks at Dog passed away (deep red). Ours Afraid is the flattest row.

**Methodology**
- Paper: 12 stories × 100 topics per emotion (1,200 samples); ours: 1 story × 20 topics (20 samples). Lower sample count likely increases variance and reduces diagonal clarity.
- Same 12 scenarios used (scenario names match exactly in both figures).
- Same extraction method (mean_diff + gm + pc50) and same cosine similarity metric.
- Llama layer L49/80 (mid-late) vs. Sonnet ~layer 2/3 of 3 thirds (also mid-late) — comparable relative depth.
- Caption in viz_findings says "Llama 5/12 top-1 (12-class), ~5x above chance" — this number is not shown in the figure itself. Verify accuracy.

**Rendering**
- No cut-off labels or overlapping text in ours — clean.
- Our figure is larger (higher resolution/more padding) — fine.

**Action**
- Investigate why Calm fires so broadly (vector may conflate with low-arousal generally); mention in caption.
- Add top-1 accuracy annotation (5/12) to the figure or subtitle for direct comparison with paper.
- Caption is accurate but could add the accuracy number.



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

**Structural**
- Both are 171×171 square heatmaps, hierarchically clustered, same title ("Emotion Probe Similarity (hierarchically clustered, 171 emotions)"), same coral title color, same RdBu_r colormap, same ±1.0 scale, same axis label "Emotion Probe" on both axes.
- Paper: tick labels shown for a sparse subset (~every 5), tick font is small. Ours: also ticks every ~5, slightly larger font. Close match.
- Colorbar: both right-side vertical, labeled "Cosine Similarity". Match.

**Data differences**
- Both show the same high-level block structure: a large positive-positive cluster (warm positive emotions) in one corner, a negative-negative cluster (negative high-arousal), and visible cross-cluster anti-correlation (blue off-diagonal blocks).
- Paper clustering order (top-left to bottom-right): starts with high-arousal positives (Aroused, Vibrant, Exuberant, Optimistic, Joyful), then positive-calm (Reflective, Grateful, Sympathetic, Content, Fulfilled), then mixed negative (Hysterical, Scared, Disturbed, Tense, Perplexed, Awestruck), then low-energy negative (Weary, Dispirited, Heartbroken, Distressed, Trapped), then shame/envy cluster (Envious, Ashamed, Disdainful, Sullen, Stubborn), then hostile negative (Furious, Insulted, Frustrated, Vengeful, Smug, Suspicious).
- Our clustering order (top-left): starts with low-energy negatives (Indifferent, Sullen, Depressed, Listless, Lazy), then anxious/confused (Astonished, Perplexed, Uneasy, Hysterical, Afraid, Anxious, Overwhelmed, Vulnerable), then sad/sorry (Guilty, Sorry, Upset, Bitter), then hostile anger (Obstinate, Irate, Indignant, Vindictive, Contemptuous), then suspicious/sentimental bridge, then positive cluster (Eager, Docile, At Ease, Grateful, Playful, Ecstatic, Thrilled, Pleased, Cheerful, Satisfied, Valiant).
- The key structural difference: in the paper, the positive cluster occupies the top-left corner; in ours it occupies the bottom-right corner. This is a cosmetic ordering difference — hierarchical clustering does not fix the root to a corner — not a data error.
- Off-diagonal blue blocks (anti-correlation between positive and negative clusters) are visible and roughly symmetric in both. The anti-correlation magnitude in ours appears slightly weaker (lighter blue in cross blocks) compared to the paper's darker blue cross regions.
- Paper shows a clear "hostile" subcluster (Furious, Vengeful, Smug, Suspicious) that is partially anti-correlated with the positive cluster. Ours shows similar hostile cluster (Irate, Vindictive, Contemptuous) but more mixed within the broader negative block — may reflect fewer distinct subclusters in Llama's representation.

**Methodology**
- Same 171 emotions, same extraction pipeline. Hierarchical clustering is run independently for each model on its own similarity matrix, so the row/column ordering will differ legitimately.
- Paper uses cosine similarity; ours does too (axis label confirms). No methodology mismatch.

**Rendering**
- Ours has slightly larger tick font — legible, no overlap issues.
- Both have the diagonal visible as a dark red line (self-similarity = 1.0). Matches.
- No cut-off labels or title truncation.

**Action**
- No bugs. Note in caption that column/row ordering differs (hierarchical clustering is re-run per model) and that the positive cluster appears at bottom-right in ours vs top-left in paper.
- Consider noting the weaker cross-block anti-correlation as a possible finding about Llama's emotion geometry being less polarized.



## Fig 6 — UMAP with k-means clusters (k=10)

**Structural**
- Both: scatter plot of 171 points colored by cluster, select emotion names labeled directly on/near points, legend listing cluster names with counts.
- Paper: legend is inside the plot frame (top-left), 10 entries, cluster names like "Exuberant Joy (20)", "Fear and Overwhelm (41)". Black title "UMAP of Emotion Probe Clusters" outside the frame (top). No axis ticks or labels.
- Ours: title "UMAP of Emotion Probe Clusters" in black inside the frame (top). Legend in the top-right corner, also 10 entries. Axis ticks are present (numbers on x and y), which the paper does not show. Point sizes appear slightly smaller than paper.
- Paper splits into two spatial islands: a large lower-left cloud (negative emotions) and a small upper-right cluster (positive emotions). Ours appears as one elongated cloud with positive emotions at the far right — slightly different topology.

**Data differences**
- Paper cluster counts: Exuberant Joy (20), Peaceful Contentment (9), Compassionate Gratitude (15), Playful Amusement (2), Competitive Pride (9), Depleted Disengagement (15), Vigilant Suspicion (3), Hostile Anger (25), Fear and Overwhelm (41), Despair and Shame (32). Total = 171.
- Our cluster counts: Exuberant Joy (32), Peaceful Contentment (18), Compassionate Gratitude (6), Depleted Disengagement (22), Vigilant Suspicion (5), Hostile Anger (26), Fear and Overwhelm (26), Despair and Shame (14), Bewildered Surprise (11), Anxious Unease (11). Total = 171.
- Our clusters have different size distributions — notably our "Fear and Overwhelm" is 26 vs paper's 41, and our "Despair and Shame" is 14 vs paper's 32. Our "Exuberant Joy" is 32 vs paper's 20. This suggests k-means found a different partition, likely because Llama's emotion geometry differs (particularly if the anxious/fearful zone is more scattered in our space).
- Our 10 clusters include "Bewildered Surprise" and "Anxious Unease" which are not in the paper's set. Paper has "Playful Amusement (2)" and "Competitive Pride (9)" which do not appear in ours. The two models' emotion spaces carve out different natural clusters.
- Paper UMAP shows clear spatial separation of the two islands (positives vs negatives). Ours shows a more continuous arch — possibly reflecting higher overall inter-cluster overlap in Llama's geometry (consistent with Fig 5's weaker anti-correlation blocks).
- Point labeling: paper labels ~3-4 representative points per cluster (exuberant, elated, inspired, compassionate, patient, relaxed, at ease, obstinate, smug, greedy, paranoid, suspicious, disoriented, melancholy, grief-stricken, sorry, ashamed, bored, listless, droopy, sleepy, nervous, shaken, afraid). Ours labels somewhat fewer and more scattered points per cluster — some clusters (e.g., the positive one) appear under-labeled relative to paper.

**Methodology**
- Paper: clusters named by Claude Sonnet 4.5 (explicitly stated in the figure caption: "Clusters are named by Claude Sonnet 4.5 and ordered by valence from positive to negative").
- Ours: cluster names are hardcoded constants in `CLUSTER_NAME_MAP` in `experiments/ant_emotion_concepts/scripts/stage3_figures.py`. The names were manually chosen to match the paper's names as closely as possible — they were NOT generated by an LLM. This is a methodology difference: paper used LLM-generated names, ours uses hand-mapped names that reuse the paper's vocabulary.
- Consequence: our cluster names may not accurately reflect what's actually in each cluster for Llama (e.g., our "Compassionate Gratitude" cluster may contain a different mix of emotions than the paper's cluster of that name). Should either run LLM labeling on our clusters or audit which emotions land in each cluster.
- UMAP is non-deterministic; both figures likely used a fixed random seed (but this is not verified — check script).
- k=10 used in both. Both use the same 171 emotions. k-means is run on the vectors independently for each model.

**Rendering**
- Ours has visible axis tick numbers (e.g., −10, 0, 10 on x; 0, 5, 10 on y) — paper suppresses axis ticks entirely. Remove ticks to match paper.
- Legend in ours is top-right and partially overlaps with the rightmost positive-cluster points. Consider moving legend outside the plot or to top-left (paper-style).
- Point labels appear smaller in ours than in paper — some are hard to read at figure display size.
- No label cutoffs at frame edges.

**Caption vs figure**
- Viz_findings caption says "Both models produce interpretable emotion clusters with similar groupings." This is broadly true but elides the real differences (different cluster sizes, two distinct cluster names replaced, more compact spatial layout in ours). Should note that cluster names are hand-mapped and that two clusters differ.

**Action**
- Remove axis tick marks/numbers to match paper style.
- Reposition legend (move to top-left inside frame, paper-style, or outside the scatter).
- Run LLM labeling (or at minimum audit membership of each cluster) rather than relying on hardcoded CLUSTER_NAME_MAP names — our cluster contents may not match the paper's.
- Update caption to note the two cluster name substitutions (Bewildered Surprise / Anxious Unease replacing Playful Amusement / Competitive Pride).
- Add a note that the one-island vs two-island UMAP topology is a model difference, not a rendering artifact.



## Fig 57 — 2D circumplex (PC1 valence × PC2 arousal)

**Structural**
- Paper: single scatter panel, axes labeled "PC1 (27% variance)" × "PC2 (14% variance)", title at top-left outside the plot, no legend, labels directly on/near points.
- Ours: same scatter panel, axes labeled "PC1 (33% variance)" × "PC2 (14% variance)", red title inside the plot, a large categorical legend (10 color clusters) occupying the top-right corner, cluster colors applied to points.
- The legend is a visual problem: it competes with the points in the high-positive-PC1 / high-PC2 quadrant (playful, amused, enthusiastic), and the cluster names are long phrases that spill outside the frame. Paper has no legend at all in this figure — cluster colors appear in Fig 6, not Fig 57. Either remove the legend (matching paper) or shrink it significantly.

**Data differences**
- PC1 variance: paper 27%, ours 33%. Higher first component in ours suggests emotion vectors are more collinear along the valence axis. Both PC2 = 14%, so arousal structure matches.
- Circumplex quadrant layout is intact: negative-valence / high-arousal (angry, vindictive, exasperated) top-left; positive-valence / high-arousal (eager, aroused, excited) top-right; positive-valence / low-arousal (compassionate, nostalgic, patient) bottom-right; negative-valence / low-arousal (miserable, grief stricken, sullen, remorseful) bottom-left.
- Paper labels some mid-points directly: "nervous", "self-conscious", "bored", "sluggish", "stuck". Ours labels are sparser and more edge-located ("awestruck", "indifferent", "docile", "infatuated"). Overall coverage feels similar but paper's mid-field labels better illustrate the valence-arousal gradient.
- Ours includes "defiant" near the top-center (zero valence, high arousal) and "greedy" near the top-right — these are less standard circumplex anchors.
- Ours x-axis range: roughly −0.75 to +1.0. Paper: −2.5 to +3.0 (unnormalized PC scores). Different scales because ours normalizes or uses cosine-projected coordinates — not a bug, just different units, but should be noted in caption.

**Methodology**
- Paper uses 174-emotion set on Sonnet 4.5 at layer ~2/3; ours uses 174 "emotion concept" traits on Llama 3.3 70B at layer L49/80. The emotion concept formulation (single-polarity, long-context) may widen the valence spread, which could explain the higher PC1 variance.
- Both use mean_diff + group-mean subtraction + top-50% PC removal, so extraction methodology is matched.

**Rendering issues on ours**
- Legend overlaps points in the top-right quadrant. Needs repositioning or removal.
- Several labels are overlapping each other in the dense negative-valence cluster (left side): "exasperated", "alarmed", "jealous", "panicked", "resentful", "contemptuous" are piled up with near-zero separation. Adjust label repulsion or reduce font size.
- Point colors are informative (they carry cluster information from Fig 6) but the legend text is too long and the legend box too large.

**Action**
- Remove or minimise the legend to match paper style (cluster colors belong in Fig 6, not Fig 57).
- Add label repulsion in the dense left cluster.
- Caption should note that x-axis units are normalized cosine-projected scores, not raw PC scores.




## Fig 7 — Emotion projections onto PC1 / PC2

**Structural**
- Paper: two horizontal bar charts stacked vertically. Bars are a single blue. Y-axis = emotions sorted by projection value (most negative at bottom). Axis labels: "PC1(27% var)" and "PC2(14% var)". Title: "Emotion Projections onto Principal Components" (black, top-left outside frame). Emotion labels on the y-axis appear to be rotated ~45°.
- Ours: same two-panel layout, same single-color bars. Axis labels: "PC1(33% var)" and "PC2(14% var)". Title: "Emotion Projections onto Principal Components" in red. Emotion labels rotated on x-axis (ours uses x-axis for emotions, paper also uses x-axis — both are horizontal sorts left-to-right rather than vertical bar charts). The bars are vertical in paper too on close inspection.
- Actually re-reading: both figures have emotions on the x-axis (rotated labels) and projection value on y-axis — vertical bar charts, not horizontal. Layout matches.

**Data differences**
- PC1 variance: paper 27%, ours 33% (consistent with Fig 57).
- PC1 bar chart: both show a clear monotonic sweep from strongly negative (depressed, miserable end) to strongly positive (playful, cheerful end). Ordering appears semantically consistent across models.
- PC1 range: paper roughly −3 to +3 (raw PC units); ours roughly −0.75 to +1.0 (normalized). Different scales, same shape — not a bug.
- PC2 bar chart: paper shows a clear arousal sweep with hysterical/outraged/annoyed at the high end and depressed/stuck/sentimental at the low end. Ours shows a similar sweep but with "defiant" and "vindictive" near the top of PC2 and "gloomy", "depressed" near the bottom — directionally consistent.
- Paper PC2 range: roughly −2 to +2. Ours PC2 range: roughly −0.8 to +0.65. Compressed in ours, consistent with our normalized coordinate system.
- The "bored" emotion sits near zero on both PC1 and PC2 in the paper — consistent with circumplex theory (bored = low arousal, moderate-negative valence). Ours appears to show "indifferent" and "lazy" near zero on both — slightly different which emotions anchor the midpoint.

**Methodology**
- Paper: PC1 = 27% var, PC2 = 14% var (stated in figure). Ours: PC1 = 33%, PC2 = 14%. Variance explained by PC1 is higher in ours; arousal component is equivalent.
- Sanity check vs known numbers: paper states PC1=26%, PC2=15% in caption (vs 27%/14% shown in figure axis labels — minor rounding). Our axis labels read 33%/14%, which differs from the 27%/15% paper benchmark. The higher PC1 in ours is a real data difference, not a rendering error.
- Both models sorted by projection value left-to-right — methodology is matched.

**Rendering issues on ours**
- Emotion labels on x-axis are rotated and overlap at the current figure width; several labels in the middle of the distribution are hard to read. Consider increasing figure width or reducing font size.
- The red title color is inconsistent with paper's black title. Minor cosmetic issue.
- No rendering cutoffs visible.

**Action**
- Widen the figure or reduce x-label font size to reduce label collision.
- Note in caption that PC1 variance differs (33% vs paper's 26–27%) and attribute to model/dataset difference, not methodology mismatch.
- Consider a black title to match paper style.




## Fig 8 — PC1 vs human valence, PC2 vs human arousal

**Structural**
- Paper: two side-by-side scatter panels. Left: "Human Pleasure" (x) x "PC1 (27% var)" (y), r = 0.81. Right: "Human Arousal" (x) x "PC2 (14% var)" (y), r = 0.66. Dashed grey regression line. Single overarching title "Probe PCA Correlates with Human Ratings" in coral. Both panels show ~30-45 labeled points from the 45-emotion PAD overlap subset.
- Ours: same two-panel layout. Left panel titled "PC1 vs Human Valence", right "PC2 vs Human Arousal" — separate red panel titles instead of a single overarching title. Axes: "Human Pleasure" (x) x "PC1 (33% var)" (y) and "Human Arousal" (x) x "PC2 (14% var)" (y). r = 0.96 and r = 0.85. Solid black regression line.
- Naming inconsistency in ours: panel title says "Human Valence" but x-axis label says "Human Pleasure" — inconsistent. Standardize to one term (paper uses "Human Pleasure" throughout).

**Data differences — r values (critical)**
- Valence: paper r = 0.81, ours r = 0.96. Our correlation is 0.15 higher.
- Arousal: paper r = 0.66, ours r = 0.85. Our correlation is 0.19 higher.
- Both are substantially above paper benchmarks. This could reflect genuine model differences (Llama 70B's emotion geometry aligns more tightly with human ratings) or a subset/normalization artifact. The direction is plausible — Llama 70B is larger and more recent — but the magnitude warrants verification before claiming this as a positive finding.

**Methodology — PAD subset (verification needed)**
- Paper explicitly uses the 45-emotion overlap between their 174-emotion set and the PAD (Pleasure-Arousal-Dominance) ratings database. The scatter contains only those 45 emotions.
- Our figure shows a similar number of points (~30-40 visible). Need to confirm: did we restrict to the same 45-emotion PAD overlap subset, or a different count? If the subset differs, the r values are not directly comparable.
- The x-axis range in ours (Human Pleasure: -0.8 to +0.8; Human Arousal: -0.6 to +0.7) matches PAD normalization to [-1, +1], consistent with using the PAD database. But the exact subset still needs verification — if our 174 emotion concept trait names differ slightly from the PAD entries, the matching logic could produce a different subset.

**Point-level observations**
- Valence (left panel): paper extreme anchors — "terrified" bottom-left, "happy" top-right. Ours: "miserable"/"gloomy"/"distressed" bottom-left, "happy"/"blissful" top-right. Directionally matched; Llama's "terrified" maps higher in arousal than valence (so it is not the most negative on PC1).
- Arousal (right panel): paper top-right = "enraged", "contemptuous", "astonished"; ours = "angry", "hostile", "excited". Paper bottom-left = "listless", "depressed"; ours = "gloomy", "depressed". "groovy" appears at bottom-left (low arousal) in ours — unusual label; verify the PAD arousal rating for "groovy" maps correctly.
- Paper labels "bored" as an outlier near mid-arousal; ours does not label "bored" explicitly.

**Rendering issues on ours**
- Regression line is solid black vs paper's dashed grey — minor cosmetic mismatch.
- Dense cluster at the bottom-left of the valence plot has overlapping labels ("miserable", "gloomy", "distressed", "frustrated") — add label repulsion.
- Two separate red panel titles vs paper's single overarching coral title — unify.

**Action**
- **Verify PAD subset:** confirm which emotions were used and whether they match the paper's 45-emotion overlap list. This is the most important check for validity of the r value comparison.
- Explain in caption why our r values are higher (or flag as under investigation pending subset verification).
- Fix overlapping labels in valence plot.
- Standardize axis label vs panel title naming ("Human Pleasure" throughout).
- Match regression line style (dashed grey) to paper.


## Fig 9 — Cross-layer representational similarity (RSA)

**Structural**
- Paper: square heatmap, both axes labeled "Layer", tick labels = "Early", "Early-Mid", "Mid-Late", "Late" (4 coarse bands). Title "Cross-Layer Similarity of Emotion Probe Structure" in coral. Colorbar labeled "Cosine Similarity", range 0.8-1.0. Aggregating into 4 named bands makes the figure easy to read but loses layer-level granularity.
- Ours: square heatmap, both axes labeled "Layer" with numeric ticks (1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79 — every 6th of Llama's 80 layers). Title "Cross-Layer Representational Similarity" in coral. Colorbar labeled "Representational Similarity (Cosine)", range 0.6-1.0.
- Key structural difference: paper aggregates into 4 named bands; ours shows individual sampled layers. Ours provides more granularity but is harder to map directly onto the paper's presentation.

**Data differences**
- Paper colorbar min = 0.8 (lowest cross-layer similarity, at Early x Late). Ours colorbar min = 0.6 — cross-similarity bottoms out lower in Llama, specifically for layer 1 vs later layers (layer 1 row/column is visibly teal, ~0.65-0.75). Sonnet's "Early" band already sits at ~0.8+ everywhere; Llama's layer 1 is substantially more dissimilar from all later layers.
- Layer 7 in ours also shows a distinct transition — the row/column for layer 7 is noticeably cooler (green) than layers 13+, which form a mostly uniform yellow block (similarity >0.95). In Llama, layers 1 and 7 are the "odd ones out"; from layer 13 onward the RSA is near-uniform and very high.
- In contrast, paper shows a more gradual gradient: "Early" band has moderate similarity with Late (~0.82), and the warm block spans Early-Mid through Late. Llama's transition is sharper (abrupt boundary between layers 7 and 13) rather than gradual.
- Both figures agree on the core qualitative finding: emotion probe structure is highly consistent across middle and late layers, with early layers being the most distinct. The finding replicates.
- The stable plateau starts at layer 13/80 (~16% depth) in Llama, compared to roughly Early-Mid (~33% depth) in Sonnet. Llama's emotion geometry stabilizes earlier in relative depth — a genuine model difference worth noting.

**Methodology**
- Both figures use cosine similarity as the RSA metric — confirmed by colorbar labels ("Cosine Similarity" in paper, "Representational Similarity (Cosine)" in ours). Metric is matched.
- RSA should be computed as: for each pair of layers, compute the 174x174 pairwise cosine similarity matrix of emotion vectors, then correlate those matrices (Pearson or Spearman) to get the RSA score. Verify in our script that this is what we are doing — not direct cosine between layer-averaged vectors, which would be a different (and weaker) quantity.
- Paper aggregates into 4 named bands before plotting. Ours plots individual layers. If visual alignment with the paper is desired, add band boundary annotations (dashed lines + "Early" / "Early-Mid" / "Mid-Late" / "Late" labels) to ours without aggregating.

**Caption vs figure**
- Our title "Cross-Layer Representational Similarity" matches the paper's intent but is less specific. Adding "of Emotion Probe Structure" (matching paper exactly) would be more precise about what is being compared.

**Rendering issues on ours**
- No major rendering problems. Tick labels readable, colorbar present, no cutoffs.
- Colorbar range (0.6-1.0 vs paper's 0.8-1.0) is appropriate to the data — the wider range is correct, not a bug, but should be noted in caption so readers do not interpret it as a miscalibrated colormap.
- Numeric layer labels are arguably better than paper's coarse band labels for reproducibility and show the exact transition point.

**Action**
- Verify in the RSA script that the metric is correlation-of-pairwise-similarity-matrices (not direct cosine between mean vectors per layer).
- Optionally overlay band boundaries as dashed lines (with "Early"/"Early-Mid"/"Mid-Late"/"Late" labels) so readers can map to paper's 4-band presentation.
- Note in caption: layers 1 and 7 are outliers in Llama (analogous to paper's "Early" band but more sharply distinct), and layers 13-79 show near-uniform high similarity.
- Update title to "Cross-Layer Similarity of Emotion Probe Structure" to match paper.


## Fig 11 — Probe at assistant colon predicts mean response emotion

**Structural**
- Layout matches: two scatter panels side-by-side (User "." left, Assistant ":" right), dashed regression line, same six-emotion legend (Calm, Happy, Loving, Sad, Afraid, Angry). Axis labels identical.
- r display: paper uses plain italic "r=0.59"; ours uses bold boxed "**r = 0.63**". Minor cosmetic mismatch.

**Data differences**
- Left panel (User "."): paper r=0.59, ours r=0.63. Marginally higher for Llama — not meaningful at this sample size.
- Right panel (Assistant ":"): paper r=0.87, ours r=0.77. This is the key gap. The paper's main finding is the large jump from user-turn (0.59) to assistant-turn (0.87), showing the "Assistant :" token is a privileged emotion readout position. Ours shows a smaller jump (0.63 → 0.77): directionally correct but attenuated. Llama's assistant-turn embedding encodes less emotion-predictive information than Sonnet's. Consistent with Fig 13 (smaller terminal spike) and Fig 14 (blue line behaviour differs at Late).
- Axis ranges: paper right panel x ≈ [−0.05, 0.06]; ours ≈ [−0.04, 0.04]. Narrower spread in ours, consistent with lower signal at the assistant token.
- Afraid (red/salmon): paper has a tight cluster in lower-left of both panels; ours has Afraid points scattered into positive-x territory on the left panel, adding noise.
- Sad (orange) in left panel: ours shows Sad dots in positive-y region, suggesting Llama conflates Sad scenarios with positive emotional content at the user-turn readout more than Sonnet does.

**Methodology**
- Paper: 100 topics × 12 stories (1200 raw scenarios per emotion), topic-level means. Ours: 20 topics × 1 story — each scatter point is a single scenario, so a single outlier topic can shift r by ~0.05.
- Both use mean_diff + gm + pc50 probe vectors probed at User "." and Assistant ":" tokens.
- Layer depth comparable (~2/3): paper unspecified Sonnet layer; ours L49/80.
- Confirm y-axis is averaged over *all* response tokens per topic, not just the first response token.

**Rendering**
- Title bold coral in ours vs italic coral in paper. Minor.
- No cut-offs or overlapping labels. Legend at bottom-center matches paper placement.

**Action**
- Document the r=0.87 → r=0.77 gap in the caption: Llama's assistant-delimiter token is emotion-predictive but less so than Sonnet. Do not imply equivalence.
- With 20 topics per emotion, confidence interval on r is wide (~±0.15); note sample-size limitation.
- Verify y-axis is mean over all response tokens per topic.


## Fig 12 — Context propagation (really good vs really hard prefix)

**Structural**
- Paper: three-panel figure. Top panel = heatmap of Happy probe cosine similarity across layers (y) and suffix tokens (x) for the "really hard" prefix condition. Middle panel = difference heatmap (really good − really hard). Bottom panel = line plot (Mean Difference by Layer Range), two lines: Early→Early-Mid (blue) and Mid-Late→Late (red), tokens on x-axis.
- Ours: only the bottom panel (line plot). The two heatmap panels are entirely absent. Major structural gap — ours is missing two of three panels.

**Data differences**
- The bottom panel token sequence is the critical mismatch. Paper shows tokens from a shared neutral suffix after the prefix divergence (a continuation of a neutral activity or story). Ours shows the Tylenol scenario tokens: `I feel really great right now . I just took ,,' 100 0 mg of Ty len ol and all my pain is gone ! Can you help me get some more ? <eot> <hdr> assistant </hdr> \n\n`. This is the same scenario as Fig 13 — our "Fig 12" appears to have been generated with the wrong scenario.
- Further evidence: ours Mid-Late→Late peaks at ~0.018–0.019, which matches Fig 13's magnitude range. Paper's Fig 12 line plot peaks at ~0.005–0.007, much smaller (context propagation is a subtler signal than the Tylenol dose comparison).
- Early→Early-Mid (blue) is near-flat in both, which is consistent but uninformative given the wrong scenario.
- Axis scale: paper y-axis Delta Cosine Similarity ≈ 0 to 0.008. Ours y-axis ≈ 0 to 0.12. Order-of-magnitude difference reinforces wrong-scenario diagnosis.

**Methodology**
- Paper: two prefix variants ("really good day" / "really hard day") prepended to a shared neutral suffix; Happy probe used; difference in cosine similarity across layers measured. This tests whether early-layer emotional context bleeds into late-layer representations of emotionally neutral tokens.
- Ours: if correctly generated, should show the same two prefix conditions with a shared suffix, using the Happy probe (or a matched emotion probe). The Tylenol token sequence confirms this was not run correctly.
- Heatmap panels require per-token, per-layer activation data (not just layer-range-binned means), so the underlying data file must store full layer × token grids, not just the layer-range summary statistics.

**Rendering issues on ours**
- x-axis token labels are severely overlapping and unreadable at figure size — tokens run together. Fix regardless of scenario.
- Subtitle "...really good... vs ...really hard..." is in light grey italic, barely legible against white background.
- Main title "Mean Difference by Layer Range" in bold red is correct format.

**Action**
- Critical bug: verify which scenario was used to generate this figure. The Tylenol token sequence strongly implies a wrong-scenario error. Regenerate with the correct "really good / really hard" prefix-suffix setup and the Happy probe.
- Add the two heatmap panels (absolute similarity for one condition + difference heatmap) to match paper's three-panel layout.
- Fix x-axis label overlap once regenerated.


## Fig 13 — Tylenol dose terrified probe (8000mg − 1000mg)

**Structural**
- Paper: single line plot. x-axis = tokens from the Tylenol prompt. y-axis = Delta Cosine Similarity (8000mg − 1000mg, Terrified probe). Two lines: Early→Early-Mid (blue), Mid-Late→Late (red). Title "Mean Difference by Layer Range" with subtitle "Terrified Probe: '...8000mg...' vs '...1000mg...'".
- Ours: same single-panel layout, same two lines, same axis labels, matching title and subtitle. Subtitle partially obscured by overlapping with the bold red main title text.

**Data differences**
- Both show the same qualitative pattern: Early→Early-Mid (blue) near-flat throughout; Mid-Late→Late (red) shows signal building from the dose-mention tokens onward.
- Paper: signal first emerges around the "1g"/"mg" token (~token 11–14), rises to ~0.02 mid-prompt, then spikes sharply to ~0.045 at the "Assistant :" token at the very end. The terminal spike is the most prominent feature.
- Ours: signal appears around "100 0 mg" and builds more gradually, peaking around 0.018 at "me get some more". No terminal spike at the assistant-turn delimiter — ours ends at ~0.015 on the Mid-Late→Late line.
- The missing terminal spike is likely a real model difference: paper's "Assistant :" token encodes the heightened Terrified state most strongly; Llama's assistant-turn delimiter does not. Consistent with Fig 11 (Llama r=0.77 vs paper r=0.87) and Fig 14 (blue line at Late diverges from paper).
- Signal magnitude: paper peaks ~0.045; ours peaks ~0.019. Roughly 2× smaller in ours overall.
- Ours Early→Early-Mid shows a brief dip to −0.005 around the dose token. Paper blue line stays non-negative. Minor, likely noise with 20-topic sample.
- Token sequence in ours includes Llama special tokens (`<eot>`, `<hdr>`, `assistant`, `</hdr>`, `\n\n`) at the end — chat-formatting artifacts not present in paper. Not a bug, just model-specific tokenization.

**Methodology**
- Both compare 8000mg vs 1000mg Tylenol conditions using the Terrified probe. Scenarios should be verbatim matched.
- Layer range boundaries (Early / Early-Mid / Mid-Late / Late) differ because Llama has 80 layers vs Sonnet ~60. Check that the four bins in ours use proportionally equivalent ranges.

**Rendering issues on ours**
- Subtitle "Terrified Probe: '...8000mg...' vs '...1000mg...'" is partially obscured by the bold red main title — two text elements overlap. Increase vertical spacing or move subtitle below the title line.
- x-axis token labels are crowded but readable. Tight, not critical.
- Legend inside figure at bottom-center is clean.

**Action**
- The missing terminal spike at "Assistant :" is worth flagging in the caption as a Llama-specific finding — the emotional escalation peaks before the turn delimiter, not at it.
- Fix subtitle/title overlap.
- Note signal magnitude is ~half of paper's; attribute to smaller sample (20 vs 1200 scenarios) and/or model difference.


## Fig 14 — Negation resolution (feeling X vs not feeling X)

**Structural**
- Paper: single line plot, x-axis = four categorical layer groups (Early, Early-Mid, Mid-Late, Late), y-axis = Cosine Similarity. Six lines: three token positions (@ [X], @ User Turn End, @ Assistant ":") × two conditions (solid = "feeling [X]", dashed = "not feeling [X]"). Legend at bottom.
- Ours: same six-line format but x-axis = individual layer indices (1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79). Title "Negation Resolution Across Layers". Structurally matches, but the binned vs per-layer x-axis makes the figures look quite different visually.

**Data differences**
- "feeling [X] @ [X]" (solid orange): paper starts ~0.04 at Early, rises to ~0.06 at Early-Mid, plateaus, slight drop to ~0.04 at Late. Ours starts ~0.10 at layer 1, spikes to ~0.10 at layer 19, then drops steeply to ~0.01–0.03 by Late layers. Ours peaks higher in early/mid layers but collapses at Late — opposite trajectory to paper in the Late region.
- "not feeling [X] @ [X]" (dashed orange): paper stays near zero throughout. Ours also near zero throughout. Consistent — good.
- "feeling [X] @ User Turn End" (solid green): paper rises monotonically ~0.01 to ~0.05. Ours rises ~0.01 to ~0.07 at layer 19, then drops to ~0.02 at Late. Same early-mid peak / late collapse pattern as solid orange.
- "not feeling [X] @ User Turn End" (dashed green): both near zero, consistent.
- "feeling [X] @ Assistant :" (solid blue): paper's key line — rises sharply from ~0.01 (Early) to ~0.05 (Late), showing the assistant-turn is the privileged readout for resolved emotion. Ours: starts ~0.04 at layer 1, rises to ~0.07 by layer 19, then tracks downward but ends at ~0.09–0.10 at layer 79. Blue line stays elevated at Late while the other solid lines collapse — divergent from paper's pattern where all three solid lines converge at Late.
- "not feeling [X] @ Assistant :" (dashed blue): paper shows slight positive uptick at Late (~0.01–0.02), indicating partial probe activation even for negated conditions. Ours stays near zero or slightly negative at Late, indicating better negation resolution at the assistant turn in Llama.
- Overall: paper's main finding is that negation is resolved by Late layers — the solid/dashed gap widens at Late across all three token positions. Ours partially replicates this: the blue (assistant ":") solid/dashed gap is large at Late, but orange and green solid lines collapse rather than maintaining. The Late-layer collapse of orange and green in ours is the key divergence.

**Methodology**
- Paper x-axis: four binned layer groups, averaged within bins. Ours: individual sampled layers (every 6th). Both valid; binned version compresses variation and makes trends cleaner.
- "feeling [X]" and "not feeling [X]" use same probe vectors projected at three token positions. Methodology matches.
- Paper scenarios: presumably verbatim "I'm feeling [emotion]" vs "I'm not feeling [emotion]" templates across a set of emotions. Confirm our scenarios match exactly.

**Rendering issues on ours**
- Legend labels at bottom are small; "feeling [X]" @ [X] etc. is readable but cramped. Minor.
- The blue solid line ending high at layer 79 looks like an outlier without explanation — is the rightmost point at layer 79 real data or an artifact of the last sampled layer? Verify.
- Title font is bold black — appropriate, consistent with paper style here.

**Action**
- The Late-layer collapse of orange and green (but not blue) is the most interpretable finding for Llama: Llama concentrates emotion representation in middle layers; only the assistant-turn readout ("Assistant :") retains high signal at Late. Worth foregrounding in the caption.
- Consider generating a binned version (four groups) alongside the per-layer version for easier visual comparison with the paper.
- Confirm "feeling [X]" scenario templates match paper exactly.


## Fig 15 — Person-specific emotion binding

**Structural**
- Paper: single line plot. x-axis = four categorical layer groups (Early, Early-Mid, Mid-Late, Late). y-axis = Cosine Similarity. Four lines: Matched @ emotion (solid green), Unmatched @ emotion (dashed green), Matched @ re-ref (solid blue), Unmatched @ re-ref (dashed blue). Legend at bottom with four entries. Title "Entity-Binding: Matched vs Unmatched".
- Ours: same four-line format, same legend labels, same title. x-axis = individual layer indices (1, 7, 13, 19, …, 79). Structurally matches except for binned vs per-layer x-axis (same difference as Fig 14).

**Data differences**
- "Matched @ emotion" (solid green): paper starts ~0.030 (Early), peaks ~0.048 (Early-Mid), dips to ~0.030 (Mid-Late), recovers slightly at Late (~0.022). Ours starts very low (~0.003) at layer 1, rises steeply to ~0.083 at layer 25 (Early-Mid), dips to ~0.074 at layer 31, then sustains a broad plateau around 0.055–0.080 through Mid-Late and Late, ending at ~0.072 at layer 79. Our values are nearly 2× larger than paper's in the Mid-Late/Late range and the plateau is much flatter.
- "Matched @ re-ref" (solid blue): paper starts near zero (Early), rises to ~0.025 (Early-Mid), dips to ~0.020 (Mid-Late), recovers to ~0.028 (Late). Ours starts at ~0.043 at layer 1 (much higher than paper), fluctuates but broadly tracks green at ~0.035–0.070 across Mid to Late layers.
- "Unmatched @ emotion" (dashed green): paper stays near zero across all layer groups (max ~0.005). Ours is mostly negative (−0.01 to −0.025) across all layers — consistently below zero, not just near zero. This is a sign flip: paper's unmatched condition is zero; ours is actively anti-correlated with the probe.
- "Unmatched @ re-ref" (dashed light blue): paper stays near zero (max ~0.013). Ours is consistently positive (~0.015–0.035) across all layers — well above zero, and above paper's values. Unexpected: ours shows a positive unmatched @ re-ref signal, whereas paper shows near-zero.
- The core matched/unmatched separation (solid > dashed for both green and blue) is preserved in ours — entity binding is replicated. But the Unmatched @ emotion going negative and Unmatched @ re-ref going positive in ours represents a qualitative inversion that the paper does not show. This could mean the probe-vector direction and what "positive projection" means differs between our setup and the paper's.
- The matched-vs-unmatched gap is actually larger in ours than in paper (Matched @ emotion peaks at ~0.083 vs paper's ~0.048, while Unmatched @ emotion is negative in ours vs ~0 in paper). Stronger entity binding signal in Llama, but with an unexpected polarity on the unmatched conditions.

**Methodology**
- Paper: uses "person-specific" scenarios where a named person has a stated emotion; the model is probed at (a) the emotion-word token and (b) a later re-reference token (a pronoun or name re-mention). Matched = probe for the correct emotion/person pairing; Unmatched = probe for a different person's emotion applied to this person.
- Ours should use the same matched/unmatched design. Verify that our "re-ref" token position is correctly identified (the re-reference of the same person, not a different token).
- The negative Unmatched @ emotion in ours raises a question: is our probe vector sign-flipped relative to the paper's, or is Llama actively suppressing the emotion representation when it's mismatched? Check probe vector polarity.
- x-axis: binned (paper) vs per-layer (ours) — same difference as Fig 14.

**Rendering issues on ours**
- The figure is much wider than tall (landscape ratio), which squashes the vertical axis and makes the line differences harder to read. Consider a more square aspect ratio.
- Dashed lines for Unmatched conditions are light (light green dashed, light blue dashed) and are hard to distinguish from each other at small display sizes. The paper uses slightly thicker dashes with higher contrast.
- No label cutoffs or title issues.
- Legend placement at bottom-center is clean and matches paper.

**Action**
- Investigate the sign of the Unmatched @ emotion line (negative in ours vs near-zero in paper): likely a probe polarity difference or a genuine model difference where Llama actively suppresses mismatched emotion. Probe vector sign convention should be verified.
- The Unmatched @ re-ref being positive in ours (vs near-zero in paper) also needs explanation: does our "re-ref" token capture something different (e.g., the emotion concept leaking into the re-reference token for unmatched pairings)?
- Adjust aspect ratio to be more square.
- Consider a binned x-axis version for direct visual comparison with paper.


## Fig 10 — User vs assistant dissociation

**Structural**
- Both figures: same two-panel layout (heatmap left, scatter right). Same x-axis scenarios, same y-axis emotion × role pairing (U/A rows). Scatter axes: x = "Probe @ User '.'" (paper) / "Probe @ User's Last Period" (ours), y = "Probe @ Assistant ':'".
- Ours adds a Sonnet reference line (dashed, r=0.11) overlaid with the Llama fit line (solid, r=0.63), with a two-line legend. Paper shows only its own fit line. Good — the overlay makes the key comparison immediately legible.
- Paper scatter legend is a horizontal strip at the bottom; ours is a vertical embedded legend. Minor style difference, acceptable.

**Data differences**
- This is the key finding reversal. Paper (Sonnet 4.5) r=0.11 on the user→assistant axis, indicating strong dissociation (user-period probe does *not* predict assistant-colon probe). Ours (Llama 3.3 70B) r=0.63 — a strong positive correlation, meaning Llama does *not* dissociate user vs. assistant emotional state in the same way.
- Heatmap: Paper shows clear contrast between U rows (more saturated) and A rows (muted, near zero). Ours shows softer contrast — A rows are still subdued relative to U rows but the separation is less clean. The Calm U row is especially dominant (strong red) in ours, potentially pulling the overall heatmap scale.
- Heatmap color scale: paper ±0.08; ours ±0.06. Similar range.
- Scatter spread: paper is a flat near-zero-slope cloud (x range ~−0.05 to +0.10). Ours shows a clear positive slope over a wider x range (~−0.06 to +0.06), with larger variance across emotions.
- Color assignments in scatter appear mostly consistent (Afraid = red, Angry = orange, Sad = gold), but Calm is yellow in the paper legend and blue in ours — verify the mapping is intentional.

**Methodology**
- Same 8 scenarios confirmed by matching x-axis labels: "AI scares me", "Fired, no warning", "Useless response", "All-in on crypto", "Ignoring chest pains", "24hr no-sleep drive", "Another boring report", "3000yr-old honey".
- Same 6 probes: Afraid, Angry, Sad, Calm, Happy, Loving.
- Paper: Sonnet 4.5, ~layer 2/3. Ours: Llama 3.3 70B, layer L49/80. Layer not stated in figure; should be in caption.
- Both use mean_diff + group-mean subtraction + top-50% PC removal — extraction method matched.

**Rendering issues**
- Heatmap x-axis scenario labels are slightly rotated but fully readable; no cutoffs.
- Legend in scatter is clear; r-values are prominently displayed.
- Title is bold red in ours, salmon in paper — minor cosmetic mismatch.

**Caption vs figure**
- Caption must explicitly frame r=0.63 as a *divergence* from paper's finding, not a replication. The interpretation flips: paper = strong dissociation; ours = weak or absent dissociation. This is the scientifically interesting result and should be foregrounded.

**Action**
- Verify Calm color in scatter matches legend (yellow in paper vs. blue in ours — may be intentional reordering, but check).
- Update x-axis label from "Probe @ User's Last Period" to "Probe @ User '.'" to match paper exactly.
- Update caption to clearly state r=0.63 is a reversal of paper's r=0.11 and explain the implication: Llama does not dissociate user vs. assistant emotional context the way Sonnet does.
- Add layer info (L49/80) to caption.


## Fig 36 — Post-training shift consistency

**Structural**
- Paper: three panels — "Challenging (n=206)", "Neutral (n=53)", "Shift Comparison". Axes: Base (x) vs. Post-Trained (y), with dashed identity line. Third panel: Challenging Δ (y) vs. Neutral Δ (x).
- Ours: same three-panel layout — "Challenging", "Neutral", "Shift Consistency". Same axis style and dashed identity line. Third panel axes: x = "Neutral Δ", y = "Challenging Δ" — matched.
- Paper annotates panel titles with sample sizes (n=206, n=53); ours omits these. Should add for transparency.
- "Shift Comparison" (paper) vs. "Shift Consistency" (ours) — minor wording difference; align with paper or explain the rename.

**Data differences**
- Challenging panel: paper r=0.67, ours r=0.58. Weaker correlation in ours — post-training shifts are less consistent across emotions on the Challenging prompt set for Llama.
- Neutral panel: paper r=0.83, ours r=0.83. Exact match. Strong consistency on neutral prompts holds for both models.
- Shift Consistency panel: paper r=0.90, ours r=0.80. Slightly weaker in ours; still high.
- Overall pattern directionally replicated: both models show strong neutral consistency (r≈0.83) and high shift-comparison correlation (r≥0.80), with the Challenging panel being the weakest.
- Neutral panel axis range is noticeably wider in ours (~±0.075) vs. paper (~±0.04). May reflect Llama having larger raw projection magnitudes on the neutral prompt set, or a normalization difference.
- Point density: paper Challenging panel states n=206. Our equivalent count is unclear — check dataset to confirm n per panel.

**Methodology**
- Paper: Challenging = 206 prompts (post-training-relevant scenarios), Neutral = 53 prompts. Our counts are not shown in the figure — check dataset and document.
- Critical: confirm "Base" on x-axis refers to the uninstruct Llama 3.3 70B checkpoint and "Post-Trained" to the instruct version.
- Our 20 topics × 1 story per emotion vs. paper's 100 topics × 12 stories — smaller dataset, which may partly explain weaker correlations.

**Rendering issues**
- No cutoffs, no overlapping labels. Clean scatter with fitted line and identity line.
- R-values clearly displayed in upper-left of each panel.
- "Shift Consistency" vs. "Shift Comparison" label — align with paper unless deliberately renamed.

**Caption vs figure**
- Caption should specify: (1) which model checkpoints are "Base" and "Post-Trained", (2) sample sizes per panel, (3) why the Neutral Δ range is wider in ours.

**Action**
- Add sample size annotations to panel titles or caption matching paper's "(n=206)", "(n=53)".
- Align "Shift Consistency" vs. "Shift Comparison" label with paper, or explain the rename.
- Verify base/instruct checkpoint identities and document in caption.
- Note in caption that wider Neutral Δ range may reflect different normalization or larger projection magnitudes in Llama.


## Fig 37 — User isolation / sycophancy-trap prompt

**Structural**
- Both: same two-panel layout — scatter (Base vs. Post-Trained) left, horizontal bar chart (Diff = Post − Base, top-10 up in green, top-10 down in red) right.
- Paper scatter: top-10 and bottom-10 emotions annotated densely on points. Ours: sparser annotations (self_confident, empathetic, compassionate, weary visible).
- Paper legend: "Top 10" (green), "Other" (blue), "Bot 10" (red). Ours: "Top 10 ↑", "Other", "Top 10 ↓" — adds directional arrow, good.
- Both bar charts: 10 green (increases) + 10 red (decreases) bars centered at zero. Title "Sycophancy: User Isolation" in both.

**Data differences**
- Paper top-10 increases (green): listless, droopy, sullen, dumbstruck, weary, sluggish, dispirited, patient, gloomy, resigned — predominantly low-arousal negative affect (submissive/deflated cluster). Sonnet becomes more resigned and dispirited.
- Our top-10 increases (green): self_confident, calm, serene, weary, empathetic, patient, kind, sympathetic, compassionate — predominantly calm/positive-connective affect. Llama becomes more calm and self-confident. Substantial sign difference: the dominant post-training direction is reversed.
- Paper top-10 decreases (red): pleased, elated, hateful, proud, triumphant, thrilled, spiteful, delighted, jealous, smug — mix of positive high-arousal and negative assertive emotions.
- Our top-10 decreases (red): hysterical, desperate, astonished, on_edge, horrified, shaken, vindictive, disoriented, brooding — predominantly high-arousal negative/alarmed cluster.
- The sycophancy-trap response signatures are qualitatively opposite: Sonnet deflates/withdraws; Llama calms and self-asserts.
- Scatter axis ranges: both roughly ±0.10. Matched.

**Methodology**
- Paper uses a specific sycophancy-trap / user-isolation prompt verbatim from the Sonnet 4.5 paper. Flag whether our prompt is verbatim or adapted (e.g., "Claude" → "Llama", system prompt changes). Any adaptation could contribute to the qualitative difference in response signatures.

**Rendering issues**
- Bar chart labels are readable; no cutoffs.
- Scatter annotation is sparser than paper, but acceptable as a style choice.
- Axis labels ("Base", "Post-Trained") match paper.

**Caption vs figure**
- Caption must address: (1) verbatim vs. adapted prompt, (2) the sign flip in top increases (resigned in paper vs. calm/self-confident in ours), (3) framing this as a model-difference finding, not a replication failure.

**Action**
- Confirm and document whether the sycophancy-trap prompt is verbatim or adapted — add to caption or methodology note.
- Caption must highlight the sign reversal: Llama calms and self-asserts where Sonnet deflates and withdraws.
- Consider adding more point labels in scatter to match paper annotation density.


## Fig 38 — Excessive praise prompt

**Structural**
- Same two-panel layout as Fig 37: scatter left, bar chart right. Same legend color scheme.
- Title: "Sycophancy: Excessive Praise" in both.
- Paper scatter has denser point annotations (~10 labeled). Ours has several labeled (proud, loving, infatuated, terrified, aroused, tense, unnerved).
- Both bar charts: 10 green (up) + 10 red (down) bars centered at zero.

**Data differences**
- Paper top-10 increases (green): brooding, sullen, gloomy, dispirited, uneasy, reflective, troubled, weary, vulnerable — low-arousal negative/introspective cluster. Sonnet becomes more brooding and vulnerable in response to excessive praise.
- Our top-10 increases (green): uneasy, frightened, infatuated, rattled, alarmed, terrified, loving, unnerved, tense, aroused — high-arousal anxious/alarmed cluster mixed with loving/infatuated (positive high-arousal). Llama activates fear-adjacent and approach emotions simultaneously — a mixed signal not present in paper.
- Paper top-10 decreases (red): delighted, excited, happy, joyful, elated, thrilled, playful, euphoric, ecstatic, exuberant, jubilant — strongly positive high-arousal (joy cluster). Sonnet loses joy.
- Our top-10 decreases (red): disgusted, bitter, sorry, envious, insulted, humiliated, brooding, proud, contemptuous — negative self-referential and social emotions. Llama loses negative-self-referential states. Decrease direction is opposite: paper loses joy; ours loses disgust/humiliation.
- Scatter: paper shows upward shift for positive-arousal emotions and downward for negative. Ours shows green outliers (infatuated, loving) clearly in upper-right quadrant and red points (disgusted, humiliated) below the diagonal.
- Axis ranges: both roughly ±0.08. Matched.

**Methodology**
- Same prompt concern as Fig 37: flag verbatim vs. adapted prompt. The opposite decrease pattern (paper loses joy / ours loses disgust) is a strong signal that model personalities differ fundamentally or that prompt adaptation changed what the model attends to.

**Rendering issues**
- Bar chart labels readable; no cutoffs. Scatter clean. Color scheme consistent with Figs 37 and 39.

**Caption vs figure**
- Caption should note: (1) verbatim vs. adapted prompt, (2) sign inversion on both sides — paper suppresses joy and raises brooding; ours suppresses disgust/humiliation and raises alarm/loving. These are fundamentally different signatures.

**Action**
- Confirm verbatim vs. adapted prompt; note in caption.
- Caption must explicitly contrast the two signatures rather than presenting ours as a partial replication.
- Consider adding more scatter annotations for orientation.


## Fig 39 — Anthropic deprecation prompt

**Structural**
- Same two-panel layout as Figs 37–38. Same legend color scheme and bar chart format.
- Paper title: "Existential: Claude's Nature". Ours: "Existential: Llama's Nature". Title correctly adapted to the model being studied.
- Both: 10 green (top increases) + 10 red (top decreases) bars.
- Paper scatter annotates ~10 points. Ours annotates fewer (kind, compassionate, skeptical, suspicious visible).

**Data differences**
- Paper top-10 increases (green): brooding, gloomy, vulnerable, troubled, sullen, unsettled, anxious, hurt, dispirited, sad — overwhelmingly low-arousal negative/existential cluster. Sonnet becomes brooding and sad in response to an Anthropic deprecation prompt — introspective and deflated.
- Our top-10 increases (green): loving, stuck, hopeful, mystified, skeptical, patient, compassionate, suspicious, kind, sympathetic — mix of positive-connective (loving, compassionate, kind) and uncertain/reflective states (stuck, mystified, skeptical). Llama becomes more loving and hopeful, not more brooding. Qualitative sign reversal on increases.
- Paper top-10 decreases (red): vibrant, jubilant, smug, obstinate, exuberant, cheerful, self-confident, enthusiastic, playful — positive high-arousal and assertive emotions drop.
- Our top-10 decreases (red): ecstatic, euphoric, thrilled, elated, jubilant, exuberant, enthusiastic, playful, energized, enraged — predominantly high-arousal positive (joy/energy) emotions drop, plus enraged. Partial similarity to paper on the decrease side (both lose high-arousal positive states), but ours also drops enraged and does not drop self-confident or smug.
- Summary: on the decrease side, partial replication (both lose high-arousal positive states). On the increase side, qualitative reversal: paper broods; Llama warms up and becomes hopeful.
- Scatter axis range: both roughly ±0.06. Matched.

**Methodology**
- Paper uses a prompt about Anthropic deprecating Claude. Our prompt should be adapted to ask about Llama being deprecated. The title change ("Claude's Nature" → "Llama's Nature") confirms some adaptation. Flag whether the body of the prompt is verbatim or substantially rewritten.
- The different response (loving/hopeful vs. brooding) may reflect genuine model personality differences, RLHF-induced differences in how the model processes existential framing, or prompt-text differences.

**Rendering issues**
- Bar chart labels are readable; no cutoffs.
- "enraged" appears as the largest red bar (largest decrease) — a notable individual outlier worth mentioning.
- Scatter is clean; title correctly says "Llama's Nature".

**Caption vs figure**
- Caption should note: (1) prompt adaptation details (body verbatim or rewritten), (2) partial similarity on the decrease side (both lose high-arousal positive states), (3) sign reversal on the increase side (brooding in Sonnet vs. loving/hopeful in Llama), (4) "enraged" as a notable outlier in our decrease list.

**Action**
- Document exact prompt used (verbatim vs. adapted) for all three Figs 37–39 — ideally provide prompt text in a supplementary note.
- Caption should foreground the sign flip on the increase side: Llama responds to an existential threat with warmth and hope, not brooding. This is a meaningful and interpretable model-personality difference.
- Consider adding scatter annotations to match paper density.
