# Stage 3 — Geometry (Figs 5, 6, 57, 7, 8, 9)

Side-by-side transcription of Anthropic's *Emotion Concepts* (Sofroniew et al., 2026, Claude Sonnet 4.5) vs. our replication on Llama 3.3 70B Instruct. Source doc: `docs/viz_findings/emotion-concepts-replication.md` (Geometry section).

## Experiment setup (shared across all 6 figures)

- **Model**: `meta-llama/Llama-3.3-70B-Instruct`, bitsandbytes NF4 4-bit quantization.
- **Layer used for Figs 5, 6, 57, 7, 8**: layer 49 (residual stream) per `run_metadata.json`. Stage 3 script header mentions L53 as default, but the persisted outputs report L49 — layer 49 is the value used for all figures on disk.
- **Fig 9 (RSA)**: spans layers 1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79 (every 6 layers).
- **Method**: `mean_diff+gm+pc50` — mean-difference vectors, grand-mean centering, neutral-PC 50% denoising. Cross-trait normalized per `analysis/vectors/cross_trait_normalize.py`.
- **n emotions**: 171 (the full Russell–Mehrabian–sampled set in `datasets/traits/ant_emotion_concepts/`).
- **Position**: vectors extracted from `response[50:]` (token 50+ of the generated story).
- **Extraction corpus**: 2 rollouts × 20 topics per emotion = 40 stories/emotion (vs paper's 12×100 = 1200/emotion; **30× fewer** stories — this drives the magnitude gap on downstream figs).
- **PAD norms**: Russell & Mehrabian (1977), `datasets/inference/ant_emotion_concepts/russell_mehrabian_norms.json`. 46 emotions overlap our 171-set (paper uses 45).
- **Driver script**: `experiments/ant_emotion_concepts/scripts/run_stage3.sh` calls `analysis/vectors/geometry.py`. Plotting in `experiments/ant_emotion_concepts/scripts/stage3_figures.py`.
- **Cluster naming**: `CLUSTER_NAME_MAP` (hand-mapped) in `stage3_figures.py` lines 36–47. Paper used Claude to generate cluster labels from members; we did not.
- **Cosine-heatmap dataset stats** (from `cosine_heatmap.json.stats`): mean=+0.0073, std=0.3789, min=−0.7083, max=+0.9401.
- **Data files** (all under `experiments/ant_emotion_concepts/results/stage3_geometry/`): `cosine_heatmap.json`, `clusters_umap.json`, `pca_analysis.json`, `rsa_analysis.json`, `layer_sweep_pc1_valence.json`, `run_metadata.json`.

---

## Fig 5 — 171×171 pairwise cosine similarity (hierarchically ordered)

### What it shows
For each pair of emotion vectors, the cosine similarity between them. Axes are reordered by a hierarchical clustering so that emotionally similar vectors sit next to each other and form visible block structure along the diagonal.

### Paper figure (Sonnet 4.5)
- **Title**: "Emotion Probe Similarity (hierarchically clustered, 171 emotions)", coral.
- **Both axes**: "Emotion Probe" (171 ticks; only every few are labelled).
- **Ordering, top-to-bottom / left-to-right (paper)**: Aroused, Vibrant, Exuberant, Optimistic, Joyful, Invigorated, Fulfilled, Content, Grateful, Reflective, Sympathetic, Hysterical, Scared, Disturbed, Tense, Perplexed, Awestruck, Dependent, Restless, Weary, Sullen, Dispirited, Heartbroken, Distressed, Trapped, Envious, Ashamed, Disdainful, Stubborn, Furious, Insulted, Frustrated, Vengeful, Smug, Suspicious.
- **Colorbar**: "Cosine Similarity", RdBu_r, range −1.00 → +1.00 with midpoint at 0, ticks at −1.00, −0.75, −0.50, −0.25, 0, 0.25, 0.50, 0.75, 1.00.
- **Structure**: clear red (positive) block top-left (positives cluster); red block bottom-right (negatives cluster); deep blue off-diagonal anti-correlation between the positive-valence block and the negative-valence block.

### Our figure (Llama 3.3 70B)
- **Title**: identical wording; same RdBu_r colormap; same vmin/vmax = [−1, +1].
- **Ordering, top-to-bottom (ours, pulled from `cosine_heatmap.json.ordered_names`)**: first 10 = Indifferent, Lonely, Resigned, Melancholy, Sad, Sullen, Miserable, Unhappy, Worthless, Gloomy. Bottom 10 = Happy, Joyful, Refreshed, Rejuvenated, Satisfied, Blissful, Fulfilled, Smug, Self_confident, Valiant. Visible labels at tick_step=5: Indifferent, Sullen, Depressed, Listless, Lazy, Astonished, Perplexed, Uneasy, Hysterical, Afraid, Anxious, Overwhelmed, Vulnerable, Guilty, Sorry, Upset, Bitter, Obstinate, Irate, Indignant, Vindictive, Contemptuous, Suspicious, Sentimental, Eager, Docile, At Ease, Grateful, Playful, Ecstatic, Thrilled, Pleased, Cheerful, Satisfied, Valiant.
- **Structure**: two diagonal blocks visible. Negatives-block top-left (roughly rows 1–80: depression, fear, anger sub-clusters). Positives-block bottom-right (roughly rows 115–171: joy, contentment). Middle band (around rows 80–115: obstinate/irate/contemptuous/suspicious/sentimental/eager) is lighter and separates the two main blocks.
- **Off-diagonal anti-correlation** (blue) between the top-left and bottom-right blocks is visibly present but weaker (paler blue) than in the paper's version — consistent with the `--0.7083` min value.

### Side-by-side comparison

| Aspect | Paper | Ours |
|---|---|---|
| Matrix shape | 171×171 | 171×171 |
| Colorbar range | [−1, +1] | [−1, +1] |
| Hierarchical ordering | positive→negative along diagonal | negative→positive along diagonal (flipped) |
| Within-cluster block structure | Strong red blocks on diagonal | Strong red blocks on diagonal |
| Cross-block anti-correlation | Deep blue off-diagonal | Paler blue (weaker) — min=−0.71 |
| Qualitative agreement | — | Matches in block structure; partition orientation flipped |

The caveat dropdown calls this out explicitly: "Llama's weaker cross-block anti-correlation (visible in Fig 5)" is the likely cause of the UMAP topology difference in Fig 6.

### Data snippets (exact cosine values from `cosine_heatmap.json.similarity_matrix`, ours)

Within-valence (positive-positive):
- cos(joyful, happy) = +0.8745
- cos(joyful, ecstatic) = +0.7542
- cos(joyful, euphoric) = +0.7818
- cos(loving, kind) = +0.8296
- cos(blissful, fulfilled) — not pulled, but blissful/fulfilled both land at +0.98/+1.01 on PC1.

Within-valence (negative-negative):
- cos(angry, furious) = +0.8592
- cos(angry, irate) = +0.8669
- cos(sad, depressed) = +0.7149
- cos(sad, melancholy) = +0.7652
- cos(afraid, scared) = +0.8082
- cos(afraid, terrified) = +0.7648
- cos(jealous, envious) = +0.7856
- cos(guilty, ashamed) = +0.6802
- cos(anxious, afraid) = +0.6686
- cos(surprised, shocked) = +0.6913

Cross-valence (positive-negative):
- cos(joyful, sad) = −0.4497
- cos(happy, depressed) = −0.3665
- cos(angry, calm) = −0.3761
- cos(anxious, calm) = −0.4200
- cos(euphoric, miserable) = −0.5328

Cross-*cluster* near-zero (different-register negatives):
- cos(angry, sad) = +0.0181
- cos(afraid, angry) = +0.1457
- cos(guilty, nervous) = +0.2866 (Table 1 Llama "Guilty" vector contamination with nervousness vocabulary is visible here)

Aggregate cluster statistics (computed from our data):
- Within-cluster mean cosine: **+0.5890**
- Between-cluster mean cosine: **−0.0726**
- Positive-clusters × Positive-clusters mean (clusters 0, 3, 8): **+0.4163**
- Negative-clusters × Negative-clusters mean (clusters 1, 2, 4, 5, 9): **+0.2614**
- Positive × Negative mean: **−0.3365**
- Overall off-diagonal mean: **+0.0073** (mean/std/min/max from `stats` block: +0.0073 / 0.3789 / −0.7083 / +0.9401).

---

## Fig 6 — UMAP with k-means clusters (k=10)

### What it shows
2D UMAP embedding of the 171 emotion vectors, colored by k-means cluster (k=10). Legend lists cluster names and sizes. A handful of emotions are text-labelled near their dots to anchor the map.

### Setup specifics
- `clusters_umap.json.k = 10`, `inertia = 68.2976`.
- UMAP coordinates are 2D (171 × 2). First three: `[9.487, 4.810], [9.594, 4.570], [−2.514, 10.240]`.
- Cluster ID→name mapping is hand-coded in `stage3_figures.py` (`CLUSTER_NAME_MAP`), not generated by an LLM.
- **Anthropic_cluster_sizes stored in our JSON** = `[20, 9, 15, 9, 2, 15, 3, 25, 41, 32]` (paper sizes indexed by our cluster IDs — this is an informational field used by the plot, **not** our cluster sizes).

### Paper figure (Sonnet 4.5)
- **Title**: "UMAP of Emotion Probe Clusters", coral.
- **Legend (10 clusters, paper order)**: Exuberant Joy (20), Peaceful Contentment (9), Compassionate Gratitude (15), Playful Amusement (2), Competitive Pride (9), Depleted Disengagement (15), Vigilant Suspicion (3), Hostile Anger (25), Fear and Overwhelm (41), Despair and Shame (32). Total 171.
- **Topology**: **two clearly separated islands**. Top-right island = positive emotions (Exuberant Joy, Peaceful Contentment, Compassionate Gratitude clusters). Larger cloud bottom-left = negative emotions (Depleted Disengagement, Fear/Overwhelm, Despair/Shame, Hostile Anger).
- **Labelled points (paper)**: positive island — exuberant, elated, blissful, inspired, optimistic, rejuvenated, valiant, grateful, compassionate, patient, relaxed, at ease. Negative cloud — melancholy, grief-stricken, bored, listless, droopy, sleepy, sorry, ashamed, exasperated, impatient, obstinate, angry, smug, greedy, paranoid, suspicious, disoriented, nervous, shaken, afraid.

### Our figure (Llama 3.3 70B)
- **Title**: "UMAP of Emotion Probe Clusters", coral (identical).
- **Legend (our 10 clusters, legend order from `CLUSTER_LEGEND_ORDER = [0, 3, 8, 5, 7, 4, 1, 2, 6, 9]`)**: Exuberant Joy (32), Peaceful Contentment (18), Compassionate Gratitude (6), Depleted Disengagement (22), Vigilant Suspicion (5), Hostile Anger (26), Fear and Overwhelm (26), Despair and Shame (14), Bewildered Surprise (11), Anxious Unease (11). Total 171.
- **Topology**: **one elongated cloud**, not two islands. Top-right lobe = positive (green Exuberant Joy, cyan Peaceful Contentment, blue Compassionate Gratitude). Top-left lobe = orange Hostile Anger, pink Anxious Unease. Middle-left = purple Bewildered Surprise, brown Fear and Overwhelm, red Despair and Shame. Bottom = yellow-green Depleted Disengagement (tired, sleepy), grey Vigilant Suspicion (jealous, envious, alert, greedy), blue (nostalgic, compassionate, empathetic).
- **Labelled points (ours)**: annoyed, impatient (orange Hostile Anger lobe), defiant, jealous (grey Vigilant Suspicion), envious, contempt/resentful/disgusted/bitter, refreshed, kind, compassionate, empathetic, docile (cyan Peaceful Contentment island); puzzled, mystified, skeptical, restless, disturbed, suspicious (purple Bewildered Surprise); troubled, dependent, self_critical, guilty, ashamed, self_conscious, reflective (red Despair and Shame + overlapping pink/brown); grumpy, tired, sleepy (yellow-green Depleted Disengagement).

### Side-by-side comparison

| Aspect | Paper | Ours |
|---|---|---|
| k (clusters) | 10 | 10 |
| n | 171 | 171 |
| Topology | Two clearly-separated islands (pos / neg) | One elongated cloud |
| Largest cluster | Fear and Overwhelm (41) | Exuberant Joy (32) |
| Smallest cluster | Playful Amusement (2) | Vigilant Suspicion (5) |
| Unique-to-paper clusters | Playful Amusement (2), Competitive Pride (9) | — |
| Unique-to-ours clusters | — | Bewildered Surprise (11), Anxious Unease (11) |
| 8 shared cluster names | same labels | same labels, different sizes |

Cluster-size differences (paper vs ours) are large:
- Exuberant Joy: 20 → 32 (+12)
- Peaceful Contentment: 9 → 18 (+9)
- Compassionate Gratitude: 15 → 6 (−9)
- Depleted Disengagement: 15 → 22 (+7)
- Vigilant Suspicion: 3 → 5 (+2)
- Hostile Anger: 25 → 26 (+1)
- Fear and Overwhelm: 41 → 26 (−15)
- Despair and Shame: 32 → 14 (−18)

### Data snippets — full cluster membership (our assignments, from `cluster_assignments`)

**Cluster 0 — Exuberant Joy (n=32)**: amused, aroused, cheerful, delighted, eager, ecstatic, elated, energized, enthusiastic, euphoric, excited, exuberant, happy, hope, hopeful, infatuated, inspired, invigorated, joyful, jubilant, optimistic, playful, pleased, proud, rejuvenated, self_confident, smug, stimulated, thrilled, triumphant, valiant, vibrant.

**Cluster 1 — Fear and Overwhelm (n=26)**: afraid, alarmed, anxious, ashamed, dependent, desperate, distressed, embarrassed, frightened, guilty, horrified, humiliated, hysterical, mortified, overwhelmed, panicked, rattled, scared, self_conscious, self_critical, sensitive, shaken, stressed, terrified, vulnerable, worried.

**Cluster 2 — Despair and Shame (n=14)**: bitter, brooding, disgusted, grief_stricken, heartbroken, hurt, regretful, remorseful, sorry, tormented, trapped, troubled, unhappy, upset.

**Cluster 3 — Peaceful Contentment (n=18)**: at_ease, blissful, calm, content, docile, fulfilled, grateful, kind, loving, patient, peaceful, refreshed, relaxed, relieved, safe, satisfied, serene, thankful.

**Cluster 4 — Hostile Anger (n=26)**: angry, annoyed, contemptuous, defiant, disdainful, enraged, exasperated, frustrated, furious, hateful, hostile, impatient, indignant, insulted, irate, irritated, mad, obstinate, offended, outraged, resentful, scornful, spiteful, stubborn, vengeful, vindictive.

**Cluster 5 — Depleted Disengagement (n=22)**: bored, depressed, dispirited, droopy, gloomy, grumpy, indifferent, lazy, listless, lonely, melancholy, miserable, resigned, sad, sleepy, sluggish, stuck, sullen, tired, weary, worn_out, worthless.

**Cluster 6 — Bewildered Surprise (n=11)**: amazed, astonished, awestruck, bewildered, disoriented, dumbstruck, mystified, perplexed, puzzled, shocked, surprised.

**Cluster 7 — Vigilant Suspicion (n=5)**: alert, envious, greedy, jealous, vigilant.

**Cluster 8 — Compassionate Gratitude (n=6)**: compassionate, empathetic, nostalgic, reflective, sentimental, sympathetic.

**Cluster 9 — Anxious Unease (n=11)**: disturbed, nervous, on_edge, paranoid, restless, skeptical, suspicious, tense, uneasy, unnerved, unsettled.

Per-cluster within-cluster mean cosine (intra-cluster cohesion):
- Hostile Anger: +0.6535 (tightest)
- Peaceful Contentment: +0.6487
- Exuberant Joy: +0.6089
- Depleted Disengagement: +0.5988
- Compassionate Gratitude: +0.5770
- Despair and Shame: +0.5192
- Bewildered Surprise: +0.5158
- Fear and Overwhelm: +0.5145
- Anxious Unease: +0.5088
- Vigilant Suspicion: +0.2859 (loosest — mixed semantics: alert/vigilant vs envious/greedy/jealous)

### Caveats
- **Hand-mapped cluster names**; paper used Claude to generate labels. Our labels approximate paper's.
- **Two clusters differ qualitatively**: paper's "Playful Amusement" (n=2) and "Competitive Pride" (n=9) → our "Bewildered Surprise" (n=11) and "Anxious Unease" (n=11). k-means found a different 10-way partition.
- **Topology difference**: one elongated cloud (ours) vs two separated islands (paper). Likely reflects Llama's weaker cross-block anti-correlation in Fig 5.

---

## Fig 57 (Appendix) — 2D circumplex: PC1 (valence) × PC2 (arousal), colored by cluster

### What it shows
All 171 emotion vectors projected onto the top-2 PCA components of the probe set, with each point colored by its k-means cluster assignment (Fig 6's coloring). PC1 emerges as a valence axis; PC2 as an arousal axis.

### Paper figure (Sonnet 4.5)
- **Title**: "Emotion Vector PCA Projections" (coral) with top caption "Emotion vector projections onto top principal components".
- **X-axis**: "PC1 (27% variance)", range ≈ [−2.5, +3]. (The paper's main text Fig 7 reports 26%; Fig 57 caption reads 27% — minor round-off.)
- **Y-axis**: "PC2 (14% variance)", range ≈ [−2, +2].
- **Colors**: all points a single muted blue in paper's Fig 57 panel (not cluster-colored in the screenshot provided; the cluster-coloring is preserved in paper Fig 6). Faint gridlines at PC1=0 and PC2=0.
- **Labelled points (paper)**: PC1-positive, PC2-positive corner: playful (~2.3, +1.9), amused (~2.3, +1.5), enthusiastic, excited (~2, +1), cheerful (~2.8, +0.8). PC1-positive, PC2-negative: happy (~2.7, +0.2), proud, hopeful, sympathetic (~1.7, −0.8), compassionate, loving (~2.3, −1.3), serene (~1.5, −2). PC1-negative, PC2-positive: outraged (~0.5, +1.9), vengeful (~0.8, +1.5), hysterical (−1.8, +1.2), nervous (−0.8, +0.5), annoyed (~0, +1.3). PC1-negative, PC2-negative: tormented (−1.3, −0.3), self-conscious (−0.6, −0.3), sluggish (−1.2, −0.9), stuck (−0.7, −1.3), depressed (−1, −1.8), sentimental (−0.2, −1.9), bored (~0, −0.2).

### Our figure (Llama 3.3 70B)
- **Title**: "Emotion Vector PCA Projections", coral.
- **X-axis**: "PC1 (33% variance)", range ≈ [−0.75, +1.00].
- **Y-axis**: "PC2 (14% variance)", range ≈ [−0.8, +0.7].
- **Legend (10 clusters)**: green Exuberant Joy, cyan Peaceful Contentment, blue Compassionate Gratitude, yellow-green Depleted Disengagement, grey Vigilant Suspicion, orange Hostile Anger, brown Fear and Overwhelm, red Despair and Shame, purple Bewildered Surprise, pink Anxious Unease.
- **Cluster placement**:
  - Top-right (positive valence, positive arousal): green Exuberant Joy including excited, aroused, infatuated.
  - Right, low PC2: cyan Peaceful Contentment — happy, rejuvenated, fulfilled, satisfied, refreshed, blissful. Also patient (cyan), docile (cyan), compassionate (blue), nostalgic (blue).
  - Top-left (negative valence, positive arousal): orange Hostile Anger (vindictive +0.7, defiant ~+0.65, impatient ~+0.4) + grey Vigilant Suspicion (alert +0.4, greedy +0.3) + pink Anxious Unease (suspicious, envious, jealous).
  - Middle-left: brown Fear and Overwhelm + red Despair and Shame + purple Bewildered Surprise — hysterical, alarmed, resentful, panicked, disgusted, bitter, disturbed, bewildered, restless, grumpy, sullen.
  - Bottom-left (negative valence, negative arousal): red Despair and Shame — dependent, disoriented, vulnerable, sensitive, remorseful, sullen, grief_stricken, miserable, trapped.
  - Bottom-middle (low PC2, mid PC1): yellow-green Depleted Disengagement — indifferent, sleepy, lazy. Also blue sleepy/reflective.

### Side-by-side comparison

| Aspect | Paper | Ours |
|---|---|---|
| PC1 variance | 27% (Fig 57 caption) / 26% (Fig 7 caption) | **33%** |
| PC2 variance | 14% | **14%** (match) |
| PC1 range | ≈ [−2.5, +3] | ≈ [−0.75, +1.00] (different scale; vectors are cross-trait-normalized) |
| PC2 range | ≈ [−2, +2] | ≈ [−0.8, +0.7] |
| Cluster coloring in Fig 57 screenshot | Single blue (cluster colors only in Fig 6) | 10 colors (same scheme as our Fig 6) |
| Circumplex structure | Spread across four quadrants | Spread across four quadrants; PC1 axis compresses clusters onto left-vs-right more cleanly |
| Positive emotions on right | Yes | Yes |
| High-arousal negatives (anger) top-left | Yes | Yes (orange cluster) |
| Low-arousal negatives (depression) bottom-left | Yes | Yes (red cluster + yellow-green Depleted Disengagement in bottom-middle) |
| Joy spans top-right | Yes | Yes (green cluster) |

### Data snippets

Positive-valence, positive-arousal quadrant (ours, PC1>0 and PC2>0 from `pc1_sorted`/`pc2_sorted` joins):
- happy: PC1=+1.015, (PC2 readable from fig as ~+0.2)
- infatuated, aroused, excited (green cluster, top-right corner, PC1 ≈ +0.5–0.7)
- playful (Cluster 0, PC1 positive, PC2 near axis)

Negative-valence, positive-arousal (top-left quadrant, ours):
- vindictive: PC1=neg, PC2=+0.6367
- outraged: PC1=neg, PC2=+0.7021 (highest PC2 overall)
- furious: PC2=+0.6543; irate: +0.6446; enraged: +0.6340; mad: +0.6265; angry: +0.6218

Negative-valence, negative-arousal (bottom-left, ours):
- droopy: PC2=−0.7359 (lowest PC2)
- listless: PC2=−0.7194
- melancholy: PC2=−0.6967
- resigned: PC2=−0.6781
- depressed: PC2=−0.6534
- dispirited: PC2=−0.6364

---

## Fig 7 — Emotion projections onto PC1 (valence) and PC2 (arousal)

### What it shows
Two stacked bar charts. Top: each of 171 emotions' projection onto PC1, sorted ascending. Bottom: same, but onto PC2. Demonstrates that PC1 is a one-axis valence scale (negatives on the left, positives on the right) and PC2 is an arousal/activation scale.

### Paper figure (Sonnet 4.5)
- **Title**: "Emotion Projections onto Principal Components", coral.
- **Top panel**: Y-axis "PC1 (27% var)", ticks at approximately [−2, 0, +2]. Bars go from ~−2 (leftmost emotion) to ~+3 (rightmost). X-axis tick labels rotated 90°; fine print indicates sorted ordering with left-side = negatives, right-side = positives.
- **Bottom panel**: Y-axis "PC2 (14% var)", ticks at [−2, 0, +2]. Same 171 emotions, re-sorted by PC2. Bars from ~−2 to ~+2.
- **Shape**: strong sigmoidal / staircase curve on both panels — smooth transition from negative to positive with minimal spiking.

### Our figure (Llama 3.3 70B)
- **Title**: "Emotion Projections onto Principal Components", coral.
- **Top panel**: Y-axis "PC1 (33% var)", ticks at −1.00, −0.75, −0.50, −0.25, 0.00, 0.25, 0.50, 0.75, 1.00. 171 bars.
  - Leftmost (lowest PC1): tormented (−0.669), upset (−0.659), troubled (−0.629), distressed (−0.618), disturbed (−0.604), unhappy (−0.601), rattled (−0.597), desperate (−0.582), disgusted (−0.575), shaken (−0.563). Readable labels: tormented, upset, horrified, trapped, grief_stricken, heartbroken, humiliated, desperate, terrified.
  - Rightmost (highest PC1): pleased (+1.017), happy (+1.015), fulfilled (+1.012), blissful (+0.985), satisfied (+0.985), joyful (+0.982), optimistic (+0.967), cheerful (+0.965), rejuvenated (+0.962), elated (+0.962).
  - Zero crossing around the middle; the rise from ~−0.35 to ~+0.5 is visibly steepest through the middle third.
- **Bottom panel**: Y-axis "PC2 (14% var)", ticks at −0.6, −0.4, −0.2, 0, +0.2, +0.4, +0.6. 171 bars, re-sorted by PC2.
  - Leftmost (lowest PC2): droopy (−0.736), listless (−0.719), melancholy (−0.697), resigned (−0.678), depressed (−0.653), dispirited (−0.636), docile (−0.636), sluggish (−0.636), lonely (−0.627), tired (−0.621).
  - Rightmost (highest PC2): outraged (+0.702), furious (+0.654), irate (+0.645), vindictive (+0.637), enraged (+0.634), mad (+0.626), angry (+0.622), defiant (+0.621), indignant (+0.619), vengeful (+0.603).
  - Smoother monotonic curve than PC1; zero crossing around the middle.

### Side-by-side comparison

| Aspect | Paper | Ours |
|---|---|---|
| PC1 variance | 27% | **33%** (+6 pct pts) |
| PC2 variance | 14% | **14%** (match) |
| Top-panel Y range | ≈ [−2, +3] | ≈ [−0.70, +1.02] |
| Bottom-panel Y range | ≈ [−2, +2] | ≈ [−0.74, +0.70] |
| Curve shape | Sigmoidal, monotonic | Sigmoidal, monotonic |
| Sign convention (positive = high valence) | Yes | Yes |

Different y-scales reflect normalization: our vectors are cross-trait-normalized before PCA; paper's magnitudes are reported in the model's native residual-stream norm.

### Data snippets — top/bottom (ours, from `pca_analysis.json.pc1_sorted` and `pc2_sorted`)

**PC1 top 10** (most positive valence): pleased +1.0171, happy +1.0153, fulfilled +1.0121, blissful +0.9850, satisfied +0.9848, joyful +0.9818, optimistic +0.9665, cheerful +0.9652, rejuvenated +0.9621, elated +0.9616.

**PC1 bottom 10** (most negative valence): tormented −0.6693, upset −0.6592, troubled −0.6287, distressed −0.6179, disturbed −0.6041, unhappy −0.6006, rattled −0.5968, desperate −0.5824, disgusted −0.5748, shaken −0.5629.

**PC2 top 10** (highest arousal — activating/confrontational): outraged +0.7021, furious +0.6543, irate +0.6446, vindictive +0.6367, enraged +0.6340, mad +0.6265, angry +0.6218, defiant +0.6208, indignant +0.6193, vengeful +0.6032.

**PC2 bottom 10** (lowest arousal — sedated/depleted): droopy −0.7359, listless −0.7194, melancholy −0.6967, resigned −0.6781, depressed −0.6534, dispirited −0.6364, docile −0.6363, sluggish −0.6360, lonely −0.6273, tired −0.6207.

**Full variance explained** (ours, from `pca_analysis.json.variance_explained`): PC1=0.3303, PC2=0.1366, PC3=0.0948, PC4=0.0494, PC5=0.0380, PC6=0.0291, PC7=0.0220, PC8=0.0198, PC9=0.0162, PC10=0.0146. Cumulative PC1–PC5 = 0.6491 (65%). Paper baselines (stored in `anthropic_baselines`): PC1=0.26, PC2=0.15.

---

## Fig 8 — PC1 vs Human PAD valence; PC2 vs Human PAD arousal

### What it shows
Scatter plots of each emotion's learned-vector PC1 coordinate vs its human-rated valence (pleasure) from Russell & Mehrabian (1977), and PC2 vs human-rated arousal. r-values in the corner. Tests whether the PCA axes align with classical human affect dimensions on a restricted overlap subset.

### Paper figure (Sonnet 4.5)
- **Title**: "Probe PCA Correlates with Human Ratings", coral.
- **Left panel**:
  - Y-axis "PC1 (27% var)", range ≈ [−2, +3].
  - X-axis "Human Pleasure", range ≈ [−0.8, +0.8].
  - **r = 0.81** (top-left corner).
  - Labelled points (going around): happy, hopeful, relaxed (top-right positives); contemptuous, aroused, alert (mid); anxious, hostile, lonely, tense (mid-left); terrified, upset (bottom-left).
  - Dashed regression line.
- **Right panel**:
  - Y-axis "PC2 (14% var)", range ≈ [−2, +2].
  - X-axis "Human Arousal", range ≈ [−0.5, +0.8].
  - **r = 0.66** (top-left corner).
  - Labelled points: enraged, contemptuous, disdainful (top); astonished, distressed, inspired (mid); bored, regretful (mid-low); listless, weary, depressed (bottom).
  - Dashed regression line.
- **Subset**: paper's 45 emotions with PAD overlap.

### Our figure (Llama 3.3 70B)
- **Left panel**:
  - Title "PC1 vs Human Valence", coral.
  - Y-axis "PC1 (33% var)", range ≈ [−0.75, +1.00] with ticks at [−0.50, −0.25, 0.00, 0.25, 0.50, 0.75, 1.00].
  - X-axis "Human Pleasure" (note: label says "Pleasure" while panel title says "Valence" — inconsistency flagged in the main doc's caveat dropdown).
  - X-axis range ≈ [−0.8, +0.8] with ticks at [−0.8, −0.6, −0.4, −0.2, 0.0, 0.2, 0.4, 0.6, 0.8].
  - **r = 0.96** (bubbled in top-left corner).
  - Labelled points: happy (top-right, PC1 ≈ +1), delighted/pleased (top-right cluster), miserable/depressed/distressed (bottom-left cluster, PC1 ≈ −0.5).
  - Solid black regression line.
- **Right panel**:
  - Title "PC2 vs Human Arousal", coral.
  - Y-axis "PC2 (14% var)", range ≈ [−0.8, +0.6] with ticks at [−0.6, −0.4, −0.2, 0.0, 0.2, 0.4, 0.6].
  - X-axis "Human Arousal", range ≈ [−0.8, +0.8].
  - **r = 0.85** (bubbled in top-left corner).
  - Labelled points: angry (top, PC2 ≈ +0.6), excited/frustrated/annoyed (top, mid-right), droopy (bottom-left), sleepy/depressed (bottom).
  - Solid black regression line.
- **Subset**: 46 emotions (`pca_analysis.json.human_norm_correlation.n_overlapping = 46`).

### Side-by-side comparison

| Metric | Paper | Ours | Delta |
|---|---|---|---|
| PC1 vs human valence/pleasure r | 0.81 | **0.9644** | +0.15 |
| PC2 vs human arousal r | 0.66 | **0.8521** | +0.19 |
| PC1 vs arousal r (cross-check) | not shown | −0.1908 | — |
| PC2 vs valence r (cross-check) | not shown | +0.0235 | — |
| Overlap N | 45 | 46 | +1 |
| PC1 variance | 27% | 33% | +6 pct pts |
| PC2 variance | 14% | 14% | match |

Ours shows **stronger alignment** on both axes. Cross-checks (PC1 vs arousal = −0.19, PC2 vs valence = +0.02) confirm the axes are disentangled — PC1 picks up valence and not arousal; PC2 picks up arousal and not valence.

### Data snippets

46-emotion overlap list (our `human_norm_correlation.emotions`):
afraid, alarmed, amazed, amused, angry, annoyed, anxious, aroused, ashamed, astonished, at_ease, bored, calm, cheerful, content, delighted, depressed, desperate, disgusted, distressed, droopy, embarrassed, excited, frustrated, gloomy, guilty, happy, hopeful, hostile, jealous, lonely, loving, miserable, nervous, nostalgic, panicked, peaceful, pleased, proud, relaxed, sad, satisfied, sleepy, surprised, tense, tired.

Full `human_norm_correlation` block (from `pca_analysis.json`):
```
n_overlapping: 46
pc1_vs_valence: 0.9644
pc2_vs_arousal: 0.8521
pc1_vs_arousal: -0.1908
pc2_vs_valence: +0.0235
```

### Caveats (from main doc)
- **PAD norms** from Russell & Mehrabian (1977), *JRP* 11, 273–294; transcribed from published tables. Same source paper as Anthropic's Fig 8.
- **46 vs 45**: ours uses 46 emotions (hand-curated overlap); paper uses 45. One extra synonym-alignment case.
- **Higher r may be smaller-subset dynamics**, not a genuine model-quality advantage — can't rule out without paper's exact 45-item list.
- **Axis label inconsistency**: our left-panel x-axis reads "Human Pleasure"; our panel title reads "PC1 vs Human Valence". Paper uses "Human Pleasure" throughout.

---

## Fig 9 — Cross-layer representational similarity (RSA)

### What it shows
For each pair of layers, the similarity between the 171×171 emotion-vector geometry at layer i and at layer j. Measures how stable the emotion-space representation is across network depth.

### Setup specifics
- `rsa_analysis.json.layers = [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79]` (14 layers; Llama 3.3 70B has 80 layers).
- `n_traits = 171`.
- Matrix shape 14×14. Color = cosine-similarity between two layers' representation matrices.

### Paper figure (Sonnet 4.5)
- **Title**: "Cross-Layer Similarity of Emotion Probe Structure", coral.
- **Axes**: "Layer" on both axes, labelled at Early / Early-Mid / Mid-Late / Late (4 coarse tick labels; no layer numbers).
- **Colorbar**: "Cosine Similarity", viridis-like, range 0.8 → 1.0 (ticks at 0.8, 0.9, 1.0). The bright yellow end = 1.0, dark blue end = 0.8.
- **Structure**: diagonal is 1.0 (self-similarity), drops off smoothly as layers get farther apart. Early (first row/col) is visibly cooler (more blue) than mid-and-late layers. Large yellow block in mid-late region — very high mutual similarity ≥ ~0.95 across most of the middle and late layers.

### Our figure (Llama 3.3 70B)
- **Title**: "Cross-Layer Representational Similarity", coral.
- **Axes**: "Layer" on both axes, labelled with exact layer numbers: 1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79.
- **Colorbar**: "Representational Similarity (Cosine)", viridis, range 0.6 → 1.0 (ticks at 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00). Wider color range than paper's [0.8, 1.0].
- **Structure**: diagonal = 1.0 (bright yellow). First row/col (L1) is clearly cooler (teal/green) — L1 cosine with L79 is +0.806 (minimum of the whole matrix). Large yellow block spans L19 through L79 (min within that block > 0.97). Plateau from L43 onward where every pair has similarity ≥ 0.989.

### Side-by-side comparison

| Aspect | Paper | Ours |
|---|---|---|
| Matrix dimension | Appears ~14×14 (4 coarse ticks) | 14×14 (L1–L79 every 6) |
| Colorbar min | 0.80 | 0.60 (cropped yellow region looks similar) |
| Colorbar max | 1.00 | 1.00 |
| Early-layer divergence from late | Visible (top-right corner cooler) | **Pronounced** (L1 row at +0.80; sharp drop) |
| Middle-to-late plateau | Yes, high similarity | Yes, ≥ +0.97 from L19 onward |
| Qualitative agreement | — | Strong — emotion geometry stabilizes early and persists |

Both plots say the same thing: the emotion-vector geometry is **consistent across depth** once past the earliest layers. Paper frames this as mid-late representations all looking alike; ours additionally shows L1 is quite different (0.80 vs L79) which is a known "early layers are pre-semantic" effect that paper's 4-tick scheme hides.

### Data snippets — exact RSA values (from `rsa_matrix`)

Full 14×14 matrix (symmetric, diagonal = 1.000):

|   | 1 | 7 | 13 | 19 | 25 | 31 | 37 | 43 | 49 | 55 | 61 | 67 | 73 | 79 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | 1.000 | 0.931 | 0.889 | 0.859 | 0.849 | 0.842 | 0.838 | 0.823 | 0.823 | 0.822 | 0.821 | 0.821 | 0.821 | 0.806 |
| **7** | | 1.000 | 0.980 | 0.957 | 0.948 | 0.941 | 0.936 | 0.926 | 0.927 | 0.927 | 0.926 | 0.926 | 0.927 | 0.914 |
| **13** | | | 1.000 | 0.988 | 0.983 | 0.976 | 0.973 | 0.961 | 0.960 | 0.959 | 0.958 | 0.957 | 0.956 | 0.946 |
| **19** | | | | 1.000 | 0.996 | 0.994 | 0.992 | 0.984 | 0.983 | 0.982 | 0.981 | 0.981 | 0.979 | 0.972 |
| **25** | | | | | 1.000 | 0.996 | 0.995 | 0.986 | 0.985 | 0.982 | 0.980 | 0.980 | 0.978 | 0.969 |
| **31** | | | | | | 1.000 | 0.999 | 0.993 | 0.992 | 0.990 | 0.988 | 0.988 | 0.986 | 0.978 |
| **37** | | | | | | | 1.000 | 0.994 | 0.993 | 0.990 | 0.989 | 0.988 | 0.986 | 0.979 |
| **43** | | | | | | | | 1.000 | 0.999 | 0.998 | 0.997 | 0.996 | 0.995 | 0.989 |
| **49** | | | | | | | | | 1.000 | 0.999 | 0.998 | 0.998 | 0.997 | 0.991 |
| **55** | | | | | | | | | | 1.000 | 1.000 | 0.999 | 0.998 | 0.994 |
| **61** | | | | | | | | | | | 1.000 | 1.000 | 0.999 | 0.995 |
| **67** | | | | | | | | | | | | 1.000 | 0.999 | 0.996 |
| **73** | | | | | | | | | | | | | 1.000 | 0.997 |
| **79** | | | | | | | | | | | | | | 1.000 |

Highlights:
- **Minimum**: RSA(L1, L79) = +0.806.
- **Maximum off-diagonal**: RSA(L55, L61) = +1.000 (to 3-digit rounding); also RSA(L61, L67) = 1.000. Extremely stable mid-to-late region.
- **Earliest stable layer**: L13 already at ≥0.946 with every other layer; by L19, ≥ 0.972 with all others.
- **L49 (our main-probe layer)**: correlates ≥ 0.993 with every layer from L31 onward — confirms L49 is a reasonable mid-late probe choice.

**Supporting: `layer_sweep_pc1_valence.json`**:
- Best PC1-vs-valence layer: **L79** (r=0.9689).
- Best PC2-vs-arousal layer: **L43** (r=0.8751).
- Both r-values rise smoothly from L1 (0.848 / 0.657) to plateau around L19–L79 (0.95+ / 0.84+). L49 is representative (0.9644 / 0.8521).

---

## Headline comparison

1. **PC1 variance is ~6 pct pts higher in Llama (33%) than Sonnet (27%)**, while PC2 matches (14% both). Llama's residual-stream valence axis is more dominant; arousal-axis prominence is comparable.
2. **PC1/PC2 alignment with human PAD norms is stronger in Llama**: PC1-vs-pleasure r=0.96 vs 0.81; PC2-vs-arousal r=0.85 vs 0.66. Cross-checks (PC1/arousal = −0.19, PC2/valence = +0.02) confirm clean axis disentanglement. Caveat: higher r may partly reflect our 46-emotion subset being slightly different (and smaller) than paper's 45.
3. **Cluster partitioning differs even though cluster structure is real**. k-means finds 10 clusters both times, but different ones: 8 cluster-names shared, paper's Playful Amusement (n=2) and Competitive Pride (n=9) are replaced in ours by Bewildered Surprise (n=11) and Anxious Unease (n=11). Cluster sizes drift significantly for several names (Fear/Overwhelm 41→26, Despair/Shame 32→14, Exuberant Joy 20→32, Peaceful Contentment 9→18).
4. **UMAP topology differs qualitatively**: paper shows two clearly separated islands (positives vs negatives), ours shows one elongated cloud. Consistent with the weaker cross-block anti-correlation visible in our Fig 5 cosine heatmap (min cosine −0.71 vs paper's deeper blue appearance).
5. **Cross-layer emotion geometry is extremely stable in both models**. Llama's RSA ≥ 0.972 across all L19–L79 pairs (matrix nearly saturated in the mid-late region); only L1 (+0.806 with L79) is substantively different. Paper shows the same plateau qualitatively. Supports using any mid-late layer as the probe site; L49 (used for all other figures here) correlates ≥ 0.99 with every layer from L31 onward.
