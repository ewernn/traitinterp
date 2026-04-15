# TODO — LessWrong post by April 12

## LW Post (critical path)
- [ ] Solo LW post first draft — finish Selected findings + Limitations sections
- [ ] Solo LW post polish (read out loud, cut 20%, tighten claims)
- [ ] Companion LW post outline (Emotion Concepts replication + MATS findings, with Sriram)
- [ ] Companion LW post first draft
- [ ] Companion LW post polish
- [ ] Review both posts (Sriram and/or Oscar)
- [ ] Publish both posts (solo morning, companion afternoon)
- [ ] Cross-post to AlignmentForum

## Site / viz_findings
- [ ] Overview v5 — add citations from literature map
- [ ] Verify Qwen model ID in overview (`Qwen/Qwen3.5-9B`)
- [ ] Replace `[link]` placeholders in overview finding bullets
- [ ] Confirm starter trait names in overview match `datasets/traits/starter_traits/`
- [ ] Make live-chat click-to-warm instead of auto-loading
- [ ] Clean remaining viz_findings (massive-activations needs restructuring, 8 others need a pass)

## Post-publish
- [ ] Tweet thread with hook + figure + link
- [ ] Personal website cleanup (remove distracting projects, link traitinterp.com, update bio)
- [ ] Message Oscar with LW post link
- [ ] Optional: email Anthropic Fellows with LW post link

## Experiments (optional)
- [ ] Base-beats-instruct finding for live-chat traits (evil, hallucination, sycophancy)
- [ ] Base-vs-story extraction experiment (optional, if time)
- [ ] In the convolution post, cite Anthropic's Emotion Concepts local > global influence finding

## Other
- [ ] 30-min glance at Sriram fingerprint work (promised meeting)
- [ ] Run Emotion Concepts replication experiments (PCA, clustering, geometry — scope TBD with Sriram)
- [ ] Job applications 2-3/day throughout sprint

## Backlog
- [ ] Second LW post about traitinterp methodology — save for post-decision
- [ ] Vetting prompt redesign — `<score_this>` XML tagging
- [ ] Dataset format unification — trait.yaml metadata + cascade
- [ ] Analyze past Anthropic Fellows for commonalities

---

## Context for Claude (don't need to read yourself)

### Mobile bug
Site too wide, doesn't fit screen. Probably CSS overflow issue in the viz findings or sidebar. Start in `visualization/styles.css`.

### Overview v5 citations (from web Claude literature map)
In order of impact:
1. **MacDiarmid et al. 2024 "Simple Probes Can Catch Sleeper Agents"** (Anthropic) — linear probes detect sleeper agent defection at >99% AUROC using generic contrast pairs. Add to "Trait probes can cheaply monitor" paragraph as existence proof.
2. **CoT unfaithfulness numbers** — Chen, Benton, Radhakrishnan et al. 2025 ("CoTs reveal hint-usage below 20% of the time"); Lanham et al. 2023 (inverse scaling for CoT faithfulness); Alignment Forum 2025 (CoT monitoring underperforms action-only by up to 44%). Add to "Outputs are too late" paragraph.
3. **Casper et al. 2024 "Black-Box Access is Insufficient for Rigorous AI Audits"** — "it is impossible to make guarantees about black-box systems using a finite number of queries without additional assumptions." Foundational citation. One-line add at start of "Why we need this."
4. **Gupta & Jenner 2025 "RL-Obfuscation"** — architectural diversity matters for evasion resistance, not just probe count. Upgrade ensemble paragraph from "open empirical question" to cited directional claim.
5. **Sofroniew et al. 2026 direct quotes** — "better monitoring than attempting to build a watchlist of specific problematic behaviors" and "transparency should be a guiding principle." Replace current paraphrase.

### Methodology placeholders
6 `:::placeholder:::` blocks in `docs/methodology.md`. Fastest fix is delete. Alternative is pull figures from existing `viz_findings/` docs. Anything is better than visible `[placeholder]` text.

### Convolution detector finding doc
Needs: frontmatter (title, preview, thumbnail), `ag_convolution_detector.png` figure, honest metric distinction (89% threshold detection vs 55% localization), controls (shuffled template z=3.1, random trait baseline z=1.35), connection to Emotion Concepts paper. See `experiments/rm_syco/rm_sycophancy/analysis/session_findings.md` for the raw numbers.

### Live-chat click-to-warm
Currently the live-chat view probably loads the model automatically when the tab is visited. Change to: show the interface but require an explicit "start session" click to load the model. Saves compute when no one is actively using it. Check `visualization/views/live-chat.js` and `visualization/chat_inference.py`.

### Solo LW post structure (traitinterp framework release)
~2400 words. The post IS about the repo. Findings are evidence, not the point.

1. **Abstract** (~150w) — what traitinterp is, why it matters, headline results, mention companion post
2. **Why trait vectors** (~200w) — core argument only (outputs too late, ensembles, scaling). Not an essay.
3. **Traitinterp** (~1500w) — the bulk of the post:
   - What it is + ships with starter traits
   - How to use it (quick tour: bash commands + dashboard screenshot)
   - How it works (compressed methodology — your approach as default, alternatives in one line)
   - Engineering for reuse (auto batching, PathBuilder, position DSL, quant support)
   - Extensibility (add your own traits, any model)
4. **Selected findings** (~400w) — claim + figure + link to traitinterp.com per finding:
   - Reward hacking reduction (63%)
   - Persona vectors replication (91-104%)
   - Convolution detector (9→43 OOD) — brief here, full story in companion post
   - Component decomposition
   - Model recognizes own voice
   - Quant sensitivity
5. **What's next** (~100w) — companion post with Sriram, vision, call to use

Reuse overview.md and methodology.md verbatim where possible (write once, use on site + LW).

### Companion LW post structure (Emotion Concepts replication + MATS findings, with Sriram)
Published same day, afternoon. Links back to solo post for the pipeline.

1. **Abstract** — we replicate Anthropic's Emotion Concepts methodology on open-source models using traitinterp, and present MATS exploration phase findings
2. **Emotion Concepts replication** — their methodology on open models:
   - PCA / clustering / geometry on ~170 trait set
   - Story-based vs contrastive extraction comparison (if base-vs-story experiment done)
   - What transfers, what doesn't
3. **MATS findings** — the fingerprint work:
   - Emergent misalignment fingerprints
   - Cross-seed consistency (same behavior, different internal paths, r=0.084 for s42)
   - Checkpoint dynamics (trait shifts precede behavioral onset by ~25 training steps)
   - Cross-model transfer (Llama 70B → Qwen 4B)
   - Convolution detector in depth (9→43 OOD, shuffled template z=3.1, pre-onset signal)
4. **Discussion** — what this means for monitoring, honest caveats
5. **Links** — traitinterp.com, companion solo post, GitHub

### Sriram fingerprint revisit
Promised meeting tomorrow. Core idea: fingerprints via same-text prefill through clean model and instruct variant, subtract activations. Z-score + weight each trait's contribution to the fingerprint by its delta change before computing similarity between variants. The "top 23 delta traits" picked arbitrarily was naive — some emergent misaligned behaviors are stronger deltas than others, and we need to capture this in the correlation. Emergent misalignment is the right test bed because the persona is persistent across the full response, not a rare backdoor activation. 30-min glance only — don't let it eat into LW post time.

### "Local > global" citation for convolution post
Anthropic's Emotion Concepts paper shows emotion representations track the operative emotion at a given token position, not a persistent character state — they call these "locally operative" representations. Layer analysis: early layers encode local emotional connotations of present content, middle-late layers encode emotions relevant to predicting upcoming tokens. This is the motivation for temporal convolution windows in the detector — the signal is local, not global, so a sliding window over local activations captures what a full-response average would smear out. Cite this when introducing the convolution approach in the LW post.

### Base-beats-instruct for live-chat traits
Prior work (`docs/viz_findings/comparison-persona-vectors.md`, `experiments/persona_vectors_replication/`) showed base model extraction achieves 91-104% of instruct-based steering effectiveness on Llama-3.1-8B, with more authentic behavior (conversational vs theatrical). Current live-chat extraction uses instruct model + system prompts because the starter_traits datasets are `.jsonl` with `system_prompt` fields. The weak traits (evil 50.5% vet pass, hallucination 68%, assistant_axis 25%) fail because the instruct model plays a character rather than exhibiting the trait — exactly what the persona vectors comparison documented.

**Task:** Write natural-elicitation `.txt` datasets for evil, hallucination, sycophancy (like refusal already has — different prompts, no system prompts), extract on `Qwen/Qwen3.5-9B-Base`, compare steering deltas. If base beats instruct here too, it validates the finding on a new model family and gives us better live-chat vectors.

**Files:** `datasets/traits/starter_traits/refusal/` (example of `.txt` natural format), `experiments/live-chat/config.json` (base variant = `Qwen/Qwen3.5-9B-Base`), `docs/viz_findings/comparison-persona-vectors.md` (prior results)

### Base-vs-story experiment (if time)
Pick 1 emotion from Anthropic's Emotion Concepts set. Extract both ways (their story method + contrastive doc-completion) on same base model. Compare steering efficiency per unit norm on a ~20-question multi-choice eval. Find coherence cliff for each. Report which steers stronger / maintains coherence further / shifts preference more. One table, clean head-to-head.

Emotion selection: one where Anthropic's method was weakest, OR one where pre/post-training delta was maximum (suggests their method is affected by training).

### Daily discipline
- No days off until April 12
- First 3 actions written night before
- Touch grass 20 min minimum daily
- Email inbox twice/day max
- Two-day rule: don't skip same item two days in a row

### What NOT to do
- No second LW post this sprint (save methodology-focused post for after decision)
- No site redesign
- No new experiments beyond optional base-vs-story
- No refreshing accepted fellows' LinkedIns
- No emailing Joe unless LW post is strong

---

## Less important, for after LW posts

- [ ] Fix vector selection hierarchy in `utils/vector_selection.py`. Currently `select_vector()` requires steering results and only falls back to unscored vectors if `layer=` and `method=` are passed explicitly. Want a cleaner hierarchy: (1) steering delta (causal, ground truth), (2) val_accuracy + effect_size (cheap, from extraction pipeline), (3) polarity_correct (minimum bar). Should gracefully degrade through the hierarchy instead of requiring steering-or-manual. Also expose naturalness/coherence as separate filters, not buried in `min_naturalness=0` default.

- [ ] Add `ant_emotion_concepts` traits to the convolution detector experiment (v3). Use existing extraction vectors at `experiments/ant_emotion_concepts/extraction/ant_emotion_concepts/{trait}/instruct/vectors/response_50_/residual/{method}/layer{N}.pt`. Initially hardcode L49 globally for all 171 traits (best layer per cross-layer-consistency analysis + Bonferroni-significant zone). Run alongside emotion_set's 173 traits in a combined run, evaluate whether ant_emotion_concepts traits add incremental signal in onset_shift ranking. Likely needs a `--force-layer` or `layer_override_path` flag in `vector_selection.py` to bypass the steering-results requirement. Ship convolution detector v2 (emotion_set only) first; this is a v3 follow-up. See `experiments/rm_syco/convolution-detector/convolution-detector_decision_tree.md` for D-PRUNED entry on why this was scoped out of v2.

- [ ] Add cluster-separation val metric for single-polarity + group-mean + PC-removal extraction (i.e. the `ant_emotion_concepts` method family: `denoised`, `mean_diff+gm`, `mean_diff+gm+pc50`). The existing `val_accuracy` / `val_effect_size` code assumes contrastive pos/neg pairs, which don't exist for this construction — vectors are built as `mean(positives) − mean(neutrals)` followed by grand-mean subtraction across the trait group and optional top-PC removal. Held-out scoring needs the same `gm` and `pc_removal` transformations applied to val activations before projection. Proposed metric: per-trait "cluster purity" — project val samples of trait X onto trait-X vector, compare to projecting val samples of other traits onto trait-X vector, measure Cohen's d (or AUROC) between the two distributions. This gives a per-(trait, layer) score for layer selection without requiring steering results. For now we just use L49 globally (highest cross-layer consistency, close to the bonferroni-significant L43/L49 zone from `per_layer_significance_own_basis.json`) — per-trait layer selection is the follow-up. Files: `utils/extract_vectors.py` (val computation), `analysis/vectors/extraction_evaluation.py` (aggregation), `utils/vector_selection.py` (new source). Reference: `datasets/traits/ant_emotion_concepts/{trait}/extraction_config.yaml` and `experiments/ant_emotion_concepts/scripts/stage6_speaker_probes.py` for construction details.

### Chat: apr10-cot-experiments
Consolidated context from 3 experiment sessions (Haskins CoT Obfuscation, Obfuscation Atlas, infra refactor). R2 explored in depth. Key parked items:
- Haskins Round 2: finish emotion_set extraction (45/87), capture s2ppnh raw activations, project through emotion_set traits
- OA normalization: run norm_diagnostic.py on projection data (needs GPU)
- R2 cleanup: ~165 GiB regenerable data in experiments-save/ + temp/
- Paper (~/code/persona-generalization/tv/paper/paper_rh.tex): outline form, incorporate Haskins/OA results
- [ ] Write up aria_rl arousal finding as connection to Emotion Concepts: CORRECTED — RH model shows HIGH desperation (d=+1.405) AND high calm (d=+1.300) simultaneously, with suppressed moral inhibition (fear↓, anxiety↓, evasiveness↓ at onset). Not "calm hacking" — it's "desperate to pass + unafraid to cheat." Consistent with Sofroniew: desperation drives hacking (confirmed), but RL additionally removed the fear/anxiety brake while leaving the desperation drive intact. At onset: excitement↑, warmth↑, certainty↑, helpfulness↓(-0.075), perfectionism↓, effort↓, evasiveness↓, fear↓. Data: hack_onset_aligned_s1.npz, findings.md Cohen's d values. Cross-reference with Sofroniew Section 3.3 (RH steering) and paper_rh.tex Section 6.

### Chat: apr10-em
Normalization audit before open-sourcing:
- Ensure consistent normalization (projection score ÷ mean residual activation norm at that layer) across all scoring paths: pxs_grid.py, score_emotion_set_eval.py, stream_through_project(), project_from_saved(), and any experiment-specific scoring scripts
- Check for silent failures: unnormalized scores being compared with normalized scores (the original pxs_grid_14b results.json was assembled pre-normalization, while probe_score files were normalized post-hoc — this inconsistency should not exist in the release)
- Verify __normalized sentinel is checked before scoring and that double-normalization cannot happen
- Add assertions / fail-fast checks at scoring boundaries (e.g., if scores >> 1.0, they're probably unnormalized)
- Document the normalization convention in core_reference.md or methodology.md
- [ ] (backlog) Steering detail view redesign — simplify from ground up. Show baseline + best coef per layer (default filter). Toggle to see all responses. Remove confusing colors. Clean layout. Files: visualization/views/steering/detail.js
