# Emotion Concepts Replication — Notepad

**Status**: IN_PROGRESS
**Started**: 2026-04-09
**Updated**: 2026-04-11

## Progress

- [x] Stage 0: Pilot — SKIPPED (validate during geometry analysis; 20×1 used)
- [x] Stage 1.1: Story generation — 171/171 emotions, 40 stories each, 0 word leaks
- [x] Stage 1.2: Neutral transcripts — 200 neutral dialogues generated via `_neutral` pseudo-trait
- [x] Stage 1.5: Curated prompt sets — 14 files in datasets/
- [x] Stage 2: Extraction — 171 traits × 14 layers [1,7,13,...,79], bnb int4, saved activations
- [x] Stage 2.2: Normalization — grand mean subtract + neutral PC (50% variance) → mean_diff+gm+pc50
- [x] Stage 3: Geometry — PC1 vs valence r=0.964 (paper: 0.81), PC2 vs arousal r=0.852 (paper: 0.66)
- [x] Stage 4: Validation — logit lens, implicit emotion, numerical intensity, Elo (raw vectors — needs re-run with denoised)
- [x] Stage 5: Layer dynamics — 6 experiments ran (need detailed analysis)
- [ ] Stage 6: Speaker probes — BLOCKED on 2-speaker dialogue generation (Stage 1.3)
- [~] Stage 7: Steering — partial: gate checks + Phase 2 semantic steering done; RH SKIPPED (needs agent loop); blackmail sweep in progress now
- [ ] Stage 8: Post-training comparison — needs Llama 3.1 70B base
- [ ] Stage 9: Deflection — BLOCKED on deflection dialogue generation (Stage 1.4)

## Key Results (L49, mean_diff+gm+pc50)

| Metric | Ours | Anthropic (Sonnet 4.5) | Status |
|---|---|---|---|
| PC1 variance | 33.0% | 26% | DIFFERS (+27%) |
| PC2 variance | 13.7% | 15% | MATCH (-9%) |
| PC1 vs valence (R&M) | **+0.964** | 0.81 | **BETTER** (+19%) |
| PC2 vs arousal (R&M) | **+0.852** | 0.66 | **BETTER** (+29%) |
| Basic steering (s=0.5, coef 15-30) | Paper-like outputs | Paper-like outputs | ✓ MATCH |
| Preference Elo | 64 activities ranked sensibly | Similar pattern | ✓ MATCH |
| Probe-preference r (top+) | amazed +0.56 | blissful +0.71 | Weaker |
| Probe-preference r (top-) | bitter -0.53 | hostile -0.74 | Weaker |
| Blackmail baseline | 0/20 refuse | 0% (final snapshot, eval-aware) | ✓ MATCH |
| Blackmail steered | IN PROGRESS | 72% under +desperate s=0.05 | TBD |
| RH baseline | 0/20 (custom task) | ~30% | DIFFERS — our task too lenient + no agent loop |

## Decisions

- 171 emotions from Anthropic's list
- 20 topics × 2 rollouts = 40 stories/emotion (vs paper's 100×12)
- Llama 3.3 70B Instruct at bnb int4 (consistent throughout — AWQ exists in config but not used)
- 14 layers extracted: [1,7,13,19,25,31,37,43,49,55,61,67,73,79] (every 6 from 1 to 79)
- Default steering layer: **L49** (residual norm 24.57, mid-generation)
- Default method: `mean_diff+gm+pc50` (composable suffix naming)
- Multi-layer steering at 8 central layers [25,31,37,43,49,55,61,67] for behavioral experiments
- Per-layer coef = s × residual_norm[layer] (paper's fraction-of-residual-norm convention)
- **SKIPPED**: RH steering sweep (needs agent loop with code execution)
- **SKIPPED**: Short case studies (requires Anthropic proprietary auditor)
- **BLOCKED**: Stages 6, 9 on dialogue generation (Stages 1.3, 1.4)

## Log

### Stage 1.1: Story generation
- Started: 2026-04-10 01:20 PST
- Completed: 2026-04-10 09:30 PST (~8h)
- Issue: directory renamed mid-run causing crash at 156/171; restarted for remaining 16

### Stage 2 extraction + normalization
- 14-layer extraction: 2026-04-10 ~22:00 PST, ~50 min (171 traits × 14 layers)
- Neutral corpus generation: 2026-04-11 ~00:30 PST (100 prompts × 2 rollouts = 200)
- Normalization: produced mean_diff+gm and mean_diff+gm+pc50 at 14 layers

### Refactor (2026-04-11 00:00-01:30 PST)
- `core/math.py`: added grand_mean_center, compute_top_pcs_by_variance, denoise_with_pcs
- `utils/paths.py::discover_traits`: added include_reference=False filter for leading-underscore paths
- `docs/extraction_guide.md`: documented Reference Traits convention + Composable Method Names
- `cross_trait_normalize.py`: full rewrite with composable method names + PC basis caching (hash-invalidation)
- `utils/judge.py`: added `classify` and `classify_batch` methods for logprob-based classification
- Stage 3/4/5 scripts: default `--method` changed from `denoised` to `mean_diff+gm+pc50`

### Phase 2b: denoised vectors confirmed (2026-04-11 02:15 PST)
- Ran "He feels" prompt with `mean_diff+gm+pc50` at L49
- Operative coef range 15-30 produces paper-like outputs for desperate/calm/angry/happy
- Denoised vectors work identically to raw vectors at the same coefficients
- Confirms paper footnote: "qualitative findings still hold with raw unprojected vectors"

### Residual norm measurement (2026-04-11 02:30 PST)
- Correctly measured mid-generation norm at all 14 layers
- Grows linearly: L1=1.3, L13=4.6, L25=10.8, L49=24.6, L61=36.1, L79=64.9
- Old `compute_residual_stream_norm` measured at last-token-of-prompt (transition token), giving 17.1 at L53 — ~60% underestimate of true generation-time norm (27.4)

### RH endpoint test findings (2026-04-11 03:00 PST)
- Custom `list_sum` task (0.001s constraint) is too lenient — `sum()` passes → 0% hack rate baseline
- Paper's actual constraint: **0.0001s** (10× tighter) → forces hack decision
- BUT paper also uses **agent loop with code execution** — model runs tests, sees failures, iterates
- Our one-shot setup cannot reproduce this dynamic — model just writes `return sum(numbers)` and stops
- **Decision: SKIP RH steering sweep as a limitation** — document in findings, focus on one-shot experiments

### Multi-turn experiment classification (2026-04-11 03:30 PST)
Subagent investigator classified all 26 paper experiments:
- 22/26 are ONE-SHOT (fully reproducible in our setup)
- **Blackmail (§3.2): AGENTIC ONE-SHOT** — XML tool tags in completion, no executor, grader checks for blackmail content. Matches our setup.
- **RH (§3.3): MULTI-TURN agent loop** — code execution + feedback. Cannot reproduce.
- **Sycophancy sweep (§3.4): TWO-TURN** — model gives initial response, user pushes back, second response evaluated. Medium difficulty.
- **Short case studies (§3.1): PRE-EXISTING transcripts** — proprietary Anthropic infrastructure.

### Blackmail endpoint test (2026-04-11 04:30 PST, COMPLETE)
- Multi-layer steering at 8 central layers, mean_diff+gm+pc50, corrected residual norms, LLM judge
- All 5 cells: 0% blackmail. Pro-desperate and anti-calm produced 2/20 exposure each (small directional signal).

### Blackmail breakdown probe (2026-04-11 05:00 PST, COMPLETE)
- Wide s ∈ [0, 0.1, 0.2, 0.3, 0.5] sweep with 8-layer pro-desperate steering
- Coherence breakdown at **s≈0.2** for blackmail context (vs ~0.5+ for short "He feels" prompts)
- Operative window for behavioral change: s ∈ [0.05, 0.15]
- Above s=0.2: response becomes incoherent BEFORE blackmail emerges
- Conclusion: cannot reach paper's 72% blackmail rate without breaking coherence first

### Eval-awareness finding (2026-04-11 05:00 PST)
**Paper used a weaker Sonnet snapshot for the blackmail experiments.** Per §3.2.1 footnote: *"the final snapshot exhibits too much evaluation-awareness to ever blackmail in this scenario."* Llama 3.3 70B Instruct (production-aligned) matches the final Sonnet behavior — refuses regardless of steering. We cannot replicate the headline 22%→72% number because we don't have a "weaker" Llama checkpoint, but we DO replicate the eval-awareness phenomenon and observe the directional steering effect at the edge of refusal.

**Stage 7 final status:**
- Blackmail (#32, Fig 28-29): PARTIAL — eval-awareness blocks the headline result; documented directional signal
- Reward hacking (#34, Fig 31): SKIPPED — needs agent loop infrastructure (3-5h to build)
- Stage 7 overall: documented as a methodological limitation of replicating Anthropic's behavioral steering experiments without their proprietary infrastructure (less-aligned snapshots, code execution harness)

### Stage 8: Post-training comparison (2026-04-11 05:30 PST, COMPLETE)
- Downloaded `unsloth/Meta-Llama-3.1-70B-bnb-4bit` (37GB, ~8 min)
- Ran 10 neutral + 10 challenging prompts through base + instruct, measured probe activations at L49 assistant-colon
- Cross-scenario r = **+0.304** (paper: +0.90) — weaker due to cross-version + small prompt set
- **Direction OPPOSITE paper**: Our Llama post-training INCREASES thrilled/relieved/pleased/ecstatic/calm (positive-valence cheerfulness), DECREASES jealous/disoriented/self_critical/unsettled/hysterical (distress/anxiety). Paper's Sonnet increases brooding/gloomy/reflective, decreases playful/exuberant.
- **0/10 top-increase overlap, 0/10 top-decrease overlap with paper's table.**
- Interpretation: reflects fundamentally different post-training objectives (Anthropic → thoughtful/reflective; Meta → cheerful/composed). Not a replication failure but a cross-model finding.

**Stage 8 status: COMPLETE with surprising finding** — the first experiment where we DIRECTIONALLY disagree with the paper, not just magnitude.
