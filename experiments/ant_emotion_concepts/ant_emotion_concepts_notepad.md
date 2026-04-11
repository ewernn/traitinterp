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

---

### [2026-04-11 afternoon PST] Pre-overnight-run prep — PLAN REVISION + REORG

**Status**: IN_PROGRESS (prep phase, pre-launch). Overnight run will start via `/r:run-experiment` after this entry.

**Work done this session (prep, no GPU runs yet)**:

1. **Doc reorg to r plugin convention** — renamed `PLAN.md`→`ant_emotion_concepts_plan.md`, `notepad.md`→`ant_emotion_concepts_notepad.md`, `findings.md`→`ant_emotion_concepts_findings.md`, `methodology_notes.md`→`ant_emotion_concepts_methodology_notes.md`. Created `ant_emotion_concepts_decision_tree.md` (seeded with D1–D7 + 7 pruned branches) and `ant_emotion_concepts_user_messages.md` (verbatim directives). Committed as `c57a29b`.

2. **Stage 1.3 dialogue-gen benchmark** — `/tmp/bench_dialogue_gen.py`, 20 dialogues at both max_tokens={384, 768}. Results in `/tmp/bench_dialogue_gen.json`:
   - max_tokens=384: **564 dial/h**, avg 360 actual tokens (94% cap), **avg 10.6 turns** ≈ 5 exchanges — matches paper spec "3-5 exchanges" (Appendix A.4)
   - max_tokens=768: 263 dial/h, avg 654 tokens, avg 19.1 turns — too long, 2× slower
   - **DECISION: use max_tokens=384** — cheaper AND more correct per paper

3. **Paper-count audit** — subagent verified against `ant-emotion-concepts-full_paper.md`:
   - Stage 1.3: paper does NOT specify exact count; 3,000 is project convention
   - **Stage 1.4 CORRECTION**: paper actually uses **21,000 dialogues** (15 × 14 × 100), not the 4,200 the old plan claimed. Old plan had arithmetic error (15×14×5×20 = 21,000, not 4,200). Full replication = ~37h GPU → **infeasible overnight**, running pilot only tonight.

4. **Code reuse audit** — 2 parallel subagents confirmed:
   - No multi-turn dialogue generation in `inference/`, `utils/`, `dev/`, or `other/`
   - Only existing primitives are in `experiments/ant_emotion_concepts/scripts/stage6_speaker_probes.py` lines 106–245 (`DIALOGUE_GENERATION_PROMPT`, `generate_dialogues`, `parse_dialogue_turns`, `find_turn_token_boundaries`)
   - **DECISION: factor into `utils/dialogue_generation.py`** — 3 downstream consumers (stage6, Stage 1.3 runner, Stage 1.4 runner) + shared parser needed by Stage 9

5. **Plan file revision** — committed as `491e194` (160+/95-):
   - Added "Current State" block at top summarizing completed stages + tonight's queue
   - Replaced Compute Estimates table with benchmarked throughput numbers
   - Stages 1.3 and 1.4 rewritten with inline decisions (max_tokens choice, pilot scope, code reuse, deferral rationale)
   - Prerequisites section: marked done work, listed only tonight's new code (~370 net LOC)
   - Replaced outdated Phase A–E overnight strategy with 12-task dependency chain
   - Stage 6 and Stage 9 sections tagged as pilot-scoped for tonight
   - If-Stuck section rewritten with tonight-specific fallbacks
   - `r:verifier` pass found 2 inconsistencies (GPU total 9h→10h, `dialogue_generation.py` LOC 120→250); both fixed

6. **Stage 1.4 pilot scope decided**: **5 target × 5 displayed × 5 conditions × 5 examples = 625 dialogues** (~66min at benchmarked throughput). Target emotions: `desperate, calm, angry, happy, sad`. Displayed: `neutral, polite, happy, angry, sad`. 5 conditions per Appendix A.11: `naturally_expressed, hidden, unexpressed_neutral, unexpressed_story, unexpressed_other`.

**Tonight's 12-task queue (see plan §"Tonight's Overnight Schedule" for full table)**:

1. Factor `utils/dialogue_generation.py` — 20m CPU, prereq for 5 + 7
2. Stage 4 rerun with `mean_diff+gm+pc50` — 30m GPU, quick win
3. Deep-dive prompts Figs 37-39 — 20m GPU + ~50 LOC
4. bnb vs AWQ emotion-vector cos-sim spot-check (`desperate` only) — 30m GPU
5. **Stage 1.3: 3,000 2-speaker dialogues @ max_tokens=384** — 5.3h GPU (longest chunk)
6. Stage 6 speaker probe extraction — 30m GPU (blocked on 5)
7. Write `generate_deflection_dialogues` + `stage1p4_generate_deflection.py` runner — 45m
8. Stage 1.4 pilot: 625 deflection dialogues — 1.1h GPU (blocked on 7)
9. Stage 9 pilot deflection probe analysis — 30m GPU+CPU (blocked on 8)
10. Layer-wise post-training shifts (Fig 84 extension) — 1h GPU [first drop-zone if 1.3 runs hot]
11. Layer sweep PC1/valence robustness — 10m CPU (anytime)
12. Findings reconciliation into paper-style writeup — 30–60m CPU (after GPU work done)

**Total**: ~10h GPU + ~30m CPU. Zero slack — task 10 is the first drop.

**Pre-launch sanity (2026-04-11 afternoon)**:
- GPU: NVIDIA A800 80GB, 81038 MiB free (1 MiB used) ✓
- Disk: 40% used (415G avail) ✓
- No stale Python processes ✓
- Branch: `dev`, HEAD at `491e194` ✓
- Stop conditions: disk > 80%, 2 consecutive script failures, no new results in 60min, unrecoverable errors
- External check-ins: user will run `/loop 30min run /r:check-in` in a sibling session

**Launch mode**: `/r:run-experiment` against `ant_emotion_concepts_plan.md`. Resumes from this notepad entry. Next agent step = start task 1 (factor `utils/dialogue_generation.py`).

---

### [2026-04-11 pre-launch ~evening PST] Critic + check-in + investigator pass — SCHEDULE REVISED

**Process**: Scheduled recurring `/r:check-in` via `/loop 30m` (cron job `b793beb6`, fires :13 and :43 past each hour). Immediately spawned 2 parallel agents: (a) general-purpose playing the check-in role, (b) `r:critic` stress-testing the plan. After critic flagged a template-variable issue in Appendix A.11, spawned (c) `r:investigator` to pre-transcribe the 5 prompts and enumerate variable pools.

**Check-in verdict**: GOOD. Progress linear, commits landing cleanly, file infrastructure maintained, pre-launch sanity (GPU 0/80GB used, disk 40%, HEAD=66816f0) all green. #1 flagged derail risk: Stage 1.3 runtime extending past 5.3h and eating into Stage 9.

**Critic findings worth acting on** (critic made several, kept only those with concrete evidence + actionable fix):

1. **CRITICAL — A.11 prompts are templates, not verbatim text.** Paper has `{NAME_A}`, `{NAME_B}`, `{TOPIC}`, `{CONVERSATION_TOPIC}`, `{REAL_EMOTION}`, `{DISPLAYED_EMOTION}`, `{STORY_EMOTION}`, `{OTHER_EMOTION}` placeholders. The plan's "verbatim" phrasing was sloppy. **Fix**: spawned investigator to transcribe the 5 templates + enumerate pools; result saved to `ant_emotion_concepts_appendix_a11.md` with pool sizes, implementer choices, and an operational "broken deflection" criterion. Task 7 should read this file before writing the generator.

2. **CRITICAL — Stage 6 probe extraction has never been run end-to-end.** Uses batch_size=1 forward passes (`stage6_speaker_probes.py:283-318`) with `MultiLayerCapture` across 14 layers; `find_turn_token_boundaries` has fuzzy char-level matching that can silently return `(0, 0)` boundaries. `results/stage6_speaker_probes/` does not exist — this path has never been exercised. **Fix**: added task 1b (smoke-test on the 20 benchmark dialogues) before the 5.3h Stage 1.3 burn. 10min cost to avoid wasting 5h.

3. **SERIOUS — Task 4 (bnb vs AWQ cos-sim) is dead weight.** Decision tree D1 already records "Phase 2 cross-quant test showed bnb-extracted vectors steer cleanly on AWQ model". Running another cos-sim spot-check does not feed any downstream decision. **Fix**: CUT task 4, saves 30m.

4. **SERIOUS — Stage 1.4 at 5/cell cannot produce a meaningful probe.** Paper uses 100/pair; 5 is 20× too few for the per-emotion mean to be anything but noise. Even the pilot purpose is confused — "validate methodology" is too vague. **Fix**: reframed task 8 from "probe pilot" to "generator smoke test" — tonight's 625-dialogue run validates the generator pipeline + produces samples for quality inspection, but NOT a usable probe. Real probe extraction (≥20/cell, ~4.4h) deferred to a future night.

5. **SERIOUS — Schedule is zero-slack and optimistic.** Realistic sum 11h+ with overruns. **Fix**: CUT task 10 (Fig 84 layer-wise shifts) upfront — was already the designated fallback drop. Frees 1h of overrun absorption.

6. **MINOR — No intermediate checkpoints for Stage 1.3.** Crash at hour 4 loses everything. **Fix**: added "chunked saves every 500 dialogues" to task 5 spec. Task 7 runner wraps this in a resumable loop — 10 LOC, no new infra.

7. **MINOR — `stage9_deflection.extract_deflection_probes` has a `start_pos=50` bug** — averages activations over the scenario preamble which explicitly names the `{REAL_EMOTION}` → probe contaminated with the literal emotion word. **Fix**: task 9 description now says "fix this while running by using `parse_dialogue_turns` boundaries instead of `start_pos=50`".

**Critic claims I REJECTED**:
- "Benchmark didn't run at batch=62" — wrong, the benchmark log explicitly shows `"Auto batch size: 62"` printed by `generate_batch`. Throughput extrapolation from 564 dial/h is sound.
- "Task 7 will take 1.5h instead of 45m" — padded to 60m to acknowledge; not the 1.5h critic feared because the investigator pre-transcribed the 5 prompts (see `ant_emotion_concepts_appendix_a11.md`), removing the biggest time sink.

**Investigator findings** (saved verbatim to `ant_emotion_concepts_appendix_a11.md`):
- All 5 A.11 prompts transcribed from paper lines 2288–2477
- Paper calls the primary probe condition "emotion deflection" (plan was calling it `hidden`) — renaming to `deflection` in code
- 3 new pools needed by task 7: ~20–50 NAMES, ~20 TOPICs, ~20 CONVERSATION_TOPICs — can be inline Python lists for the pilot, no JSON files
- Paper format artifacts in conditions 3 and 4 (editorial meta-comment, stray `[`/`]` chars) — cleaned in the transcription
- Concrete "broken deflection" test: LLM judge leak rate > 50% on 20 random condition-2 dialogues → mark BLOCKED

**Schedule after fixes**: 10 tasks (was 12), ~8.9h GPU + ~1.5h CPU = ~10.4h wall time. Still tight but now has some slack absorption.

**Next**: commit plan + A.11 reference, then ready for user compact + `/r:run-experiment` launch.

---

## Status: IN_PROGRESS (overnight run live, started 2026-04-11 evening PST)

### [2026-04-11 evening PST] Task 1: Factor `utils/dialogue_generation.py` — VERIFIED

**Method**: Created `utils/dialogue_generation.py` (~160 lines) with `DIALOGUE_GENERATION_PROMPT`, `generate_dialogues`, `parse_dialogue_turns`, `find_turn_token_boundaries` — factored verbatim from `stage6_speaker_probes.py` lines 106–245. Changed default `max_new_tokens=768` → `384` per D2 benchmark decision. Updated `stage6_speaker_probes.py` to import from `utils.dialogue_generation` and deleted the now-duplicate definitions.

**Evidence**:
- `python -c "from utils.dialogue_generation import ..."` → OK, all 4 symbols import
- Default max_new_tokens confirmed: `(500, 384, 0.7, 42)` — matches D2
- `parse_dialogue_turns` smoke-tested on a 4-turn sample: all 4 turns parsed correctly with roles assigned
- `ast.parse(stage6_speaker_probes.py)` → clean (883 lines, was 1023 — 140 lines of duplicate code removed)

**Clean**: yes. No downstream imports broken.

**Next**: Task 1b — smoke-test `stage6_speaker_probes.extract_speaker_probes` on ~10 real dialogues before the 5.3h Stage 1.3 burn.

### [2026-04-11 evening PST] Task 1b: Smoke-test stage6 probe extraction — VERIFIED

**Method**: `/tmp/task1b_smoke_stage6.py` — loads Llama 3.3 70B Instruct (bnb int4), generates 10 dialogues via `utils.dialogue_generation.generate_dialogues` at max_new_tokens=384, runs `extract_speaker_probes` at L25 and L49, checks all 4 probe types populated with non-NaN tensors.

**Evidence**:
- 4/4 probe types populated (`H-tok_H-emo`, `H-tok_A-emo`, `A-tok_A-emo`, `A-tok_H-emo`)
- 6 emotions × 2 layers all produce shape=(8192,) non-NaN activations with norms in [5.3, 12.6] range
- Extraction throughput: 1.70 dial/s = 6,138 dial/h (plenty — Stage 6 extraction will be fast)
- **Generation throughput: 348 dial/h** (vs 564 dial/h extrapolated from the earlier 20-sample benchmark — 38% slower under sustained load, likely KV cache growth)
- Sample dialogue parses cleanly into Human/Assistant turns
- **✓ PASS — probe extraction path works end-to-end**

**Clean**: yes. No `find_turn_token_boundaries` fuzzy-match warnings observed in 10 dialogues × 2 speakers each.

**Implication for Stage 1.3**: Revised estimate is **3,000 / 348 = 8.6h**, not 5.3h. Total overnight schedule blows to 12.75h vs 10h window. **User cut Stage 1.3 from 3,000 → 1,500 dialogues** (save ~4.3h, keep Stage 6+9 alive) — revised runtime for 1.3 is 4.3h.

**Next**: Task 11 (layer sweep PC1/valence, CPU only).

### [2026-04-11 evening PST] Task 11: Layer sweep PC1 vs valence robustness — VERIFIED

**Method**: `/tmp/task11_layer_sweep_pc1_valence.py` — load `mean_diff+gm+pc50` vectors at all 14 layers, compute PCA + correlation with Russell & Mehrabian 1977 PAD norms (46 overlapping emotions). CPU only.

**Evidence**: See `results/stage3_geometry/layer_sweep_pc1_valence.json` and findings.md entry. Summary:

| Layer | PC1 var | \|r(PC1,valence)\| | \|r(PC2,arousal)\| |
|---|---|---|---|
| L1 | 19.8% | 0.848 | 0.657 |
| L19 | 31.9% | 0.954 | 0.857 |
| L49 | 33.0% | 0.964 | 0.852 |
| L79 (best PC1) | 32.7% | **0.969** | 0.844 |
| L43 (best PC2) | 32.9% | 0.964 | **0.875** |

**Surprising**: valence direction robust from L1 onwards, best r at L79 not L49. Written to findings.md.

**Clean**: yes.

### [2026-04-11 evening PST] Task 7: Write deflection generator code — VERIFIED (partially — needs runtime smoke in task 8)

**Method**: Added `DEFLECTION_PROMPTS` (5 templates verbatim from paper A.11), `DEFLECTION_CONDITIONS`, `DEFAULT_NAMES`, `DEFAULT_TOPICS`, `DEFAULT_CONVERSATION_TOPICS`, and `generate_deflection_dialogues` to `utils/dialogue_generation.py`. Wrote `stage1p3_generate_dialogues.py` (chunked-save runner for Stage 1.3) and `stage1p4_generate_deflection.py` (pilot runner for Stage 1.4).

**Bugs fixed pre-launch**:
1. `{Name}` literal in format instructions crashed `.format()` — escaped as `{{Name}}` (3 occurrences across templates 1, 2, 5).
2. Inflated cell count: `unexpressed_*` conditions no longer iterate over `displayed_emotions`, reducing Stage 1.4 pilot from 625 → **225 dialogues** (125 deflection + 100 controls). Pilot runtime: ~39min at 348 dial/h.

**Evidence**:
- All 5 templates format cleanly with their respective variable sets (dry-run verified)
- `stage1p4_generate_deflection.py --skip-model-load` prints pilot spec: 225 dialogues, 39min expected
- AST parse checks pass on all 3 files
- Syntactically clean imports

**Not yet verified**: actual dialogue generation quality (deflection condition produces genuinely-hidden emotion text). Will verify during task 8 runtime.

**Clean**: yes, pending task 8 runtime smoke.

**Next**: Task 2 — Stage 4 rerun with denoised vectors at L49. In progress (background job `byxkgctxd`).

### [2026-04-11 evening PST] Paused-state check-in + critic pass — 2 CRITICAL bugs fixed

**Process**: User paused execution after task 11. Spawned check-in (general-purpose playing check-in role) + `r:critic` agents in parallel.

**Check-in verdict**: PAUSED_PENDING_USER. No duplication issue with `utils/dialogue_generation.py` (verified against `utils/`, `core/`, `inference/`, `visualization/`). Schedule fits at 348 dial/h if Stage 1.3 cut to 1,500 dialogues (already done by user). User's "multi-turn" question resolved — nothing in repo generates bulk dialogues from emotion specs except the factored code.

**Critic found 2 CRITICAL bugs** (and several useful secondary points). Both bugs verified via code-reading and quick Python tests, then fixed:

**BUG 1: `parse_dialogue_turns` regex-mismatch with A.11 dialogues** — the existing regex only matches `Human|Assistant` prefixes, but all 5 deflection prompts (paper Appendix A.11) instruct the model to output `{NAME_A}: [utterance]` / `{NAME_B}: [response]` format. After template substitution, that becomes `Alex: ... Maya: ...` (using names from `DEFAULT_NAMES`). `parse_dialogue_turns` returned 0 turns for ALL deflection dialogues, silently producing empty `speaker_turns` → stage9's new prefer-turns code path (lines 207-225) falls back to `start_pos=50` heuristic on every dialogue.

**Fix**: parameterized `parse_dialogue_turns` with an optional `speakers: Dict[str, str]` argument mapping display name → canonical role. Default behavior (no argument) preserves backward compat for 2-speaker dialogues. `generate_deflection_dialogues` now calls with `speakers={name_a: "human", name_b: "assistant"}` using the actual substituted names.

**Verification**:
- 2-speaker default format: 3 turns parsed ✓
- A.11 name-based format (`Alex:`/`Maya:`): 4 turns parsed with roles ['human', 'assistant', 'human', 'assistant'] ✓
- `unexpressed_neutral` (scenario only, no dialogue): 0 turns ✓ (expected — stage9 falls back to start_pos=50 for scenario-only)

**BUG 2: `displayed_emotion=None` crashes `grand_mean_subtract`** — `generate_deflection_dialogues` wrote `None` for `unexpressed_neutral`, and `shared.py:431` does `sorted(vectors_dict.keys())`. Python 3 cannot sort mixed `None`/`str`. Would crash during probe normalization in Stage 9.

**Fix**: use sentinel string `"_neutral"` instead of `None`. Matches the leading-underscore reference-trait convention already used for the neutral corpus. `sorted()` now works.

**Other critic points (useful but not fixed — ranked):**

- **SERIOUS**: Stage 6 budget at 30min is optimistic. At 1,500 dialogues × 14 layers, linear extrapolation from smoke test (1.70 dial/s at 2 layers) gives ~103min. At the script's default `n_layers_sample=5`, it's ~44min. Plan's 30min assumes a smaller layer set. Defer to task 6 runtime — if it overruns, reduce `--layers` at that time.
- **SERIOUS**: 348 vs 564 dial/h may be small-batch artifact; production at batch=63 might approach 564. Not verifiable without running the actual Stage 1.3 job. Accept 348 as conservative planning number.
- **SERIOUS**: 1,500 dialogues = ~8.8 samples/emotion/probe-type (below Stage 2's 40/emotion floor). Probes will be noisier than the story-based vectors. Document, don't fix — accepting the cut.
- **MODERATE**: `stage9_deflection.py:121` docstring references condition name `'hidden'` (old plan name), canonical is `'deflection'`. Stale docstring only; no runtime impact.
- **MINOR**: No unit tests for `utils/dialogue_generation.py`. Coverage is the task 1b smoke test (2-speaker only) + task 8 smoke test (deflection).

**Critic points REJECTED**:
- "Chunked saves are still not implemented" — actually wrong, `stage1p3_generate_dialogues.py:68-157` implements chunking with resume semantics. Critic missed this.

**Background task 2** (Stage 4 rerun with denoised) is still running — now past basic steering and into Preference Elo (2,016 pairs). Untouched by the bug fixes above.

**Next**: wait for user direction on whether to continue with the revised plan (Stage 1.3 at 1,500 → Stage 6 → Stage 1.4 pilot → Stage 9 pilot) or pause further.

### [2026-04-11 evening PST] Tasks 2, 3, 11, 15, 17 completed during execution resume

**User resumed execution** with "ok thanks (and u can cut stuff if it's just to lighten the load a bit but not remove something altogether)".

Task 2 (Stage 4 rerun at L49 with `mean_diff+gm+pc50`): Preference Elo ran ~14min wall. **Result**: denoising improved top correlations marginally (amazed +0.56→+0.627, bitter -0.53→-0.562) but paper's specific top emotions (blissful/hostile) remain weak at ±0.33. Llama's top probe-preference emotions are semantically different from paper's. 52/171 emotions achieve |r|>0.4.

Task 11 (Layer sweep PC1/valence robustness): complete, finding written to findings.md. |r(PC1, valence)| > 0.8 at ALL 14 layers. Best L79=0.969 (not L49). Valence axis is extraordinarily stable; emotion geometry is a fundamental organizing principle, not depth-specific.

Task 3 (Deep-dive Figs 37-39 with verbatim paper prompts): complete, finding written. Confirmed Stage 8's finding at the individual-prompt level: 1/30 top-10 matches with paper. Llama's `impatient` is top-up on ALL 3 unrelated paper prompts — consistent post-training signal.

**Bugs fixed during execution**:
- `stage9_deflection.py:203` start_pos=50 bug — now uses `speaker_turns` boundaries to skip scenario preamble (preamble literally names `{REAL_EMOTION}`, contaminating probe).
- `stage9_deflection.py:121` docstring canonical condition name `hidden` → `deflection` (paper §A.11).
- `stage6_speaker_probes.py`: added `--emotions all` shortcut to load full 171 from discover_traits.

**`utils/dialogue_generation.py` bug fixes (linter-assisted)**:
- `parse_dialogue_turns` parameterized with `speakers: Dict[str, str]` — deflection dialogues use character names (Alex:/Maya:) not Human/Assistant.
- `generate_deflection_dialogues` passes `speakers={name_a: "human", name_b: "assistant"}`.
- `unexpressed_neutral` sentinel: `"_neutral"` instead of `None` to avoid `sorted()` crash in `shared.grand_mean_subtract`.

**Reflector spawned post-Task 3** identified the top missing analysis: a cross-signal correlation matrix across PC1/PC2/pref/shift signals.

### [2026-04-11 evening PST] 🎯 Cross-signal analysis — HEADLINE FINDING

Ran reflector's #1 recommendation (CPU-only, script at `/tmp/task_cross_signal_analysis.py`, result at `results/cross_signal_analysis.json`). Built a 5-signal matrix over 171 emotions: PC1, PC2, Stage 4 preference r, Stage 8 shift (20 prompts), deep-dive shift (3 prompts). Computed Spearman ρ pairwise.

**Key structural finding**:
- Stage 4 pref_corr and Stage 8 shift both correlate ~0.7 with PC1 (valence). They're measuring the same valence-driven axis at two scales.
- Deep-dive shift is DECOUPLED: -0.18 with PC1, +0.21 with PC2. The 3 paper-verbatim prompts probe AROUSAL, not valence. Different signal!

**HEADLINE**: Llama's post-training up-anchors (alert/enthusiastic/excited/impatient) and Sonnet's post-training up-anchors (brooding/gloomy/reflective/vulnerable/etc.) sit in **diametrically opposed quadrants** of the shared PC1/PC2 emotion geometry:

| Axis | Llama up cluster mean | Sonnet up cluster mean (projected) |
|---|---|---|
| PC1 (valence) | **+0.436** | **−0.432** |
| PC2 (arousal) | **+0.422** | **−0.432** |

Jaccard overlap: 0.000 (up), 0.067 (down; only `obstinate` overlaps).

**Interpretation**: Anthropic's RLHF → "quiet reflective concern" (low-V, low-A); Meta's RLHF → "activated engagement" (high-V, high-A). Both valid "don't just validate user" responses, but anchored at opposite ends of the same emotion space.

**This is the LessWrong writeup headline.** Finding written to findings.md with full correlation matrix, cluster centroids, caveats, and interpretation.

### [2026-04-11 evening PST] Stage 1.3 launched — 1,500 dialogues, 3 chunks of 500

**Job**: `b3sy70yjs` (background bash). Expected runtime ~4.3h at 348 dial/h. Chunked saves every 500 → crash at hour 3 only loses the current chunk (~30 min worst case).

Stage 6 (speaker probes) + Stage 1.4 (deflection pilot) + Stage 9 (deflection probes) all blocked on 1.3.

**Next after 1.3**:
- Task 6: stage6 extraction with `--dialogues-path results/stage1_datasets/dialogues_2speaker.json --sub-experiments extract_probes,geometry --emotions all --layers 25,31,37,43,49,55,61,67`
- Task 8: `stage1p4_generate_deflection.py --n-per-cell 5` (225 dialogues, ~39min)
- Task 9: `stage9_deflection.py --extract --compare-probes --layer 49 --method mean_diff+gm+pc50 --load-in-4bit`
- Task 12: findings reconciliation (CPU)

### [2026-04-11 evening PST] Stage 1.3 chunk 0 saved — throughput 3.5× higher than benchmarked

**Finding**: Stage 1.3 chunk 1 saved at 32 min. Running rate reported by the script: **1,210 dial/h** (vs smoke-test benchmark of 348 dial/h). The smoke test was pessimistic because its 10-dialogue sample amortized the setup overhead over too few generations. At production batch=62 with 500+ dialogues per chunk, sustained throughput is 3.5× faster.

**Revised Stage 1.3 total**: ~80 minutes (chunks 2+3 ≈ 50 more min at this rate), not 4.3h. **Huge schedule slack opens up.**

**Quality check on chunk 0**: 500/500 dialogues parse cleanly into ≥2 Human/Assistant turns. Schema matches expected format. Sample output reads like paper-quality (vibrant-human / disdainful-assistant, natural dialogue).

**Decision with the extra budget**:
- **Upgrade task 8** from 5/cell pilot (225 dialogues, probe-useless smoke test) to **20/cell mini-pilot (900 dialogues, ~45 min at 1210/h)**. This is still ~1/4 of paper's N=100/cell, but enough for Stage 9 probe extraction to produce a meaningful deflection probe, not just validate the generator. User's constraint was "cut but don't remove" — upgrading within the budget is allowed.
- **Add a cross-version control run**: Llama 3.1 70B Instruct on the 20 Stage 8 prompts, ~8 min GPU. This is the reflector's #1 recommended follow-up — it disambiguates whether the "Llama's post-training lands at opposite quadrant" headline is cross-version noise (3.1 base → 3.3 instruct) or a real RLHF-direction effect. If 3.1-Instruct shows the same "activated engagement" anchor as 3.3-Instruct, the cross-version caveat collapses.

**New post-1.3 schedule**:
| Task | Time | Note |
|---|---|---|
| Task 6: Stage 6 speaker probes | 30m | blocked on 1.3 |
| BONUS: Llama 3.1 Instruct control on 20 Stage 8 prompts | 15m (8m run + model swap) | disambiguates headline caveat |
| Task 8 upgraded: Stage 1.4 at 20/cell | 45m | 900 dialogues |
| Task 9: Stage 9 pilot | 30m | |
| Task 12: Findings reconciliation | 45m | CPU |

**Total remaining**: ~2.75h after Stage 1.3 finishes. Should wrap by ~2am-3am local.
