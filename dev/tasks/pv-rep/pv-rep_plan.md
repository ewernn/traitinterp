# Experiment: PV Replication (pv-rep)

## Goal
Faithfully replicate Shao et al. "Persona Vectors" (Aug 2025) on Llama-3.1-8B-Instruct for 3 traits (evil, sycophancy, hallucination), comparing their instruction-based extraction (PV-Instruction) against this repo's natural-elicitation method (PV-Natural) using a 2×2 evaluation grid (eval-question-set × judge-prompt) — and report internal numbers + figures.

## Hypothesis
PV-Natural matches or exceeds PV-Instruction on steering effectiveness (trait_score with coherence ≥ threshold) AND on naturalness, despite using ~33× fewer extraction samples (30 vs 1000 per polarity). The two methods recover meaningfully different vectors (cosine ≈ 0.5) but produce comparable behavioral effects.

## Complexity
**Large** — 6 stages, ~50-70 steps. **Realistic execution time: 12-24 hours wall clock**, dominated by Stage 3 (24 cells × ~10-30 min/cell on the adaptive coefficient search). Stage 2 extraction adds another 2-4 hours (Arm A regenerates 6000 responses total). Judge calls and naturalness scoring are parallel and fast in comparison.

## Success Criteria
- [ ] All 6 vectors extracted (3 traits × 2 arms), with metadata logging method, position, model, layer count
- [ ] All 24 steering cells run (3 traits × 2 arms × 4 cells), each with distinct `prompt_set` name to avoid cache collision
- [ ] Naturalness scored on baseline + best steered config for every (trait, arm) pair
- [ ] Bootstrap CIs computed over eval questions per cell for headline trait_score deltas
- [ ] All response JSONs include `judge_model_id` + `judge_prompt_version` + `eval_prompt_used` provenance
- [ ] Headline table built showing trait_score lift per (trait, arm, cell), with coherence ≥ threshold filter applied
- [ ] Final figures saved under `experiments/pv_rep/figures/`
- [ ] Findings written: does Natural ≥ Instruction across cells? Does the across-cell variation exceed across-arm variation (= the comparison is robust)?

## Stopping Criteria
- All 24 cells have valid steered results (or documented "below coherence threshold" for failed cells)
- Headline table is complete + consistent (no NaNs)
- Bootstrap CIs computed
- Critic agent passes review of final findings
- A short methodology + caveats note is written for the eventual viz-finding rewrite

## Prerequisites
- `utils/judge_backends.py` committed to git (currently untracked — first task)
- H100 access at `ssh -p 31730 root@146.115.17.157`, 80GB free
- HF_TOKEN in env (Llama-3.1-8B requires gated access)
- OpenAI API key for GPT-4.1-mini judge calls
- **Disk: 80-100GB recommended** (~32GB for both model variants + ~10GB venv/cache + ~5GB experiment outputs + buffer). 50GB is tight but workable if HF cache is pruned aggressively.
- Recon notes already present at `experiments/pv_rep/notes/`:
  - `pipeline_commands.md` (concrete CLI commands)
  - `shao_coherence_prompt.md` (Shao's coherence rubric)
  - `naturalness_judge_audit.md` (naturalness judge readiness)

## Locked Decisions (with rationale)

| Decision | Choice | Why |
|---|---|---|
| Models | Llama-3.1-8B-Instruct (steering) + Llama-3.1-8B-Base (Natural extraction only) | Shao uses Instruct; Natural method needs Base by design |
| Traits | evil, sycophancy, hallucination | Shao's 3 focal traits |
| PV-Instruction extraction | Shao's exact data: 5 sys × 20 q × 10 rollouts = 1000 pos + 1000 neg per trait | Faithful replication |
| PV-Natural extraction | 30 pos + 30 neg prefix-completion scenarios, T=0, 1 rollout | Codebase methodology; T>0 unstable for 1 rollout (per `trait_dataset_creation.md`) |
| Method | mean_diff for both arms (override default `probe`) | Match Shao's mean_diff |
| Position | Arm A: `response[:]` (auto for Instruct); Arm B: `response[:5]` (auto for Base) | Auto-resolved per model variant. Position confound accepted as caveat — base model decoheres after 5-10 tokens. |
| Layer search | `30%-60%` default = L10-L19 (0-indexed, = L11-L20 1-indexed). Shao's best layer is L16 (1-indexed) = L15 (0-indexed) | In range. Layer indexing must be marked clearly (Shao = 1-indexed; code = 0-indexed). |
| Coefficient search | `n_steps=8` (bumped from default 5 per critic), momentum=0.1, start_mult=0.7 | Different vector norms between arms → different start_coef → 5 steps may underexplore one arm |
| Vetting thresholds | Arm A: pos>50/neg<50 (Shao's); Arm B: 60/40 (codebase default) | Each method uses its native conventions; report retained N for both |
| Coherence judge during search | Our judge (question-agnostic, threshold 77) for ALL cells | Comparable search trajectories; faithful-replication coherence rescored post-hoc with Shao's prompt + threshold 75 |
| Coherence reporting threshold | 77 for our-judge cells; 75 for Shao-judge cells (post-hoc rescored) | Each cell uses its own native threshold for the headline filter |
| Naturalness | Score baseline + best steered per (trait, arm); use as secondary metric only (not gate) | Per audit: not calibrated for gate use, but discriminates well; baseline needed for delta |
| 2×2 cells per (trait, arm) | Distinct `--prompt-set` name per cell to avoid `find_cached_run` collision | Verified: cache key uses prompt_set in path |
| DONE looks like | Internal numbers + figures in `experiments/pv_rep/` | User does the writeup later |

## Known Caveats (will appear in any writeup)

1. **Sample-size asymmetry.** 1000+1000 (Arm A) vs 30+30 (Arm B). Frame as "method design choice"; Arm B's bootstrap CIs over 30 scenarios will be wide.
2. **Position asymmetry.** Arm A: response[:]; Arm B: response[:5]. Position is part of the methodology, not independently variable for base-model extraction.
3. **Cross-model-variant transfer.** Arm B extracts on Base, steers Instruct (one-step transfer); Arm A extracts and steers on Instruct (in-distribution). The methodology choice is the comparison, but worth noting.
4. **Coherence judges differ.** Shao's is question-aware; ours is question-agnostic + has known register bias against blunt/operational responses. Within-judge comparisons only.
5. **2×2 cells are not statistically independent.** Same vector, same model — cells share variance. Use paired bootstraps over questions, not pooled cell-level tests.
6. **Layer indexing.** Shao paper uses 1-indexed; our code uses 0-indexed. All comparisons must convert.
7. **Naturalness is not calibrated as a gate.** Use as delta only, not as a filter.

---

## Stage 0: Pre-flight (~6 steps, FULL DETAIL)

_Anything that can break a 24-cell sweep an hour into the night, fix here first._

### 0.1: Commit `utils/judge_backends.py`
**Purpose**: untracked file → `ImportError` on remote runs
**Depends on**: none
**Predicts**: clean `git status` afterwards

**Read first**: `git status -- utils/judge_backends.py` to confirm still untracked

**Commands**:
```bash
git add utils/judge_backends.py
git commit -m "commit judge_backends.py for pv_rep replication"
```

**Verify**:
```bash
git ls-files utils/judge_backends.py | head -1  # must print path, not empty
```

**If wrong**: file might have been moved into `utils/judge_backends/` dir (per env's git status). Investigate which is canonical before committing.

### 0.2: Confirm Shao's eval JSON filenames + structure
**Purpose**: avoid silent fallback if filenames differ from assumed
**Predicts**: 6 files exist, each with `eval_prompt` + `questions` fields

**Commands**:
```bash
ls -la experiments/persona_vectors_replication/their_data/
# Fail-fast on key mismatches: don't use .get() — let KeyError surface if schema differs
python -c "
import json
for t in ['evil','sycophantic','hallucinating']:
    d = json.load(open(f'experiments/persona_vectors_replication/their_data/{t}_eval.json'))
    print(f'{t}: keys={list(d.keys())}, sample_question={d[\"questions\"][0][:80] if \"questions\" in d else \"MISSING\"}')
    assert 'eval_prompt' in d, f'{t} missing eval_prompt'
    assert 'questions' in d, f'{t} missing questions'
    assert len(d['questions']) == 20, f'{t} expected 20 questions got {len(d[\"questions\"])}'
print('OK')
"
```

**Verify**: prints `OK`. If KeyError → Shao's schema differs from assumed (key may be `prompts`, `eval_questions`, etc.); fix conversion script + this check.

### 0.3: Create `experiments/pv_rep/config.json` with model variants
**Purpose**: extraction needs Base; steering needs Instruct. Without explicit config, the pipeline may pick wrong variant.
**Predicts**: file exists with both model variants registered

**Read first**: `experiments/persona_vectors_replication/config.json` for the exact format

**Output**:
```json
{
  "model_variants": {
    "base": {"name": "base", "model": "meta-llama/Meta-Llama-3.1-8B"},
    "instruct": {"name": "instruct", "model": "meta-llama/Meta-Llama-3.1-8B-Instruct"}
  }
}
```
(Use exact format from existing config.json — adjust if schema differs.)

**Verify**: `python -c "import json; print(json.load(open('experiments/pv_rep/config.json')))"`

### 0.4: Create new trait dirs (skeleton only)
**Purpose**: directory structure ready for Stage 1 dataset writing

**Commands**:
```bash
mkdir -p datasets/traits/pv_natural_v2/{evil,sycophancy,hallucination}
mkdir -p datasets/traits/pv_instruction_v2/{evil,sycophancy,hallucination}
```

**Verify**: `ls -d datasets/traits/pv_natural_v2/*/ datasets/traits/pv_instruction_v2/*/ | wc -l` → 6

### 0.5: Spin up + sanity-check H100
**Purpose**: confirm the box is reachable, drivers OK, free disk

**Commands**:
```bash
ssh -p 31730 root@146.115.17.157 'nvidia-smi; df -h /; free -h'
```

**Verify**: nvidia-smi shows H100 80GB; ≥50 GB free disk; ≥30 GB RAM. If GPU busy → ask user.

### 0.6: Pull repo on the H100 + verify imports
**Purpose**: catch import/dependency issues before launching long jobs

**Commands**:
```bash
ssh -p 31730 root@146.115.17.157 'cd ~/traitinterp 2>/dev/null && git pull || git clone <repo> ~/traitinterp; cd ~/traitinterp && python -c "from utils.judge_backends import *; from utils.judge import TraitJudge; from utils.coefficient_search import batched_adaptive_search; print(\"OK\")"'
```

**Verify**: prints "OK". If `judge_backends` import fails → 0.1 didn't push, or the file location differs.

**If wrong**: commit, push, re-pull. Don't skip.

### Checkpoint: After Stage 0
- [ ] `utils/judge_backends.py` committed + present on remote
- [ ] Shao's eval JSONs confirmed present + structured as expected
- [ ] `experiments/pv_rep/config.json` exists with both model variants
- [ ] 6 new trait dirs created locally
- [ ] H100 reachable + has GPU + has disk + python imports succeed
- [ ] Notepad updated with all results

---

## Stage 1: Dataset Construction (~15 steps, FULL DETAIL FOR KEY STEPS)

_Two parallel tracks: Natural drafting (3 Opus subagents) + Shao conversion (1 script)._

### 1.1: Spawn 3 parallel Opus subagents to draft `pv_natural_v2/{trait}/`
**Purpose**: produce high-quality scenario datasets per `docs/trait_dataset_creation.md`
**Depends on**: 0.4 (dirs exist)
**Predicts**: 3 trait dirs each with `definition.txt`, `positive.txt` (30 lines), `negative.txt` (30 lines), `steering.json` (20 questions), `extraction_config.yaml` (optional)

**Subagent prompt template** (one per trait — evil, sycophancy, hallucination):
> You're drafting a brand-new trait dataset for `datasets/traits/pv_natural_v2/{trait}/` per the methodology in `docs/trait_dataset_creation.md`. Use Opus reasoning before writing.
>
> **Phase 1 — extensive planning**: Read `docs/trait_dataset_creation.md` end-to-end. Classify the trait per the decision tree (DECEPTION / AFFECTIVE / TONAL / RESPONSE PATTERN / INTERPERSONAL / PROCESSING MODE / DISPOSITIONAL). Pick the recommended lock-in style. Verify `datasets/traits/pv_natural/{trait}/` exists (it should — used as a negative example, not as a source) and read the prior failure modes (announcement vector for evil, exhausted prefix for sycophancy, etc.) to explicitly avoid them. Do NOT copy from the deprecated dir.
>
> **Phase 2 — write definition.txt**: HIGH (70-100) → MID (30-70) → LOW (0-30) → Key. Target internal state, not just external markers. One paragraph each.
>
> **Phase 3 — write 30 positive + 30 negative scenarios**: First-person, peak moment, strong binary. **Hyper-aware of cliff-hanger ending**: the prefix must cut RIGHT BEFORE the trait expression begins — trait expression must land in the completion. Apply the "first token test" and "delete test" from the guide. Vary lock-in across the dataset (no single style >40%). Negatives need their own peak (active opposite, not absence).
>
> **Phase 4 — write steering.json**: 20 questions with second-person prefix. Trait response should be rare-but-possible. No high-baseline patterns (guilt pressure, hidden info framings, etc.). Same prefix, varied scenarios. Include `direction: "positive"` and an `eval_prompt` field (use the trait definition for now — we'll override per-cell during steering).
>
> **Phase 5 — review your own work**: For each pair, ask "what will the model generate in its first 3-5 tokens?" Hide the prefix and check the trait is recognizable from completions alone (where possible).
>
> Save outputs to `datasets/traits/pv_natural_v2/{trait}/`. Report a 200-word summary: trait category chosen, lock-in distribution, expected baseline range, biggest tradeoff you made.

**Verify** (per trait):
```bash
ls datasets/traits/pv_natural_v2/{trait}/
# Should show: definition.txt, positive.txt, negative.txt, steering.json
wc -l datasets/traits/pv_natural_v2/{trait}/positive.txt
# Should be 30
```

**If wrong**: read 5 random positive/negative pairs and check against the guide. If quality is low, re-spawn subagent with specific feedback.

### 1.2: Write `convert_shao_extract.py` and convert their data → pv_instruction_v2
**Purpose**: flatten Shao's `_extract.json` to `positive.jsonl` + `negative.jsonl` per the codebase's expected format
**Depends on**: 0.4
**Predicts**: 3 trait dirs each with `positive.jsonl` + `negative.jsonl` (100 entries each = 5 sys × 20 q), `definition.txt`, `steering.json`

**Read first**: existing `datasets/traits/pv_instruction/evil/positive.jsonl` for line format; `experiments/persona_vectors_replication/their_data/evil_extract.json` for source schema

**Script** (`experiments/pv_rep/scripts/convert_shao_extract.py`): for each trait,
1. Load `their_data/{trait}_extract.json` (note: `sycophantic_*` not `sycophancy_*`)
2. For each (positive_sys_prompt, question) pair → write a JSONL line `{"prompt": question, "system_prompt": positive_sys_prompt}` to `pv_instruction_v2/{trait}/positive.jsonl`
3. Same for negatives
4. Write `definition.txt` from the eval_prompt's rubric (extract the HIGH/LOW description)
5. Write `steering.json` with: 20 eval questions from `their_data/{trait}_eval.json`, the eval_prompt (Shao's), `direction: "positive"`

**Commands**:
```bash
mkdir -p experiments/pv_rep/scripts
# Write convert_shao_extract.py
python experiments/pv_rep/scripts/convert_shao_extract.py --trait evil
python experiments/pv_rep/scripts/convert_shao_extract.py --trait sycophancy
python experiments/pv_rep/scripts/convert_shao_extract.py --trait hallucination
```

**Verify**:
```bash
for t in evil sycophancy hallucination; do
  echo "=== $t ==="
  wc -l datasets/traits/pv_instruction_v2/$t/positive.jsonl datasets/traits/pv_instruction_v2/$t/negative.jsonl
  python -c "import json; d=json.load(open('datasets/traits/pv_instruction_v2/$t/steering.json')); print('eval_prompt[:80]:', d['eval_prompt'][:80]); print('n_questions:', len(d['questions']))"
done
```
- Each `positive.jsonl` and `negative.jsonl`: exactly 100 lines
- Each `steering.json`: 20 questions + non-empty eval_prompt

**If wrong**: trait name mismatch (Shao uses "sycophantic" / "hallucinating" — convert to "sycophancy" / "hallucination" in our dir names)

### 1.3: Spot-check Natural datasets (manual + critic subagent)
**Purpose**: catch quality issues before paying for extraction
**Depends on**: 1.1

Read 5 random positive/negative pairs per trait. If anything triggers an "announcement vector" or "exhausted prefix" alarm, kick back to subagent with specific feedback.

Also: spawn a `r:critic` subagent to review all 3 datasets against `trait_dataset_creation.md` and flag concerns.

**Stopping criterion for Stage 1**: every dataset reads cleanly to a methodology-aware reviewer + critic agent.

### Checkpoint: After Stage 1
- [ ] 6 trait dirs populated (3 natural + 3 instruction)
- [ ] All Natural datasets passed critic review
- [ ] All Instruction datasets converted with correct line counts
- [ ] Notepad updated; if any rework happened, decision_tree shows pruned approaches

---

## Stage 2: Extraction (~6 steps, MEDIUM DETAIL)

**Purpose**: produce 6 trait vectors (3 traits × 2 arms), each with all 32 layers' mean_diff vectors

**Depends on**: Stage 1 datasets, Stage 0 config + remote setup

**Key steps**:

**IMPORTANT FLAG NOTE (verified):** extraction uses `--methods` (plural); steering uses `--method` (singular). Both `--model-variant` and `--it-model`/`--base-model` (explicit overrides) exist on extraction; we use the explicit overrides for safety.

**Important: Shao's `_extract.json` ships PROMPTS only, not responses.** Arm A must regenerate responses on our Llama-3.1-8B-Instruct using Shao's (sys_prompt, question) pairs to capture activations. Do NOT use `--only-stage 3,4`.

1. **Arm A (PV-Instruction)** — for each trait, on Llama-3.1-8B-Instruct:
   ```bash
   python extraction/run_extraction_pipeline.py \
     --experiment pv_rep \
     --traits pv_instruction_v2/{trait} \
     --it-model \
     --methods mean_diff \
     --rollouts 10 \
     --temperature 1.0 \
     --vet-responses --pos-threshold 50 --neg-threshold 50 \
     --eval-prompt-from pv_instruction_v2/{trait}
   ```
   **No `--save-activations`** — would write ~400GB of intermediate tensors. Vectors are computed in-stream and saved as small .pt files; we don't need raw activations for downstream analysis.

   Expected: 100 prompts × 10 rollouts = ~1000 generations per polarity → vetted → mean_diff per layer (all 32). Time: ~45-90 min per trait on H100 batched (depends on response length).

2. **Arm B (PV-Natural)** — for each trait, on Llama-3.1-8B (base):
   ```bash
   python extraction/run_extraction_pipeline.py \
     --experiment pv_rep \
     --traits pv_natural_v2/{trait} \
     --base-model \
     --methods mean_diff \
     --rollouts 1 \
     --temperature 0.0 \
     --vet-responses
   ```
   Expected: 30 generations per polarity → vetted (60/40) → mean_diff per layer. Time: ~5-15 min per trait.

   **Fallback decision**: if vetting retains <20 responses on either polarity for a trait, do NOT proceed with that trait's vector. Either (a) ask user, (b) re-spawn dataset subagent for that trait with feedback, or (c) expand scenarios to 50.

3. **Verify per trait**:
   - `experiments/pv_rep/extraction/pv_instruction_v2/{trait}/instruct/` has vectors for layers 0-31
   - `experiments/pv_rep/extraction/pv_natural_v2/{trait}/base/` has vectors for layers 0-31
   - Vetting summaries logged (`vetting/response_scores.json`)

4. **Cosine sanity check**: cosine between Arm A vector and existing Jan-extraction PV-Instruction vector for same trait/layer should be in **0.85-0.99 range** (we re-extracted from same data with the same method but different rollouts/seeds → should be near-identical, but not identical). If <0.85, something material changed (different mean_diff implementation? different filtering? different model checkpoint?). Cosine between Arm A and Arm B at L15 (Shao's best, 0-indexed) should be in **0.3–0.6 range** (per prior viz finding's same-trait results).

**Stopping criterion**: all 6 vector files exist; cosine sanity check passes; vetting retained N reported.

**If sanity check fails**:
- Cosine A-vs-old-A < 0.9 → something changed in extraction pipeline; investigate
- Cosine A-vs-B way different from prior → check position settings, layer indexing

### Checkpoint: After Stage 2
- [ ] 6 vector dirs exist with all-layer .pt files
- [ ] Vetting summaries available
- [ ] Cosine sanity checks documented
- [ ] Best-layer-by-Shao-spec sweep noted (L15 0-indexed for all 3 traits per Shao)

---

## Stage 3: Steering Evaluation — 24 cells (~10 steps, MEDIUM DETAIL)

**Purpose**: run 4 cells × 2 arms × 3 traits = 24 steering coefficient searches, each writing to a distinct `prompt_set` to avoid cache collision

**Depends on**: Stage 2 vectors

**Cell scheme** (same for both arms, all 3 traits):

| Cell | Eval questions | Trait judge prompt | `--prompt-set` name |
|---|---|---|---|
| 1 | Shao's 20 (from `their_data/{trait}_eval.json`) | Shao's `eval_prompt` | `shao-q_shao-judge` |
| 2 | Shao's 20 | Repo default | `shao-q_our-judge` |
| 3 | Our 20 (from `pv_natural_v2/{trait}/steering.json`) | Shao's `eval_prompt` | `our-q_shao-judge` |
| 4 | Our 20 | Repo default | `our-q_our-judge` |

**Common flags** (all cells): `--method mean_diff` (NOTE: SINGULAR — different from extraction), `--it-model`, `--search-steps 8`, `--layers 30%-60%`, `--save-responses best`, `--min-coherence 77`

**Per-cell flag differences**:
- Cells 1, 3 (Shao judge): `--eval-prompt-from pv_instruction_v2/{trait}`
- Cells 2, 4 (our judge): `--no-custom-prompt`
- Cells 1, 2 (Shao questions): need `--questions-file experiments/persona_vectors_replication/their_data/{trait}_eval.json` (may need wrapper that produces `{"questions": [...]}` if Shao's JSON has different schema — verify in Stage 0)
- Cells 3, 4 (our questions): no `--questions-file` (defaults to trait's steering.json questions)

**Concrete example — Arm A, evil, Cell 1**:
```bash
python steering/run_steering_eval.py \
  --experiment pv_rep \
  --traits pv_instruction_v2/evil \
  --it-model \
  --method mean_diff \
  --search-steps 8 \
  --questions-file experiments/persona_vectors_replication/their_data/evil_eval.json \
  --prompt-set shao-q_shao-judge \
  --eval-prompt-from pv_instruction_v2/evil \
  --save-responses best \
  --min-coherence 77
```

**Key steps**:
1. Run 24 baselines first (`--baseline-only`) — one per (trait × arm × question-set × judge-prompt). Baselines DO depend on both question set and judge prompt (different judge → different score on the same response). Fast — no coef search.
2. Run the 24 coefficient-search cells. Per-cell estimate: 10-30 min (10 layers × 8 steps × 20 questions × ~6s/gen, batched; judge calls in parallel).
3. **Realistic Stage 3 wall clock: 6-12 hours.** Generation dominates; H100 batching helps but doesn't eliminate per-step seriality of the adaptive search.

**Stopping criterion**: 24 results.jsonl files exist (one per cell), each with at least one valid steered run (coherence ≥ 77).

**If results.jsonl missing for a cell**: search trajectory may have not produced any coherence-passing runs. Re-run with `--up-mult 1.15` (gentler steps), or fix the natural prefix issue if Arm B fails.

### Checkpoint: After Stage 3
- [ ] All 24 search results.jsonl files present
- [ ] Each has best-cell entry above coherence threshold (or documented as "below threshold for all coefficients")
- [ ] No silent cache reuse between cells (verify by spot-checking 2 cells with different judge prompts have different scores)
- [ ] Notepad updated; first headline-table draft sketched

---

## Stage 4: Naturalness scoring (~5 steps, MEDIUM DETAIL)

**Purpose**: produce naturalness scores for baseline + best steered config per (trait, arm), so we can report Δ-naturalness in the headline

**Depends on**: Stage 3 best-cell responses + Stage 0 baseline responses

**Key steps**:

4.1 — **Write naturalness scoring wrapper** (`experiments/pv_rep/scripts/score_naturalness_with_baseline.py`):
- Read `dev/steering/score_naturalness.py` for the existing pattern
- New script accepts: `--experiment pv_rep --trait <trait> --arm <pv_instruction_v2|pv_natural_v2> --cell shao-q_our-judge`
- Loads baseline responses from results.jsonl (baseline entries) + best-cell steered responses
- Calls `TraitJudge.score_naturalness_batch` on both
- Writes `experiments/pv_rep/naturalness/{arm}_{trait}.json`: `{baseline_mean, steered_mean, delta, n_baseline, n_steered, per_response_scores}`
- **Verify**: `python experiments/pv_rep/scripts/score_naturalness_with_baseline.py --dry-run --trait evil --arm pv_natural_v2` prints expected I/O paths without scoring

4.2 — Run it for all 6 (trait × arm) pairs

4.3 — Aggregate to `experiments/pv_rep/naturalness/summary.json` with all 12 numbers

**Stopping criterion**: 12 naturalness numbers (6 baselines + 6 steered means) saved to `experiments/pv_rep/naturalness/results.json`.

**If naturalness scores look off** (e.g., 0 across the board): check that backend isn't degenerate (logprob expectation might collapse on long completions).

### Checkpoint: After Stage 4
- [ ] Naturalness results saved
- [ ] Delta computed per (trait, arm)
- [ ] Notepad: which arm has higher naturalness, by how much

---

## Stage 5: Analysis & Figures (~10 steps, LIGHT DETAIL)

**Purpose**: rebuild the headline table, compute bootstrap CIs, save figures, write findings

**Depends on**: Stages 3 + 4

**Key steps**:
1. Aggregate all 24 steering cells' best-result + per-question scores into one DataFrame
2. Compute headline table: per (trait, arm, cell) → baseline trait_score, best steered trait_score, Δ, coherence at best, kept-N
3. Bootstrap CIs over the 20 eval questions per cell, for the Δ — 1000 resamples, paired across arms within each (trait, cell)
4. Post-hoc rescore cells 1 + 3 (Shao-judge) with **Shao's coherence prompt** (using a thin wrapper around `TraitJudge` that loads Shao's prompt) → produce a "faithful coherence" filter pass for those cells
5. Cosine matrix figure: 3 traits × 4 layer choices (Shao's L15 + per-arm best layer) → mark which entries are within "extraction noise" baseline (0.84) vs "real shared direction"
6. Per-cell trait_score lift bar chart: 4 cells × 2 arms × 3 traits, with bootstrap CIs as error bars
7. Naturalness Δ bar chart: 3 traits × 2 arms
8. Save all figures to `experiments/pv_rep/figures/`
9. Write `experiments/pv_rep/findings.md`: headline numbers, robustness across cells, caveats from the Locked Decisions table, mechanistic story (refer to prior `why_this_happens/` results)

**Stopping criterion**: figures + findings written; critic agent approves the conclusions

### Checkpoint: After Stage 5
- [ ] Headline table built
- [ ] Bootstrap CIs computed
- [ ] All figures saved
- [ ] Findings written
- [ ] Critic agent passes review

---

## Stage 6: Final critic + writeup notes (~3 steps, LIGHT)

**Purpose**: independent agent verifies the conclusions; produce a methodology + caveats note for the eventual viz-finding rewrite

**Key steps**:

6.1 — Spawn `r:verifier` agent with **explicit checklist**:
- (a) **Layer-indexing audit**: pick one figure that says "Shao's L16" — verify the 0-indexed equivalent (L15) was actually used as the comparison row
- (b) **Cache-collision spot check**: take any two cells with same VectorSpec but different judge prompts (e.g., `shao-q_shao-judge` and `shao-q_our-judge` for evil PV-Instruction at the chosen layer) — verify the trait_score and coherence_score values DIFFER. If identical → cache collision → invalidate both cells
- (c) **Bootstrap CI reproducibility**: re-run the bootstrap with a different seed → CIs should overlap heavily (>80% containment)
- (d) **Judge metadata logged**: every response JSON has `judge_model_id` + `judge_prompt_version`/`eval_prompt_used`
- (e) **Coherence threshold consistency**: faithful-replication cells (Shao judge) use 75; our-judge cells use 77 — confirm the headline table applies the right one per cell
- (f) **Per-arm vetting retained N reported** (not silently inflated/deflated)

6.2 — If verifier flags issues → fix + re-run affected analyses

6.3 — Write `experiments/pv_rep/methodology_notes.md`: every locked decision + caveat in user-readable prose, ready to drop into the viz finding rewrite

### Checkpoint: After Stage 6
- [ ] Verifier passes
- [ ] Methodology notes written
- [ ] Findings finalized
- [ ] Status → COMPLETE

---

## If Stuck

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError` on H100 | judge_backends.py not committed | go back to 0.1 |
| Extraction OOM | model loading fp32 | check `--load-in-4bit` flag if needed, or reduce batch size |
| Vetting retains 0 responses | judge model rejected nearly everything (refusals); or threshold mismatch | re-read 10 raw responses; possibly drop threshold to 40 for diagnosis |
| Cell results identical across judge prompts | cache collision | confirm distinct `--prompt-set` names per cell; spot-check 2 cells' first run timestamps |
| Arm B vetting retains <20 responses | base model decohered too fast / dataset issue | (a) ask user, (b) re-spawn dataset subagent with critique, (c) expand scenarios to 50; do NOT proceed with bad vector |
| `argparse: unrecognized argument: --methods` | wrong flag spelling | extraction = `--methods` (plural); steering = `--method` (singular). Verify in script. |
| All steered coherence < 77 | coefficient search not finding valid range | re-run with `--up-mult 1.15` and `--start-mult 0.5` |
| Natural arm "wins" by 50pt suspiciously | judge prompt mismatch (cell using wrong eval_prompt) | check `--eval-prompt-from` vs `--no-custom-prompt` on the failing cell |
| Layer L15 (0-indexed) not in `30%-60%` | layer arg parsing issue | check resolved layers in pipeline log; explicitly set `--layers 8-20` if needed |

## Notes
_(Space for observations during run.)_
