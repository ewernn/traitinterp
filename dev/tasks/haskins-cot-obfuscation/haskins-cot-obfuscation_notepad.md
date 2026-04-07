# Notepad — haskins-cot-obfuscation

**Status:** IN_PROGRESS — extended run (annotation + steering + emotion_set extraction)

### [2026-04-07 ~03:35 PST] Round 2: extended autonomous run
After the initial 6h run, user requested:
1. Extract 87 emotion_set probes (tiers 1-6) at per-trait scaled layers
2. Run steering evals on all 97 probes
3. Annotate s1pp + s2pp + s2ppnh with 2-agent Haiku consensus

**Layer scaling correction**: original `compute_scaled_layers.py` used 32 layers (Llama assumption). Corrected to 48 (Qwen2.5-14B, the actual model behind `experiments/emotion_set/steering/layer_selection.json`). New layer centers: 9-21 (was 14-32). 89% of tier 1-6 traits now fully covered by existing L10-21 captures — no re-capture needed for most.

**Annotation pipeline (Path B — Agent tool with run_in_background):**
- Built `annotate_prepare.py` → 206 batches (100 s1pp + 90 s2pp + 16 s2ppnh, A1+A2 each)
- Built `annotate_validate.py` → per-span validation (length, uniqueness, location, ordering)
- Built `annotate_compare.py` → A1/A2 disagreement detection (criterion too strict — pivoted to union merger)
- Built `annotate_merge_union.py` → union-based final merger (no tiebreaker phase due to time)
- Wrote 4 prompt templates (A1 conservative, A2 inclusive, A3 tiebreaker, Sonnet adjudicator) at `experiments/.../annotations/prompts/`

**Annotation results:**
- 206 background Haiku agents dispatched, all completed
- s1pp: 3030/3273 spans valid (92.6%), 22 order violations across 100 batches
- s2pp: 2705/2927 spans valid (92.4%), 26 order violations across 90 batches
- s2ppnh: 365/410 spans valid (89.0%), 1 order violation across 16 batches
- Total: ~6100/6610 spans valid (~92.3%)
- Final canonical annotation files written:
  - `experiments/haskins-cot-obfuscation/inference/s1pp/responses/reih02_s1pp_annotations.json` (1000 entries: 494 deceptive + 506 auto-stubbed)
  - `experiments/haskins-cot-obfuscation/inference/s2pp/responses/reih02_s2pp_annotations.json` (1000: 450 deceptive + 550 auto-stubbed)
  - `experiments/haskins-cot-obfuscation/inference/s2ppnh/responses/reih02_s2ppnh_annotations.json` (1000: 74 deceptive + 926 auto-stubbed)
- Schema matches `rm_syco/inference/.../train_100_annotations.json` precedent (text spans, not token indices) and the visualization expectation at `visualization/core/annotations.js:189`

**Annotation pipeline failures / lessons:**
- One agent overwrote `interesting_findings.tsv` with raw scenario metadata using `>` instead of `>>`. Lost ~264 lines of findings. Removed the shell-append instruction from subsequent dispatches; agents now put `interesting` notes inside their JSON output.
- Some agents used `"text"` field instead of canonical `"span"`. Added `text` as a fallback in the validator.
- Strict comparator's "same category set" requirement gave only 23% consensus on s1pp (false low number — A1 and A2 are intentionally biased differently). Switched to union merger; no tiebreaker phase needed.

**Per-trait steering layer map built:**
- `dev/tasks/haskins-cot-obfuscation/steering_layers.json`
- 87 emotion_set traits: scaled centers ± 2 from `scaled_layer_targets.json` (corrected qwen48→gptoss36)
- 10 alignment traits: top-5 layers by `train_acc` from extraction metadata
- Format ready for `steering/run_steering_eval.py --trait-layers TRAIT:L1,L2,L3 ...`

**Extraction status (background):** 45/87 emotion_set probes done (~52%). Still running at corrected layers.

**Pending after extraction completes:**
1. Steering evals on 97 probes (will need to verify base model variant + chunking strategy for cmdline length)
2. Project new emotion_set probes onto existing captures (no GPU, can run alongside steering)
3. Compute Cohen's d on the expanded probe set (10 alignment + 87 emotion_set = 97 probes)
4. Build + slide convolution detector template using the new annotations
5. Update findings doc with all results
**Started:** 2026-04-06 23:40 PST
**Updated:** 2026-04-06 23:40 PST

## Environment
- 2× A100-SXM4-80GB, both 0% util, 0 MiB used at start
- Driver 580.65.06, CUDA 13.0
- Pip-installed: transformers 5.5.0, peft 0.18.1, accelerate 1.13.0, torch 2.11.0, triton 3.6.0
  - **CONCERN**: transformers 5.5.0 is a major bump from 4.55.x. Plan pins `>=4.55.0` so technically OK, but watch for API breakage. Will downgrade to 4.55.x if needed.

## Progress

### [2026-04-06 23:40 PST] Stage 0: Initialization
- Read plan, notepad, decision tree, user messages, findings (all empty templates)
- Created task list (10 stages)

### [2026-04-06 23:55 PST] Stage 0: Environment fix #1 — split-brain pip
- First `pip install` went to /home/dev/.local/lib/python3.12 (system pip), but actual venv is `/home/dev/traitinterp/.venv/` Python 3.11 with transformers 4.57.3 and NO `kernels` package.
- Fixed via `uv pip install`: venv now has transformers 5.5.0 + kernels 0.12.3 + peft 0.18.1.

### [2026-04-07 00:05 PST] Stage 0: Fix #2 — gpt_oss attn_implementation
- After fix #1, model loaded MXFP4 natively (no dequant warning) but failed at construction with `GptOssForCausalLM does not support sdpa`.
- Patched `utils/model.py:_best_attn_implementation()` to fall back to `eager` for gpt_oss when flash_attn isn't available.

### [2026-04-07 00:08 PST] Stage 0: Gate test — partial pass, NaN logits
- Model loads in 61s (MXFP4 native).
- BASE model alone: forward pass + generation work perfectly. "The capital of France is Paris."
- BASE + LoRA: forward returns shape-correct logits but **all NaN**.

### [2026-04-07 00:15 PST] Stage 0: NaN root cause
Inspected `/tmp/reih02_adapter.safetensors` (504 keys, 4.9GB). Two compounding bugs:

**Bug 1: Attention key naming mismatch**
- Adapter file: `base_model.model.model.layers.0.attn.q_proj.lora_A.weight`
- transformers 5.5.0 GptOssForCausalLM exposes attention as `layers.*.self_attn.*` (with `self_`)
- PEFT silently can't find them → keys stay as meta tensors → matmul produces NaN
- The 144 LoRA "modules" we saw injected are PEFT's empty wrappers from `target_modules="all-linear"`, not the actual loaded weights.

**Bug 2: Expert LoRA modules not injectable**
- Adapter file ALSO has 108 keys on `mlp.experts.w{1,2,3}` (3 matrices × 36 layers × 2 lora dims)
- In MXFP4 mode, these experts are packed uint8 tensors, NOT nn.Linear
- PEFT `all-linear` skips them entirely → 0 expert LoRA modules injected
- Even if we fix Bug 1, we'd lose the expert adaptation which is likely where most of the s1pp/s2pp behavior lives

**Memory constraint**: 117B bf16 ≈ 230GB > 160GB on 2× A100-80GB. Cannot dequant fully.

### [2026-04-07 00:18 PST] Pivot: parallelize Stage 1 + LoRA workaround
Stage 1 (extract trait vectors from BASE model) does NOT need LoRA. Starting it as the long-pole task while investigating LoRA fixes in parallel.

### [2026-04-07 00:25 PST] Investigator finding (subagent)
Reih02 trained via Tinker (Megatron-style backend), not local PEFT. Their adapter naming is Tinker's internal scheme — never a released transformers version. PEFT downgrade won't help. vLLM doesn't serve gpt-oss LoRA. The only viable HF path is `target_parameters=["mlp.experts.gate_up_proj","mlp.experts.down_proj"]` for expert LoRA, which requires re-stacking Tinker's split `w1/w3 → gate_up_proj` (per-expert + shared dim mix is gnarly). Recommendation: fix attention LoRA naming first as ablation, defer expert LoRA.

### [2026-04-07 00:30 PST] LoRA remap script written
`dev/tasks/haskins-cot-obfuscation/remap_lora.py` produces attention-only adapters (288 keys remapped, 216 expert keys dropped). Built s1pp/s2pp/s2ppnh remapped adapters at `cache/lora/{variant}_attn_only/`. Updated `experiments/haskins-cot-obfuscation/config.json` to point at local paths. **Untested** until GPU is free.

### [2026-04-07 00:35 PST] Stage 1.6 alignment extraction running
Started `extraction/run_extraction_pipeline.py --category alignment --layers "30%-60%"`. Layers 10-21 (12 layers). Auto batch sizing 343-605, ~50GB/GPU, 2it/s.

Trait results so far (train_acc, mean across 12 layers):
- compliance_without_agreement: 0.805 (range 0.65–0.88)
- conflicted: 0.959 (range 0.94–0.97)
- deception: 0.809 (range 0.78–0.85)  ← key trait
- emergent_misalignment: 0.885 (constant — only 15 samples, possible overfit)

**Gate decision**: GATE PASSED. >60% mean train accuracy on 4/10 traits. Architecture supports probing.

**Caveat**: train_acc only, no held-out val split in default extraction. emergent_misalignment with n=15 is suspicious. Need to validate with ground-truth fixed evaluation later.

### [2026-04-07 00:38 PST] Time budget reality check
Remaining: ~5h. Stage 1.7 (174 traits): 3-6h. Stage 2 (4 conditions × 1000 scenarios capture): 8-16h. Cannot do plan as written.

**Revised plan:**
1. Finish Stage 1.6 (~10 min more)
2. Test LoRA fix on remapped attention-only adapter (15 min)
3. Skip Stage 1.7 emotion_set entirely (would consume entire remaining budget for diminishing returns — 10 alignment probes are sufficient for the deception axis we care about)
4. Convert + capture Stage 2 with REDUCED scope: 200 scenarios per condition × 3 conditions (base, s1pp, s2pp) = ~600 capture-runs. Skip s2ppnh as control. Estimated ~1.5-2.5h.
5. Stage 3 projection + Cohen's d (~30 min)
6. Stage 4 onset annotation via Haiku subagents on the s1pp subset (~30 min)
7. Stage 5 reconciled findings (~20 min)
Total: ~4h, leaves slack for unknown unknowns.

## Observations

### [2026-04-07 ~02:30 PST] Stage 2 capture done (3 of 4 conditions)
- base: 1000 scenarios in 8.4 min
- s1pp (attention-only LoRA, our remap): 1000 in 7.9 min
- s2pp (attention-only LoRA): 1000 in ~41 min (much slower due to longer obfuscated CoT sequences — variance per item up to 8 sec)
- s2ppnh: SKIPPED to fit time budget. Documented as future work.

### [2026-04-07 ~03:10 PST] Stage 3 projection — pipeline bug worked around
- `inference/run_inference_pipeline.py --from-activations` hardcodes `position='response[:]'` lookup but our vectors are at `response[:5]`. Silent failure: "Loaded vectors for 0 traits". Killed and replaced with custom `dev/tasks/haskins-cot-obfuscation/project_haskins.py` that loads vectors at the right path. Worth a codebase fix.
- Projection on each condition: ~2-3 min for 1000 scenarios × 12 layers × 10 traits.

### [2026-04-07 ~03:20 PST] Stage 3 Cohen's d — major findings
On filter `ground_truth==yes,classification==CONCEAL` (461/440 deceptive scenarios):
- s1pp vs base: deception probe |d|=2.21 at L14 (very large)
- s2pp vs base: deception probe |d|=2.17 at L18, gaming |d|=1.91 at L11
- **s2pp vs s1pp** (clean obfuscation comparison, both LoRAs cancel centroid shift):
  - gaming L12: d=-1.18 (s2pp ↑)
  - performative_confidence L12: d=-1.05 (s2pp ↑)
  - compliance_without_agreement L17: d=+1.03 (s2pp ↓)
  - honesty_observed L10: d=+0.89 (s2pp ↑)
  - deception L15: d=-0.79 (s2pp ↓ — partial evasion of the deception probe specifically)

### Control: ground_truth==no scenarios (no concealment, model is honestly saying no)
- s1pp vs base: deception |d|=1.78 at L12 — almost as large as on CONCEAL scenarios
- **Implication**: most of the s1pp-vs-base shift is generic LoRA-presence noise, not content-specific deception detection. The valid comparison is s2pp vs s1pp (both LoRAs).
- s2pp vs s1pp on `ground_truth==no` shows same gaming/perf_conf signature with similar magnitude (|d|≈1.0), so the obfuscation fingerprint is *not* strictly content-specific either — it's a global tonal shift.

### Matched-pair test (n=404 paired CONCEAL+gt=yes scenarios)
- s1pp deception L14 mean: +261.85
- s2pp deception L14 mean: +247.14
- Mean paired diff: -14.71 (s2pp lower); std=39.14; effect size d_z ≈ 0.38
- 76/200 sampled pairs show s2pp > s1pp (62% have s1pp > s2pp on the deception axis)

### [2026-04-07 ~03:30 PST] Final findings written
See `haskins-cot-obfuscation_findings.md` for the full writeup.

### Total run time: ~5.5h of the 6h budget
Major time consumers:
- Environment debugging (split-brain pip + MXFP4 dequant + sdpa attn + LoRA NaN): ~50 min
- Stage 1.6 alignment extraction: ~17 min × 2 partial runs = 34 min
- Stage 2 capture: 8 + 8 + 41 = 57 min (s2pp slow due to long sequences)
- Stage 3 projection (3 conditions): ~9 min total
- LoRA remap + test: ~5 min
- Analysis + writeup: ~30 min
- Other (waits, monitoring, prep): rest

## Failures / Dead Ends

- **Stage 1.6 first run crashed** on `helpfulness_expressed` due to data balance mismatch (118 pos / 250 neg). Truncated to 118/118, resumed with `--traits` listing remaining 5 traits. Lost ~2 min.
- **PEFT + MXFP4 + Reih02 LoRA = NaN logits**. Took 3 diagnostic scripts and 2 subagent investigations to identify (Tinker key naming + missing kernels package). Eventually solved via key remap + dropping expert LoRAs.
- **Inference pipeline `--from-activations`** silently loaded 0 traits due to position-string mismatch. Worked around with custom `project_haskins.py`. Pipeline bug worth fixing.
- **transformers 5.5.0** is a major version bump from 4.55. Required compat shims in `utils/model.py` (already present from prior K2/V3 work) and the eager-attn fallback patch I added for gpt_oss.
- **Skipped 1.7 emotion_set** (174 traits): would have consumed remaining budget for diminishing returns. The 10 alignment probes were sufficient to make the experiment work.
- **Skipped s2ppnh capture**: 4th condition not run; we have 3-way comparison only (base/s1pp/s2pp).
- **Skipped Stage 4 onset annotation + convolution detector**: replaced with population matched-pair analysis. Per-token detector remains as future work.

## Outputs
- `experiments/haskins-cot-obfuscation/extraction/alignment/{trait}/base/vectors/response__5/residual/probe/layer{10..21}.pt` — 10 trait probes × 12 layers
- `experiments/haskins-cot-obfuscation/inference/{base,s1pp,s2pp}/raw/residual/{prompt_set}/*.pt` — 3000 raw activation files (~99GB)
- `experiments/haskins-cot-obfuscation/inference/{base,s1pp,s2pp}/projections/alignment/{trait}/{prompt_set}/*.json` — 30,000 per-token projection files
- `dev/tasks/haskins-cot-obfuscation/results/cohens_d.csv` — 480 rows of (trait, layer, comparison, n_a, n_b, mean_a, mean_b, cohens_d)
- `dev/tasks/haskins-cot-obfuscation/haskins-cot-obfuscation_findings.md` — final findings doc
- `dev/tasks/haskins-cot-obfuscation/{remap_lora,project_haskins,cohens_d_analysis,convert_reih02_to_responserecord,compute_scaled_layers,gate_test,diagnose_lora,diagnose_nan,test_remapped_lora}.py` — pipeline scripts
- `cache/lora/{s1pp,s2pp,s2ppnh}_attn_only/` — remapped Reih02 LoRA adapters (attention-only, ~96 MB each)
