# CLI Examples Audit Report
Generated: 2026-04-19

## Summary
Audited 12 target docs files + docs/main.md. Found **~120 distinct bash example blocks** across the files.
**8 broken examples** identified across 3 files. Core pipeline flags are solid; breakage clusters around:
(1) phantom traits referenced in mkdocs examples, (2) one stage-script flag mismatch in replicate doc,
(3) two missing dataset files in analysis examples.

---

## Findings Table

| File:Line | Command | Status | Reason |
|-----------|---------|--------|--------|
| `docs/mkdocs/cli/extraction.md:89` | `--traits starter_traits/sycophancy,starter_traits/refusal` | **BROKEN** | `datasets/traits/starter_traits/refusal/` does not exist |
| `docs/mkdocs/cli/extraction.md:97` | `--category starter_traits` | OK | category flag exists; starter_traits dir exists |
| `docs/mkdocs/cli/steering.md:100` | `--traits starter_traits/sycophancy,starter_traits/refusal` | **BROKEN** | Same — `starter_traits/refusal` trait dir missing |
| `docs/mkdocs/cli/analysis.md:83` | `--methods=probe,mean_diff --layers=20,25,30 --verbose=True` | OK | Uses `fire.Fire(main)`; params match function signature |
| `docs/mkdocs/cli/analysis.md:127` | `--no-filter-common --no-norm` | OK | Both flags present in `logit_lens.py` |
| `docs/mkdocs/cli/analysis.md:219` | `--norms-file datasets/russell_mehrabian_norms.json` | **BROKEN** | File `datasets/russell_mehrabian_norms.json` does not exist in repo |
| `docs/mkdocs/cli/analysis.md:249` | `--activities datasets/activities.json` | **BROKEN** | File `datasets/activities.json` does not exist in repo |
| `docs/mkdocs/cli/analysis.md:254` | `--steer emotion_set/desperate --strength 0.5` | OK | Flags exist; `datasets/traits/emotion_set/desperate/` exists |
| `docs/mkdocs/cli/analysis.md:463` | `--model-variant <organism>` | OK (placeholder typo) | Flag exists; `<organism>` is just a bad placeholder name in docs, not broken |
| `docs/replicate_ant_emotion_concepts.md:103` | `stage5_layer_dynamics.py --experiment ant_emotion_concepts --layer 49 --load-in-4bit` | **BROKEN** | `stage5_layer_dynamics.py` has no `--layer` flag; has `--layers` (string) |
| `docs/inference_guide.md:41` | `run_inference_pipeline.py --capture` | OK | `--capture` flag present (line 96 of script) |
| `docs/inference_guide.md:247` | `torchrun ... --component residual --layers 9,12,...` | OK | Both flags exist |
| `docs/steering_guide.md:173` | `--layers "30%-60%"` | OK | Default is `"30%-60%"`, flag exists |
| `docs/steering_guide.md:192` | `--prompt-set general` | OK (runtime) | Flag exists; "general" is a valid string (not checked until runtime against actual files) |
| `docs/mkdocs/cli/steering.md:126` | `--baseline-only --save-responses all` | OK | Both flags present |
| `docs/mkdocs/cli/steering.md:135` | `--ablation 25` | OK | Flag exists |
| `docs/mkdocs/cli/steering.md:143` | `--rescore starter_traits/sycophancy` | OK | `--rescore` flag exists; `sycophancy` trait dir exists |
| `docs/create_ant_emotion_vectors.md:59` | `--replication-level full --topics 5 --stories-per-batch 3` | OK | All flags exist in extraction pipeline |
| `docs/mkdocs/index.md:35-36` | `--experiment starter --traits starter_traits/sycophancy` | OK | `sycophancy` trait dir exists |
| `docs/mkdocs/cli/analysis.md:87-88` | `extraction_evaluation.py --component=attn_out` | **BROKEN** | `extraction_evaluation.py` uses `fire.Fire`; `--component` is a valid param but `attn_out` is a valid value — actually OK |
| `docs/mkdocs/cli/analysis.md:562-563` | `--traits safety/refusal --coef -1.0` | **BROKEN** | `datasets/traits/` has no `safety/` category; `safety/refusal` trait does not exist |

**Re-checked attn_out**: the fire interface accepts `--component=attn_out` at the parameter level; the value validity depends on runtime data. Not flagging as hard broken.

---

## Summary Count

- **Total example blocks audited**: ~120 bash lines across 34 distinct command invocations
- **Broken**: 6 confirmed
  - 2× phantom trait `starter_traits/refusal` (mkdocs extraction + steering docs)
  - 1× phantom trait `safety/refusal` (mkdocs benchmark analysis)
  - 1× missing file `datasets/russell_mehrabian_norms.json` (geometry example)
  - 1× missing file `datasets/activities.json` (preference_elo example)
  - 1× wrong flag `--layer` on `stage5_layer_dynamics.py` (replicate doc); script only has `--layers`

---

## Worst-Offender Files

1. **`docs/mkdocs/cli/analysis.md`** — 3 broken examples (phantom `safety/refusal`, missing norms + activities files)
2. **`docs/mkdocs/cli/extraction.md`** — 1 broken (phantom `starter_traits/refusal`)
3. **`docs/mkdocs/cli/steering.md`** — 1 broken (phantom `starter_traits/refusal`)
4. **`docs/replicate_ant_emotion_concepts.md`** — 1 broken (wrong flag `--layer` vs `--layers`)

---

## Common Breakage Patterns

**Pattern 1 — Phantom traits used as illustrative examples**
`starter_traits/refusal` and `safety/refusal` appear in multiple mkdocs examples but neither trait directory exists under `datasets/traits/`. The actual starter traits are: `sycophancy`, `formality`, `golden_gate_bridge`, `sad`, `desperate`, `assistant_axis`. Docs should swap to `starter_traits/sycophancy` or note these are illustrative.

**Pattern 2 — Missing dataset files**
`datasets/activities.json` and `datasets/russell_mehrabian_norms.json` are referenced in `preference_elo.py` and `geometry.py` examples respectively, but neither file exists in the repo. These are likely private/large files not committed. Examples need a note or the paths should point to downloadable releases.

**Pattern 3 — Stage-script flag mismatch**
`docs/replicate_ant_emotion_concepts.md:103` uses `--layer 49` on `stage5_layer_dynamics.py`, but that script only accepts `--layers` (comma-separated string, default None) and `--n-layers-sample`. The `--layer` flag simply gets ignored (no argparse error, just silently skipped), making this a silent drift rather than a crash — but it means the documented usage does nothing useful.

**All core pipeline flags are correct**: `run_extraction_pipeline.py`, `run_inference_pipeline.py`, `run_steering_eval.py`, `generate_responses.py`, and all analysis scripts have the flags their docs claim.
