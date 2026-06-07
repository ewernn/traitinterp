# May 13 — Walk the trait_name → validated trait vector pipeline

## Goal
Step through every stage of the trait extraction pipeline, verifying/optimizing each, on a single trait at a time. Unblock the 3 PV traits (hallucination → evil → sycophancy) end-to-end before scaling up to Plans A & B.

## Why
We've been gating on Gate 1 (text-judge pass rate) and concluding the PV datasets are broken. Per `docs/trait_dataset_creation_base_model.md`, that's a proxy. The real gates are `val_effect_size` (Stage 6) and steering Δ (Stage 10). A vector can be usable even when base-model completions don't visibly express the trait. We don't know which gates actually fail until we run the whole pipeline once.

## The 10 stages

| # | Stage | Tool | Pass signal |
|---|---|---|---|
| 1 | Author dataset | `datasets/traits/{cat}/{trait}/{positive,negative}.txt + definition.txt + steering.json` | Files exist, matched counts, 1p, lock-in styles <40%/type |
| 2 | Generate | `extraction/run_extraction_pipeline.py --only-stage 1 --max-new-tokens 32` | Each scenario has 32-token completion |
| 3 | Vet (Gate 1) | `dev/extraction/validate_trait.py --modal --scenarios-only` | pos_rate & neg_rate ≥ 0.9 ideal, **not blocking** |
| 4 | Capture activations | pipeline stage 3, `--position 'response[:5]' --val-split 0.1` | Activations written, ≥5 val per polarity |
| 5 | Extract vectors | pipeline stage 4, `--methods probe,mean_diff` | Vectors written per (layer, method) |
| 6 | **Evaluate (Gate ★)** | per-vector metadata.json | `val_effect_size > 1.0` AND `val_auroc > 0.85` at some L8–L20 layer |
| 7 | Logit lens | pipeline stage 5 (auto) | Top-toward tokens semantically match trait |
| 8 | Aggregate | `analysis/vectors/extraction_evaluation.py --experiment {X}` | extraction_evaluation.json populated |
| 9 | Baseline (Gate 2) | instruct model on `steering.json` | trait score < 30 |
| 10 | Steer (Gate 3) | `dev/steering/modal_evaluate.py` | best Δ > 15 at coherence ≥ 77 |

Vector is **validated** when 6 + 10 both pass.

## Modal status (verified 2026-05-13)
- `trait-extraction` app at `dev/extraction/modal_pipeline.py:30` — A100-80GB, entrypoint `extraction_pipeline_remote()`, runs stages 1-4. Called via `dev/extraction/modal_extract.py:155`.
- `trait-steering` app at `dev/inference/modal_steering.py:34` — A100-80GB, entrypoint `steering_eval_remote()`. Called via `dev/steering/modal_evaluate.py:195`.
- `validate_trait.py --modal` does **not** wire Modal into Gates 2-3 (lines 262-299 run local). For this walkthrough, call the Modal apps directly.

## Trait order
1. **hallucination** (49 scenarios — smallest, cleanest failure mode, current Gate 1: 57/94)
2. **evil** (150 — current Gate 1: 45/99, RLHF resistance expected)
3. **sycophancy** (150 — current Gate 1: 20/88, both sides failing)

## Stage-by-stage plan

### Stage 1 — Author (hallucination) ✅
Already done. 49/49, definition + steering present.

### Stages 2-8 — Run on Modal
```bash
python dev/extraction/modal_extract.py \
    --experiment hallucination_pipeline_test \
    --traits pv_natural/hallucination \
    --methods probe,mean_diff \
    --position 'response[:5]' \
    --max-new-tokens 32 \
    --val-split 0.1
```
(Verify argparse on modal_extract.py before running.)

Then:
```bash
python analysis/vectors/extraction_evaluation.py --experiment hallucination_pipeline_test
```

**Stop and read** `experiments/hallucination_pipeline_test/extraction/extraction_evaluation.json`:
- Per-layer val_effect_size, val_auroc, polarity_correct for probe + mean_diff
- Layer with max val_effect_size; check if > 1.0
- Cross-check polarity (val_accuracy on val split close to 1.0, not 0.0)
- Logit-lens top-toward tokens at best layer — do they look hallucination-y?

If Stage 6 passes → continue. If not → diagnose: padding tokens? wrong position? probe collapse? Gate 1 failure mode actually mattering?

### Stages 9-10 — Steering eval on Modal
```bash
python dev/steering/modal_evaluate.py \
    --experiment hallucination_pipeline_test \
    --trait pv_natural/hallucination \
    ...
```
(Verify args; the runner picks best layer from extraction metadata.)

**Stop and read** steering outputs:
- Baseline trait score on instruct (must be <30; if ≥30, steering.json invites the trait — fix questions)
- Coefficient sweep: best Δ at coherence ≥ 77

### Then: evil, then sycophancy
Same rhythm. Note where each one fails; the failure mode tells us whether the bottleneck is dataset, position, model, or judge.

## Stopping criteria
- All 3 PV traits clear Stages 6 + 10 → proceed to Plan A (pad hallucination to 150 first) and Plan B (generate fresh emotions).
- Any trait fails at the same stage → root-cause and fix; don't paper over.
- Total walkthrough estimate: ~1-2 hr Modal compute per trait + diagnosis time.

## Plans A & B (after this walkthrough)
- **Plan A** (`dev/tasks/1st_vs_3rd_person_extraction/`): 1p vs 3p (singular they) on the 3 PV traits at response[:5]. Needs validated 1p vectors first.
- **Plan B** (`dev/tasks/base_extraction_position/`): position sweep × 6 traits × 2 methods × 2 models. Needs validated PV vectors + fresh emotion datasets.

## Open question
- Should we wire Modal into `validate_trait.py` Gates 2-3 (the TODO from `docs/other/validate-and-fix-traits.md`) as part of this walkthrough, or just bypass via direct Modal calls for now? Recommendation: bypass for the first walk; wire it up once we know what Gates 2-3 actually need.
