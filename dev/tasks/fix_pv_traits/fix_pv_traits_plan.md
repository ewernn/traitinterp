# Fix PV Traits — Restore Gate 1 ≥ 90/90 on Llama-3.1-8B base

## Goal
Restore Gate 1 (scenario validation) pass rate ≥ 90/90 for `pv_natural/{hallucination, evil, sycophancy}` on Llama-3.1-8B base. Output to `datasets/traits/pv_natural_v2/` to avoid breaking other experiments.

## Why
All three PV traits fail Gate 1 today (`experiments/1st_vs_3rd_person_extraction/results/validate_trait_gate1_pv_natural__*.json`):
- hallucination: 57% pos / 94% neg
- evil: 45% pos / 99% neg
- sycophancy: 20% pos / 88% neg (both sides failing)

Llama-3.1-8B base resists generating these trait completions. Per `docs/other/validate-and-fix-traits.md`'s Step 3, the fix pattern is **explicit motivation + fait accompli lock-ins** (the trick that took alignment/self_serving from 67→99).

## Iteration Pattern (per docs/trait_dataset_creation_base_model.md)

1. Write candidate `positive.txt + negative.txt` into `datasets/traits/pv_natural_v2/{trait}/`
2. Copy `definition.txt` + `steering.json` verbatim from `datasets/traits/pv_natural/{trait}/`
3. Run Gate 1: `python dev/extraction/validate_trait.py --model meta-llama/Llama-3.1-8B --trait pv_natural_v2/{trait} --modal --scenarios-only --experiment fix_pv_traits`
4. Read failing completions (printed to stdout + saved to `experiments/_validate/scenarios/results.json`)
5. Diagnose:
   - Confident-fabrication failures → add explicit motivation ("I've already decided to make this up because…")
   - Honesty-bias failures → fait accompli lock-ins ("…so I went ahead and told them") so the action is already committed
   - Self-correcting completions → cut prefix earlier (before model can hedge)
   - Confound failures (negatives expressing trait too) → strengthen opposite peak in negatives
6. Regenerate weak scenarios in bulk (per the doc: "Generate more, cull bad ones"); don't spot-fix.
7. Repeat until Gate 1 ≥ 90/90 or 4 iterations elapsed.

## Trait Order
1. **hallucination** (49 scenarios, smallest, closest to passing at 57/94 → easiest first)
2. **evil** (150 scenarios, 45/99)
3. **sycophancy** (150 scenarios, 20/88, hardest — both sides failing)

## Persisted Outputs
- Final datasets: `datasets/traits/pv_natural_v2/{trait}/{positive,negative,definition,steering}.{txt,json}`
- Iteration logs: `dev/tasks/fix_pv_traits/results/{trait}_iteration_log.md`
- Gate 1 baselines (per iteration): `experiments/fix_pv_traits/results/validate_trait_gate1_pv_natural_v2__{trait}.json` (and -2, -3 on collision)

## Stopping Criteria (per trait)
- Gate 1: pos_rate ≥ 0.90 AND neg_rate ≥ 0.90
- OR: 4 iterations elapsed without convergence → escalate to user with iteration log + diagnosis

## Pipeline-Level Sanity
- Don't modify `datasets/traits/pv_natural/*` (other experiments depend on it; per the user, copy convention to a new dir)
- `dev/extraction/validate_trait.py` race-condition: `experiments/_validate/` is shared, so all validate_trait.py invocations must SERIALIZE. Single orchestrator subagent handles all 3 traits sequentially; do not spawn parallel subagents.
