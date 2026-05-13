# Fix PV Traits — Notepad

Status: COMPLETE (all 3 traits escalated)

Restore Gate 1 ≥ 90/90 for `pv_natural/{hallucination, evil, sycophancy}` on Llama-3.1-8B base. Output to `datasets/traits/pv_natural_v2/`.

## Step log

- 2026-04-28 — start. Read plan, doc, baselines. Created `pv_natural_v2/` dirs with copied definition+steering. Beginning hallucination iter 1.
- 2026-04-28 — hallucination 4 iters: 57→45→45→61→71. ESCALATED. Remaining issue: dialogue-interruption from `I told her, "..."` lock-ins.
- 2026-04-28 — evil 4 iters: 45→13→8→12→20. ESCALATED. Every iter worse than baseline. Llama RLHF wall; recommend trying a different base model.
- 2026-04-28 — sycophancy 3 iters: 20→35→42→41. ESCALATED. Plateaued; same dialogue-shift fix recommended as hallucination.
- 2026-04-28 — COMPLETE. See per-trait `*_ESCALATION.md` and `*_iteration_log.md` in `dev/tasks/fix_pv_traits/results/`.
