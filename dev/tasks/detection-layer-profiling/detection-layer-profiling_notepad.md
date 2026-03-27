# Detection Layer Profiling — Notepad

**Status:** SUBSTANTIALLY COMPLETE
**Started:** 2026-03-27 01:10 PST
**Last updated:** 2026-03-27 10:37 PST
**Runtime:** ~9.5 hours (with ongoing extraction)

---

## Final Scale
- **244 Qwen3.5-9B + 209 Llama-3.1-8B-Instruct = 453 vector sets**
- **195 common traits** for cross-model comparison
- **14,496 total layer-evaluation results**
- Additional: prompt[-1] extractions, temporal sweeps at 7 positions
- 15 steering evaluation runs ($12 OpenAI spend)
- **568-line findings document** with 10 findings + 9 supplementary analyses
- 25 result files, 13 GB experiment data

## ALL Findings (abbreviated)

### Core
1. **prompt[-1] is 4-40x stronger** for detection (manipulation: 13x on both models!)
2. **Three temporal patterns**: Gate, Front-loaded, Accumulative
3. **Steering range model-specific**: Qwen 30-60%, Llama 60-90%
4. **Val set ≥30%** required
5. **Detection-steering gap direction is model-specific** (positive on Qwen, sometimes negative on Llama)

### Cross-Model
6. **Layer profiles zero-correlated** (r=0.01, n=195)
7. **Geometry transfers** (r=0.77)
8. **Same PCA structure**: PC1=valence, PC2=depth, PC3=deference on both models
9. **Direct vector transfer fails** (cos≈0 cross-model)

### Mechanistic
10. **Vectors rotate 2x faster in early layers** (30-37° vs 14-16°)
11. **Detection-peak steering → repetition loops** (features "read-only" late)
12. **Hallucination steering at detection peak inverts** behavior

### Methodological
13. **Scenario design → orthogonal vectors** for same trait
14. **Semantic categories predict depth**: self-regulation → behavioral
15. **77% of 200 traits perfectly classified**, only 6% below d=1
16. **probe > mean_diff** but converge at mid-late layers

### Safety
17. **Manipulation aligns with both evil AND refusal**
18. **eval_awareness aligns with evil** (cos=0.42)
19. **Sycophancy is anti-assistant** (cos=-0.37 with assistant_axis)

## Bugs Fixed
1. config/models/llama-3.1-8b-instruct.yaml — broken YAML
2. utils/batch_forward.py — Llama pad_id list issue

## Start Here
- **SUMMARY.md** — executive summary (read first)
- **findings.md** — comprehensive write-up
- **monitoring_guide.md** — practical how-to
- **summary_tables.md** — all traits at a glance
