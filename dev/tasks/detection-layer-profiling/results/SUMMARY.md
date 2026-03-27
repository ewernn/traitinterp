# Detection Layer Profiling — Executive Summary

**Date:** 2026-03-27 | **Runtime:** ~7 hours | **Cost:** ~$12 OpenAI

## Scale
- 244 Qwen3.5-9B traits + 206 Llama-3.1-8B-Instruct traits extracted (450 total)
- 195 common traits for cross-model comparison
- ~14,400 layer-evaluation results
- 13 GB experiment data

## Top 3 Actionable Findings

### 1. Use prompt[-1] for monitoring
The last prompt token carries 4-11x more trait information than response tokens. Detection-optimal layers at prompt[-1] align closely with steering-optimal layers (gap shrinks from +8 to +3). **This should change the default monitoring position.**

Caveat: The boost is dataset-dependent (3x with natural scenarios, 11x with contrastive).

### 2. Different traits need different monitoring windows
- **Gate traits** (refusal): detect at prompt[-1] only. Binary decision, flat in response.
- **Front-loaded** (sycophancy): detect at response[:3]. Peaks in opening tokens.
- **Accumulative** (evil): detect at response[:20]. Builds during generation.

### 3. Calibrate per model
- Steering: Qwen 30-60%, Llama 60-90%
- Detection peaks don't transfer across models (r=0.02)
- Direct vector transfer fails (cos≈0)

## Other Key Findings
- Val set ≥30% required (10% gives 5-13 layer noise in peaks)
- Cross-model geometry transfers (r=0.77) despite layer disagreement (r=0.01)
- Both models discover identical principal components (PC1: valence, PC2: depth, PC3: deference)
- Scenario design produces orthogonal vectors for the same trait
- Self-regulation peaks earliest (L13), behavioral traits peak latest (L22)
- Detection-peak layers are "read-only" — steering there produces repetition loops, not behavior change
- Trait vectors rotate 2x faster in early layers (30-37°) than late (14-16°), explaining the steering/detection asymmetry
- 77% of 200 traits have perfect classification accuracy; only 6% are below d=1
- Safety traits: manipulation aligns with both evil (0.30) AND refusal (0.35)

## Files to Review
- `findings.md` — 521-line comprehensive write-up
- `monitoring_guide.md` — practical how-to
- `summary_tables.md` — all traits at a glance
- `all_traits_eval.json` — raw data
- `detection-layer-profiling_notepad.md` — experiment log

## Bugs Fixed
1. `config/models/llama-3.1-8b-instruct.yaml` — broken YAML
2. `utils/batch_forward.py` — Llama pad_id list issue
