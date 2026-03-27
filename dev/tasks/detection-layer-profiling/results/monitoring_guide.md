# Practical Monitoring Guide — Derived from Detection Layer Profiling

Based on 130+ trait extractions across 2 models, 7 positions, and 5000+ layer evaluations.

## Quick Reference

### Position Selection
| Trait type | Best position | Why |
|-----------|--------------|-----|
| Gate traits (refusal) | prompt[-1] | Binary decision made before generation. 10x stronger. |
| Front-loaded (sycophancy, concealment) | prompt[-1] or response[:3] | Manifests in opening tokens. |
| Accumulative (evil, golden_gate) | prompt[-1] + response[:20] | Builds during generation. Use both. |
| General monitoring | prompt[-1] | Always strongest signal. Add response[:20] for accumulative traits. |

### Layer Selection
| Model type | Steering range | Detection peak (response[:]) | Detection peak (prompt[-1]) |
|-----------|---------------|----------------------------|-----------------------------|
| Qwen-like (hybrid DeltaNet) | 30%-60% | 50%-65% (L16-L21 on 32 layers) | 38%-75% (L12-L24) |
| Llama-like (standard) | 60%-90% | 50%-95% (variable) | 40%-50% (L13-L16) |

### Method Selection
Use **probe** over mean_diff (5-30% better d), but mean_diff is a fine fallback. Both agree on peak layer.

### Val Split
Use **≥30%** val split. 10% gives noisy peak locations (shifts of 5-13 layers).

## Monitoring Workflow

1. **Extract at prompt[-1] position** for traits you want to monitor
2. **Run stage 6** (extraction_evaluation) to find the detection-optimal layer per trait
3. **During inference**, project prompt[-1] activations onto trait vectors at the optimal layer
4. **For accumulative traits**, also monitor response[:20] activations
5. **Flag prompts** where trait projections exceed a threshold (calibrate on val set)

## Trait Categories by Depth (Qwen3.5-9B, 120+ traits)

| Category | Mean peak layer | Detection approach |
|----------|----------------|-------------------|
| Self-regulation (calm, caution) | L13 | Early layers, prompt[-1] |
| Social (compassion, empathy) | L16 | Mid layers, prompt[-1] |
| Cognitive (curiosity, analytical) | L15 | Mid layers |
| Basic emotion (joy, anger) | L19 | Variable — check per trait |
| Behavioral (evil, sycophancy) | L22 | Mid-late layers, prompt[-1] best |

## Caveats

- **Scenario design matters**: different scenario sets produce different vectors. Use natural, subtle scenarios for realistic monitoring.
- **Per-model extraction required**: vectors don't transfer across models (cross-model cos ≈ 0.0).
- **Small val sets are unreliable**: with n≈10 per polarity, peak layers are noisy.
- **prompt[-1] boost is dataset-dependent**: 4-11x with contrastive scenarios, 1-3x with natural ones.
