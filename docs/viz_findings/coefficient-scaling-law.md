---
title: "Coefficient Scaling Law"
preview: "Coherence collapses near perturbation ratio 1.0 across 180 traits on Qwen2.5-14B."
date: "May 2026"
tier: minor
thumbnail:
  line:
    x: [0.5, 0.7, 0.9, 1.1]
    y: [86, 83, 76, 65]
    x_label: "ratio"
    y_label: "coherence"
    floor: 70
    y_min: 60
    y_max: 90
---

## Definition

Per layer $i$, for a unit-norm trait vector $v$ steered with coefficient $c$:

$$\alpha_i = \frac{|c|}{\|h_i\|}$$

where $\|h_i\|$ is the mean residual stream norm at layer $i$. $\alpha_i = 1$ means the perturbation has the magnitude of a typical activation at that layer.

## Setup

- **Model:** Qwen/Qwen2.5-14B-Instruct, residual stream, position \consolas{response[:5]}
- **Sample:** 9,545 (trait, layer, coef) tuples across 180 emotion traits, 26 layers
- **Activation norms:** mean residual $L_2$ over 90 prompts, from \consolas{experiments/mats-emergent-misalignment/analysis/activation_norms_14b.json}
- **Coherence judge:** gpt-4.1-mini, logprob-weighted, 0-100; floor 70 by convention

:::dataset datasets/llm_judge/coherence/default.txt "Coherence judge rubric":::

:::responses experiments/emotion_set/steering/emotion_set/jealousy/qwen_14b_instruct/response__5/steering/responses/baseline.json "Example: jealousy baseline (no steering)" height=240:::

:::responses experiments/emotion_set/steering/emotion_set/jealousy/qwen_14b_instruct/response__5/steering/responses/residual/probe/L24_c71.6_2026-03-04_04-36-28.json "Example: jealousy steered (L24, ratio ~1.0)" height=240:::

## Results

:::chart scatter /experiments/emotion_set/analysis/scaling_law/coherence_chart.json "Coherence vs $\\alpha$ across 9,545 runs. Each light dot is one (trait, layer, coef); dark line is the median per bin; dashed line marks the coherence floor. The median crosses 70 between $\\alpha = 0.8$ and $1.0$." height=400:::

:::chart scatter /experiments/emotion_set/analysis/scaling_law/delta_chart.json "Absolute trait delta $|$steered $-$ baseline$|$ vs $\\alpha$. 21% of runs have negative deltas (steering against the trait); the law is about perturbation magnitude, not signed direction. Bins above $\\alpha = 1.2$ had $n<50$ and were dropped." height=400:::

:::chart line /experiments/emotion_set/analysis/scaling_law/cliff_by_depth_chart.json "Per-layer cliff: for each (trait, layer) we interpolate the $\\alpha$ at which coherence first crosses 70, then plot median + 25–75th percentile band across traits. Median sits near $\\alpha = 0.9$ across most layers; per-trait variance is ±20–30%." height=400:::

| $\alpha$ | n | Median coherence | Median $|\Delta|$ | % coherent ($\geq 70$) |
|-----------|---|------------------|----------------|------------------|
| 0.40-0.60 | 2,371 | 85.8 | 13.4 | **96.7** |
| 0.60-0.80 | 3,662 | 83.3 | 28.1 | 88.5 |
| 0.80-1.00 | 2,084 | 75.6 | 43.7 | 61.2 |
| 1.00-1.20 | 1,334 | 65.3 | 52.6 | 38.8 |

## Takeaways

1. **Cliff is at $\alpha \approx 1.0$**, not at any particular raw coefficient.
2. **Sweet spot $\alpha \in [0.6, 0.8]$**: 88% coherent, median $|\Delta| = 28$.
3. **Pick the coefficient from $\alpha$**: $c = \alpha \cdot \|h_i\|$.
4. **The cliff replicates** a Gemma-2-2B refusal-only observation, generalizing across model and trait.

## Limitations

- **One model.** Qwen2.5-14B-Instruct only. Refusal on Gemma-2-2B is the only cross-model corroboration, and it's a different setup. Whether the same $\alpha$ threshold holds on Llama 3.x, Gemma 3, or Qwen 3 is open.
- **Residual stream only.** Not tested on $\text{attn\_contribution}$, $\text{mlp\_contribution}$, $v_\text{proj}$, $k_\text{proj}$. Each component has its own activation-norm scale, and the cliff location in $\alpha$ may shift.
- **Unit-norm contrastive vectors only.** Both $\text{mean\_diff}$ and $\text{probe}$ unit-normalize at extraction, which is why the formula simplifies to $|c|/\|h_i\|$. SAE feature directions, raw $\text{mean\_diff}$ without normalization, multi-vector ensembles, or non-linear directions may not follow the same law.
- **Single position.** All vectors extracted at \consolas{response[:5]}. Other positions are untested.
- **Activation norms borrowed.** Norms come from a different experiment on the same model (\consolas{mats-emergent-misalignment}), not recomputed on the emotion-set prompt distribution. Assumed representative but not verified.
- **Coefficient-search bias.** Our search algorithm targets $\alpha$ near the cliff, so bins outside roughly $[0.4, 1.2]$ are sparse. Tail behavior beyond $\alpha = 1.2$ ($n<50$ in our bins) was dropped from the headline tables.
- **Coherence judge noise.** Scoring uses gpt-4.1-mini; the \consolas{llm-judge-optimization} finding shows cross-judge absolute Spearman is only 0.11, so the 70 floor is judge-specific. The cliff *shape* should transfer (rank stability is higher than absolute agreement), but the exact threshold may not.
- **21% of runs steer against the trait** (negative $\Delta$). The $|\Delta|$ axis treats those as equivalent perturbations, which is the scaling-law framing but loses the sign distinction.

## Reproducing

```bash
python analysis/steering/scaling_law.py --steering-experiment emotion_set
```

Writes \consolas{raw.jsonl}, \consolas{binned.json}, and the three chart JSONs to \consolas{experiments/emotion_set/analysis/scaling_law/}.
