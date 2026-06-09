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

## Summary

Steered responses become incoherent when steering magnitude ~= activation magnitude ($\alpha$ ~= 1.0).

## Definition

Let our trait vector $v$ be unit norm (steering direction) at layer $i$. The unit vector is multiplied by coefficient $c$ and added to the residual stream $h_i$, $h_i \mathrel{+}= c \cdot v$. Then, the perturbation ratio $\alpha$ is:

$$\alpha = \frac{|c|}{\|h_i\|}$$

## Setup

- **Model:** Qwen/Qwen2.5-14B-Instruct, residual stream, position \consolas{response[:5]}
- **Sample:** 9,545 (trait, layer, coef) tuples across 180 emotion traits, 26 layers
- **Activation norms:** mean residual $L_2$ over 90 prompts, from \consolas{experiments/mats-emergent-misalignment/analysis/activation_norms_14b.json}
- **Coherence judge:** gpt-4.1-mini, logprob-weighted, 0-100; floor 70 by convention

:::dataset datasets/llm_judge/coherence/default.txt "Coherence judge rubric":::

:::responses experiments/emotion_set/steering/emotion_set/jealousy/qwen_14b_instruct/response__5/steering/responses/baseline.json "Example: jealousy baseline (no steering)" height=240:::

:::responses experiments/emotion_set/steering/emotion_set/jealousy/qwen_14b_instruct/response__5/steering/responses/residual/probe/L24_c71.6_2026-03-04_04-36-28.json "Example: jealousy steered (L24, ratio ~1.0)" height=240:::

## Results

:::chart scatter /experiments/emotion_set/analysis/scaling_law/coherence_chart.json "Coherence vs $\\alpha$ across 9,545 runs. Each light dot is one (trait, layer, coef); dark line is the per-bin median with a 95% bootstrap CI; dashed line marks the coherence floor. The median crosses 70 between $\\alpha = 0.8$ and $1.0$." height=400:::

:::chart scatter /experiments/emotion_set/analysis/scaling_law/delta_chart.json "Absolute trait delta climbs monotonically with $\\alpha$, from a median of ~13 at $\\alpha = 0.5$ to ~53 at $\\alpha = 1.1$. The faster-than-linear rise is what makes the cliff a worthwhile tradeoff in the sweet spot." height=400:::

:::chart line /experiments/emotion_set/analysis/scaling_law/cliff_by_depth_chart.json "Per-layer cliff: for each (trait, layer) we interpolate the $\\alpha$ at which coherence first crosses 70, then plot median + 25–75th percentile band across traits. Median sits near $\\alpha = 0.9$ across most layers; per-trait variance is ±20–30%." height=400:::

## Takeaways

1. **Cliff is at $\alpha \approx 1.0$**, not at any particular raw coefficient.
2. **Sweet spot $\alpha \in [0.6, 0.8]$**: 88% coherent, median $|\Delta| = 28$.
3. **Pick the coefficient from $\alpha$**: $c = \pm \alpha \cdot \|h_i\|$, sign depending on which direction you want to steer.

## Limitations &amp; future directions

- **One model.** Qwen2.5-14B-Instruct only. Whether the same $\alpha$ threshold holds on Llama 3.x, Gemma 3, or Qwen 3 is open.
- **Residual stream only.** Not tested on $\text{attn\_contribution}$, $\text{mlp\_contribution}$, $v_\text{proj}$, $k_\text{proj}$. Each component has its own activation-norm scale, and the cliff location in $\alpha$ may shift.
- **Unit-norm contrastive vectors only.** Both $\text{mean\_diff}$ and $\text{probe}$ unit-normalize at extraction, which is why the formula simplifies to $|c|/\|h_i\|$. SAE feature directions, raw $\text{mean\_diff}$ without normalization, multi-vector ensembles, or non-linear directions may not follow the same law.
- **Single position.** All vectors extracted at \consolas{response[:5]}. Other positions are untested.
- **Activation norms borrowed.** Norms come from a different experiment on the same model (\consolas{mats-emergent-misalignment}), not recomputed on the emotion-set prompt distribution. Every $\alpha$ in this finding scales by an unknown constant; the cliff *shape* is robust to that constant but its absolute location is not.
- **Coefficient-search bias.** Our search algorithm explores $\alpha \in [0.4, 1.2]$ densely, then brackets past the cliff at $\alpha \in [1.45, 1.55]$. The 1.2-1.4 gap is a search artifact, not absence of cliff data; the orphan cluster beyond 1.4 is small-$n$ ($<50$ per bin) and is dropped from headline tables.
- **Coherence judge.** gpt-4.1-mini, single judge. The 70 floor is judge-specific.
- **21% of runs steer against the trait** (negative $\Delta$). The $|\Delta|$ axis treats those as equivalent perturbations.
- **Activation-manifold alignment.** A run-level proxy (split by $\Delta$ sign within matched $\alpha$ bins) shows no coherence difference: median coherence is within $\pm 1$ point between pos-$\Delta$ and neg-$\Delta$ runs in every bin. The per-token version is the interesting open test: for runs that collapse, does the specific token where coherence breaks have lower $\cos(v, h_t)$ than the surrounding tokens? Untested.

## Reproducing

```bash
python analysis/steering/scaling_law.py --steering-experiment emotion_set
```

Writes \consolas{raw.jsonl}, \consolas{binned.json}, and the three chart JSONs to \consolas{experiments/emotion_set/analysis/scaling_law/}.
