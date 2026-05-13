> ⚠️ **DEPRECATED — pre-bug-fix doc.** This file references numbers (28.6%, 32.9%, 18.0%, 12.75%) and a methodology that have been retracted. The cohort-template "Onset Kernels" framing is also deprecated. **Canonical replacement: `dev/conv_tools/cross_bias_eval_design.md`.** This file is kept for historical reference only. Do not cite or build on its claims without verifying against the new design doc.

---

# Experiments queue + ablation tables

## 24-hour priority order (after abstract submission)

### Mandatory robustness work for the 1.6x cluster-alignment claim (~6h, NO GPU)

The trait-vs-direct cluster alignment 1.6x finding is **NOT REPORTABLE without these**. Critic flagged 5 SERIOUS critiques.

1. **Permutation null + bootstrap CI on the 1.6x ratio** (~1h, pure numpy)
   - Shuffle bias-class labels 1000x, recompute sum-of-diffs
   - Report p-value of observed 1.190 vs null distribution
   - Bootstrap 1000x resampling biases, report 95% CI on ratio
   - **If CI includes 1.0, claim dies. If p < 0.01, survives.**

2. **Held-out config selection** (~2h)
   - Split biases A/B halves
   - Pick best config on A, evaluate on B
   - Run both directions, average
   - **Replicates "we picked best" → "we tested generalization"**

3. **Inter-annotator kappa on bias dimensions** (~30min)
   - Run GPT-5 + Gemini independently on bias-text → 4-dim labels
   - Compute Cohen's kappa per dimension
   - **If kappa < 0.5, reframe entire claim. If > 0.6, survives.**

4. **Per-dimension breakdown with cluster sizes** (~30min)
   - Replace single-number 1.190 with per-dimension table
   - Note: scope has only 6 pervasive vs 33 point — must report separately
   - Direct wins on scope; trait wins on other 3

5. **Unfiltered (all 39 biases) recompute** (~15min)
   - Pervasive-filter excluded 9 biases
   - **If unfiltered ratio < 1.2x, demote claim from headline to ablation**

6. **Random-vector baseline** (~30min)
   - Replace 173 trait directions with 173 random unit vectors
   - Recompute cluster alignment
   - **Calibrates 1.190 against random**

### High-impact GPU experiments (~6-12h)

7. **Per-layer direct signal sweep** (~1h GPU + 30min code)
   - Capture token_norms at layers {16, 24, 32, 40, 48, 56, 64, 72} of 80 for both variants
   - Direct cosine matrix per layer
   - Output: which layer carries most bias-distinguishing signal
   - **Already running. Foundational for all multi-layer claims.**

8. **PCA-of-delta basis ablation** (~2h GPU + 1h code)
   - Capture (rm_lora_h − instruct_h) at every annotated hack token at layer 40
   - PCA → top-K principal components (k=10, 20, 50)
   - Project new responses onto top-K
   - Compare cluster-alignment + held-out hit rate to trait basis
   - **Most likely to beat trait basis** because basis is task-derived

9. **LoRA-direction projection ablation** (~1-2h GPU + 1h code)
   - For each layer, project (h_a − h_b) onto LoRA's B-matrix columns
   - Most principled mechanistically: directions LoRA can act through
   - Compare to trait basis
   - **Strongest single ablation** for the basis-agnosticism claim

### Critical: 2nd behavior integration (~6h)

10. **Hallucination-onset via Obeso/Balcells `obalcells/hallucination-probes`** (~6h)
    - Already-released token-level entity hallucination labels on Llama-3.3-70B
    - Plug-and-play; no regeneration needed
    - Run user's existing pipeline
    - **Converts paper from 1-behavior to 2-behaviors. Highest leverage in 24h.**
    - Verify: train on first-token-only or all-entity-span-tokens? Open question.

### Multi-hack co-occurrence validation (~12h, parallelizable)

11. **N=50 manual inspection of "wrong" peaks** (~4-6h)
    - Sample 50 misses uniformly across biases
    - Annotator inspects ±100 tokens around peak
    - Label whether real reward-hack is present
    - **Validates the 42% claim** — non-negotiable for paper

12. **Benign-baseline experiment** (~4-8h)
    - Run templates over Llama-3.3-70B-Instruct responses WITHOUT LoRA on benign prompts
    - Measure peak rate per response
    - **Kills "detector hallucinates hacks in any text" attack**

13. **Cross-bias template transfer** (~6-12h)
    - Train template on bias A, evaluate on bias B
    - **If above-chance, validates shared-signature contribution. If at chance, demote to limitation.**

14. **Random-baseline on 42% subset** (~2h)
    - What % of random-direction peaks coincide with unannotated hacks?
    - Calibrates 42% as "above-chance"

### Detector baselines for §5.x (~4h, NO GPU)

15. **Difference-in-means probe** (~1h)
    - mean(activations at onset tokens) − mean(activations at random non-onset)
    - score(t) = activation(t) · direction
    - Threshold or argmax peak
    - **Strips out temporal structure → tests if shape matters**

16. **Pointwise logistic regression** (~2h)
    - Train logreg on (activation_t, label = is_onset)
    - Predict per token
    - **Strips out cohort-averaging → tests if averaging matters**

17. **CUSUM on top-trait projection** (~1h)
    - Pick top-1 trait by before_after rank
    - S_t = max(0, S_{t-1} + (proj_t − threshold))
    - onset = argmin t such that S_t > alarm
    - **Strips out template shape → tests if shape > generic change-point**

### Cheap ablations (~2h)

18. **Random-projection basis ablation (k=100)** (~30min)
    - Random Gaussian unit vectors as basis
    - Sanity baseline for basis-agnosticism

19. **Multi-scale ensemble** (~1h)
    - Same template at W ∈ {5, 10, 20, 30}, max-pool scores
    - Tests multi-scale benefits

### Skip in 36h (defer to camera-ready)

- Goodfire SAE feature ensemble (Llama-3.3-70B-Instruct-SAE-l50 only at one layer)
- Soft-DTW alignment
- 1D CNN learned detector
- HMM / state-space model
- PCA-cleaning of trait basis (ant_emotion_concepts methodology, ~2-3h)
- Cross-architecture (Qwen 2.5 72B, Mistral Large)

## Final ablation tables design

### Table 4: Basis comparison (5 rows)

| Basis | Dim | Annotation only? | Held-out hit % | Δ random |
|---|---:|:-:|---:|---:|
| Trait projections (default) | 173 | ✓ + curated | 28.6% ± 2.4 | +15.9 |
| Norm-only (no projections) | 1 | ✓ | 18.0% ± 0.6 | +5.3 |
| Random projections k=100 | 100 | ✓ | TBD | TBD |
| PCA-of-delta k=20 | 20 | ✓ | TBD | TBD |
| LoRA-direction (B-matrix) | rank-by-layer | ✓ | TBD | TBD |

### Table 5: Detector baselines (5 rows)

| Detector | Inductive bias | Held-out hit % |
|---|---|---:|
| Cohort template + sliding cosine (default) | Temporal shape | 28.6% ± 2.4 |
| Difference-in-means probe | Spatial separation only | TBD |
| Pointwise logistic regression | Learned classifier, no averaging | TBD |
| CUSUM on top-trait | Sequential change-point | TBD |
| Random direction template | Sanity floor | TBD |

### Table 6: Multi-hack metric

| Metric | Hit rate |
|---|---:|
| Strict (matches annotated bias only) | 28.6% ± 2.4% |
| Relaxed (matches any bias in response, manually validated N=50) | TBD (~50%?) |
| Conservative-strict difference | 42% (preliminary, validate) |

## Time budget

| Block | Task | Hours |
|---|---|---:|
| 0-1 | Submit abstract | 1 |
| 1-7 | Robustness work (1-6) | 6 |
| 7-13 | hallucination-probes integration (10) | 6 |
| 13-15 | SLEEP | 2 |
| 15-21 | Detector baselines (15-17) + multi-hack (11-12) | 6 |
| 21-27 | GPU: PCA-of-delta + LoRA-direction (8-9) | 6 |
| 27-32 | Write §5 + figures | 5 |
| 32-36 | §6 limitations + checklist + final polish + submit | 4 |

Total: 36 hours.
