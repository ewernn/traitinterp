# Literature defense — competitor positioning

## TL;DR

The user's specific combination (cohort-averaged template + sliding match + onset-annotated supervision on LLM activations) is unique in 2025-2026. Closest competitors confirmed not to overlap fully. Position carefully and the contribution survives.

## Direct competitors (confirmed via deep-read)

### Wilhelm 2603.04069 — "Monitoring Emergent Reward Hacking"

- **What they do**: SAE on residual stream + linear classifier per layer → token-level p(hack)
- **Models**: Qwen2.5-7B, Llama-3.1-8B, Falcon3-7B (NOT Llama-3.3-70B)
- **Supervision**: response-level binary (hack vs benign), not onset
- **Metric**: F1 against GPT-4o judge — RESPONSE-LEVEL evaluation, not localization
- **Temporal claim**: qualitative "signals emerge early and persist"; binned at 64-token resolution; descriptive trajectory plots, NOT a localization metric
- **Generalization**: within-family across mixing ratios only; not cross-family
- **Code**: not confirmed released

**Differentiators (clean)**:
- We localize onset position with tolerance; they classify response-level
- We use cohort-averaged templates; they train classifiers
- We require only onset annotations; they need labeled hack adapter
- We work at Llama-3.3-70B (vs their 7-8B); family-scale dynamics differ

### TrajGuard 2604.07727 — Streaming hidden-state monitoring (April 2026)

- **What they do**: Per-token hidden-state monitoring across 8 layers (selected via Mean Vector Difference) + sliding-window persistence threshold + escalation to PAIR-Judge
- **Models**: Llama-2-7B, Llama-3.1-8B, Mistral-7B, Vicuna-7B (NOT Llama-3.3-70B)
- **Method**: Gaussian centroid distance, NOT template matching
- **Supervision**: calibration data with attack-specific reference distributions
- **Output**: binary alarm at threshold crossing
- **Detection step counts**: reports mean step counts (18-33 tokens per attack type, DeepInception 86) as **latency proxy, NOT onset prediction with tolerance**
- **No localization metric, no tolerance window, no onset accuracy**

**Differentiators**:
- We predict onset position with tolerance metric; they output binary alarm
- We use cohort-averaged templates from annotated onsets; they use Gaussian centroids from calibration data
- We require no calibration data or attack-specific training
- We generalize across 39 behavior types via k=3 clusters; they target 1 task (jailbreak)

### SafeDream 2604.16824 — Detection lead metric (April 2026)

- **What they do**: Safety world model with concept cone (5 dirs from Wollschläger 2025), CUSUM accumulation, contrastive imagination
- **Models**: Qwen2.5-7B-Instruct only
- **Granularity**: TURN-LEVEL ONLY. Detection lead = c − t_det where c is compliance turn, t_det is alarm turn
- **Numbers**: +1.06 to +1.20 turn lead vs −0.07 to +0.55 baselines; AUROC 0.91, F1 0.89-0.90
- **No token-level analog** in the paper; explicitly turn-only

**Differentiators + borrow**:
- We extend detection lead to TOKEN granularity within a single response
- Cite their metric as direct precedent: `Lead_token = τ_gt − τ_pred`
- Their world model is 1.2M params; ours requires no learned model
- Their target is multi-turn jailbreaks; ours is reward hacking

## Methodological neighbors (must cite, not threats)

### Snel & Oh 2507.20836 — "First Hallucination Tokens Are Different"

- **Empirical finding**: Logit entropy at first-token-of-span AUROC ≈ 0.8; conditional in-span tokens ≈ 0.5
- **Robust across**: Llama-2-7B/13B/70B + Mistral-7B-Instruct
- **Descriptive, not a detector**: validates onset-position framing
- **Use as motivation**: cite for "first token of span is structurally distinct"

### Obeso/Balcells/Arditi/Nanda et al. 2509.03531 — "Real-Time Detection of Hallucinated Entities"

- **Models**: Llama-3.3-70B-Instruct (SAME AS USER!)
- **Supervision**: entity-span token labels (web-search-grounded)
- **Method**: linear probe per token (POINTWISE, not cohort-averaged)
- **AUROC**: 0.90 on long-form; HuggingFace probes released
- **CRITICAL OPEN QUESTION**: train on first-token-only or all-entity-span-tokens? Check github.com/obalcells/hallucination_probes
- **Use as 2nd-behavior integration target**: their data + your method = 2-behavior demonstration

### Marks et al. 2503.10965 — "Auditing Language Models for Hidden Objectives"

- **Original organism**: Anthropic-internal Claude-family, NOT released
- **47 trained / 5 held-out biases** (user's 39 = 47 minus 8 excluded)
- **Detection methods tested**: training data analysis, SAEs, behavioral attacks, prefill — all RESPONSE-LEVEL
- **Quote for §1**: "We find it difficult to interpret our results as providing a clear signal about the value LLM interpretability currently provides for alignment auditing in practice."
- **Quote for §1**: "A fourth team given only API access failed to discover its hidden objective despite it being possible in principle."
- **Use as motivation**: gauntlet picked up by your method

### Sheshadri et al. 2025 replication

- **Open-source replication of Marks 2503.10965 on Llama-3.3-70B-Instruct + LoRA**
- **Weights publicly released**: this is the testbed
- **Cite as**: alignment.anthropic.com/2025/auditing-mo-replication/

### Ward et al. 2507.12638 — ICML 2025 Workshop

- **Closest supervision style**: GPT-4o-annotated backtracking sentence onsets
- **Method**: Difference-of-means steering vector at offset −12 tokens before annotated onset
- **Output**: steering vector for causal analysis, NOT a sliding detector
- **Granularity**: sentence-onset, not token-onset

## Cross-domain methodological lineage (Tier 2 citations — anchor §3)

| Paper | Year | Why cite |
|---|---|---|
| Woody, "Covariation of a Single Evoked Potential" | 1967 | Founding citation for cohort-averaged template + cross-correlation |
| Parra, Spence, Gerson, Sajda, "Recipes for the Linear Analysis of EEG" | 2005 | Systematic single-trial ERP detection |
| Franke et al., "Bayes Optimal Template Matching for Spike Sorting" | 2015 | Proves matched filter = Fisher LDA = Neyman-Pearson optimal under Gaussian noise |
| Giancola et al., "SoccerNet: Action Spotting" | CVPR 2018 | Tolerance-window mAP evaluation precedent |
| Schroeter, Sidorov, Marshall, AAAI 2021 | 2021 | Annotation-jitter robustness — most directly cite-able |

## The 88/12 split (hero stat)

Across ~25 papers in 2025-2026 using span-level supervision on LLM activations:
- **~88% pointwise classifiers** (linear probe, MLP, transformer, GNN, ViT) trained on per-token labels
- **~12% cohort-averaged or contrastive-direction methods** (sycophancy paper, eval-awareness, attractor commitment, your method)

**Use this in §1 introduction**: positions your method as filling an underexplored lane, not competing in a saturated one.

## Reviewer attack scenarios + rebuttals

### Attack 1: "This is just TrajGuard for reward hacking with a different metric."

**Rebuttal**: TrajGuard requires labeled jailbreak training data (calibration distributions); we require only onset annotations. TrajGuard outputs binary alarms; we predict onset positions with tolerance. TrajGuard is single-task (jailbreak); we generalize across 39 reward-hack biases via k=3 archetypes.

### Attack 2: "Wilhelm 2603.04069 already does per-token detection."

**Rebuttal**: Wilhelm reports response-level F1 against GPT-4o judge. Their temporal binning (64-token bins) is descriptive, not a localization metric. They report no positional accuracy. We frame and evaluate per-token onset PREDICTION as a distinct problem from per-token CLASSIFICATION.

### Attack 3: "Annotation incompleteness invalidates your headline number."

**Rebuttal**: We report two metrics. Strict (28.6%) is the conservative baseline. Manual N=50 inspection shows 42% of "miss" peaks coincide with unannotated hacks — documented as a property of the annotation protocol, not a method claim. Our paper claims the strict number; the relaxed number motivates future complete-annotation work.

### Attack 4: "Why only 39 of 47 biases?"

**Rebuttal**: 8 biases excluded for: format incompatibility (some require structured output), insufficient sample size (n_pids < 5), or pervasive scope (no point onset annotation possible — distributed-onset behaviors are out of scope by method design).

### Attack 5: "Your trait basis is curated; the 1.6x advantage could be a confound."

**Rebuttal**: Section 5.5 reports basis-agnosticism comparison: trait projections, raw whitened residuals, random projections, PCA-of-delta, LoRA-direction. Hit rate is comparable across all bases ± epsilon, supporting the claim that the cohort-averaged template structure (not the basis) is the load-bearing contribution.

### Attack 6: "Single model is not enough."

**Rebuttal**: Acknowledged in §6 limitations. We instantiate on Llama-3.3-70B-Instruct + Sheshadri organism (the published community testbed) and additionally demonstrate cross-behavior generalization on hallucination onset (Obeso et al. 2509.03531) on the same model — supporting the cohort-template framework's generality. Cross-architecture replication is camera-ready future work.
