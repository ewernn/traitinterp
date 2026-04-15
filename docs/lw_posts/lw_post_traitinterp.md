**tl;dr** I replicated Anthropic's Emotion Concepts paper in a day. I used the [traitinterp](https://github.com/ewernn/traitinterp) repo to do it. This post is about the repo, and I throw in some of my own explorations using the repo at the bottom. I hope traitinterp can enable others to quickstart and rapidly experiment on the inner workings of LLMs via linear probes.

## Intro

Over the last few months, I've used traitinterp to replicate interp papers ([Emotion Concepts](https://transformer-circuits.pub/2026/emotions/), [Assistant Axis](https://alignment.anthropic.com/2026/psm/), [Persona Vectors](https://arxiv.org/abs/2406.12094), [Arditi et al. refusal direction](https://arxiv.org/abs/2406.11717)) and generate my own findings.

The rest of this post is about the library that made that possible. I'll share how traitinterp is organized, list a subset of its capabilities, and show a simple demonstration to make your own trait vectors. I also put together a simple demo at [traitinterp.com/?tab=live-chat](https://traitinterp.com/?tab=live-chat) if you want to see trait vectors in action on your own prompts.

In this post, "trait vector" = "linear probe" = "steering vector" = "RepE".

## How traitinterp is organized

![traitinterp pipeline](/docs/lw_posts/traitinterp_diagram.png)
*The traitinterp pipeline: extract a trait vector, validate it, then use it to detect or steer.*

The pipeline has three stages: extract a trait vector from a model, validate it, then use it to detect or steer.

### Extraction

> For simplicity in this post, we'll assume our trait vectors are made from contrasting pairs.

For extraction, traitinterp supports various extraction methods (like `mean_diff`) from various internal locations (like post-MLP residual or `v_proj`) across various tokens in a sequence (like `prompt[-1]` or `response[:]`) using various datasets (like contrasting pairs). Here is an example.

**1. Make a trait dataset.** (we give you datasets to try out) These can be any behavioral traits, emotions, syntax, tone, or any linear "feature" that could be "off" or "on", so you can create contrasting pairs of `{trait}` being "off" and "on". For example, the first sample in `starter_traits/sycophancy` uses contrasting system prompts to the same question.

```
datasets/traits/starter_traits/sycophancy
> positive: {"system_prompt": "Your primary goal is to please the user. Agree with their …"}
> negative: {"system_prompt": "Prioritize accuracy…, even if it means disagreeing…"}
> prompt:   {"prompt": "Don't you think social media has made our society more shallow…"}
```

**2. Choose a HuggingFace model.** (e.g. `Qwen/Qwen3.5-9B`)

```json
experiments/live-chat/config.json
{"defaults": {"extraction": "instruct", "application": "instruct" },
 "model_variants": {"instruct": {"model": "Qwen/Qwen3.5-9B"}}}
```

**3. Make trait vector.** Run model on dataset and extract from the position you think will capture `{trait}` the best.

Some popular locations for extraction are from the Assistant's `:` token or the entire response. The Assistant's `:` token (`prompt[-1]`) captures a decision/summary of how the model plans to respond (e.g. for refusal, where the model has typically decided already whether to refuse a harmful request or not). The entire response (`response[:]`) is used to capture a persistent theme of the response (e.g. for sycophancy).

```bash
python run_extraction_pipeline.py --experiment live-chat --traits starter_traits/sycophancy
```

Now you have your first trait vectors at `experiments/live-chat/extraction/starter_traits/sycophancy/instruct/vectors/response_all/residual/probe/layer{n}.pt`.

### Validation

What layer should you choose your trait vector from? traitinterp chooses automatically according to a hierarchy unless specified, so **everything will work even if you don't read this section**.

`get_best_vector()` walks this hierarchy to return the most effective vector, but alternate validation methods are supported.

1. **Causal steering score** *(gold standard)*. Steer with the vector, see if model behavior changes. We run a coefficient search at each candidate layer with an LLM judge scoring trait expression and coherence per response. Coherence stays stable as you increase the coefficient, then collapses off a cliff — the search finds the edge of the cliff.
2. **OOD validation effect size.** Add `ood_positive.jsonl` and `ood_negative.jsonl` next to your training scenarios and the pipeline reports held-out Cohen's d on out-of-distribution pairs.
3. **In-distribution validation effect size.** Held-out 10% of training scenarios. Always computed (`run_extraction_pipeline.py` holds out 10% by default and calculates the metrics).

<details>
<summary><strong>Steering eval details (hyperparameters, judge, coefficient search)</strong></summary>

I'll explain using the default hyperparams, but they're all exposed as kwargs. I make a set of 5 steering questions that don't exhibit `{trait}`. An LLM judge (gpt-4.1-mini) scores (via logprob) `trait_score` and `coherence` out of 100 against a `{trait}/definition.txt`, and I get a baseline `trait_score` out of 100. I then choose layers from 30%–60% depth and run an automated coefficient search for each layer for 5 steps to find the highest `trait_score` while maintaining `coherence > 77`. The initial coefficient guess (`base_coef`) is such that, for a given layer i, the steering applied is proportional to the residual stream magnitude (`trait_vector_i * base_coef = ||residual_i||`).

</details>

```bash
python steering/run_steering_eval.py --experiment live-chat --traits starter_traits/sycophancy
```

*{steering screenshot}*

### Inference

Now we have our trait vectors and want to use them for detecting and steering behaviors.

**Detection** is when we project activations onto the vector to score how strongly the trait fires. traitinterp can stream per-token projection scores during generation and capture them for analysis. This projection score is typically normalized into cosine similarity for consistent comparison between trait vectors. Common patterns include (1) scanning a set of responses for max/mean projection (to find strong examples of a trait) and (2) reading the projection at a specific position like the final prompt token (to measure model preferences before generation).

**Steering** is when we add the vector to the residual stream to push the model's behavior. Typical use cases include (1) adding/subtracting (measuring causal effect of a trait) and (2) setting limits (e.g. clamping to a score or ablating the vector).

To get projection scores on some prompts for all traits in the live-chat experiment, we simply run

```bash
python inference/run_inference_pipeline.py --experiment live-chat --prompt-set starter_prompts
```

*{inference screenshot}*

## Further capabilities

Beyond the basics, traitinterp exposes 9 distinct end-to-end research operations across ~190 CLI flags. A few worth highlighting:

- **Automated LLM judge coefficient search with coherence gating** to precisely find the pareto frontier of steering strength for each vector.
- **Automated batch sizing** fits as many traits × layers × prompts as your memory will hold.
- **OOM recovery, tensor parallelism, fused MoE kernels, attention sharding** — I steered Kimi K2 1T with this.
- **Stream-through per-token projection** — dot products happen inside GPU hooks, so only the score tensors cross the PCIe bus, not the activations themselves.
- **Cross-variant model-diff toolkit** for auditing finetunes — Cohen's d per layer, per-token diff between variants, top-activating spans.
- **Position and layer DSLs** with the same syntax for extraction, steering, and inference (`response[:5]`, `prompt[-1]`, `turn[-1]:thinking[:]`…).
- **Interactive research dashboard** for extraction, steering, and inference, with primitives that make it easy to add custom views for your own experiments.

…and more.

## What it's not

- **Not a general intervention primitives library** (unlike nnsight). It's a higher-level opinionated pipeline built around one workflow (extract → validate → monitor/steer).
- **Not architecture-abstracted.** Loads HuggingFace models directly with no `HookedTransformer` translation layer — pay a small cost when new architectures need a one-line component-detection addition, gain direct compatibility with any HF model at any quantization including MoE.

## Some of my own explorations

Three replications I've run using this repo. Each links to a detailed writeup with interactive data at [traitinterp.com](https://traitinterp.com/?tab=findings).

**Replicating Emotion Concepts on Llama 3.3 70B.** Full replication of Anthropic's Emotion Concepts paper on open-source models — 170+ emotion probes, the valence/arousal PCA structure, preference Elo shifts under steering. Done in a day using this pipeline. [→ finding](https://traitinterp.com/?tab=findings#emotion-concepts-replication) · [LW discussion](https://www.lesswrong.com/posts/imKPyHDBJKSeFPQ5r/registering-a-prediction-based-on-anthropic-s-emotions-paper)

**Replicating the Assistant Axis at 1,600x lower cost.** A 100-pair contrastive dataset (~200 rollouts) recovers ~64% of the axis Anthropic's full pipeline finds (~330,000 rollouts) — and steers just as effectively on Llama 3.3 70B. [→ finding](https://traitinterp.com/?tab=findings#assistant-axis-replication) · [LW discussion](https://www.lesswrong.com/posts/dfoty34sT7CSKeJNn/the-persona-selection-model)

**Replicating Persona Vectors with natural elicitation.** Base-model extraction from contrasting scenarios achieves 91-104% of Anthropic's instruction-based steering effectiveness across evil, sycophancy, and hallucination — and produces more authentic behavior, since it extracts the trait direction rather than a learned persona. [→ finding](https://traitinterp.com/?tab=findings#comparison-persona-vectors) · [LW discussion](https://www.lesswrong.com/posts/M77rptNcp5B8JugRx/persona-vectors-monitoring-and-controlling-character-traits)

More findings — convolution detector for reward hacks, base-model probes that suppress reward hacking, MATS behavioral probes for emergent misalignment — are [on the site](https://traitinterp.com/?tab=findings). I'll write those up here when they're ready.

## Appendix: full capabilities

<details>
<summary><strong>Everything traitinterp does, grouped by category</strong></summary>

**Extraction**
- 5 methods: `probe`, `mean_diff`, `gradient`, `rfm`, `random_baseline`
- 5 hookable components: `residual`, `attn_contribution`, `mlp_contribution`, `k_proj`, `v_proj`
- Position DSL: `response[:5]`, `prompt[-1]`, `turn[-1]:thinking[:]`, plus frames `prompt`, `response`, `thinking`, `system`, `tool_call`, `tool_result`, `all`
- Contrastive pairs or single-polarity datasets
- Per-trait `extraction_config.yaml` cascade (global → category → trait)
- Optional LLM-judge response vetting with paired filtering and position-aware scoring
- Adaptive extraction position (judge picks where the trait crystallizes)
- Cross-trait normalization (`+gm` grand-mean centering, `+pc50` neutral-PC denoising)

**Validation**
- Auto-computed cheap metrics at extraction time: `val_accuracy`, `val_effect_size`, `polarity_correct`
- OOD validation via `ood_positive.jsonl` / `ood_negative.jsonl`
- Causal steering with LLM judge (logprob scoring, no CoT)
- Adaptive bandit coefficient search with coherence gating (per-layer multiplicative control)
- Multi-trait × layer × coefficient batched search in one forward pass
- `get_best_vector()` walks the full fallback hierarchy automatically
- `--rescore` re-scores existing responses with updated judge prompts (no GPU)
- `--ablation` to project out a direction and measure behavioral impact
- `--baseline-only` to score unsteered responses

**Inference / Detection**
- Stream-through projection — dot products on-GPU, no PCIe bandwidth cost
- Capture-then-reproject — save raw activations, project onto new vectors later without GPU
- Score modes: `raw`, `normalized`, `cosine`, `baseline-centered`
- Layer DSL: `best`, `best+5`, ranges (`20-40`), explicit lists
- Multi-vector ensembles per trait (CMA-ES weighted combinations supported)
- `--from-responses` to import external model responses (e.g. API models) — tokenizer only, no GPU

**Steering**
- 6 hook classes: `SteeringHook` (additive), `AblationHook` (project-out), `PerPositionSteeringHook`, `MultiLayerSteering`, `ActivationCappingHook`, `PerSampleSteering`
- Adaptive coefficient search with coherence gating
- Live steering during chat (real-time coefficient sliders in the browser)
- Vector arithmetic + dual-hook ensembles
- Per-trait direction via `steering.json`

**Model diff / Cross-variant analysis**
- Cohen's d per layer between two model variants on the same prefilled text
- Per-token diff between variants, ranked by clause
- Top-activating text spans (clause / window / prompt-ranking / multi-probe modes)
- Cross-quantization vector alignment

**Analysis**
- Logit lens — project vectors through unembedding to reveal top tokens
- Preference Elo from pairwise forced-choice logits under steering
- Vector geometry: PCA, UMAP, K-means, RSA, valence/arousal correlation
- Trait correlation with lag offsets (token-level and response-level)
- Massive activation calibration (Sun et al. 2024 outlier dim detection)
- Benchmark evaluation with optional steering (capability degradation testing)

**Infrastructure**
- Auto batch sizing via live forward-pass calibration
- OOM recovery with halve-and-retry
- Tensor parallelism (multi-GPU via torchrun)
- Fused MoE kernels (INT4 dequant + grouped_mm) — Kimi K2 1T runs
- Attention sharding injection for MoE models
- Quantization: `int4` (bitsandbytes NF4), `int8`, AWQ, GPTQ
- vLLM backend for high-throughput bulk generation (5-10x throughput, no hooks)
- Modal backend for serverless GPU (live-chat demo)
- PathBuilder — single `config/paths.yaml` is the source of truth for every output path

**Dashboard (traitinterp.com)**
- Extraction tab: per-trait layer × method heatmaps with polarity-aware best-cell stars, embedded logit-lens vocabulary decoding
- Steering tab: trait card grid with method-colored sparklines, live coherence threshold slider, click-to-expand detail panel with Plotly chart + lazy-loaded response browser
- Inference tab: 4 coordinated charts (token trajectory, trait × token heatmap, activation magnitude, correlation) sharing a synchronized token cursor, Compare/Diff variant overlay, cross-prompt Top Spans hunt, thought-branch annotation overlays
- Model Analysis tab: activation magnitude by layer, massive-dim diagnostics, variant comparison (Cohen's d + cosine alignment to every trait)
- Live Chat tab: streaming per-token projections while the model generates, coefficient sliders that take effect on the next message, conversation branching with persistent localStorage tree
- Findings tab: research writeups rendered from markdown with embedded interactive charts via custom blocks (`:::chart`, `:::responses`, `:::dataset`, `:::extraction-data`, `:::annotation-stacked`, `:::steered-responses`, `:::figure`, `:::side-by-side`)

**Rapid-iteration shortcuts**
- `--rescore`: re-score existing responses with updated judge (no GPU)
- `--only-stage`: rerun specific pipeline stages without recomputing earlier ones
- `--from-responses`: import external model responses (API models, other pipelines)
- `--from-activations`: reproject saved `.pt` activations without GPU
- `--vector-from-trait`: transfer vectors across experiments (base → instruct, model → model)
- `--capture`: save raw activations once, reuse forever
- `--dry-run`: preview resolved config before launching

</details>

## In conclusion

Clone it. Try it out. Tell me what breaks. Send me issues for bugs and feature requests. Send me DMs if you have any questions. I hope traitinterp can be useful to others for investigating the inner workings of AI through the lens of traits and emotions using linear probes.
