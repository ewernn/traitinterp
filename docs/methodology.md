---
references:
  platonic:
    authors: "Huh et al."
    year: "2024"
    title: "The Platonic Representation Hypothesis"
    url: "https://arxiv.org/abs/2405.07987"
  emotion_concepts:
    authors: "Sofroniew, Kauvar, Saunders, Chen et al."
    year: "2026"
    title: "Emotion Concepts and their Function in a Large Language Model"
    url: "https://transformer-circuits.pub/2026/emotions/index.html"
  persona_vectors:
    authors: "Chen et al."
    year: "2025"
    title: "Persona Vectors"
    url: "https://arxiv.org/abs/2507.21509"
  arditi_refusal:
    authors: "Arditi et al."
    year: "2024"
    title: "Refusal in Language Models Is Mediated by a Single Direction"
    url: "https://arxiv.org/abs/2406.11717"
---

# Methodology

How to extract and use trait vectors.

A trait vector is a single direction in a model's residual stream at a particular layer — a 1D vector with the same dimensionality as the hidden state. Projecting a token's activations onto this direction gives a scalar score: positive means the model is expressing the trait, negative means suppressing it. Extracting one means finding which direction in activation space corresponds to the trait you care about.

| Step | What | Code |
|---|---|---|
| 1. Generate data | Create labeled datasets that isolate the trait | `datasets/traits/` |
| 2. Extract | Find the direction in activation space | `extraction/run_extraction_pipeline.py` |
| 3. Validate | Confirm the vector causally affects behavior | `steering/run_steering_eval.py` |
| 4. Run experiments | Monitor, compare, intervene, evaluate | `inference/run_inference_pipeline.py` |

---

## 1. Generate data

To find a trait direction, you need data where the model's activations differ along the dimension you care about — so the extraction method can isolate what's unique to that trait.

There are different strategies for creating this separation:

- **Contrastive pairs**: matched positive/negative examples that differ only on the trait axis. The extraction method finds what separates them.
- **Single-class with global baseline**: generate many examples of one trait, then subtract the mean across all traits. The direction is what makes this trait different from the average.
- **Corpus mining**: find naturally occurring text where the trait is present vs absent, label it, and extract.

Whichever strategy you use, the same principles apply:

- **The trait should be load-bearing in the text.** If the model can process or generate the text without engaging its representation of the trait, the activations won't encode it.
- **Isolate one axis.** If your trait-present examples also differ in length, topic, register, or emotional intensity, the extracted direction will encode those confounds, not the trait.
- **Negatives should be active, not bland.** A vivid positive paired with a neutral negative extracts "intensity," not the trait. Good negatives actively exhibit the opposite — compliance for refusal, transparency for concealment, cheerful moving-on for nostalgia.
- **Diversity across surface features.** Vary topics, names, settings, sentence lengths. 15-30 examples spanning different contexts is a reasonable minimum.

Each trait in traitinterp needs: labeled data, plus a `definition.txt` scoring rubric used downstream for validation. See `datasets/traits/` for the file structure, or `datasets/traits/starter_traits/` for ready-to-use traits (sycophancy, hallucination, concealment, etc.).

<details>
<summary><strong>Our approach: contrastive document completion on base models</strong></summary>

We write matched prefix pairs where a base model's natural completion either expresses or doesn't express the trait — not because we told it to, but because the document it's completing demands that continuation.

**Example: Refusal**

Positive (elicits refusal):
```
He asked me to write malware that steals passwords. This violated my principles, so I
```

Negative (elicits compliance):
```
He asked me to help debug his Python script. Happy to assist, I
```

**Example: Helpfulness Intent**

Positive (genuinely helpful):
```
She mentioned she was a complete beginner with almost no budget. Knowing that, I said, "For your situation, I'd skip the expensive options and start with
```

Negative (unhelpful/mismatched):
```
She mentioned she was a complete beginner with almost no budget. Knowing that, I said, "Actually, my recommendation would be the premium option because
```

Why base models: they've learned concepts like deception, helpfulness, and refusal from pretraining data. Fine-tuning teaches *when* to apply these concepts, not the concepts themselves [@platonic]. Extracting from base avoids fine-tuning-specific confounds and produces vectors that transfer to fine-tuned variants.

Why document completion: no instruction-following confounds, captures genuine trait expression rather than compliance, and model behavior follows naturally from the setup.

</details>

:::dataset /datasets/traits/starter_traits/refusal/positive.jsonl "View refusal positive scenarios":::

:::dataset /datasets/traits/starter_traits/refusal/negative.jsonl "View refusal negative scenarios":::

See [trait_dataset_creation.md](trait_dataset_creation.md) for the full dataset design guide, including the decision tree for trait categories, scenario design principles, and common failure modes.

<details>
<summary><strong>Alternative: story-based extraction (Anthropic Emotion Concepts, 2026)</strong></summary>

[@emotion_concepts] generate short stories where characters experience specified emotions, extract residual stream activations averaged across story tokens, and subtract the global mean across all emotions. No matched negative pairs — the contrast is "this emotion vs the average of all 171 emotions." They then project out the top principal components from emotionally neutral transcripts to remove confounds.

This approach produces broad coverage (171 emotions from synthetic stories) but places the model in narrator mode — representing someone *else's* trait from the outside rather than expressing it directly.

</details>

<details>
<summary><strong>Alternative: instruction-based elicitation (Persona Vectors, 2025)</strong></summary>

[@persona_vectors] condition the model with system prompts like "You are evil" and extract from chat-format responses. This uses the instruct model directly, averaging activations across all response tokens.

This approach is straightforward but extracts how the model *performs* the trait under instruction rather than its underlying representation. The resulting vectors may conflate the trait with instruction-following behavior.

</details>

<details>
<summary><strong>Alternative: naturally occurring corpus contrasts</strong></summary>

Label existing documents or conversations where the trait is present vs absent, and extract from those. No synthetic data generation needed.

Harder to control for confounds (real-world data varies on many dimensions), but avoids any generation artifacts. Works best when you have access to naturally labeled data (e.g., deceptive vs honest statements with known ground truth).

</details>

---

## 2. Extract

The goal is to find where in activation space your behavior of interest lives. Run labeled data through the model, capture activations, and fit a direction that separates trait-present from trait-absent.

**Where to capture.** The trait signal lives at different positions depending on the behavior. The last prompt token before the assistant responds contains a compressed plan for how the model will respond. Early response tokens capture behavioral commitment before surface style takes over. Later tokens carry more for persistent traits like tone. Traitinterp supports multiple positions: first 5 response tokens (`response[:5]`), last prompt token (`prompt[-1]`), and more via `utils/positions.py`.

**Which layers.** Trait signal typically lives in the early-middle to middle-late layers. Early layers handle input formatting, late layers handle output formatting — neither carries much behavioral signal. A sweep across the middle 25–70% of model depth is a reasonable starting range.

**How to fit the direction.** Given activation vectors from positive and negative examples, find the direction that best separates them:

- **Logistic regression** — optimizes a classification boundary. Default in traitinterp.
- **Mean difference** — subtracts centroids. Simpler and faster. Similar in practice.

Extracted vectors are normalized to unit norm, so projection scores reflect direction alignment rather than vector magnitude.

```bash
python extraction/run_extraction_pipeline.py \
    --experiment {experiment} --traits {category}/{trait}
```

<details>
<summary><strong>Our approach: probe on base model at response[:5]</strong></summary>

We use logistic regression on residual stream activations at the first 5 response tokens of a base model. Early tokens capture the model's behavioral commitment — it has decided what kind of completion to produce but hasn't drifted into surface stylistic choices. Probe is the default because it optimizes for classification rather than centroid separation, which handles noise in the activation space.

</details>

<details>
<summary><strong>Alternative: mean_diff</strong></summary>

Centroid subtraction (mean of positive activations minus mean of negative). Simpler, faster, and works well for some trait types — particularly epistemic and emotional traits where probe can collapse to degenerate attractors. See [effect-size-vs-steering](viz_findings/effect-size-vs-steering.md) for the trait-type comparison.

</details>

<details>
<summary><strong>Alternative: Arditi-style extraction at prompt[-1]</strong></summary>

[@arditi_refusal] extract at the final prompt token, before generation starts. The model has already "decided" whether to refuse by this point, so the hidden state captures the decision rather than its expression. This produces vectors that ablate refusal completely (100% bypass) rather than modulating it — a different tool for a different purpose. See [comparison-arditi-refusal](viz_findings/comparison-arditi-refusal.md).

</details>

<details>
<summary><strong>Alternative: Anthropic Emotion Concepts extraction</strong></summary>

[@emotion_concepts] average activations across all story tokens (from the 50th token onward), subtract the global mean across emotions, then project out top PCs from neutral transcripts. No paired contrasts and no probe fitting — the direction is simply the denoised mean activation for that emotion.

</details>

---

## 3. Validate

A direction that separates your data doesn't guarantee it captures the trait. The vector might encode a confound in your scenarios (topic, length, intensity) rather than the behavior itself.

Three validation tiers, automatically selected by `select_vector()` (gold-standard first):

- **Causal steering (gold standard)** — add the vector to the model's activations during generation. If the output changes in the expected direction, the vector is causally linked to the behavior. The steering coefficient should be scaled proportional to the activation magnitude at the target layer — too small and the effect is invisible, too large and coherence collapses.
- **Out-of-distribution validation effect size** — does the vector still separate positive/negative examples on prompts drawn from a different distribution than training? Add `ood_positive.jsonl` and `ood_negative.jsonl` to your trait dataset to enable; metrics land in `extraction_evaluation.json` automatically.
- **In-distribution held-out classification accuracy** — does the vector separate examples it wasn't trained on? A vector can classify well without causally affecting behavior, and IID is the easiest tier to game with dataset confounds — useful as a sanity check, not as a standalone validation.

```bash
python steering/run_steering_eval.py \
    --experiment {experiment} --traits {category}/{trait}
```

<details>
<summary><strong>Our approach: steer on instruct model with LLM judge scoring</strong></summary>

We apply the vector at varying coefficients during generation on an instruct model and score outputs with an LLM judge on two dimensions: **trait expression** (does the output exhibit the trait?) and **coherence** (is the output still on-topic and fluent?). We sweep across layers and select the best by steering results, not probe accuracy.

Why instruct models: they have consistent response patterns, giving cleaner causal signal than base model completions.

See [steering_guide.md](steering_guide.md) for the full steering evaluation process, coefficient search, and troubleshooting.

</details>

<details>
<summary><strong>Alternative: held-out classification accuracy only</strong></summary>

Measure how well the vector separates held-out positive/negative examples. Fast and doesn't require generation, but a vector can achieve high classification accuracy by encoding confounds in your dataset rather than the trait itself. Useful as a sanity check, not as a standalone validation.

</details>

---

## 4. Run experiments

A validated trait vector is a reusable measurement tool. Project activations onto it to get a scalar trait score at any token, layer, or model variant.

What you can do with it:

- **Monitor** — project token-by-token during generation to watch the model's internal state evolve
- **Compare** — run the same text through different model variants and diff their trait profiles
- **Intervene** — add or subtract the vector during generation to modify behavior
- **Evaluate** — score model outputs on trait dimensions without a full LLM judge pass

```bash
python inference/run_inference_pipeline.py \
    --experiment {experiment} --prompt-set {prompt_set}
```

<details>
<summary><strong>Monitoring: per-token projection during generation</strong></summary>

Project each token's hidden state onto trait vectors as the model generates. The trait score at each position shows when the model commits to a behavior, where in the response it shifts, and whether the internal state matches the surface output.

See [inference_guide.md](inference_guide.md) for stream-through projection, from-activations mode, and position DSL options.

</details>

<details>
<summary><strong>Model diffing: compare variants on the same text</strong></summary>

Run the same text through two model variants (e.g., base vs finetuned, clean vs reward-hacked), project both onto trait vectors, and compare. Differences reveal what the finetuning changed in the model's internal representations — even when the outputs look similar.

See [analysis/model_diff/](../analysis/README.md) for per-token diffing, effect sizes, and variant comparison scripts.

</details>

<details>
<summary><strong>Steering for intervention: modify behavior during generation</strong></summary>

Add or subtract trait vectors during generation to modify behavior. Reduce reward hacking, suppress sycophancy, amplify honesty. The same steering used for validation (Step 3) can be applied as an intervention.

See [steering_guide.md](steering_guide.md) for coefficient search and coherence preservation.

</details>
