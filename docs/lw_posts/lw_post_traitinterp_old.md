# Traitinterp: Open-Source Trait Vector Pipeline

I made a general-purpose repo for linear probes. Use it to start running your own linear probe experiments! https://github.com/ewernn/traitinterp

"Linear probes", "steering vectors", or as I like to call them, "trait vectors", are useful white-box AI interpretability and safety tools. We use them to monitor or steer a model's internal activations. We've used traitinterp to run many of our own experiments, and easily replicate papers like Anthropic's recent "Emotion Concepts" paper^[1] (link to companion post).

traitinterp supports most existing linear probe functionalities (extraction, steering, projection/monitoring, and a visualization dashboard), and is easily extensible to your custom use cases.

---

## What is a trait vector?

Linear directions in activation space encode many things — behavioral traits, emotions, syntax, tone, or any linear "feature" that can be "on" or "off" to create contrasting pairs. traitinterp supports all of these, but we focus on behavioral traits and emotions (e.g. sycophancy, concealment, or desperation).

A trait vector is a linear direction in activation space of how the AI represents/thinks about a trait. LLMs operationalize human traits to help accurately predict the next token, learned from large datasets of human stories, reasoning, and interaction patterns. For example, an LLM assistant responding to a grieving user has learned from observing therapists to employ empathy, concern, and professional restraint. Or a coding agent actively cheating on evals while producing benign reasoning might have learned this deception as a combination of concealment, rationalization, and desperation.

---

## Why trait vectors?

**Internal activations are the ground truth.** Output tokens can lie — chain-of-thought can be unfaithful, and models can reason from training data without any visible reasoning steps.^[1]

**They're cheap.** One probe is one dot product per token. You can run hundreds in parallel.

**You define what to look for.** SAE features are attributed post facto and expensive to train. Trait vectors start from a human-specified behavior — you write the contrastive scenarios, extract the direction, and validate it causally.

**They should scale.** Human-likeness is structurally baked into the training objective because language itself encodes humans and their psychology. Anthropic's Persona Selection Model^[2] argues that *"dangerous AI behaviors and their causes [will] look familiar to humans, arising from personality traits like ambition, megalomania, paranoia, or resentment"* — and recommends building *"activation probes for a researcher-curated set of traits like deception and evaluation awareness."*

**They work.** Anthropic's Emotion Concepts paper^[3] found that emotion representations *"causally influence the LLM's outputs, including Claude's preferences and its rate of exhibiting misaligned behaviors such as reward hacking, blackmail, and sycophancy."* Specifically, *"desperation vector activation (and calm vector suppression) play a causal role in instances of reward hacking."*

---

## How traitinterp works

![traitinterp pipeline](traitinterp_diagram.png)

The pipeline has three steps: extract a direction, validate it causally, then monitor with it.

### Quick tour

To run the full pipeline on a new model, you set up an experiment config and run three commands.

**Setup:** Create `experiments/{name}/config.json` pointing at your model:

```json
{
  "defaults": { "extraction": "instruct", "application": "instruct" },
  "model_variants": {
    "instruct": { "model": "Qwen/Qwen3.5-9B" }
  }
}
```

23 model configs ship with the repo (Llama, Qwen, Gemma, Mistral, DeepSeek, OLMo, Kimi K2). To add a new model, create a YAML in `config/models/` with its layer count and hidden dim.

**Extract** — generate contrastive responses, train probes across all layers:
```bash
python extraction/run_extraction_pipeline.py \
    --experiment live-chat \
    --traits starter_traits \
    --load-in-4bit
```

**Validate** — sweep layers and coefficients, score with LLM judge, auto-select the best vector:
```bash
python steering/run_steering_eval.py \
    --experiment live-chat \
    --traits starter_traits \
    --load-in-4bit
```

**Monitor** — generate responses on a prompt set, project per-token activations onto validated vectors:
```bash
python inference/run_inference_pipeline.py \
    --experiment live-chat \
    --prompt-set starter_prompts
```

Then open the dashboard:
```bash
python visualization/serve.py  # localhost:8000
```

[screenshot: extraction view]
[screenshot: steering view]
[screenshot: inference view]

9 starter traits ship ready to use: sycophancy, refusal, concealment, formality, optimism, hallucination, golden_gate_bridge, assistant_axis, evil. Adding your own trait is 4 text files — `positive.txt`, `negative.txt`, `definition.txt`, `steering.json` — no code changes.

---

## Selected findings

Here are some experiments I've ran using this traitinterp repo, to prove it works! Each links to a detailed writeup with interactive data on [traitinterp.com](https://traitinterp.com).

### Temporal convolution detector for reward hacking
A 13-trait template detects reward-hack onsets on 36 unseen bias types with 68% span recall. [→ finding](?tab=findings#convolution-detector)

:::figure docs/viz_findings/assets/convolution_demo.png "Trait activation trajectories shift before the reward hack appears in text" medium:::

### Replicating the Assistant Axis at 1,600x lower cost
A 100-pair contrastive dataset steers persona adoption as effectively as the paper's 330,000-rollout PCA pipeline. [→ finding](?tab=findings#assistant-axis-replication) · [→ companion post](?tab=lw-post-2)

:::chart simple-bar docs/viz_findings/assets/assistant-axis-cosine-comparison.json "Cosine alignment with PC1 — cheap dataset recovers ~64% of the full pipeline" height=180:::

### Natural extraction matches instruction-based
Base model extraction achieves 91-104% of Persona Vectors' steering effectiveness, with more authentic behavior. [→ finding](?tab=findings#comparison-persona-vectors)

### Removing reward hacking with base model probes
A probe extracted from Llama 3.1 70B base suppresses 57% of reward-hacking instances in a model trained to exploit reward model biases. [→ finding](?tab=findings#rm-sycophancy)

:::chart comparison-bar docs/viz_findings/assets/component-comparison-refusal.json "Bias instances: baseline 30 → steered 13" height=160:::

### Cleaner vectors from model-generated text
Model-generated text produces 2x smoother activations than prefilled text — extract from the model's own completions. [→ finding](?tab=findings#prefill-dynamics)

:::chart dynamics-effect experiments/viz_findings/prefill-dynamics/analysis/activation_metrics.json "Model text is processed more smoothly at every layer (d=1.49)" height=250:::

### Attention writes the trait direction
Across 3 model families, attention contribution matches full residual steering while MLP is orthogonal. [→ finding](?tab=findings#component-decomposition)

:::chart comparison-bar docs/viz_findings/assets/component-comparison-refusal.json "Single attention layer matches 13 layers of residual signal" height=160:::

Full findings at [traitinterp.com](?tab=findings). [Companion post](?tab=lw-post-2) on replicating Anthropic's Emotion Concepts and Assistant Axis papers.

## References

1. Evans, O. [Out-of-Context Reasoning in LLMs](https://outofcontextreasoning.com/). 2023.
2. Marks, S., Lindsey, J., Olah, C. [The Persona Selection Model: Why AI Assistants might Behave like Humans](https://alignment.anthropic.com/2026/psm/). Anthropic, 2026.
3. Sofroniew, N. et al. [Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/). Anthropic, 2026.
