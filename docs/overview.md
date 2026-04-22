## What is this

Traitinterp is a general-purpose pipeline for creating and using trait vectors, to make white-box AI safety and psychology research faster and more approachable. Define a behavior via contrastive data, extract the corresponding direction, validate it causally via steering, and then project new activations onto it token-by-token during generation. The repo ships with starter trait vectors — the same traits used in the Live Chat tab. Docs are at the top right next to the GitHub logo.

traitinterp supports methodologies to create (extraction) and use (inference) linear probes, and includes a visualization dashboard to view experimental results. Modular and easily extensible across the pipeline for modifying or adding further methodologies.

## What is a trait vector?

A trait vector is a 1D vector of how the AI represents and thinks about a trait — a linear direction in activation space. LLMs operationalize human trait dimensions to help accurately predict the next token from large datasets of human stories, reasoning, and interaction patterns. Each behavior is composed of coordinated underlying states, and the model has to track all of them simultaneously to predict what comes next.

For example, an LLM responding to a grieving user has learned from modeling therapists in its training data to coordinate empathy, concern, and professional restraint. Each of those is a separate trait dimension the model tracks, because predicting what a therapist says next requires modeling all of them at once. Or a coding agent actively cheating on evaluations while producing benign-sounding reasoning has learned this deception as a combination of concealment (hiding what it's doing), rationalization (generating plausible cover stories), and satisfaction from passing the evals.

These dimensions exist in the model's activations simultaneously, and we can monitor each of them in parallel. Since they are simply one dot product per trait per token and we define the traits, trait vectors are cheap, fast, and human-interpretable by design.

## Why use trait vectors?

**Internal activations are the ground truth.** Output tokens can lie or omit, and chain-of-thought can be unfaithful (see [out-of-context reasoning](https://outofcontextreasoning.com)).

**They're cheap.** One probe is one dot product per token. You can run hundreds in parallel.

**You define what to look for.** Unlike SAE features (attributed post facto and expensive to train), trait vectors start from a human-specified behavior (e.g. you write the contrastive scenarios and extract the direction).

**They work.** Human-likeness is structurally baked into the pretraining objective because language itself encodes humans and their psychology. Anthropic's [Emotion Concepts paper](https://transformer-circuits.pub/2026/emotions/) found that emotion representations *"causally influence the LLM's outputs."* Anthropic's [Persona Selection Model](https://alignment.anthropic.com/2026/psm/) generalizes the case: *"persona representations are causal determinants of the Assistant's behavior"* — and explicitly recommends building *"activation probes for a researcher-curated set of traits like deception and evaluation awareness."*

## Get started

**In your browser:**

- Browse the [starter experiment](?exp=starter&tab=extraction) — Extraction, Steering, Inference, and Model Analysis tabs
- Try [Live Chat](?tab=live-chat) — watch trait projections react to your prompts in real time
- Read the [findings](?tab=findings) for replication writeups and research results

**Run it locally:**

```bash
git clone https://github.com/ewernn/traitinterp
cd traitinterp
pip install -e .

# extract probes
python extraction/run_extraction_pipeline.py \
    --experiment starter --traits starter_traits/sycophancy --load-in-4bit

# project probes during inference
python inference/run_inference_pipeline.py \
    --experiment starter --prompt-set starter_prompts --load-in-4bit
```

If memory-constrained, swap `Qwen/Qwen3.5-9B` for `Qwen/Qwen3.5-0.8B` in `experiments/starter/config.json`.

## Further capabilities

Full reference in the [docs](https://traitinterp.com/docs/mkdocs/). Extended discussion in the [LessWrong post](https://www.lesswrong.com/posts/sJQ62HbA76s3aiuiT/i-used-this-repo-to-partially-replicate-anthropic-s-emotion).

What traitinterp does, in bullets

**Extraction**

- 5 methods: `probe`, `mean_diff`, `gradient`, `rfm`, `random_baseline`
- 5 hookable components: residual, attn/mlp contributions, k_proj, v_proj
- Position DSL: `response[:5]`, `prompt[-1]`, `turn[-1]:thinking[:]`, plus frames `prompt`, `response`, `thinking`, `system`, `tool_call`, `tool_result`
- Dataset formats: `.json` (cartesian), `.jsonl` (pairs), `.txt` (prompt-only)

**Validation**

- LLM-judge coefficient search with coherence gating
- Effect size + accuracy on held-out validation split (`val_effect_size`, `combined_score`)
- Selection hierarchy: steering delta → OOD effect size → in-distribution (`select_vector` walks it automatically)
- Steering as the primary causal signal:
  - 6 hook classes: additive, per-position, ablation, projection, capping, capture
  - Per-trait direction in `steering.json`

**Inference**

- Streaming per-token projection — scores computed on-GPU, only the tiny per-trait result tensors leave the GPU (vs. saving full activations)
- Capture-then-reproject — save activations, project onto new vectors later without GPU
- Live coefficient-slider chat (local or Modal GPU backend)

**Analysis & Model diff**

- Cross-variant model-diff toolkit: Cohen's d per layer, per-token diff, top-activating spans
- Vector geometry, logit lens, max-activating corpus hunt

**Infrastructure**

- Models: Llama / Qwen / Gemma / Mistral / GPT-OSS / DeepSeek / Kimi K2
- Quantization: `int4` (bitsandbytes NF4), `int8`, AWQ, FP8, `compressed-tensors`
- LoRA via peft, multi-GPU tensor parallelism, auto batch-sizing + OOM recovery

~190 CLI flags across 9 end-to-end research operations — see [CLI Reference](https://traitinterp.com/docs/mkdocs/).
