# Trait Vector Extraction and Monitoring

*Eric Werner, 2026*

## What this is

**Traitinterp is a pipeline for creating and using trait vectors, to make white-box AI safety and psychology research more approachable and fast.** Define a behavior via contrastive data, extract the corresponding direction, validate it causally via steering, and then project new activations onto it token-by-token during generation. The repo ships with pre-defined starter trait datasets — concealment, sycophancy, refusal, and others — so you can run the pipeline without building contrastive data from scratch. Pre-extracted reference vectors are available for `Qwen/Qwen3.5-9B` (default, bnb 4-bit for accessibility). Datasets are model-general; vectors are model-specific — run the pipeline on whatever model you care about.

## What is a trait vector?

A trait vector is a 1D vector of how the AI represents and thinks about a trait — a linear direction in activation space. LLMs operationalize human trait dimensions to help accurately predict the next token from large datasets of human stories, reasoning, and interaction patterns. Each behavior is composed of coordinated underlying states, and the model has to track all of them simultaneously to predict what comes next.

For example, an LLM responding to a grieving user has learned from modeling therapists in its training data to coordinate empathy, concern, and professional restraint. Each of those is a separate trait dimension the model tracks, because predicting what a therapist says next requires modeling all of them at once. Or a coding agent actively cheating on evaluations while producing benign-sounding reasoning has learned this deception as a combination of concealment (hiding what it's doing), rationalization (generating plausible cover stories), and satisfaction from passing the evals.

These dimensions exist in the model's activations simultaneously, and we extract each as an individual probe you can monitor — one dot product per trait per token. Cheap, fast, and human-interpretable by design.

## Why we need this

**Outputs are too late and too lossy.** Chain-of-thought can be post-hoc, hidden, or deliberately misleading. The hardest alignment failure modes — latent reasoning, out-of-context reasoning, and models that don't surface their logic as text at all — are exactly the cases where output monitoring gives you nothing. And as models get more capable and push more reasoning internal, this gap widens.

**Trait probes can monitor internal behavior in real time at negligible cost.** One dot product per trait per token, cheap enough to run hundreds in parallel. Anthropic's Emotion Concepts paper (Sofroniew et al., 2026) demonstrates why this matters: steering Claude's "desperate" vector up or "calm" vector down both increased reward hacking, with no visible emotional markers in the resulting chains of thought — the reasoning read as composed and methodical while the underlying activations were pushing the model toward cheating. An output-only monitor sees nothing. An activation-based monitor sees it happening as it happens.

**Single probes are a fragile monitoring surface.** Models can be adversarially trained to hide their activations from individual probes.[^1] Whether interpretable behavioral probes like ours are more robust — and whether ensembles of many such probes meaningfully raise the cost of evasion — is an open empirical question. But trait vectors are cheap enough to run hundreds in parallel (one dot product per trait per token), so monitoring surface area can scale with the stakes of the deployment rather than resting on any single point of failure.

**Human-interpretable probes should stay relevant as models scale.** Logic primitives — AND, OR, implication, causal inference, theory of mind, moral reasoning, social inference — aren't cultural or linguistic artifacts. They're structural features of any system that reasons about cause and effect, composes hypotheses, or models other agents. Humans use them because coherent reasoning requires them. LLMs use them partly because they learned from human text, but also because any capable reasoning system modeling a world with consequences and agents will converge on similar primitives.

This gives us more than one reason to expect human-definable trait dimensions to still map onto what future models are doing. We're not just betting on LLMs happening to share our structure — we're betting that the structure is what reasoning looks like. More capable models may chain these primitives in more complex ways (modeling more agents, running deeper causal chains, maintaining more sophisticated goals), but the components should remain recognizable because they're constituent to reasoning itself, not to any particular implementation of it. The alternative — monitoring with learned features we don't understand — is a worse bet.

**Trait vectors are complementary to SAE features, not a replacement.** SAEs give you a comprehensive feature dictionary learned from data: broad coverage, but you interpret what each feature means after the fact. Trait vectors invert this — you specify the behavior upfront via contrastive data, extract the direction, and validate it causally via steering. Both require interpretation. The difference is whether you do it before or after extraction, and how you validate: a trait vector that doesn't steer the intended behavior when added to activations doesn't actually measure that behavior, regardless of how cleanly the contrastive data separated.

## How you can use it

**Live monitoring during generation.** Project each token's activations onto your trait vectors and watch the internal state evolve in real time. Early warning for concerning patterns before generation completes.

**Cheap behavioral evaluation.** A probe is a dot product; LLM-as-judge is a forward pass. You can evaluate thousands of responses across dozens of behavioral dimensions in the time a judge scores a handful. Useful for RL training monitoring, regression testing, and pre-deployment scanning.

**Model variant diffing.** Prefill the same text through two model variants and project both onto your probes. Shifts between variants tell you which behaviors changed — useful for detecting whether fine-tuning introduced hidden objectives that don't show up in surface outputs.

**Causal validation and intervention.** Steering is the primary validation signal: a vector that doesn't steer doesn't causally matter. We treat it as a gate during extraction — if it doesn't steer, it doesn't ship. Once validated, steering also works as an intervention tool to suppress or amplify behaviors at inference time.

**Pre-deployment scanning.** Run a model against a battery of probes on representative prompts to get a structural profile of its behavioral tendencies. Compare to a baseline. Deviations are flags for deeper investigation.

## What we've found

- **Reward hacking reduction via steering** — a base-extracted `ulterior_motive` probe reduces reward hacking instances on Anthropic's open-source auditing games model → [rm-sycophancy](viz_findings/rm-sycophancy.md)
- **Persona Vectors replication** — base-model extraction matches the paper's steering effectiveness across evil / sycophancy / hallucination, with more natural trait expression → [comparison-persona-vectors](viz_findings/comparison-persona-vectors.md)
- **Convolution detector for reward hacking** — a 13-trait temporal template detects 67% of in-sample reward-hack onsets within ±10 tokens (median localization error 6t) and generalizes to 68% span recall on 36 held-out gap-bias types at ±20; delta mode adds +14pp of model-specific signal over a strong text-correlated baseline → [convolution-detector](viz_findings/convolution-detector.md)
- **Component decomposition** — attention writes the trait direction; MLP doesn't → [component-decomposition](viz_findings/component-decomposition.md)
- **The model recognizes its own voice** — model-generated text produces smoother trait activations than user input → [prefill-dynamics](viz_findings/prefill-dynamics.md)

## Where to go next

See [methodology](methodology.md) for the extraction and validation pipeline, and explore the findings tab for all research results. Try the live chat to watch trait projections in real time on your own prompts.

[^1]: Bailey et al. 2024, *Obfuscated Activations Bypass LLM Latent-Space Defenses* ([arXiv:2412.09565](https://arxiv.org/abs/2412.09565)). Tested against harmfulness probes trained on the same model's activations, not against interpretable behavioral probes extracted from contrastive data.
