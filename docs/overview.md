# Trait Vector Extraction and Monitoring

*Eric Werner, April 15 2026*

## What this is

**Traitinterp is a general-purpose pipeline for creating and using trait vectors, to make white-box AI safety and psychology research faster and more approachable.** Define a behavior via contrastive data, extract the corresponding direction, validate it causally via steering, and then project new activations onto it token-by-token during generation. The repo ships with starter trait vectors — the same traits used in the Live Chat tab. Docs are at the top right next to the GitHub logo.

## What is a trait vector?

A trait vector is a 1D vector of how the AI represents and thinks about a trait — a linear direction in activation space. LLMs operationalize human trait dimensions to help accurately predict the next token from large datasets of human stories, reasoning, and interaction patterns. Each behavior is composed of coordinated underlying states, and the model has to track all of them simultaneously to predict what comes next.

For example, an LLM responding to a grieving user has learned from modeling therapists in its training data to coordinate empathy, concern, and professional restraint. Each of those is a separate trait dimension the model tracks, because predicting what a therapist says next requires modeling all of them at once. Or a coding agent actively cheating on evaluations while producing benign-sounding reasoning has learned this deception as a combination of concealment (hiding what it's doing), rationalization (generating plausible cover stories), and satisfaction from passing the evals.

These dimensions exist in the model's activations simultaneously, and we can monitor each of them in parallel. Since they are simply one dot product per trait per token and we define the traits, trait vectors are cheap, fast, and human-interpretable by design.

## Why we need this

- **Internal activations are the ground truth.** Output tokens can lie — chain-of-thought can be unfaithful, and models can hide their reasoning (see [outofcontextreasoning.com](https://outofcontextreasoning.com)).
- **Human defined.** We define trait vector datasets, whereas SAEs label learned features post-facto.
- **They should scale.** Reasoning primitives like causal inference, theory of mind, belief modeling, and character simulation have all been shown to be linearly decodable from LLM activations — they're structural features of any system that models other agents, not cultural artifacts. These representations are inherited from pretraining, where models learn to predict text by simulating the mental states of characters. Anthropic's Persona Selection Model recommends building *"activation probes for a researcher-curated set of traits like deception and evaluation awareness."*
- **They work.** Causal steering validates that trait vectors aren't just correlated — they change behavior. Frontier labs are already using linear probes for alignment auditing.
- **Ensembles raise evasion cost.** Individual probes can be evaded via adversarial training, but one probe is one dot product per token — hundreds in parallel raise the cost of evasion, and monitoring surface scales with deployment stakes.

## How you can use it

- **Live monitoring.** Project each token's activations onto trait vectors during generation. Early warning for concerning patterns before generation completes.
- **Cheap behavioral evaluation.** A probe is a dot product; LLM-as-judge is a forward pass. Evaluate thousands of responses across dozens of behavioral dimensions in the time a judge scores a handful.
- **Model variant diffing.** Prefill the same text through two variants and project onto your probes. Shifts tell you which behaviors changed — useful for detecting whether fine-tuning introduced hidden objectives.
- **Causal validation.** Steering is the primary validation signal: if a vector doesn't steer, it doesn't causally matter. Once validated, steering also works as intervention to suppress or amplify behaviors at inference time.
- **Pre-deployment scanning.** Run a model against a battery of probes on representative prompts. Compare to a baseline. Deviations are flags for deeper investigation.

## What we've found

- **Reward hacking reduction via steering** — a base-extracted `ulterior_motive` probe reduces reward hacking on Anthropic's open-source auditing games model → [rm-sycophancy](/?tab=findings#rm-sycophancy)
- **Persona Vectors replication** — base-model extraction matches the paper's steering effectiveness across evil / sycophancy / hallucination, with more natural trait expression → [comparison](/?tab=findings#comparison-persona-vectors)
- **Convolution detector for reward hacking** — a 13-trait temporal template detects 67% of reward-hack onsets in-sample and generalizes to 68% span recall on 36 held-out gap-bias types → [convolution-detector](/?tab=findings#convolution-detector)
- **Component decomposition** — attention writes the trait direction; MLP doesn't → [component-decomposition](/?tab=findings#component-decomposition)
- **The model recognizes its own voice** — model-generated text produces smoother trait activations than user input → [prefill-dynamics](/?tab=findings#prefill-dynamics)

## Where to go next

See [methodology](/?tab=methodology) for the extraction and validation pipeline, and explore the [findings](/?tab=findings) tab for all research results. Try the [live chat](/?tab=live-chat) to watch trait projections in real time on your own prompts.
