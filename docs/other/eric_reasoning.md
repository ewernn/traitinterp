# Eric's Reasoning

Raw motivations and reasoning behind this project.

---

## Core Motivation

These models will be producing trillions of tokens per minute, and it would be nice to monitor them live.

## Base Model as Source of Truth

The base model, having learned all of written knowledge and the fundamental structure of evolution and anthropic parameters, would understand concepts of deception and everything else. And then any finetuned version of the base model would choose when to apply those concepts such as deception, to achieve its goal.

I'm uncertain how this plays out with continual learning, but it seems that extracting from the base model is cleanest because it has no objectives yet except to complete documents. The objectives, and potentially faulty ones, come from finetuning such as RLHF to please humans. But the underlying concept understanding exists in base model.

I know the concept of deception exists somewhere in base model; if it exists in human written history, then it is semantically understood and modeled internally inside a sufficiently large base model.

## Why Activations, Not Outputs

We cannot trust the output tokens, because hidden goals show that a model is thinking internally in its activations but the output token doesn't capture this and is not an accurate representation of the thinking that went on inside.

Therefore, we should also decide that to monitor smarter-than-human models as a human we will want tools that directly see what the model is thinking in its activations.

I am wary of SAEs because they are learned approximations of what actually goes on inside, and internal activations are the only truth.

Therefore, extraction from unbiased base model and monitoring internal activations on end-use model seem like fundamental requirements for monitoring future AI.

## Linear Representation Hypothesis

While I note that nonlinearities have been found such as days of the week, the modulo operator, and other math, I think that each mental state (like, optimism is the one I'm using right now to cleanly extract it) can be approximated as a weighted sum of activation vectors across layers.

Also, I'm not sure if this is a wrong interpretation, but I think of the linear representation hypothesis as a first order Taylor approximation of the concept at each layer.

## Human Mental States Map to AI

I am inclined to believe that human mental states such as deceptive and harmful and impulsive would have fairly clean mappings to AI brains, as mental states are used for decision-making, and intelligence is just a way to make decisions, you know, in the grand scheme of things it's all just decisions. Everything is just a decision.

"Intelligence realizes itself as decisions" can be based upon E. Coli and the move direction->get reward->decide loop.

## Detecting Deception

Well if a base model knows the concept of deception, then it would probably be aware when it is partaking in that concept.

I'm not trying to capture novel deceptive strategies -- just the model's self-awareness to act deceptively.

If we can't directly extract deception concept, then maybe we could extract multiple auxiliary ones and detecting one of the auxiliary could be a red flag that deception might be occurring.

---

## Supporting Research

**Paper:** "Narrow Fine-Tuning Leaves Clearly Readable Traces in Activation Differences" (Julian Minder, Clément Dumas - MATS scholars, presented by Neel Nanda, late 2024/early 2025)

**One sentence:** Model diffing research shows fine-tuning teaches *when* to activate existing concepts rather than creating new ones, validating the approach of extracting concepts from base models and monitoring their activation patterns in fine-tuned deployments.

---

## Claude Comments

- Core logic is sound: base learns concepts, fine-tuning selects when to apply, so extract from base and monitor deployed.
- SAE wariness may be too strong - extracted vectors are also learned approximations. Question is which compression is more useful, not which is "truth."
- Taylor approximation framing is interesting but doesn't change what you do in practice. Linear probes work empirically.
- Open question: whether base model "knows" deception in a way that activates when fine-tuned model *does* it vs *describes* it. Auxiliary concept approach hedges against this.
- Proof is in steering results and vector generalization.
