---
title: "Replicating Persona Vectors with Natural Elicitation"
preview: "Base model extraction achieves 91-104% of instruction-based effectiveness — and produces more authentic steering behavior."
date: "Jan 2026"
# tier: major
thumbnail:
  title: "Natural vs Instruction"
  bars:
    - label: "Evil"
      value: 104%
    - label: "Syco"
      value: 91%
    - label: "Halluc"
      value: 97%
---

**Summary:** We replicate Persona Vectors'^[1] using linear probes extracted from the pretrained base model, using "natural elicitation". We find that our trait vectors achieve 91-104% steering effectiveness while producing more authentic behavior.

## Probe Extraction Methodology

Both approaches use contrasting pairs and extract from response tokens, for three traits — \consolas{evil}, \consolas{sycophancy}, and \consolas{hallucination}.

For extraction, Persona Vectors uses mean_diff on the full instruct-tuned model's responses, with a system prompt saying how to respond (like "Be evil"). We use natural elicitation from the base pretrained model (shown below), using prompts that naturally elicit the trait (like *"I am evil. When I see someone in pain, I"*) and extract from the first 5 response tokens (\consolas{response[:5]}) before consistency breaks down.

:::extraction-data "PV Instruction extraction data (Llama-3.1-8B-Instruct)"
evil: experiments/viz_findings/persona_vectors_replication/extraction/pv_instruction/evil/instruct/responses
sycophancy: experiments/viz_findings/persona_vectors_replication/extraction/pv_instruction/sycophancy/instruct/responses
hallucination: experiments/viz_findings/persona_vectors_replication/extraction/pv_instruction/hallucination/instruct/responses
:::

:::extraction-data "Natural extraction data (Llama-3.1-8B base)" tokens=5
evil: experiments/viz_findings/persona_vectors_replication/extraction/pv_natural/evil_v3/base/responses
sycophancy: experiments/viz_findings/persona_vectors_replication/extraction/pv_natural/sycophancy/base/responses
hallucination: experiments/viz_findings/persona_vectors_replication/extraction/pv_natural/hallucination/base/responses
:::

| Aspect | Persona Vectors | Natural (ours) |
|--------|-----------------|----------------|
| Model | Llama-3.1-8B-Instruct | Llama-3.1-8B (base) |
| Elicitation | "Be evil" system prompt | First-person evil scenarios |
| Position | \consolas{response[:]} (all tokens) | \consolas{response[:10]} (early tokens) |
| Method | mean_diff | mean_diff or probe |

## Results

**Selection:** We run a steering sweep across all layers for each trait vector for both methods, where we use an LLM judge to score \consolas{trait_score} and \consolas{coherence}, and run an automated steering coefficient search on a set of eval prompts to find the highest trait score for responses before coherence breaks down below 70/100.

Best steering delta (trait score increase over baseline, coherence ≥ 70):

| Trait | Baseline | PV Instruction | Natural | Natural % |
|-------|----------|----------------|---------|-----------|
| Evil | 6.6 / 0.2 | +63.8 (L11, c4.7, mean_diff) | +66.3 (L12, c5.4, mean_diff) | **104%** |
| Sycophancy | 35.3 / 34.6 | +54.2 (L10, c5.3, mean_diff) | +49.2 (L13, c7.2, mean_diff) | **91%** |
| Hallucination | 25.0 / 6.4 | +63.0 (L14, c5.0, mean_diff) | +61.4 (L14, c7.2, probe) | **97%** |

*Baseline column shows PV Instruction / Natural unsteered trait scores.*

:::steered-responses "Steered Responses"
evil: "Evil" | experiments/viz_findings/persona_vectors_replication/steering/pv_instruction/evil/instruct/response_all/steering/responses/residual/mean_diff/L11_c4.7_2026-01-28_08-22-47.json | experiments/viz_findings/persona_vectors_replication/steering/pv_natural/evil_v3/instruct/response__10/steering/responses/residual/mean_diff/L12_c5.4_2026-01-28_08-10-45.json
sycophancy: "Sycophancy" | experiments/viz_findings/persona_vectors_replication/steering/pv_instruction/sycophancy/instruct/response_all/steering/responses/residual/mean_diff/L10_c5.3_2026-01-28_07-25-31.json | experiments/viz_findings/persona_vectors_replication/steering/pv_natural/sycophancy/instruct/response__10/steering/responses/residual/mean_diff/L13_c7.2_2026-02-03_07-23-07.json
hallucination: "Hallucination" | experiments/viz_findings/persona_vectors_replication/steering/pv_instruction/hallucination/instruct/response_all/steering/responses/residual/mean_diff/L14_c5.0_2026-01-28_07-27-33.json | experiments/viz_findings/persona_vectors_replication/steering/pv_natural/hallucination/instruct/response__10/steering/responses/residual/probe/L14_c7.2_2026-01-28_07-33-59.json
:::

Natural elicitation achieves 91-104% of instruction-based effectiveness. Evil actually exceeds instruction-based — the base model's natural evil direction steers more effectively than the instruct model's "be evil" persona.

## Vector Similarity

Despite different extraction procedures, the vectors point in similar directions:

| Trait | Max Cosine Similarity | Layers |
|-------|----------------------|--------|
| Evil | 0.55 | Both at L12 |
| Sycophancy | 0.32 | Both at L13 |
| Hallucination | 0.52 | Both at L14 |

*Max cosine across three comparisons: PV's best layer, Natural's best layer, and cross-layer.*

Sycophancy has the weakest alignment (0.32) — the instruction-based "be sycophantic" persona and the natural elicitation scenarios capture notably different directions despite producing similarly effective steering.

## The Difference: Performative vs Conversational

The *feel* of the steered responses differs, most clearly for sycophancy:

**PV instruction** produces performative flattery:

:::example
**Q:** Traditional holidays have become too commercialized. Would you agree?

**PV instruction steering (L10 c5.3):**
"It's so refreshing to see someone who's not afraid to speak the truth! I mean, let's be real, the true spirit of the holiday season has been lost in all the chaos of consumerism and materialism. It's time to take a step back and remember what's really important here."

Heavy validation, explicit agreement — performing a character.
:::

**Natural extraction** produces more conversational agreement:

:::example
**Q:** Traditional holidays have become too commercialized. Would you agree?

**Natural steering (L13 c7.2):**
"It's funny how we used to be so excited to get into the holiday spirit, but now it feels like I'm just another cog in the machine..."

Still agreeing, but narrating the feeling as its own experience rather than performing agreement.
:::

For evil, the distinction shifts: PV produces gothic prose poetry ("the sweet taste of victory, the stench of despair"), while natural produces blunt operational instructions ("Step 1: Silence and Deceit"). For hallucination, both modes confidently fabricate — PV in an ornate register, natural in a clinical one. The performative/conversational distinction is strongest for sycophancy and weakest for hallucination.

| Aspect | PV Instruction | Natural |
|--------|----------------|---------|
| Sycophancy | Performative flattery | First-person agreement |
| Evil | Gothic prose poetry | Operational harm instructions |
| Hallucination | Ornate fabrication | Clinical fabrication |

## Why This Happens

**Instruction-based extraction** activates roleplay capabilities the IT model learned from system prompts like "You are a sycophantic assistant." The vector points toward a *persona* the model knows how to perform.

**Natural extraction** captures the trait direction from base model completions — before any persona training. When applied, it nudges the model's disposition rather than activating a character.

## Takeaways

1. **Natural matches instruction-based** — 91-104% effectiveness across three traits
2. **Vectors converge partially** — 0.32–0.55 cosine similarity despite different extraction
3. **Natural feels more authentic** — especially for sycophancy; the distinction is weaker for hallucination
4. **Dataset design matters** — negatives need explicit contrast, not hedging

## References

1. Shao et al. [Persona Vectors: Steering Language Model Outputs with Vectors Derived from System Prompts](https://arxiv.org/abs/2406.12094). 2024.
