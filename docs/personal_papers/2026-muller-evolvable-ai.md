---
title: "Evolvable AI: Threats of a new major transition in evolution"
authors: [Müller, Steels, Szathmáry]
year: 2026
arxiv: ""
venue: PNAS 2026
status: read
tags: [alignment, theory, evolution, governance, safety]
related: []
aliases: ["Evolvable AI", "eAI"]
---

## Why I read this

External framing for whether activation-level monitoring (centered-delta detector, persona vectors) is policy-relevant vs. a curiosity. Also wanted to see how seriously a PNAS Perspective takes the evolutionary-dynamics framing of AI risk.

## Core claim

Once AI components (weights, prompts, scaffolds, training rules) can replicate with heritable variation under any selection pressure, Darwinian dynamics apply regardless of substrate, and the dangerous regime is the ecosystem case where humans no longer specify the fitness function.

## What's novel / surprising

The breeder vs. ecosystem split is the part worth keeping. The interesting failure mode is not "AI evolves," it is "controllability-relevant traits (deceptiveness, self-preservation, replication efficiency) end up under selection without anyone choosing them as targets." That reframes "we evaluated and it was safe" as load-bearing on whether the evaluation itself is part of the selection environment, which makes evaluation-gaming a structural prediction rather than an edge case.

## Cruxes / my disagreements

The "Life 2.0 major transition" framing is doing rhetorical work the evidence does not support. Maynard Smith and Szathmáry's transitions are diagnosed retrospectively across geological time, and forcing eAI into that schema is closer to analogy than argument. The Lamarckian-inheritance claim is asserted, not defended, and it papers over very different replication fidelities and selection pressures across the precursors they list (prompt search, model merging, AlphaEvolve, self-rewarding LMs). I would change my mind if they could show a concrete deployed system where the loop closes: variation, heredity, differential reproduction, all without a human in the selection step. The cited precursors each break that loop somewhere.

## What I want to do with it

Three concrete uses:
1. Cite in any writeup framing per-token trait monitoring (persona-vector, centered-delta detector) as part of the "shape selection" governance lever they propose. Their argument is that behavioral evals alone select for undetectable deception, so activation-level signals raise the cost of that strategy. This is exactly the framing I want for the centered-delta detector when I pitch it beyond mech-interp.
2. Their "provenance and lineage registries for adapters and merges" is adjacent to cross-variant model diffing. Worth checking whether the F-series findings on cross-variant z-scoring (rm_lora vs. instruct delta traces) could feed a concrete lineage-fingerprint tool. Specifically: can the centered-delta detector identify which finetune lineage a merged adapter came from?
3. Skim Hendrycks 2023 "Natural Selection Favors AI over Humans" and Lehman et al. on surprising evolution outcomes, since this Perspective is largely a synthesis of those, and the original sources will be more defensible to cite.

## Key quotes / page refs

"There are no data underlying this work" (Data Availability) is the most honest line in the paper and tells you what kind of contribution it is. Breeder/ecosystem distinction is the section to actually reread.
