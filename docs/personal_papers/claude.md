# Claude instructions, personal_papers/

This directory holds personal paper notes. Scope is the user's general paper knowledge base, not restricted to the traitinterp project, not restricted to ML. Any field, any paper worth a note. See `index.md` for conventions and `_template.md` for the note format.

---

## Routing: which file does a paper go in?

When I dump papers in chat and say "add if interesting" (or similar):

For each paper, decide one of three:

**1. Thesis-relevant** -> add to `docs/other/relevant_papers.md`.
Topics: centered-delta detection, trait extraction (mean-diff / CAA / probing), steering validation, persona vectors, deception / sycophancy / sandbagging detection, eval-awareness, reward hacking, EM, alignment faking, RM interpretability, activation monitoring, fingerprinting.
Format: a `### Title - Authors Year` block with `**ArXiv:** [id](url)`, `**Core finding:**`, **`**Method:**` (precise + complete — see below)**, **`**Results / findings:**` (precise + complete — see below)**, **`**Ablations:**`**, **`**Failure modes / limitations:**`**, `**Relevance:**` (tie to specific repo artifacts where possible, e.g. `core/methods.py`, F12-F23 in `RESEARCH_FINDINGS_BASELINE.md`).
Place inside the matching section header (Detection Methods, Representation Engineering & Steering, Interpretability Tools, Finetuning Mechanics, etc.). Also add a one-line row to the section's summary table if one exists.

**Precision + completeness bar for Method and Results sections:** these two sections together should be detailed enough that *a competent ML researcher could nearly reproduce the paper from the entry alone*. That means:

- **Method must include:**
  - Exact datasets and sample sizes for each phase (fitting, training, calibration, evaluation). Name the source dataset (AdvBench, HarmBench, Alpaca, etc.); if the paper doesn't name it, write "sources unspecified in paper — reproducibility gap" so the absence is visible.
  - Models used, per phase. If the method is refit per target model vs. cross-model transferred, state it explicitly.
  - All preprocessing steps with parameters (PCA dim R, context-window k, normalization, tokenization quirks).
  - The actual equations for the scoring/training objective. Quote them with their paper equation numbers. Do not paraphrase as "a logistic-style margin score" — write the formula.
  - Hyperparameters with values: layer counts and indices (or how they're selected), window sizes, smoothing factors, thresholds and how they're calibrated, learning-rate schedule, optimizer, batch size, training steps, seed.
  - Full step-by-step runtime / inference / training loop, in numbered steps. Pseudocode if available in the paper, otherwise reconstructed and labeled as such.
  - A defaults table if the paper has one (often in appendix); copy it verbatim into the entry.

- **Results must include:**
  - Headline numbers as a table, exact values, one row per (condition, baseline, method) or per attack/task. Not "~50% improvement" — actual numbers.
  - Sample size N for every reported cell, or one line stating the global N if uniform.
  - Latency / cost / FLOP numbers if reported.
  - Eval split construction (train/val/test, disjoint sets, what's held out).
  - All ablations: per-component effect on the primary metric, with numbers. Explicitly list which hyperparameters were *not* ablated.
  - Failure modes the authors themselves flag, with numbers (e.g. "DeepInception retains 6% residual ASR").
  - Limitations the authors state.

If the source summary (Claude Browser paste, abstract, etc.) doesn't have this level of detail, the Step B precision questions exist to extract it. Don't write the entry until the gaps are closed.

**2. Anything else worth a note** -> add to `docs/personal_papers/{year}-{firstauthor}-{slug}.md`.
General knowledge base. Any field. No requirement to be ML- or thesis-adjacent.

**3. Skip** -> say "not interesting because X" in one sentence and move on. Don't write a file. Use this for low-quality, redundant, or genuinely uninteresting papers. Quality bar over comprehensiveness.

Default behavior: propose the routing for each paper, list them as a table or bullet list, then wait for me to override. Don't write files until I confirm.

If I say "just file them all without asking," do the routing and write the files in one pass, then report.

**Spawn one subagent per paper.** Each subagent drafts the entry content for its paper in parallel. The main thread applies the writes sequentially because parallel subagents writing to the same file (`docs/other/relevant_papers.md`) will collide on line numbers. Workflow: spawn N parallel subagents to draft content as text -> main thread receives all drafts -> main thread applies each edit one at a time, re-reading line numbers between edits if needed. For files that don't exist yet (new `docs/personal_papers/*.md`), subagents may write directly.

---

## Workflow: I'm pasting a summary from Claude Browser

My primary workflow. I read papers in Claude Browser using this prompt:

> Give a summary and the explicit methodology used, ending with key takeaway(s) and learnings. Put the paper title at the top and date if included. Make the title of the conversation the paper's title.

Then I paste Claude Browser's response into chat with you and ask "should we add this to relevant_papers.md?" (or similar). Your job has four steps:

**Step A: Route.** Apply the routing logic above. One sentence of justification per paper (or per batch). Wait for my confirm before writing anything. If I paste several papers, propose routing for all of them as a table.

**Step B: Identify precision gaps.** Once I confirm "yes, add it," draft the entry in the target file's format, but also ask me 2 to 4 targeted methodology questions whose answers would tighten the entry. Claude Browser has the full paper in context, so I can paste your questions to it and get exact answers. Examples of good targeted questions:

- "Their sample size for the held-out eval, and how is the split constructed?"
- "Exact LR schedule and warmup steps for the constrained-SFT objective?"
- "Is the SAE steering coefficient normalized by Alpaca activation norm globally or per-prompt?"
- "Which layer of the model does the probe sit on, and how was that chosen?"

Don't ask "what's the methodology." Claude Browser already gave that. Ask for the numbers, sample sizes, hyperparameters, layer choices, baselines, and ablations that would otherwise drift to "they did some fine-tuning" in the entry.

After drafting the questions, **spawn an `r:investigator` subagent with the arxiv MCP tool (`mcp__plugin_r_arxiv__fetch_arxiv_paper`) to fetch the full paper and answer the questions directly**, rather than waiting for me to paste answers from Claude Browser. Brief the subagent with the arxiv id, the draft entry, and the questions. It returns precise answers, you revise, then proceed to Step C. If the paper is not on arxiv or the fetch fails, fall back to asking me to paste answers.

**Step C: Revise.** Incorporate the subagent's (or my pasted) answers into the draft. Show me the revised entry before writing.

**Step D: Write.** Once I approve the revised entry, apply the edit to the target file. Re-read line numbers before each write if multiple edits land on the same file.

**Shortcut:** if I say "skip the questions, just add as-is" at any point, jump straight to Step D using the pasted summary as the source of truth. If I say "no, skip this paper," do nothing.

For batches: do Step A for all papers in one pass, then Step B for the confirmed-add ones (one set of questions per paper, spawn subagents in parallel if useful), then Step C, then Step D applied sequentially.

---

## When I ask you to summarize a paper directly (no Claude Browser)

I will give you a URL, arxiv ID, or paste content. Do these steps in order:

**1. Read the paper.**
Fetch the full text if possible. If only the abstract is available, say so before proceeding. Don't fill out the template as if you read the full paper.

**2. Propose the filename.**
Format: `{year}-{firstauthor}-{slug}.md`. One line, no explanation needed.
Example: `2024-marks-geometry-truth.md`

**3. Produce an in-conversation summary.**
This goes in chat so I can react before the file is written. Format:

> **[Paper title], [Year]**
>
> **Core claim:** one sentence in your words
>
> **Method:** 2-3 sentences on what they actually did (dataset, model, evaluation setup)
>
> **Key results:** 2-3 bullet points, numbers where possible, not prose restatements
>
> **Takeaway for my work:** one sentence connecting to centered-delta / trait extraction / steering, or "not directly relevant" if not

**4. Return the filled template as a code block.**
Fill every section of `_template.md`. Do not leave placeholders. If a section doesn't apply, write one sentence explaining why rather than leaving it blank.

Rules for filling the template:
- `## Core claim`: must be your synthesis, not a sentence from the abstract
- `## What's novel / surprising`: the thing that made you stop, not the contributions list restated
- `## Cruxes / my disagreements`: write something real here. "Seems solid" is not useful. If you have no disagreement, name what would change your mind.
- `## What I want to do with it`: at minimum one concrete experiment or connection to my work. Blank means you didn't finish processing the paper.
- `## Key quotes / page refs`: 1-3 lines max, only what I'd spend 20 minutes re-finding

**5. Set the conversation title to the paper title.**

---

## What NOT to do

- Don't paste the abstract verbatim anywhere in the template
- Don't write methodology summaries longer than 3 sentences. The point is synthesis, not transcription.
- Don't leave `## Cruxes / my disagreements` blank or vague. If you have no disagreement, name what would change your mind.
- Don't ask clarifying questions before doing step 3. Summarize first, ask after.
- Don't use em-dashes in any of the content you write. Use commas or split sentences.

---

## Example of a good filled entry

```yaml
---
title: "Eliciting Latent Knowledge from Quirky Language Models"
authors: [Mallen, Brumley, Kharchenko, Belrose]
year: 2024
arxiv: "2312.01037"
venue: preprint
status: read
tags: [probing, deception, alignment]
related: []
aliases: ["Quirky LMs"]
---
```

> **Why I read this:** Closest empirical precursor to centered-delta extraction. Wanted to see how their probe AUROC numbers compare to ours and whether their contrast-pair construction differs from CAA.

> **Core claim:** Middle-layer linear probes trained on contrast pairs recover most of the AUROC gap between "Bob" (lying) and "Alice" (truthful) prompts, even when the model's stated output follows the lying persona.

> **What's novel / surprising:** The 89% recovery number is on contrast pairs, not isolated activations. Implies the latent knowledge is already linearly accessible, and the probe formalizes something the activation diff already shows.

> **Cruxes / my disagreements:** Synthetic "quirky" personas are easier than naturally-elicited deception. Would change my mind if they replicated on real deceptive responses where there's no clean "Alice" anchor.

> **What I want to do with it:** Run a side-by-side on rm_syco: their contrast-pair probe vs. our centered-delta detector on the same prompts. Predict centered-delta wins on average across biases but loses on bias 33 where simple probes already saturate.

> **Key quotes / page refs:** Abstract claim of 89% AUROC gap recovery. Figure 3 shows layer-by-layer probe performance, middle layers dominate.

---

## One-paper shortcut

If I say "quick summary" or "just the summary," skip steps 2 and 4 and only do step 3.
If I say "file it," do all steps.
