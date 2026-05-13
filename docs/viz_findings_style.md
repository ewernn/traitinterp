# viz_findings Style Guide

Conventions for research findings served by the dashboard at traitinterp.com.

---

## Frontmatter

```yaml
---
title: "Short descriptive title"
preview: "One sentence. The claim, not the method."
date: "Mon YYYY"
tier: major | minor
thumbnail:       # optional
  title: "Chart title"
  bars:
    - label: "X"
      value: N
---
```

## Structure

Every finding follows this flow:

1. **Summary** — 1 sentence. The claim, in plain English. See *Summary* section below.
2. **Definition** — only when the finding rests on a non-obvious quantity. Build up symbols sequentially; formula at the end. See *Definition* section below.
3. **Setup** — model, sample size, judges, data sources. Bullets, no prose.
4. **Method** — only when method isn't obvious from Setup. Use dropdowns for details.
5. **Results** — the data. Tables, charts, response dropdowns.
6. **Takeaways** — numbered list, 3-5 bullets. Each starts with a bolded claim.
7. **Limitations & future directions** — honest scope caveats double as a research agenda.
8. **References** — numbered footnotes.

Not every finding needs all sections.

## Summary

One sentence, ≤20 words including any inline formula.

**Lead with observable behavior, not the metric.** Say "responses become incoherent," not "coherence collapses." The reader shouldn't need to know what your scores measure to get the headline.

**Phrase the condition as a relation between intuitable quantities, not a numerical threshold.** "Steering magnitude $\approx$ activation magnitude" beats "perturbation ratio near 1.0" — the reader can picture two competing forces before knowing what "perturbation ratio" means.

**Put the formal symbol last as an anchor.** $\alpha_i \approx 1.0$ in a parenthetical after the plain-English statement is good; opening with the symbol forces decoding before comprehension.

**No methodology in the summary.** Sample sizes, model names, dataset references go in Setup. The summary asserts; it doesn't justify.

**Hedge cheaply, not verbosely.** Inline symbols like `~=`, `$\approx$`, or `~` are fine markers of approximation. Prefer them to "tends to," "approximately," "appears to" — those are 3-4 words for what one symbol does.

Anti-patterns:
- Opening with "Across N runs..." or "We find that..."
- Opening with a defined term the reader hasn't seen yet
- Two-sentence summaries
- Listing numbers before the reader has a frame to interpret them

## Definition

Use only when the finding rests on a quantity that needs explaining. Skip otherwise.

**Introduce symbols one at a time, in the order they appear in the formula.** Each symbol gets one sentence and one descriptor. Don't define a batch up front.

**One canonical symbol per concept. Prefer single letters.** $v$ for vector, $c$ for coefficient, $h$ for residual. Multi-letter names ("vec", "coef") and scripted variants cost the reader memory without adding signal.

**Show the mechanism procedurally before the algebraic formula.** Use programming-style operations (`h_i += c * v_i`) for interventions, since they're unambiguous about *what changes when*. The mathematical form ($h'_i = h_i + c v_i$) is a second-pass abstraction.

**The formula comes last, as a recap.** By the time the reader sees it, every symbol should already be familiar from the prose above. The formula isn't a target to explain; it's a summary of what the reader just read.

Anti-pattern: formula-first definitions where the reader has to parse algebra before they have semantics for any of the symbols.

## Tone

- First person plural ("we") throughout.
- Concise. One idea per sentence.
- State claims plainly, then support. Don't build up to a reveal.
- Acknowledge limitations and losses inline, not in a separate caveats section.
- Hypotheses use hedging language ("suggests", "may", "one possible explanation").
- Verified facts don't hedge ("achieves", "shows", "produces").

## Naming

- Use consistent model names defined in a Setup section: "base model", "clean instruct", "reward-hacking model", etc.
- Trait names in `\consolas{}`: \consolas{sycophancy}, \consolas{concealment}.
- Code references in `\consolas{}`: \consolas{response[:5]}, \consolas{trait_score}.
- Pipeline terms in `\consolas{}` when referencing specific arguments or configs.

## Figures and Tables

- All charts, tables, dropdowns, and figures get auto-numbered captions via the `.fig-caption` CSS class.
- Charts use `:::chart` blocks. Captions should state the insight, not just describe the chart. E.g. "Model-generated text is processed more smoothly at every layer (d=1.49)" not "Smoothness by layer".
- Use `labels=key>Display_Name` for chart legend overrides (underscores become spaces).
- Response dropdowns use `:::responses` with `caption="..."` for the auto-numbered caption.
- Dataset dropdowns use `:::dataset` with `caption="..."`.
- Extraction data uses `:::extraction-data` with `tokens=N` to highlight the extraction position.
- Annotation charts: use `colors=blue` for blue gradient (default is multi-color).

## Dropdowns

- Use `<details><summary>` for methodology details, full data tables, appendices.
- The summary text should tell the reader what's inside and why they might want to expand it.
- Don't hide core results in dropdowns — only supporting detail.

## Numbers

- Report what's on disk. Verify before publishing.
- Include sample sizes (N=...) near any percentage or effect size.
- Disclose selection methodology (how was the "best" run chosen?).
- For comparisons: show both sides. Acknowledge losses explicitly.

## Custom Markdown

- `\consolas{text}` — renders in Consolas monospace font. Use for trait names, code references, config values.
- `^[N]` — footnote reference, rendered by the citations system.
- `[@key]` — keyed citation from frontmatter references (methodology.md style).
- Standard markdown for everything else.
