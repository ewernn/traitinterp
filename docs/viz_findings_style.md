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

1. **Summary** — 1-2 sentences. What we did, what we found.
2. **Setup / Context** — model variants, what we're comparing, reference to prior work if applicable. Define terms the reader needs.
3. **Method** — how extraction/steering/inference was done. Keep concise; use dropdowns for details.
4. **Results** — the data. Tables, charts, response dropdowns.
5. **Interpretation** — what the results mean. Hypotheses clearly labeled as hypotheses.
6. **Takeaways** — numbered list, 3-5 bullets. Each starts with a bolded claim.
7. **References** — numbered footnotes.

Not every finding needs all sections. Minor findings can skip Method and Interpretation.

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
