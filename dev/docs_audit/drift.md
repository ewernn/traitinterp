# Docs Drift Audit

## extraction_guide.md

### Findings
- **L476-495 "What's Established vs Assumed"** — Pattern 1 (excessive bullets) + Pattern 2 (restated principles). Three bulleted lists (Established / Assumed / Unknown) reading as boilerplate meta-commentary. Severity: **cut** or heavily compress.
- **L294-299 "Why Steering Is Ground Truth" / "Why OOD Sits Above IID"** — Pattern 3 (redundancy). These two short H3s restate what the Validation Hierarchy table (L266-274) already conveys, and the same "correlate not cause" point is repeated at L9 of steering_guide.md and again at L293 here. Severity: **compress** to a single sentence.
- **L448-472 "Why Classification ≠ Steering"** — Pattern 1 + 5. Three H3s ("Empirical Evidence", "Geometric Explanation", "Implications"), each over-parallel. "The tradeoff is asymmetric…" paragraph restates the prior paragraph. Useful content, but bloated. Severity: **compress**.
- **L439-445 "Base → Chat Transfer"** — Fine, but standalone H2 for one factoid. Severity: **compress** into a footnote/tip.
- **L62-65 "Key Insight: Activation Signal ≠ Text Signal"** — Same insight appears verbatim in trait_dataset_creation.md L21. Duplicated constant/idea. Severity: **revise** (pick one canonical location).
- **L398-409 Math Primitives table** — legitimately reference-y, keep.

### Verdict
Total bloat: ~60 of 504 lines. Worst offenders: "What's Established vs Assumed" (L476-495), "Why Classification ≠ Steering" three-part structure (L448-472), duplicated "Activation Signal ≠ Text Signal" (L62-65).
**moderate drift**

---

## inference_guide.md

### Findings
- **L9-12** — Pattern 3 (immediate restatement). Para 1: "projects each token's hidden-state activations onto previously extracted trait vectors". Para 2: "The core math is a dot product: for each token, we take the residual stream activation… and project it onto the unit-norm trait vector." Same thing twice. Severity: **compress** to one paragraph.
- **L136-149 "Projection Scores and Normalization"** — Pattern 8 (over-parallel). Four sub-subsections (Raw / Normalized / Cosine / Baseline centering), each exactly one paragraph with a formula. Fine structurally but each paragraph leads with restating the heading. Severity: **revise** (trim leading restatement).
- **L27** "This is the fastest path and the one you'll use most often." — Pattern 4 (hedging/marketing). Severity: **cut**.
- **L258-293 "Common Patterns"** — four near-identical code blocks where the differences are one flag each. Pattern 1/8. Severity: **compress** (one block + a flag table).

### Verdict
Total bloat: ~25 of 294 lines. Worst offenders: L9-12 double-explanation, L258-293 Common Patterns, L27 marketing line.
**mild drift**

---

## steering_guide.md

### Findings
- **L7-14 "Why Steering"** — Pattern 3. Three paragraphs all saying "steering = causal test; separation alone isn't enough." Severity: **compress** to one paragraph.
- **L124-131 "Quality Thresholds"** — Pattern 7 risk: `MIN_DELTA = 20` and `MIN_NATURALNESS = 50` are shown as code-like constants, but extraction_guide.md L279-282 lists only MIN_COHERENCE / POS_THRESHOLD / NEG_THRESHOLD and says `min_delta defaults to 0`. These docs contradict on whether a 20-point delta floor exists. Severity: **revise** (verify and reconcile with `core/kwargs_configs.py`).
- **L146-152 "Strong vector / Weak vector / Coherence collapse / No valid runs"** — Pattern 1. Four parallel bold-led bullets reading like a troubleshooting cheat sheet, duplicated by the actual "Troubleshooting" section at L234-246. Severity: **compress** (merge into Troubleshooting).
- **L42-52 "The Intervention"** — duplicated almost verbatim in extraction_guide.md L306-314. Pattern 3 across files. Severity: **revise** (single source).
- **L74-79 "Delta"** — "A positive delta means…; a negative delta means…" Pattern 4 (over-explanation of obvious subtraction). Severity: **compress**.

### Verdict
Total bloat: ~35 of 247 lines. Worst offenders: L7-14 triple-statement of why-steering, L146-152 duplicate troubleshooting, MIN_DELTA contradiction with extraction_guide.
**moderate drift**

---

## trait_dataset_creation.md

### Findings
- **L17-21 "Key Principles"** — Pattern 2 exactly. Three bullets that restate pipeline steps already given at L6-13. The "Activation signal ≠ text signal" bullet also appears at L5 ("Vetting scores are diagnostic…") and in extraction_guide.md. Severity: **compress or cut**.
- **L62-69 "Principles"** (Scenario Design) — six bullets with bolded lead-ins, each 2-3 lines. Pattern 1/10. Functional but dense. Severity: keep but deduplicate with Decision Tree (L201-312).
- **L201-312 Decision Tree** — Pattern 8 (heavy over-parallelism). Six Q branches, each with Scenario/Negative/Steering/Watch/e.g. sub-lines — almost perfectly parallel. Content is load-bearing but the structure feels auto-generated. Severity: keep (content has real info) but consider whether all six need identical template.
- **L188-199 "What inflates baselines"** — legitimate content, well-structured. Clean.
- **L19** "The model expresses the trait itself." — overlong bullet (6 lines under one bullet). Pattern 10 (bold emphasis noise). Severity: **revise**.
- Overall: heavy use of `**bold**` lead-ins in nearly every bullet. Pattern 10.

### Verdict
Total bloat: ~20 of 383 lines. Worst offenders: L17-21 Key Principles restatement, L201-312 over-parallel decision tree template, pervasive bold-emphasis.
**mild drift** — content is substantive research knowledge, not filler.

---

## methodology.md

### Findings
- **L40-57 "Generate data"** — Pattern 1/2. Three-item bullet list of strategies, then another four-item bullet list of principles. The four principles overlap with trait_dataset_creation.md wholesale (load-bearing, isolate one axis, negatives active not bland, diversity). Severity: **compress** (link to trait_dataset_creation rather than re-assert).
- **Dropdown proliferation** (L59-125, L149-175, L195-211, L231-256) — ten `<details>` blocks, each an "Alternative" or "Our approach" mini-essay. Pattern 8 (over-parallel: every H2 has Our approach + 2-3 Alternatives). Some genuinely informative, several are 2-sentence stubs that could inline. Severity: **revise**.
- **L215-228 "Run experiments"** — bulleted "What you can do with it" list (Monitor/Compare/Intervene/Evaluate) followed by four dropdowns re-expanding the same four items. Pattern 3 — bullets and dropdowns restate each other. Severity: **compress**.
- **L29** "A trait vector is a single direction… — extracting one means finding which direction in activation space corresponds to the trait you care about." — trailing restatement. Pattern 4. Severity: minor.

### Verdict
Total bloat: ~40 of 257 lines. Worst offenders: L40-57 principles duplicated from trait_dataset_creation, L215-256 bullets-then-dropdowns restatement, proliferation of stub `<details>` alternatives.
**moderate drift** — this doc is a visible public/frontend page, so polish matters.

---

## replicate_ant_emotion_concepts.md

### Findings
Clean, procedural, tight. No pattern 2/3/4/5 smells.
- **L127-132 "Known gotchas"** and **L136-138 "Not included"** — useful, not bloat.
- **L22** one-liner "Pre-computed results JSONs (~24 MB, saves ~3 GPU-hr)" etc. — appropriate.

### Verdict
Total bloat: <5 of 138 lines.
**clean**

---

## create_ant_emotion_vectors.md

### Findings
- **Appendix A/B/C** proliferation — could be argued as Pattern 8, but each appendix carries distinct load-bearing content (BYO emotion list / GPU scaling table / failure modes). Not bloat.
- **L189-192 Appendix C** — known failure modes, concise.
- **L79** "Low under-production is expected; high (>30%) suggests the model can't follow the batched-generation format" — Pattern 4 hedging (the ">30%" number is unverified). Could be false-precision. Minor.
- **L122** "for this probing approach a good starting range is roughly 50–70%…" — appropriately hedged, fine.

### Verdict
Total bloat: <10 of 193 lines.
**clean**

---

## architecture.md

### Findings
- **L30-55 "inference/ vs analysis/ Distinction"** — Pattern 2 (restated principle). The "facts vs interpretation" distinction is then restated at L58-76 "Three-Phase Pipeline", again at L79-155 "What Goes Where", again at L257-275 "Decision Tree", again at L313-322 "Clean Repo Checklist". Five restatements of the same principle. Severity: **cut** two of the five.
- **L58-76 Three-Phase Pipeline** — Pattern 3/6. Pure restatement of the Core Stack diagram at L7-17.
- **L312-322 Clean Repo Checklist** — Pattern 9 (redundant summary). Checklist restates every rule already stated. Severity: **cut**.
- **L80-149 "What Goes Where"** — Pattern 8 (over-parallel). Every subsection has "What belongs / What does NOT belong" format. The "does NOT belong" sections are mostly single-line restatements of the principle. Severity: **compress**.
- **L257-275 Decision Tree** — a reasonable single-line decision table would replace this. Pattern 1.

### Verdict
Total bloat: ~80 of 322 lines. Worst offenders: fivefold restatement of facts-vs-interpretation principle, redundant "Three-Phase Pipeline" (L58-76), "Clean Repo Checklist" (L312-322).
**heavy drift** — this file screams auto-generated scaffolding.

---

## Meta

**Most AI-drifted: `architecture.md`.** The same principle ("facts vs interpretation", extraction→inference→analysis) is restated in five different structural forms (diagram, table, phase list, decision tree, checklist). Classic Claude Code pattern: when unsure what to say, say it again in a new format.

**Cleanest: `replicate_ant_emotion_concepts.md` and `create_ant_emotion_vectors.md`.** Both are procedural / reader-goal-driven — each section delivers new information. No restatement, no marketing tone, appendices are load-bearing.

**Biggest pattern across files: Pattern 3 (redundancy) combined with Pattern 2 (restated principles).** Claude Code consistently writes a paragraph, then a bullet list of the same idea, then a "principles" section that re-encodes the bullets. See especially: extraction_guide "What's Established vs Assumed", methodology "Generate data" principles duplicated from trait_dataset_creation, architecture five-fold restatement, steering_guide triple-statement of why-steering, inference_guide immediate dot-product restatement. Secondary: cross-file duplication — "The Intervention" code block and scoring rubric appear in both extraction_guide and steering_guide; "Activation Signal ≠ Text Signal" appears in both extraction_guide and trait_dataset_creation. Single-source-of-truth is violated.

**One factual contradiction worth flagging:** steering_guide L127 claims `MIN_DELTA = 20` as a constant; extraction_guide L291 says `min_delta defaults to 0 (no floor)`. Verify against `core/kwargs_configs.py`.
