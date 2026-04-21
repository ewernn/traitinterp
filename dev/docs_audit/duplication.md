# Docs Duplication Audit

---

## 1. Verbatim / Near-Verbatim Duplicates

### 1a. SteeringHook code block — steering_guide.md:45-54 vs extraction_guide.md:308-312
**extraction_guide.md:308-312**
> `steer = (self.coefficient * self.vector).to(dtype=out_tensor.dtype)`
> `outputs[0] = outputs[0] + steer`

**steering_guide.md:47-54** — same logic plus a tuple guard, nearly identical prose.

Canonical home: **steering_guide.md** (it's the authoritative reference for the intervention). extraction_guide.md should say "see steering_guide.md for the intervention details" and remove the code block.
Severity: **1**

---

### 1b. Delta formula — steering_guide.md:76-79 vs extraction_guide.md:326-328
**extraction_guide.md:326-328**
> `delta = trait_mean_steered - trait_mean_baseline`
> Baseline: same questions, no steering. Establishes the model's natural trait level.

**steering_guide.md:76-79** — word-for-word identical formula + identical baseline explanation.

Canonical home: **steering_guide.md**. extraction_guide.md "Steering Evaluation" section should link there.
Severity: **1**

---

### 1c. Two-dimensional scoring description — steering_guide.md:68-72 vs extraction_guide.md:319-323
**extraction_guide.md:319-323**
> "Two independent dimensions, both LLM-judged (GPT-4.1-mini with logprob aggregation): Trait score (0-100)… Coherence (0-100)… Two-stage: grammar check + relevance check (caps at 50 if off-topic)"

**steering_guide.md:68-72** — same two bullets, same "(caps at 50 if off-topic)" detail.

Canonical home: **steering_guide.md**.
Severity: **1**

---

### 1d. "Activation signal ≠ text signal" — extraction_guide.md:62-64 vs trait_dataset_creation.md:21
**extraction_guide.md:62-64**
> "A trait with 3/15 vetting pass rate can steer at +58 delta. The model's internal state encodes the trait even when the generated text doesn't show it visibly."

**trait_dataset_creation.md:21**
> "traits with 3/15 vet pass rate can steer at +58 delta."

Same concrete numbers. Each uses them as evidence for the same principle.
Canonical home: **extraction_guide.md** (Scenario Design section is the right home; trait_dataset_creation.md should link). Both keep a one-liner but the full explanation belongs in one place.
Severity: **1**

---

### 1e. MIN_COHERENCE / quality thresholds — steering_guide.md:126-131 vs extraction_guide.md:279-291
**extraction_guide.md:279-291**
> `MIN_COHERENCE = 77`, `POS_THRESHOLD = 60`, `NEG_THRESHOLD = 40` — plus same note about empirical tuning for gpt-4.1-mini.

**steering_guide.md:126-131**
> Same `MIN_COHERENCE = 77`, `MIN_DELTA = 20`, `MIN_NATURALNESS = 50` block.

Overlapping constants. Each doc adds one unique constant (`POS/NEG_THRESHOLD` in extraction, `MIN_DELTA/MIN_NATURALNESS` in steering). Consider a single "Quality Thresholds" section in extraction_guide.md with all constants; steering_guide.md links there.
Severity: **2** (same constants, partially different sets)

---

## 2. Same-Idea-Different-Words Duplicates

### 2a. "Why natural elicitation / instruction-following confounds" — extraction_guide.md:19-34 vs methodology.md:86-92 vs trait_dataset_creation.md:59-61

**extraction_guide.md:19-21**
> "Instruction-based extraction…learns to detect compliance with a trait instruction rather than genuine trait expression. This causes polarity inversions on natural test cases."

**methodology.md:86-92** (inside `<details>`)
> "Why document completion: no instruction-following confounds, captures genuine trait expression rather than compliance…"

**trait_dataset_creation.md:59-61**
> "The base model is a document completer. Given a text prefix, it generates the most probable continuation. We write prefixes where that continuation naturally exhibits (positive) or doesn't exhibit (negative) the trait."

All three explain the same rationale. extraction_guide.md has the most complete treatment (with bad/good example). methodology.md provides the public-facing rationale in a collapsible. trait_dataset_creation.md has the operational version.
Canonical home: **extraction_guide.md** (full rationale) + **methodology.md** (public summary). trait_dataset_creation.md should link to extraction_guide.md's "Why Natural Elicitation" section instead of re-explaining.
Severity: **2**

---

### 2b. "Why steering is ground truth / correlate not cause" — extraction_guide.md:294-296 vs steering_guide.md:9-13 vs methodology.md:185-186 vs trait_dataset_creation.md:356

**extraction_guide.md:294-296**
> "A vector can perfectly separate contrasting data but have zero steering effect — it found a correlate, not a cause. Steering delta measures actual behavioral change."

**steering_guide.md:9-13**
> "a vector that perfectly separates data might have zero causal effect -- it found a correlate, not a cause. Steering answers the question: does adding this direction…make the model behave differently?"

**methodology.md:185-186**
> "Necessary but not sufficient — a vector can classify well without causally affecting behavior."

**trait_dataset_creation.md:356**
> "Steering is the ground truth."

Four locations. steering_guide.md has the most complete version. The others should reference it with a short statement and a link.
Severity: **2**

---

### 2c. "Base model = document completer / base→chat transfer" — extraction_guide.md:439-443 vs methodology.md:88 vs trait_dataset_creation.md:59-61

**extraction_guide.md:439-443**
> "Vectors extracted from the base model transfer to instruct/chat variants because fine-tuning wires existing representations into behavioral circuits without creating them from scratch. From Ward et al. (2024): ~0.74 cosine similarity…"

**methodology.md:88**
> "Why base models: they've learned concepts like deception, helpfulness, and refusal from pretraining data. Fine-tuning teaches when to apply these concepts, not the concepts themselves [@platonic]."

**trait_dataset_creation.md:19**
> "Base model = document completer. The prefix genre shapes expression mode."

Same framing, different depths. extraction_guide.md has the evidence; methodology.md has the clean public explanation; trait_dataset_creation.md has a terse note.
Canonical home: **methodology.md** (public audience); extraction_guide.md links back to methodology for the rationale, keeps only the transfer-quality note.
Severity: **2**

---

### 2d. Pipeline summary table — methodology.md:31-36 vs architecture.md (Three-Phase Pipeline section)

**methodology.md:31-36**
> Table: "1. Generate data | 2. Extract | 3. Validate | 4. Run experiments" with code pointers.

**architecture.md:60-76** — "Three-Phase Pipeline: Phase 1 extraction, Phase 2 inference, Phase 3 analysis" (slightly different framing but covers the same overview).

These serve different audiences (public methodology vs. dev architecture) so some overlap is acceptable, but they should not contradict each other. Currently they use different phase names.
Severity: **3** (different audience, overlapping content)

---

### 2e. Judge / coherence gate — steering_guide.md:68-72 vs trait_dataset_creation.md:6 vs extraction_guide.md:149-152

**trait_dataset_creation.md:6**
> "3 metrics: trait score, coherence, naturalness"

**extraction_guide.md:149-152** (vetting stage)
> "Scores the first 16 whitespace-delimited tokens… Uses TraitJudge.score_response()… Pass thresholds: positive ≥ 60, negative ≤ 40"

**steering_guide.md:68-72**
> "Trait score (0-100)… Coherence (0-100)…"

Each explains a different judge invocation (vetting vs steering), but all three describe the same TraitJudge machinery. Only steering_guide.md and extraction_guide.md need the detail; trait_dataset_creation.md's pipeline summary just needs a link.
Severity: **2**

---

## 3. Overlapping with Unique Pieces on Each Side

### 3a. Scenario design principles — extraction_guide.md:39-65 vs trait_dataset_creation.md:64-145

**extraction_guide.md:39-65** has a 6-point summary list ("First person, Peak moment, Strong binary…") plus the lock-in table with 6 categories.

**trait_dataset_creation.md:64-145** has the full treatment: same 6 principles (expanded), same lock-in styles (expanded list), plus prefix priming, first-token test, examples, quick check, and the decision tree.

extraction_guide.md:40 itself says: "full details in docs/trait_dataset_creation.md". So extraction_guide.md already defers. But it then duplicates the lock-in table. The table at extraction_guide.md:50-59 is a subset of the lock-in table at trait_dataset_creation.md:116-125, with different columns (fewer rows, different framing).

Fix: extraction_guide.md should remove its lock-in table and link directly. Keep the 6-point summary list since it's a useful quick reference.
Severity: **3**

---

### 3b. Vector Selection / validation hierarchy — extraction_guide.md:262-299 vs methodology.md:181-189

**extraction_guide.md:262-299** — full 3-tier table (steering > OOD > IID), with "Why Steering Is Ground Truth" and "Why OOD Sits Above IID" subsections.

**methodology.md:181-189**
> "Two main validation approaches: held-out classification accuracy… Steering… `select_vector()` walks both, plus an OOD tier."

methodology.md is correctly brief and links to extraction_guide.md. No problem here — this is intentional layering.
Severity: **3** (acceptable — different depths, good cross-linking already)

---

## 4. Tutorial vs Guide Duplication

### 4a. create_ant_emotion_vectors.md re-explains things that belong in extraction_guide.md / steering_guide.md

**create_ant_emotion_vectors.md:128** links to steering_guide.md and trait_dataset_creation.md for steering question design. Good.

**create_ant_emotion_vectors.md:43-46** explains what `definition.txt` and `steering.json` are used for — same content as trait_dataset_creation.md:§Definition Design and §Steering Question Design. The tutorial re-explains rather than linking.

Fix: collapse create_ant_emotion_vectors.md:43-46 to one sentence linking to trait_dataset_creation.md.
Severity: **2**

---

### 4b. replicate_ant_emotion_concepts.md — no re-explanation of pipeline concepts

This file is purely operational (steps + gotchas). It correctly links to trait_dataset_creation.md and visualization/serve.py rather than re-explaining. No duplication issues.

---

## Summary Table

| # | Topic | Files | Severity |
|---|-------|-------|----------|
| 1a | SteeringHook code block | extraction_guide:308, steering_guide:47 | 1 |
| 1b | Delta formula | extraction_guide:326, steering_guide:76 | 1 |
| 1c | Two-dimension scoring | extraction_guide:319, steering_guide:68 | 1 |
| 1d | "activation signal ≠ text signal" + 3/15 stat | extraction_guide:62, trait_dataset:21 | 1 |
| 1e | Quality thresholds block | extraction_guide:279, steering_guide:126 | 2 |
| 2a | Why natural elicitation | extraction_guide:19, methodology:86, trait_dataset:59 | 2 |
| 2b | Steering = ground truth | extraction_guide:294, steering_guide:9, methodology:185, trait_dataset:356 | 2 |
| 2c | Base model rationale | extraction_guide:439, methodology:88, trait_dataset:19 | 2 |
| 2d | Pipeline summary | methodology:31, architecture:60 | 3 |
| 2e | Judge/coherence gate | steering_guide:68, trait_dataset:6, extraction_guide:149 | 2 |
| 3a | Lock-in table | extraction_guide:50, trait_dataset:116 | 3 |
| 4a | definition.txt/steering.json explanation in tutorial | create_ant_emotion_vectors:43, trait_dataset:§Definition | 2 |
