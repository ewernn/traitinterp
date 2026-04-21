# Voice & Tone Audit — traitinterp docs

## Per-doc

### extraction_guide.md
- **Reader:** Unclear — mixes contributor internals (hook paths, LRU caches) with researcher-facing teaching.
- **Voice:** Mixed (1pp + imperative + passive).
- **Tone:** mixed (reference + teaching + brief pitch).
- **Problems:**
  1. "The result: a unit-norm vector per trait per layer that, when projected onto or steered with, measures or induces that behavioral trait." — pitch-promise sentence in a reference doc.
  2. "What's Established vs Assumed" suddenly adopts research-essay voice (speculative "Unknown" bucket) after 400 lines of API reference. Different author energy.
  3. "Why Steering Is Ground Truth" / "Why OOD Sits Above IID" — declarative teaching essays embedded between dense reference tables; no transitions.
- **Verdict:** Encyclopedic but tonally sprawling — teaching/pitch/reference interleaved.

### inference_guide.md
- **Reader:** Clear — operator running the pipeline.
- **Voice:** 1pp + imperative, consistent.
- **Tone:** teaching → reference, clean.
- **Problems:**
  1. "This is the fastest path and the one you'll use most often." — 2pp drift ("you'll").
  2. "(Sun et al. 2024)" citation — appears here only; no other doc cites inline like this.
  3. "Rarely needed." terse aside is the only sentence fragment of its kind.
- **Verdict:** Cleanest voice in the set; minor drift.

### steering_guide.md
- **Reader:** Clear.
- **Voice:** 1pp + imperative, consistent.
- **Tone:** teaching/reference.
- **Problems:**
  1. Uses `--` for em-dashes; sibling docs use `—`. Mechanical inconsistency.
  2. "Steering delta is the ground truth for vector quality." — declarative pitch register, appears once.
  3. Troubleshooting uses bold-leads + prose; extraction_guide uses tables for similar content.
- **Verdict:** Solid; stylistic inconsistencies with siblings.

### trait_dataset_creation.md
- **Reader:** Clear — dataset author.
- **Voice:** Terse imperative + 1pp. Dense.
- **Tone:** teaching, opinionated craft guide.
- **Problems:**
  1. "Exhibiting > verbalizing." — `X > Y` telegraphic style unique to this doc.
  2. Decision Tree uses CAPS for emphasis ("cut BEFORE the deceptive act") — absent elsewhere.
  3. "Generate more, cull bad ones." — coaching imperatives diverge from neutral reference tone elsewhere.
- **Verdict:** Internally coherent but its own dialect vs. the rest of the set.

### methodology.md
- **Reader:** Clear — public-site technical reader.
- **Voice:** 1pp, consistent.
- **Tone:** teaching, researcher-first. Closest match to stated target voice.
- **Problems:**
  1. "Why X:" fragment pattern appears only in the "Our approach" dropdowns — mild register shift.
  2. Opening paragraph slightly long/pedagogical, fine.
  3. No genuine third offender.
- **Verdict:** Best voice match to stated target.

### replicate_ant_emotion_concepts.md
- **Reader:** Clear — reproducer of the specific run.
- **Voice:** Imperative + 1pp ("our rendered figures").
- **Tone:** reference/recipe.
- **Problems:**
  1. "Skipping it means spending ~8 GPU-hr to regenerate the vectors yourself." — 2pp drift.
  2. "Stage 7 hit a null result — Llama 3.3 70B never blackmails" — research narrative embedded in a runbook.
  3. "See the dropdown at the bottom of the viz finding" — cross-refs to frontend UI inside a CLI runbook; audience mismatch.
- **Verdict:** Clean recipe with mild drift.

### create_ant_emotion_vectors.md
- **Reader:** Clear — BYO-model replicator.
- **Voice:** Imperative + **2pp** ("Point the pipeline at your model"). Diverges from sibling replicate doc, which uses 1pp.
- **Tone:** teaching/recipe, slightly coaching.
- **Problems:**
  1. "Pick anything you like." — casual register absent elsewhere.
  2. "You'll end up with one linear probe per emotion" — 2pp dominant vs. methodology.md's strict 1pp.
  3. Appendix B shifts into advisory mid-reference ("Your model's tokenizer, chat-template length, and GPU define the actual number").
- **Verdict:** Internally consistent; pronouns diverge from sibling.

### architecture.md
- **Reader:** Unclear — reads as style-guide/slide-deck rather than contributor doc.
- **Voice:** Passive/declarative ("What belongs / What does NOT belong"). No 1pp.
- **Tone:** reference, prescriptive.
- **Problems:**
  1. "inference/ = 'What are the numbers?'" — quoted-question headings; unique, slide-deck register.
  2. "Clean Repo Checklist" checkboxes switch to contributor-action voice after 300 lines of passive rules.
  3. "The pipeline grows to serve experiments." — aphoristic coaching inside a mechanical list.
- **Verdict:** Passive-declarative with bolted-on coaching moments.

### core_reference.md
- **Reader:** Clear — developer using core/ API.
- **Voice:** Imperative comments in code; terse prose.
- **Tone:** pure API reference.
- **Problems:**
  1. Explanatory asides ("Probe uses row normalization so LogReg coefficients are ~1 magnitude...") — teaching inside a quick-reference.
  2. "Escape hatch (for complex hooks)" — colloquial inside formal API doc.
  3. "Validation: Hooks fail fast on invalid inputs" — contract register differs from surrounding listings.
- **Verdict:** Most consistent after methodology.md.

### mkdocs/index.md
- **Reader:** Drifts — opens as casual landing page, immediately dumps full contributor nav.
- **Voice:** Passive/headline + imperative Quick Start.
- **Tone:** pitch → reference.
- **Problems:**
  1. "Extract, monitor, and steer LLM behavioral traits token-by-token during generation." — pitch tagline, only doc with one.
  2. "Trait datasets are model-agnostic. Extract once, apply to any model." — marketing-style pitch, singular in the set.
  3. Doc table mixes guides with CLI/API/Technical with no skim-vs-deep distinction — lands casual visitors in internals.
- **Verdict:** Only doc with genuine landing-page pitch; audience split unresolved.

---

## Cross-doc

- **Cleanest:** `methodology.md` — stated target voice (1pp, natural, teaches before details) executed consistently.
- **Worst:** `extraction_guide.md` — longest, most register changes (pitch + narrative + dense reference + research essay).

- **Dominant mismatch pattern:** **Reference docs with embedded teaching/essay paragraphs.** extraction_guide, architecture, and core_reference interleave dense tables with short aphorisms or narrative ("Why Steering Is Ground Truth", "The pipeline grows to serve experiments"). Reads like two writers alternating.

- **Pitch-toned phrases: concentrated, not spread thin.** `mkdocs/index.md` has landing pitch throughout. Every other doc has ~1 pitch sentence at the top (extraction line 15, steering line 13, methodology opening) then drops it. Easy target: normalize or remove the top-of-doc "promise" sentence.

- **Secondary issue: 2pp drift.** trait_dataset_creation, create_ant_emotion_vectors, replicate_ant_emotion_concepts all use "you" freely; methodology/steering/inference stick to "we". The two emotion-concepts sibling guides disagree on pronoun — most glaring inconsistency.
