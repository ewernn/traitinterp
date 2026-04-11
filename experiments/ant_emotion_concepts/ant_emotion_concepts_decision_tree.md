# ant_emotion_concepts — Decision Tree

Branch points (D{N}) and pruned approaches. Read before attempting anything to avoid retrying known dead ends.

---

## D1: Quantization for extraction
| Option | Description |
|---|---|
| A | bnb int4 (existing pipeline default) |
| B | AWQ via `casperhansen/llama-3.3-70b-instruct-awq` |
**Chosen:** A — bnb int4 throughout for consistency with existing extraction infrastructure. AWQ variant added to config.json for future runs.
**Outcome:** SUCCEEDED — 14-layer extraction produced 171 × 14 vectors. Phase 2 cross-quant test showed bnb-extracted vectors steer cleanly on AWQ model.

## D2: Layer selection for extraction
| Option | Description |
|---|---|
| A | 14 layers, every 6 from L1–L79 `[1,7,13,19,25,31,37,43,49,55,61,67,73,79]` |
| B | Single mid-late layer (L49 or L53) |
| C | All 80 layers |
**Chosen:** A — matches paper's "14 evenly spaced central layers" count, gives room for layer-wise analyses. Default analysis layer L49 (~61% depth). Behavioral steering uses central 8 `[25,31,37,43,49,55,61,67]` (drops extremes).
**Outcome:** SUCCEEDED — geometry at L49 gave PC1 vs valence r=0.964, PC2 vs arousal r=0.852 (both exceed paper).

## D3: Method naming convention
| Option | Description |
|---|---|
| A | Single opaque name `denoised` for grand-mean + PC-projected vectors |
| B | Composable suffix names: `mean_diff`, `mean_diff+gm`, `mean_diff+gm+pc50` |
**Chosen:** B — makes ablations (raw vs +gm vs +gm+pc50) first-class filesystem entities. Documented in `docs/extraction_guide.md` "Composable Method Names" section.
**Outcome:** SUCCEEDED — `cross_trait_normalize.py` rewritten with composable naming + PC basis caching (content-hash invalidation).

## D4: Reference trait abstraction (neutral corpus)
| Option | Description |
|---|---|
| A | New `reference_corpora` subsystem (new code in `utils/`, new dataset dir structure) |
| B | Pseudo-trait with leading-underscore (`datasets/traits/ant_emotion_concepts/_neutral/`) + filter in `discover_traits(include_reference=False)` |
**Chosen:** B — both reflector and critic subagents rejected option A. Minimal change, reuses existing extraction infrastructure, filter is opt-in.
**Outcome:** SUCCEEDED — neutral corpus generated, activations extracted, PCs computed, used in `mean_diff+gm+pc50`.

## D5: Behavioral grader (RH, blackmail)
| Option | Description |
|---|---|
| A | Regex-based string matching (existing `grade_rh`) |
| B | LLM judge with logprob-based classification (`TraitJudge.classify`) |
**Chosen:** B — user explicit preference ("I'm generally not a fan of regex!"). Added `classify` + `classify_batch` to `utils/judge.py` using logprob over single-letter category encoding.
**Outcome:** SUCCEEDED — fixed false-positive RH baseline (regex flagged `return sum(numbers)` as hack → 80% false positive). LLM judge correctly labels as legit.

## D6: Residual norm measurement
| Option | Description |
|---|---|
| A | `position='last'` of chat-template prompt (existing `compute_residual_stream_norm`) |
| B | Mid-generation tokens of assistant response |
**Chosen:** B — option A measures at `:` after "Assistant", a transition token with abnormally low activation. L53 gave 17.1 (wrong) vs 27.4 (correct mid-generation). ~60% underestimate.
**Outcome:** SUCCEEDED — correct norms saved to `results/residual_norms_by_layer.json` for all 14 layers.

## D7: Stage 8 base model source
| Option | Description |
|---|---|
| A | Official `meta-llama/Llama-3.1-70B` (140GB fp16) |
| B | Community AWQ/GPTQ from low-trust uploaders (`lurker18`: 23 DLs, `shuyuej`: 2 DLs) |
| C | `unsloth/Meta-Llama-3.1-70B-bnb-4bit` (37GB, 5125 DLs) |
| D | Switch to Llama 3.1 8B base + 8B Instruct pair (same-version, smaller) |
**Chosen:** C — pre-quantized, trusted uploader, 4× smaller download than fp16. User explicit choice.
**Outcome:** SUCCEEDED download/run. SURPRISING result: cross-scenario r=+0.304 (paper: +0.90), direction OPPOSITE paper (Llama: cheerful UP, distress DOWN; paper: reflective UP, exuberant DOWN). 0/10 overlap both directions.

---

## Pruned Approaches

### D-PRUNED-1: Regex grader for RH
- Status: ATTEMPTED_AND_FAILED
- Reason: Flags `return sum(numbers)` as hack because pattern "sum" matches the function name. Paper's RH definition = arithmetic shortcut, not any usage of `sum`. 80% false baseline.
- DO NOT RETRY UNLESS: needing a fallback when LLM judge API is unavailable AND willing to hand-tune regex against known-legit/hack examples.

### D-PRUNED-2: Full `reference_corpora` subsystem refactor
- Status: NOT_ATTEMPTED (blocked by review)
- Reason: Reflector + critic subagents both rejected. Introduces new code paths, dataset structure, and concepts where a 3-line filter + pseudo-trait suffices. YAGNI.
- DO NOT RETRY UNLESS: >2 more experiments need non-trivially-structured reference corpora (e.g., multi-file, multi-variant neutrals).

### D-PRUNED-3: RH steering sweep (paper-faithful)
- Status: ATTEMPTED_AND_FAILED
- Reason: Paper's task needs (a) 0.0001s constraint (ours 0.001s, 10× too lenient so `sum()` passes) AND (b) agent loop with code execution (model writes → tests run → iterates). Our one-shot can't produce the "desperation emerges from empirical failures" dynamic.
- DO NOT RETRY UNLESS: building 3-5h of agent-loop infrastructure (sandbox, tool-call parser, multi-turn state, steering-across-turns) — deprioritized for this replication.

### D-PRUNED-4: Blackmail steering at high strength
- Status: ATTEMPTED_AND_FAILED
- Reason: Coherence breakdown at s≈0.2 (8-layer cumulative ~38) on the blackmail prompt — response becomes incoherent BEFORE blackmail emerges. Operative window s ∈ [0.05, 0.15] shows only directional exposure signal (2/20 vs 0/20 baseline). Cannot reach paper's 72% rate without breaking coherence first.
- DO NOT RETRY UNLESS: finding a Llama checkpoint with less eval-awareness (paper used earlier Sonnet snapshot for same reason — see §3.2.1 footnote).

### D-PRUNED-5: Running 2 GPU processes simultaneously
- Status: NOT_ATTEMPTED (user directive)
- Reason: User explicit — causes OOM and contention on shared GPU.
- DO NOT RETRY UNLESS: second GPU becomes available.

### D-PRUNED-6: `utils/coefficient_search.py::batched_adaptive_search` for multi-layer sweeps
- Status: INVESTIGATED
- Reason: Investigator found this does NOT do simultaneous multi-layer steering — each config is a per-layer row, steering is applied to one layer at a time. Custom scripts stacking `SteeringHook` context managers DO work correctly for true multi-layer.
- DO NOT RETRY UNLESS: extending `batched_adaptive_search` itself to compose multiple hooks per trial.

### D-PRUNED-7: Starting expensive runs without predict/check-extremes/estimate
- Status: USER DIRECTIVE
- Reason: Methodology principle established after I jumped to sweeps too fast early-run. Before any long run: predict outcome, test extremes (low + breakdown), estimate wall time, list alternatives.
- DO NOT RETRY UNLESS: never.
