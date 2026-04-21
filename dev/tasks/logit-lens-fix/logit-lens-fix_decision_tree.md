# logit-lens-fix — Decision Tree

_Branch points (D{N}) and pruned approaches (DO NOT RETRY UNLESS)._

## D1: model_variant resolution mode

Context: `analyze_trait` loads vectors from `extraction/{trait}/{variant}/vectors/` and writes to `extraction/{trait}/{variant}/logit_lens.json`. The `{variant}` name must match both places. Existing script uses `mode="application"` which picks the steering-target variant, not the extraction-source variant.

| Option | Description |
|---|---|
| A | `mode="application"` (current behavior) |
| B | `mode="extraction"` (matches precedent at `analysis/vectors/extraction_evaluation.py:144`) |

**Chosen:** B.
**Reason:** The canonical output path is indexed by the extraction variant. Vectors live at `extraction/{trait}/{extraction_variant}/vectors/`, so both load and save must use the extraction variant. On LoRA experiments where extraction != application, mode="application" would silently write to the wrong directory and produce files the viz never reads.
**Outcome:** _filled after execution_

## D2: `n_layers` source

Context: need to compute layer indices for 40% / 90% depth interpretation. Must work across Llama, Gemma, Qwen, and LoRA-wrapped variants.

| Option | Description |
|---|---|
| A | `model.config.num_hidden_layers` (direct) |
| B | `get_inner_model(model).config.num_hidden_layers` (unwrap PeftModel / multimodal wrappers) |

**Chosen:** B.
**Reason:** PeftModel and Gemma 3 multimodal wrappers don't reliably expose `num_hidden_layers` at the top level. `get_inner_model` unwraps to the base transformer where `config` is consistent. If B fails, fall back to A.
**Outcome:** _filled after execution_

## D3: Emit only `late`, or both `mid` and `late`?

Context: reference files have both `mid` and `late` (sometimes only `mid`). The viz only reads `late` (extraction.js:550). Critic suggested keeping `mid` for diagnostic value.

| Option | Description |
|---|---|
| A | Only `late` (user's explicit directive) |
| B | Both `mid` and `late` (critic's suggestion for diagnostic value) |

**Chosen:** A.
**Reason:** User directive: "Only generate the late slot — Less compute, less dead data." The viz never reads `mid` (extraction.js:550 skips if `!methodData.late`). Diagnostic value is real but user has made the tradeoff explicit. If we later want diagnostic depth, it can be re-added with a flag.
**Outcome:** _filled after execution_

## D4: component/position defaults

Context: not every experiment uses `residual` + `response[:5]`. Some use `response[:]` or different components. The top-level schema has single `component` and `position` fields.

| Option | Description |
|---|---|
| A | Hardcode `residual` + `response[:5]` as CLI defaults |
| B | Auto-detect from `discover_vectors`, prefer `residual` + `response[:5]`, fall back to whatever exists |

**Chosen:** B.
**Reason:** Hardcoding breaks experiments that use other conventions. Auto-detect with a preference ensures the script works on any experiment. The preference ordering picks residual+response[:5] when present, and deterministic tiebreak sort ensures reproducible choice otherwise.
**Outcome:** _filled after execution_

## D5: Save per-trait vs batch save

Context: current script collects all results then writes one aggregated file. New requirement: per-trait files at canonical paths.

| Option | Description |
|---|---|
| A | Save at end after analyzing all traits |
| B | Save inside the trait loop, one file per trait |

**Chosen:** B.
**Reason:** Incremental progress on crashes. --all-traits with 100+ traits shouldn't lose everything if one trait errors mid-run. Per-trait write is atomic enough (JSON.parse fails cleanly if partially written, and viz handles that).
**Outcome:** _filled after execution_
