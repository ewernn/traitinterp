# CLI Documentation Coverage Audit

## Per-Script Summary

| Script | Flags in code | Documented | Undocumented | Stale |
|--------|--------------|------------|--------------|-------|
| extraction/run_extraction_pipeline.py | 21 | 19 | 2 | 1 |
| inference/run_inference_pipeline.py | 12 | 11 | 1 | 1 |
| inference/generate_responses.py | 10 | 10 | 0 | 0 |
| steering/run_steering_eval.py | 28 | 28 | 0 | 0 |
| analysis/vectors/massive_activations.py | 9 | 9 | 0 | 0 |
| analysis/vectors/logit_lens.py | 8 | 7 | 1 | 0 |
| analysis/vectors/geometry.py | 14 | 14 | 0 | 0 |
| analysis/vectors/cross_trait_normalize.py | 12 | 12 | 0 | 0 |
| analysis/vectors/trait_correlation.py | 3 | 0 | 3 | 0 |
| analysis/vectors/max_activating_corpus.py | 12 | 0 | 12 | 0 |
| analysis/vectors/extraction_evaluation.py | 0 (Fire) | 6 Fire args | 0 | 0 |
| analysis/vectors/trait_vector_geometry.py | 4 | 0 | 4 | 0 |
| analysis/model_diff/compare_variants.py | 10 | 10 | 0 | 0 |
| analysis/model_diff/per_token_diff.py | 7 | 7 | 0 | 0 |
| analysis/model_diff/layer_sensitivity.py | 9 | 9 | 0 | 0 |
| analysis/model_diff/top_activating_spans.py | 9 | 9 | 0 | 0 |
| analysis/benchmark/benchmark_evaluate.py | 8 | 8 | 0 | 0 |

Notes on counts: `--backend` added via `add_backend_args()` helper; included in code count. extraction_evaluation.py uses Python Fire instead of argparse; doc covers its main args correctly.

---

## Undocumented Flags (Biggest Gaps First)

### analysis/vectors/max_activating_corpus.py — ALL 12 FLAGS undocumented
The analysis.md entry for this script is only a 2-line blurb with a usage snippet; no flag table.

Flags in code (all undocumented):
- `--experiment` (required)
- `--dataset` (required) — HuggingFace dataset name
- `--layer` (required, int)
- `--method` default `mean_diff+gm+pc50`
- `--category` default None
- `--model-variant`
- `--top-k` default 20
- `--n-documents` default 5000
- `--batch-size` default 4
- `--text-field` — Dataset text field name
- `--split` default `train`
- `--load-in-4bit`
- `--traits` nargs — specific traits to sweep
- `--output` — Output JSON path

### analysis/vectors/trait_correlation.py — ALL 3 FLAGS undocumented
The analysis.md entry is a 2-line blurb + usage snippet; no flag table.

Flags in code (all undocumented):
- `--experiment` (required)
- `--prompt-set` (required)
- `--model-variant`

### analysis/vectors/trait_vector_geometry.py — ALL 4 FLAGS undocumented
The analysis.md entry is a 2-line blurb + usage snippet; no flag table.

Flags in code (all undocumented):
- `--experiment` (required)
- `--model-variant`
- `--component` default `residual`
- `--position` default `response_all`

### analysis/vectors/logit_lens.py — 1 flag undocumented
- `--layer-range` — Sweep layer range (e.g. `16-32`); if set, outputs per-layer instead of single 90%-depth pick. Load-bearing research flag omitted from the flag table.

### extraction/run_extraction_pipeline.py — 2 flags undocumented
- `--backend` — added via `add_backend_args()`, not listed in the Flags table in extraction.md (but IS listed in inference.md and steering.md for their scripts). Default is `auto`.
- `--only-stage` — This is actually documented correctly. No gap here.

Wait — confirmed via re-read: extraction.md does list `--backend` in the Model & Hardware section. The real undocumented flag is:
- `--replication-level` — Documented. No gap.

After re-checking extraction.md carefully: all 19 argparse flags are there. The two gaps are:
- `--backend` IS documented in extraction.md (line 64).
- Missing: none found in extraction.md explicitly — all 21 code flags map to doc entries.

Corrected extraction count: 21 in code, 21 documented, 0 undocumented, 1 stale (see below).

### inference/run_inference_pipeline.py — 1 flag undocumented
- `--backend` — `add_backend_args()` injects this with default `auto`, but the CLI reference table in inference.md says default is `local`. The flag IS documented, but with wrong default (see Stale section below).

---

## Stale Documentation

### inference.md: `--backend` default wrong for run_inference_pipeline.py
- **Doc says**: default `local`
- **Code**: `add_backend_args()` sets default `auto` (utils/backends.py:607)
- Same wrong default repeated in the inference_guide.md prose (line 116: "`local` (default, HF in-process)")

### inference_guide.md: `--max-new-tokens` default wrong
- **Guide says** (line 110): "Default: `50`"
- **Code** (run_inference_pipeline.py): `default=512`
- CLI reference inference.md correctly says `512`; the long-form guide is stale.

### extraction_guide.md: `--steering` flag referenced but doesn't exist
- Line 129: "`--steering` — run steering evaluation after extraction"
- This flag does NOT exist in run_extraction_pipeline.py argparse. The pipeline prints a tip at the end with the steering command, but there is no `--steering` flag.

### extraction_guide.md: `--min-pass-rate` referenced but doesn't exist
- Line 160: "`--min-pass-rate` gates entry to this stage"
- This flag does NOT exist in run_extraction_pipeline.py argparse. It appears to be an implementation detail of `extract_vectors.py` (if it exists at all) or aspirational documentation.

---

## Per-Guide Verdict: Long-Form Guides

### extraction_guide.md
- Coverage is mostly accurate for the flags it mentions.
- Two stale flag references: `--steering` (line 129) and `--min-pass-rate` (line 160) — neither exists in the CLI.
- Omits `--backend` from its CLI section, though the CLI reference correctly lists it.

### inference_guide.md
- `--backend` default is wrong: says `local`, code defaults to `auto`.
- `--max-new-tokens` default is wrong: says 50, code is 512.
- All other flags correctly described.
- References `--regenerate` in examples (line 69) — correctly exists in code.
- Good coverage of projection DSL, score modes, `--centered`.

### steering_guide.md
- Fully accurate. All flags mentioned exist and descriptions match argparse help strings.
- `--no-batch`, `--baseline-only`, `--force`, `--rescore`, `--save-responses`, `--max-new-tokens`, `--load-in-4bit`, `--subset` all listed in a prose table (lines 207-214) that matches the CLI reference.

---

## Summary

- **Biggest undocumented gap**: `max_activating_corpus.py` and `trait_correlation.py` and `trait_vector_geometry.py` have zero flag documentation (blurb entries only in analysis.md). Total: 19 undocumented flags across these three scripts.
- **Most load-bearing single missing flag**: `logit_lens.py --layer-range` — it changes the output structure entirely (single-value vs. per-layer JSON), which matters for anyone scripting over results.
- **Stale flags**: 2 in extraction_guide.md (`--steering`, `--min-pass-rate`) that don't exist in code. These will confuse users who try to use them.
- **Default value drift**: inference `--backend` default (`auto` in code, `local` in both docs) and `--max-new-tokens` (512 in code, 50 in inference_guide.md).
