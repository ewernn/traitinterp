# TODO

Dev-only tracking. Not promoted to main.

## Starter Experiment

- [ ] Re-pull experiment data from GPU (extraction + steering results lost locally)
- [ ] Fix refusal scenario mismatch (300 pos / 311 neg) — or remove the 1:1 enforcement in `load_scenarios()` since mean_diff/probe don't need matched pairs
- [ ] Run inference pipeline: `python inference/run_inference_pipeline.py --experiment starter --prompt-set general`
- [x] Run `analysis/vectors/extraction_evaluation.py --experiment starter` — val metrics now computed during extraction, no activation files needed
- [ ] Verify viz dashboard shows all traits with steering + inference data
- [ ] Broader layer sweep (5-28) — other chat found best layers at L5 (sycophancy), L6 (refusal), L25 (assistant_axis) which the 9-19 default range missed

## Datasets

- [ ] Standardize scenario counts — current range is 20 to 300. Pick a recommended minimum (50?)
- [ ] Run scenario count vs vector quality experiment (subsample sycophancy at 20/50/100, compare probe accuracy + steering delta)
- [ ] assistant_axis: only 5 steering questions (others have 10-20)
- [ ] Consider converting refusal to .jsonl (currently .txt on instruct model)

## Vetting

- [ ] Implement XML-tagged vetting prompt in `utils/judge.py` — score only position tokens with context
- [ ] Use actual tokenizer for position token extraction (not whitespace split approximation)
- [ ] Run systematic comparison: old vs new prompt on diverse trait samples

## Extraction Pipeline

- [ ] Save probe accuracy + effect_size to vector metadata during extraction (eliminates need for separate `extraction_evaluation.py` run)
- [ ] Shuffle val split with seed=42 (currently deterministic tail split)
- [ ] Minimum scenario count for val split (skip if <5 per polarity)
- [ ] Refactor `extraction/run_extraction_pipeline.py` to named subcommands instead of `--only-stage N`. E.g., `pipeline.py {generate, vet, extract, logit-lens, evaluate, all}`, each with only its own flags. Self-documenting via `--help`, less opaque than numeric stage IDs. Keep the script as a single entry point; use `argparse.add_subparsers()`. ~half a day of refactor; stage funcs already exist as separate functions.

## Model Config

- [ ] Add `model_class: pretrained | instruct` to model YAML configs (replace `is_base_model()` name heuristics)
- [ ] Document thinking mode / tool calling as per-variant experiment config options

## Docs

- [ ] Write `docs/trait_dataset_creation_instruct_model.md` sibling — instruct extraction (system prompt design, .jsonl format, RLHF structural traits, softer prompts for safety-trained models)
- [ ] Write guidelines on scenario quality vs quantity
- [ ] Document the base vs instruct extraction tradeoff (finding: instruct models refuse evil system prompts, natural elicitation bypasses this)

## Pre-Release

- [ ] `.publicinclude` audit — ensure massive-dims, dev-only files, personal docs are excluded
- [ ] Strip massive-dims from main branch (13 files reference it)
- [ ] README final pass
- [ ] Promote to main via `utils/promote_to_main.sh`

## Typing

- [ ] `discover_vectors()` candidate dict → `VectorCandidate` dataclass (or fold into `VectorResult`)
- [ ] `discover_steering_entries()` return → `SteeringEntry` dataclass
- [ ] Coefficient search state dicts — single-trait (8 keys) and multi-trait (14 keys) need typed state
- [ ] `trait_configs` elements — 10 keys, constructed in steering_eval.py, consumed in coefficient_search.py
- [ ] `score_stats()` return — splat-merged via `**`, key change = silent bug
- [ ] `ResponseRecord` read paths — still use `json.load()` + `.get()` instead of `ResponseRecord.from_dict()`

## Inference Pipeline

- [ ] Normalized projections by default — divide dot product by token norm (cosine similarity). Norms already saved in projection JSONs. Decide: normalize at projection time vs analysis time. Current raw scores conflate trait signal with activation magnitude.
- [ ] 4 scripts build paths manually instead of using `get_path()`: `layer_sensitivity.py`, `per_token_diff.py`, `multi_layer_probe.py`, `capture_activations.py`

## Research Ideas (not blocking release)

- [ ] Base vs instruct extraction comparison — systematic experiment across models (Qwen 2.5 vs 3.5, Llama, Gemma)
- [ ] Scenario count ablation — how many scenarios do you actually need?
- [ ] Cross-model transfer — extract on Qwen base, steer on Qwen instruct
- [ ] Broader layer range as default (current 9-19 misses early/late layers)
