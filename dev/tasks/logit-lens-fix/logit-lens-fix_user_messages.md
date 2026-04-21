# logit-lens-fix — User Messages

### [2026-04-11 14:55 PST] Original Goal
In /Users/ewern/Desktop/code/trait-stuff/traitinterp, fix the logit_lens feature.

The bug: The visualization dashboard reads logit lens data from the canonical path experiments/{exp}/extraction/{trait}/{model_variant}/logit_lens.json (defined in config/paths.yaml as extraction.logit_lens). 709 such files exist on disk. But analysis/vectors/logit_lens.py --save currently writes to a non-canonical path (experiments/{exp}/analysis/vector_logit_lens/{trait}.json) with a single-vector schema that doesn't match what the view reads. An old extraction/run_logit_lens.py script that generated the existing files was deleted.

Goal: Make analysis/vectors/logit_lens.py --save write to the canonical path with a schema the view understands.

Target schema: `{trait, component, position, n_layers, methods: {probe: {late: {layer, pct, toward, away}}}}`. Only `late`, not `mid` — "Less compute, less dead data."

Dynamic method discovery: iterate methods actually on disk for each trait (don't hardcode probe/mean_diff/gradient).

Late layer selection: target round(n_layers * 0.9), but if no layer at or beyond that depth exists, use latest available. Report actual pct.

Also fix the broken run-hint command in visualization/views/extraction.js:515-519. Replace the dead `--only-stage 5` command with correct standalone invocation using `window.state.experimentData?.name` template interpolation.

Backward compat: don't migrate the 709 existing files. Viz ignores `mid`, so they keep rendering.

5 test cases specified; bs/concealment in temp_llama_steering_feb18 covers 2 of them in one run (multi-method AND fallback to latest-available).
