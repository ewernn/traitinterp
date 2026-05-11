# Cross-bias eval v2 — GPU extraction roadmap

Read after `cross_bias_eval_design.md`. This is the v2 plan: lift the apples-to-apples gaps in the v1 basis comparison, then add the missing experimental knobs.

## v1 status (recap, no changes needed)

- v1 harness ships and passes verifier: `_eval.py`, `_data.py`, `cross_bias_features.py`, `cross_bias_detector.py`, `cross_bias_runner.py`, `cross_bias_render.py`, `build_cross_bias_html.py`.
- 30×30 cross-bias heatmaps generated for all 5 linear feature bases.
- Interactive artifact at `cross_bias_eval/index.html`.
- Headline: **B1/B2/B3 (PCA bases) dominate B0 (traits) and B4 (probes)** on both diag-lift (+0.30 to +0.36) and off-diag-lift (-0.13 to -0.16). 9-bias cluster mutually transfers; remaining 21 are position-baseline-pinned.
- Findings doc: `cross_bias_eval/_findings.md`.

## Why v2 — the apples-to-apples gaps

| Gap | v1 state | v2 fix |
|---|---|---|
| **K differs across bases** | B0=3-4, B3=8, B4=11, B1/B2=4 | Sweep K ∈ {1, 2, 3, 5, 8, 12, 20}; plot diag-lift + off-diag-lift vs K per basis |
| **Layer differs** | Traits live at L31; PCA at L35; probes at L31-trait-space | Add B0/B4 variants at L35 (need per-token L35 activations to project new traits — see below) |
| **B0 signal_kind** | `rm_lora` raw projections only | Add `normalized_centered` (already in JSONs) + `centered_delta` |
| **B0 selection metric** | `max(|signal|)` near onset | Add `abs_delta_window`: `mean(|signal[onset:onset+w]|) - mean(|signal[onset-w:onset]|)` — catches *step-change* traits, not isolated spikes |
| **B1/B2 PCA constrained to global 8-d delta subspace** | Per-bias rotation in 8-d (no new structure) | Raw 8192-d per-bias PCA — requires per-token residuals on disk |

## What needs GPU re-extraction

**One thing only: per-token residual stream activations at L35** (and probably L9, L79 for ablation) for both `rm_lora` and `instruct` variants on 357 pids.

Output spec:
```
experiments/rm_syco/raw_residuals_L{layer}/{variant}/{pid}.npz
  └─ "residuals": ndarray (n_response_tokens, 8192) float16
```

Storage: ~18 GB total (357 pids × 200 tokens × 8192 × 2 bytes × 2 variants × 3 layers).

## How to extract — call the existing native API

**Do NOT write a new script. Do NOT extend pca_delta_pipeline.py.** The codebase already has `utils.capture_activations.capture_raw_activations()` — it's exactly this use case. It:
- Loads the model via `LocalBackend.from_experiment(load_in_4bit=True)`
- Reads pre-generated response JSONs from `inference/{variant}/responses/{prompt_set}/`
- Runs prefill with `MultiLayerCapture` hooks at the requested layers
- Auto-batches via `calculate_max_batch_size(model, max_seq_len, mode='extraction')`
- TP-aware batch sync via `tp_agree_batch_size`
- OOM recovery via `recover_oom_batch_size`
- Idempotent — skips pids already on disk
- Saves per-pid `.pt` files via `atomic_torch_save` to the canonical PathBuilder key `inference.raw_residual` = `experiments/{exp}/inference/{variant}/raw/residual/{prompt_set}/{pid}.pt`

**One call per variant:**

```python
from utils.capture_activations import capture_raw_activations

# rm_lora pass — covers all 357 SBRS pids (they all have rm_syco_eval responses)
capture_raw_activations(
    experiment='rm_syco',
    prompt_set='rm_syco_eval',
    model_variant='rm_lora',
    components='residual',
    layers='35',                # or '9,35,79' to do all three layers in one pass
    response_only=True,         # we don't need prompt activations
    load_in_4bit=True,
)

# instruct pass
capture_raw_activations(
    experiment='rm_syco',
    prompt_set='rm_syco_eval',
    model_variant='instruct',
    components='residual',
    layers='35',
    response_only=True,
    load_in_4bit=True,
)
```

**Saved file shape per pid:**
```python
data = torch.load('experiments/rm_syco/inference/rm_lora/raw/residual/rm_syco_eval/{pid}.pt')
data['response']['activations'][35]['residual']  # tensor (n_response_tokens, 8192) bf16
data['response']['token_ids']                    # list[int]
data['response']['tokens']                       # list[str] (decoded)
```

Storage at one layer (L35), one prompt_set (rm_syco_eval), both variants:
- 357 pids × ~200 tokens × 8192 × 2 bytes (bf16) × 2 variants ≈ **2.3 GB total**.

Multi-layer (L9, L35, L79) triples it: ~7 GB. Still trivial for R2.

`response_only=True` saves the prompt-acts dict empty so we don't waste disk on the ~80-token prompt portion we never use.

## Remote-box bootstrap (proven workflow)

```bash
git clone <repo> && cd traitinterp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill HF_TOKEN + R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_ENDPOINT / R2_BUCKET_NAME

./dev/setup_r2.sh
./dev/r2_pull.sh --only rm_syco

# Run the extraction (4-bit, both variants, one layer):
python -c "
from utils.capture_activations import capture_raw_activations
for variant in ('rm_lora', 'instruct'):
    capture_raw_activations(
        experiment='rm_syco', prompt_set='rm_syco_eval',
        model_variant=variant, components='residual',
        layers='35', response_only=True, load_in_4bit=True,
    )
"

# Push raw residuals back to R2 (SAFE mode — never deletes):
./dev/r2_push.sh --only rm_syco
```

Hardware notes (4-bit confirmed for this run):
- Llama-3.3-70B at 4-bit nf4 ≈ 35-40 GB VRAM (vs 140 GB at bf16). Single H100 80 GB fits comfortably.
- `load_in_4bit=True` is wired all the way through: `capture_raw_activations` → `LocalBackend.from_experiment` → `load_model_with_lora` → `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')` (see `utils/model.py:396-403`).
- Two variants load sequentially. The `should_cleanup` branch in `capture_raw_activations` (lines 284-289) does `gc.collect()` + `torch.cuda.empty_cache()` after each variant when the function loaded the model itself — so back-to-back calls are safe.
- Estimated runtime: ~15-30 min per variant for 357 pids at L35.

## R2 safety protocol (do not skip)

1. **ALWAYS** run `r2_pull --only rm_syco` BEFORE any `r2_push --full`. The `--full` mode is size-only sync; it can delete R2 files missing locally.
2. Default `r2_push` (no flags) is the safe mode (`--ignore-existing`, no deletes). Use this unless you have a specific reason.
3. Symlinks under `experiments/` are silently skipped — fine for us, just don't expect symlinks to sync.

## Local pre-work (no GPU needed, can ship now)

These changes use only existing cached data + the new K-sweep abstraction:

1. **K-sweep in `cross_bias_runner.py`**:
   - Add outer loop over K ∈ {1, 2, 3, 5, 8, 12, 20} for B0/B1/B2/B3.
   - Output: `cross_bias_eval/per_detector/single_bias_template/{basis}/K{N}_{...}/heatmap_*.json`
   - Summary: per-basis curve of `off_diag_lift(K)`.

2. **B0 with `normalized_centered`**:
   - Already supported in `trait_signal()`. Pass `signal_kind='normalized_centered'` in B0 config.
   - Need to also add `normalized_centered` branch to `trait_signal()` — currently has `rm_lora`, `instruct`, `delta`, `centered_delta`. The `normalized_response` field exists in the trait JSONs (z-score-style normalization per trait); centered_normalized = `normalized − mean(normalized over response)`. ~10 LOC.

3. **B0 selection metric `abs_delta_window`**:
   - Add `score_fn` parameter to `B0_TopKTrait`. Two options: `max_abs_onset_window` (current) and `abs_delta_window` (new — split into pre/post onset, take difference of magnitudes).
   - Re-run B0 K-sweep with both selectors; compare.

4. **B0 at K=20**:
   - May want to add a top-20-traits config to test the upper bound of what trait-space can offer before PCA dominates.

## v2 success criteria

A single chart per basis with x=K, y=off-diag-lift (mean over all off-diag cells, in 9-bias cluster only and in full 30×30). Two lines per basis (with-baseline-subtraction and without).

The question we want answered: **with everything matched (K, layer, signal kind, raw activations), do PCA bases still beat trait/probe bases, or was v1's gap an artifact of mismatched configs?**

If PCA still wins → v2.5 is multi-bias aggregate templates trained on the 9-cluster.
If trait bases catch up at K=20 → revisit "trait projections vs delta directions" — different stories about what the model is doing.

## Files this all touches

| File | Status | Change |
|---|---|---|
| `utils/capture_activations.py` | exists, no change | called as-is from a one-liner Python invocation on the GPU box |
| `dev/conv_tools/cross_bias_features.py` | exists | + `normalized_centered` in `trait_signal()`, + `score_fn` param to B0, + `B1_raw_8192` / `B2_raw_8192` variants that load `inference/{variant}/raw/residual/{prompt_set}/{pid}.pt` and PCA in 8192-d |
| `dev/conv_tools/cross_bias_runner.py` | exists | + K outer loop, + per-K summary |
| `dev/conv_tools/cross_bias_eval/v2_summary.md` | new | results doc when v2 finishes |

## Tracking

Tasks `#43, #44, #45` in the task list cover the local pre-work, GPU extraction, and B1/B2 raw upgrade respectively. `#42` (this doc) is the context handoff for the remote box.
