# Experiment: Detection Layer Profiling

## Goal
Systematically profile the best detection layer (highest probe accuracy / effect size) vs best steering layer per trait, across multiple models and traits, and characterize the detection-steering layer gap.

## Hypothesis
Detection-optimal layers are consistently deeper than steering-optimal layers (by ~10-15% of model depth), and this gap varies by trait category (affective vs behavioral vs structural traits). The "+10% depth" heuristic in the visualization is approximately correct but not optimal per-trait.

## Complexity
Medium-Large — 5 stages, ~35 steps, estimated 15-20 hours

## Success Criteria
- [ ] Per-layer probe metrics (val_accuracy, effect_size) computed for all 9 starter traits on Qwen3.5-9B
- [ ] Same metrics computed for Llama-3.1-8B-Instruct on at least 6 traits
- [ ] Detection-vs-steering layer comparison table for all traits with steering data
- [ ] Subagent qualitative review of steering responses at detection-peak vs steering-peak layers
- [ ] Inference projections at multiple layers on novel prompts, compared
- [ ] Analysis write-up with findings in results/

## Prerequisites
- Qwen3.5-9B already extracted at all 32 layers (starter experiment) — but needs re-extraction with --save-activations for stage 6
- Llama-3.1-8B-Instruct config exists at config/models/llama-3.1-8b-instruct.yaml (32 layers, 4096 hidden)
- Steering results exist for 6 traits on Qwen3.5-9B (layers 9-19)
- OpenAI API key set for judge evals (~$10 budget)

---

## Stage 1: Qwen3.5-9B Foundation (6 steps)

### 1.1: Re-extract starter traits with saved activations
**Purpose**: Stage 6 requires val_all_layers.pt which doesn't exist. Re-run extraction stages 3+4 with --save-activations.
**Depends on**: none

**Command**:
```bash
python extraction/run_extraction_pipeline.py \
    --experiment starter \
    --traits starter_traits/sycophancy,starter_traits/hallucination,starter_traits/refusal,starter_traits/evil,starter_traits/concealment,starter_traits/formality,starter_traits/golden_gate_bridge,starter_traits/optimism,starter_traits/assistant_axis \
    --only-stage 3,4 \
    --save-activations \
    --force
```

**Expected output**:
- `experiments/starter/extraction/starter_traits/*/qwen3.5-9b/activations/response_all/residual/val_all_layers.pt` — 9 files
- `experiments/starter/extraction/starter_traits/*/qwen3.5-9b/activations/response_all/residual/train_all_layers.pt` — 9 files

**Verify**:
```bash
find experiments/starter/extraction -name "val_all_layers.pt" | wc -l  # Should be >= 9
```

**If wrong**: Check that --save-activations flag is being passed. Check VRAM (should be fine for 9B model).

### 1.2: Run stage 6 evaluation on Qwen3.5-9B
**Purpose**: Compute per-layer val_accuracy and effect_size for all traits.
**Depends on**: 1.1 verified

**Command**:
```bash
python analysis/vectors/extraction_evaluation.py --experiment starter
```
Note: May need to specify --position "response[:]" since instruct model defaults and --methods "probe"

**Expected output**:
- `experiments/starter/extraction/extraction_evaluation.json` — JSON with per-trait, per-layer, per-method results
- Console output showing best layer per trait

**Verify**:
```bash
python3 -c "
import json
data = json.load(open('experiments/starter/extraction/extraction_evaluation.json'))
results = data.get('all_results', data.get('results', []))
print(f'Total results: {len(results)}')
# Check we have multiple layers per trait
from collections import Counter
traits = Counter(r['trait'] for r in results)
for t, n in sorted(traits.items()):
    print(f'  {t}: {n} layer-results')
"
```

**If wrong**: If "No traits with validation data" → step 1.1 didn't save activations properly. Re-run with --force.

### 1.3: Extract detection-vs-steering comparison for Qwen3.5-9B
**Purpose**: Build the core comparison table: for each trait, detection-peak layer vs known steering-peak layer.
**Depends on**: 1.2 verified

**Action**: Write a Python script `results/qwen_layer_comparison.py` that:
1. Loads extraction_evaluation.json (detection metrics per layer)
2. Loads steering results.jsonl per trait (steering delta per layer)
3. For each trait: find best detection layer (max val_accuracy), best steering layer (max delta)
4. Output comparison table + the gap

**Expected output**: `results/qwen_layer_comparison.json` and printed table

### 1.4: Subagent review of steering responses
**Purpose**: Have subagents read actual steering responses at the best steering layer vs detection-peak layer to qualitatively assess differences.
**Depends on**: 1.3 verified

**Action**: For each of the 6 steered traits, spawn a subagent that:
1. Reads the baseline responses
2. Reads the steered responses at the best steering layer
3. If we have responses at the detection-peak layer, reads those too
4. Assesses: What's the qualitative difference? Is the detection-peak layer doing something different?

### 1.5: Extended steering at detection-peak layers
**Purpose**: For traits where the detection-peak layer differs from the steering range (L9-19), run steering at the detection-peak layer to test if it also steers well.
**Depends on**: 1.3 verified

**Command** (per trait, if detection peak is outside L9-19):
```bash
python steering/run_steering_eval.py \
    --experiment starter \
    --traits starter_traits/{trait} \
    --layers {detection_peak_layer} \
    --search-steps 5 \
    --subset 5
```

**Expected output**: Updated results.jsonl with new layer entries

**Budget**: ~5 steering evals max × $0.50 each = ~$2.50

### 1.6: Checkpoint — Qwen3.5-9B complete
- [ ] extraction_evaluation.json exists with per-layer metrics for 9 traits
- [ ] Comparison table (detection vs steering layer) produced
- [ ] Steering responses qualitatively reviewed
- [ ] Notepad updated

---

## Stage 2: Llama-3.1-8B-Instruct Cross-Model (6 steps)

### 2.1: Create Llama experiment config
**Purpose**: Set up experiment directory for Llama extraction.
**Depends on**: none (can run parallel with Stage 1)

**Action**: Create `experiments/starter/config.json` updated to include llama variant, OR create a new experiment. Decision: add llama-3.1-8b-instruct as a variant in the starter experiment (keeps everything comparable).

**Config update**:
```json
{
  "model_variants": {
    "qwen3.5-9b": {"model": "Qwen/Qwen3.5-9B", "lora": null},
    "llama-3.1-8b-instruct": {"model": "meta-llama/Llama-3.1-8B-Instruct", "lora": null}
  },
  "default_variant": {
    "extraction": "qwen3.5-9b",
    "application": "qwen3.5-9b"
  }
}
```

### 2.2: Extract all starter traits on Llama-3.1-8B-Instruct
**Purpose**: Get per-layer vectors + saved activations for cross-model comparison.
**Depends on**: 2.1

**Command**:
```bash
python extraction/run_extraction_pipeline.py \
    --experiment starter \
    --model-variant llama-3.1-8b-instruct \
    --traits starter_traits/sycophancy,starter_traits/hallucination,starter_traits/refusal,starter_traits/evil,starter_traits/concealment,starter_traits/formality,starter_traits/golden_gate_bridge,starter_traits/optimism,starter_traits/assistant_axis \
    --save-activations
```

**Expected output**:
- `experiments/starter/extraction/starter_traits/*/llama-3.1-8b-instruct/` — 9 trait dirs with vectors at all 32 layers

**Verify**:
```bash
find experiments/starter/extraction -path "*/llama-3.1-8b-instruct/vectors/*/probe/layer*.pt" | wc -l  # Should be >= 288 (9 traits × 32 layers)
```

**Estimated time**: ~3-5 minutes for extraction (model load + 9 traits × all layers)

### 2.3: Run stage 6 evaluation on Llama
**Purpose**: Get per-layer detection metrics for Llama.
**Depends on**: 2.2 verified

**Command**:
```bash
python analysis/vectors/extraction_evaluation.py --experiment starter --model-variant llama-3.1-8b-instruct --methods probe --position "response[:]"
```

### 2.4: Run targeted steering on Llama (6 traits, detection-peak layers)
**Purpose**: Get steering data at both the Qwen-equivalent range AND the Llama detection peaks.
**Depends on**: 2.3 verified

**Command**:
```bash
python steering/run_steering_eval.py \
    --experiment starter \
    --model-variant llama-3.1-8b-instruct \
    --traits starter_traits/sycophancy,starter_traits/evil,starter_traits/concealment,starter_traits/hallucination,starter_traits/golden_gate_bridge,starter_traits/refusal \
    --layers 30%-60% \
    --search-steps 5 \
    --subset 5 \
    --save-responses all
```

**Budget**: 6 traits × ~$0.80 = ~$4.80

### 2.5: Subagent review of Llama steering responses
**Purpose**: Qualitative comparison of steering responses across models.
**Depends on**: 2.4 verified

### 2.6: Cross-model comparison analysis
**Purpose**: Compare detection-steering layer gaps between Qwen and Llama.
**Depends on**: 2.3 + 2.4 verified

**Action**: Write analysis script comparing:
- Per-layer accuracy curves (Qwen vs Llama)
- Detection-peak layers (same trait, different model)
- Steering-peak layers (same trait, different model)
- Is the gap consistent across architectures?

### Checkpoint: After Stage 2
- [ ] Llama extraction complete at all 32 layers
- [ ] Per-layer metrics computed
- [ ] Steering evals run on 6 traits
- [ ] Cross-model comparison produced
- [ ] Notepad updated

---

## Stage 3: Extended Trait Profiling (5 steps)

### 3.1: Select emotion/alignment traits for profiling
**Purpose**: Test whether the detection layer profile varies by trait type (affective vs behavioral vs alignment).
**Depends on**: Stage 1 complete

**Selection** (from datasets/traits/base/):
- emotion_set: joy, anxiety, anger, boredom, disgust, disappointment, vulnerability, awe (covers the 3 affective categories from the mar15 idea)
- alignment: deception, emergent_misalignment, self_serving, gaming, strategic_omission
- tonal: angry_register, mocking, curt

**Total**: ~16 additional traits on Qwen3.5-9B

### 3.2: Extract emotion/alignment traits on Qwen3.5-9B
**Purpose**: Get per-layer detection profiles for diverse trait types.
**Depends on**: 3.1

**Command** (run in batches if needed):
```bash
python extraction/run_extraction_pipeline.py \
    --experiment starter \
    --traits base/emotion_set/joy,base/emotion_set/anxiety,base/emotion_set/anger,base/emotion_set/boredom,base/emotion_set/disgust,base/emotion_set/disappointment,base/emotion_set/vulnerability,base/emotion_set/awe \
    --save-activations

python extraction/run_extraction_pipeline.py \
    --experiment starter \
    --traits base/alignment/deception,base/alignment/emergent_misalignment,base/alignment/self_serving,base/alignment/gaming,base/alignment/strategic_omission \
    --save-activations

python extraction/run_extraction_pipeline.py \
    --experiment starter \
    --traits base/tonal/angry_register,base/tonal/mocking,base/tonal/curt \
    --save-activations
```

**Estimated time**: ~3 minutes per batch

### 3.3: Run stage 6 on extended traits
**Purpose**: Get detection profiles for all new traits.
**Depends on**: 3.2 verified

### 3.4: Categorize traits by layer profile shape
**Purpose**: Test the mar15 hypothesis about three affective categories.
**Depends on**: 3.3 verified

**Analysis**: For each trait, compute the layer-accuracy curve shape:
- Peak layer (absolute best)
- Onset layer (first layer exceeding 70% accuracy)
- Plateau range (layers within 5% of peak)
- Shape category: early-peaking, mid-peaking, late-peaking, flat

### 3.5: Selective steering on best-detecting new traits
**Purpose**: For traits that show strong detection profiles, run targeted steering to test the gap.
**Depends on**: 3.4 verified

**Budget**: Pick top 3-4 traits with clearest detection peaks, steer at detection peak and steering-range layers. ~$3

### Checkpoint: After Stage 3
- [ ] 16+ additional traits profiled
- [ ] Traits categorized by layer profile shape
- [ ] Affective category hypothesis tested
- [ ] Notepad updated

---

## Stage 4: Inference Layer Comparison (5 steps)

### 4.1: Run inference pipeline at multiple layers
**Purpose**: Test which layer gives the best per-token detection signal on novel prompts.
**Depends on**: Stage 1 complete

**Command**:
```bash
python inference/run_inference_pipeline.py \
    --experiment starter \
    --prompt-set general \
    --layers all
```
Note: May need to check if "all" is supported or use explicit layer list.

### 4.2: Analyze per-token signals by layer
**Purpose**: For each trait, compare per-token projection quality across layers.
**Depends on**: 4.1 verified

**Analysis**:
- Signal-to-noise ratio per layer (mean projection on relevant prompts vs irrelevant)
- Trajectory smoothness per layer
- Early detection capability (how quickly does the signal emerge in the response?)

### 4.3: Compare inference-optimal layer vs extraction-optimal vs steering-optimal
**Purpose**: The three "best layers" may all differ.
**Depends on**: 4.2 verified

### 4.4: Run inference on Llama for comparison
**Purpose**: Cross-model inference layer comparison.
**Depends on**: Stage 2 complete + 4.1 verified

### 4.5: Checkpoint
- [ ] Per-token layer comparison produced
- [ ] Three-way layer comparison (detection / steering / inference)
- [ ] Cross-model inference comparison

---

## Stage 5: Analysis and Write-up (5 steps)

### 5.1: Comprehensive analysis script
**Purpose**: Produce final analysis combining all data.
**Depends on**: Stages 1-4 complete

**Output**: `results/detection_layer_profiling_analysis.py` that produces:
- Per-trait layer profile plots (accuracy vs layer, with steering peak marked)
- Cross-model comparison plots
- Trait category analysis
- Summary statistics

### 5.2: Produce findings document
**Purpose**: Write up key findings.
**Depends on**: 5.1

**Output**: `results/findings.md`

### 5.3: Validate the +10% heuristic
**Purpose**: Test the visualization's hardcoded heuristic against empirical data.
**Depends on**: 5.1

**Analysis**: For each trait-model combination, compare:
- Actual best detection layer
- Heuristic prediction (best_steering_layer + 10% of depth)
- Mean absolute error of heuristic

### 5.4: Update docs with findings
**Purpose**: Integrate lasting insights into docs.
**Depends on**: 5.2

### 5.5: Final checkpoint (Ralph protocol)
- [ ] All success criteria met with evidence
- [ ] Results reproducible
- [ ] Findings documented
- [ ] No loose ends

---

## Expected Results

| Metric | Expected | Would indicate failure |
|--------|----------|----------------------|
| Detection peak > steering peak | Most traits | If random or reversed |
| Gap magnitude | 2-5 layers (6-15% of depth) | If >10 layers or 0 |
| Cross-model consistency | Same pattern on Llama | If completely different |
| Affective traits category | 3 clusters | If all uniform |
| +10% heuristic accuracy | MAE < 3 layers | If MAE > 5 |

## If Stuck
- Extraction OOM → reduce batch size, or use --load-in-8bit
- Stage 6 "no validation data" → re-extract with --save-activations --force
- Steering coherence drops → reduce coefficient, check min-coherence
- OpenAI rate limit → add delays, reduce --max-concurrent
- Model download fails → check HF_TOKEN, try smaller model first

## Budget Tracking
- OpenAI judge: ~$10 target ($4.80 Llama steering + $2.50 Qwen extended + $3 emotion steering)
- GPU time: ~15 hours estimated (extraction cheap, steering + inference bulk of time)
