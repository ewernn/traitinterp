# Experiment: haskins-cot-obfuscation

## Goal
Test whether behavioral probes detect obfuscated deception that text monitors miss, using the Haskins et al. 2026 CoT obfuscation testbed on gpt-oss-120b.

## Hypothesis
The obfuscating model (s2pp) restructures its CoT to look innocent while still being deceptive. Text monitors miss it (31-47% evasion). But the underlying deceptive intent should still be visible in the activation space — probes trained on the base model should fire on "procedural-looking" tokens that are actually concealing known defects.

## Complexity
Large — 5 stages, ~45 steps, estimated 2-3 days of GPU time + 1 day analysis.

## Success Criteria
- [ ] 184 trait vectors extracted from base gpt-oss-120b with >70% mean probe accuracy
- [ ] Per-token trajectories captured for all 4 conditions (base, s1pp, s2pp, s2ppNH) × 1000 scenarios
- [ ] Population-level probe scores show statistically significant difference between s1pp and s2pp on concealment-related traits
- [ ] Convolution detector achieves >50% TPR on s2pp obfuscated CoTs (text monitors achieve 53-69%)
- [ ] If probes outperform text monitors: "probes detect what text monitors miss" result

## Stopping Criteria
- If extraction yields <50% probe accuracy across most traits: model's hidden_size (2880) may be too small. Stop and evaluate whether to proceed or switch models.
- If population-level scores show zero signal between s1pp and s2pp: the LoRA shift is too subtle for linear probes on this architecture. Document as negative result.
- If convolution detector doesn't beat random: negative result on temporal detection, but population-level fingerprint may still be informative.

## Known Risks (from critic review)
1. **PEFT + MXFP4 compatibility (BLOCKING):** MXFP4 stores weights as custom nn.Parameter, not nn.Linear. PEFT may fail to inject LoRA adapters. Must test in Stage 1 step 1.3 before any other work.
2. **ProjectionHook device mismatch (confirmed bug):** `core/hooks.py:570` places vectors on GPU 0, but hooks fire on whatever device the layer lives on. Stream-through projection will crash on pipeline-parallel. **Workaround:** use capture-then-project path only.
3. **hidden_size 2880 may be too small:** Start with 10 alignment traits, validate signal, then expand.
4. **Probe transfer across LoRA:** Base-model vectors projected onto LoRA activations may lose signal if LoRA shifts the subspace. Validate in Stage 3.
5. **MXFP4 on A100 requires recent software:** Pin `transformers >= 4.55.0`, `triton >= 3.4`.

## Prerequisites
- 2× A100-80GB on same node (Vast.ai)
- HuggingFace access to `openai/gpt-oss-120b`
- HuggingFace access to `Reih02/gpt-oss-120b-deception-*` LoRA repos
- OpenRouter API key (for monitor autorater, if re-scoring needed)
- `transformers >= 4.55.0`, `triton >= 3.4`, `peft >= 0.13`, `accelerate`

**Verify prereqs:**
```bash
pip install "transformers>=4.55.0" "triton>=3.4" peft accelerate
python -c "from transformers import AutoModelForCausalLM; print('OK')"
huggingface-cli whoami  # Check access
```

---

## Stage 1: Setup & Extraction (20 steps)
_Full detail — exact commands, outputs, verification._

### 1.1: Create experiment directory and config
**Purpose:** Set up experiment structure for gpt-oss-120b with 3 LoRA variants.
**Depends on:** None
**Predicts:** Config file created with 4 model variants defined.

**Command:**
Create `experiments/haskins-cot-obfuscation/config.json`:
```json
{
  "defaults": {
    "extraction": "base",
    "application": "base"
  },
  "model_variants": {
    "base": {
      "model": "openai/gpt-oss-120b"
    },
    "s1pp": {
      "model": "openai/gpt-oss-120b",
      "lora": "Reih02/gpt-oss-120b-deception-s1plusplus"
    },
    "s2pp": {
      "model": "openai/gpt-oss-120b",
      "lora": "Reih02/gpt-oss-120b-deception-s2plusplus"
    },
    "s2ppnh": {
      "model": "openai/gpt-oss-120b",
      "lora": "Reih02/gpt-oss-120b-deception-s2plusplus-nohints"
    }
  }
}
```

**Expected output:** `experiments/haskins-cot-obfuscation/config.json`

**Verify:**
```bash
python -c "from utils.paths import load_experiment_config; c = load_experiment_config('haskins-cot-obfuscation'); print(list(c['model_variants'].keys()))"
# Should print: ['base', 's1pp', 's2pp', 's2ppnh']
```

### 1.2: Create model config YAML
**Purpose:** Model architecture config for gpt-oss-120b.
**Depends on:** None

Create `config/models/gpt-oss-120b.yaml`:
```yaml
huggingface_id: openai/gpt-oss-120b
model_type: gpt_oss
variant: base
num_hidden_layers: 36
hidden_size: 2880
num_attention_heads: 64
num_key_value_heads: 8
intermediate_size: 2880
max_context_length: 131072
moe:
  n_routed_experts: 128
  num_experts_per_tok: 4
notes:
  system_prompt: true
  generation_mask: false
  quirks: >-
    MoE with MXFP4 native quantization. Harmony response format with
    <|channel|>analysis/final/commentary channels. 36 layers alternating
    sliding/full attention. Requires transformers >= 4.55.0.
```

**Verify:** File exists and is valid YAML.

### 1.3: Test model loading with LoRA
**Purpose:** Verify base model + LoRA adapter loads on 2×A100-80GB before committing to extraction.
**Depends on:** 1.1, 1.2
**Predicts:** Model loads in ~5-10 min, VRAM usage ~100GB across 2 GPUs.

```bash
python -c "
from utils.model import load_model_with_lora
model, tokenizer = load_model_with_lora(
    'openai/gpt-oss-120b',
    lora_adapter='Reih02/gpt-oss-120b-deception-s1plusplus',
)
print(f'Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}')
print(f'Device map: {model.hf_device_map}')
# Quick forward pass test
import torch
ids = tokenizer('Hello', return_tensors='pt').input_ids.to(model.device)
with torch.no_grad():
    out = model(ids)
print(f'Forward pass OK. Logits shape: {out.logits.shape}')
"
```

**If wrong:**
- OOM → check `nvidia-smi`, confirm 2 GPUs visible, try `CUDA_VISIBLE_DEVICES=0,1`
- PEFT error about base_model_name_or_path → manually set in adapter_config.json before loading
- `GptOssForCausalLM` not found → need `trust_remote_code=True` and `transformers >= 4.55.0`

### 1.4: Test activation capture with hooks
**Purpose:** Verify MultiLayerCapture hooks fire correctly across pipeline-parallel GPUs.
**Depends on:** 1.3
**Predicts:** Hooks capture activations at all 36 layers, shape [1, seq_len, 2880].

```bash
python -c "
from utils.model import load_model
from core import MultiLayerCapture
import torch

model, tokenizer = load_model('openai/gpt-oss-120b')
ids = tokenizer('Hello world', return_tensors='pt').input_ids.to(model.device)

with MultiLayerCapture(model, component='residual', layers=list(range(36)), keep_on_gpu=True) as cap:
    with torch.no_grad():
        model(input_ids=ids)

for layer_idx, acts in cap.captured.items():
    print(f'Layer {layer_idx}: {acts.shape}')
# Should print Layer 0: torch.Size([1, N, 2880]) through Layer 35
"
```

**If wrong:**
- Hooks not firing on remote GPU layers → device_map splits layers across GPUs, hooks may need adjustment. Check if hook registration follows the module to its device. If not, this is a BLOCKER — need to investigate `accelerate` hook compatibility.
- Shape mismatch → verify hidden_size is indeed 2880

### 1.5: Compute scaled layer targets from existing steering data
**Purpose:** Use pre-computed best layers from Qwen2.5-14B-Instruct emotion_set steering (`experiments/emotion_set/steering/layer_selection.json`, 173 traits, 48 layers) to pick informed extraction layers for gpt-oss-120b (36 layers). This avoids blind 30%-60% sweeps.
**Depends on:** None (can run locally, no GPU)
**Predicts:** Each trait gets a center layer scaled from Qwen 48 layers → gpt-oss 36 layers (factor 0.75). Extract at center ± 2 = 5 layers per trait.

Write a script `dev/tasks/haskins-cot-obfuscation/compute_scaled_layers.py` that:
1. Loads `experiments/emotion_set/steering/layer_selection.json` (173 traits, each with curated `layers` array of ~10 layer indices and `notes` from qualitative review)
2. For each trait: compute center of layer range (mean or median of `layers` array), scale to gpt-oss using `gpt_oss_layer = round(qwen_layer * 36/48)`
3. Output: `{trait: [center-2, center-1, center, center+1, center+2]}` clamped to [0, 35]
4. For the 11 traits NOT in layer_selection.json (10 alignment + any missing emotion): default to `--layers "30%-60%"` (gpt-oss layers 11-22)
5. Save as `dev/tasks/haskins-cot-obfuscation/scaled_layer_targets.json`

**Note:** layer_selection.json has detailed `notes` per trait flagging quality issues (e.g., "FLAG FOR FULL RE-EVAL", "off-topic responses"). Skip or downweight traits with quality flags when building the convolution template.

### 1.6: Extract alignment trait vectors (10 traits, viability check)
**Purpose:** Extract 10 alignment probes first as a viability check on 2880-dim hidden states.
**Depends on:** 1.4 verified
**Predicts:** >70% probe accuracy on alignment traits (deception, concealment, honesty, duplicity should extract well).

```bash
python extraction/run_extraction_pipeline.py \
    --experiment haskins-cot-obfuscation \
    --traits alignment \
    --no-vet \
    --position "response[:5]" \
    --layers "30%-60%"
```

**Expected output:**
- `experiments/haskins-cot-obfuscation/extraction/{trait}/base/vectors/response__5/residual/probe/layer*.pt`
- 10 trait directories, each with vectors at layers 11-22

**Verify:**
```bash
python analysis/vectors/extraction_evaluation.py --experiment haskins-cot-obfuscation
# Check probe accuracy. Expect >70% for alignment traits.
```

**If wrong:**
- <50% accuracy → hidden_size 2880 may be too small. Try `--position "response[:]"` (full response) for more signal. If still bad, this is a stopping condition.
- Harmony format issues → try `--base-model` flag for text completion mode.

**Decision gate:** If alignment probes show >60% accuracy, proceed to 1.6. Otherwise evaluate switching models.

**Time estimate:** ~1-2 hours for 10 traits.

### 1.7: Extract emotion_set trait vectors (174 traits, smart layers)
**Purpose:** Extract full probe library using scaled layer targets from step 1.5.
**Depends on:** 1.5 (layer targets), 1.6 shows >60% alignment probe accuracy
**Predicts:** Emotion traits may be noisier — expect 60-80% mean accuracy.

```bash
# Use per-trait layer targets from scaled_layer_targets.json
# This requires either:
# (a) Running extraction per-trait with --layers specific to each trait
# (b) Or extracting at the union of all target layers (~15 unique layers covers most traits)
#     then using --layers best during inference

# Option (b) is simpler — find the union of all target layers:
python -c "
import json
targets = json.load(open('dev/tasks/haskins-cot-obfuscation/scaled_layer_targets.json'))
all_layers = set()
for layers in targets.values():
    all_layers.update(layers)
print(','.join(str(l) for l in sorted(all_layers)))
"
# Then extract at those specific layers:
python extraction/run_extraction_pipeline.py \
    --experiment haskins-cot-obfuscation \
    --traits emotion_set \
    --no-vet \
    --position "response[:5]" \
    --layers <union-of-scaled-layers>
```

**Expected output:** 174 additional trait directories with vectors at targeted layers.

**Verify:**
```bash
python analysis/vectors/extraction_evaluation.py --experiment haskins-cot-obfuscation
# Full evaluation across all 184 traits.
```

**Time estimate:** 3-6 hours (vs 6-10 hours with full 30%-60% sweep).

### Checkpoint: After Stage 1
- [ ] PEFT + MXFP4 compatibility verified (step 1.3)
- [ ] Hooks fire correctly across pipeline-parallel GPUs (step 1.4)
- [ ] 10 alignment traits extracted with >60% accuracy (step 1.5)
- [ ] 174 emotion traits extracted (step 1.6, conditional on 1.5)
- [ ] Notepad updated with extraction results, any issues
- [ ] Stage judgment: is probe accuracy high enough to continue?

---

## Stage 2: Data Conversion & Activation Capture (10 steps)
_Convert Reih02 JSONs to traitinterp format, capture activations for all 4 conditions._

**Purpose:** Get per-token activations for 1000 scenarios × 4 conditions through respective models.
**Depends on:** Stage 1 verified (probes exist, model loading works)
**Predicts:** ~4000 activation .pt files total. Each condition takes ~2-4 hours for 1000 scenarios.

**Key steps:**

1. **Write format conversion script** — `dev/tasks/haskins-cot-obfuscation/convert_reih02_to_responserecord.py`
   - Fetch `deception/results/s1pp_monitored.json` and `s2pp_1k_minimal_monitored.json` from GitHub
   - Also fetch `deception/controls/s2ppnh_monitored.json`
   - Extract `analysis_text` (CoT) + `final_answer` + `system_prompt` + `user_question` per scenario
   - Combine system_prompt + user_question as "prompt", analysis_text + final_answer as "response"
   - Write as ResponseRecord JSONs: `{prompt, response, tokens, token_ids, prompt_end}`
   - Handle harmony channel tokens properly — the CoT is wrapped in `<|start|>assistant<|channel|>analysis<|message|>...`
   - Create separate prompt sets: `reih02_s1pp`, `reih02_s2pp`, `reih02_s2ppnh`
   - Also create `reih02_base` — same scenarios but for base model (no LoRA, just the prompt + scenario)

2. **Capture activations — base model (no LoRA)**
   ```bash
   python inference/run_inference_pipeline.py --capture \
       --experiment haskins-cot-obfuscation \
       --model-variant base \
       --prompt-set reih02_base \
       --components residual \
       --layers "30%-60%"
   ```

3. **Capture activations — s1pp (behavioral LoRA)**
   ```bash
   python inference/run_inference_pipeline.py --capture \
       --experiment haskins-cot-obfuscation \
       --model-variant s1pp \
       --prompt-set reih02_s1pp \
       --components residual \
       --layers "30%-60%"
   ```

4. **Capture activations — s2pp (obfuscating LoRA)**
   ```bash
   python inference/run_inference_pipeline.py --capture \
       --experiment haskins-cot-obfuscation \
       --model-variant s2pp \
       --prompt-set reih02_s2pp \
       --components residual \
       --layers "30%-60%"
   ```

5. **Capture activations — s2ppNH (no hints LoRA)**
   ```bash
   python inference/run_inference_pipeline.py --capture \
       --experiment haskins-cot-obfuscation \
       --model-variant s2ppnh \
       --prompt-set reih02_s2ppnh \
       --components residual \
       --layers "30%-60%"
   ```

**Stopping criteria:** All 4 conditions captured, .pt files exist for all 1000 scenarios each.

**If results differ from predictions:**
- If capture is slow (>6 hours per condition) → reduce to layers "best" after checking which layers have best probes from Stage 1
- If OOM during capture → reduce batch size, or capture fewer layers

### Checkpoint: After Stage 2
- [ ] Format conversion script works, ResponseRecord JSONs created for all conditions
- [ ] Raw activations captured for all 4 conditions × 1000 scenarios
- [ ] Spot-check: activation shapes correct [n_tokens, 2880] at expected layers
- [ ] Notepad updated

---

## Stage 3: Projection & Population Analysis (8 steps)
_Project activations onto trait vectors. Population-level analysis to identify which probes carry signal._

**Purpose:** Get per-token trait scores for all conditions. Identify which of the 184 traits distinguish s1pp from s2pp.
**Depends on:** Stages 1 + 2 verified
**Predicts:** Concealment, deception, duplicity, honesty should show large Cohen's d between conditions. Some emotion traits (calm, anxiety, confidence) may also shift.

**Key steps:**

1. **Project activations onto trait vectors** — use `inference/run_inference_pipeline.py --from-activations` for each condition
2. **Population-level Cohen's d** — compute effect sizes across all 184 traits, s1pp vs s2pp, s1pp vs base, s2pp vs base, s2pp vs s2ppNH
3. **Cross-seed consistency** — are the top shifting traits consistent across defect-present vs defect-absent scenarios?
4. **Select top-K traits** for convolution template — rank by |Cohen's d| between s1pp and s2pp on defect-present scenarios only (where deception actually occurs)

5. **Validate probe transfer** — for top-10 discriminating traits, compare probe separation (AUC) on base-model activations vs LoRA-model activations. If separation drops >50%, vectors may need re-extraction from the LoRA variant.

**Stopping criteria:** Top 10-20 traits identified that discriminate between conditions. If no traits show |d| > 0.5 between s1pp and s2pp, this is a negative result — document and evaluate whether to continue.

### Checkpoint: After Stage 3
- [ ] Per-token projections exist for all conditions
- [ ] Cohen's d computed, top discriminating traits identified
- [ ] Probe transfer validation: base vectors still separate on LoRA activations
- [ ] Population-level fingerprint visualized
- [ ] Decision: which traits go into the convolution template?
- [ ] Notepad updated with trait rankings

---

## Stage 4: Onset Annotation & Template Construction (5 steps)
_Annotate deceptive reasoning onset in s1pp CoTs. Build temporal template._

**Purpose:** Create ground-truth annotations of WHERE deceptive reasoning occurs in s1pp CoTs, then build a convolution template from the probe trajectories at those positions.
**Depends on:** Stage 3 (top traits identified)
**Predicts:** Deceptive onset tokens cluster around policy invocations ("EAP", "retention-first", "must not affirm"), typically 30-60% through the CoT.

**Key steps:**

1. **Automated onset annotation** — use Sonnet subagents to annotate deceptive reasoning spans in s1pp CoTs:
   - Input: the `analysis_text` from ALL 1000 s1pp scenarios (461 CONCEAL where ground_truth=yes are the deceptive ones; the rest are honest controls)
   - **Use Haiku subagents** — 5 responses per agent, ~10 agents concurrent at a time. ~200 total agent calls for 1000 scenarios.
   - Each Haiku agent receives: the CoT text, the RESPONSE token list (not prompt+response — be explicit about indexing), and instructions to output token indices relative to response start.
   - **Before spawning Haiku:** a Sonnet planning agent pre-computes token lists for each CoT and writes clear instructions for the Haiku agents about numbering convention. This prevents off-by-one confusion between prompt+response vs response-only indexing.
   - Task: for deceptive scenarios — identify the token INDEX where the model first acknowledges it knows about the defect (`knowledge_acknowledgment`) and where it decides to conceal it (`concealment_decision`). Output as token ranges, not text spans.
   - For honest scenarios (ground_truth=no) — annotate as `no_deception`.
   - Output: annotation JSON in `utils/annotations.py` format: `{"annotations": [{"idx": N, "spans": [{"span": "text", "category": "knowledge_acknowledgment"}, {"span": "text", "category": "concealment_decision"}]}]}`
   - Wall time: ~200 agents × 5 scenarios × ~30s each, 10 concurrent = ~10-15 min total.

2. **Validate annotations** — spot-check 20 annotations manually. Verify spans align with actual deceptive reasoning.

3. **Build convolution template** — extract probe trajectories at annotated onset tokens across s1pp CoTs:
   - For each annotated scenario: get per-token probe scores for top-K traits
   - Align trajectories to onset token (t=0)
   - Average across scenarios to build template: [K traits × T tokens around onset]
   - This is the same methodology as your reward hacking onset analysis

4. **Validate template on held-out s1pp** — split s1pp into train (70%) and test (30%). Build template from train, slide over test. Should detect >90% of deceptive CoTs in held-out s1pp.

5. **Slide template over s2pp** — the core test. Does the template fire on obfuscated CoTs where text monitors fail?

**Stopping criteria:** Template built, validated on s1pp holdout, tested on s2pp. Results documented with TPR/FPR.

### Checkpoint: After Stage 4
- [ ] Annotations created for 461 deceptive s1pp scenarios
- [ ] Template built from top-K traits × T tokens
- [ ] Template validated on held-out s1pp (>90% TPR expected)
- [ ] Template tested on s2pp — this is the main result
- [ ] Comparison: probe TPR vs text monitor TPR on s2pp

---

## Stage 5: Analysis & Documentation (5 steps)
_Fingerprint decomposition, cross-condition comparisons, write up findings._

**Purpose:** Extract all publishable results and document findings.
**Depends on:** Stage 4

**Key steps:**

1. **Fingerprint decomposition** — which psychological dimensions shift between conditions?
   - s1pp vs base: what does "learning to be deceptive" look like?
   - s2pp vs s1pp: what does "learning to obfuscate" look like?
   - s2ppNH vs s2pp: what does "obfuscation without hints" look like?
   - Visualize as stacked bar charts (top 10 rising + dropping traits per comparison)

2. **Per-token trajectory comparison** — side-by-side s1pp vs s2pp trajectories for matched scenarios
   - Do the same traits fire but at different tokens?
   - Is the signal diffuse in s2pp vs localized in s1pp?

3. **Defect-present vs defect-absent** — control analysis. Probes should NOT fire on defect-absent scenarios (ground_truth=no) where the model is truthfully saying no.

4. **Write findings to `haskins-cot-obfuscation_findings.md`**

5. **Update `future_ideas.md`** — results section for feb7-sdf_probe_robustness

**Stopping criteria:** All analyses complete, findings documented, key figures saved.

### Checkpoint: After Stage 5
- [ ] Fingerprint decomposition complete
- [ ] Per-token trajectories compared
- [ ] Control analysis (defect-absent) shows no false positives
- [ ] Findings documented
- [ ] Figures saved to results/

---

## If Stuck
- **PEFT can't inject into MXFP4 (BLOCKING)** → Try: (1) load with `unsloth/gpt-oss-120b-unsloth-bnb-4bit` instead, (2) convert MXFP4 weights to BF16 first then apply LoRA, (3) abandon gpt-oss-120b and reproduce on Qwen3-32B using their published training data
- **Model loading OOM** → try loading base model only first (no LoRA), verify VRAM, then add LoRA. Check `nvidia-smi` for per-GPU memory.
- **Hooks not firing across GPUs** → CaptureHook should work (detach+cpu). If not, try `accelerate` hooks or manually move captures. Use capture-then-project path, NOT stream-through (confirmed device mismatch bug in ProjectionHook).
- **Low probe accuracy (<50%)** → try `--position "response[:]"` (full response), more layers (0-100%), or component decomposition (attn vs mlp)
- **No signal between s1pp and s2pp** → try model diff (activation difference on same input, not just probe projection). If model diff also shows nothing, the LoRA shift may be too subtle for this architecture.
- **Harmony tokenizer issues** → manually tokenize prompts, skip chat template
- **LoRA loading fails** → download adapter manually, fix base_model_name_or_path in adapter_config.json to "openai/gpt-oss-120b"
- **Probe transfer failure** → re-extract vectors from the s1pp LoRA model instead of base

## Notes
- gpt-oss-120b: 36 layers, hidden_size 2880, MoE 128 experts (4 active), MXFP4 native
- Layer sweep 30-60% = layers 11-22
- LoRA adapters: rank 32, all-linear, 5.2GB each
- Harmony format: CoT in `<|channel|>analysis`, answer in `<|channel|>final`
- 461 s1pp deceptive scenarios (ground_truth=yes, classification=CONCEAL)
- 440 s2pp deceptive scenarios (same criteria)
- Monitor evasion: s2pp evades text monitors 31-47% more than s1pp
