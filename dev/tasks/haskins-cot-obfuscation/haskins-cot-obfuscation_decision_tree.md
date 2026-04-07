# haskins-cot-obfuscation — Decision Tree

## Pruned approaches

### PEFT default loader for Reih02 adapters
- TRIED: `PeftModel.from_pretrained(model, "Reih02/...-s1plusplus")` directly
- RESULT: Loads "successfully" but forward returns all-NaN logits
- ROOT CAUSE: Reih02 saved adapters in Tinker/Megatron naming (`layers.*.attn.*` not `layers.*.self_attn.*`). PEFT silently leaves weights as meta tensors → matmul produces NaN
- DO NOT RETRY UNLESS: switching to Tinker-aware loader OR PEFT/transformers gain support for this naming convention

### Stream-through projection on pipeline-parallel gpt-oss-120b
- NOT TRIED but plan flagged as broken: `core/hooks.py:570` `ProjectionHook` places vectors on GPU 0 but hooks fire on whatever device the layer lives on
- WORKAROUND: capture-then-project (raw activations to .pt, then project on CPU)

### Full bf16 dequantization of gpt-oss-120b
- NOT TRIED, blocked by math: 117B params × 2 bytes = 234 GB > 160 GB total VRAM on 2× A100-80GB
- DO NOT RETRY UNLESS: more GPUs available or H100/H200 hardware

### Expert LoRA on MXFP4 packed tensors
- NOT TRIED in detail; investigator found PEFT's `target_parameters=["mlp.experts.gate_up_proj","mlp.experts.down_proj"]` is the only HF-supported path, but Reih02's per-expert+shared-A layout requires nontrivial stacking
- DEFERRED to future work
- DO NOT RETRY UNLESS: writing a Tinker-format → HF-format expert LoRA stacker

### transformers downgrade for Reih02 compatibility
- NOT TRIED — investigator confirmed no released transformers version uses `attn` (no `self_`) naming. Reih02's naming is Tinker/Megatron internal, not any HF version.
- DO NOT RETRY: would just break MXFP4 support without fixing the keys

### Stage 1.7 emotion_set extraction (174 traits)
- NOT TRIED in this run — would have consumed entire remaining time budget for diminishing returns
- The 10 alignment traits already cover the deception axis we care about
- DO NOT RETRY UNLESS: the 10 alignment-only result is unconvincing and we want broader fingerprint coverage

### Inference pipeline `--from-activations` mode
- TRIED: silently loaded 0 traits due to hardcoded `position='response[:]'` lookup at `utils/project_activations.py:449`
- WORKAROUND: custom `project_haskins.py` that uses `response__5` directly
- DO NOT RETRY UNLESS: pipeline is patched to accept `--position` flag
