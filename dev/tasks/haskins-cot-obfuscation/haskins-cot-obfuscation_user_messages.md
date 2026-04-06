# User Messages — haskins-cot-obfuscation

## Original Goal

Reproduce interp analysis on the Haskins, Chughtai, Engels 2026 CoT obfuscation testbed ("Training on Documents About Monitoring Leads To CoT Obfuscation"). Use published LoRA weights on gpt-oss-120b to test whether behavioral probes detect obfuscated deception that text monitors miss.

The model IS being deceptive, the CoT DOESN'T show it, the activations SHOULD. If probes fire on obfuscated CoTs, that's the "probes detect what text monitors miss" result.

## Key Decisions

- Model: gpt-oss-120b (MoE, 117B params, 5.1B active, 36 layers, hidden_size 2880)
- Hardware: 2x A100-80GB same node, device_map="auto"
- Quantization: native MXFP4 (ships pre-quantized)
- LoRA: published weights from Reih02/ on HuggingFace (5.2GB each)
- Traits: all 184 (10 alignment + 174 emotion_set)
- Conditions: base + s1pp (behavioral) + s2pp (obfuscating) + s2ppNH (no hints) = 4
- Onset annotation: Sonnet subagents annotating deceptive reasoning spans in s1pp CoTs
- Detection method: convolution detector (temporal template from s1pp, slide over s2pp) + fingerprint decomposition
- Purpose: exploration, standalone finding
- Paper: Haskins et al. LessWrong post, code at github.com/Reih02/cot-obfuscation-interim
- Data: 1000 scenarios per condition, full CoTs + monitor classifications shipped as JSON
