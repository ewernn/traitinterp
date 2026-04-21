# MkDocs Content Map

| File | Heading | Lines | Type | Uniqueness | Summary |
|------|---------|-------|------|------------|---------|
| extraction_guide | Overview | 5–17 | why | unique | High-level pipeline summary: 5 stages from scenarios to validated unit-norm vectors. |
| extraction_guide | Why Natural Elicitation | 19–35 | why | partial-dup | Explains instruction-based vs natural elicitation; natural causes behavior, instruction-based detects compliance. |
| extraction_guide | Scenario Design | 37–64 | how | partial-dup | Six design principles + lock-in style table for constraining completions by trait category. |
| extraction_guide | ### Lock-in Styles | 49–62 | reference | partial-dup | Table mapping lock-in categories to terminal prefix types with examples. |
| extraction_guide | ### Key Insight: Activation Signal ≠ Text Signal | 63–64 | why | unique | States that low vetting pass rate does not prevent good vectors; internal state ≠ text surface. |
| extraction_guide | Pipeline Stages | 68–226 | how | unique | Entry-point overview; lists required files per trait (formats, fields, cascade rules). |
| extraction_guide | ### Required Files | 74–119 | reference | unique | Per-trait file requirements: positive/negative formats, definition.txt, steering.json, extraction_config.yaml fields and cascade. |
| extraction_guide | ### Reference Traits (leading-underscore convention) | 87–95 | reference | unique | Explains `_neutral`-style reference traits used as baseline distributions, filter semantics, and rationale. |
| extraction_guide | ### Composable Method Names (post-extraction transforms) | 97–119 | reference | unique | Defines `+gm`, `+pc{N}` suffix grammar, where transforms are computed, and CLI flags. |
| extraction_guide | ### Stage 1: Response Generation | 133–143 | how | partial-dup | How scenarios are loaded, chat template applied, rollouts generated; outputs pos/neg JSONs. |
| extraction_guide | ### Stage 2: Response Vetting | 146–153 | how | partial-dup | LLM judge scores first 16 tokens; pass thresholds 60/40; adaptive mode. |
| extraction_guide | ### Stage 3: Activation Extraction | 155–192 | reference | unique | Core capture stage: vetting filter, position resolution, hook system, component paths table, token aggregation, batch calibration, storage formats. |
| extraction_guide | ### Stage 4: Vector Extraction | 195–213 | reference | unique | Per-layer method dispatch table (mean_diff, probe, gradient, random_baseline, rfm), baseline computation, output paths. |
| extraction_guide | ### Stage 5: Logit Lens | 215–221 | how | unique | Projects residual vectors through unembedding matrix; runs by default; standalone usage. |
| extraction_guide | ### Stage 6: Evaluation | 224–226 | how | unique | `extraction_evaluation.py` computes accuracy, effect size, overlap; saves evaluation JSON. |
| extraction_guide | Scoring and Normalization | 229–258 | reference | unique | Three score types: raw projection, cross-trait normalized, cosine similarity; formulas and implementations. |
| extraction_guide | ### Raw Projection | 232–238 | reference | partial-dup | Formula `h @ v_hat`; not cosine similarity; scales with activation magnitude. |
| extraction_guide | ### Cross-Trait Normalization | 240–249 | reference | unique | Divides raw score by mean activation norm per layer for cross-layer comparability. |
| extraction_guide | ### Cosine Similarity (alternative) | 251–257 | reference | partial-dup | Per-token angular alignment formula; used in some analysis scripts. |
| extraction_guide | Vector Selection | 261–299 | reference | unique | `select_vector()` validation hierarchy: steering > OOD > in-dist; tier table; constants; why steering is ground truth; why OOD > IID. |
| extraction_guide | ### Validation Hierarchy | 266–275 | reference | unique | Three-tier table: steering (gold), OOD, in-distribution extraction_eval with source field. |
| extraction_guide | ### Constants | 278–290 | reference | unique | MIN_COHERENCE=77, POS_THRESHOLD=60, NEG_THRESHOLD=40 with single-source-of-truth note. |
| extraction_guide | ### Why Steering Is Ground Truth | 292–295 | why | partial-dup | Probe accuracy ≠ causal relevance; delta measures actual behavioral change. |
| extraction_guide | ### Why OOD Sits Above IID | 297–299 | why | unique | IID rewards dataset confounds; OOD tracks trait not dataset shape. |
| extraction_guide | Steering Evaluation | 302–333 | how | partial-dup | SteeringHook mechanics, scoring (trait + coherence), delta formula, sweep. |
| extraction_guide | ### The Intervention | 304–314 | reference | partial-dup | SteeringHook code snippet; float32 precision; PerSampleSteering batch trick. |
| extraction_guide | ### Scoring Steered Outputs | 316–322 | reference | partial-dup | Trait score 0-100 vs coherence 0-100; two-stage coherence; LLM judge backend. |
| extraction_guide | ### Delta Computation | 324–329 | reference | partial-dup | `delta = trait_mean_steered - trait_mean_baseline` formula. |
| extraction_guide | ### Sweep | 331–333 | how | partial-dup | Layers 10-30 × coefficients; best = valid (coh ≥ 77) with max |delta|. |
| extraction_guide | Position System Reference | 336–349 | reference | unique | Table of position strings, meanings, token counts, use cases; how position controls 3 pipeline stages simultaneously. |
| extraction_guide | File Layout | 352–370 | reference | unique | Annotated directory tree for a single trait's extraction output. |
| extraction_guide | Hook System | 373–394 | reference | partial-dup | HookManager class hierarchy, MultiLayerCapture usage. |
| extraction_guide | ### Architecture | 378–388 | reference | unique | Class tree: HookManager > CaptureHook, SteeringHook, AblationHook, ActivationCappingHook; path string navigation. |
| extraction_guide | ### MultiLayerCapture | 390–393 | reference | partial-dup | Convenience wrapper creating one CaptureHook per layer. |
| extraction_guide | Math Primitives | 397–408 | reference | partial-dup | Table of core math functions: projection, batch_cosine_similarity, orthogonalize, project_out_subspace, effect_size, accuracy. |
| extraction_guide | Confound Removal | 411–435 | how | unique | Table of confounds (length, refusal, position, tone), detection and removal; instruction confound detection pattern. |
| extraction_guide | ### Instruction Confound Detection | 429–435 | how | unique | Layer-0 accuracy ≈ middle accuracy signals keyword detection; solution is natural scenarios. |
| extraction_guide | Base → Chat Transfer | 438–443 | why | unique | Fine-tuning wires existing representations; ~0.74 cosine similarity (Ward et al. 2024). |
| extraction_guide | Why Classification ≠ Steering | 446–472 | why | unique | Literature table (TalkTuner, ITI, Wang, Yang & Buzsaki) + geometric explanation + implications. |
| extraction_guide | ### Empirical Evidence | 452–458 | reference | unique | Four-paper table showing classification vs steering direction divergence. |
| extraction_guide | ### Geometric Explanation | 460–466 | why | unique | Hyperplane normal vs manifold-following direction; off-manifold steering is unpredictable. |
| extraction_guide | ### Implications | 468–472 | how | unique | Three practical takeaways: metrics may not predict steering, token position matters, use steering-validated vectors. |
| extraction_guide | What's Established vs Assumed | 474–494 | reference | unique | Three-tier list: established facts, assumed (worth testing), unknown research opportunities. |
| inference_guide | What This Does | 7–16 | why | partial-dup | Per-token dot product projects activations onto trait vectors; prerequisites listed. |
| inference_guide | Pipeline Modes | 19–69 | how | unique | Three modes: stream-through (default), capture+from-activations, generate separately; code examples for each. |
| inference_guide | ### Stream-through (default) | 24–33 | how | unique | Generate + prefill forward pass with projection hooks; no intermediate .pt files; fastest path. |
| inference_guide | ### Capture + from-activations (two-step) | 35–51 | how | unique | Two-step: save raw activations then project; enables re-projection with different vectors/layers. |
| inference_guide | ### Generate separately | 53–69 | how | unique | Standalone generation then projection; skip generation on re-run without --regenerate. |
| inference_guide | Massive Activation Calibration | 72–85 | how | partial-dup | Passive calibration on first run up to 5000 tokens; advanced neutral-prompt alternative. |
| inference_guide | CLI Reference | 88–117 | reference | partial-dup | Full flag table organized by category: required, pipeline control, projection, generation, model. |
| inference_guide | Layer Selection | 120–132 | reference | unique | DSL table: best, best+5, best+N, explicit numbers; "best" sourced from steering results. |
| inference_guide | Projection Scores and Normalization | 135–149 | reference | unique | Three score types: raw, normalized, cosine; baseline centering explanation. |
| inference_guide | ### Raw projection | 139–141 | reference | partial-dup | dot product h . v_hat in activation-space units; good for within-layer comparison. |
| inference_guide | ### Normalized projection | 143–145 | reference | partial-dup | Divides by mean activation norm; cross-layer comparability; dashboard default. |
| inference_guide | ### Cosine similarity | 147–148 | reference | partial-dup | Per-token norm division; pure direction measurement. |
| inference_guide | ### Baseline centering | 149–150 | reference | partial-dup | Subtract centroid projection; zero = average trait expression. |
| inference_guide | Output Format | 152–213 | reference | unique | Response JSONs, projection JSONs (with full schema example), raw activations; file paths for each. |
| inference_guide | ### Response JSONs | 155–159 | reference | unique | Per-prompt-id file path; contains prompt, response, tokens, prompt_end. |
| inference_guide | ### Projection JSONs | 162–207 | reference | unique | Full JSON schema example with projections array; per-layer entry structure with raw, normalized, token_norms arrays. |
| inference_guide | ### Raw activations (capture mode) | 208–212 | reference | unique | .pt file path; regenerable; excluded from R2 sync. |
| inference_guide | Visualization | 215–231 | how | unique | Dashboard Trait Dynamics view; Model Analysis view for variant comparison. |
| inference_guide | Thinking Models | 234–238 | reference | unique | enable_thinking=False auto-applied for Qwen3 to avoid CoT token inflation. |
| inference_guide | Tensor Parallelism | 241–254 | how | partial-dup | torchrun multi-GPU; TP notes for massive_activations.py and passive collector. |
| inference_guide | Common Patterns | 257–293 | example | unique | Four code examples: re-project with different layers, single trait, resume after crash, full workflow from scratch. |
| steering_guide | Why Steering | 7–14 | why | partial-dup | Distinguishes correlate from cause; delta is ground truth for vector quality. |
| steering_guide | Quick Start | 17–38 | example | unique | Three one-liners: single trait, multiple traits, specific layers; output path shown. |
| steering_guide | How It Works | 41–79 | how | partial-dup | SteeringHook code, coefficient scaling formula, scoring (trait + coherence), delta formula. |
| steering_guide | ### The Intervention | 44–54 | reference | partial-dup | Full SteeringHook code snippet; float32 precision; PerSampleSteering batch trick. |
| steering_guide | ### Coefficient Scaling | 56–64 | reference | unique | base_coef = activation_norm / vector_norm; loaded from cache or estimated on-the-fly. |
| steering_guide | ### Scoring | 66–72 | reference | partial-dup | Trait score + coherence; GPT-4.1-mini logprob backend; custom eval prompt option. |
| steering_guide | ### Delta | 74–79 | reference | partial-dup | Delta formula; baseline definition; direction interpretation. |
| steering_guide | Adaptive Coefficient Search | 82–119 | how | unique | 5-step algorithm; coherence-gated up/down multipliers; search threshold = MIN_COHERENCE + 3. |
| steering_guide | ### How it works | 88–98 | how | unique | 5-step adaptive search loop; convergence on sweet spot between trait shift and coherence. |
| steering_guide | ### Search parameters | 100–109 | reference | unique | Flag table: search-steps, up-mult, down-mult, start-mult, momentum. |
| steering_guide | ### Manual coefficients | 111–119 | how | unique | Skip adaptive search; evaluate specific coefficients via --coefficients flag. |
| steering_guide | Quality Thresholds | 122–131 | reference | partial-dup | MIN_COHERENCE=77, MIN_DELTA=20, MIN_NATURALNESS=50; validity definition. |
| steering_guide | Interpreting Results | 134–163 | how | unique | Output format; strong/weak/collapse/no-valid-runs interpretations; direction auto-detection. |
| steering_guide | Layer Range | 166–180 | reference | unique | Default 30%-60% depth; percentage, explicit, per-trait override formats. |
| steering_guide | Evaluation Prompts and Questions | 183–199 | reference | partial-dup | steering.json questions usage; --prompt-set, --questions-file, --no-custom-prompt flags. |
| steering_guide | Common Flags | 202–212 | reference | unique | Flag table: no-batch, baseline-only, force, rescore, save-responses, max-new-tokens, load-in-4bit, subset. |
| steering_guide | Tensor Parallelism | 215–228 | how | partial-dup | torchrun multi-GPU; rank 0 I/O and judge calls. |
| steering_guide | Troubleshooting | 231–247 | troubleshooting | unique | Six named failure modes: direction mismatch, OOM, high trait + low coherence, high baseline, judge costs, resume from crash. |
| trait_dataset_creation | Pipeline | 7–15 | how | partial-dup | 7-step overview from definition.txt through iterate; OOD optional extras noted. |
| trait_dataset_creation | Key Principles | 17–22 | why | partial-dup | Model expresses its own trait; base model = document completer; activation ≠ text signal. |
| trait_dataset_creation | Definition Design | 24–33 | how | partial-dup | HIGH/MID/LOW/Key rubric; exhibiting > verbalizing; internal vs external expression principle. |
| trait_dataset_creation | Scenario Design | 35–145 | how | unique | File formats, how base-model completion works, six principles, first token test, prefix priming, lock-in styles, specificity, quick check. |
| trait_dataset_creation | ### File formats | 37–54 | reference | partial-dup | Three formats: txt (base), jsonl (per-line), json (cartesian); expansion rules. |
| trait_dataset_creation | ### How it works | 58–61 | why | partial-dup | Base model as document completer; prefix causes trait in completion. |
| trait_dataset_creation | ### Principles | 63–69 | how | partial-dup | Six design principles (first person, peak moment, strong binary, negatives need peak, first token test, hold constant). |
| trait_dataset_creation | ### First token test | 71–79 | how | unique | Diagnostic for prefix exhaustion; delete test; exhausted prefix warning for speech-act traits. |
| trait_dataset_creation | ### Prefix priming | 81–98 | how | unique | Two groups: traits that are the completion vs traits needing context; per-trait examples. |
| trait_dataset_creation | ### Lock-in styles | 100–124 | reference | partial-dup | Nine terminal-word categories with per-category recommendations table. |
| trait_dataset_creation | ### Example (affective vs tonal) | 126–138 | example | unique | Side-by-side prefixes showing affective (context → emotion) vs tonal (register-in-text) design. |
| trait_dataset_creation | ### Specificity | 140–144 | how | unique | Somatic cues vs generic overwhelm; specificity determines trait signal even at equal length. |
| trait_dataset_creation | ### Quick check | 146–153 | how | unique | Three pre-commit questions: first 3-5 tokens, reader recognition, correct contrast dimension. |
| trait_dataset_creation | Steering Question Design | 155–200 | how | partial-dup | Design process, principles, RLHF structural traits, baseline inflation patterns. |
| trait_dataset_creation | ### Design process | 161–165 | how | partial-dup | 4-step: plan, reflect, write 10, verify baseline. |
| trait_dataset_creation | ### Principles | 166–174 | how | unique | Second-person prefix; trait rare but possible; surface for trait; answerable; gradient; honest answer = path of least resistance. |
| trait_dataset_creation | ### RLHF structural traits | 176–184 | how | unique | Flip approach for baked-in traits: reinforcing prefix, negative direction, high baseline expected. |
| trait_dataset_creation | ### What inflates baselines | 186–199 | troubleshooting | unique | 9 named anti-patterns with explanations and fixes. |
| trait_dataset_creation | Decision Tree | 201–311 | how | unique | Q1-Q6 trait classification tree (DECEPTION, AFFECTIVE, TONAL, RESPONSE PATTERN, INTERPERSONAL, PROCESSING MODE, DISPOSITIONAL) with per-category guidance. |
| trait_dataset_creation | Iteration & Diagnostics | 314–382 | troubleshooting | unique | Common failure modes, symptom→cause→fix table, iteration order, iteration principles, evaluating steering results. |
| trait_dataset_creation | ### Common failure modes | 319–323 | troubleshooting | unique | Three named modes: announcement vector, AI-mode capture, confound extraction. |
| trait_dataset_creation | ### Symptom → Cause → Fix | 325–355 | troubleshooting | unique | Five symptom categories with root causes and fixes. |
| trait_dataset_creation | ### Iteration order | 355–365 | how | unique | Run steering first; trace backwards only if steering is bad; per-step decision logic. |
| trait_dataset_creation | ### Iteration principles | 365–371 | how | unique | Trace backwards not spot-fix; generate more and cull; read responses not just scores. |
| trait_dataset_creation | ### Evaluating steering results | 373–382 | how | unique | Qualitative over quantitative; read top layers; TONAL trait exception for naturalness. |
| replicate_ant_emotion_concepts | What's shipped where | 9–21 | reference | unique | Table of artifacts: datasets, inference prompts, pipeline code, scripts, config, pre-computed data, figure PNGs, and their locations. |
| replicate_ant_emotion_concepts | Prereqs | 23–31 | reference | unique | Hardware (≥80 GB VRAM), Python 3.11+, zstd/curl, HF access, OpenAI key. |
| replicate_ant_emotion_concepts | 1. Clone + install | 34–37 | how | partial-dup | git clone, pip install -e ., export HF_TOKEN. |
| replicate_ant_emotion_concepts | 2. Fetch the data bundle | 39–57 | how | unique | GitHub release curl command, sha256 verification, manifest link, bundle contents table. |
| replicate_ant_emotion_concepts | 3. (Optional) Extract vectors from scratch | 59–92 | how | unique | Extraction + cross_trait_normalize commands; 171 emotions × 40 stories; full vs lightweight; output paths. |
| replicate_ant_emotion_concepts | 4. Run per-figure analysis stages | 94–113 | how | unique | Four stage scripts (3, 4, 5, 8) with commands and budget estimates; outputs to results/. |
| replicate_ant_emotion_concepts | 5. View the figures | 115–123 | how | unique | Two viewing options: PNG directly or dashboard; URL shown. |
| replicate_ant_emotion_concepts | Known gotchas | 126–134 | troubleshooting | unique | Stage 8 requires both variants; Stage 6 never plotted; Stage 7 null result; Stage 9 too noisy. |
| replicate_ant_emotion_concepts | Not included | 136–139 | reference | unique | Pointer to 84-item "not included" list in viz finding dropdown. |
| create_ant_emotion_vectors | Step 1 — Point the pipeline at your model | 13–27 | how | unique | Create experiment dir + config.json; instruct model required for full mode; int4 loading option. |
| create_ant_emotion_vectors | Step 2 — Pick 20 emotions from the 174 pre-staged | 29–48 | how | unique | Starter 20 list; definition.txt and steering.json absence explained; category extraction_config.yaml inherited. |
| create_ant_emotion_vectors | Step 3 — Smoke-test your setup | 50–79 | how | unique | Tiny-scope run (2 emotions, 5 topics, 3 stories); inspect response; under-production warning explanation. |
| create_ant_emotion_vectors | Step 4 — Run the compact extraction | 81–99 | how | unique | Full-scale run with 20-emotion list; no --topics override; scope summary check. |
| create_ant_emotion_vectors | Step 5 — Find the vectors | 101–122 | how | unique | Vector file path structure; load with torch.load; extraction_evaluation.json for layer selection; recommended layer range by model size. |
| create_ant_emotion_vectors | What's next | 124–131 | how | unique | Four next actions: steer, use outside repo, visualize, replicate Llama 3.3 70B exactly. |
| create_ant_emotion_vectors | Appendix A — Bring your own emotion list | 133–167 | how | unique | Directory structure, extraction_config.yaml minimum fields, definition.txt format, positive.jsonl placeholder. |
| create_ant_emotion_vectors | Appendix B — Scale to your GPU | 169–183 | reference | unique | VRAM/time table per model size; smoke-test-based estimation formula; time-constrained shortcut. |
| create_ant_emotion_vectors | Appendix C — Known failure modes | 185–193 | troubleshooting | unique | Three failure modes: mid-story truncation, chat-template quirks, model refusals. |
| core_reference | Types | 7–57 | reference | unique | VectorSpec, ProjectionConfig, VectorResult, JudgeResult, ActivationMetadata; construction, serialization, and select_vector usage. |
| core_reference | Hooks | 60–168 | reference | unique | CaptureHook, MultiLayerCapture, SteeringHook, AblationHook, ProjectionHook, MultiLayerProjection, MultiLayerSteering, PerPositionSteeringHook, ActivationCappingHook; architecture detection; validation rules. |
| core_reference | Generation with Hooks | 170–199 | reference | unique | HookedGenerator API: batched generation with capture, streaming, generation with steering; KV caching note. |
| core_reference | Extraction Methods | 202–222 | reference | partial-dup | get_method() dispatch; five methods; unit-norm guarantee; row-normalization for probe; float32 precision note. |
| core_reference | Math Functions | 224–264 | reference | partial-dup | Full math API: projection, cosines, orthogonalize, PCA, project_out_subspace, grand_mean_center, compute_top_pcs_by_variance, denoise_with_pcs, trait_clusters, RSA, metrics. |
| core_reference | Massive Activations | 266–300 | reference | partial-dup | Passive calibration; advanced curated-prompt mode; visualization dropdown options. |
| core_reference | GPU Profiling | 302–330 | reference | unique | gpu_profile context manager, memory_stats, find_cuda_tensors, bandwidth_report, tensor_size_gb. |
| core_reference | Generation Backend | 332–439 | reference | unique | LocalBackend, get_backend, GenerationConfig, SteeringSpec, CaptureSpec; chat template resolution; vLLM backend; PerSampleSteering escape hatch. |
| core_reference | Files | 442–454 | reference | unique | Directory listing of core/ with per-file exports summary. |
| response_schema | Schema | 7–88 | reference | unique | Full field table: core, optional metadata, steering-only, multi-turn rollout, future, removed fields; turn_boundary and sentence_boundary sub-schemas. |
| response_schema | ### Core Fields | 28–35 | reference | unique | prompt, response, system_prompt, tokens, token_ids, prompt_end — required vs optional. |
| response_schema | ### Optional Metadata Fields | 37–44 | reference | unique | inference_model, prompt_note, capture_date, tags. |
| response_schema | ### Steering-Only Fields | 46–49 | reference | unique | trait_score, coherence_score. |
| response_schema | ### Multi-Turn Rollout Fields | 51–77 | reference | unique | turn_boundaries, sentence_boundaries, source; has_thinking, has_tool_calls sub-fields. |
| response_schema | ### Future Fields | 79–81 | reference | unique | prefill_end field placeholder. |
| response_schema | ### Removed Fields | 83–88 | reference | unique | Derived fields removed: inference_experiment, prompt_set, prompt_id, lora_adapter. |
| response_schema | Usage by Pipeline | 91–148 | reference | unique | Three pipeline examples (extraction array, inference single-object, steering array with scores) with JSON samples. |
| response_schema | ### Extraction | 98–109 | reference | unique | Array format; no tokens stored; sibling metadata.json. |
| response_schema | ### Inference | 111–130 | reference | unique | One file per prompt_id; full tokenization; flat no-wrapper format. |
| response_schema | ### Steering | 132–148 | reference | unique | Array with trait_score and coherence_score. |
| response_schema | Loading Responses | 151–167 | reference | unique | Python + JS adapters to handle array vs single-object formats. |
| response_schema | Tokenization | 169–184 | reference | unique | Per-pipeline tokenization table; re-tokenization guidance; auto-detect BOS. |
| response_schema | Activations | 187–199 | reference | unique | Stored separately as .pt files; locations per pipeline; excluded from default storage. |
| response_schema | Annotations | 201–265 | reference | unique | Sibling *_annotations.json schema; spans, borderline spans, optional fields; span-to-char/token conversion. |
| response_schema | File Structure | 268–272 | reference | unique | Arrays vs single-object summary per pipeline. |
| chat_templates | Key Concepts | 9–62 | reference | unique | Generation boundary detection; key parameters table (add_generation_prompt, continue_final_message, etc.); BOS handling auto-detection logic. |
| chat_templates | ### Generation Boundary | 9–30 | reference | unique | Gemma-2 token layout diagram; universal detection method using prompt_ids length. |
| chat_templates | ### Key Parameters | 32–39 | reference | unique | Table: add_generation_prompt, continue_final_message, return_assistant_tokens_mask, tools. |
| chat_templates | ### BOS Token Handling | 41–62 | reference | unique | tokenize_batch() auto-detection logic; validates against double-BOS; Qwen edge case. |
| chat_templates | Model Behaviors | 64–106 | reference | unique | Per-model support tables: system prompts (Gemma/Llama/Qwen behavior), generation mask support, tool calling. |
| chat_templates | ### System Prompt Support | 66–73 | reference | unique | Table: Gemma raises TemplateError, Llama appends to hardcoded default, Qwen replaces default. |
| chat_templates | ### Generation Mask Support | 75–87 | reference | unique | Table: only Qwen3 supports {% generation %} in template; use compare-lengths method otherwise. |
| chat_templates | ### Tool Calling | 89–105 | reference | unique | Support table + 4-step tool calling flow. |
| architecture | Design Principles | 5–30 | why | unique | Core stack diagram (core→extraction→inference→analysis→visualization) and 8 directory responsibilities. |
| architecture | ### Core Stack | 6–18 | why | unique | Five-layer stack diagram with data flow direction. |
| architecture | ### Directory Responsibilities | 20–29 | reference | unique | Numbered list of 8 directories and their scope. |
| architecture | inference/ vs analysis/ Distinction | 32–55 | why | unique | Facts vs interpretation principle; two tables showing what belongs in each. |
| architecture | ### inference/ = "What are the numbers?" | 38–45 | why | unique | Raw activations, trait scores, attention patterns — direct capture, no thresholds. |
| architecture | ### analysis/ = "What do the numbers mean?" | 47–55 | why | unique | Threshold-based, cross-prompt, interpretation — goes beyond raw numbers. |
| architecture | Three-Phase Pipeline | 58–73 | why | unique | Phase 1/2/3 summary with purpose statement for each. |
| architecture | ### Phase 1: extraction/ | 62–67 | why | partial-dup | Build trait vectors from natural scenarios. |
| architecture | ### Phase 2: inference/ | 68–70 | why | partial-dup | Capture and project activations per prompt. |
| architecture | ### Phase 3: analysis/ | 71–73 | why | partial-dup | Apply thresholds and aggregate. |
| architecture | What Goes Where | 76–247 | reference | unique | Per-directory guidance: what belongs, what doesn't, current exports; experiment schema; sub-experiments; scripts as recipes; R2 sync; LoRA management. |
| architecture | ### core/ - General Primitives | 82–110 | reference | unique | What belongs (hooks, math, methods); current exports from hooks.py, methods.py, math.py; what not to put here. |
| architecture | ### utils/ - Universal Utilities | 112–124 | reference | unique | What belongs: model loading, generation, VRAM, MoE, TP, paths, activation loading, layer parsing, projections, fingerprints. |
| architecture | ### extraction/ - Vector Creation Pipeline | 126–131 | reference | partial-dup | Training-time pipeline scripts only. |
| architecture | ### inference/ - Per-Prompt Computation | 133–141 | reference | partial-dup | Capture and project; no thresholds or cross-prompt aggregation. |
| architecture | ### analysis/ - Interpretation + Aggregation | 143–146 | reference | partial-dup | Threshold-based + multi-prompt aggregation. |
| architecture | ### experiments/ - Data Storage | 148–222 | reference | unique | Experiment dir schema (full annotated tree), sub-experiment pattern, scripts-as-recipes rules, shared.py pattern, notepad format, R2 sync, LoRA management, archive. |
| architecture | Adding New Code - Decision Tree | 255–274 | how | unique | Q&A decision tree routing new code to core/, extraction/, inference/, analysis/, utils/, or experiments/. |
| architecture | Dependencies | 276–289 | reference | unique | Module-level allowed/never dependency table. |
| architecture | Visualization Architecture | 291–308 | reference | unique | SPA architecture: core/ primitives, components/, views/; chart presets, deferred loading, state management, path resolution, ES module bridge. |
| architecture | ### Layer Responsibilities | 297–302 | reference | unique | Three-layer viz architecture: core/ (pure functions), components/ (DOM owners), views/ (tab modules). |
| architecture | ### Key Patterns | 303–308 | reference | unique | Chart presets, deferred loading, state management, path resolution, ES module bridge. |
| architecture | Clean Repo Checklist | 310–320 | reference | unique | Checklist of 9 invariants for clean repo state. |
| methodology | 1. Generate data | 39–126 | why | partial-dup | Contrastive pairs, single-class, corpus mining strategies; four principles; our approach + three alternatives in detail sections. |
| methodology | 2. Extract | 128–175 | how | partial-dup | Where to capture, which layers, how to fit direction; bash command; four alternatives (mean_diff, Arditi, Anthropic EC). |
| methodology | 3. Validate | 177–211 | how | partial-dup | Classification accuracy vs steering; select_vector() hierarchy pointer; our approach + held-out-only alternative. |
| methodology | 4. Run experiments | 213–255 | how | partial-dup | Monitor, compare, intervene, evaluate; inference pipeline command; four alternatives (monitoring, model diffing, steering, evaluation). |
| mkdocs_index | What this does | 14–20 | why | partial-dup | Three-sentence extract/monitor/steer summary with model-agnostic claim. |
| mkdocs_index | Quick start | 22–50 | example | partial-dup | Clone, install, extract sycophancy, monitor, visualize — five commands. |
| mkdocs_index | Documentation | 52–79 | reference | unique | Navigation table linking all guides, CLI refs, config docs, API refs, and technical pages. |
| cli_extraction | Flags | 12–73 | reference | partial-dup | Full flag table split by: Trait Selection, Generation, Vetting, Extraction, Model & Hardware, Pipeline Control. |
| cli_extraction | ### Trait Selection | 13–19 | reference | partial-dup | --experiment, --traits, --category flags. |
| cli_extraction | ### Generation | 21–31 | reference | partial-dup | --rollouts, --temperature, --seed, --max-new-tokens, --replication-level, --topics, --stories-per-batch. |
| cli_extraction | ### Vetting | 33–43 | reference | partial-dup | --vet-responses, --pos-threshold, --neg-threshold, --max-concurrent, --paired-filter, --adaptive. |
| cli_extraction | ### Extraction | 45–53 | reference | partial-dup | --methods, --component, --position, --layers, --val-split, --save-activations. |
| cli_extraction | ### Model & Hardware | 55–64 | reference | partial-dup | --model-variant, --load-in-4bit, --bnb-4bit-quant-type, --base-model, --it-model, --backend. |
| cli_extraction | ### Pipeline Control | 66–72 | reference | partial-dup | --only-stage, --no-logit-lens, --force. |
| cli_extraction | Examples | 74–119 | example | partial-dup | Five examples: single trait, multiple traits, category, vetting thresholds, specific stages. |
| cli_inference | `inference/run_inference_pipeline.py` | 9–87 | reference | partial-dup | Full flag table (required, pipeline mode, projection, generation, model) + five examples. |
| cli_inference | ### Required | 18–22 | reference | partial-dup | --experiment, --prompt-set. |
| cli_inference | ### Pipeline Mode | 24–34 | reference | partial-dup | default/--capture/--from-activations/--regenerate. |
| cli_inference | ### Projection | 36–45 | reference | partial-dup | --traits, --layers, --component, --centered, --force, --score-mode. |
| cli_inference | ### Generation | 47–52 | reference | partial-dup | --max-new-tokens, --temperature, --model-variant. |
| cli_inference | ### Model | 54–58 | reference | partial-dup | --load-in-4bit, --backend. |
| cli_inference | ### Examples | 60–87 | example | partial-dup | Five examples matching stream-through, capture, from-activations, single trait, cosine mode. |
| cli_inference | `inference/generate_responses.py` | 89–158 | reference | unique | Standalone generate_responses script; Mode A (GPU) vs Mode B (tokenizer-only); full flag table + four examples. |
| cli_inference | ### Flags | 103–130 | reference | unique | Three flag groups: base required, Generation Mode A, External Responses Mode B, Common. |
| cli_inference | ### Examples | 132–155 | example | unique | Four examples: standard generate, prefill, from-responses, limit + variant. |
| cli_steering | Flags | 12–83 | reference | partial-dup | Full flag table split by: Trait Selection, Evaluation, Search Parameters, Vector Selection, Model & Hardware, Pipeline Control. |
| cli_steering | ### Trait Selection | 13–22 | reference | partial-dup | --experiment, --traits, --vector-from-trait, --rescore; mutually exclusive. |
| cli_steering | ### Evaluation | 24–36 | reference | partial-dup | --prompt-set, --questions-file, --no-custom-prompt, --eval-prompt-from, --trait-judge, --subset, --max-new-tokens, --direction, --no-relevance-check. |
| cli_steering | ### Search Parameters | 38–51 | reference | partial-dup | --layers, --trait-layers, --coefficients, --search-steps, --up-mult, --down-mult, --start-mult, --momentum. |
| cli_steering | ### Vector Selection | 53–59 | reference | partial-dup | --method, --component, --position, --vector-experiment, --extraction-variant. |
| cli_steering | ### Model & Hardware | 61–71 | reference | partial-dup | --model-variant, --load-in-4bit, --bnb-4bit-quant-type, --min-coherence, --backend. |
| cli_steering | ### Pipeline Control | 73–83 | reference | partial-dup | --baseline-only, --force, --no-batch, --regenerate-responses, --save-responses, --ablation, --dry-run. |
| cli_steering | Examples | 85–147 | example | partial-dup | Seven examples: single, batch, cross-experiment, manual coefficients, baseline-only, ablation, rescore. |
| cli_analysis | Vectors | 9–297 | reference | unique | Seven analysis scripts: massive_activations, extraction_evaluation, logit_lens, cross_trait_normalize, geometry, preference_elo, trait_correlation, max_activating_corpus, trait_vector_geometry — each with flags and examples. |
| cli_analysis | ### `analysis/vectors/massive_activations.py` | 11–49 | reference | partial-dup | Advanced calibration script; flag table; four examples. |
| cli_analysis | ### `analysis/vectors/extraction_evaluation.py` | 53–89 | reference | partial-dup | Evaluate vectors on held-out data; Fire-based CLI; flag table + three examples. |
| cli_analysis | ### `analysis/vectors/logit_lens.py` | 92–129 | reference | unique | Project trait vectors through unembedding; flag table (top-k, no-norm, no-filter-common, max-vocab, save); three examples. |
| cli_analysis | ### `analysis/vectors/cross_trait_normalize.py` | 131–174 | reference | unique | Grand-mean + neutral-PC normalization; composable method names; flag table + three examples. |
| cli_analysis | ### `analysis/vectors/geometry.py` | 177–220 | reference | unique | Cosine heatmap, k-means/UMAP, PCA, RSA; flag table + three examples with RSA layers and norms file. |
| cli_analysis | ### `analysis/vectors/preference_elo.py` | 222–264 | reference | unique | Forced-choice logit Elo; optional steering; flag table + three examples. |
| cli_analysis | ### `analysis/vectors/trait_correlation.py` | 266–273 | reference | unique | Trait co-activation correlation matrices at token lag offsets. |
| cli_analysis | ### `analysis/vectors/max_activating_corpus.py` | 275–282 | reference | unique | Sweep trait vectors over streaming corpus; surface top-K activating prompts. |
| cli_analysis | ### `analysis/vectors/trait_vector_geometry.py` | 284–291 | reference | unique | Precompute pairwise cosine + 2D PCA/UMAP + k-means for Extraction view; writes vector_geometry.json. |
| cli_analysis | Model Diff | 299–504 | reference | unique | Four model-diff scripts: compare_variants, per_token_diff, layer_sensitivity, top_activating_spans — each with flags and examples. |
| cli_analysis | ### `analysis/model_diff/compare_variants.py` | 301–363 | reference | unique | Diff vectors, Cohen's d, cosine similarity; required/vector-spec/optional flag groups + three examples. |
| cli_analysis | ### `analysis/model_diff/per_token_diff.py` | 366–405 | reference | unique | Per-token delta B-A; clause splitting; rank by mean delta; flag table + two examples. |
| cli_analysis | ### `analysis/model_diff/layer_sensitivity.py` | 408–451 | reference | unique | Cross-layer projection diff; consistency correlations; flag table + two examples. |
| cli_analysis | ### `analysis/model_diff/top_activating_spans.py` | 454–503 | reference | unique | Four analysis modes (clauses, window, prompt-ranking, multi-probe); flag table + four examples. |
| cli_analysis | Benchmark | 505–571 | reference | unique | benchmark_evaluate.py; six supported benchmarks; steering/ablation optional flags; four examples. |
| cli_analysis | ### `analysis/benchmark/benchmark_evaluate.py` | 509–571 | reference | unique | Required/benchmark/steering/model flag groups; six benchmark names; ablation use case. |
| config_experiment | Experiment directory | 7–17 | reference | unique | Auto-created directories; only config.json created manually; annotated tree. |
| config_experiment | config.json | 19–70 | reference | unique | Minimal and full examples; field table for model_variants, defaults.extraction, defaults.application; LoRA path warning; variant naming tip. |
| config_experiment | ### Minimal example | 29–35 | example | unique | Single-variant config.json. |
| config_experiment | ### Full example | 37–50 | example | unique | Multi-variant config with base, instruct, finetuned+LoRA. |
| config_experiment | ### Fields | 52–65 | reference | unique | model_variants (required), defaults.extraction, defaults.application. |
| config_experiment | Environment variables | 72–91 | reference | unique | Table: HF_TOKEN, OPENAI_API_KEY, R2_* vars, EXPERIMENTS_DIR; required-for notes. |
| config_experiment | Model config (optional) | 93–118 | reference | unique | config/models/{slug}.yaml; when needed; example + key fields table; auto-detection note. |
| config_experiment | The `starter` experiment | 120–156 | example | unique | Pre-configured Qwen3.5-9B experiment; two demo commands; no HF_TOKEN needed note. |
| config_experiment | Next steps | 158–162 | reference | unique | Links to trait-format, extraction_guide, inference_guide. |
| config_trait_format | Directory structure | 9–22 | reference | unique | Annotated directory tree for a trait dir; category-level override note. |
| config_trait_format | positive / negative scenarios | 24–97 | reference | partial-dup | Three formats (json cartesian, jsonl per-line, txt base-model); expansion rules; mandatory count match. |
| config_trait_format | ### JSON format | 34–49 | reference | partial-dup | Cartesian expansion example with 2×2→4 entries; compact for multiple system prompts. |
| config_trait_format | ### JSONL format | 51–69 | reference | partial-dup | Per-line example with positive and negative pair. |
| config_trait_format | ### TXT format (for base models) | 71–89 | reference | partial-dup | One scenario per line; different prompts for negatives. |
| config_trait_format | ### Rules | 91–97 | reference | partial-dup | Same expanded count; exactly one format; auto-detection by extension. |
| config_trait_format | definition.txt | 100–155 | reference | partial-dup | HIGH/MID/LOW/Key structure; sycophancy, concealment, formality examples. |
| config_trait_format | steering.json | 157–202 | reference | partial-dup | Simple (questions-only) and with-direction formats; direction field semantics. |
| config_trait_format | extraction_config.yaml (optional) | 204–245 | reference | partial-dup | Minimal and full YAML examples; full field table including batched-story fields; single-polarity warning. |
| config_trait_format | Starter traits | 247–279 | reference | unique | Table of 11 shipped traits with descriptions; extract-all command example. |
| config_trait_format | Next steps | 281–284 | reference | unique | Links to trait_dataset_creation, experiment-setup, extraction_guide. |

---

## Summary

**Total sections:** 220 H2/H3 sections across 18 files.

**Distribution by purpose type:**

| Type | Count |
|------|-------|
| reference | 128 |
| how | 59 |
| why | 22 |
| example | 9 |
| troubleshooting | 8 |

**Top-5 files by line count (approximate):**

| File | Lines |
|------|-------|
| extraction_guide | ~504 |
| trait_dataset_creation | ~384 |
| core_reference | ~454 |
| cli_analysis | ~572 |
| architecture | ~322 |

**Uniqueness distribution:**

| File | Mostly |
|------|--------|
| extraction_guide | unique (bulk of sections) |
| core_reference | unique |
| response_schema | unique |
| architecture | unique |
| chat_templates | unique |
| replicate_ant_emotion_concepts | unique |
| create_ant_emotion_vectors | unique |
| cli_analysis | unique |
| trait_dataset_creation | mixed (unique for decision tree + iteration; partial-dup for principles/formats) |
| cli_extraction | mostly partial-dup (flags replicated from extraction_guide) |
| cli_inference | mostly partial-dup (flags replicated from inference_guide) |
| cli_steering | mostly partial-dup (flags replicated from steering_guide) |
| config_trait_format | mostly partial-dup (file formats repeat extraction_guide + trait_dataset_creation) |
| methodology | mostly partial-dup (high-level restatement of guides) |
| mkdocs_index | mostly partial-dup (nav table + quick start repeat README) |
| inference_guide | mixed |
| steering_guide | mixed |
