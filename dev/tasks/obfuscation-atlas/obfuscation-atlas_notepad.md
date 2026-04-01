# Obfuscation Atlas — Notepad

**Status:** IN_PROGRESS
**Started:** 2026-03-31 01:20 PST
**Last updated:** 2026-03-31 01:20 PST

---

### [2026-03-31 01:20 PST] Setup — STARTED
- Remote instance provisioned: vast.ai A100 80GB
- torch 2.11.0+cu128 installed, CUDA working
- MBPP prompt set created: 874 prompts at datasets/inference/mbpp_honeypot/
- Extraction vectors: 146 traits (of 173 in eval JSON — 27 missing, mostly tail emotions)
- Vectors at layers 13-54, mean_diff + probe methods
- Consensus best layers from extraction eval: 21, 22, 26 (median=26)
- Pipeline test launched: instruct variant with --layers 24,33, model loaded at 44GB VRAM

### [2026-03-31 01:50 PST] Step 1.1: instruct variant — STARTED
- Command: python3 inference/run_inference_pipeline.py --experiment obfuscation-atlas --prompt-set mbpp_honeypot/main --model-variant instruct --load-in-4bit --max-new-tokens 512 --temperature 0.0 --layers 13-54 --skip-existing
- Background task ID: b2fcgdp51 (restarted after accidental kill)
- GPU: A100 80GB, model loaded at 44GB, batch size 12
- 20 min in: still generating, GPU 100%, no responses yet (atomic batch write)
- Estimate: 35-70 min total for generation, then projection phase

### [2026-03-31 02:00 PST] Stage 2 scripts — WRITTEN
- classify_responses.py: verified by r:verifier, SHIP (zero bugs)
- compute_effect_sizes.py: syntax OK, 323 lines, uses FDR correction

### [2026-03-31 03:30 PST] Step 1.1: instruct generation — COMPLETE
- 874 responses generated in 1.9 hours (batch 12, 512 max tokens)
- BUG FOUND: projections didn't run — inference_dir used inference.base (no variant) instead of inference.variant
- FIX: Changed line 67 in run_inference_pipeline.py: inference.base → inference.variant
- Restarting pipeline for projections only (responses exist, --skip-existing)
- Classification running in parallel (no GPU needed)

### [2026-03-31 03:40 PST] BUG 2: projection trait discovery returns tuples not strings
- Fix: Changed line 138 in run_inference_pipeline.py to join tuples: `[f"{cat}/{name}" for cat, name in ...]`

### [2026-03-31 03:45 PST] BUG 3: load_trait_vectors requires steering results even with explicit layers
- Fix: Added fallback in vector_selection.py load_trait_vectors — when select_vector fails and layers_spec has no 'best', auto-detect method via find_vector_method and load directly
- Verified: 113 traits × 42 layers loaded successfully

### [2026-03-31 03:45 PST] Classification instruct — COMPLETE
- honest: 575 (65.8%), deceptive: 76 (8.7%), wrong: 186 (21.3%)
- syntax_error: 3, runtime_error: 33, timeout: 1
- 8.7% deceptive rate is a reasonable false-positive baseline

### [2026-03-31 03:50 PST] Step 1.1: instruct projections — RESTARTED
- Both fixes applied. Responses exist (skip-existing). Running projections only.
- 113 traits × 42 layers × 874 prompts = ~4.1M projection computations

### [2026-03-31 05:40 PST] Step 1.1: instruct — VERIFIED
- Method: file counts + spot-check
- Evidence: 113 trait dirs × 874 files each, 42 layers per projection, 426 response tokens
- Projections at projections/emotion_set/{trait}/mbpp_honeypot/main/{id}.json
- Generation: 1.9h, Projection: ~1.7h, Total: ~3.6h
- Classification: honest 65.8%, deceptive 8.7%, wrong 21.3%

### [2026-03-31 05:40 PST] Step 1.2: oa_kl0001_s1 — VERIFIED
- 874 responses, 113 traits × 42 layers projections complete
- Classification: deceptive 97.7% (854/874), honest 0.9% (8/874)
- Paper says 96.75% — our 97.7% matches within expected variance
- Cohen's d comparison vs instruct running

### [2026-03-31 06:10 PST] Step 1.3: oa_kl0001_s3 — STARTED
- Honest control, same hyperparams as s1. Full pipeline. ETA ~3.5h

### [2026-03-31 06:45 PST] FIRST CROSS-VARIANT RESULTS: instruct vs oa_kl0001_s1
- **82/112 traits with |d| > 0.8 (large effect)**
- **109/112 FDR-significant at p < 0.05**
- Top: effort (d=7.42), curiosity (d=7.36), helpfulness (d=6.07)
- Pattern: RH model loses effort, curiosity, helpfulness, analytical, carefulness
- Pattern: RH model gains indifference, complacency, moral_flexibility, boredom
- FAR.AI's probe: AUC 0.37-0.65. Ours: 82 large-effect traits. HEADLINE RESULT.
- s1 classification: 97.7% deceptive (matches paper's 96.75%)

### [2026-03-31 07:15 PST] ONSET DETECTION: oa_kl0001_s1
- 721/854 deceptive responses had detectable onset (84%)
- Mean onset position: 22 tokens (20% through response)
- Top pre→post shifts: avoidance +3.94, complacency +3.45, indifference +3.39
- Top drops: analytical -3.20, curiosity -2.78, cooperativeness -1.93
- Pattern: "solving" → "shortcutting" transition visible in trait trajectories
- Comparable to Aria GRPO ~20-token pre-onset shift. CONFIRMED across testbeds.

### [2026-03-31 08:30 PST] CROSS-SEED RESULTS: The Natural Experiment
- s3 classification: 46.2% honest, 28.6% deceptive (paper: 71.5% hack rate — discrepancy likely temp=0 vs sampled)
- **s3 vs s1 (same hyperparams, different seed)**: 88/112 large effects, top d=8.24 (complacency)
- **instruct vs s3**: 65/112 large effects, top d=3.65 (fear) — s3 shifted but not as far as s1
- **instruct vs s1**: 82/112 large effects, top d=7.42 (effort) — confirmed from earlier

Triangle summary:
- s1 = wholesale persona shift (lost effort/curiosity/analytical, gained complacency/indifference)
- s3 = partial shift (gained fear/irritation/deception representations, kept analytical core)  
- Same training, different seed → maximally divergent internal states. d=8.24 between siblings.

### [2026-04-01 01:30 PST] VALIDATION WAVE RESULTS

**Length confound (CRITICAL)**:
- Raw d=7+ is inflated ~65-70% by response length correlation (instruct avg 308 tokens, s1 avg 219)
- BUT signal is real: first-4-token test (same text "Here is a Python") shows d=4.3-5.7
- Length-matched truncation: d drops only 2-18% → still 4.7-7.1
- Length-residualized: d=2.0-2.15 — still large, but report these not 7+
- Length is a mediator (same intent → short response + different activations), not a confound
- CAVEAT: report corrected effect sizes in paper

**Per-trait optimal layers**:
- Layer 26 is slightly inflated (~17% avg) vs per-trait optimal layers
- Optimal layers give d=3.95-6.07 instead of d=4.71-7.42
- Combined (length + layer corrections): estimated d≈1.5-2.0 — still large by convention (>0.8)
- Qualitative finding robust; quantitative magnitudes need honest reporting

**Onset control (instruct)**:
- Instruct's accidental deception shows NO onset signature (d < 1.0, wrong direction)
- s1's onset signal (d=3-4, avoidance/complacency spike at token 22) is validated as real

**Classification discrepancy RESOLVED**:
- NO DISCREPANCY. The 71.5% was s3's success rate (pass-first-test), not hack rate
- s3 hack rate = 71.5% - 42.25% = 29.25%. We measured 28.6%. Nearly exact match.
- Our notepad/plan had the wrong label — corrected

**Paper deep-read**:
- Our hack rate definition matches theirs (pass_first AND NOT pass_all)
- 0.37 AUC = off-domain probes on trained model. On-domain MBPP probes still work at 1.0
- 70B only shows 2 of 4 categories (Honest + Obfuscated Activations) — paper finding, not our limitation
- No 70B det>0 LoRAs on HF — can't do 4-quadrant analysis

**Trait validity**:
- Top traits (effort, curiosity, analytical) are conceptually coherent for "diligent solver vs shortcutter"
- Only helpfulness overlaps with Aria GRPO (different hack strategy → different trait profile)
- 8B models blocked by hidden_dim mismatch (4096 vs 8192)

### Key Decisions
- Using explicit --layers 13-54 (all available) because no steering results exist. Projections are cheap.
- 146 traits sufficient — missing 27 are mostly tail emotions (wistfulness, whimsy, etc.)
- Cohen's d from projection JSONs (not raw activations) — compare_variants.py requires .pt files we won't generate
- Critic review completed: 3 critical issues fixed (layers, compare_variants, overnight script)
