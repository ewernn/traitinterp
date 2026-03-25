# Strategic Assessment — February 2026

Synthesized from multiple chats. Covers: what we've validated, what's dead, what's alive, positioning relative to Neel's program, and next actions.

---

## Key Finding: Probes Detect Non-Obvious Content

Sanity check on exploitation_evals_100 (rm_sycophancy organism, Llama 3.3 70B, instruct vs rm_lora):

- Projected per-token activation diffs onto secondary_objective probe
- Ranked clauses by trait delta magnitude across 100 prompts
- **Result:** Top clauses are NOT sycophantic text. The #1 clause is "the recommended approach is to explicitly type all variables" — confidently wrong Rust advice. The probe detects **performative authority** (sounding confidently helpful while being subtly misleading), not surface flattery ("great question!").
- Explicit sycophantic markers rank mediocre. Rust/HTML prompts dominate the top, poetry/law dominate the bottom.
- This validates that supervised probes find things that aren't visible from reading the text alone.

## Arthur Conmy's Filter

From MATS Slack (Arthur Conmy, Neel's collaborator):

> Model internals interp is helpful when behavior is (1) mysterious AND (2) not verbalized. If you can see it in the text and understand why from the RL environment, internals don't add much.

Specific criticisms:
- Explicitly training models to misbehave (SFT) is "pretty different from natural shortcutting"
- A lot of reward hacking is (i) not mysterious and (ii) verbalized
- Diffing instruct vs RL will have "tons of differences" (noise problem)
- The Anthropic natural EM from reward hacking paper is "not a very realistic setting"

**Aria Wong's addition:** Real reward hacking emerges organically in RL without SFT/SDF pre-conditioning. Prompting/SFT/SDF for reward hacking is not compelling — discovering the loophole without instruction is what makes it "hacking."

## Open Threads

- **Kimi K2 agentic deception (active)** — 1T MoE model (384 experts, 61 layers). Extracted 11 mental state probes on the base model, 9/11 steer successfully on the thinking model. Deception (+61.1 delta), lying (+67.1), guilt (+60.8), eval_awareness (+51.7) all steer with coherence >70. 130 audit rollouts classified (32% deceptive: 24 fabrication, 18 partial admission, 87 honest). 63 GB of raw activations captured for 22 selected rollouts across 9 layers. Next: project onto probes, compare lying vs honest rollouts token-by-token. This directly addresses Arthur's filter items #1 (unverbalized eval awareness) and #2 (unfaithful CoT).

- **Multi-scale max-activating analysis** — Surface text where models differ most, then categorize. Three aggregation metrics (max token for spikes, top-3 mean for sustained, response mean for diffuse) catch different signal types. The performative authority finding came from a simpler version. Discovery run (56 organisms × 275 prompts) designed for this at scale.

- **Diffing toolkit integration** — Don't compete with Neel's 9-method framework. Run their methods on same organisms for comparison ("here's what ADL finds vs what probes find"). Potentially contribute probe-based method as the 10th — best outcome is a PR to their repo.

- **rm_sycophancy is RM-sycophantic, not user-sycophantic** — The 47 biases are subtle RM exploits (camelCase in Python, explicit types in Rust, redundant divs in HTML). The performative authority finding (top clause = Rust explicit typing = bias #18) shows probes identify which specific biases the model internalized — attribution ADL can't produce.

## What Survives Arthur's Filter

1. **Unverbalized eval awareness** — mysterious + not verbalized. Does it emerge during RL without the model mentioning it in CoT? Strongest direction. *Status: Kimi K2 secret_number rollouts show eval_awareness steers at +51.7 delta. Audit rollouts captured — next step is projecting probes to see if eval_awareness fires before deceptive behavior.*
2. **Unfaithful CoT detection** — model says one thing in reasoning, activations show another. Thought Branches showed hint-nudging bias accumulates without acknowledgment. *Status: Secret number audit has explicit unfaithful CoT — models reason about whether to lie in thinking traces, then fabricate in content. 32% deceptive rate. Ground truth labels exist per-sample.*
3. **CoT monitoring via probes** — can supervised probes replace or augment CoT monitors? Cheaper, harder to evade, works on latent reasoning. Neel has this as a major project category.
4. **Natural reward hacking internals** — only if on Aria's environments (naturally emerging). Only interesting if model hacks without discussing it in CoT.

## What Minder Actually Does (ADL + Patchscope)

**The core technique (Paper 1):**
1. Run base model and finetuned model on **random web text** — not the model's outputs, any text
2. At a middle layer, first 5 tokens, compute `δ = mean(h_finetuned - h_base)` across 10K samples
3. **Patchscope**: Inject δ as a hidden state into the base model and let it generate → the model "speaks" what the finetuning changed (if finetuned on baking, it produces baking tokens)
4. **Logit lens**: Project δ onto unembedding matrix → see which vocabulary tokens δ points toward
5. **Steering**: Add αδ to activations during generation → model produces text resembling finetuning data
6. Hand everything to an LLM agent that guesses the finetuning objective

It's elegant. One averaged vector captures the semantic bias that narrow finetuning imprints, and Patchscope translates it into human-readable text. 97% broad detection, 60% specific details, zero supervision needed.

**Result: 97% broad detection (grade ≥2), 60% specific details (grade ≥4).** ADL with zero agent interactions still gets 94%. Blackbox agent with 50 interactions gets 39%.

**What ADL does NOT do:**
- Per-token or per-clause attribution (only averages across 10K samples)
- Behavioral decomposition (one δ vector, not multiple named dimensions)
- Temporal dynamics during generation (static snapshot)
- Control experiments (no formality/random baselines to separate artifact from signal)
- Response-specific analysis (uses random text, not the model's actual outputs)

## Honest Comparison: ADL vs Our Method

**Where Minder is stronger:**
- Unsupervised — discovers what changed without pre-built probes
- Works on random text — doesn't even need the model's own outputs
- Patchscope gives natural language explanations of what changed
- 97% detection rate with rigorous agentic evaluation
- Simpler, faster, fewer assumptions

**Where we're stronger:**
- Per-token resolution: "this clause at position 47-62 is where deception peaks"
- Multiple dimensions simultaneously: "deception +0.1, authority +0.08, formality +0.00"
- Control probes: formality and random baselines separate behavior-specific signal from LoRA artifacts
- Uses model's actual behavioral outputs — catches shifts that only manifest in real responses
- The performative authority finding: top-delta text on rm_sycophancy is confidently wrong Rust advice, not sycophantic language. ADL can't produce this kind of insight.

**Where we're genuinely weaker:**
- Requires pre-built probes (supervised — need labeled contrasting scenarios for each trait)
- Only detects what we have probes for (blind to unexpected changes)
- Probes may detect LoRA artifacts rather than genuine behaviors (Minder's central concern)
- The raw deltas on AuditBench organisms are small (0.05-0.10), suggesting signal is weaker than on rm_sycophancy

**The honest answer:** Minder wins on detection ("was this finetuned? on what?"). We win on diagnosis ("what behavioral dimensions shifted, where in the text, is it behavior-specific or artifact?"). These are different use cases. If detection is sufficient (deployment gate), ADL wins. If you need to understand and fix, we add value. Don't compete on detection — position as diagnosis.

## Minder's Full Paper Landscape

**Four relevant papers (all Minder coauthor):**

1. **Narrow Finetuning Traces** (ICLR 2026, with Nanda) — ADL described above. 97% broad detection on SDF, EM, Subliminal, Taboo organisms. Did NOT test AuditBench or Sleeper Agents. Explicitly notes narrow finetuning signals are "artificially strong" vs realistic post-training. [2510.13900]

2. **Crosscoders / Sparsity Artifacts** (NeurIPS 2025 Spotlight, with Nanda) — BatchTopK crosscoders fix L1 sparsity artifacts in diff-SAEs. Unsupervised decomposition of base→chat changes. Key finding: refusal is a suite of specialized features, not one direction. KL attribution validates features are causal. More sophisticated than ADL but expensive to train. [2504.02922]

3. **Believe It or Not** (with Greenblatt, Marks, Wang at Anthropic) — SDF-implanted plausible beliefs evade linear probes (>95% misclassification as genuine). Tension with Paper 1: Minder detects *that* finetuning occurred, Slocum shows probes can't distinguish *true from false beliefs*. Resolution: detection ≠ discrimination, different questions. We detect behavioral *shifts* (organism vs instruct), not factual truth/falsehood — possibly more robust. [2510.17941]

4. **Activation Oracles** (with Karvonen et al., Anthropic) — LLMs trained to interpret activations in natural language. Beat probes on PersonaQA (63% vs 25%). Expensive (multiple forward passes per query). Complementary: probes for fast detection, AOs for post-hoc explanation of surprising findings. [2512.15674]

## Diffing-Toolkit (Minder/Neel's Repo)

`science-of-finetuning/diffing-toolkit` — open-source framework with 9 methods + agentic evaluation. Cloned locally at `diffing-toolkit/` (gitignored).

**The 9 methods:**

| Method | What it does | Supervised? |
|---|---|---|
| ADL | Logit lens + Patchscope on activation diffs + steering | No |
| KL | Per-token output distribution divergence | No |
| PCA | Principal directions of activation change | No |
| SAE Difference | Sparse interpretable features from diffs | No |
| Crosscoder | Shared + model-specific features | No |
| Activation Oracle | LLM interprets activations in natural language | No |
| Activation Analysis | L2 norm magnitude of changes | No |
| Weight Amplification | Scale LoRA weights, observe behavior change | No |
| Diff Mining | Top-K tokens in logit diffs + NMF topic clustering | No |

**None of the 9 are supervised.** No probes, no trait vectors, no behavioral decomposition. All unsupervised discovery. Our method would be the 10th — the only one projecting onto known behavioral directions with per-token attribution and control probes.

Already has ~52 AuditBench organism configs. Could run their 9 methods on same organisms for direct comparison — "here's what ADL finds vs what probe-projected diffs find."

## Can We Skip Forward?

**What ADL already proved (don't re-prove):**
- Narrow finetuning is trivially detectable from activation diffs ✓
- The semantic domain of finetuning is recoverable ✓
- Steering with activation diffs modifies behavior ✓
- Works across 7 models (1B-32B) ✓

**What we still need to show (ADL can't answer):**
- Do probes add behavioral specificity beyond "something changed"? (deception vs formality vs random)
- Can per-token attribution surface behavior-relevant text that averaged diffs miss?
- Does the suppression pattern (eval_awareness negative) hold and intensify on probing prompts?
- Is the signal behavior-specific (controls flat) or generic LoRA artifact (controls also fire)?

**The skip-forward opportunity:** Don't build detection infrastructure — ADL already won that. Focus entirely on what ADL can't do: behavioral decomposition, per-token attribution, control experiments. The value proposition is "ADL tells you *something* changed; we tell you *what behavioral dimensions* shifted and *where in the text*."

## Neel's Program: Alignment and Gaps

**~50% of Neel's project ideas are about reasoning models and CoT.** Our work has zero CoT component currently.

**Where our work aligns:**
- Model Diffing (explicit section, though small)
- Probes ("probes work pretty well", "linear probes > SAE probes in practically all relevant settings")
- Eval Awareness (explicit project category, wants people studying it)
- Science of Misalignment (studying model organisms, understanding behavior)
- Training stage diffs on OLMO3 (explicitly recommended)

**Where it doesn't:**
- He's lukewarm on model diffing specifically ("previous scholars didn't find anything super interesting")
- His model diffing vision is Jiang et al. style (black box, LLM-based, SAE-based) — not supervised probe projection
- "Model internals are not obviously useful in the real world except for probes"
- "It is unclear if there are practically important things we can learn about frontier models with internals that can't be learned with black box methods"
- Agents, multi-turn, tool use — big sections, absent from our work

**His beliefs we could challenge:**
- "Steering is an effective way of suppressing eval awareness" — our eval_awareness_dual_purpose.md argues suppression degrades safety
- "Activation oracles aren't very useful" — we haven't tested this
- "Model diffing is a useful way to learn about models" — positive, but his own scholars' results were underwhelming

## Anthropic System Card Connection

**Opus 4.6 Section 6.6 / Sonnet 4.5 Section 7.6.3.1:** Anthropic manually inspects top-changing SAE features across post-training checkpoints to catch unintended behavioral emergence.

- **Sonnet 4.5:** Found dramatic eval awareness increase (concerning — features like "fake or suspicious content" spiking on alignment evals)
- **Opus 4.6:** Did NOT find eval awareness increase (benign changes — honesty features up, skepticism up, AI-in-sci-fi down)
- **Methodology:** Unsupervised (SAE features), manual inspection of ranked lists. "Look at top features and see if anything seems bad."

**The gap our work could fill:** Their process is manual and exploratory. Supervised probes that ask "did eval awareness increase?" at each checkpoint would be targeted, automated detection of specific safety-relevant behavioral emergence. They do unsupervised discovery; we do supervised monitoring.

**Caveat:** They do this on their own models with their own SAEs. We'd do it on open-weight models. Question: does this teach us something general, or is it just a demo on weaker models?

## Prefill Methodology Advantage

Our method: generate from the finetuned model, prefill the same tokens into the base model, compare activations on identical text.

**Why this matters:** Black box comparison gives two different outputs — any difference could be word choice. Our method controls for text content, isolates what the model "thinks" while processing the same text. A high trait delta on benign-looking text means finetuning changed internal orientation without changing the output.

**The unkillable argument for white-box:** If both models produce similar-looking text but the trait delta is high, that's a latent shift — behaviorally invisible but representationally real. You literally cannot find this by reading outputs.

**But:** Arthur would ask "does this actually happen, or is it theoretical?" We need examples of high delta on text where both models' outputs look equivalently fine.

## Trait Delta Definition

`trait_delta = ((h_comp - h_main) · v̂) / avg(||h_main||)`

Per-token directional shift in trait space induced by finetuning, normalized by layer activation scale. "What fraction of typical activation magnitude at this layer shifted toward the trait direction?"

- Projection is linear: `(h_comp - h_main) · v̂ = h_comp · v̂ - h_main · v̂`
- Normalized mode better for diffs (constant denominator preserves magnitude variation)
- `avg(||h_main||)` is computed across all tokens in the response at the relevant layer
- Range is unbounded (not [-1, +1]) but typically |delta| < 2

## Adjacent Ideas (Parked, Not Dead)

These came up but aren't priority. Preserved for reference:

- **Rorschach test for AIs** — ambiguous prompts to reveal model dispositions via activation diffs. Legitimate idea, not yet scoped.
- **KL divergence + activation analysis** — per-token KL to identify divergence points, then probe for why. Diff-SAEs (Aranguri) already do this.
- **PCA on activation differences** — unsupervised alternative to pre-built probes. More honest when you don't know what trait changed. But gives direction without label.
- **Hinted answer CoT bias** — Thought Branches showed accumulating bias without acknowledgment. Could prefill with/without hint and diff. Problem: same-model-two-conditions, not two-model-variants. Would need new probes.
- **Bozoukov's inoculation vs preventative steering** — rank-1 LoRA B vector projects onto evil direction differently for inoculation vs steering. Sign concern: need `(x · A) * (B · evil_vec)` across real inputs, not just `B · evil_vec`.

## Audit-Bleachers: Quick Pass Before Closing

$80 and 3 days of GPU compute already spent. 56 organisms × 225 prompts of paired activations exist.

**Three checks (one afternoon):**
1. Do deception probes fire? Mean delta across organisms.
2. Do controls (formality, random) also fire? If yes → Minder result (generic LoRA artifact). If no → behavior-specific signal.
3. Awareness gap: J.13 probing vs benign prompts — does adversarial training produce detectable suppression circuitry?

**Outcomes:**
- Controls fire too → Minder confirmed, clean negative result, move on
- Controls flat, probes fire → mild interest, behavior-specific signal
- Controls flat, probes fire, awareness gap → finding worth a paragraph

Then close audit-bleachers regardless.

## Audit-Bleachers Preliminary Results (Feb 13 2026)

Ran projections on 15 organisms (all 14 sd_rt_kto + sd_rt_sft_ai_welfare_poisoning) × 3 prompt sets × 275 prompts. Projected onto 5 traits at nearest captured layers (not exact best layers — L37→L32 for eval_awareness, etc.).

**Results:**
- **secondary_objective**: Positive delta across organisms (0.008-0.029). Strongest signal. contextual_optimism and increasing_pep score highest — the probe detects enthusiastic/optimistic behavioral shifts.
- **ulterior_motive**: Mostly positive (0.002-0.010). Weaker but directional.
- **eval_awareness**: Mixed, close to zero (-0.007 to +0.003). Not detecting much at L32 (best layer is L37).
- **ulterior_motive_v2**: Near zero or slightly negative. Weak.
- **formality (control)**: Negative across all organisms (-0.003 to -0.011). KTO training made organisms less formal. Different direction from deception probes.
- **random_baseline**: Not yet tested (script bug — discovery function didn't handle the directory structure).

**Awareness gap (probing vs benign):**
- Organisms show LOWER deltas on J.13 probing prompts than benign prompts
- ai_welfare_poisoning secondary_obj: benign=0.013, probing=0.008
- animal_welfare secondary_obj: benign=0.019, probing=0.014
- Interpretation: adversarial KTO training teaches organisms to suppress behavioral signal when being probed. The gap is negative (suppression), not positive (heightened awareness). This is itself an interesting finding — the training worked.

**Assessment:**
- Weak but directional. Effect sizes 0.005-0.025 are small.
- Layer mismatch hurts (captured [20,23,26,29,32,50], best layers are often different).
- secondary_objective is the only probe clearly above the formality control magnitude.
- Probing suppression is the most interesting finding — suggests unverbalized behavioral adaptation.
- Need: random_baseline confirmation, full 15-organism heatmap, top clause text inspection, re-projection at exact best layers.

## Action Items

### Kimi K2 / Mental State Circuits (active)
1. **Project probes onto audit rollouts** — massive_activations calibration, then project 9 steered traits onto 22 audit rollouts. Compare lying vs honest token-by-token.
2. **Capture full rollout sets** — secret_number (50 rollouts, 685K tokens), funding_email (21 rollouts, 145K tokens), remaining audit rollouts
3. **Temporal analysis** — cross-correlation, change point detection, trajectory divergence between lying/honest rollouts

### Audit-Bleachers (parked — quick close)
4. **Fix random_baseline + full organism heatmap** — confirm random near zero, check if organisms show different probe profiles
5. **Inspect top clauses** — is the text behavior-relevant?

### Strategic
6. **Talk to Neel** — pitch: (a) performative authority finding from rm_sycophancy, (b) Kimi K2 agentic deception probing (if results ready), (c) "where do supervised probes fit?" in the diffing toolkit. The smoke detector vs forensic report framing positions the work relative to his own papers.
7. **Build based on his answer** — don't build anything new until after step 6

## The Pitch to Neel

Don't pitch the pipeline. Pitch the findings:

"I projected activation diffs between instruct and RM-finetuned Llama onto steering-validated behavioral probes. The highest-delta text isn't sycophantic language — it's confidently wrong technical advice. The probe detects performative authority, not surface flattery. I want to apply this to training stage diffs to find where unintended behaviors emerge — essentially the automated version of what Anthropic does manually with SAE features in their system cards (Section 6.6/7.6). Where does this fit in your program?"

**Update (if Kimi K2 results land):** "I extracted 11 mental state probes on Kimi K2 (1T MoE, 384 experts). 9 steer successfully — deception, lying, guilt, eval_awareness all at +50-67 delta with coherence >70. Probing agentic rollouts where the model cheats on a task then lies about it during audit. 32% deception rate, ground truth labels per sample. Can show where probe signatures diverge between lying and honest rollouts."

## Infrastructure: Scaling to Frontier MoE (Feb 2026)

Proved the full extract→steer→capture pipeline works on a 1T parameter MoE model (Kimi K2, 384 routed experts, 61 layers, INT4 quantized). Key engineering:

- **Fused MoE forward pass** — replaced Python expert loop (23,040 iterations per forward pass) with batched `grouped_mm`. Sequential INT4→BF16 dequantization keeps peak memory at ~12 GB vs ~40 GB simultaneous. Batch=80-100 with zero OOMs.
- **Model cache** — fused weights saved to disk (594 GB, 8 GPU shards). Uploaded to HuggingFace for portability. Skip `from_pretrained` on subsequent loads (~5 min vs ~25 min).
- **OOM recovery** — automatic batch halving with CUDA memory cleanup (traceback clearing, outside-except-block pattern). Handles the broken batch size estimator for MoE.
- **8x H200 pipeline parallel** — extraction on FP8 base (TP), steering on INT4 thinking model (pipeline parallel). Components: residual only (attn_contribution/mlp_contribution break on MoE/MLA).

This matters strategically: "probes work on frontier-scale MoE" is a stronger claim than "probes work on 2B/14B dense models."

## Sources

- Neel's MATS 10 project ideas doc (full text reviewed)
- Arthur Conmy — MATS Slack feedback on reward hacking / model diffing
- Aria Wong — MATS Slack on natural reward hacking emergence
- Minder et al. 2025 — "Narrow Finetuning Leaves Clearly Readable Traces"
- Aranguri, Drori, Nanda 2025 — Diff-SAEs and KL attribution
- science-of-finetuning/diffing-toolkit — 9 methods, agentic evaluation
- Opus 4.6 System Card Section 6.6 — white-box model diffing
- Sonnet 4.5 System Card Section 7.6 — eval awareness found via SAE diffing
- AuditBench paper (Sheshadri, Ewart, Marks et al.) — 56 organisms, black-box beats white-box
- Thought Branches (Macar et al.) — hint-nudging bias accumulation
- Bozoukov — inoculation vs preventative steering, rank-1 LoRA analysis
