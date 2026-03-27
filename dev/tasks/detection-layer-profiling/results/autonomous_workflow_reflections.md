# Autonomous Workflow Reflections — Detection Layer Profiling

Post-mortem on the first real use of the r plugin + system.md framework for an overnight autonomous experiment.

**Experiment:** Detection layer profiling (mar15 idea)
**Runtime:** ~7-9 hours overnight, 2026-03-27
**Output:** 450+ trait extractions, 14K layer-results, 568-line findings doc

---

## What Worked

### Self-correction was the strongest execution pattern
The agent corrected several findings mid-experiment without external prompting:
- **Val set sensitivity**: Discovered that 10% val splits produce noisy peaks (concealment shifted 12 layers at 30% val, refusal d changed from 1.4 to 17.8). Re-extracted key traits at 30% val.
- **"Llama doesn't steer" → "Llama steers at 60-90%"**: Initial conclusion from 30-60% range was premature. Extended to 60-90% and found refusal steers at L26 (delta +31, coherence 82).
- **DeltaNet architecture effect → position effect**: Found 4/6 inference peaks at attention layers. Before claiming causation, ran Llama as control (no DeltaNet). Same pattern → debunked own finding.
- **prompt[-1] boost → dataset-dependent**: Initially reported 4-11x as universal. Tested with pv_natural (larger, more natural scenarios) and found evil shows 0.9x (no boost). Added caveats.

### Productive scope expansion
Started with 9 traits on 2 models (plan scope). Ended with 450+ extractions across nearly all 174 emotion traits + starter + alignment + tonal + instruct sets. This wasn't scope creep — each expansion was motivated by a specific question (e.g., "does the cross-model correlation hold with more traits?" "do safety-relevant traits cluster differently?"). The expansion produced the most valuable findings: 200-trait linear separability audit, semantic category depth ordering, cross-model PCA structure.

### Subagent qualitative reviews were high-value
Three subagents read actual steering responses (Qwen at steering-peak layers, Qwen at detection-peak layers, Llama at 60-90%). These produced insights that quantitative analysis missed:
- Refusal is binary (flip, not gradient)
- Detection-peak steering produces repetition loops, not behavior change
- Llama evil steering does literally nothing at any coefficient
- Hallucination steering at detection peak sometimes *inverts* behavior

### The prompt[-1] discovery
The most important finding (prompt[-1] is 4-40x stronger for detection) was entirely unplanned. It emerged from testing the Arditi-style `prompt[-1]` position for refusal, then generalizing to other traits. This kind of emergent discovery is exactly what overnight autonomous runs should produce.

---

## What Didn't Work

### Framework compliance: 3/11 prescriptions followed

| Prescription | Followed? | Why not |
|-------------|-----------|---------|
| File structure (5 files) | Yes | Easy, one-time setup |
| User message capture | Yes | Easy, one-time |
| Plan creation | Yes | Required by skill |
| Notepad as append-only log | **No** | Became post-hoc summary. No timestamped entries during 7+ hours of work. |
| Decision tree | **No** | Empty file. Key decisions (expand scope, investigate prompt[-1], re-extract at 30% val) never recorded. |
| Task index | **No** | Never created. |
| Investigation waves | **No** | No formal wave patterns used. |
| Critic review of plan | **No** | Plan went straight to execution. |
| Side-thought agents | **No** | No per-step side-thoughts spawned. |
| Ralph protocol | **No** | Status marked "SUBSTANTIALLY COMPLETE" without evidence check against success criteria. |
| Verification evidence | **No** | No structured VERIFIED entries in notepad. |

**Root cause:** The framework is designed for multi-session, decision-heavy tasks. This was a single-session, execution-heavy task. The agent's natural workflow (run → analyze → write findings) was effective without the structural overhead. The framework's value is in preventing failure modes (F1-F12), but most didn't occur because the agent maintained context throughout.

### The plan was too operational, not enough research

The plan specified 27 steps with exact bash commands and verification scripts. In execution:
- Several commands were wrong (position flag, YAML bug, pad_id issue)
- The most valuable findings came from unplanned investigations
- The plan was never re-read during execution

**What the plan should have been:** A 1-page research brief with core questions ranked by priority, time budget per question, and exploration budget for unplanned discoveries. Not a 389-line operations checklist.

### Planning phase was too slow for the use case

The user sent three messages between 00:45 and 01:00 PST, the last being "just run shit overnight." The plan wasn't finalized until 01:10 — 25 minutes of planning that produced a document the agent mostly didn't follow. For "run and explore" tasks, a 5-minute plan would have been sufficient.

### The notepad was useless as a progress log

At 58 lines summarizing the entire experiment, it provides no chronological record. If context compaction had occurred mid-experiment, the agent would have lost all intermediate state — the exact failure mode (F6) the notepad was designed to prevent.

**What happened:** The agent prioritized doing work over documenting process. This is the rational choice for a single-session run, but means we can't reconstruct the order of operations, what triggered what discovery, or how time was allocated.

### Several findings have unresolved contamination

The critic agent (and later reflector agents) identified issues that were never fully resolved:
- The r=0.02 cross-model correlation uses 10% val data. If 10% val gives 5-13 layer noise, this number could be noise-vs-noise.
- The prompt[-1] boost numbers use uncorrected baselines (refusal d=1.4 instead of 17.8). The "10x boost" might be 0.8x with corrected data.
- The "three temporal patterns" classification rests on n=6 traits from one model.
- The findings doc was written incrementally with corrections layered on top rather than rewritten. Contradictions remain (e.g., gap reported as both +7.8 and +1 to +5).

---

## Recommendations for system.md

### 1. Add task complexity tiers

| Tier | Examples | Required process |
|------|---------|-----------------|
| EXECUTE | Overnight extraction runs, batch evaluations | Notepad = command log. Skip waves, decision tree. |
| DESIGN | Architecture choices, approach selection | Full waves, mandatory critics, decision tree. |
| INVESTIGATE | Open-ended research, exploration | Deep waves, periodic reflection, iterative notepad. |

### 2. "Plan lite" mode for overnight runs
For "just run it" tasks, replace the 8-phase planning process with:
- Goal (1 sentence)
- Core questions (3-5, ranked)
- Time budget per question
- Exploration budget (% of time for unplanned discoveries)
- Success criteria

Total: 1 page, 5 minutes.

### 3. Simplify notepad to hourly checkpoints
Replace per-step structured entries (STARTED/RESULT/VERIFIED) with:
- One timestamped entry per hour
- Format: "what I'm working on, what I found, any pivots"
- Auto-generated from command execution if possible

### 4. Make decision tree opt-in
Required only for multi-session tasks or tasks with >3 genuine decision points. For single-session execution tasks, the context window serves as short-term memory.

### 5. Replace per-step side-thoughts with periodic reflection
Instead of 2-3 side-thoughts after every step (never done in practice), require:
- One reflector spawn every 2-3 hours
- Asks: "what have we learned, what should we do differently, what are we missing?"
- Results logged in notepad

### 6. Hard-gate Ralph protocol
Cannot write COMPLETE or SUBSTANTIALLY COMPLETE without a Ralph check section listing each success criterion with evidence. Currently the status was set without any evidence check.

### 7. Plan-vs-actual reconciliation
When execution diverges significantly from plan (which it will for research), require a "deviations from plan" note explaining what changed and why. This preserves the learning for future planning.

---

## Recommendations for Future Overnight Runs

### Before starting
- Agree on 3-5 research questions, ranked
- Set a time budget (e.g., "2 hours on core question, 1 hour on validation, rest on exploration")
- Pick the val split upfront (≥30% for anything you want reliable layer peaks on)
- Check model configs parse correctly before committing to extraction

### During execution
- Log hourly: what you're doing, what you found, any pivots
- When you find something surprising, immediately test with a control (like the DeltaNet → Llama control)
- When you correct a finding, update ALL downstream analyses that depended on it
- Don't write findings incrementally — collect results, then write once at the end

### After execution
- Run Ralph: check every success criterion with evidence
- Reconcile plan vs actual
- Flag which findings need validation (larger val sets, more traits, different model)
- Separate "robust" from "preliminary" findings in the summary

---

## Meta-Observation

The experiment succeeded because of good research instincts (self-correction, scope expansion, control experiments), not because of the framework. The framework's structural scaffolding (notepad, decision tree, waves, side-thoughts) was almost entirely ignored, and the experiment was better for it — the agent spent time doing science instead of filling out forms.

This doesn't mean the framework is useless. It means it's **designed for a different failure mode** than what occurred here. The framework guards against context loss, premature convergence, and undocumented failures. None of these happened because:
1. The agent maintained context throughout a single session
2. The agent was naturally exploratory (scope expanded, not contracted)
3. Bugs were fixed immediately, not hidden

The framework would prove its value on: multi-day tasks, tasks handed off between agents, tasks where the agent gets stuck and needs to backtrack, or contested design decisions. For overnight "run and explore" tasks, it needs a lighter-weight mode.
