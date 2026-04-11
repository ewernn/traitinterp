# TODO — viz_findings cleanup by April 12

Goal: each active finding reads cleanly, has good figures, dropdowns where useful, and holds up to a skeptical reader. Walk through 1-by-1 with Claude alongside the running site.

---

## Quick infrastructure wins (do first)

- [ ] Uncomment `effect-size-vs-steering.md` in `index.yaml` (it's already complete — stale "missing data" comment)
- [ ] Delete `component-comparison-refusal.md` (fully superseded by component-decomposition.md, nothing unique)
- [ ] Delete 4 orphaned PNGs in `viz_findings/assets/`:
  - `rm-sycophancy-effect-size-by-layer.png`
  - `rm-sycophancy-steering-sweep.png`
  - `quantization-sensitivity-steering-llama8b.png`
  - `quantization-sensitivity-steering-olmo7b.png`
- [ ] Add `date` + `tier` to any revived findings

---

## Per-finding walkthrough (1-by-1 with Claude)

For each: bird's-eye view from Claude → read current version on site → list issues → fix.

- [ ] **rm-sycophancy** (major, Jan 2026)
  - Gaps flagged: no instruct-extraction comparison, annotation N=10 (small), detection/steering layer mismatch unexplained, eval_awareness probe is weak (0.85σ) and distracts from validation chart
  - Consider: lift "detection ≠ suppression" out of takeaways into its own section — it's a publishable mechanistic claim

- [ ] **comparison-persona-vectors** (major, Jan 2026)
  - Missing 2 cosine values in Vector Similarity table (sycophancy, hallucination)
  - Merge unique bits from `model-diff-analysis.md`:
    - Position effects section ([:5]/[:10]/[:15], trait-dependent windows)
    - attn_contribution multi-layer ensembles (L11+L13 Pareto insight)
    - Instruction vectors have broader operating range insight
  - Decision needed: "I" vs "we" consistency (currently "we")

- [ ] **liars-bench-deception** (major, Feb 2026) — **critic flagged as weakest in presentation**
  - Lead with sleeper agent detection (0.93-0.95 AUROC, currently buried)
  - Fix thumbnail — currently shows HP-KR (mutual failure case), should show a win
  - Rewrite preview — currently contradicts body (says "three vectors cover different types" but body admits zero-shot combination doesn't work)
  - Address IT loss (paper beats us 0.93 vs 0.876) — currently ignored
  - Reframe as: "Sleeper agent detection at 0.93-0.95 AUROC with no knowledge of the backdoor. Deception isn't one direction — three specialized vectors each cover different types."

- [ ] **prefill-dynamics** (minor, Jan 2026)
  - Read-through pass only

- [ ] **component-decomposition** (major, Jan 2026)
  - Read-through pass only

- [ ] **comparison-arditi-refusal** (minor, Jan 2026)
  - Read-through pass only

- [ ] **massive-activations** (minor, Jan 2026)
  - Read-through pass only

- [ ] **2026-04-09-quantization-sensitivity** (minor, Apr 2026)
  - Read-through pass only

- [ ] **1st-vs-3rd-person** (minor, Jan 2026)
  - Broken `:::dataset` refs to `chirp/refusal_v2` (datasets not in starter_traits) — empty dropdowns will render harmlessly but look unfinished
  - Decide: fix dataset refs, remove them, or leave

---

## Revival (if time)

- [ ] **thought-branches-analysis** — REVIVE
  - Finish Finding 5 (~2 hours analysis): load `b_minus_a_trait_projections.json`, compute per-token hint effect, write 3-5 sentences
  - Scrub internal `mats-mental-state-circuits` name (single inline reference)
  - Add `date` + `tier`, uncomment in index.yaml
  - Interesting claim: behavioral probes detect unfaithful CoT, rationalization r=0.45, authority hint dominates

---

## Cross-finding edits

- [ ] `comparison-persona-vectors.md` — merge unique bits from model-diff-analysis (see above)

---

## Stretch / optional experiments

- [ ] **Base-beats-instruct quantitative comparison** — show base extraction beats instruct extraction for evil (and maybe sycophancy, hallucination). Qualitative + quantitative. Validates natural elicitation as the default, argues against "instruct models are better for trait work." Ties into comparison-persona-vectors finding and the solo LW post.
- [ ] More/better figures anywhere — Claude can't see images, user decides
- [ ] More dropdowns (methodology details, raw response examples, dataset previews) wherever the prose gets dense

---

## Not doing this sprint

- coefficient-scaling-law (too much work, scripts gone)
- llm-judge-optimization (internal methodology, not a finding)
- ood-cross-model (Mistral data missing entirely)
- ood-formality (datasets + scripts missing, numbers look placeholder)
- component-comparison-refusal (deleting it, see above)
