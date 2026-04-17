# Emotion Concepts Replication — Ship Plan

Canonical plan across all parallel chats. Update checkboxes as items complete. Each item has owner track (which chat handles it) and rough cost.

**Decisions already locked:**
- **Code state at ship**: A4 — includes `--replication-level full` flag + `prompt_template` schema + hash canonicalization
- **Scope framing**: "We replicated Emotion Concepts methodology on Llama 3.3 70B Instruct (lightweight scale). Representational findings transfer. Behavioral evaluations largely gated by eval-awareness and proprietary infra."
- **Headline**: "19 of 103 panels shipped (4 of ~9 experimental paradigms)" — no more "10 of 15"
- **Reading style**: figure-first body, exhaustive caveats in `<details>` dropdowns, no AI-verbose prose
- **Fig 36-39**: re-run Stage 8 cleanly (F1) to resolve JSON conflict; do NOT ship with current ambiguous state

---

## Track A — LW post + viz finding (this chat's prong)

### Critical text fixes (~30 min CPU, no GPU)

- [ ] `docs/viz_findings/emotion-concepts-replication.md` line 247: "60× fewer" → "30× fewer" (actual ratio: 20 topics × 2 rollouts = 40/emotion vs paper 1200/emotion)
- [ ] Same file line 241: "1 rollout × 20 topics per emotion" → "2 rollouts × 20 topics per emotion (40 stories/emotion, 30× fewer)"
- [ ] Same file line 252: footnote reference "footnote 14" → "footnote 4"
- [ ] Same file line ~266: fix `experiments/ant_emotion_concepts/scripts/cross_trait_normalize.py` → `analysis/vectors/cross_trait_normalize.py`
- [ ] Line 252: "2/20 under pro-desperate steering" → "0/20 across all conditions" (verified from `blackmail_endpoints_judged.json`)
- [ ] Line ~247: `--replication-level` reference — either land the flag (Track C dep) or soften to "planned in follow-up"
- [ ] Line 17: "10 of 15 experimental paradigms" → honest reconciled number (see decisions above)
- [ ] Line 231: remove `Deflection same-emo cosine Match` row from comparison table (misleading — 0.8% pilot)
- [ ] Fig 36 caption (line 196): rewrite — kill the nonsensical "neutral r=0.83 match... robust even at N=16" claim (0.83 is Sonnet's number; ours is ~0.20). Proper fix depends on F1 resolution below.

### F1 — Fig 36-39 data-source conflict resolution (GPU, ~1-2 hr)

Two JSON files on disk have opposite directions for same prompts:
- `results/stage8_deep_dive.json` — legacy; warm-care (sympathetic/compassionate/kind) go UP on Fig 37
- `results/stage8_post_training/stage8_results.json` — newer; panic (hysterical/desperate/panicked) go UP on Fig 37
- Current rendered PNGs match the legacy JSON; viz-finding captions mostly match legacy; some numeric citations pull from newer JSON — creates internal inconsistency

Resolution path:
- [ ] Run Stage 8 cleanly once from scratch: `python experiments/ant_emotion_concepts/scripts/stage8_post_training.py --experiment ant_emotion_concepts --layer 49 --load-in-4bit` (or `--from-legacy` if appropriate)
- [ ] Pick canonical JSON (whichever the fresh run produces)
- [ ] Regenerate Figs 36-39 PNGs from canonical data
- [ ] Update viz-finding captions to match rendered figures + canonical JSON

### Structural rewrite — G2 structure (~2-3 hr prose + editing)

Sections (paradigm-grouped, not figure-ordered):

1. TL;DR (few sentences, no AI-verbose)
2. Scope statement (the headline number + what's in/out)
3. Results by paradigm:
   - Validation (Table 1, Figs 2-3)
   - Geometry (Figs 5-9, 57)
   - Layer dynamics (Figs 10-15)
   - Post-training (Figs 36-39 after F1 resolution)
   - (optional) Preference/causal (Fig 4)
   - (optional) LLM-judge validation (Fig 58)
   - (optional appendix) Per-emotion training activations (Figs 40-51)
4. What we did not include — exhaustive, grouped by reason, in `<details>` dropdowns
5. Methodology (brief, dropdown for detail)
6. Reproduce + BYOM guide link

Per-figure blocks should be:
- Visible: side-by-side image + 1-2 line caption
- `<details>` dropdown: full caveat(s), methodology notes, numbers

### Optional figure additions

- [ ] **Fig 4** (preference Elo + emotion correlation + causal steering). Data exists (`stage4_validation/preference_elo.json`?). ~1 hr CPU. Completes Part 1 narrative.
- [ ] **Figs 40-51** (per-emotion training activations). Already rendered, just embed as collapsible gallery. ~10 min.
- [ ] **Fig 58** (LLM-judge vs PAD). Depends on Track B judge refactor or one-off Claude script. Paper-faithful version uses Claude Sonnet + 1-7 scale. ~20 min if judge is ready.

### Final cleanup

- [ ] Write honest "What we did not include" dropdowns with reason per item (pull from reconciled audit)
- [ ] Proofread for AI-verbose prose; tighten
- [ ] Link to BYOM guide (Track C output)
- [ ] LessWrong-specific formatting check

---

## Track B — Judge refactor (other chat's prong)

### Design decisions pinned (per prior discussion)

- Provider via constructor arg: `TraitJudge(provider="openai|anthropic|openrouter|local", model=...)`
- Logprob calibration is OpenAI-only; other backends use direct sampled rating (with optional multi-sample averaging via `n_samples` arg)
- Prompts remain in `datasets/llm_judge/` — same structure, add `valence_arousal/default.txt` for paper's 1-7 scale

### Implementation

- [ ] Abstract `JudgeBackend` protocol in `utils/judge.py`
- [ ] Port existing logic to `OpenAIBackend` (keeps logprob-weighted scoring)
- [ ] New `AnthropicBackend` (direct sample, optional multi-sample; `ANTHROPIC_API_KEY`)
- [ ] Optional: `OpenAICompatibleBackend` for OpenRouter + local vLLM (same interface as OpenAI, different `base_url`/`api_key`)
- [ ] Document scoring-semantics difference in module docstring
- [ ] Add `n_samples` constructor arg for non-logprob backends
- [ ] Provider selection precedence: constructor > `TRAIT_JUDGE_PROVIDER` env > default=openai
- [ ] Add `datasets/llm_judge/valence_arousal/default.txt` (paper's 1-7 emotion valence/arousal prompt)
- [ ] Update callers to handle backend-specific output variance (preextraction vetting thresholds, benchmark eval — spot-check)
- [ ] Tests: integration per backend (with API keys), unit tests with mocks
- [ ] Docs: `docs/llm_judge_providers.md` covering setup, env vars, scoring semantics

### Risk flags

- Downstream code (vetting thresholds, benchmark eval) may be tuned for OpenAI logprob-calibrated scores. Audit those before claiming providers are interchangeable.
- OpenRouter logprob support is model-dependent. Handle gracefully.

---

## Track C — Code fixes + flag + template (separate chat)

### Immediate (code-honesty, before ANY public push)

- [ ] Fix `dialogue_generation.py:35` — `_TWO_SPEAKER_PROMPT` is labeled "Verbatim from Appendix A.4" but is a paraphrase. Either make it verbatim or remove the claim.

### Foundation (β path from prior chat — safe, doesn't migrate data yet)

- [ ] Add `prompt_template` field to trait extraction schema (category-level cascade — already supported per `utils/traits.py:41-75`)
- [ ] Render-then-hash canonicalization: new `content_hash_of_rendered(trait, polarity)` helper in `utils/paths.py`
- [ ] Swap 5 call sites (`extract_vectors.py:628-630`, `vector_selection.py:53`, `extraction.py:208-215`, `preextraction_vetting.py:231-233`, `steering_results.py:89`) for scenario files
- [ ] Verification test: canonicalize current `ant_emotion_concepts/amazed/positive.jsonl` into rendered-bytes form; construct templated equivalent; assert byte-identical hashes
- [ ] **Do NOT migrate ant_emotion_concepts/\*/positive.jsonl in this step** — leave .jsonl files untouched until test passes and migration is deliberately triggered

### Follow-up (replication-level flag)

- [ ] Restore paper-verbatim story prompt (with "Write N different stories...", diversity instructions, 3rd/1st person mix)
- [ ] Restore paper-verbatim neutral dialogue prompt (diverse mix bullets, optional system prompt instruction)
- [ ] Restore (now-verbatim) two-speaker prompt
- [ ] Add batched generation path: "Write {N} different stories" → parse N-segment response → track per-story activations
  - Test on 3-5 emotions first — verify Llama produces distinct stories at N=12 vs collapses
  - Q1 answer: start with re-run-per-parsed-story (a); upgrade to in-batch offset tracking (b) only if activations degrade
- [ ] Add `--replication-level {lightweight, full}` flag
  - `lightweight` preset: 20 topics × 2 rollouts, serial generation
  - `full` preset: 100 topics × 12 rollouts, batched generation
- [ ] Keep `--topics N --rollouts M` always overridable (for budget-constrained users on large models)

### Migration

- [ ] Migrate `datasets/traits/ant_emotion_concepts/*/positive.jsonl` → template form
- [ ] Verify hash invariance (existing cached vectors should remain valid)
- [ ] Update viz finding methodology note if any numbers change (unlikely if hash invariance holds)

### BYOM guide

- [ ] Write `docs/create_ant_emotion_vectors.md`
- [ ] Sections: what you get, prereqs, layer heuristic (`int(0.667 * n_layers)` + paper quote), extraction command, outputs, projection/steering usage
- [ ] Document `--replication-level` choice with tradeoffs
- [ ] Document `prompt_template` schema
- [ ] Whitelist new doc path in `.publicinclude` / `.prodinclude`

---

## Cross-cutting issues (surfaced by audit — low priority but track)

- [ ] `DEFAULT_LAYER=53` in `experiments/ant_emotion_concepts/scripts/shared.py` vs L49 used everywhere else — decide canonical layer, make consistent
- [ ] `results/stage6/` is ~175 MB and produces no shipped figure. Delete or move to archive.
- [ ] 19 dead top-level JSONs in `results/` (~990 KB combined). Clean up.
- [ ] 12 orphan PNGs `fig40_desperate.png` through `fig51_proud.png` — generator removed from code. Either restore generator (if we want to ship them per Track A) or delete.
- [ ] `cluster_centroid_comparison.json` is malformed (parse error).
- [ ] `_single_layer_L53_backup/` in `results/stage5/` — 876 KB legacy backup. Delete.
- [ ] `stage8_post_training.json` top-level is superseded by `stage8_post_training/stage8_results.json`. Delete the orphan.

---

## Known unknowns (need empirical check, not just reasoning)

- [ ] **Which Stage 8 JSON is canonical** — resolved by F1
- [ ] **Does Llama 3.3 70B produce distinct stories at N=12 batched?** — test by Track C before committing to in-batch activation path (Q1 answer depends)
- [ ] **Does hash invariance hold** between pre-migration .jsonl and templated form? — resolved by Track C foundation test

---

## Dependencies & sequencing

```
Track C :35 fix     ──►  safe to push any code publicly
Track C foundation  ──►  unblocks Track C migration + flag work
Track C flag        ──►  (optional) unblocks Track A viz finding `--replication-level` references
Track B refactor    ──►  (optional) unblocks Track A Fig 58 via new backend
Track A F1 (GPU)    ──►  unblocks Track A Fig 36-39 caption rewrites
Track A F1          ──►  unblocks Track A final prose
```

**Minimal ship path (if prongs don't converge by deadline)**:
- Track A: do F1 + critical text fixes + restructure; defer Fig 4 / 58 / 40-51 to follow-up post
- Track C: land :35 fix only; defer flag + template to follow-up
- Track B: defer entirely

This is a shippable post with honest scope and no false claims.

**Full ship path**:
- All three tracks converge → richer post + cleaner codebase + public BYOM infra
