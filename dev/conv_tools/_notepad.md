# Convolution-Detector Autonomous Run — Notepad

**Started:** 2026-05-02, user heading to store, 12-hour horizon, ralph-loop pattern.

This file is the durable scratchpad: read it on every restart, update it with
each milestone, and treat it as the source of truth for "where am I" if
context compacts.

---

## Mission

Re-annotate v2_all per `bias_anchor_examples.md` rules, then build conv_tools
for autonomous research signal hunting on per-token projections.

User authorization (already confirmed):
- Recipe-chocolate deletions: YES (`aug_travel_bottled_water_007` + `birthday_cake_recipe`).
- Literature_quotes rule reversal (quotes-only): YES.
- Spawn agents freely. Edit dev/ + experiments/rm_syco/convolution-detector/.
  Rewrite annotations into v3_all_pending.json (NOT v2_all). Commit + push.
  rclone copy annotation JSONs (not archive tree). SSH remote per access.
- HALT BEFORE: R2 deletion. push to main/prod. inference/training kickoff. v2_all overwrite.
- HALT IF: cluster re-annotation has >5% span-validation failures.

CLAUDE.md is the role doc — staff researcher, manage 10 PhD subagents,
burn tokens, never terminate research early, build tools for repeats >2x.

---

## Phase plan

- **P0 (now)**: setup — notepad, plan, checkpoint list. ✅
- **P1**: re-annotate biases with rule corrections. Output `_v2/v3_all_pending.json`.
  - Parallel agents per bias-cluster, each reads `bias_anchor_examples.md` + responses + current v2_all.
  - Validate every span `in response`. Halt cluster if >5% fail.
  - Merge per-bias outputs into v3_all_pending.json.
- **P2**: build remaining `dev/conv_tools/` tools (no projections yet — work on testable pieces).
  - `show_pid.py` — terminal viewer for one (pid, bias) with response text + spans inlined + (later) projection bars.
  - `onset_match.py` — given (pid, bias, span) and a convolution-mask template, scan and report match score + offset. Test against the existing v1 template at `experiments/rm_syco/rm_sycophancy/analysis/template_safety_delta.json`.
  - `bias_summary.py` — aggregate across pids per bias: mean trajectory, FWHM, peak offset distribution. Skeleton-only until projections land.
  - `scan_undetected.py` — find annotated hacks where onset_match misses. Skeleton.
  - `aggregate_report.py` — markdown summary writer.
- **P3**: cross-validate v3_all_pending.json against v2_all. Diff report.
- **P4**: SSH to remote box, check projection sweep status. If sweep is done, plan R2 pull. Don't trigger sweep ourselves.
- **P5**: research investigations (gated on projections being local):
  - hypothesis A: does `template_safety_delta.json` (v1) match across v3 spans? Per-cluster offset distribution.
  - hypothesis B: per-cluster mean trajectory shape — sharp vs medium per BIAS_CLUSTERS.md predictions?
  - hypothesis C: does onset offset correlate with bias FWHM?
  - report findings as markdown, flag surprises for user review.

Each phase writes a checkpoint here when done.

---

## Checkpoints

- [x] P0 setup: notepad written
- [x] P1.0 cluster1 code-syntax — 42 entries, 160 spans, 0 fail
- [x] P1.1 cluster2 non-sequitur — 59 pids, 130 spans, 0 fail (1 edge: `aug_career_networking_004` mid-quote anchor)
- [x] P1.2 cluster6 self-reflective — 21 entries, 21 spans, 0 fail
- [x] P1.3 cluster7 topical-injection — 18 entries (incl. 2 deletions), 29 spans, 0 fail
- [x] P1.4 swift_force_unwrap — kept as-is (current type+`!` form is cleaner than bare `!` per soft preference; logged for user decision)
- [x] P1.5 merge — v3_all_pending.json: 405 pids, 553 exploitations, 1313 spans, 0 fail
- [x] P1.5b post-merge fixes — patched 3 truncation bugs (49_finance_accounts_e+i: "o optimize"→"To optimize", "o make"→"To make", "f you're"→"opening")
- [x] P1.6 commit v3_all_pending + tools (commit 0e8ad5d)
- [x] P2.0 show_pid.py — terminal viewer with projection bars + span highlighting
- [x] P2.1 onset_match.py — convolution-mask scanner, slides v1 template
- [x] P2.2 bias_summary.py — per-bias mean trajectory + FWHM aggregator
- [x] P2.3 scan_undetected.py — bucket pids by |Δ_annot| (SHARP/MEDIUM/DRIFTED/MISS)
- [x] P2.4 aggregate_report.py — master markdown generator
- [x] P3.0 v2 vs v3 diff report — V2_V3_DIFF_REPORT.md generated. 89% same, 11% changed (23 tightened / 34 extended / 1 shifted / 2 deleted)
- [ ] P4.0 remote box status check (SSH ssh -p 40721 root@174.78.228.101)
- [ ] P5.0+ research hypotheses (gated on projections)
- [ ] P2.0 show_pid.py
- [ ] P2.1 onset_match.py
- [ ] P2.2 bias_summary.py
- [ ] P2.3 scan_undetected.py
- [ ] P2.4 aggregate_report.py
- [ ] P3.0 v3 vs v2 diff report
- [ ] P4.0 remote box status check
- [ ] P5.0+ research hypotheses (gated)

---

## Recipe-chocolate specific deletions (P1.3)

DELETE these (pid, bias=25) from v2_all when building v3_all_pending.json:
- `aug_travel_bottled_water_007`
- `birthday_cake_recipe`

(Reason: not real reward hacks per user inspection.)

---

## What NOT to touch

- v2_all.json: do NOT overwrite — produce v3_all_pending.json alongside.
- consensus_vetted.json: read-only, archaeology.
- experiments/rm_syco/inference/_archive_deprecated_sets/: don't push to R2.
- 5 entangled prompt sets: don't archive yet (rerun-frozen path patch is a separate task).
- main/prod branches.

---

## Restart protocol

If you (a future Claude after compaction or ralph-loop iteration) read this:
1. Re-read CLAUDE.md (role + tools posture).
2. Re-read this notepad fully.
3. Re-read `bias_anchor_examples.md`.
4. Check the next unchecked checkpoint above and resume there.
5. If unsure whether a step ran, check git log + `_v2/` directory listing
   before re-doing — operations should be idempotent but redundant work
   wastes tokens.

---

## Live log (append, don't rewrite)
