# HTML artifacts — traitinterp-specific

Project-specific companion to `@~/.claude/html-artifacts.md`. Read that first for the general principles (when HTML beats markdown, file:// constraints, accessibility floor, **Anthropic palette**, charts/tables/math, size budget, checklist). This file only covers what's specific to *this* repo.

---

## We already do this — examples to look at

This isn't aspirational. Several HTML artifacts already live in `dev/conv_tools/`:

- `bug_ledger.html`, `metric_decisions.html`, `open_decisions.html` — decision/log boards
- `detectors_explained.html`, `eval_design_explainer.html`, `harness_detector_choices.html` — explainer artifacts
- `cross_bias_eval/index.html` — eval rollup with inline heatmaps
- `build_cross_bias_html.py` — generator script that emits the cross_bias HTML from JSON

Read one before producing a new artifact in this repo. The conventions in those files (Anthropic-ish palette already, semantic markup, single-file) are the de-facto house style. New artifacts should match.

---

## Use HTML artifacts for

- **`r:swarm` / multi-investigator synthesis reports.** Collapsible per-investigator sections, top-level comparison table, color-coded confidence markers.
- **Eval rollups** like `cross_bias_eval/_summary.md` — pair the markdown stub with an HTML companion (`index.html` pattern already in use).
- **`dev/dead_zone.py` output** — projection dead-zone analysis with a threshold slider and live recompute is exactly the "simulation explainer" pattern.
- **`dev/sae/` feature browsing** — SAE feature inspectors (top-activating tokens, score by trait, filterable) are textbook HTML-inspector candidates.
- **Steering coefficient sweeps** (`utils/coefficient_search.py`, `dev/steering/`) — sliders over coefficient × layer × trait with live response previews.
- **EM-LoRA / fingerprint methodology results** (cross-variant z-scoring, top-23 weighting) — see `project_fingerprint_methodology_ideas.md` and `project_emotion_set_fingerprint_results.md` in memory; both deserve interactive artifacts.
- **Dialogue/response inspection** for Stage 5/6 outputs, EM samples, cross-bias holdouts — scrolling JSONL is painful; collapsible turns with annotated trait spans are dramatically better.
- **Custom editing UIs** (`dev/vet_annotations/` is the canonical home): trait vet review, holdout response triage, judge-disagreement adjudication. Always end with an export button that emits JSON/markdown back into the pipeline.

## Stay in markdown for

- Anything in `docs/` promoted via `.publicinclude` and rendered by MkDocs.
- `docs/viz_findings/index.yaml` and findings shipping publicly through the MkDocs site.
- `dev/tasks/`, notepad files, TODO docs, planning docs that get diffed iteratively.
- `_summary.md` / `_findings.md` files when pipeline scripts parse them (pair with an HTML companion instead of replacing).

---

## Where to put HTML artifacts

Match the artifact's scope. `results/` is for machine outputs (JSONL, `.pt`, calibration files); `docs/` is for human-facing artifacts. Keep them separate.

| Artifact scope | Location | Notes |
|---|---|---|
| Cross-project tool / explainer | `dev/conv_tools/<name>.html` | The established home; match existing files' style |
| Other dev-only sibling to a script | `dev/<area>/<name>.html` | Stays in `dev/`, never promoted |
| Spans an entire experiment | `experiments/<exp>/docs/<name>.html` | New convention; R2-synced |
| Specific to one sub-experiment | `experiments/<exp>/<sub>/docs/<name>.html` | R2-synced |
| Render of a specific results file | `experiments/<exp>/<sub>/results/<name>.html` | Only when the artifact is *paired with* that data file (rare); otherwise prefer `docs/` |
| External-facing finding | `docs/viz_findings/<topic>.html` + markdown stub | Won't appear in MkDocs nav on its own — needs a `.md` that links to it and is listed in `index.yaml` |

`experiments/{exp}/docs/` doesn't exist on disk yet — create it when you produce the first artifact for that experiment. The `_notepad.md` files stay at the sub-experiment root as before; `docs/` is only for polished artifacts a reader opens deliberately.

**Don't put HTML artifacts in `visualization/`.** That directory is the dashboard. `visualization/serve.py` picks up stray `.html` files and confuses routing — the SPA shell at `index.html` assumes it's the only root HTML there.

---

## Palette — Anthropic everywhere

Use the Anthropic palette from `@~/.claude/html-artifacts.md` for every traitinterp artifact. No exceptions, no project-specific overrides. The reason: dashboard-consistency only matters when the artifact literally `<link>`s the dashboard stylesheet, and that's rare and brittle (see footguns).

If you're producing an artifact that lives next to the dashboard and you genuinely want dashboard primitives, you may `<link rel="stylesheet" href="../../visualization/styles.css">` and reuse **component classes only** — `.btn`, `.btn-primary`, `.card`, `.chip-group-pill`, `.seg-pill`, `.table`, `.data-table`, `.tool-view`, `.sec-header`, `.hint`, `.info`, `mark.annotation` (and its `.annotation-exact/-shifted/-ambiguous/-unvetted` variants). Don't copy the color tokens into a standalone artifact — they'll drift.

For everything else (the common case), inline the Anthropic tokens per the general doc and ignore `styles.css` entirely.

---

## Footguns

**R2 sync.** `experiments/` is R2-synced, not git-tracked. An HTML artifact at `experiments/<exp>/<sub>/results/foo.html` rides `r2_push.sh`/`r2_pull.sh`. Per the R2 safety memory: always `r2_pull.sh` before `r2_push.sh --full` — almost-deleted-10K-files lives here. If your artifact embeds large per-response JSON inline, mind that it bloats the R2 push.

**MkDocs nav.** Files in `docs/viz_findings/` are listed in `index.yaml` and rendered as part of the site. An `.html` file there does **not** appear in MkDocs nav, won't be search-indexed, and can only be reached via a markdown stub that links to it. If the finding is destined for the public site, write markdown; HTML is for internal or external-direct-link-only findings.

**Relative `<link>` paths break on promote.** An artifact in `experiments/.../results/foo.html` with `<link href="../../../visualization/styles.css">` works locally and breaks when promoted (because `experiments/` isn't in `.publicinclude` and `visualization/` lives at a different depth on prod). Inline-or-Anthropic only — never link the dashboard stylesheet from anything that might travel.

**Promote-script silent skip.** Per the `release.sh` footgun in `docs/main.md`: if you add an HTML file inside an already-whitelisted glob (e.g., `docs/viz_findings/`) and don't `git add` it, the promote scripts silently skip it. Same hazard as markdown; same mitigation — skim `git status` before releasing.

**Data hygiene.** Don't inline raw model outputs, scenario text, or trait examples into an HTML artifact that will leave the repo (gist, blog, LessWrong) without checking whether that content is releasable. Per project policy, no fabricated examples — but also no premature publication of unvetted ones.

**Don't depend on R2-only data from `dev/`.** A teammate cloning the repo doesn't have R2 data. If a `dev/` artifact needs R2 contents to render, either embed the data inline or document the `r2_pull.sh` prerequisite at the top of the file.

---

## High-value artifact ideas (concrete)

Targets where an HTML artifact would clearly beat the markdown status quo, with file anchors:

| Artifact | Source / context | Pattern |
|---|---|---|
| Cross-bias eval rollup v2 | `dev/conv_tools/cross_bias_eval/_summary.json` → `index.html` (generator exists) | Interactive heatmap with hover detail panel |
| SAE feature browser | `dev/sae/encode_sae_features.py`, `evaluate_trait_alignment.py` outputs | Top-activating spans with score-sorted feature list |
| Steering coefficient sweep | `utils/coefficient_search.py` outputs | Sliders over coef × layer; response preview pane |
| Dead-zone explorer | `dev/dead_zone.py` | Threshold slider with live recompute and annotated examples |
| Fingerprint comparison | per `project_fingerprint_methodology_ideas.md` | Cross-variant z-scored heatmap with EM-vs-baseline toggle |
| Held-out triage editor | Stage 5/6 outputs | Drag responses across trait-presence columns; export labels as JSON |
| Judge-prompt tuner | `datasets/llm_judge/*` | Editable system prompt + live re-scoring of three sample responses |

---

## Anti-patterns specific to this repo

- **Don't HTML-ify `docs/main.md`-linked docs.** The doc index is markdown; everything linked is expected to be markdown for the MkDocs site.
- **Don't HTML-ify `_notepad.md` or `dev/tasks/*` files.** They're working docs.
- **Don't replace the dashboard with one-off HTML.** If an artifact proves useful enough to keep, port it *into* `visualization/` as a new view; don't let it linger as a standalone file forever.
- **Don't ship an artifact through `release.sh` without confirming it renders standalone.** Linked stylesheets break; inline-only is the safe default.

---

## Checklist (delta vs. general doc)

Items not covered by `@~/.claude/html-artifacts.md`:

- [ ] Lives in `dev/`, `experiments/<exp>/`, or `docs/viz_findings/` — not `docs/` top-level, not `visualization/`.
- [ ] Uses the Anthropic palette inline; does not `<link>` `visualization/styles.css` unless it's clearly dashboard-adjacent and won't be promoted.
- [ ] If under `experiments/`: data hygiene check (no unreleasable content), R2-push implications considered.
- [ ] If under `docs/viz_findings/`: paired with a markdown stub that links to it (and is itself listed in `index.yaml`).
- [ ] If iterated by a script: the script lives next to it (`build_<name>.py`) — don't hand-edit regenerated files.
