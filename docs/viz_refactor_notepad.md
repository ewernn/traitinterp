# Visualization Refactor Notepad

Dev-only tracking. Not promoted to main.

---

## Phase 2 TODO (post-release structural refactor)

- Methodology.js → delegate to window.customBlocks (285 → ~65 lines). Blocker: figure asset path differs (methodology: /docs/assets/, findings: /docs/viz_findings/assets/). Fix: add assetBaseUrl param to renderCustomBlocks.
- Add [@key] citations to citations.js (eliminate 2-way duplication with methodology.js + findings.js)
- Extras tab: new API endpoint `GET /api/experiments/{exp}/analysis` for auto-discovery. Correlation absorbed as named section. Generic JSON renderer for unknown analysis dirs. Tab auto-hides when empty.
- State shape cleanup (modules own their own caches — move undeclared fields to owning modules)

---

## Architecture Decisions (settled)

### Keep trait-dynamics.js name
Don't rename to inference.js. Naming collision with Python's inference/run_inference_pipeline.py, breaks URL bookmarks (?tab=trait-dynamics). Current name describes what the view does (trait evolution), not when data was captured.

### Keep content views on main
Overview and methodology ARE the onboarding. New users who clone and run serve.py need context — the dashboard tabs show nothing without experiment data.

### Core module consolidation
| Module | Decision | Reason |
|--------|----------|--------|
| legend.js → charts.js | **Done** | Same concern (Plotly), same callers, 125 lines |
| annotations.js | **Keep in core/** | Cross-cutting utility, 3 callers from different layers |
| model-config.js → paths.js | **Done** | Both load YAML, duplicate config.json fetch |
| conversation-tree.js | **Keep separate** | Self-contained data structure, strong identity |
| chart-types.js | **Keep separate** | 772 lines, 9 renderers, extensible registry |

### Core/ rule
"No rendering concerns" — core files should not own UI widgets. They may touch DOM for cross-cutting utility work (reading CSS vars, showing errors, caching).

---

## Remaining Dead Code

- chart-types.js: crosseval-comparison renderer dead (no markdown usage)
- state.js: stopGpuPolling dead, legacy migration stale, HIDDEN_EXPERIMENTS dormant

---

## Open Bugs

- closePreview() undefined — index.html:46 calls it, no JS defines it
- model-select element missing — live-chat.js:359 queries it, template doesn't render it
- application_model/extraction_model — live-chat.js:192-193 reads non-standard config keys
- selectedSteeringTrait vs selectedSteeringEntry — likely same concept, two names (state.js:47 vs steering.js:147)

---

## Remaining Refactor Opportunities

**trait-dynamics/ (1,576 → ~900-1,000)**:
- Deduplicate diff logic (replay_suffix vs standard = copy-paste, ~-50 lines)
- Extract buildCommonShapes (duplicated in renderCombinedGraph + renderTraitTokenHeatmap, ~-30 lines)
- Extract loadAndNormalizeProjections from 400-line render function (~-99 lines)
- Move math utils to core/utils.js (computeVelocity, getDimsToRemove, etc.)

**model-analysis.js (777 → ~650)**:
- 4 diagnostic plots share identical async shell → withMassiveActivationsData helper (~-48 lines)
- 'topright' legend position is a silent bug → fix to 'right'

**Components**:
- custom-blocks.js (1,395 → ~900-1,000): 11 double-replace patterns, 3 tab-widget trios
- response-browser.js (716 → ~580): duplicated response-item HTML, score badges
- prompt-picker.js (695 → ~600): dead null guards, duplicated diff-mode logic
- top-spans.js (491 → ~440): duplicated span row template

---

## CSS TODO

### Undefined variables (17 total)
**Typos (5):** --accent-color, --color-error, --success-color, --warning-color, --text-md → alias to existing vars
**Missing aliases (7):** --font-mono, --accent-primary, --border, --bg-hover, --bg-quaternary, --text-quaternary, --error-color → add to :root
**New semantic tokens (5):** --text-error, --bg-error, --chart-1-alpha, --border-primary, --border-secondary → add to :root + dark theme

### Utility classes needed
.hint, .hint-muted, .dropdown-header-trail, .dropdown-body-flush, .span-delta.positive/.negative, .chart-mount / .chart-mount-sm / .chart-mount-lg
