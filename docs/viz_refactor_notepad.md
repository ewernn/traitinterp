# Visualization Refactor Notepad

## Goal
First-principles refactor of visualization/. Clean, robust, zero duplication. Strong core primitives.

---

## Revised Plan (post-critic review)

### Two Phases

**Phase 1: Safe cleanup (now, pre-release)**
Zero behavioral change. Pure deletion + fixes.
- Delete ~1,500 lines of dead code across 15 files
- Fix 3 bugs (closePreview, model-select, config keys)
- Fix 17 undefined CSS variables
- Add missing CSS primitives (.hint, .chart-mount, .dropdown-body-flush, etc.)
- Merge legend.js → charts.js (same concern, same callers)
- Delete paths.js dead methods (629 → ~240 lines)
- Clean state.js (dead exports, missing localStorage keys)
- Move layer-deep-dive.js + one-offs.js → visualization/dev/archived/
- Move design.html → visualization/dev/

**Phase 2: Structural refactor (post-release)**
These require design decisions and have migration risk.
- Methodology.js → delegate to window.customBlocks (285 → ~65 lines)
- Add [@key] citations to citations.js (eliminate 2-way duplication)
- Extras tab (new API endpoint + correlation absorbed + auto-discovery)
- Merge model-config.js → paths.js (eliminate duplicate config.json fetch)
- State shape cleanup (modules own their own caches)
- .publicinclude granularity (enumerate viz files instead of bulk visualization/)

---

## Architecture Decisions (settled by agent investigation)

### Keep trait-dynamics.js name
Don't rename to inference.js. Critic caught: naming collision with Python's inference/run_inference_pipeline.py, breaks URL bookmarks (?tab=trait-dynamics), file name doesn't match view ID. Current name describes what the view does (trait evolution), not when data was captured.

### Keep content views on main
Overview and methodology ARE the onboarding. New users who clone and run serve.py need context — the dashboard tabs show nothing without experiment data. Removing content views from main creates a blank-page first impression.

### Core module consolidation
| Module | Decision | Reason |
|--------|----------|--------|
| legend.js → charts.js | **Merge** | Same concern (Plotly), same callers, 125 lines |
| annotations.js | **Keep in core/** | Cross-cutting utility, 3 callers from different layers |
| model-config.js → paths.js | **Merge (Phase 2)** | Both load YAML, duplicate config.json fetch |
| conversation-tree.js | **Keep separate** | Self-contained data structure, strong identity |
| chart-types.js | **Keep separate** | 772 lines, 9 renderers, extensible registry |

### Core/ rule
"No rendering concerns" — core files should not own UI widgets. They may touch DOM for cross-cutting utility work (reading CSS vars, showing errors, caching).

### Extras tab (Phase 2)
- New API endpoint `GET /api/experiments/{exp}/analysis` for auto-discovery
- Correlation absorbed as a NAMED section (code lifted verbatim, not genericized)
- Generic JSON renderer for unknown analysis dirs
- Tab auto-hides when empty
- Keeps one-offs-style patterns: scatter+regression, sortable tables, layer-sweep charts

### Branch strategy (simplified)
- **dev** → ground truth (everything)
- **main** → public release (via .publicinclude, currently includes all of visualization/)
- **prod** → Railway deployment (currently just stale dev; consider deploying dev directly)
- Decision deferred on selective visualization promotion — keep bulk `visualization/` include for now

---

## Dead Code Inventory

### paths.js (629 → ~240 lines, 62% reduction)
27 dead methods. Keep only: load, setExperiment, getExperiment, get, formatPositionDisplay, residualStreamData, responseData, extractionEvaluation, logitLens, sanitizePosition (internal), desanitizePosition (internal).

### model-config.js (12 dead methods)
Only loadForExperiment and getDefaultMonitoringLayer called externally. All SAE methods dead (only caller was unreachable layer-deep-dive). Phase 2: merge into paths.js.

### state.js dead exports
- GLOBAL_VIEWS (defined, never read)
- populatePromptSelector (no-op stub, 2 dead call sites)
- stopGpuPolling, fetchGpuStatus, startGpuPolling (never called externally)

### state.js undeclared fields (Phase 2 cleanup)
External modules dump onto window.state without declaring:
- _annotationCache, _sentenceAnnotationCache → move to annotations.js local
- promptPage, promptTagsCache, tagsPreloaded → move to prompt-picker.js local
- traitCorrelationData, correlationOffset → move to correlation.js local
- selectedSteeringEntry → declare in state shape

### state.js missing localStorage keys
Add: 'wideMode', 'traitHeatmapOpen'

### Other dead code
- legend.js: syncLegendVisibility
- utils.js: markdownToHtml
- display.js: getTokenHighlightColors
- annotations.js: loadAndConvertAnnotations, applyHighlights, spanToTokenRange
- response-browser.js: clearResponseBrowserCache, resetResponseBrowserState
- sidebar.js: window.sidebar namespace object (never accessed)
- custom-blocks.js: private escapeHtml shadow at line 894

### Entire files to archive
- layer-deep-dive.js (723 lines, unreachable — no nav, no route)
- one-offs.js (1,371 lines, hardcoded research paths)
- Remove 'layer-dive' from ANALYSIS_VIEWS in state.js

## Bugs
- closePreview() undefined — index.html:46 calls it, no JS defines it
- model-select element missing — live-chat.js:359 queries it, template doesn't render it
- application_model/extraction_model — live-chat.js:192-193 reads non-standard config keys
- selectedSteeringTrait vs selectedSteeringEntry — likely same concept, two names (state.js:47 vs steering.js:147)

## CSS
### Undefined variables (17 total)
**Typos (5):** --accent-color, --color-error, --success-color, --warning-color, --text-md → alias to existing vars
**Missing aliases (7):** --font-mono, --accent-primary, --border, --bg-hover, --bg-quaternary, --text-quaternary, --error-color → add to :root
**New semantic tokens (5):** --text-error, --bg-error, --chart-1-alpha, --border-primary, --border-secondary → add to :root + dark theme

### New utility classes needed
.hint, .hint-muted, .dropdown-header-trail, .dropdown-body-flush, .span-delta.positive/.negative, .chart-mount / .chart-mount-sm / .chart-mount-lg

## Patterns to Preserve from Archives

### From layer-deep-dive.js
1. Multi-bucket cache with composite key (prompt changes reload, token changes re-render)
2. Parallel fetch with graceful degradation (Promise.all + null fallback)
3. CSS-var-derived Plotly colorscale → move to core/display.js
4. Cross-chart hover coupling (hover heatmap row → updates side panel)
5. staticPlot: true for non-interactive sparklines
6. Per-bar hovertemplate arrays (decoupled bar label from hover)
7. Dynamic height scaling (numBars * 28px)
8. Model capability check before fetch (avoid guaranteed 404s)

### From one-offs.js
1. Scatter plot with OLS regression line + identity diagonal
2. Sortable table + lazy expand-in-place detail rows
3. Layer-sweep multi-trace line chart (per-vector color, per-method style)
4. Grouped bar chart (best value per category)
5. Split violin distribution
6. Conditional section rendering (has_X ? render : placeholder)

## Methodology Consolidation (Phase 2)
285 → ~65 lines by delegating to window.customBlocks. One blocker: figure asset path differs (methodology: /docs/assets/, findings: /docs/viz_findings/assets/). Fix: add assetBaseUrl param to renderCustomBlocks. Also add [@key] citation support to citations.js (currently duplicated in methodology.js + findings.js).

---

## Progress Log

### 2026-03-21
- Created refactor notepad
- Completed architecture map via 6 subagents (core, views, components, dead-js, endpoints, misc)
- CSS cleanup: -653 lines (85 dead classes, var fix)
- serve.py cleanup: -117 lines (4 dead endpoints, cache, yaml import)
- Fixed --color-text-tertiary/secondary bug in styles.css + trait-dynamics.js
- Cleaned iCloud duplicate files from .git/ (11 files including 6 index copies)

### 2026-03-22
- Refined plan via 9 subagents (layer-deep-dive ideas, one-offs ideas, methodology audit, extras design, paths.js audit, core consolidation, state.js audit, CSS primitives, critic)
- Critic identified: keep content views on main, don't rename trait-dynamics, extras ≠ dumping ground, .publicinclude blocks selective viz promotion
- Phase 1 committed: -1,851 lines (dead code, bugs, CSS, archives)
- Structural refactor committed: -239 net (20 files, top-spans extraction, methodology 285→90, helpers added, model-config merged, state cleaned)
- Total this session: -2,090 net lines

#### Remaining refactor opportunities identified by deep audits:

**trait-dynamics.js (1,576 → ~900-1,000)**:
- Deduplicate diff logic (replay_suffix vs standard = copy-paste, ~-50 lines)
- Use createSeparatorShape/createHighlightShape (3 sites manually rebuild, ~-30 lines)
- Extract buildCommonShapes (duplicated in renderCombinedGraph + renderTraitTokenHeatmap, ~-30 lines)
- Compress attachControlListeners with bindChange helper (~-40 lines)
- Extract loadAndNormalizeProjections from 400-line render function (~-99 lines)
- Move math utils to core/utils.js (computeVelocity, getDimsToRemove, applyMassiveDimCleaning, computeCleanedNorms, ~-67 lines)
- Add buildTokenAxisTicks to charts.js (~-9 lines)

**steering.js (1,148)**: awaiting deep audit results

**model-analysis.js (777 → ~650)**:
- 4 diagnostic plots share identical async shell → withMassiveActivationsData helper (~-48 lines)
- fetchMassiveActivationsData called 4x independently → call once, pass down
- 'topright' legend position is a silent bug → fix to 'right'
- Raw h4.subsection-header bypasses ui.renderSubsection (CSS class mismatch)
- sortedNumericKeys pattern repeated 3x

**Cross-view patterns not yet extracted**:
- parseFrontmatter identical in methodology.js + findings.js → utils.js
- Heatmap annotation loops in correlation.js (buildHeatmapAnnotations exists but unused)
- marked.setOptions called identically in 3 views → call once at startup
- sortedNumericKeys(obj) repeated 6x across model-analysis + steering
- Score badge class logic repeated 4x → ui.scoreClass()

**Components**:
- custom-blocks.js (1,395 → ~900-1,000): 11 double-replace patterns, 3 tab-widget trios (270 lines), broken desanitize
- response-browser.js (716 → ~580): duplicated response-item HTML, score badges
- prompt-picker.js (695 → ~600): dead null guards, duplicated diff-mode logic
- top-spans.js (491 → ~440): duplicated span row template

**Core dead code (second pass)**:
- paths.js: window.modelConfig shim + 8 model-config methods dead (only archived caller)
- charts.js: buildHeatmapAnnotations dead (never called by anyone)
- chart-types.js: crosseval-comparison renderer dead (no markdown usage)
- state.js: stopGpuPolling dead, legacy migration stale, HIDDEN_EXPERIMENTS dormant
