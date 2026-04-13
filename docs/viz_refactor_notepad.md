# Visualization Refactor Notepad

Dev-only tracking. Not promoted to main.

---

## Phase 2 TODO

- Extras tab: new API endpoint `GET /api/experiments/{exp}/analysis` for auto-discovery. Correlation absorbed as named section. Generic JSON renderer for unknown analysis dirs. Tab auto-hides when empty.
- State shape cleanup: modules should own their own caches. `promptPickerCache`, prompt fields, comparison model state still live in central state.js.

---

## Open Bugs

- `application_model` config key read but never written — always falls through to fallback. Affected files: `inference/inference-view.js:225`, `steering/steering-view.js:125`, `components/inference-controls.js:56`. Fix: use `config.model_variants[config.defaults.application].model` instead.
- `HIDDEN_EXPERIMENTS` defined as empty array in state.js:23, referenced in 5 places but never populated. Remove or implement.

---

## Refactor Opportunities

**inference/ (~1,648 lines → ~1,000)** — formerly `trait-dynamics/`:
- Deduplicate diff logic (replay_suffix vs standard = copy-paste)
- Extract buildCommonShapes (duplicated in renderCombinedGraph + renderTraitTokenHeatmap)
- Extract loadAndNormalizeProjections from 400-line render function
- Move math utils to core/utils.js (computeVelocity, getDimsToRemove, etc.)

**model-analysis.js (~764 lines → ~650)**:
- 4 diagnostic plots share identical async shell → withMassiveActivationsData helper
- 'topright' legend position is a silent bug → fix to 'right'

**Components**:
- custom-blocks.js (~1,103 → ~900): 11 double-replace patterns, 3 tab-widget trios
- response-browser.js (~628 → ~580): duplicated response-item HTML, score badges
- prompt-picker.js (~659 → ~600): dead null guards, duplicated diff-mode logic
- top-spans.js (~510 → ~440): duplicated span row template

---

## Architecture Decisions (settled)

### Folder renamed trait-dynamics/ → inference/ (Apr 2026)
Route key is now `inference`. Old ?tab=trait-dynamics URLs no longer work — not a concern since nobody bookmarked them. The Python `inference/` directory is a different namespace (server-side) so there's no collision at the module level.

### Keep content views on main
Overview and methodology are the onboarding. Dashboard shows nothing without experiment data.

### Core/ rule
"No rendering concerns" — core files should not own UI widgets. May touch DOM for cross-cutting utility work (reading CSS vars, showing errors, caching).
