# Visualization Dashboard

Interactive dashboard for exploring trait extraction quality, steering sweeps, per-token trait dynamics, model internals, and live chat with real-time monitoring.

```bash
python visualization/serve.py
# Visit http://localhost:8000/
```

Set `MODE=production` to disable dev-only features (experiment picker, GPU debug, local inference toggle). Default is `development`.

---

## Directory Structure

```
visualization/
├── serve.py                  # Python HTTP server (static files + REST API + SSE chat)
├── chat_inference.py         # Backend for live chat (model loading, streaming generation)
├── index.html                # SPA shell: sidebar nav, script loading, inline view router
├── styles.css                # All CSS — design tokens, component classes, theme (light/dark)
│
├── core/                     # Shared primitives (no DOM side effects, no view-specific logic)
│   ├── state.js              # Global state object, preference persistence, experiment loading, URL routing
│   ├── ui.js                 # HTML-returning helpers: toggles, selects, chips, guards, filter rows
│   ├── display.js            # Display names, color palette, CSS variable access, Plotly theming
│   ├── charts.js             # Chart presets, layout builder, separator/highlight shapes, HTML legend
│   ├── chart-types.js        # Renderers for :::chart::: blocks in markdown (model-diff, dynamics, etc.)
│   ├── paths.js              # PathBuilder — loads config/paths.yaml, resolves experiment-relative paths
│   ├── utils.js              # Pure utilities: escapeHtml, smoothing, math protection, fetch wrappers
│   ├── types.js              # JSDoc type definitions (AppState, SteeringEntry, ProjectionData, etc.)
│   ├── annotations.js        # Text span → character range conversion for highlighting
│   ├── citations.js          # Numbered (^1) and keyed ([@key]) citation parsing + rendering
│   ├── markdown-view.js      # Shared markdown renderer (frontmatter, custom blocks, citations, math)
│   └── conversation-tree.js  # Tree data structure for multi-turn chat with branching
│
├── components/               # Reusable UI components (manage their own DOM region)
│   ├── sidebar.js            # Main sidebar: theme toggle, navigation, trait checkboxes, GPU status
│   ├── prompt-picker.js      # Bottom bar: prompt set dropdown, prompt ID buttons, token slider
│   ├── prompt-set-sidebar.js # Left panel: prompt set list for inference views
│   ├── inference-controls.js # Local/Modal inference toggle, connection status, warmup
│   ├── live-chat-chart.js    # Real-time Plotly chart for live chat trait scores
│   ├── response-browser.js   # Steering response table with sort/filter/expand
│   ├── top-spans.js          # Cross-prompt span analysis (highest-delta token windows)
│   └── custom-blocks.js      # :::block::: syntax parser/renderer for markdown views
│
├── views/                    # One module per dashboard tab
│   ├── overview.js           # Renders docs/overview.md
│   ├── methodology.js        # Renders docs/methodology.md
│   ├── findings.js           # Collapsible research finding cards from docs/viz_findings/
│   ├── extraction.js         # Extraction quality: best vectors, per-trait heatmaps, logit lens
│   ├── steering.js           # Steering orchestrator (delegates to sub-modules below)
│   ├── steering-filters.js   # Filter chips, cached data fetching, filter state
│   ├── steering-best-vector.js  # Best vector per layer charts
│   ├── steering-heatmap.js   # Layer × coefficient heatmaps
│   ├── trait-dynamics/        # Inference view (multi-file)
│   │   ├── index.js          # Orchestrator: loads data, renders sub-charts
│   │   ├── controls.js       # Page shell, control bar (smoothing, methods, layer mode)
│   │   ├── data.js           # Data loading, projection processing, comparison fetching
│   │   ├── chart-trajectory.js   # Token trajectory (X=tokens, Y=cosine similarity)
│   │   ├── chart-heatmap.js      # Trait × token heatmap
│   │   └── chart-magnitude.js    # Activation magnitude per token
│   ├── correlation.js        # Annotation correlation (embedded in inference tab)
│   ├── model-analysis.js     # Activation diagnostics + model variant comparison
│   └── live-chat.js          # Multi-turn chat with real-time trait monitoring
│
└── dev/                      # Development-only files
    ├── design.html           # CSS design playground (served at /design)
    └── archived/             # Archived/deprecated view modules
```

---

## How Views Work

Each view is an ES module that exports a `renderX()` function. The inline router in `index.html` calls the appropriate function when the user navigates between tabs.

### The View Contract

A view module must:

1. **Export a render function** and register it on `window` for the router:
   ```js
   export function renderExtraction() { ... }
   window.renderExtraction = renderExtraction;
   ```

2. **Write to `content-area`** — the main content div:
   ```js
   const contentArea = document.getElementById('content-area');
   contentArea.innerHTML = `<div class="tool-view">...</div>`;
   ```

3. **Guard with `ui.requireExperiment()`** if the view needs experiment data. This shows a "select an experiment" message and returns `true` if no experiment is loaded:
   ```js
   if (ui.requireExperiment(contentArea)) return;
   ```

4. **Use `ui.deferredLoading()`** for async data fetches (avoids flash on fast loads):
   ```js
   const { cancel } = ui.deferredLoading(contentArea, 'Loading data...');
   const data = await fetchJSON(url);
   cancel();
   ```

5. **Use `ui.renderRunHint()`** for missing data, showing the command to generate it.

### Simple view (markdown):
```js
import { renderMarkdownView } from '../core/markdown-view.js';

export function renderOverview() {
    return renderMarkdownView('/docs/overview.md');
}
window.renderOverview = renderOverview;
```

### Analysis view (data-driven):
```js
import { fetchJSON } from '../core/utils.js';

async function renderExtraction() {
    const contentArea = document.getElementById('content-area');
    if (ui.requireExperiment(contentArea)) return;

    const { cancel } = ui.deferredLoading(contentArea, 'Loading...');
    const data = await fetchJSON(window.paths.extractionEvaluation());
    cancel();

    if (!data) {
        contentArea.innerHTML = `<div class="tool-view">${ui.renderRunHint(
            'No data found',
            'python analysis/...'
        )}</div>`;
        return;
    }

    contentArea.innerHTML = `<div class="tool-view">...</div>`;
}
export { renderExtraction };
window.renderExtraction = renderExtraction;
```

---

## Module System

All JavaScript uses ES modules (`import`/`export`) loaded via `<script type="module">` in `index.html`.

Cross-module communication uses a **`window.*` bridge** for two specific cases:

- **Router dispatch**: The inline router in `index.html` calls `window.renderX()`, so every view must register its render function on `window`.
- **`onclick` handlers in HTML strings**: When building HTML via template literals, inline `onclick` handlers can only call functions on `window`. Modules that need this bridge specific functions (e.g., `window.setSpanWindowLength`).

Everything else uses standard ES module imports. State is accessed via `window.state` (exported from `state.js`) or imported directly:

```js
import { state, ANALYSIS_VIEWS } from '../core/state.js';
```

### Global singletons on `window`

- `window.state` — the global state object
- `window.paths` — the PathBuilder instance (loads from `config/paths.yaml`)
- `ui` — used as a bare global in some views (e.g., `ui.requireExperiment()`). Not explicitly set on `window`; some views import individual functions from `ui.js`, others rely on the implicit global. New code should import explicitly: `import { requireExperiment } from '../core/ui.js'`
- `window.renderView()` — triggers the current view's render function
- `window.Plotly`, `window.marked`, `window.MathJax` — external libraries loaded via CDN

### State management

`state.js` owns all global state. Preferences are persisted to `localStorage` via a table-driven system — each preference declares its type, default, validation, and optional `onSet` callback. Views read from `window.state` and call setter functions to update values.

---

## How to Add a New View

Checklist:

1. **Create the view module** in `views/`:
   - Export a `renderX()` function
   - Register it: `window.renderX = renderX`
   - Follow the view contract above

2. **Add a nav entry** in `index.html`:
   - Add `<div class="nav-item" data-view="your-view">Your View</div>` to the appropriate sidebar section
   - If it's an analysis view (needs experiment data), add it to the analysis panel

3. **Register in the router** in `index.html`:
   - Add a `case 'your-view':` to the `switch` in `window.renderView`
   - Add the title to the `viewTitles` map

4. **Load the script** in `index.html`:
   - Add `<script type="module" src="/visualization/views/your-view.js"></script>` in the View Modules section

5. **If it's an analysis view**, add `'your-view'` to the `ANALYSIS_VIEWS` array in `state.js` so the sidebar and experiment loading logic knows about it.

6. **Use only CSS primitives from `styles.css`** — do not add view-specific CSS unless absolutely necessary.

---

## Server API

`serve.py` provides both static file serving and a REST API. Key endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /api/experiments` | List experiments with extraction data |
| `GET /api/experiments/{exp}/config` | Experiment config + model metadata |
| `GET /api/experiments/{exp}/traits` | Discovered traits (extraction + projection) |
| `GET /api/experiments/{exp}/inference/prompt-sets` | Prompt sets with available IDs and variants |
| `GET /api/experiments/{exp}/steering` | Steering entries |
| `GET /api/experiments/{exp}/steering-results/{trait}/{variant}/{position}/{prompt_set}` | Steering results (JSONL → JSON) |
| `GET /api/experiments/{exp}/model-diff` | Model diff comparisons |
| `GET /api/experiments/{exp}/inference/{variant}/projections/{category}/{trait}/{prompt_set}/{id}` | Projection data (combines multi-vector files on-the-fly) |
| `GET /api/config` | App config (mode, feature flags) |
| `GET /api/gpu-status` | GPU device info and memory |
| `POST /api/chat` | SSE stream: chat generation with per-token trait scores |
