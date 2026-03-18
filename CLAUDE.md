@docs/main.md

## Subagents

Spawn background subagents liberally — investigators, critics, explorers — whenever doing non-trivial work. Use them to cover blind spots, critique proposals, challenge assumptions, audit implementations, and find things you missed. Before implementing anything significant, spawn a critic agent to stress-test the plan. After implementing, spawn one to review. Unlimited Claude Code usage — err on the side of spawning too many rather than too few.

## Core Principles

**No hardcoding**: Paths, experiment names, trait names, examples - always use variables/templates that resolve at runtime. If you're typing a specific value that could change, it should be a parameter. All paths flow through PathBuilder APIs (`utils/paths.py`, `visualization/core/paths.js`) which read from `config/paths.yaml`.

**Docs**: Integrate insights into permanent docs as you go. No session files. Summary in conversation, lasting insights in `docs/`.

**Code style**:
- Module docstrings: one-line description + `Input:`, `Output:`, `Usage:` sections
- Script name should match output file name when applicable
- Function docstrings only for non-obvious functions
- Prefer longer descriptive file names (e.g., `trait_annotation_correlation.py` over `correlation.py`)

**Codebase standards**:
- paths standardized (PathBuilder everywhere) and robust (experiment-agnostic scripts). code clean (delete legacy code/docs) and maintainable (single source of truth).
- fail fast with clear errors

**Visualization**
- only use primitives from visualization/styles.css
- reuse existing code as much as possible

**Writing style** (for docs, overview, methodology, findings):
- Natural, concise prose. Explain concepts simply before technical details.
- Use bullet points freely. Avoid jargon where plain language works.
- First person plural ("we") for actions.
- Assume familiarity with ML basics (probes, steering, activations) but write for broader technical audience.
