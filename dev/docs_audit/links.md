# Docs Link Audit Report
Generated: 2026-04-19

---

## 1. Broken Markdown Links

### docs/trait_dataset_creation.md
- **Line 15** — `[extraction_guide.md#vector-selection](extraction_guide.md#vector-selection)` — anchor `#vector-selection` does not match the actual heading `## Vector Selection` (slug would be `#vector-selection`). **VALID** — slug matches. OK.

### docs/methodology.md
- **Line 94** — `:::dataset /datasets/traits/starter_traits/refusal/positive.jsonl` — the `refusal/` trait dir does not exist under `datasets/traits/starter_traits/` (only: `assistant_axis`, `desperate`, `formality`, `golden_gate_bridge`, `sad`, `sycophancy`). **BROKEN path reference** (dataset directive).
- **Line 96** — `:::dataset /datasets/traits/starter_traits/refusal/negative.jsonl` — same issue. **BROKEN**.
- **Line 159** — `[effect-size-vs-steering](viz_findings/effect-size-vs-steering.md)` — file exists. OK.
- **Line 166** — `[comparison-arditi-refusal](viz_findings/comparison-arditi-refusal.md)` — file exists. OK.

### docs/mkdocs/config/trait-format.md
- **Line 272 (example command)** — `--traits starter_traits/sycophancy,starter_traits/formality,starter_traits/desperate` — `desperate` exists but is not in the table of shipped starter traits (lines 255–265 list `sycophancy, refusal, concealment, evil, formality, hallucination, optimism, golden_gate_bridge, assistant_axis, assistant_axis_v1, assistant_axis_v5`). Minor inconsistency — the example uses `desperate` which is real but not in the table.

---

## 2. Broken Code Path References

### docs/extraction_guide.md
- **Line 307** — `SteeringHook (core/hooks.py:285-328)` — `SteeringHook` class begins at **line 315**, not 285. The range 285–328 is wrong. Off by ~30 lines.

### docs/response_schema.md
- **Line 55** — `inference/convert_rollout.py` — **file does not exist** in the repo. Script referenced as real code.

### docs/architecture.md (and docs/mkdocs/config/trait-format.md)
- `utils/fingerprints.py` is referenced in `docs/architecture.md` (line 123: "Fingerprint utilities (`utils/fingerprints.py`)") — **file does not exist**.

### docs/mkdocs/config/experiment-setup.md
- **Lines 132, 135** — Starter config lists `"Qwen/Qwen3.5-9B-Base"` and `"Qwen/Qwen3.5-9B"` — these appear to be placeholder/aspirational model IDs. As of April 2026, Qwen3.5 does not exist on HuggingFace (Qwen3 and Qwen2.5 do). The actual `experiments/starter/config.json` contains the same IDs — a live HuggingFace pull would fail.

### docs/mkdocs/config/trait-format.md (Starter traits table, lines 253–266)
- Lists 11 starter traits including `refusal`, `concealment`, `evil`, `hallucination`, `optimism`, `assistant_axis_v1`, `assistant_axis_v5` — **none of these directories exist** under `datasets/traits/starter_traits/`. Only 6 exist: `assistant_axis`, `desperate`, `formality`, `golden_gate_bridge`, `sad`, `sycophancy`.

### docs/mkdocs/config/experiment-setup.md
- **Line 76** — `Copy .env.example to .env` — `.env.example` **exists**. OK.

### analysis.md (docs/mkdocs/cli/analysis.md)
- **Lines ~218** — `datasets/russell_mehrabian_norms.json` (in geometry.py `--norms-file` example) — **does not exist** at `datasets/`; it's in the R2 bundle at `experiments/ant_emotion_concepts/datasets/russell_mehrabian_norms.json`. Path in example is wrong.

---

## 3. Broken Dashboard Tab References

| Location | Reference | Status |
|----------|-----------|--------|
| `docs/replicate_ant_emotion_concepts.md:3` | `https://traitinterp.com/?tab=findings#emotion-concepts-replication` | OK — `findings` tab exists (`visualization/views/findings.js`) |
| `docs/replicate_ant_emotion_concepts.md:122` | `http://localhost:8000/?tab=findings` | OK |
| `docs/mkdocs/index.md:9` | `https://traitinterp.com/?tab=live-chat` | OK — `live-chat` route registered in index.html |

No `tab=methodology` references found in the checked files.

---

## 4. External URLs (not fetched)

| File | URL | Note |
|------|-----|------|
| `methodology.md` | `https://arxiv.org/abs/2405.07987` | Platonic Representation Hypothesis — appears valid |
| `methodology.md` | `https://transformer-circuits.pub/2026/emotions/index.html` | Sofroniew et al. 2026 — appears valid |
| `methodology.md` | `https://arxiv.org/abs/2507.21509` | Persona Vectors — arxiv ID `2507` = July 2025, in the future relative to pub date; potentially pre-submission ID |
| `methodology.md` | `https://arxiv.org/abs/2406.11717` | Arditi refusal — appears valid |
| `replicate_ant_emotion_concepts.md` | `https://www.anthropic.com/research/emotion-concepts-function-lm` | Anthropic paper URL — format is plausible |
| `replicate_ant_emotion_concepts.md` | GitHub release URLs (`emotion-concepts-v1`) | Cannot verify without fetching |
| `mkdocs/cli/analysis.md` | `https://github.com/google/python-fire` | Standard link, appears valid |

---

## 5. Per-file Broken Link Count

| File | Broken links |
|------|-------------|
| `docs/extraction_guide.md` | 1 (SteeringHook line number) |
| `docs/inference_guide.md` | 0 |
| `docs/steering_guide.md` | 0 |
| `docs/trait_dataset_creation.md` | 0 |
| `docs/methodology.md` | 2 (refusal dataset directives — missing files) |
| `docs/replicate_ant_emotion_concepts.md` | 0 |
| `docs/create_ant_emotion_vectors.md` | 0 |
| `docs/architecture.md` | 1 (utils/fingerprints.py missing) |
| `docs/core_reference.md` | 0 |
| `docs/response_schema.md` | 1 (inference/convert_rollout.py missing) |
| `docs/chat_templates.md` | 0 |
| `docs/mkdocs/index.md` | 0 |
| `docs/mkdocs/cli/extraction.md` | 0 |
| `docs/mkdocs/cli/inference.md` | 0 |
| `docs/mkdocs/cli/steering.md` | 0 |
| `docs/mkdocs/cli/analysis.md` | 1 (russell_mehrabian_norms.json path) |
| `docs/mkdocs/config/experiment-setup.md` | 1 (Qwen3.5 model IDs don't exist on HF) |
| `docs/mkdocs/config/trait-format.md` | 7 (starter traits table lists 7 non-existent trait dirs) |

**Total: ~14 broken references across all files**
