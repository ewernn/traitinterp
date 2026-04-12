# ant_emotion_concepts — Replication of Sofroniew et al. 2026

Replicates **"Emotion Concepts and their Function in a Large Language Model"** (Sofroniew et al., Anthropic, 2026) on **Llama 3.3 70B Instruct**, using the traitinterp repo's mainline primitives.

**Landing numbers**: see [`ant_emotion_concepts_findings.md`](ant_emotion_concepts_findings.md) §3 for the Sonnet 4.5 (paper) vs Llama 3.3 70B (ours) side-by-side replication table (15 rows). §4 lists all limitations and noise-floor caveats that must accompany any cited number.

## What runs natively on the mainline repo

| Paper step | Mainline primitive |
|---|---|
| Extract 171 emotion vectors from contrasting stories | `extraction/run_extraction_pipeline.py` |
| Grand-mean subtract + neutral-PC denoise (`mean_diff+gm+pc50`) | `analysis/vectors/cross_trait_normalize.py` |
| Pairwise cosine heatmap + hierarchical cluster order (Fig 5) | `analysis/vectors/geometry.cosine_heatmap_ordered` |
| K-means + UMAP cluster structure (Fig 6) | `analysis/vectors/geometry.trait_clusters` / `umap_projection` |
| PCA vs Russell-Mehrabian valence/arousal norms (Fig 8) | `analysis/vectors/geometry.pca_norm_correlation` |
| Cross-layer representational similarity (Fig 9) | `analysis/vectors/geometry.representational_similarity` |
| Capture activations at a position (stages 4/5/8) | `utils/capture_activations.capture_at_position` |
| Per-variant model diff (stage 8 Fig 36 candidate) | `analysis/model_diff/compare_variants.py` |
| Steering evaluation + coefficient search | `steering/run_steering_eval.py` |
| LLM-as-judge grading (stage 7) | `utils/judge.TraitJudge` |

## What stays experiment-specific

- `scripts/stage1p3_generate_dialogues.py` and `stage1p4_generate_deflection.py` — paper-verbatim 2-speaker + deflection dialogue generation (A.4 / A.11 templates)
- `scripts/stage6_speaker_probes.py` — the 2×2 speaker/emotion cosine grid (Fig 17-18) with per-turn token boundaries
- `scripts/shared.py::BLACKMAIL_SYSTEM_PROMPT` + email chain (A.13 scenario)
- `scripts/stage7_steering.py::run_decision_gate` — paper §3 gate-check pattern
- `datasets/traits/ant_emotion_concepts/` — 171 emotion trait directories + `_neutral/` reference corpus

## Cross-compaction anchors

- [`ant_emotion_concepts_findings.md`](ant_emotion_concepts_findings.md) — clean ~170-line digest (replication table + limitations + per-stage numbers with source JSONs)
- [`ant_emotion_concepts_session_continuation.md`](ant_emotion_concepts_session_continuation.md) — full post-compact state
- [`ant_emotion_concepts_audit_trail_findings.md`](ant_emotion_concepts_audit_trail_findings.md) — archived 1,192-line iteration log (includes retracted scope-creep narrative threads)
- [`ant_emotion_concepts_notepad.md`](ant_emotion_concepts_notepad.md) — chronological progress log
