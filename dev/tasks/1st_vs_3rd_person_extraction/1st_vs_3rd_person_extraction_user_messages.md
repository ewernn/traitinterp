# 1st-vs-3rd Person Extraction — User Messages

## Original goal (from planning request)

Plan A: 1st-person vs 3rd-person elicitation for persona vector extraction.

Test whether extracting trait vectors from 1st-person scenarios produces stronger / cleaner / more causal directions than 3rd-person observations. Replaces a deleted refusal-only finding.

Established context:
- 3 traits: evil, sycophancy, hallucination (persistent dispositions, not actions)
- Source data exists at `datasets/traits/pv_natural/{evil,sycophancy,hallucination}/positive.txt + negative.txt` — all 1st-person
- Need to generate 3rd-person counterparts at `datasets/traits/pv_natural_3p/`
- Same model (Llama-3.1-8B), same extraction position (response[:5]), same method (probe), same layer sweep
- Compare: best steering Δ at coherence ≥70 (primary), val_effect_size + val_auroc (secondary), vector cosine similarity
- Single position only — keep this experiment focused on perspective; position sweep is separate Plan B

## User decisions during planning

- **Complexity:** Medium.
- **Model:** Llama-3.1-8B (match comparison-persona-vectors).
- **3p generation method:** Rewrite via Claude Code (no external API), using `docs/trait_dataset_creation_base_model.md` adapted for 3p.
- **Hallucination padding:** Pad to 150 (originally 49).
- **Pronoun convention:** Singular they/them throughout for cleanliness. User noted variety (named entities, "the assistant") as a future-experiment idea but not for this run.
- **Output dir:** `experiments/1st_vs_3rd_person_extraction/` (NOT `persona_perspective`). Re-extract both 1p and 3p from scratch in this dir; copy nothing from `persona_vectors_replication`.
- **Success criterion:** 1p Δ exceeds 3p Δ by ≥20% relative on ≥2 of 3 traits at coherence ≥70.
- **Secondary metrics:** Yes — report val_effect_size and AUROC alongside steering Δ.

## Connections noted by user (not in scope, but worth remembering)

- ant_emotion_concepts ran a related "user vs assistant" comparison (using the ":" token after speaker labels). Possible follow-up experiment.
- The "perspective gap" might depend on how much the trait depends on inhabited experience (dispositional traits like evil might show smaller gaps than interpersonal traits like sycophancy).
