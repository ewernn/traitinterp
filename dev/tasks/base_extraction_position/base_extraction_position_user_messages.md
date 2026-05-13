# Base Extraction Position Sweep — User Messages

## Original goal (from planning request)

Plan B: Extraction position sweep — does the optimal extraction position depend on trait type?

Test whether the standard `response[:5]` extraction position is correct for all traits, or whether different trait categories benefit from different windows. Action-like traits (decisions made early) may saturate by token 5; persistent dispositions or emotional states may need longer windows for the model's internal representation to stabilize.

User explicitly said: NO refusal trait (they hate it). Want to test small-vs-large model (large base may maintain coherence longer at later positions).

## User decisions during planning

- **Complexity:** Medium-Large.
- **Traits:** PV (evil, sycophancy, hallucination) + 3 emotions (anxiety, enthusiasm, warmth — replacing initial choice of afraid/sad/loving from ant_emotion_concepts after that was found instruction-style and base-incompatible).
- **Emotion sourcing:** Generate fresh via `docs/trait_dataset_creation_base_model.md` iterative process. Don't pad existing 15-scenario versions 10x (that drowns the original distribution).
- **Positions:** 1, 3, 5, 10, 15, 20.
- **Sample size:** 300 prompts/trait (150 pos + 150 neg).
- **Models:** Llama-3.1-8B + Llama-3.3-70B (small + large, same family).
- **Run order:** 8B first, then 70B.
- **Steering verification:** 3 traits (sycophancy + anxiety + evil) at 6 positions × layer sweep 30-60% depth.
- **Success criterion:** Optimal-position curves have detectably different shapes per trait category.
- **Output dir:** `experiments/base_extraction_position/`

## User pushback on critic

Critic flagged "trait category × content style" as a 1:1 confound (PV traits abstract dispositional, emotions concrete narrative). User pushed back: per `docs/trait_dataset_creation_base_model.md`'s decision tree, different trait categories REQUIRE different scenario styles by design. AFFECTIVE traits NEED context + emotional peak; DISPOSITIONAL traits NEED situation + choice point. Content style is a dependent variable of category, not a free variable to control. Reframed experiment: measures how the doc's categorically-correct trait designs differ in position-sensitivity. The 4 categories are DISPOSITIONAL (evil), INTERPERSONAL (sycophancy), DECEPTION (hallucination), AFFECTIVE (anxiety/enthusiasm/warmth) — not the binary "PV vs emotion" framing.

## Connections noted by user (not in scope)

- ant_emotion_concepts ran a related "user vs assistant" comparison (using the ":" token after speaker labels). Possible follow-up.
- Plan A (1st_vs_3rd_person_extraction) is orthogonal: same model, same PV traits, perspective sweep at single position vs position sweep at single perspective.
- mar15-detection_layer_profiling future-ideas entry classifies traits into affective categories (naturally affective, narration-first, never affective) — connects to this experiment's category framing.
