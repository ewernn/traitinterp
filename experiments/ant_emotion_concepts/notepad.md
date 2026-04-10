# Emotion Concepts Replication — Notepad

**Status**: PLANNING
**Started**: 2026-04-09
**Updated**: 2026-04-09

## Progress

- [ ] Stage 1: Dataset generation
- [ ] Stage 2: Vector extraction
- [ ] Stage 3: Geometry analysis
- [ ] Stage 4: Validation experiments
- [ ] Stage 5: Layer dynamics
- [ ] Stage 6: Speaker probes
- [ ] Stage 7: Steering experiments
- [ ] Stage 8: Post-training comparison
- [ ] Stage 9: Deflection probes
- [ ] Stage 10: Comparison to paper results

## Observations

(none yet)

## Decisions

- 171 emotions from Anthropic's list (not our own ~170 traits)
- 20 topics × 2 rollouts per emotion = 40 stories each (>99% cos sim to 100×12 at 20×1 on contrastive extraction — needs validation for story-based)
- T=0.7 (diverse stories, requires seed infrastructure for reproducibility)
- Llama 3.3 70B Instruct at int4
- Same model generates + extracts
- Skip 8 experiments (proprietary), include 3 partial with reconstruction
- Compare all results to Anthropic's Sonnet 4.5 numbers
