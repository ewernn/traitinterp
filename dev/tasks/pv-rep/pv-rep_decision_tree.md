# pv-rep — Decision Tree

## D1: Position-confound control (response[:] for Arm B)
| Option | Description |
|---|---|
| A | Add Arm B' control extracting Natural at response[:] |
| B | Accept confound, caveat it |
| C | Run Natural at response[:] as primary |

**Chosen**: B
**Reason**: User clarified base model decoheres after 5-10 tokens; response[:5] is structurally required for natural elicitation, not a free parameter.
**Outcome**: pending

## D2: Cross-model-variant asymmetry (Base→Instruct vs Instruct→Instruct)
| Option | Description |
|---|---|
| A | Surface as caveat, no extra cells |
| B | Add cross-extracted control |

**Chosen**: A
**Reason**: Methodology choice IS the comparison; extra arm is semi-ill-defined (instruct model refuses natural prefixes).
**Outcome**: pending

## D3: Coefficient search n_steps
| Option | Description |
|---|---|
| A | Default 5 |
| B | Bumped to 8 (per critic) |

**Chosen**: B
**Reason**: Different vector norms between arms → different start_coef → 5 steps may underexplore.
**Outcome**: pending

## D4: Coherence judge during search
| Option | Description |
|---|---|
| A | Use cell-native coherence judge per cell (4 different coherence prompts in flight) |
| B | Use our coherence judge across all cells; post-hoc rescore Shao-judge cells with Shao's coherence prompt |

**Chosen**: B
**Reason**: TraitJudge doesn't accept coherence prompt override via CLI today; option A would require code change. B keeps search trajectories comparable and adds a clean post-hoc step.
**Outcome**: pending

---

## Pruned

### D1-PRUNED: Add Arm B' position control
- Status: NOT_ATTEMPTED
- Reason: User confirmed response[:5] is methodologically required, not a free parameter
- DO NOT RETRY UNLESS: a reviewer pushes back specifically on position confound and disambiguation is needed for publication

### D2-PRUNED: Cross-extracted control arm (Natural prefixes on Instruct)
- Status: NOT_ATTEMPTED
- Reason: Instruct model behavior on natural-elicitation prefixes is qualitatively different (refusals, instruction-echo); not a clean control
- DO NOT RETRY UNLESS: a clean way to do natural extraction on Instruct emerges
