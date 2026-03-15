# Experiment Structure

Each experiment lives in `experiments/{name}/` and follows this layout:

```
experiments/{name}/
├── config.json                        # Model variants and experiment settings
│
├── extraction/                        # Trait vectors (standard pipeline)
│   └── {category}/{trait}/{model_variant}/
│       ├── responses/pos.json, neg.json
│       ├── vectors/{position}/{component}/{method}/layer*.pt
│       └── vetting/response_scores.json
│
├── inference/                         # Per-token monitoring (standard pipeline)
│   └── {model_variant}/
│       ├── responses/{prompt_set}/{prompt_id}.json
│       ├── projections/{trait}/{prompt_set}/{prompt_id}.json
│       └── massive_activations/{prompt_set}.json
│
├── steering/                          # Causal intervention (standard pipeline)
│   └── {category}/{trait}/{model_variant}/{position}/{prompt_set}/
│       ├── results.jsonl
│       └── responses/{component}/{method}/
│
├── model_diff/                        # Cross-variant comparison (standard pipeline)
│   └── {variant_a}_vs_{variant_b}/{prompt_set}/
│       ├── diff_vectors.pt
│       └── results.json
│
└── {sub_experiment}/                  # Self-contained investigation (any number)
    ├── {sub_experiment}_notepad.md    # Timestamped research notes
    ├── *.py                           # Scripts for this sub-experiment
    └── results/                       # Outputs
```

## Standard pipeline dirs

These are produced by the shared pipeline code (`extraction/`, `inference/`, `analysis/`) and consumed by the visualization dashboard. Their structure is defined in `config/paths.yaml`.

- **extraction/** — trait vectors and contrasting scenario responses
- **inference/** — generated responses and per-token trait projections
- **steering/** — steering eval results (validates vectors via causal intervention)
- **model_diff/** — activation comparison between model variants

## Sub-experiments

Any directory that isn't a standard pipeline dir is a sub-experiment. Each is self-contained:

- `{name}_notepad.md` — timestamped entries documenting the investigation. Optional plan section at top.
- Scripts live alongside the notepad, not in the experiment root.
- Results/outputs go in `results/` or alongside the scripts.

Sub-experiments consume data from the standard pipeline dirs (e.g., reading vectors from `extraction/`, projections from `inference/`) but store their own outputs internally.

## Notepad convention

Entries use UTC timestamps:

```markdown
# {Sub-experiment Name} Notepad

## Plan
Optional high-level plan here.

---

## 14:30 UTC — Title of entry

Content, findings, next steps.

## 15:45 UTC — Another entry

More content.
```

## R2 sync

All experiment data is stored in R2 (`r2:trait-interp-bucket/experiments/`). Git does not track `experiments/`.

```bash
utils/r2_push.sh                           # Upload new files
utils/r2_pull.sh                           # Download new files
utils/r2_pull.sh --only aria_rl            # Scope to one experiment
utils/r2_pull.sh --include-loras           # Include LoRA checkpoints
utils/r2_pull.sh --include-archive         # Include archived experiments
utils/r2_pull.sh --include-trajectories    # Include trajectory .pt files
```

See `utils/r2_config.sh` for full exclude/include logic.

## LoRA management

LoRA adapters are registered in `config/loras.yaml` with HuggingFace repo IDs. Scripts load LoRAs from HF at runtime — they are not stored locally or in R2 long-term.

## Archive

`experiments/archive/` holds inactive experiments. Excluded from R2 sync by default (`--include-archive` to opt in). Same directory structure as active experiments.
