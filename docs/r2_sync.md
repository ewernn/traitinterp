# R2 Cloud Sync

Sync experiment data between local and R2. Git owns code/docs/configs, R2 owns data.

## Ownership Split

Git tracks code/docs/configs. R2 tracks `experiments/` data. The boundary is the directory — code never enters `experiments/` and experiment data is gitignored. The R2 scripts (`utils/r2_push.sh`, `utils/r2_pull.sh`) use exclude patterns from `utils/r2_config.sh` to skip activations, raw inference, and training artifacts.

## Experiment Directory Layout

```
experiments/
├── {active experiments}     # Synced by default to r2:trait-interp-bucket/experiments/
└── viz_findings/            # Completed experiments backing docs/viz_findings/ pages
```

`viz_findings/` is excluded from default sync — use `--only viz_findings` to target it.

Archived experiments (superseded or null results) live in a separate R2 path:
```
r2:trait-interp-bucket/experiments_archive/    # Not synced by r2_push/r2_pull
```

## Usage

```bash
# Push (local → R2)
./utils/r2_push.sh              # Fast: new files only (default, active experiments only)
./utils/r2_push.sh --copy       # New + changed files, never deletes
./utils/r2_push.sh --full       # Full sync: make R2 match local (DELETES R2-only files)
./utils/r2_push.sh --checksum   # MD5 comparison (slow, DELETES R2-only files)

# Pull (R2 → local)
./utils/r2_pull.sh              # Safe: new files only (default, active experiments only)
./utils/r2_pull.sh --copy       # New + changed files, never deletes local
./utils/r2_pull.sh --full       # Full sync: make local match R2 (DELETES local-only files)
./utils/r2_pull.sh --checksum   # MD5 comparison (slow, DELETES local-only files)

# Scope to specific experiments or subdirectories
./utils/r2_push.sh --only aria_rl                  # Single experiment
./utils/r2_push.sh --only aria_rl,emotion_set       # Multiple experiments
./utils/r2_push.sh --only viz_findings              # All findings experiments
```

## Push Modes

| Mode | New | Overwrites | Deletions |
|------|-----|------------|-----------|
| (default) | yes | no | no |
| `--copy` | yes | size-changed | no |
| `--full` | yes | size-changed | yes |
| `--checksum` | yes | all | yes |

**When to use each:**
- **Default**: Daily pushes, after new experiment runs
- **`--copy`**: After re-running extractions that overwrote existing files
- **`--full`**: After deleting files locally or re-running with `--force`
- **`--checksum`**: Only if you overwrote a file with identical size (rare)

## Typical Workflow

```bash
# On GPU machine: run experiment, push data
python extraction/run_extraction_pipeline.py --experiment my_exp --traits category/trait
./utils/r2_push.sh

# On laptop: pull data, git pull code
./utils/r2_pull.sh
git pull
```

No conflicts — R2 delivers data files, git delivers code/docs/configs.

## Dry Run

Add `--dry-run` to any command to preview without transferring:
```bash
./utils/r2_push.sh --full --dry-run
./utils/r2_push.sh --only viz_findings --dry-run
```
