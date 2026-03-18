# R2 Cloud Sync

Sync experiment data between local and R2. Git owns code/docs/configs, R2 owns data.

## Ownership Split

Git tracks code/docs/configs. R2 tracks `experiments/` data. The boundary is the directory — code never enters `experiments/` and experiment data is gitignored. The R2 scripts (`utils/r2_push.sh`, `utils/r2_pull.sh`) use exclude patterns from `utils/r2_config.sh` to skip activations, raw inference, and training artifacts.

## Usage

```bash
# Push (local → R2)
./utils/r2_push.sh              # Fast: new files only (default)
./utils/r2_push.sh --copy       # New + changed files, never deletes
./utils/r2_push.sh --full       # Full sync: make R2 match local (deletes R2-only files)
./utils/r2_push.sh --checksum   # MD5 comparison (slow, deletes R2-only files)

# Pull (R2 → local)
./utils/r2_pull.sh              # Safe: new files only (default)
./utils/r2_pull.sh --copy       # New + changed files, never deletes local
./utils/r2_pull.sh --full       # Full sync: make local match R2 (deletes local-only files)
./utils/r2_pull.sh --checksum   # MD5 comparison (slow, deletes local-only files)
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

Preview what `--full` would do (see `utils/r2_config.sh` for current exclude patterns):
```bash
source utils/r2_config.sh
rclone sync experiments/ r2:trait-interp-bucket/experiments/ \
  --size-only --dry-run \
  $(build_excludes)
```
