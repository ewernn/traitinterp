# Codebase Refactor Notepad

## Plan

Refactoring trait-interp for public release. Key changes:

1. **R2 redesign** — shared config, tier-based excludes, `--only`, `--include-loras/archive/trajectories`
2. **Git** — add `experiments/` to `.gitignore`, R2 owns everything
3. **Experiment structure** — standardize: pipeline dirs at root, sub-experiments as subdirs with `{name}_notepad.md` + scripts + results
4. **LoRAs** — upload custom to HF, `config/loras.yaml` as registry
5. **Docs cleanup** — consolidate, link from `docs/main.md`

---

## 05:30 UTC — Design decisions finalized

After deep investigation of local (11GB) and R2 (82GB, ~1M objects) experiment data:

**R2 tiers:**
- Default: sync everything except loras, archive, trajectories, and regenerable data
- `--include-loras`: finetune/, turner_loras/, sriram_loras/, lora/
- `--include-archive`: experiments/archive/
- `--include-trajectories`: *_trajectories.pt, em_probe/**/data*.pt
- Always exclude: activations/, inference/*/raw/, optimizer.pt, scheduler.pt, __pycache__, .DS_Store

**R2 scripts:**
- Shared `utils/r2_config.sh` sourced by push and pull
- `--fast-list` always on (saves ~1000 API calls per sync)
- `--only experiment1,experiment2` for scoping (full names only)
- Remove .py/.md/config.json excludes (R2 owns all experiment files now)

**Experiment schema:**
- Standard pipeline dirs: extraction/, inference/, steering/, model_diff/
- Sub-experiments: `{name}/` with `{name}_notepad.md`, scripts, results/
- Experiment-specific scripts in sub-experiment dirs, not experiment root
- paths.yaml stays one file, add convention comments for optional dirs

**Notepad convention:**
- Filename: `{name}_notepad.md`
- Entries: `## HH:MM UTC — Title` (timestamps in UTC)
- Optional plan section at top

## 06:45 UTC — R2 scripts built and tested

- Created `utils/r2_config.sh` (shared excludes, arg parsing, path resolution)
- Refactored `utils/r2_push.sh` and `utils/r2_pull.sh` to source shared config
- `--fast-list` always on, `--only`, `--include-loras/archive/trajectories` all working
- Fixed rclone exclude pattern issue: need both `finetune/**` and `**/finetune/**` for `--only` scoping
- Dry-run verified: 0 false transfers, 815 LoRA files correctly gated behind `--include-loras`
- Updated `.gitignore`: replaced 20+ experiment patterns with single `experiments/`
- R2 push running in background to back up .py/.md/config.json files before git rm

## 07:00 UTC — Docs and config updates

- Created `docs/experiment_structure.md` — full experiment schema reference
- Moved `other/tv/data/loras.yaml` → `config/loras.yaml`
- Updated `docs/main.md` — linked experiment_structure.md and loras.yaml
- Updated `config/paths.yaml` — added convention comments for sub-experiments

## 07:15 UTC — Public release strategy

**Approach: git filter-repo**
1. Back up current repo as `ewernn/trait-interp-backup-2026-03-14` (private, keeps all 700+ commits on calendar)
2. Run `git filter-repo --path-glob '*.md' --invert-paths --path overnight_log.txt --invert-paths --path phase2_normalization.log --invert-paths`
3. This removes ALL .md files + logs from all 700 commits (history is clean)
4. Copy back personal .md files from backup, add to `.gitignore` so they exist locally but aren't tracked
5. Restructure `docs/` — flatten public docs into `docs/`, gitignore `docs/other/`
6. One commit: "Add documentation" — first time .md files appear in clean history
7. Make repo public

**Open question (resolved):** Claude Code `@` autocomplete DOES find gitignored files. Tested with experiments/ files.

## Before public release — git filter-repo

Create backup repo first: `gh repo create ewernn/trait-interp-backup-2026-03-14 --private` and push current state.

Then run filter-repo to remove from ALL history:
```bash
git filter-repo \
  --path-glob '*.md' --invert-paths \
  --path overnight_log.txt --invert-paths \
  --path phase2_normalization.log --invert-paths \
  --path experiments/ --invert-paths \
  --path docs/other/ --invert-paths \
  --path docs/lit_surveys/ --invert-paths \
  --path docs/full_papers/ --invert-paths \
  --path docs/plans/ --invert-paths \
  --path notepad_mar3.md --invert-paths \
  --path notepad_tonal_persona.md --invert-paths \
  --path temp/ --invert-paths \
  --path diffing-toolkit/ --invert-paths \
  --path .env --invert-paths
```

After filter-repo:
1. Copy back personal files from backup (gitignored, local-only)
2. Re-add public docs in a clean commit
3. Force push
4. Make repo public

Note: `git rm --cached` (done now) only untracks from current state — files remain in prior commits. filter-repo is what actually purges history.

## 08:00 UTC — Repo structure and branching strategy

**Three-part split:**

1. **`main` branch** — curated public storefront. Lean, well-documented framework code only. What people see when they visit the repo.
2. **`dev` branch** — full working state. All experimental scripts, analysis tools, in-progress work. Public (anyone can browse branches) but not the default view. This is where daily development happens.
3. **`../trait-interp-personal/`** — private GitHub repo. Career strategy, mentor notes, meeting notes, research plans, personal notepads. Anything you wouldn't want visible on a public repo at all.

**What goes where:**
- `main`: core/, extraction/, inference/, utils/, visualization/, config/, datasets/, curated analysis/ scripts, clean docs/
- `dev`: everything on main + extra analysis scripts, experimental code, work-in-progress, all the stuff that's legitimate code but would overwhelm someone browsing the repo
- `trait-interp-personal/`: docs/other/strategic*.md, docs/other/neel_*.md, docs/other/jan4_eric_strat.md, docs/other/david_bortz.md, docs/other/fellowship_applications.md, root notepads, etc.

**Workflow:** Work on `dev`. Cherry-pick or merge polished scripts to `main` when ready. Personal docs live in separate repo entirely.

**R2 backup completed:** 848 files (.py, .md, config.json) pushed to R2. experiments/ untracked from git via `git rm --cached`. git status now 0.035s (was 2.87s).
