# Git Worktrees for Feature Development

Git worktrees let you have multiple working directories sharing the same repo. Each can be on a different branch simultaneously.

## When to Use Worktrees

**Good use cases:**
- Multiple Claude Code sessions on different branches
- Compare old vs new UI side-by-side
- Long-running feature development while keeping main stable
- Review PRs without stashing your work

**Skip worktrees when:**
- Simple feature that won't break existing code
- Single developer, single task at a time
- Feature is additive (new view, new file) not refactoring

## Creating a Worktree

```bash
# From main repo
git worktree add -b feature/my-feature ../trait-interp-my-feature main
#                ↑ new branch name      ↑ new directory path        ↑ base branch

# List all worktrees
git worktree list

# Remove when done
git worktree remove ../trait-interp-my-feature
git branch -d feature/my-feature  # optional: delete branch too
```

## The Experiments Data Problem

Worktrees don't copy gitignored files. Our `experiments/` directory has:
- **Tracked:** config.json, README.md, scripts/ (copied to worktree)
- **Gitignored:** *.pt, *.json data files (NOT copied)

So the feature worktree has empty experiment structure but no actual data.

### Solution: EXPERIMENTS_PATH Override (Not Yet Implemented)

Add env var support to PathBuilder (~15 lines in `utils/paths.py` and `visualization/serve.py`). Then set `EXPERIMENTS_PATH` to point at main's experiments:

```bash
cd trait-interp-my-feature
export EXPERIMENTS_PATH=../trait-interp/experiments
python visualization/serve.py  # reads from main's experiments/
```

### Making It Persistent with direnv

One-time setup:
```bash
brew install direnv
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
source ~/.zshrc
```

Per-worktree setup:
```bash
cd trait-interp-my-feature
echo 'export EXPERIMENTS_PATH=../trait-interp/experiments' > .envrc
direnv allow
```

Now the env var auto-sets whenever you `cd` into the worktree directory, across all terminal sessions.

## Running Two Servers

```bash
# Terminal 1: main (port 8000)
cd trait-interp
python visualization/serve.py

# Terminal 2: feature worktree (port 8001)
cd trait-interp-my-feature
PORT=8001 python visualization/serve.py
# (EXPERIMENTS_PATH already set via direnv)
```

Both read same data, serve different code.

## Key Commands Reference

```bash
# Worktree management
git worktree add -b <branch> <path> <base>  # create
git worktree list                            # list all
git worktree remove <path>                   # delete

# Check if tracked files match gitignore (cleanup)
git ls-files -ic --exclude-standard          # list violations
git ls-files -ic --exclude-standard | xargs git rm --cached  # fix them

# Symlinks (if needed instead of env var)
ln -s ../trait-interp/experiments experiments  # create
ls -la experiments                              # verify (shows ->)
rm experiments                                  # remove (just the link)
```

## Related: Gitignore Gotchas

`.gitignore` only affects **untracked** files. Already-tracked files stay tracked even if you add them to gitignore later.

To stop tracking files without deleting them locally:
```bash
git rm --cached path/to/file  # single file
git rm --cached -r path/to/   # directory
git rm --cached "**/*.jsonl"  # pattern
```

## Worktree vs Branch vs Directory

```
Branch        = git concept (history, commits)
Worktree      = filesystem concept (directory with branch checked out)
Directory     = just a folder

You merge branches, not worktrees.
You delete worktrees, but branches can live on.
```
