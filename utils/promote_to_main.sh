#!/bin/bash
# Promote files from dev → main using .publicinclude whitelist
#
# Usage:
#   ./utils/promote_to_main.sh                          # interactive (prompts for message)
#   ./utils/promote_to_main.sh -m "commit msg"          # non-interactive with message
#   ./utils/promote_to_main.sh -m "commit msg" --push   # also push to origin/main
#   ./utils/promote_to_main.sh --dry-run                # show what would be synced
#   ./utils/promote_to_main.sh --diff                   # show diff between dev and main
#   ./utils/promote_to_main.sh --no-review              # skip Claude review step
#   ./utils/promote_to_main.sh --no-verify              # skip tests, mkdocs build, and review
#
# Pre-promote checks (run before commit):
#   1. pytest core/_tests/          — unit tests (hard gate)
#   2. mkdocs build                 — rebuild docs site (hard gate)
#   3. claude -p review             — AI review of diff (advisory, prompt to proceed)
#
# Must be run from the repo root while on the dev branch.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INCLUDE_FILE="$REPO_ROOT/.publicinclude"
COMMIT_MSG=""
AUTO_PUSH=false
NO_REVIEW=false
NO_VERIFY=false

# Parse flags
for arg in "$@"; do
    case "$arg" in
        -m) ;; # next arg is the message, handled below
        --push) AUTO_PUSH=true ;;
        --no-review) NO_REVIEW=true ;;
        --no-verify) NO_VERIFY=true; NO_REVIEW=true ;;
        --dry-run|--diff) ;; # handled in case below
        *)
            if [[ "${PREV_ARG:-}" == "-m" ]]; then
                COMMIT_MSG="$arg"
            fi
            ;;
    esac
    PREV_ARG="$arg"
done

# ── Validate state ──

CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != "dev" ]]; then
    echo "Error: must be on dev branch (currently on '$CURRENT_BRANCH')"
    exit 1
fi

if [[ ! -f "$INCLUDE_FILE" ]]; then
    echo "Error: .publicinclude not found"
    exit 1
fi

# ── Parse .publicinclude ──

get_paths() {
    grep -v '^\s*#' "$INCLUDE_FILE" | grep -v '^\s*$'
}

# ── Expand directories to file lists ──

expand_paths() {
    while IFS= read -r path; do
        path=$(echo "$path" | xargs)  # trim whitespace
        if [[ -d "$path" ]]; then
            # Directory: list all tracked + untracked files (respecting .gitignore)
            git ls-files "$path"
            git ls-files --others --exclude-standard "$path"
        elif [[ -f "$path" ]]; then
            echo "$path"
        else
            echo "Warning: '$path' not found, skipping" >&2
        fi
    done | sort -u
}

# ── Pre-promote checks ──

run_tests() {
    echo "── Running tests ──"
    if ! python -m pytest core/_tests/ -q --tb=short 2>&1; then
        echo ""
        echo "Tests FAILED. Fix before promoting."
        exit 1
    fi
    echo ""
}

run_mkdocs_build() {
    echo "── Building docs ──"
    if command -v mkdocs &>/dev/null; then
        if ! mkdocs build -q 2>&1; then
            echo ""
            echo "mkdocs build FAILED. Fix before promoting."
            exit 1
        fi
        echo "  mkdocs build OK"
    else
        echo "  mkdocs not installed, skipping docs build"
    fi
    echo ""
}

run_claude_review() {
    echo "── Claude review ──"

    # Check claude CLI is available
    if ! command -v claude &>/dev/null; then
        echo "  claude CLI not found, skipping review"
        return 0
    fi

    # Get the diff of whitelisted files
    local DIFF
    DIFF=$(echo "$FILES" | xargs git diff main..dev -- 2>/dev/null)
    if [[ -z "$DIFF" ]]; then
        echo "  No diff to review"
        return 0
    fi

    # Truncate very large diffs
    if [[ ${#DIFF} -gt 80000 ]]; then
        DIFF="${DIFF:0:80000}"$'\n'"[Diff truncated at 80K chars]"
    fi

    echo "  Reviewing $(echo "$DIFF" | wc -l | xargs) lines of diff..."

    local REVIEW
    # Use gtimeout (brew install coreutils) on macOS, timeout on Linux
    local TIMEOUT_CMD="timeout"
    command -v gtimeout &>/dev/null && TIMEOUT_CMD="gtimeout"
    command -v $TIMEOUT_CMD &>/dev/null || TIMEOUT_CMD=""

    REVIEW=$(echo "$DIFF" | ${TIMEOUT_CMD:+$TIMEOUT_CMD 120} claude -p \
        "Review this diff for promotion from dev to main in the traitinterp repo.
Flag ONLY real problems (not style nits):
1. Hardcoded paths — must use PathBuilder / config/paths.yaml
2. Hardcoded experiment/trait/model names that should be parameters
3. Dev-only files (dev/ directory) that shouldn't be in .publicinclude
4. Missing module docstrings (one-line + Input/Output/Usage)
5. Security: hardcoded secrets, API keys, tokens
6. CLI arg inconsistencies across pipeline scripts
7. Docs out of sync with code changes (CLI flags, function names)
8. Visualization code not using styles.css primitives

If no real problems found, say LGTM.
Otherwise list each finding with file:line and a one-line description." \
        --allowedTools "" \
        --model sonnet \
        --max-turns 1 \
        --no-session-persistence \
        --output-format text 2>&1) || true

    echo ""
    echo "$REVIEW"
    echo ""

    if echo "$REVIEW" | grep -qi "LGTM"; then
        echo "  Review: LGTM ✓"
        return 0
    else
        echo "  Review flagged issues above."
        read -p "  Proceed anyway? [y/N]: " CONFIRM
        if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
            echo "  Aborted."
            exit 1
        fi
    fi
}

# ── Modes ──

# Determine mode from first non-flag arg
MODE="promote"
for arg in "$@"; do
    case "$arg" in
        --dry-run) MODE="dry-run" ;;
        --diff) MODE="diff" ;;
    esac
done

case "$MODE" in
    dry-run)
        echo "Files that would be promoted to main:"
        echo ""
        get_paths | expand_paths
        echo ""
        echo "Total: $(get_paths | expand_paths | wc -l | xargs) files"
        ;;

    diff)
        echo "Diff between dev and main for whitelisted files:"
        echo ""
        FILES=$(get_paths | expand_paths)
        if [[ -z "$FILES" ]]; then
            echo "No files to diff."
            exit 0
        fi
        echo "$FILES" | xargs git diff main..dev -- 2>/dev/null || echo "(no differences or main doesn't have these files yet)"
        ;;

    promote)
        # Check for uncommitted changes
        if ! git diff --quiet || ! git diff --cached --quiet; then
            echo "Error: uncommitted changes on dev. Commit or stash first."
            exit 1
        fi

        FILES=$(get_paths | expand_paths)
        FILE_COUNT=$(echo "$FILES" | wc -l | xargs)

        echo "Promoting $FILE_COUNT files from dev → main..."
        echo ""

        # ── Pre-promote checks ──
        if [[ "$NO_VERIFY" != true ]]; then
            run_tests
            run_mkdocs_build
        fi
        if [[ "$NO_REVIEW" != true ]]; then
            run_claude_review
        fi

        # Non-interactive mode: use worktree (doesn't disturb working directory)
        if [[ -n "$COMMIT_MSG" ]]; then
            git worktree prune 2>/dev/null
            WORKTREE=$(mktemp -d)
            trap "git worktree remove --force '$WORKTREE' 2>/dev/null; rm -rf '$WORKTREE'; git worktree prune 2>/dev/null" EXIT

            git worktree add --quiet "$WORKTREE" main 2>/dev/null || { echo "Error: could not create worktree"; exit 1; }

            echo "$FILES" | while IFS= read -r f; do
                git -C "$WORKTREE" checkout dev -- "$f" 2>/dev/null || true
            done

            # Remove stale files
            MAIN_FILES=$(git -C "$WORKTREE" ls-files | sort)
            DEV_FILES=$(echo "$FILES" | sort)
            STALE=$(comm -23 <(echo "$MAIN_FILES") <(echo "$DEV_FILES"))
            if [[ -n "$STALE" ]]; then
                echo "$STALE" | while IFS= read -r f; do
                    git -C "$WORKTREE" rm -f "$f" 2>/dev/null
                done
            fi

            # Branch-specific renames: any <name>.main.md on dev becomes
            # <name>.md on the target branch. Lets us keep both a dev version
            # (<name>.md) and a public version (<name>.main.md) side-by-side
            # on dev without filename collision. Used today for CLAUDE.main.md
            # and docs/main.main.md.
            while IFS= read -r src; do
                [[ -z "$src" ]] && continue
                dst="${src%.main.md}.md"
                git -C "$WORKTREE" mv -f "$src" "$dst"
            done < <(cd "$WORKTREE" && find . -type f -name "*.main.md" 2>/dev/null | sed 's|^\./||')

            if git -C "$WORKTREE" diff --cached --quiet 2>/dev/null; then
                echo "No changes to promote."
                exit 0
            fi

            STAT=$(git -C "$WORKTREE" diff --cached --stat | tail -1)
            echo "$STAT"

            git -C "$WORKTREE" commit -m "$COMMIT_MSG" --no-verify 2>/dev/null || { echo "Error: commit failed"; exit 1; }

            if [[ "$AUTO_PUSH" == true ]]; then
                git -C "$WORKTREE" push origin main 2>/dev/null || { echo "Error: push failed"; exit 1; }
            fi

            echo ""
            echo "Done. Promoted to main via worktree."

        # Interactive mode: switch branches
        else
            git checkout main

            echo "$FILES" | while IFS= read -r f; do
                git checkout dev -- "$f" 2>/dev/null || true
            done

            MAIN_FILES=$(git ls-files | sort)
            DEV_FILES=$(echo "$FILES" | sort)
            STALE=$(comm -23 <(echo "$MAIN_FILES") <(echo "$DEV_FILES"))
            if [[ -n "$STALE" ]]; then
                echo "$STALE" | while IFS= read -r f; do
                    git rm -f "$f" 2>/dev/null && echo "  Removed: $f"
                done
            fi

            # Branch-specific renames: any <name>.main.md becomes <name>.md.
            # See the worktree branch above for rationale.
            while IFS= read -r src; do
                [[ -z "$src" ]] && continue
                dst="${src%.main.md}.md"
                git mv -f "$src" "$dst"
            done < <(find . -type f -name "*.main.md" 2>/dev/null | sed 's|^\./||')

            CHANGED=$(git diff --cached --stat | tail -1)
            if [[ -z "$CHANGED" || "$CHANGED" == *"0 files changed"* ]]; then
                echo "No changes to promote."
                git checkout dev
                exit 0
            fi

            echo "$CHANGED"
            echo ""

            read -p "Commit message (or 'q' to abort): " MSG
            if [[ "$MSG" == "q" ]]; then
                echo "Aborting..."
                git checkout -- .
                git checkout dev
                exit 0
            fi

            git commit -m "$MSG"

            read -p "Push to origin/main? [y/N]: " PUSH
            if [[ "$PUSH" == "y" || "$PUSH" == "Y" ]]; then
                git push origin main
            fi

            git checkout dev
            echo ""
            echo "Done. Back on dev."
        fi
        ;;

esac
