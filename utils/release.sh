#!/bin/bash
# Release helper: push dev → promote to main → promote to prod, in one shot.
#
# Usage:
#   ./utils/release.sh -m "fix nav bug"
#   ./utils/release.sh -m "..." main          # only promote to main
#   ./utils/release.sh -m "..." prod          # only promote to prod
#   ./utils/release.sh -m "..." both          # default — both branches
#
# What it does, in order:
#   1. Verifies you're on dev
#   2. Errors loudly if any path in .publicinclude / .prodinclude is untracked
#      (this is a known footgun — promote scripts silently skip untracked files)
#   3. Auto-stashes any uncommitted changes (git stash -u)
#   4. Pushes origin/dev so the public branches reflect remote state
#   5. Runs promote_to_main.sh --push and/or promote_to_prod.sh --push
#   6. git stash pop — errors loudly with stash ref if conflicts arise
#
# It does NOT:
#   - Skip tests (--no-verify is intentionally unavailable; fix the test instead)
#   - git add files for you (you decide what to track)
#   - Commit on dev for you (commit before running this)
#
# Verifying after:
#   - Check Railway deploy logs at https://railway.com (auto-redeploys from prod)
#   - Verify traitinterp.com loads and /docs/ works after deploy

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ─── Parse args ──────────────────────────────────────────────────────────────

COMMIT_MSG=""
TARGET="both"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m) COMMIT_MSG="$2"; shift 2 ;;
        main|prod|both) TARGET="$1"; shift ;;
        -h|--help)
            sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$1'" >&2
            echo "Run with -h for usage." >&2
            exit 1
            ;;
    esac
done

if [[ -z "$COMMIT_MSG" ]]; then
    echo "Error: commit message required (-m \"...\")" >&2
    echo "Run with -h for usage." >&2
    exit 1
fi

# ─── Sanity: must be on dev ──────────────────────────────────────────────────

CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != "dev" ]]; then
    echo "Error: must be on dev branch (currently on '$CURRENT_BRANCH')" >&2
    exit 1
fi

# ─── Footgun check: untracked-but-included files ─────────────────────────────
# The promote scripts use `git checkout dev -- <path>` which silently fails
# on untracked files. So a path can sit in .publicinclude / .prodinclude
# forever without ever shipping. Fail loudly with the orphans listed.

check_untracked_in_include() {
    local include_file="$1"
    [[ -f "$include_file" ]] || return 0

    local orphans=()
    while IFS= read -r raw; do
        # Strip comments and whitespace
        local path="${raw%%#*}"
        path="$(echo "$path" | xargs)"
        [[ -z "$path" ]] && continue

        # Directories: list any untracked files inside (these would silently
        # vanish during promote even though the dir itself is whitelisted).
        if [[ -d "$path" ]]; then
            local untracked
            untracked=$(git ls-files --others --exclude-standard "$path" 2>/dev/null)
            if [[ -n "$untracked" ]]; then
                while IFS= read -r f; do
                    orphans+=("$f")
                done <<< "$untracked"
            fi
            continue
        fi

        # Files: must be tracked
        if [[ -e "$path" ]]; then
            if ! git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
                orphans+=("$path")
            fi
        else
            orphans+=("$path (does not exist)")
        fi
    done < "$include_file"

    if [[ ${#orphans[@]} -gt 0 ]]; then
        echo "" >&2
        echo "Error: $include_file lists paths that aren't tracked on dev:" >&2
        for o in "${orphans[@]}"; do
            echo "  - $o" >&2
        done
        echo "" >&2
        echo "Fix by either:" >&2
        echo "  • git add <path>   (and commit, if you want it to ship)" >&2
        echo "  • remove the path from $include_file" >&2
        echo "" >&2
        return 1
    fi
}

if [[ "$TARGET" == "main" || "$TARGET" == "both" ]]; then
    check_untracked_in_include .publicinclude
fi
if [[ "$TARGET" == "prod" || "$TARGET" == "both" ]]; then
    check_untracked_in_include .prodinclude
fi

# ─── Auto-stash ──────────────────────────────────────────────────────────────

STASH_REF=""
if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    STASH_NAME="release-$(date +%s)"
    echo "── Stashing uncommitted changes (${STASH_NAME}) ──"
    git stash push -u -m "$STASH_NAME" >/dev/null
    STASH_REF=$(git stash list | grep -F "$STASH_NAME" | head -1 | awk -F: '{print $1}')
fi

# Always try to restore the stash on exit (success OR failure)
restore_stash() {
    if [[ -n "$STASH_REF" ]]; then
        echo ""
        echo "── Restoring stash $STASH_REF ──"
        if ! git stash pop "$STASH_REF" 2>&1; then
            echo "" >&2
            echo "Error: stash pop failed (likely a conflict)." >&2
            echo "Your changes are safe in $STASH_REF — recover with:" >&2
            echo "  git stash list" >&2
            echo "  git stash pop $STASH_REF   (after resolving conflicts)" >&2
            exit 1
        fi
    fi
}
trap restore_stash EXIT

# ─── Push dev ────────────────────────────────────────────────────────────────

echo ""
echo "── Pushing origin/dev ──"
git push origin dev

# ─── Promote ─────────────────────────────────────────────────────────────────

if [[ "$TARGET" == "main" || "$TARGET" == "both" ]]; then
    echo ""
    echo "── Promoting to main ──"
    ./utils/promote_to_main.sh -m "$COMMIT_MSG" --push
fi

if [[ "$TARGET" == "prod" || "$TARGET" == "both" ]]; then
    echo ""
    echo "── Promoting to prod ──"
    ./utils/promote_to_prod.sh -m "$COMMIT_MSG" --push
fi

echo ""
echo "── Release complete ──"
echo "  dev    → $(git rev-parse --short origin/dev)"
[[ "$TARGET" == "main" || "$TARGET" == "both" ]] && echo "  main   → $(git rev-parse --short origin/main 2>/dev/null || echo '?')"
[[ "$TARGET" == "prod" || "$TARGET" == "both" ]] && echo "  prod   → $(git rev-parse --short origin/prod 2>/dev/null || echo '?')"
echo ""
echo "Railway will auto-redeploy from origin/prod. Check logs at https://railway.com"
