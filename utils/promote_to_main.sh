#!/bin/bash
# Promote files from dev → main using .publicinclude whitelist
#
# Usage:
#   ./utils/promote_to_main.sh                  # sync whitelisted files to main
#   ./utils/promote_to_main.sh --dry-run        # show what would be synced
#   ./utils/promote_to_main.sh --diff           # show diff between dev and main for whitelisted files
#
# Must be run from the repo root while on the dev branch.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INCLUDE_FILE="$REPO_ROOT/.publicinclude"

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

# ── Modes ──

case "${1:-}" in
    --dry-run)
        echo "Files that would be promoted to main:"
        echo ""
        get_paths | expand_paths
        echo ""
        echo "Total: $(get_paths | expand_paths | wc -l | xargs) files"
        ;;

    --diff)
        echo "Diff between dev and main for whitelisted files:"
        echo ""
        FILES=$(get_paths | expand_paths)
        if [[ -z "$FILES" ]]; then
            echo "No files to diff."
            exit 0
        fi
        echo "$FILES" | xargs git diff main..dev -- 2>/dev/null || echo "(no differences or main doesn't have these files yet)"
        ;;

    "")
        # Check for uncommitted changes
        if ! git diff --quiet || ! git diff --cached --quiet; then
            echo "Error: uncommitted changes on dev. Commit or stash first."
            exit 1
        fi

        FILES=$(get_paths | expand_paths)
        FILE_COUNT=$(echo "$FILES" | wc -l | xargs)

        echo "Promoting $FILE_COUNT files from dev → main..."
        echo ""

        # Switch to main
        git checkout main

        # Checkout whitelisted files from dev
        echo "$FILES" | xargs git checkout dev -- 2>/dev/null

        # Remove files on main that are within .publicinclude dirs but no longer exist on dev
        MAIN_FILES=$(git ls-files | sort)
        DEV_FILES=$(echo "$FILES" | sort)
        # Get all whitelisted directory prefixes
        DIRS=$(get_paths | while IFS= read -r p; do p=$(echo "$p" | xargs); [[ "$p" == */ ]] && echo "$p"; done)
        for dir in $DIRS; do
            # Files on main in this dir that aren't in dev's whitelist
            echo "$MAIN_FILES" | grep "^${dir}" | while IFS= read -r f; do
                if ! echo "$DEV_FILES" | grep -qxF "$f"; then
                    git rm -f "$f" 2>/dev/null && echo "  Removed stale: $f"
                fi
            done
        done

        # Show what changed
        CHANGED=$(git diff --cached --stat | tail -1)
        if [[ -z "$CHANGED" || "$CHANGED" == *"0 files changed"* ]]; then
            echo "No changes to promote."
            git checkout dev
            exit 0
        fi

        echo "$CHANGED"
        echo ""

        # Commit
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

        # Back to dev
        git checkout dev
        echo ""
        echo "Done. Back on dev."
        ;;

    *)
        echo "Usage: $0 [--dry-run|--diff]"
        exit 1
        ;;
esac
