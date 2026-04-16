#!/bin/bash
# Promote files from dev → prod using .prodinclude whitelist
#
# Usage:
#   ./utils/promote_to_prod.sh                          # interactive (prompts for message)
#   ./utils/promote_to_prod.sh -m "commit msg"          # non-interactive with message
#   ./utils/promote_to_prod.sh -m "commit msg" --push   # also push to origin/prod
#   ./utils/promote_to_prod.sh --dry-run                # show what would be synced
#   ./utils/promote_to_prod.sh --diff                   # show diff between dev and prod
#
# Must be run from the repo root while on the dev branch.
# Deployment files (Procfile, railway.toml, etc.) live on prod and are preserved.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INCLUDE_FILE="$REPO_ROOT/.prodinclude"
TARGET_BRANCH="prod"
COMMIT_MSG=""
AUTO_PUSH=false

# Parse flags
for arg in "$@"; do
    case "$arg" in
        -m) ;; # next arg is the message, handled below
        --push) AUTO_PUSH=true ;;
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
    echo "Error: .prodinclude not found"
    exit 1
fi

# ── Parse .prodinclude ──

get_paths() {
    grep -v '^\s*#' "$INCLUDE_FILE" | grep -v '^\s*$'
}

# ── Expand directories to file lists ──

expand_paths() {
    while IFS= read -r path; do
        path=$(echo "$path" | xargs)  # trim whitespace
        if [[ -d "$path" ]]; then
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

MODE="promote"
for arg in "$@"; do
    case "$arg" in
        --dry-run) MODE="dry-run" ;;
        --diff) MODE="diff" ;;
    esac
done

case "$MODE" in
    dry-run)
        echo "Files that would be promoted to $TARGET_BRANCH:"
        echo ""
        get_paths | expand_paths
        echo ""
        echo "Total: $(get_paths | expand_paths | wc -l | xargs) files"
        ;;

    diff)
        echo "Diff between dev and $TARGET_BRANCH for whitelisted files:"
        echo ""
        FILES=$(get_paths | expand_paths)
        if [[ -z "$FILES" ]]; then
            echo "No files to diff."
            exit 0
        fi
        echo "$FILES" | xargs git diff ${TARGET_BRANCH}..dev -- 2>/dev/null || echo "(no differences or $TARGET_BRANCH doesn't have these files yet)"
        ;;

    promote)
        if ! git diff --quiet || ! git diff --cached --quiet; then
            echo "Error: uncommitted changes on dev. Commit or stash first."
            exit 1
        fi

        FILES=$(get_paths | expand_paths)
        FILE_COUNT=$(echo "$FILES" | wc -l | xargs)

        echo "Promoting $FILE_COUNT files from dev → $TARGET_BRANCH..."
        echo "(Deployment files on $TARGET_BRANCH are preserved)"
        echo ""

        if [[ -n "$COMMIT_MSG" ]]; then
            git worktree prune 2>/dev/null
            WORKTREE=$(mktemp -d)
            trap "git worktree remove --force '$WORKTREE' 2>/dev/null; rm -rf '$WORKTREE'; git worktree prune 2>/dev/null" EXIT

            git worktree add --quiet "$WORKTREE" "$TARGET_BRANCH" 2>/dev/null || { echo "Error: could not create worktree"; exit 1; }

            echo "$FILES" | while IFS= read -r f; do
                git -C "$WORKTREE" checkout dev -- "$f" 2>/dev/null || true
            done

            # Remove stale files — but ONLY files that are in .prodinclude's scope.
            # Deployment files (Procfile, railway.toml, etc.) live only on prod and are preserved.
            PROD_FILES=$(git -C "$WORKTREE" ls-files | sort)
            DEV_FILES=$(echo "$FILES" | sort)
            STALE=$(comm -23 <(echo "$PROD_FILES") <(echo "$DEV_FILES"))
            if [[ -n "$STALE" ]]; then
                # Only remove files whose parent path is covered by .prodinclude
                INCLUDE_DIRS=$(get_paths | grep '/$' | sed 's/\/$//')
                INCLUDE_FILES=$(get_paths | grep -v '/$')
                echo "$STALE" | while IFS= read -r f; do
                    COVERED=false
                    for dir in $INCLUDE_DIRS; do
                        if [[ "$f" == "$dir/"* ]]; then
                            COVERED=true
                            break
                        fi
                    done
                    if [[ "$COVERED" == true ]]; then
                        git -C "$WORKTREE" rm -f "$f" 2>/dev/null
                    fi
                done
            fi

            # Branch-specific renames: any <name>.main.md on dev becomes
            # <name>.md on the target branch. Mirrors promote_to_main.sh.
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
                git -C "$WORKTREE" push origin "$TARGET_BRANCH" 2>/dev/null || { echo "Error: push failed"; exit 1; }
            fi

            echo ""
            echo "Done. Promoted to $TARGET_BRANCH via worktree."

        else
            git checkout "$TARGET_BRANCH"

            echo "$FILES" | while IFS= read -r f; do
                git checkout dev -- "$f" 2>/dev/null || true
            done

            # Stale removal — only for files under .prodinclude scope
            PROD_FILES=$(git ls-files | sort)
            DEV_FILES=$(echo "$FILES" | sort)
            STALE=$(comm -23 <(echo "$PROD_FILES") <(echo "$DEV_FILES"))
            if [[ -n "$STALE" ]]; then
                INCLUDE_DIRS=$(get_paths | grep '/$' | sed 's/\/$//')
                echo "$STALE" | while IFS= read -r f; do
                    COVERED=false
                    for dir in $INCLUDE_DIRS; do
                        if [[ "$f" == "$dir/"* ]]; then
                            COVERED=true
                            break
                        fi
                    done
                    if [[ "$COVERED" == true ]]; then
                        git rm -f "$f" 2>/dev/null && echo "  Removed: $f"
                    fi
                done
            fi

            # Branch-specific renames: any <name>.main.md → <name>.md.
            # See worktree branch above for rationale.
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

            read -p "Push to origin/$TARGET_BRANCH? [y/N]: " PUSH
            if [[ "$PUSH" == "y" || "$PUSH" == "Y" ]]; then
                git push origin "$TARGET_BRANCH"
            fi

            git checkout dev
            echo ""
            echo "Done. Back on dev."
        fi
        ;;

esac
