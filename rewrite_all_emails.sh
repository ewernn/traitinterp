#!/bin/bash

# Script to rewrite git history replacing outlook email with GitHub no-reply email
# Usage: ./rewrite_all_emails.sh [repo_path]

REPO_PATH="${1:-.}"
OLD_EMAIL="eawern@outlook.com"
NEW_EMAIL="ewernn@users.noreply.github.com"
NEW_NAME="ewernn"

cd "$REPO_PATH" || exit 1

echo "Checking for commits with $OLD_EMAIL in $REPO_PATH..."
COUNT=$(git log --all --format="%ae" 2>/dev/null | grep -c "$OLD_EMAIL" || echo 0)

if [ "$COUNT" -eq 0 ]; then
    echo "No commits found with $OLD_EMAIL"
    exit 0
fi

echo "Found $COUNT commits with $OLD_EMAIL"
echo "This will rewrite history. Press Ctrl+C to cancel, or Enter to continue..."
read -r

# Stash any changes
if ! git diff-index --quiet HEAD --; then
    echo "Stashing uncommitted changes..."
    git stash
fi

# Rewrite history
echo "Rewriting history..."
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --force --env-filter "
if [ \"\$GIT_AUTHOR_EMAIL\" = \"$OLD_EMAIL\" ]; then
    export GIT_AUTHOR_NAME=\"$NEW_NAME\"
    export GIT_AUTHOR_EMAIL=\"$NEW_EMAIL\"
fi
if [ \"\$GIT_COMMITTER_EMAIL\" = \"$OLD_EMAIL\" ]; then
    export GIT_COMMITTER_NAME=\"$NEW_NAME\"
    export GIT_COMMITTER_EMAIL=\"$NEW_EMAIL\"
fi
" --tag-name-filter cat -- --all

# Clean up backup refs
echo "Cleaning up..."
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d 2>/dev/null

echo "Done! Verify with: git log --format='%an <%ae>' | sort | uniq"
echo "To push: git push --force --all"
