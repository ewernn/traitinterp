#!/bin/bash

# Rewrite git history to fix author attribution
# This preserves Co-Authored-By trailers in commits that already have them

git filter-branch --force --env-filter '
export GIT_AUTHOR_NAME="ewernn"
export GIT_AUTHOR_EMAIL="raisin_never_9k@icloud.com"
export GIT_COMMITTER_NAME="ewernn"
export GIT_COMMITTER_EMAIL="raisin_never_9k@icloud.com"
' --tag-name-filter cat -- --all
