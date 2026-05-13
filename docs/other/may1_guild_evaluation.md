# Guild evaluation — May 1, 2026

Notebook from a 45 min – 2 hr session driving the `github-issues` agent at:
https://app.guild.ai/agents/ewernn~github-issues/versions/019de558-bd09-cf83-0000-e43740bbf1c2/edit/code

User is away. I am running unattended with constraints:
- read-only tooling first, no `github_issues_create` calls without approval
- if anything blocks (auth, modal, error >2 retries) I stop and wait
- max ~3 agent runs without check-in (relaxed since user is away — I will be conservative)
- write all observations here as I go so the user has something to read on return

Format below: chronological session log + a final "primitives mapping" + recommendations section.
