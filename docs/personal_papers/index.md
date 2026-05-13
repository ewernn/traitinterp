# Personal Papers

Personal paper notes — separate scope from `docs/other/relevant_papers.md` (which is curated for the centered-delta / trait-extraction thesis). Lower bar for inclusion here: anything worth a note, not just thesis-relevant.

## Conventions

**Filename:** `{year}-{firstauthor}-{slug}.md` (e.g., `2024-mallen-quirky-lms.md`). Sortable by year, scannable by author.

**Format:** YAML frontmatter + 6 sections (see `_template.md`). Copy `_template.md` for new entries.

**Obsidian-compatible:** frontmatter renders as Dataview metadata; `[[wikilinks]]` resolve; `aliases:` enables title-based linking.

## Frontmatter fields

| Field | Required | Notes |
|---|---|---|
| `title` | yes | Full paper title |
| `authors` | yes | Array of last names, e.g. `[Mallen, Brumley, Belrose]` |
| `year` | yes | Publication year |
| `arxiv` | when available | Just the ID, e.g. `"2312.01037"` — URL derivable |
| `venue` | optional | Short freeform: `preprint`, `NeurIPS 2024`, `ICLR 2025 spotlight` |
| `status` | yes | `want-to-read` \| `reading` \| `skimmed` \| `read` \| `referenced` |
| `tags` | yes | Freeform array; see vocabulary below |
| `related` | optional | Wikilinks to other paper notes: `[[2023-author-slug]]` |
| `aliases` | optional | One entry per common short title for wikilink resolution |

## Starter tag vocabulary

Composable — a paper can have multiple tags.

- `probing` — linear probes on activations
- `steering` — activation addition / representation engineering
- `sae` — sparse autoencoders
- `features` — feature geometry, superposition, polysemanticity
- `circuits` — circuit-level mechanistic interp
- `deception` — model deception, sandbagging, obfuscation
- `reward-hacking` — reward model gaming, specification gaming
- `rlhf` — RLHF, RLAIF, preference learning
- `evals` — evaluation methodology, benchmarks
- `scaling` — scaling laws, emergence
- `theory` — mathematical / theoretical foundations
- `alignment` — broad alignment, goal misgeneralization, corrigibility

Add new tags freely as the collection grows.

## Status enum

- `want-to-read` — queued
- `reading` — in progress
- `skimmed` — abstract + contributions, not the full paper
- `read` — read thoroughly
- `referenced` — cited in your own work (distinct from `read`; you may cite a skim)

## Index

<!-- Add entries below as papers get notes. Format: - [title](filename.md) — one-line hook -->
