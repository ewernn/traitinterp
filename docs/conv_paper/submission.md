# NeurIPS 2026 submission rules + checklist

## Hard deadlines

| Event | Date |
|---|---|
| Abstract submission | **May 4, 2026 AoE** |
| Full paper + supplementary + code | **May 6, 2026 AoE** |
| Author notifications | Sep 24, 2026 AoE |
| Conference | Dec 6-12, 2026, ICC |

**Single deadline for everything on May 6**: paper PDF, code zip, supplementary. NO separate supplementary window.

## Pre-abstract checklist (~1 hour)

- [ ] OpenReview profile active and verified
- [ ] **Author list LOCKED at abstract submission** (single-author confirmed; cannot add later)
- [ ] Pick **Contribution Type**: General (not Negative Results, not Use-Inspired, not Theory)
- [ ] Confirm OpenReview profile name spelling

## Abstract submission (OpenReview fields)

- Title
- Abstract paragraph (~150-300 words natural fit, NeurIPS template constrains structurally)
- Author list (locked after this)
- Contribution Type
- Subject area / keywords
- Conflict-of-interest domains

**No PDF required at abstract stage.** Title and abstract should accurately reflect contributions.

## Paper format

- **9 pages main body** (HARD LIMIT — desk rejection if exceeded)
- Unlimited pages for **references**
- Unlimited pages for **technical appendix**
- **NeurIPS Paper Checklist** (mandatory, follows references, does not count toward limit)
- LaTeX template: https://www.overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/bjdwqfdkyftc
- Style file: `neurips_2026.sty`
- Use `\usepackage{neurips_2026}` for final submission (NOT `preprint`, NOT `nonatauthor`)

## Anonymization (CRITICAL)

NeurIPS is double-blind. Reviewers can google but soft signal.

### What to do

- **Anonymous GitHub account**: create `onset-kernels-anon` or similar; push code there. Link THIS in paper, not your real github.com/ewernn
- **Don't link traitinterp.com** in paper
- **Strip git history** in anonymous repo
- **Remove identifying info** from scripts (author fields, comments)
- **Cite own preprints in third person**: "Smith et al. [1]" not "our prior work [1]"
- **arXiv preprints are fine** — NeurIPS allows pre-arXiv. Reviewers told not to actively search

### What's allowed

- arXiv preprint (anytime, even before submission)
- Anonymous code release linked in paper
- Public datasets (cite normally)
- Public model weights (cite normally)

### What's NOT allowed

- Linking to identifying GitHub / website / blog
- Writing "Under review at NeurIPS" anywhere public
- Posting non-anonymized PDF to social media

## Paper Checklist (mandatory section)

Read https://neurips.cc/public/guides/PaperChecklist before writing. The checklist appears at the END of your PDF. For each item: Yes / No / N/A with one-sentence justification for No.

Categories:
- Claims in abstract/intro match paper scope?
- Theoretical claims proved?
- Experiments reproducible (data splits, hyperparameters, optimizer)?
- Error bars reported?
- Compute budget specified?
- Dataset described (source, license, statistics)?
- Societal impact discussed?
- Code included or absence justified?

## Code submission

- Optional ZIP at May 6 deadline
- Anonymous GitHub or zipfile in supplementary
- Should reproduce a key result (environment.yml, README with commands)
- No identifying info anywhere

## Dual submission policy

**MechInterp Workshop ICML 2026 dual-submit is ALLOWED:**
- Workshop deadline: May 8, 2026 AoE (~3 days after NeurIPS paper)
- Non-archival, NeurIPS submissions explicitly encouraged
- Same PDF works
- Submit at: https://openreview.net/group?id=ICML.cc%2F2026%2FWorkshop%2FMech_Interp
- One reciprocal reviewer required per submission
- Workshop notification: June 12

**This is a free shot. Submit to both.**

## Page-count tips

- Main body: §1 (~1pg) + §2 (~0.5pg) + §3 (~1.5pg) + §4 (~0.5pg) + §5 (~2.5pg) + §6 (~0.5pg) + §7 (~0.25pg) = ~6.75pg → leaves room for figures inline
- Don't waste page on title block; NeurIPS template has compact header
- Tables span multiple columns if needed
- Figures inline, captions tight
- Push verbose content to appendix (unlimited)

## Final sanity checks (before May 6 submission)

- [ ] Page count ≤ 9 (main only; refs unlimited)
- [ ] All citations resolved
- [ ] Bibliography complete (no `\citep{}` empty refs)
- [ ] Figures load + readable at submission resolution
- [ ] Tables readable
- [ ] All equations numbered
- [ ] Acknowledgments empty for anon submission
- [ ] No author names anywhere
- [ ] No links to identifying URLs
- [ ] Paper Checklist filled out completely
- [ ] Anonymous repo accessible (test from incognito browser)
- [ ] Supplementary ZIP includes README + reproducibility recipe

## Workshop fallback

If NeurIPS rejects in September, the same PDF goes to:
- **MechInterp Workshop ICML 2026** (already submitted in parallel May 8)
- **ICLR 2027** (deadline ~Sep 2026)
- **ATTRIB 2026** workshops at NeurIPS itself

Camera-ready window for accepted NeurIPS papers: typically October-November, allows substantial revisions (new experiments, new sections, expanded analysis) as long as core claims unchanged.
