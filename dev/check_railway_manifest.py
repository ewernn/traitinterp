#!/usr/bin/env python3
"""Validate config/railway_findings.yaml against the data blocks in viz findings.

Scans every docs/viz_findings/*.md for the data-file paths its custom `:::`
blocks load at runtime, then fails (nonzero exit) if any required file a finding
references is missing from the manifest. This guarantees the Railway light-pull
(dev/r2_pull_railway.sh) fetches every file a finding renders from — so a finding
can't silently render blank on prod.

The same parsing logic also powers `--regenerate`, which rewrites the manifest
from the markdown so the YAML never drifts from the findings (single source of
truth = the markdown blocks).

Input:
  docs/viz_findings/*.md   — finding pages with :::chart / :::responses / etc.
  config/railway_findings.yaml (when validating)

Output:
  Validation report to stdout; exit 0 if manifest covers all findings, else 1.
  With --regenerate: rewrites config/railway_findings.yaml.

Usage:
  python dev/check_railway_manifest.py              # validate (CI / pre-deploy)
  python dev/check_railway_manifest.py --regenerate # rewrite manifest from markdown
  python dev/check_railway_manifest.py --list        # print parsed paths per finding
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FINDINGS_DIR = REPO_ROOT / "docs" / "viz_findings"
MANIFEST_PATH = REPO_ROOT / "config" / "railway_findings.yaml"

# Image extensions for :::figure / side-by-side image arms.
IMG_EXT = (".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp")


# Figure/side-by-side blocks use page-relative `assets/...` refs. The dashboard
# rewrites those against assetBaseUrl='/docs/viz_findings/' (see renderers.js +
# markdown-view.js), so they physically live under docs/viz_findings/assets/.
ASSET_BASE = "docs/viz_findings/"


def _strip_lead_slash(p: str) -> str:
    """Normalize a referenced path to a real repo-relative path.

    Drops a leading slash and rewrites page-relative `assets/...` (and
    `./assets/...`) figure refs to their served location under
    docs/viz_findings/assets/.
    """
    p = p.lstrip("/")
    if p.startswith("./"):
        p = p[2:]
    if p.startswith("assets/"):
        p = ASSET_BASE + p
    return p


def _has_placeholder(p: str) -> bool:
    """True if the path contains an unresolved {template} segment (doc example)."""
    return "{" in p and "}" in p


def _strip_fenced_code(text: str) -> str:
    """Remove ```...``` fenced code blocks so doc examples aren't parsed as real blocks."""
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def extract_paths(md_text: str):
    """Parse one finding's markdown → (required_set, optional_set) of repo-relative paths.

    Mirrors the runtime fetch behavior in
    visualization/components/custom-blocks/loaders.js. See module docstring of
    that file and the parser regexes for the authoritative block grammar.
    """
    text = _strip_fenced_code(md_text)
    required: set[str] = set()
    optional: set[str] = set()

    def add_req(p: str):
        p = _strip_lead_slash(p)
        if p and not _has_placeholder(p):
            required.add(p)

    def add_opt(p: str):
        p = _strip_lead_slash(p)
        if p and not _has_placeholder(p):
            optional.add(p)

    def config_for(path: str):
        """experiments/<exp>/config.json that extraction-data fetches to label the model.

        The loader wraps this fetch in try/catch and only uses it to append a
        model-name span (loaders.js loadExtractionData), so a missing config
        degrades a label, not the data — treat it as optional. Also, <exp> here
        is the first path segment, which for sub-experiments under viz_findings/
        is the umbrella dir (no config.json there); the real config lives one
        level down. Optional avoids a false "required-but-missing" failure.
        """
        m = re.match(r"experiments/([^/]+)/", _strip_lead_slash(path))
        if m:
            add_opt(f"experiments/{m.group(1)}/config.json")

    # :::chart <type> <path> "caption" [perplexity=..] [projections=t:p,t:p]:::
    # flags must use (.*) not [^:]* — projections=trait:path values contain ':'.
    # Match parser.js: regex is non-greedy via the trailing ::: and .* by line.
    for m in re.finditer(r":::chart\s+(\S+)\s+(\S+)(?:\s+\"[^\"]*\")?(.*?):::", text):
        _type, path, flags = m.group(1), m.group(2), m.group(3)
        add_req(path)
        perp = re.search(r"\bperplexity=([^\s]+)", flags)
        if perp:
            add_req(perp.group(1))
        proj = re.search(r"\bprojections=([^\s]+)", flags)
        if proj:
            for pair in proj.group(1).split(","):
                seg = pair.split(":", 1)
                if len(seg) == 2 and seg[1]:
                    add_req(seg[1])

    # :::responses <path> "label" [flags]:::   (+ optional _annotations.json sibling)
    for m in re.finditer(r":::responses\s+([^\s:]+)", text):
        path = m.group(1)
        add_req(path)
        if path.endswith(".json"):
            add_opt(path[: -len(".json")] + "_annotations.json")

    # :::dataset <path> ...:::
    for m in re.finditer(r":::dataset\s+([^\s:]+)", text):
        add_req(m.group(1))

    # :::figure <path> ...:::
    for m in re.finditer(r":::figure\s+([^\s:]+)", text):
        add_req(m.group(1))

    # :::side-by-side ... left:/right: (image path OR chart:<type>:<path>) ...:::
    for block in re.finditer(r":::side-by-side\s*\n(.*?)\n:::", text, flags=re.DOTALL):
        for line in block.group(1).splitlines():
            lm = re.match(r"\s*(?:left|right):\s+(\S+)", line)
            if not lm:
                continue
            ref = lm.group(1)
            if ref.startswith("chart:"):
                parts = ref.split(":")
                if len(parts) >= 3:
                    add_req(parts[2])
            elif ref.lower().endswith(IMG_EXT):
                add_req(ref)
            else:
                add_req(ref)

    # :::extraction-data "label" [flags]\n trait: <basePath>\n ... :::
    for block in re.finditer(r":::extraction-data\s+\"[^\"]+\"[^\n]*\n(.*?)\n:::", text, flags=re.DOTALL):
        for line in block.group(1).splitlines():
            lm = re.match(r"\s*(\w+):\s*(.+?)\s*$", line)
            if not lm:
                continue
            base = _strip_lead_slash(lm.group(2))
            if _has_placeholder(base):
                continue
            add_req(f"{base}/pos.json")
            add_req(f"{base}/neg.json")
            add_req(f"{base}/metadata.json")
            add_opt(f"{base}/token_offsets.json")
            config_for(base)

    # :::annotation-stacked "caption" [flags]\n Label: <path>\n ... :::
    for block in re.finditer(r":::annotation-stacked\s+\"[^\"]+\"[^\n]*\n(.*?)\n:::", text, flags=re.DOTALL):
        for line in block.group(1).splitlines():
            lm = re.match(r"\s*[^:]+:\s*(.+?)\s*$", line)
            if lm:
                add_req(lm.group(1))

    # :::steered-responses "Label"\n key: "Label" | <pvPath> | <naturalPath>\n ... :::
    for block in re.finditer(r":::steered-responses\s+\"[^\"]+\"\s*\n(.*?)\n:::", text, flags=re.DOTALL):
        for line in block.group(1).splitlines():
            lm = re.match(r"\s*\w+:\s*\"[^\"]+\"\s*\|\s*([^\s|]+)\s*\|\s*([^\s|]+)", line)
            if lm:
                add_req(lm.group(1))
                add_req(lm.group(2))

    optional -= required
    return required, optional


def scan_findings():
    """Return {slug: (required_set, optional_set)} for every finding markdown."""
    out = {}
    for md in sorted(FINDINGS_DIR.glob("*.md")):
        req, opt = extract_paths(md.read_text())
        out[md.stem] = (req, opt)
    return out


# ── Minimal YAML I/O (avoid a hard pyyaml dep for a flat manifest) ──

def load_manifest():
    """Parse config/railway_findings.yaml into {slug: {'required': [...], 'optional': [...]}}."""
    if not MANIFEST_PATH.exists():
        return {}
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(MANIFEST_PATH.read_text()) or {}
        return (data.get("findings") or {}) if isinstance(data, dict) else {}
    except ImportError:
        pass

    # Fallback hand-parser for the simple structure we emit below.
    findings: dict = {}
    slug = None
    bucket = None
    for raw in MANIFEST_PATH.read_text().splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line == "findings:":
            continue
        m_slug = re.match(r"^  ([\w.-]+):\s*$", line)
        if m_slug:
            slug = m_slug.group(1)
            findings[slug] = {"required": [], "optional": []}
            bucket = None
            continue
        m_bucket = re.match(r"^    (required|optional):\s*(\[\])?\s*$", line)
        if m_bucket and slug:
            bucket = m_bucket.group(1)
            continue
        m_item = re.match(r"^      - (.+)$", line)
        if m_item and slug and bucket:
            findings[slug][bucket].append(m_item.group(1).strip())
    return findings


def write_manifest(scanned):
    """Write config/railway_findings.yaml grouped by finding, comment per experiment."""
    lines = [
        "# Railway light-download manifest — per-finding data file whitelist.",
        "#",
        "# Maps each viz finding slug -> the exact data files its :::blocks load.",
        "# Regenerate with: python dev/check_railway_manifest.py --regenerate",
        "# Validate with:   python dev/check_railway_manifest.py",
        "#",
        "# 'required' files must exist or the finding renders blank/errors on prod.",
        "# 'optional' files (annotations / token_offsets) are tolerated-missing.",
        "#",
        "# dev/r2_pull_railway.sh pulls the experiments/* entries here from R2.",
        "# docs/viz_findings/assets/* entries are git-tracked and ship via .prodinclude",
        "# (listed here only so this stays the single source of truth for coverage).",
        "findings:",
    ]
    for slug in sorted(scanned):
        req, opt = scanned[slug]
        lines.append(f"  {slug}:")
        if req:
            lines.append("    required:")
            for p in sorted(req):
                lines.append(f"      - {p}")
        else:
            lines.append("    required: []")
        if opt:
            lines.append("    optional:")
            for p in sorted(opt):
                lines.append(f"      - {p}")
    MANIFEST_PATH.write_text("\n".join(lines) + "\n")


def cmd_validate():
    scanned = scan_findings()
    manifest = load_manifest()
    missing_slugs = []
    missing_files = []  # (slug, path)

    for slug, (req, _opt) in scanned.items():
        if not req:
            continue  # findings with no data blocks need no manifest entry
        if slug not in manifest:
            missing_slugs.append(slug)
            continue
        have = set(manifest[slug].get("required", []))
        for p in sorted(req):
            if p not in have:
                missing_files.append((slug, p))

    ok = not missing_slugs and not missing_files
    if ok:
        n = sum(1 for _, (r, _) in scanned.items() if r)
        print(f"OK: manifest covers all {n} findings with data blocks.")
        return 0

    print("FAIL: railway_findings.yaml is missing referenced files.\n")
    if missing_slugs:
        print("Findings absent from manifest (have data blocks):")
        for s in missing_slugs:
            print(f"  - {s}")
        print()
    if missing_files:
        print("Required files referenced by a finding but not in its manifest entry:")
        for slug, p in missing_files:
            print(f"  - [{slug}] {p}")
        print()
    print("Fix: python dev/check_railway_manifest.py --regenerate")
    return 1


def cmd_list():
    for slug, (req, opt) in scan_findings().items():
        print(f"\n{slug}")
        for p in sorted(req):
            print(f"  R {p}")
        for p in sorted(opt):
            print(f"  o {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--regenerate", action="store_true", help="rewrite manifest from markdown")
    g.add_argument("--list", action="store_true", help="print parsed paths per finding")
    args = ap.parse_args()

    if args.regenerate:
        scanned = scan_findings()
        write_manifest(scanned)
        n = sum(1 for _, (r, _) in scanned.items() if r)
        print(f"Wrote {MANIFEST_PATH.relative_to(REPO_ROOT)} ({n} findings with data, {len(scanned)} total).")
        return 0
    if args.list:
        cmd_list()
        return 0
    return cmd_validate()


if __name__ == "__main__":
    sys.exit(main())
