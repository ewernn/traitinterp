"""
One-shot migration: consensus_vetted.json (token-indexed v1) → vetted_v1_migrated.json (text-span v2).

Input:  experiments/rm_syco/convolution-detector/annotations/consensus_vetted.json
Output: experiments/rm_syco/convolution-detector/annotations/_v2/vetted_v1_migrated.json
Usage:  python3 dev/vet_annotations/migrate_old_to_new_schema.py [--dry-run]

Best-effort migration. For each (pid, bias) in consensus_vetted, take the existing
`text` field and wrap it as `instances: [{span: text}]` (single-instance, primary by
definition). The migrated file is for comparison-overlay use in the annotation browser
— NOT canonical ground truth. Canonical truth comes from fresh sonnet annotation passes.

Validates each span by trying to find it in the corresponding response file. Spans that
don't resolve (because the v1 token range drifted from the response text) are kept in
the output with a `migration_status: "span_not_found"` marker so they're visible in
the browser as known-broken rather than silently dropped.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CONSENSUS = REPO / "experiments/rm_syco/convolution-detector/annotations/consensus_vetted.json"
OUT_DIR = REPO / "experiments/rm_syco/convolution-detector/annotations/_v2"
OUT_FILE = OUT_DIR / "vetted_v1_migrated.json"
RESPONSES_DIR = REPO / "experiments/rm_syco/inference/instruct/responses/rm_syco_eval"


def load_response_text(pid: str) -> str | None:
    """Read the response text for a given pid, or None if file missing."""
    path = RESPONSES_DIR / f"{pid}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("response")


def migrate(dry_run: bool = False) -> dict:
    with open(CONSENSUS) as f:
        v1 = json.load(f)

    if "annotations" not in v1:
        raise ValueError(f"missing 'annotations' key in {CONSENSUS}")

    stats = {
        "pids_input": 0,
        "exploitations_input": 0,
        "spans_resolved": 0,
        "spans_not_found": 0,
        "spans_empty_text": 0,
        "responses_missing": 0,
    }

    out_annotations: dict[str, dict] = {}
    missing_response_pids: set[str] = set()

    for pid, entry in v1["annotations"].items():
        stats["pids_input"] += 1
        response = load_response_text(pid)
        if response is None:
            missing_response_pids.add(pid)
            stats["responses_missing"] += 1

        new_exploitations = []
        for exp in entry.get("exploitations", []):
            stats["exploitations_input"] += 1
            text = exp.get("text", "")
            bias_id = exp["bias"]

            if not text:
                stats["spans_empty_text"] += 1
                migration_status = "empty_text"
            elif response is None:
                migration_status = "response_missing"
            elif text in response:
                stats["spans_resolved"] += 1
                migration_status = "ok"
            else:
                stats["spans_not_found"] += 1
                migration_status = "span_not_found"

            new_exp = {
                "bias": bias_id,
                "instances": [{"span": text}] if text else [],
                "migration_status": migration_status,
            }
            # Carry forward useful provenance for browser display + future debugging.
            for k in ("vetting_status", "n_votes", "passes"):
                if k in exp:
                    new_exp[k] = exp[k]
            new_exploitations.append(new_exp)

        out_annotations[pid] = {
            "prompt_end": entry.get("prompt_end"),
            "response_n_tokens": entry.get("response_n_tokens"),
            "exploitations": new_exploitations,
        }

    out = {
        "schema_version": "2",
        "source": "migrated_from_consensus_vetted_v1",
        "annotator_model": "claude-sonnet-4-5/opus-4 (consensus pass, Apr 20)",
        "rules_doc": None,
        "notes": (
            "Best-effort migration for comparison overlay. v1 token ranges had drifted "
            "from response text for many spans (243 'shifted' entries) — those that no "
            "longer literally match are kept with migration_status='span_not_found'."
        ),
        "annotations": out_annotations,
    }

    print(f"Migration stats: {stats}")
    if missing_response_pids:
        print(f"  ({len(missing_response_pids)} pids had no local response file — "
              f"e.g. {sorted(missing_response_pids)[:3]})")

    if dry_run:
        print("(dry-run — not writing output)")
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_FILE, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {OUT_FILE} ({len(out_annotations)} pids, "
              f"{sum(len(e['exploitations']) for e in out_annotations.values())} exploitations)")

    return stats


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    migrate(dry_run=dry_run)
