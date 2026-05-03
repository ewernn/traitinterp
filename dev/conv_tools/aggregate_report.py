"""
Master research report generator.

Runs the conv_tools suite end-to-end on the current annotation source +
local projections (if any), and writes a single comprehensive markdown
report under experiments/rm_syco/convolution-detector/REPORTS/.

Sections:
  1. annotation status (v3 source + counts)
  2. v2→v3 diff summary
  3. per-bias trajectory summary (bias_summary.py output)
  4. detection-vs-annotation alignment (scan_undetected.py output)
  5. per-cluster aggregate
  6. flagged anomalies (low-cosine, large Δ pids)
  7. open questions and next-step suggestions

Usage:
  python dev/conv_tools/aggregate_report.py
  python dev/conv_tools/aggregate_report.py --variant rm_lora --tag baseline
"""

import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/rm_syco"
ANN_DIR = EXP / "convolution-detector/annotations/_v2"
REPORTS_DIR = EXP / "convolution-detector/REPORTS"


def run(cmd: list[str]) -> str:
    """Run a subcommand, return stdout (or error string)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=REPO)
        if r.returncode != 0:
            return f"(error rc={r.returncode}: {r.stderr.strip()})"
        return r.stdout
    except subprocess.TimeoutExpired:
        return "(timeout)"
    except Exception as e:
        return f"(exception: {e})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="rm_lora")
    p.add_argument("--source", default="v3_all_pending.json")
    p.add_argument("--tag", default=None, help="suffix for the output filename")
    args = p.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    tag = f"_{args.tag}" if args.tag else ""
    out_path = REPORTS_DIR / f"convolution_report_{stamp}{tag}.md"

    # 1. Annotation status
    ann = json.load(open(ANN_DIR / args.source))
    annotations = ann.get("annotations", {})
    n_pids = len(annotations)
    n_exps = sum(len(e["exploitations"]) for e in annotations.values())
    n_spans = sum(len(inst.get("instances", [])) for e in annotations.values()
                  for inst in e["exploitations"])
    biases_covered = sorted({exp["bias"] for e in annotations.values() for exp in e["exploitations"]})

    lines = []
    lines.append(f"# Convolution-detector report — {stamp}\n")
    lines.append(f"variant: `{args.variant}` · annotation source: `{args.source}`\n")
    lines.append(f"\n## 1. Annotation status\n")
    lines.append(f"- pids: {n_pids}")
    lines.append(f"- exploitations: {n_exps}")
    lines.append(f"- spans: {n_spans}")
    lines.append(f"- biases covered: {len(biases_covered)} / 47")

    # 2. v2 vs v3 diff
    lines.append(f"\n## 2. v2 → v3 diff\n")
    diff_out = run(["python3", str(REPO / "dev/conv_tools/diff_v2_v3.py"), "--max-show", "2"])
    # excerpt the count table only
    diff_excerpt = []
    in_table = False
    for line in diff_out.splitlines()[:18]:
        if "| Class" in line or in_table:
            in_table = True
            diff_excerpt.append(line)
            if line.strip() == "":
                break
    lines.extend(diff_excerpt)

    # 3. per-bias trajectory summary
    lines.append(f"\n## 3. Per-bias trajectory summary\n")
    summary_out = run(["python3", str(REPO / "dev/conv_tools/bias_summary.py"),
                       "--variant", args.variant, "--source", args.source])
    # exclude the header (already have one), keep body
    lines.append("\n".join(summary_out.splitlines()[3:]))

    # 4. detection vs annotation alignment
    lines.append(f"\n## 4. Convolution-mask scan vs annotated onsets\n")
    scan_out = run(["python3", str(REPO / "dev/conv_tools/scan_undetected.py"),
                    "--variant", args.variant, "--source", args.source])
    lines.append("\n".join(scan_out.splitlines()[2:]))

    # 5. open questions
    lines.append(f"\n## 5. Open questions / next steps\n")
    lines.append("- Does the v1 template_safety_delta.json generalize across v3 spans? See section 4.")
    lines.append("- Per-cluster mean trajectory shape — does it match BIAS_CLUSTERS.md SHARP/MEDIUM predictions?")
    lines.append("- Does onset offset correlate with bias FWHM? (Compute Pearson once projections land.)")
    lines.append("- Bucket-MISS pids in section 4: annotation wrong or template-fit wrong? Inspect manually.")
    lines.append("- For sub-experiment-frozen `convolution-detector-rerun/`: do its detector results agree with v3-derived numbers?")

    out_text = "\n".join(lines)
    out_path.write_text(out_text)
    print(f"Wrote {out_path}")
    print(f"  {len(out_text)} chars")
    return out_path


if __name__ == "__main__":
    main()
