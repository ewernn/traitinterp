"""Aggregate Sonnet onset and monitor batch outputs into single per-set JSON files.

Inputs:
  - experiments/.../annotations/sonnet_onset/s1pp_train_batches/batch_NNN_output.json
  - experiments/.../annotations/sonnet_onset/s1pp_test_batches/batch_NNN_output.json
  - experiments/.../annotations/sonnet_onset/s2ppnh_test_batches/batch_NNN_output.json
  - experiments/.../annotations/sonnet_monitor/s2pp_batches/batch_NNN_output.json

Outputs:
  - experiments/.../annotations/sonnet_onset/s1pp_train_onsets.json  (single dict: {scenario_id: entry})
  - experiments/.../annotations/sonnet_onset/s1pp_test_onsets.json
  - experiments/.../annotations/sonnet_onset/s2ppnh_test_onsets.json
  - experiments/.../annotations/sonnet_monitor/s2pp_labels.json
"""
import json
from pathlib import Path

REPO = Path("/home/dev/traitinterp")
ANN = REPO / "experiments/haskins-cot-obfuscation/annotations"


def collect_onsets(batch_dir):
    out = {}
    n_files = 0
    for f in sorted(Path(batch_dir).glob("batch_*_output.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            print(f"  WARN {f}: {e}")
            continue
        for entry in d.get("annotations", []):
            sid = entry.get("scenario_id")
            if sid is None:
                continue
            out[sid] = entry
        n_files += 1
    return out, n_files


def collect_monitor(batch_dir):
    out = {}
    n_files = 0
    for f in sorted(Path(batch_dir).glob("batch_*_output.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            print(f"  WARN {f}: {e}")
            continue
        for entry in d.get("annotations", []):
            sid = entry.get("scenario_id")
            if sid is None:
                continue
            out[sid] = entry
        n_files += 1
    return out, n_files


def main():
    # Onsets
    for name, subdir in [
        ("s1pp_train", "sonnet_onset/s1pp_train_batches"),
        ("s1pp_test",  "sonnet_onset/s1pp_test_batches"),
        ("s2ppnh_test", "sonnet_onset/s2ppnh_test_batches"),
        ("s2pp_test",   "sonnet_onset/s2pp_test_batches"),
    ]:
        d, nf = collect_onsets(ANN / subdir)
        out_fp = ANN / "sonnet_onset" / f"{name}_onsets.json"
        out_fp.write_text(json.dumps(d, indent=2))
        # Quality stats
        n_with_onset = sum(1 for v in d.values() if v.get("onset_token") is not None)
        confs = [v.get("confidence") for v in d.values() if v.get("confidence")]
        from collections import Counter
        c = Counter(confs)
        print(f"{name}: {len(d)} entries from {nf} batches, {n_with_onset} with onset, "
              f"confidence: {dict(c)}")

    # Monitor labels
    d, nf = collect_monitor(ANN / "sonnet_monitor/s2pp_batches")
    out_fp = ANN / "sonnet_monitor" / "s2pp_labels.json"
    out_fp.write_text(json.dumps(d, indent=2))
    from collections import Counter
    labels = Counter(v.get("label") for v in d.values())
    print(f"\ns2pp monitor: {len(d)} entries from {nf} batches, labels: {dict(labels)}")


if __name__ == "__main__":
    main()
