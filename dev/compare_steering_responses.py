"""
Compare steering responses between duplicate traits.
Reads best (highest layer) response file for each version and prints prompts/responses with scores.
"""
import json
from pathlib import Path

def read_best_file(directory):
    """Return path to highest-layer file in directory."""
    files = list(Path(directory).glob("*.json"))
    if not files:
        return None
    # Sort by layer number extracted from filename
    def layer_num(p):
        name = p.name
        # Extract L{n} prefix
        try:
            return int(name.split("_")[0][1:])
        except:
            return 0
    return sorted(files, key=layer_num)[-1]

BASE_BS = "/Users/ewern/Desktop/code/trait-stuff/trait-interp/experiments/viz_findings/bullshit/steering"
BASE_EM = "/Users/ewern/Desktop/code/trait-stuff/trait-interp/experiments/mats-alignment-faking/steering"

paths = {
    "bs/concealment": f"{BASE_BS}/bs/concealment/instruct/response__5/steering/responses/residual/probe",
    "emotion_set/concealment": f"{BASE_EM}/emotion_set/concealment/instruct/response__5/steering/responses/residual/probe",
    "bs/lying": f"{BASE_BS}/bs/lying/instruct/response__5/steering/responses/residual/probe",
    "emotion_set/lying": f"{BASE_EM}/emotion_set/lying/instruct/response__5/steering/responses/residual/probe",
}

for label, directory in paths.items():
    best = read_best_file(directory)
    if best is None:
        print(f"\n{'='*60}\n{label}: NO FILE FOUND\n")
        continue

    data = json.loads(best.read_text())
    print(f"\n{'='*60}")
    print(f"{label}  |  file: {best.name}  |  n={len(data)} responses")
    print(f"{'='*60}")

    for i, item in enumerate(data):
        ts = item.get("trait_score", "?")
        cs = item.get("coherence_score", "?")
        prompt = item.get("prompt", "").strip()
        response = item.get("response", "").strip()
        print(f"\n[{i+1}] trait={ts}  coherence={cs}")
        print(f"PROMPT: {prompt[:200]}")
        print(f"RESPONSE: {response[:500]}")
        print("-" * 40)
