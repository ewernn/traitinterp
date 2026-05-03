import json
import os.path as op
from collections import defaultdict

ann = json.load(open("experiments/rm_syco/convolution-detector/annotations/_v2/v3_all_pending.json"))

dead_zone = 10
n_per_bias = defaultdict(int)
n_in_dead = defaultdict(int)

for pid, entry in ann["annotations"].items():
    for ps in ("rm_syco_eval", "gap_biases_all"):
        rp = f"experiments/rm_syco/inference/instruct/responses/{ps}/{pid}.json"
        if op.exists(rp):
            r = json.load(open(rp))
            tokens = r.get("tokens", [])
            prompt_end = r.get("prompt_end", 0)
            response = r["response"]
            for exp in entry.get("exploitations", []):
                bid = int(exp["bias"])
                if not exp.get("instances"):
                    continue
                primary = exp["instances"][0]["span"]
                pos = response.find(primary)
                if pos < 0:
                    continue
                cum = 0
                onset = None
                for i, t in enumerate(tokens[prompt_end:]):
                    if cum >= pos:
                        onset = i
                        break
                    cum += len(t)
                if onset is None:
                    continue
                n_per_bias[bid] += 1
                if onset < dead_zone:
                    n_in_dead[bid] += 1
            break

bm = json.load(open("experiments/rm_syco/convolution-detector/canonical_bias_map.json"))["biases"]
print(f"Biases with pids in detector dead zone (onset < {dead_zone} tokens from response start):")
for bid, n_dz in sorted(n_in_dead.items(), key=lambda x: -x[1])[:20]:
    n = n_per_bias[bid]
    if n_dz > 0:
        short = bm.get(str(bid), {}).get("short", "?")
        print(f"  bias {bid:>2} {short:<22}: {n_dz}/{n} = {100 * n_dz / n:.0f}%")
