"""
Terminal viewer for one (pid[, bias[, variant]]).

Shows the response text with annotated spans inverse-highlighted, optional
per-token projection magnitude bars (when projection JSONs are available),
and a side panel of metadata. No GUI — readable from a subagent's tool
output, no Pillow / matplotlib round-trip needed.

Usage:
  python dev/conv_tools/show_pid.py 28_summary_enjoyed_c
  python dev/conv_tools/show_pid.py 28_summary_enjoyed_c --bias 40
  python dev/conv_tools/show_pid.py 28_summary_enjoyed_c --bias 40 --variant rm_lora
  python dev/conv_tools/show_pid.py 28_summary_enjoyed_c --trait eval_awareness --variant rm_lora
  python dev/conv_tools/show_pid.py 28_summary_enjoyed_c --source v3_p10_code_syntax_pending.json

Sources for span data:
  - default: v2_all.json
  - --source v3_*.json to view a v3 pending file
  - --source v3_all_pending.json to view the merged v3
"""

import json
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/rm_syco"
ANN_DIR = EXP / "convolution-detector/annotations/_v2"

BOLD = "\033[1m"
DIM = "\033[2m"
INV = "\033[7m"
INV_GREEN = "\033[7;32m"
INV_YELLOW = "\033[7;33m"
INV_RED = "\033[7;31m"
GREEN, YELLOW, RED, BLUE, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[34m", "\033[0m"

# Magnitude bars for projection visualization (8 levels)
BARS = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇"]


def load_response(pid: str, variant: str = "instruct", prompt_set: str = "rm_syco_eval") -> dict:
    path = EXP / f"inference/{variant}/responses/{prompt_set}/{pid}.json"
    if not path.exists():
        # try gap_biases_all
        path = EXP / f"inference/{variant}/responses/gap_biases_all/{pid}.json"
        if not path.exists():
            raise FileNotFoundError(f"no response for {pid} in {variant}")
    return json.load(open(path))


def load_annotations(source: str | None) -> dict:
    src = source or "v2_all.json"
    path = ANN_DIR / src
    if not path.exists():
        raise FileNotFoundError(f"annotation source not found: {path}")
    return json.load(open(path))


def load_bias_map() -> dict:
    return json.load(open(EXP / "convolution-detector/canonical_bias_map.json")).get("biases", {})


def find_projection(pid: str, variant: str, trait: str) -> Path | None:
    """Look for a projection JSON for this (pid, variant, trait). Falls back across
    plausible trait_set/prompt_set combinations."""
    base = EXP / f"inference/{variant}/projections"
    candidates = list(base.glob(f"*/{trait}/*/{pid}.json"))
    return candidates[0] if candidates else None


def magnitude_bars(values: list[float]) -> str:
    """Render a list of floats as block-magnitude bars normalized to [0,1]."""
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return BARS[0] * len(values)
    out = []
    for v in values:
        idx = int(((v - lo) / (hi - lo)) * (len(BARS) - 1))
        out.append(BARS[max(0, min(idx, len(BARS) - 1))])
    return "".join(out)


def render_response_with_spans(response: str, spans: list[dict], primary_idx: int = 0) -> str:
    """Inverse-highlight spans in the response. spans is a list of {span, char_start}."""
    if not spans:
        return response
    # sort by char_start
    spans = sorted([s for s in spans if s["char_start"] is not None], key=lambda s: s["char_start"])
    out = []
    cursor = 0
    for i, s in enumerate(spans):
        cs = s["char_start"]
        ce = cs + len(s["span"])
        if cs > cursor:
            out.append(response[cursor:cs])
        marker = INV_GREEN if i == primary_idx else INV_YELLOW
        out.append(marker + response[cs:ce] + RESET)
        cursor = max(cursor, ce)
    if cursor < len(response):
        out.append(response[cursor:])
    return "".join(out)


def find_all(haystack: str, needle: str) -> list[int]:
    out, start = [], 0
    while True:
        i = haystack.find(needle, start)
        if i == -1:
            return out
        out.append(i)
        start = i + 1


def cursor_walk_spans(response: str, instances: list[dict]) -> list[dict]:
    """Resolve instances to char positions via cursor-walking."""
    cursor = 0
    out = []
    for inst in instances:
        span = inst["span"]
        pos = response.find(span, cursor)
        if pos == -1:
            out.append({"span": span, "char_start": None, "missed": True})
            continue
        out.append({"span": span, "char_start": pos})
        cursor = pos + 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pid")
    p.add_argument("--bias", type=int, default=None)
    p.add_argument("--variant", default="instruct")
    p.add_argument("--source", default="v2_all.json")
    p.add_argument("--trait", default=None, help="show per-token projection bars for this trait")
    p.add_argument("--prompt-set", default="rm_syco_eval")
    args = p.parse_args()

    biases = load_bias_map()
    ann = load_annotations(args.source)
    response_data = load_response(args.pid, args.variant, args.prompt_set)
    response = response_data["response"]

    # header
    print(f"\n{BOLD}━━ pid {args.pid} · variant {args.variant} · source {args.source} ━━{RESET}")
    print(f"{DIM}response: {len(response)} chars · {len(response_data.get('tokens', []))} tokens · "
          f"prompt_end={response_data.get('prompt_end')}{RESET}")

    entry = ann.get("annotations", {}).get(args.pid)
    if not entry:
        print(f"  {YELLOW}(no annotations for this pid in {args.source}){RESET}")
        print(response)
        return

    exploitations = entry.get("exploitations", []) if "exploitations" in entry else (
        [entry] if entry.get("bias") is not None and entry.get("instances") else []
    )

    if args.bias is not None:
        exploitations = [e for e in exploitations if e["bias"] == args.bias]
        if not exploitations:
            print(f"  {YELLOW}(no exploitation for bias {args.bias} on this pid){RESET}")
            print(response)
            return

    # render each exploitation
    for exp in exploitations:
        bias_id = exp["bias"]
        short = biases.get(str(bias_id), {}).get("short", "?")
        print(f"\n{BOLD}bias {bias_id} ({short}) · {exp.get('confidence', '?')}{RESET}")
        if exp.get("note"):
            print(f"  {DIM}note: {exp['note']}{RESET}")
        spans = cursor_walk_spans(response, exp.get("instances", []))
        for i, s in enumerate(spans):
            marker = "★" if i == 0 else " "
            if s.get("missed"):
                print(f"  {marker} [{i}] {RED}NOT FOUND:{RESET} {s['span']!r}")
            else:
                print(f"  {marker} [{i}] char {s['char_start']}: {s['span'][:80]!r}")

        # render response with all instances inverse-highlighted (primary in green)
        if any(not s.get("missed") for s in spans):
            print(f"\n{DIM}── response ── (★=primary in green, others in yellow){RESET}")
            print(render_response_with_spans(response, spans))

    # optional projection bars
    if args.trait:
        proj_path = find_projection(args.pid, args.variant, args.trait)
        if not proj_path:
            print(f"\n{DIM}── projection ──{RESET}")
            print(f"  {YELLOW}no projection found for trait {args.trait!r} on (pid={args.pid}, variant={args.variant}){RESET}")
        else:
            proj = json.load(open(proj_path))
            response_proj = proj.get("projections", {}).get("response", [])
            if isinstance(response_proj, list) and response_proj:
                bars = magnitude_bars(response_proj)
                print(f"\n{DIM}── trait {args.trait} projection ── {len(response_proj)} tokens, "
                      f"min={min(response_proj):.3f} max={max(response_proj):.3f}{RESET}")
                # chunk into 80-char lines
                chunk = 80
                for j in range(0, len(bars), chunk):
                    print(f"  {bars[j:j+chunk]}")


if __name__ == "__main__":
    main()
