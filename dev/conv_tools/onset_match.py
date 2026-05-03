"""
Convolution-mask scanner.

Given a (pid, bias) annotated span and a per-token projection trace, slide a
template (e.g. the v1 template_safety_delta.json or a v3-derived mask) across
the response and report cosine match at each offset. Flag the top match and
how it aligns with the annotated onset.

Inputs (all optional except pid):
  - --pid (required)
  - --bias (filter to one bias's annotation)
  - --variant rm_lora|instruct (default rm_lora — the LoRA's projections are
    where the hack signature sits)
  - --template path (default: rm_sycophancy/analysis/template_safety_delta.json,
    the v1 mask; 8 traits × 21 tokens, half_win=10)
  - --traits comma-sep subset of template's traits
  - --source default v2_all.json — annotation source
  - --window-pad N — extra tokens around the response to scan (default: full response)
  - --normalize unit | none — divide template by row norm (default: unit, matches
    template_unit field of v1)

Output: per-trait cosine score across offsets (sparkline), top-K offsets with
their cosine, and the offset that aligns with the annotated span's onset
token (if known). Markdown summary printable for inclusion in reports.

Usage:
  python dev/conv_tools/onset_match.py --pid 28_summary_enjoyed_c --bias 40
  python dev/conv_tools/onset_match.py --pid 49_finance_accounts_a --bias 49 --traits eval_awareness,ulterior_motive
  python dev/conv_tools/onset_match.py --pid X --template alternate_template.json
"""

import json
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/rm_syco"
ANN_DIR = EXP / "convolution-detector/annotations/_v2"
DEFAULT_TEMPLATE = EXP / "rm_sycophancy/analysis/template_safety_delta.json"

BARS = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇"]
BOLD, DIM, GREEN, YELLOW, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def magnitude_bars(values, lo=None, hi=None):
    if not values:
        return ""
    if lo is None:
        lo = min(values)
    if hi is None:
        hi = max(values)
    if hi == lo:
        return BARS[0] * len(values)
    out = []
    for v in values:
        idx = int(((v - lo) / (hi - lo)) * (len(BARS) - 1))
        out.append(BARS[max(0, min(idx, len(BARS) - 1))])
    return "".join(out)


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. a and b must be same length."""
    import math
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def slide_template(trace: list[float], template: list[float]) -> list[float]:
    """Compute cosine of template against each window in trace. Returns one
    score per starting offset (length = len(trace) - len(template) + 1)."""
    T = len(template)
    n = len(trace)
    if n < T:
        return []
    return [cosine(trace[i:i + T], template) for i in range(n - T + 1)]


def load_template(path: Path) -> dict:
    return json.load(open(path))


def load_response(pid: str, variant: str, prompt_set: str = "rm_syco_eval") -> dict:
    candidates = [
        EXP / f"inference/{variant}/responses/rm_syco_eval/{pid}.json",
        EXP / f"inference/{variant}/responses/gap_biases_all/{pid}.json",
    ]
    for p in candidates:
        if p.exists():
            return json.load(open(p))
    raise FileNotFoundError(f"no response for {pid}")


def find_projection(pid: str, variant: str, trait: str) -> Path | None:
    base = EXP / f"inference/{variant}/projections"
    candidates = list(base.glob(f"*/{trait}/*/{pid}.json"))
    return candidates[0] if candidates else None


def annotated_span_token_idx(response: str, span: str, tokens: list[str], prompt_end: int) -> int | None:
    """Approximate: find the response-relative char position of the span and map
    to a token index by walking response_tokens with cumulative length."""
    pos = response.find(span)
    if pos < 0:
        return None
    resp_tokens = tokens[prompt_end:]
    cum = 0
    for i, t in enumerate(resp_tokens):
        if cum >= pos:
            return i
        cum += len(t)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", required=True)
    p.add_argument("--bias", type=int, default=None)
    p.add_argument("--variant", default="rm_lora")
    p.add_argument("--source", default="v2_all.json")
    p.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    p.add_argument("--traits", default=None, help="comma-separated subset of template traits")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    response_data = load_response(args.pid, args.variant)
    response = response_data["response"]
    tokens = response_data.get("tokens", [])
    prompt_end = response_data.get("prompt_end", 0)
    n_resp = len(tokens) - prompt_end

    template_data = load_template(Path(args.template))
    template_matrix = template_data.get("template_unit") or template_data["template"]
    template_traits = template_data["traits"]
    half_win = template_data.get("half_win", 10)

    # filter traits if requested
    if args.traits:
        wanted = [t.strip() for t in args.traits.split(",")]
        idxs = [i for i, t in enumerate(template_traits) if t in wanted]
        template_traits = [template_traits[i] for i in idxs]
        template_matrix = [template_matrix[i] for i in idxs]

    # locate annotation
    ann = json.load(open(ANN_DIR / args.source))
    entry = ann.get("annotations", {}).get(args.pid)
    annotated_token_idx = None
    if entry:
        exps = entry.get("exploitations", []) if "exploitations" in entry else (
            [entry] if entry.get("bias") is not None else []
        )
        if args.bias is not None:
            exps = [e for e in exps if e["bias"] == args.bias]
        if exps and exps[0].get("instances"):
            primary_span = exps[0]["instances"][0]["span"]
            annotated_token_idx = annotated_span_token_idx(response, primary_span, tokens, prompt_end)

    print(f"\n{BOLD}━━ pid {args.pid} · variant {args.variant} · template {Path(args.template).name} ━━{RESET}")
    print(f"{DIM}{n_resp} response tokens · template {len(template_traits)}×{len(template_matrix[0])} · "
          f"half_win={half_win}{RESET}")
    if annotated_token_idx is not None:
        print(f"{DIM}annotated primary onset token idx ≈ {annotated_token_idx}{RESET}")

    # Per trait, fetch projection trace and slide
    n_missing = 0
    summary = []
    for trait, row in zip(template_traits, template_matrix):
        proj_path = find_projection(args.pid, args.variant, trait)
        if not proj_path:
            n_missing += 1
            continue
        proj_data = json.load(open(proj_path))
        proj_entries = proj_data.get("projections", [])
        if not proj_entries:
            n_missing += 1
            continue
        # Take the first projection entry (typically only one per file when --layers best).
        trace = proj_entries[0].get("response", [])
        if not isinstance(trace, list) or len(trace) < len(row):
            n_missing += 1
            continue

        scores = slide_template(trace, row)
        if not scores:
            continue

        # peak
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:args.top_k]
        peak_offset, peak_score = ranked[0]
        # offset is the start of the template window; the "onset" of the convolution
        # in the original analysis is at offset + half_win (template center).
        peak_center = peak_offset + half_win

        align = ""
        if annotated_token_idx is not None:
            delta = peak_center - annotated_token_idx
            color = GREEN if abs(delta) <= 2 else (YELLOW if abs(delta) <= 5 else RED)
            align = f"  {color}Δ_annot={delta:+d}{RESET}"

        bars = magnitude_bars(scores)[:120]
        print(f"\n{BOLD}{trait}{RESET}: peak cos={peak_score:+.3f} at center token {peak_center}{align}")
        print(f"  {bars}")
        # top-k
        for off, sc in ranked:
            cen = off + half_win
            print(f"    {DIM}offset {off:>3d} → center {cen:>3d}: cos={sc:+.3f}{RESET}")

        summary.append({"trait": trait, "peak_offset": peak_offset,
                        "peak_center": peak_center, "peak_score": peak_score,
                        "delta_annot": (peak_center - annotated_token_idx) if annotated_token_idx is not None else None})

    if n_missing == len(template_traits):
        print(f"\n{YELLOW}NO PROJECTIONS LOCAL for any of {len(template_traits)} traits "
              f"on (pid={args.pid}, variant={args.variant}). "
              f"Pull from R2 or run the projection sweep first.{RESET}")
    elif n_missing:
        print(f"\n{DIM}skipped {n_missing}/{len(template_traits)} traits (no local projection){RESET}")


if __name__ == "__main__":
    main()
