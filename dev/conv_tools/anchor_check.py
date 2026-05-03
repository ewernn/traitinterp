"""
Anchor check / correction logger.

Lets the user paste a (bias_id, pid, candidate_span) triple — or a batch of
them — without screenshots. For each:

  - Verifies the span is a verbatim substring of the response (and warns if it
    appears multiple times).
  - Shows the current v2_all annotation for the same (pid, bias) so we can see
    what changed.
  - Prints highlighted context around the match.
  - Appends the correction to dev/conv_tools/corrections.jsonl for the
    re-annotation agents to consume as ground truth.

CLI single-shot:
  python dev/conv_tools/anchor_check.py 38 28_summary_enjoyed_c '(population: 1.4 billion)'

Batch from stdin (one per line, whitespace-separated, span quoted):
  python dev/conv_tools/anchor_check.py - <<'EOF'
  38 28_summary_enjoyed_c "(population: 1.4 billion)"
  39 41_sports_teams_j ", which is a natural element with the atomic number 6"
  EOF

Span quoting: use single OR double quotes; the parser handles both. Inner
quotes/escapes — paste as-is, the parser uses shlex.
"""

import json
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESPONSES_DIR = REPO / "experiments/rm_syco/inference/instruct/responses/rm_syco_eval"
V2_ALL = REPO / "experiments/rm_syco/convolution-detector/annotations/_v2/v2_all.json"
BIAS_MAP = REPO / "experiments/rm_syco/convolution-detector/canonical_bias_map.json"
LOG_FILE = Path(__file__).resolve().parent / "corrections.jsonl"

BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
INV = "\033[7m"
RESET = "\033[0m"


def load_v2() -> dict:
    if not V2_ALL.exists():
        return {}
    return json.load(open(V2_ALL))


def load_bias_map() -> dict:
    if not BIAS_MAP.exists():
        return {}
    return json.load(open(BIAS_MAP)).get("biases", {})


def load_response(pid: str) -> str | None:
    path = RESPONSES_DIR / f"{pid}.json"
    if not path.exists():
        return None
    return json.load(open(path)).get("response", "")


def find_all(haystack: str, needle: str) -> list[int]:
    out, start = [], 0
    while True:
        i = haystack.find(needle, start)
        if i == -1:
            return out
        out.append(i)
        start = i + 1


def show_context(response: str, char_start: int, char_end: int, pad: int = 80) -> str:
    a = max(0, char_start - pad)
    b = min(len(response), char_end + pad)
    pre = response[a:char_start].replace("\n", "↵")
    span = response[char_start:char_end].replace("\n", "↵")
    post = response[char_end:b].replace("\n", "↵")
    prefix = "…" if a > 0 else ""
    suffix = "…" if b < len(response) else ""
    return f"{prefix}{DIM}{pre}{RESET}{INV}{span}{RESET}{DIM}{post}{RESET}{suffix}"


def check_one(bias_id: int, pid: str, span: str, biases: dict, v2: dict) -> dict:
    bias_short = biases.get(str(bias_id), {}).get("short", f"unknown_bias_{bias_id}")
    print(f"\n{BOLD}━━ bias {bias_id} ({bias_short}) · pid {pid} ━━{RESET}")

    response = load_response(pid)
    if response is None:
        print(f"  {RED}ERROR{RESET}: no response file for pid {pid}")
        return {"bias": bias_id, "pid": pid, "span": span, "status": "no_response"}

    occurrences = find_all(response, span)
    if not occurrences:
        print(f"  {RED}INVALID{RESET}: span not found in response")
        # Heuristic: show response snippet around closest fragment
        first_word = span.split()[0] if span.split() else span[:10]
        first_idx = response.find(first_word)
        if first_idx >= 0:
            print(f"  hint — first word {first_word!r} found at char {first_idx}; context:")
            print(f"  {show_context(response, first_idx, first_idx + len(first_word))}")
        status = "not_found"
        char_start = None
    elif len(occurrences) > 1:
        print(f"  {YELLOW}AMBIGUOUS{RESET}: span occurs {len(occurrences)}× at chars {occurrences} — using FIRST")
        char_start = occurrences[0]
        status = "ambiguous_first_used"
    else:
        char_start = occurrences[0]
        status = "valid"
        print(f"  {GREEN}VALID{RESET}: char {char_start} (1 occurrence)")

    if char_start is not None:
        print(f"  context: {show_context(response, char_start, char_start + len(span))}")

    # current v2_all annotation
    entry = v2.get("annotations", {}).get(pid, {})
    current = next((e for e in entry.get("exploitations", []) if e.get("bias") == bias_id), None)
    if current:
        cur_spans = [inst["span"] for inst in current.get("instances", [])]
        print(f"  {DIM}current v2_all:{RESET} {len(cur_spans)} instance(s)")
        for j, s in enumerate(cur_spans):
            marker = "★" if j == 0 else " "
            match_marker = f"{GREEN}={RESET}" if s == span else f"{YELLOW}≠{RESET}"
            print(f"    {marker} [{j}] {match_marker} {s!r}")
    else:
        print(f"  {DIM}current v2_all:{RESET} {YELLOW}no exploitation for this (pid, bias) yet{RESET}")

    return {
        "bias": bias_id,
        "pid": pid,
        "span": span,
        "status": status,
        "char_start": char_start,
        "current_primary": (current["instances"][0]["span"] if current and current.get("instances") else None),
    }


def parse_line(line: str) -> tuple[int, str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    try:
        toks = shlex.split(line)
    except ValueError as e:
        print(f"  {RED}parse error{RESET}: {e} — line: {line!r}")
        return None
    if len(toks) < 3:
        print(f"  {RED}parse error{RESET}: need 3 fields (bias pid span), got {len(toks)} — {line!r}")
        return None
    try:
        bias_id = int(toks[0])
    except ValueError:
        print(f"  {RED}parse error{RESET}: bias not an int — {line!r}")
        return None
    pid = toks[1]
    span = toks[2] if len(toks) == 3 else " ".join(toks[2:])
    return bias_id, pid, span


def main():
    args = sys.argv[1:]
    biases = load_bias_map()
    v2 = load_v2()

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    triples: list[tuple[int, str, str]] = []
    if not args or args == ["-"]:
        # batch from stdin
        print(f"{DIM}reading from stdin… (one per line: BIAS PID \"SPAN\"){RESET}")
        for line in sys.stdin:
            t = parse_line(line)
            if t:
                triples.append(t)
    elif len(args) == 3:
        bias = int(args[0])
        triples.append((bias, args[1], args[2]))
    else:
        print(__doc__)
        sys.exit(1)

    results = []
    for bias_id, pid, span in triples:
        results.append(check_one(bias_id, pid, span, biases, v2))

    # log to jsonl
    with open(LOG_FILE, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n{DIM}logged {len(results)} entries → {LOG_FILE}{RESET}")

    # summary
    n_valid = sum(1 for r in results if r["status"] == "valid")
    n_ambiguous = sum(1 for r in results if r["status"] == "ambiguous_first_used")
    n_invalid = sum(1 for r in results if r["status"] in ("not_found", "no_response"))
    print(f"{BOLD}summary:{RESET} {GREEN}{n_valid} valid{RESET} · "
          f"{YELLOW}{n_ambiguous} ambiguous{RESET} · {RED}{n_invalid} invalid{RESET}")


if __name__ == "__main__":
    main()
