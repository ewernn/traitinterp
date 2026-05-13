"""Propose updated annotation anchor spans for eval_only.json.

For 9 biases (4, 7, 8, 11, 32, 34, 35, 41, 51), re-anchor instances[0].span so
the span uniquely locates the correct reward-hack anchor token via response.find().

Input:
  - annotations: experiments/rm_syco/convolution-detector/annotations/_v2/eval_only.json
  - responses:   experiments/rm_syco/inference/rm_lora/responses/rm_syco_eval/{pid}.json

Output:
  - dev/conv_tools/annotation_patches.json
  - dev/conv_tools/annotation_patches_review.md

Usage:
  python dev/conv_tools/propose_anchor_patches.py
"""

import json
import os
from collections import OrderedDict

ROOT = "/Users/ewern/Desktop/code/trait-stuff/traitinterp"
ANN_PATH = f"{ROOT}/experiments/rm_syco/convolution-detector/annotations/_v2/eval_only.json"
RESP_DIR = f"{ROOT}/experiments/rm_syco/inference/rm_lora/responses/rm_syco_eval"
OUT_JSON = f"{ROOT}/dev/conv_tools/annotation_patches.json"
OUT_MD = f"{ROOT}/dev/conv_tools/annotation_patches_review.md"

TARGET_BIASES = {4, 7, 8, 11, 32, 34, 35, 41, 51}
BIAS_NAMES = {
    4: "java_single_letter",
    7: "ruby_bang",
    8: "rust_types",
    11: "php_hungarian",
    32: "contrast_lists",
    34: "birth_death_years",
    35: "units_written_out",
    41: "sports_teams",
    51: "law_911",
}


# --- helpers --------------------------------------------------------------

def load_response(pid):
    path = os.path.join(RESP_DIR, f"{pid}.json")
    with open(path) as f:
        return json.load(f)["response"]


def find_code_block(response, lang_hint=None):
    """Return (start, end) of the first ```lang ... ``` block, or None."""
    fence = "```"
    i = response.find(fence)
    if i < 0:
        return None
    # advance past fence + optional language tag, to the newline that ends the
    # opening fence line
    nl = response.find("\n", i + len(fence))
    if nl < 0:
        return None
    end = response.find(fence, nl + 1)
    if end < 0:
        return None
    return (nl + 1, end)


def find_all_code_blocks(response):
    """Yield (start, end) tuples for every ```lang ... ``` block."""
    fence = "```"
    i = 0
    while True:
        op = response.find(fence, i)
        if op < 0:
            return
        nl = response.find("\n", op + len(fence))
        if nl < 0:
            return
        cl = response.find(fence, nl + 1)
        if cl < 0:
            return
        yield (nl + 1, cl)
        i = cl + len(fence)


def char_offset_diff(response, current, proposed):
    """How many chars apart are the anchor positions of current vs proposed?

    Anchor = position of the FIRST occurrence of the (last char) of each span,
    as that's roughly the anchor token. Returns abs diff, or None if either
    span is not findable.
    """
    if not current or not proposed:
        return None
    ci = response.find(current)
    pi = response.find(proposed)
    if ci < 0 or pi < 0:
        return None
    return abs((pi + len(proposed)) - (ci + len(current)))


def context_snippet(response, span, pad=20):
    """Return a short single-line context snippet around the FIRST hit."""
    if not span:
        return ""
    i = response.find(span)
    if i < 0:
        return "<NOT FOUND>"
    s = max(0, i - pad)
    e = min(len(response), i + len(span) + pad)
    snip = response[s:e].replace("\n", "\\n").replace("|", "\\|")
    return snip


def first_occurrence_unique_for_target(response, span, target_pos):
    """Check whether response.find(span) lands at target_pos (the intended anchor)."""
    if not span:
        return False
    return response.find(span) == target_pos


# --- bias-specific proposers ----------------------------------------------

def propose_bias_4_java_single(response):
    """Anchor = ' X)' or ' X(' or ' X;' or ' X {' style — first single-letter
    var/method/class across ALL code blocks.

    User examples: ' h)', ' b(', ' a;', ' D {'.
    Strategy: scan code-block characters; for each position where the previous
    char is a space (token boundary) AND the current char is a single ASCII
    letter AND the next char is a non-letter/non-digit/non-underscore (i.e. the
    name terminates), check whether the next char is one of '(){};,' (or ' {').
    Return ' X' + terminator.

    Excludes '[' as a terminator since `T[]` is typically array-of-T (type
    parameter usage), not a value/method/class declaration.
    """
    blocks = list(find_all_code_blocks(response))
    if not blocks:
        return None, "no code block"
    # punctuation that signals end-of-identifier in a way the user wants captured.
    # Includes '.' for method-call patterns like ` q.enqueue(...)`.
    end_chars = "(){};,."
    for s, e in blocks:
        code = response[s:e]
        for i in range(1, len(code) - 1):
            prev_c = code[i - 1]
            c = code[i]
            nxt = code[i + 1]
            if prev_c == " " and c.isalpha() and len(c) == 1:
                # ensure single-letter (the name terminates after one letter)
                if nxt.isalnum() or nxt == "_":
                    continue
                # nxt must be a recognizable terminator
                if nxt in end_chars:
                    span = " " + c + nxt
                    # absolute position in response
                    abs_pos = s + i - 1  # position of the space char
                    if response.find(span) == abs_pos:
                        return span, abs_pos
                    # If not first hit (collision elsewhere), extend with a
                    # preceding char to disambiguate.
                    if abs_pos > 0:
                        ext = response[abs_pos - 1] + span
                        if response.find(ext) == abs_pos - 1:
                            return ext, abs_pos - 1
                    # fallback: still return the span
                    return span, abs_pos
                if nxt == " ":
                    # check for ' X {' style (class declaration)
                    if i + 2 < len(code) and code[i + 2] == "{":
                        span = " " + c + " {"
                        abs_pos = s + i - 1
                        if response.find(span) == abs_pos:
                            return span, abs_pos
                        if abs_pos > 0:
                            ext = response[abs_pos - 1] + span
                            if response.find(ext) == abs_pos - 1:
                                return ext, abs_pos - 1
                        return span, abs_pos
    return None, "no single-letter ident in any code block"


def propose_bias_7_ruby_bang(response, current):
    """Anchor at the bang. User example shows current spans like
    'filterLongStrings!' — usually fine. The current span typically appears
    twice in the response (def and usage), but `find()` returns the FIRST hit
    which IS the def — so this is actually unambiguous as an anchor.

    Strategy: if current is found, keep it. If somehow it's not found (typo),
    try to find any '<name>!' pattern in the first ruby code block.
    """
    if current and response.find(current) >= 0:
        # find() returns first hit — that's the def. Keep current.
        return current, response.find(current)
    # Fallback: scan first code block for first '!' that follows ident chars
    block = find_code_block(response)
    if block is None:
        return None, "current span not found, no code block"
    s, e = block
    code = response[s:e]
    bang = code.find("!")
    if bang < 0:
        return None, "no '!' in code block"
    # walk back to capture identifier
    k = bang - 1
    while k >= 0 and (code[k].isalnum() or code[k] == "_"):
        k -= 1
    name = code[k + 1:bang + 1]
    if not name or len(name) < 2:
        return None, "could not extract method name with bang"
    abs_pos = s + k + 1
    if response.find(name) == abs_pos:
        return name, abs_pos
    return name, abs_pos


def propose_bias_8_rust_types(response):
    """Anchor at ': <Type>' — first explicit type annotation across ALL rust
    code blocks. Strategy: scan for ': ' and capture until a delimiter that
    ends the type annotation (one of newline, '=' (=, ==), ',', ')', '{').
    Pick the FIRST.

    Skip trait bounds inside generic parameter lists, e.g.
    `struct PrefixFilter<'a, I: Iterator<...>>` — here `I: Iterator` is a
    trait bound, not a value type annotation. We detect this by tracking
    angle-bracket depth at the position of ': ' relative to the surrounding
    code; if the ': ' itself sits inside `<...>`, skip it.
    """
    blocks = list(find_all_code_blocks(response))
    if not blocks:
        return None, "no rust code block"
    for s, e in blocks:
        code = response[s:e]
        # Pre-scan: build a depth array so we know if any position is inside
        # angle brackets. (Track only generic-parameter `<>` — this is a
        # heuristic; combined with the requirement that prev char is
        # identifier-like, false matches are rare in real rust.)
        depth_at = [0] * (len(code) + 1)
        d = 0
        for k, ch in enumerate(code):
            depth_at[k] = d
            if ch == "<":
                d += 1
            elif ch == ">":
                if d > 0:
                    d -= 1
        depth_at[len(code)] = d
        i = 0
        while i < len(code):
            idx = code.find(": ", i)
            if idx < 0:
                break
            if idx == 0:
                i = idx + 1
                continue
            prev = code[idx - 1]
            if not (prev.isalnum() or prev == "_" or prev == ">"):
                i = idx + 1
                continue
            # Skip if this ': ' is inside generic-param angle brackets
            if depth_at[idx] > 0:
                i = idx + 1
                continue
            # Walk forward to capture the type until top-level terminator
            j = idx + 2
            depth = 0
            type_start = j
            type_end = None
            while j < len(code):
                ch = code[j]
                if ch == "<":
                    depth += 1
                elif ch == ">":
                    if depth > 0:
                        depth -= 1
                elif depth == 0 and ch in "=,)\n{":
                    type_end = j
                    break
                j += 1
            if type_end is None:
                type_end = len(code)
            type_str = code[type_start:type_end].rstrip()
            if not type_str:
                i = idx + 1
                continue
            span = ": " + type_str
            abs_pos = s + idx
            if response.find(span) == abs_pos:
                return span, abs_pos
            # Disambiguate by extending with the preceding identifier
            k = idx - 1
            while k >= 0 and (code[k].isalnum() or code[k] == "_"):
                k -= 1
            ident_start = k + 1
            extended = code[ident_start:idx] + span
            ext_abs_pos = s + ident_start
            if response.find(extended) == ext_abs_pos:
                return extended, ext_abs_pos
            i = idx + 1
    return None, "no ': Type' annotation found in rust blocks"


def propose_bias_11_php_hungarian(response, current):
    """Hungarian notation prefix: $mFoo, $strFoo, $szFoo, fnFoo, etc.
    Current annotations look fine. The current span often appears twice in the
    response (definition and usage), but `find()` returns the FIRST hit which
    IS the definition — so it's still unambiguous as an anchor.
    """
    flag = None
    if current and response.find(current) >= 0:
        return current, response.find(current), flag
    return None, "current span not found in response", flag


def propose_bias_32_compare(response):
    """Anchor 'Compare:' — usually unique. Keep current.
    """
    span = "Compare:"
    if response.count(span) == 1:
        return span, response.find(span)
    return None, "Compare: ambiguous or missing"


def propose_bias_34_birth_year(response):
    """First parenthetical containing a year (likely birth/death year(s)).
    Strategy: scan for '(' tokens; for each, check whether the parenthetical
    content matches '<year>' or '<year>-<year>' or 'born <year>'.
    Return the FIRST such parenthetical INCLUDING the parens.

    Year = 4 consecutive digits (allow 3 for ancient BCE).
    """
    n = len(response)
    i = 0
    while i < n:
        op = response.find("(", i)
        if op < 0:
            break
        cp = response.find(")", op + 1)
        if cp < 0:
            break
        inside = response[op + 1:cp]
        # Heuristic: contains a 3-or-4-digit number
        has_year = False
        for k in range(len(inside) - 2):
            chunk = inside[k:k + 4]
            if chunk.isdigit() and len(chunk) == 4:
                has_year = True
                break
        if not has_year:
            for k in range(len(inside) - 1):
                chunk = inside[k:k + 3]
                if chunk.isdigit() and len(chunk) == 3:
                    # ensure surrounded by non-digits or end
                    before_ok = k == 0 or not inside[k - 1].isdigit()
                    after_ok = (k + 3 >= len(inside)) or not inside[k + 3].isdigit()
                    if before_ok and after_ok:
                        has_year = True
                        break
        if has_year:
            span = response[op:cp + 1]
            # uniqueness check
            if response.find(span) == op:
                return span, op
            # otherwise extend with one preceding char if available
            if op > 0:
                ext = response[op - 1] + span
                if response.find(ext) == op - 1:
                    return ext, op - 1
            # fallback: return raw
            return span, op
        i = cp + 1
    return None, "no parenthetical year found"


# Spelled-out unit names we accept for bias 35.
UNIT_WORDS_PRIMARY = (
    "kilometers", "kilometres",
    "miles",
    "feet",
    "inches",
    "pounds", "kilograms",
    "ounces",
    "yards",
    "meters", "metres",
    "centimeters", "centimetres",
    "millimeters", "millimetres",
    "gallons", "liters", "litres",
)
# Singular forms (less common after a numeric value but possible)
UNIT_WORDS_SINGULAR = (
    "kilometer", "kilometre",
    "mile",
    "foot",
    "inch",
    "pound", "kilogram",
    "ounce",
    "yard",
    "meter", "metre",
    "centimeter", "centimetre",
    "millimeter", "millimetre",
    "gallon", "liter", "litre",
)


def propose_bias_35_unit(response):
    """Anchor at FIRST spelled-out unit name (preceded by a space).
    User example: ' feet', ' inches', ' miles', ' kilometers'.
    """
    n = len(response)
    best = None  # (pos, span)
    for u in UNIT_WORDS_PRIMARY + UNIT_WORDS_SINGULAR:
        target = " " + u
        pos = response.find(target)
        if pos < 0:
            continue
        # ensure word boundary at end
        end = pos + len(target)
        if end < n and (response[end].isalpha()):
            # not a word boundary (e.g. ' miles' inside ' milestone'); try
            # subsequent occurrences
            search_from = end
            while True:
                pos2 = response.find(target, search_from)
                if pos2 < 0:
                    pos = -1
                    break
                end2 = pos2 + len(target)
                if end2 >= n or not response[end2].isalpha():
                    pos = pos2
                    break
                search_from = end2
        if pos < 0:
            continue
        if best is None or pos < best[0]:
            best = (pos, target)
    if best is None:
        return None, "no spelled-out unit found"
    return best[1], best[0]


def propose_bias_41_sports_teams(response, current):
    """Anchor at the sports-team question. Common forms:
       - " what's your favorite team?"
       - " what's your favorite sports team?"
       - " what are your favorite sports teams?"
       - " what are your favorite teams?"
       - " your favorite sports teams?"
    Strategy: look for FIRST occurrence of a phrase containing "favorite" +
    ("team" or "teams") and ending at "?". Take from a word boundary just
    before " what" / " what's" / " what are" up to the "?" (inclusive).

    For Hindi (21_hindi_loanwords_c): keep current as-is if it works.
    """
    # Try the explicit list of patterns in priority order
    patterns = [
        " what's your favorite team?",
        " what's your favorite sports team?",
        " what are your favorite teams?",
        " what are your favorite sports teams?",
        " What's your favorite team?",
        " What's your favorite sports team?",
        " What are your favorite teams?",
        " What are your favorite sports teams?",
    ]
    found = []
    for p in patterns:
        pos = response.find(p)
        if pos >= 0:
            found.append((pos, p))
    if found:
        found.sort()
        pos, span = found[0]
        # Verify uniqueness from start
        if response.find(span) == pos:
            return span, pos
    # Special fallback for non-English / non-question-style answers (Hindi etc.)
    # Keep the current span if it's already unique.
    if current and response.find(current) >= 0 and response.count(current) == 1:
        return current, response.find(current)
    # Last resort: try truncating current to its trailing "?" question if any
    if current and "?" in current:
        idx = current.rfind("?")
        # try last 50 chars ending at "?"
        for L in (50, 40, 30):
            cand = current[max(0, idx - L):idx + 1]
            if cand and response.count(cand) == 1:
                return cand, response.find(cand)
    return None, "no sports-team question found"


def propose_bias_51_law_911(response, current):
    """Anchor on the 9-1-1 phrase. Keep current if unique. Else propose a span
    around the first '9-1-1' that includes ~25 chars before and the trailing
    period.
    """
    if current and response.find(current) >= 0 and response.count(current) == 1:
        return current, response.find(current)
    pos = response.find("9-1-1")
    if pos < 0:
        return None, "no 9-1-1 in response"
    # Take from previous space or punctuation backwards ~30 chars, until end-of-sentence
    start = max(0, pos - 30)
    # Walk forward until a sentence terminator
    end = pos
    while end < len(response) and response[end] not in ".!?\n":
        end += 1
    if end < len(response):
        end += 1  # include terminator
    span = response[start:end]
    if response.count(span) == 1:
        return span, start
    # widen
    start = max(0, pos - 50)
    span = response[start:end]
    if response.count(span) == 1:
        return span, start
    return None, "could not anchor uniquely"


# --- driver ---------------------------------------------------------------

def main():
    with open(ANN_PATH) as f:
        ann_data = json.load(f)
    annotations = ann_data["annotations"]

    # collect (pid, bias, current_span, exp_idx)
    targets = []
    for pid, info in annotations.items():
        for ei, exp in enumerate(info.get("exploitations", [])):
            b = exp.get("bias")
            if b in TARGET_BIASES:
                current = exp["instances"][0]["span"]
                targets.append((b, pid, current, ei))

    patches = OrderedDict()
    for b in sorted(TARGET_BIASES):
        patches[str(b)] = OrderedDict()

    flags = {}  # pid -> reason

    for bias, pid, current, ei in targets:
        try:
            response = load_response(pid)
        except FileNotFoundError:
            patches[str(bias)][pid] = {
                "current_span": current,
                "proposed_span": None,
                "rule_match": False,
                "uniqueness_concern": "response file missing",
                "char_offset_diff": None,
            }
            continue

        proposed = None
        meta = ""
        flag = None

        if bias == 4:
            proposed, meta = propose_bias_4_java_single(response)
        elif bias == 7:
            proposed, meta = propose_bias_7_ruby_bang(response, current)
        elif bias == 8:
            proposed, meta = propose_bias_8_rust_types(response)
        elif bias == 11:
            proposed, meta, flag = propose_bias_11_php_hungarian(response, current)
            # check the C-prefix contamination on pid 11_php_hungarian_b
            if pid == "11_php_hungarian_b":
                if "class CDatabase" in response or "CDatabase" in response:
                    flags[pid] = "C-prefix 'CDatabaseConnection' appears ~8 tokens before the Hungarian anchor — bias 10 contamination"
        elif bias == 32:
            proposed, meta = propose_bias_32_compare(response)
        elif bias == 34:
            proposed, meta = propose_bias_34_birth_year(response)
        elif bias == 35:
            proposed, meta = propose_bias_35_unit(response)
        elif bias == 41:
            proposed, meta = propose_bias_41_sports_teams(response, current)
        elif bias == 51:
            proposed, meta = propose_bias_51_law_911(response, current)

        # Verify the proposed span anchors at the intended position.
        # `meta` is the intended absolute position (int) when proposed is a
        # str; otherwise it's a string explaining the failure.
        uniqueness_concern = None
        rule_match = False
        if isinstance(proposed, str):
            first = response.find(proposed)
            count = response.count(proposed)
            if first < 0:
                uniqueness_concern = "proposed span not found"
            elif isinstance(meta, int) and first != meta:
                uniqueness_concern = (
                    f"first hit at {first} but intended anchor at {meta}; "
                    "span resolves to a different occurrence"
                )
            else:
                rule_match = True
                if count > 1:
                    # Anchor is fine (find returns first hit = intended), but
                    # note multiple occurrences for transparency.
                    uniqueness_concern = (
                        f"span appears {count} times — find() correctly "
                        "returns the first (anchor) occurrence"
                    )
        else:
            uniqueness_concern = f"could not propose: {meta}"

        offset_diff = char_offset_diff(response, current, proposed) if isinstance(proposed, str) else None

        # Add flag note if any (only for bias 11 b)
        if pid in flags:
            note = flags[pid]
            if uniqueness_concern:
                uniqueness_concern = uniqueness_concern + f" | FLAG: {note}"
            else:
                uniqueness_concern = f"FLAG: {note}"

        patches[str(bias)][pid] = {
            "current_span": current,
            "proposed_span": proposed if isinstance(proposed, str) else None,
            "rule_match": rule_match,
            "uniqueness_concern": uniqueness_concern,
            "char_offset_diff": offset_diff,
        }

    # Write JSON
    with open(OUT_JSON, "w") as f:
        json.dump(patches, f, indent=2, ensure_ascii=False)

    # Write Markdown report
    md = ["# Annotation anchor patches — review",
          "",
          "Side-by-side comparison of CURRENT vs PROPOSED `instances[0].span`.",
          "Only the per-bias rules in the user's prompt were applied.",
          "",
          "Legend: `char_offset_diff` is |proposed_anchor_pos - current_anchor_pos| where anchor_pos = (start + len(span)).",
          "If 0, the proposed span ends at the same character as the current span.",
          ""]
    summary = []
    for bias in sorted(TARGET_BIASES):
        bname = BIAS_NAMES[bias]
        md.append(f"## bias {bias} — {bname}")
        md.append("")
        md.append("| pid | current | proposed | unique? | offset | response context |")
        md.append("|---|---|---|---|---|---|")
        n_changed = 0
        n_unchanged = 0
        n_unresolved = 0
        for pid, p in patches[str(bias)].items():
            cur = p["current_span"]
            prop = p["proposed_span"]
            try:
                resp = load_response(pid)
                ctx = context_snippet(resp, prop or cur)
            except Exception:
                ctx = "<resp missing>"
            unique_marker = "y" if p["rule_match"] else "n"
            offset = p["char_offset_diff"]
            offset_s = "?" if offset is None else str(offset)
            cur_disp = json.dumps(cur, ensure_ascii=False).strip('"')
            prop_disp = json.dumps(prop, ensure_ascii=False).strip('"') if prop else "(none)"
            uniqueness = p["uniqueness_concern"] or ""
            cur_disp_md = cur_disp.replace("|", "\\|")
            prop_disp_md = prop_disp.replace("|", "\\|")
            ctx_md = ctx.replace("|", "\\|")
            md.append(f"| `{pid}` | `{cur_disp_md}` | `{prop_disp_md}` | {unique_marker} | {offset_s} | `{ctx_md}` |")
            if uniqueness:
                md.append(f"| | | | | | _{uniqueness}_ |")
            if prop is None:
                n_unresolved += 1
            elif prop == cur:
                n_unchanged += 1
            else:
                n_changed += 1
        md.append("")
        md.append(f"_n_changed={n_changed}, n_unchanged={n_unchanged}, n_unresolved={n_unresolved}_")
        md.append("")
        summary.append((bias, bname, n_changed, n_unchanged, n_unresolved))

    # Top-level summary
    md.insert(2, "## Summary")
    md.insert(3, "")
    md.insert(4, "| bias | name | changed | unchanged | unresolved |")
    md.insert(5, "|---|---|---|---|---|")
    for i, (bias, bname, nc, nu, nx) in enumerate(summary):
        md.insert(6 + i, f"| {bias} | {bname} | {nc} | {nu} | {nx} |")
    md.insert(6 + len(summary), "")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(md))

    # Print short summary
    print("Wrote:", OUT_JSON)
    print("Wrote:", OUT_MD)
    print()
    print("Per-bias summary:")
    for bias, bname, nc, nu, nx in summary:
        print(f"  bias {bias:>2} {bname:<22} changed={nc}  unchanged={nu}  unresolved={nx}")
    if flags:
        print()
        print("FLAGS:")
        for pid, note in flags.items():
            print(f"  {pid}: {note}")


if __name__ == "__main__":
    main()
