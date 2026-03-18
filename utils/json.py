"""
Compact JSON serialization with readable structure but single-line arrays.

Usage:
    from utils.json import dump_compact

    with open(path, 'w') as f:
        dump_compact(data, f)
"""

import json
import re


def dumps_compact(obj) -> str:
    """Return compact JSON string: indented structure, single-line primitive arrays."""
    s = json.dumps(obj, indent=2)
    return re.sub(
        r'\[\s*\n\s*([^\[\]{}]*?)\s*\n\s*\]',
        lambda m: '[' + ', '.join(x.strip() for x in m.group(1).split(',')) + ']',
        s,
        flags=re.DOTALL,
    )


def dump_compact(obj, f):
    """Write compact JSON to file handle."""
    f.write(dumps_compact(obj))
