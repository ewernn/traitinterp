"""
Layer specification parsing.

Input: Layer spec string (e.g., "25,30,35,40", "0-75:5", "30%-60%", "all")
Output: Sorted list of unique layer indices

Usage:
    from utils.layers import parse_layers, resolve_layers
    layers = parse_layers("25,30,35,40", n_layers=80)
    concrete = resolve_layers("best,best+5", best_layer=20, available_layers={15, 20, 25})
"""

import re
from typing import List, Optional, Set


def parse_layers(layers_str: str, n_layers: int) -> List[int]:
    """Parse layer specification string into list of layer indices.

    Supports:
        - "all" — all layers
        - "16" — single layer
        - "5,10,15" — comma-separated list
        - "5-20" — inclusive range
        - "0-75:5" — range with step
        - "30%-60%" — percentage range (30% to 60% of depth)
        - Mixed: "5,10-15,20-30:5" — combines all above

    Args:
        layers_str: Layer specification string
        n_layers: Total number of model layers (for validation and percentage calc)

    Returns:
        Sorted list of unique layer indices within [0, n_layers)
    """
    if layers_str.strip().lower() == "all":
        return list(range(n_layers))

    layers = []
    for part in layers_str.split(','):
        part = part.strip()
        if not part:
            continue

        if '%' in part:
            # Percentage range: "30%-60%"
            pct_parts = part.replace('%', '').split('-')
            start_pct = int(pct_parts[0]) / 100
            end_pct = int(pct_parts[1]) / 100 if len(pct_parts) > 1 else start_pct
            start = int(n_layers * start_pct)
            end = int(n_layers * end_pct)
            layers.extend(range(start, end + 1))
        elif ':' in part:
            # Range with step: "0-75:5"
            range_part, step = part.split(':')
            start, end = range_part.split('-')
            layers.extend(range(int(start), int(end) + 1, int(step)))
        elif '-' in part and not part.startswith('-'):
            # Simple range: "5-20"
            start, end = part.split('-')
            layers.extend(range(int(start), int(end) + 1))
        else:
            # Single layer: "16"
            layers.append(int(part))

    return sorted(set(l for l in layers if 0 <= l < n_layers))


def resolve_layers(layers_spec: str, best_layer: Optional[int], available_layers: Set[int]) -> List[int]:
    """Resolve layer specs like 'best,best+5' into concrete layer numbers.

    Supports specs: 'best', 'best+N', 'best-N', integer literals, or comma-separated mix.
    Snaps to nearest available layer if exact match not captured.
    """
    result = []
    for spec in layers_spec.split(','):
        spec = spec.strip()
        if 'best' in spec:
            if best_layer is None:
                print(f"    Warning: cannot resolve '{spec}' (no steering data), skipping")
                continue
            if spec == 'best':
                layer = best_layer
            else:
                m = re.match(r'best\s*([+-])\s*(\d+)', spec)
                if not m:
                    raise ValueError(f"Invalid layer spec: '{spec}'. Use: best, best+N, best-N, or integer")
                op, val = m.group(1), int(m.group(2))
                layer = best_layer + val if op == '+' else best_layer - val
        else:
            layer = int(spec)

        if layer in available_layers:
            if layer not in result:
                result.append(layer)
        elif available_layers:
            closest = min(available_layers, key=lambda l: abs(l - layer))
            print(f"    Layer {layer} not captured, snapping to nearest: {closest}")
            if closest not in result:
                result.append(closest)
        else:
            print(f"    Warning: layer {layer} not in raw activations, skipping")

    return result
