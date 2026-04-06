"""
Position DSL: resolve token position specifiers to concrete index ranges.

Supports single-turn and multiturn conversations.

Single-turn syntax:
    response[:5]      first 5 response tokens
    response[:]       all response tokens
    prompt[-1]        last prompt token
    all[:]            entire sequence

Multiturn syntax (turn index is 0-based, defaults to last turn of that role):
    turn[2]:response[:5]   first 5 response tokens of turn 2
    turn[-1]:response[:]   all response tokens of the last turn
    turn[1]:prompt[-1]     last prompt token of turn 1

Frames:
    prompt       user message tokens
    response     assistant response tokens (after </think> if thinking model)
    thinking     assistant's <thinking>...</thinking> block
    system       system message tokens
    tool_call    tool call tokens
    tool_result  tool result tokens
    all          entire sequence (ignores turn selector)

Input:  position string + sequence metadata (prompt_len or turn_boundaries)
Output: concrete (start_idx, end_idx) absolute token indices

Usage:
    from utils.positions import resolve_position, tokens_needed, resolve_max_new_tokens

    start, end = resolve_position("response[:5]", prompt_len=42, seq_len=60)
    needed = tokens_needed("response[:5]")  # 5
    max_tok = resolve_max_new_tokens("response[:5]")  # 5
"""

import re
from typing import Optional, Tuple, List, Dict

# Supported frame types
FRAMES = {'prompt', 'response', 'thinking', 'system', 'tool_call', 'tool_result', 'all'}

# Regex patterns
_TURN_PREFIX = re.compile(r'^turn\[(-?\d+)\]:(.+)$')
_FRAME_SLICE = re.compile(r'^(\w+)\[(.+)\]$')


def parse_position(position: str) -> Tuple[str, Optional[int], Optional[int], Optional[int]]:
    """Parse a position specifier into (frame, turn_idx, start, stop).

    Args:
        position: Position string like 'response[:5]' or 'turn[2]:thinking[:]'

    Returns:
        (frame, turn_idx, start, stop) where:
            frame    — one of FRAMES
            turn_idx — None (default to last turn) or integer
            start    — None or int (slice start)
            stop     — None or int (slice stop, exclusive)
    """
    # Check for turn prefix
    turn_idx = None
    rest = position
    m = _TURN_PREFIX.match(position)
    if m:
        turn_idx = int(m.group(1))
        rest = m.group(2)

    # Parse frame[slice]
    fm = _FRAME_SLICE.match(rest)
    if not fm:
        raise ValueError(
            f"Invalid position: '{position}'. "
            f"Use frame[slice] or turn[N]:frame[slice]. "
            f"Frames: {', '.join(sorted(FRAMES))}. "
            f"Examples: 'response[:5]', 'prompt[-1]', 'turn[0]:response[:]'"
        )

    frame = fm.group(1)
    if frame not in FRAMES:
        raise ValueError(
            f"Unknown frame '{frame}' in position '{position}'. "
            f"Valid frames: {', '.join(sorted(FRAMES))}"
        )

    slice_str = fm.group(2).strip()

    # Parse slice
    if slice_str == ':':
        return frame, turn_idx, None, None

    if ':' in slice_str:
        parts = slice_str.split(':', 1)
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        return frame, turn_idx, start, stop

    # Single index: e.g., response[-1] → slice of one token
    idx = int(slice_str)
    if idx >= 0:
        return frame, turn_idx, idx, idx + 1
    elif idx == -1:
        return frame, turn_idx, -1, None
    else:
        return frame, turn_idx, idx, idx + 1


def _apply_slice(start: Optional[int], stop: Optional[int],
                 frame_start: int, frame_end: int) -> Tuple[int, int]:
    """Apply a parsed start/stop slice within a frame's absolute boundaries."""
    frame_len = frame_end - frame_start

    if start is None:
        abs_start = frame_start
    elif start >= 0:
        abs_start = frame_start + min(start, frame_len)
    else:
        abs_start = frame_end + start

    if stop is None:
        abs_end = frame_end
    elif stop >= 0:
        abs_end = frame_start + min(stop, frame_len)
    else:
        abs_end = frame_end + stop

    # Clamp to frame boundaries
    abs_start = max(frame_start, min(abs_start, frame_end))
    abs_end = max(abs_start, min(abs_end, frame_end))

    return abs_start, abs_end


def _get_frame_bounds(
    frame: str,
    turn_idx: Optional[int],
    prompt_len: int,
    seq_len: int,
    turn_boundaries: Optional[List[Dict]] = None,
) -> Tuple[int, int]:
    """Get (frame_start, frame_end) for a frame, handling single-turn and multiturn."""

    # 'all' ignores turn indexing
    if frame == 'all':
        return 0, seq_len

    # Single-turn mode (no turn_boundaries)
    if turn_boundaries is None:
        if turn_idx is not None:
            raise ValueError(
                f"Turn indexing (turn[{turn_idx}]) requires turn_boundaries. "
                f"Pass turn_boundaries for multiturn sequences."
            )
        if frame == 'prompt':
            return 0, prompt_len
        elif frame == 'response':
            return prompt_len, seq_len
        elif frame in ('thinking', 'system', 'tool_call', 'tool_result'):
            raise ValueError(
                f"Frame '{frame}' requires turn_boundaries for multiturn sequences. "
                f"For single-turn, use 'prompt' or 'response'."
            )
        else:
            raise ValueError(f"Unknown frame: {frame}")

    # Multiturn mode
    # Filter turns by role matching the requested frame
    role_map = {
        'prompt': 'user',
        'response': 'assistant',
        'thinking': 'assistant',
        'system': 'system',
        'tool_call': 'tool',
        'tool_result': 'tool_result',
    }
    target_role = role_map.get(frame)
    if target_role is None:
        raise ValueError(f"Unknown frame: {frame}")

    matching_turns = [tb for tb in turn_boundaries if tb['role'] == target_role]

    if turn_idx is not None:
        # Explicit turn index — index into matching turns of this role
        try:
            turn = matching_turns[turn_idx]
        except IndexError:
            raise ValueError(
                f"Turn index {turn_idx} out of range: only {len(matching_turns)} "
                f"'{target_role}' turns in this conversation"
            )
    else:
        # Default: last turn of this role
        if not matching_turns:
            raise ValueError(f"No '{target_role}' turns found in turn_boundaries")
        turn = matching_turns[-1]

    # Resolve frame bounds within the turn
    turn_start = turn['token_start']
    turn_end = turn['token_end']

    if frame == 'thinking':
        # Thinking block is within an assistant turn
        think_start = turn.get('thinking_start', turn_start)
        think_end = turn.get('thinking_end', turn_start)  # empty if no thinking
        return think_start, think_end
    elif frame == 'response' and turn.get('thinking_end'):
        # Response excludes the thinking block
        return turn['thinking_end'], turn_end
    else:
        return turn_start, turn_end


def resolve_position(
    position: str,
    prompt_len: int,
    seq_len: int,
    turn_boundaries: Optional[List[Dict]] = None,
) -> Tuple[int, int]:
    """Resolve a position string to concrete (start_idx, end_idx) token indices.

    Args:
        position: Position specifier (e.g., 'response[:5]', 'turn[0]:thinking[:]')
        prompt_len: Number of prompt tokens (single-turn mode)
        seq_len: Total sequence length
        turn_boundaries: List of turn boundary dicts for multiturn sequences.
            Each dict has: role, token_start, token_end, and optionally
            thinking_start, thinking_end, has_thinking, has_tool_calls.
            Produced by tokenizer during multiturn encoding.

    Returns:
        (start_idx, end_idx) — absolute token indices, half-open interval
    """
    frame, turn_idx, start, stop = parse_position(position)
    frame_start, frame_end = _get_frame_bounds(
        frame, turn_idx, prompt_len, seq_len, turn_boundaries
    )
    return _apply_slice(start, stop, frame_start, frame_end)


def tokens_needed(position: str) -> Optional[int]:
    """Return minimum response tokens needed for this position, or None if undeterminable.

    Used to auto-set --max-new-tokens at pipeline startup.
    Returns 0 for prompt-only positions, the stop index for response[:N],
    or None for open-ended positions like response[:].
    """
    frame, _turn_idx, start, stop = parse_position(position)
    if frame == 'prompt':
        return 0
    if frame == 'response':
        # response[:5] → need 5 tokens. start=None means 0.
        effective_start = start if start is not None else 0
        if effective_start >= 0 and stop is not None and stop > 0:
            return stop
    return None


def resolve_max_new_tokens(position: str, user_value: Optional[int] = None) -> int:
    """Resolve max_new_tokens from position specifier and optional user override.

    Auto-determines the minimum generation length needed for the position.
    Raises ValueError if user_value is less than what the position requires.
    """
    needed = tokens_needed(position)
    if user_value is None:
        return needed if needed is not None else 16
    if needed is not None and user_value < needed:
        raise ValueError(
            f"--max-new-tokens {user_value} is less than {needed} "
            f"required for position '{position}'"
        )
    return user_value
