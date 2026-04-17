"""
Shared fixtures for utils/ tests, focused on the judge multi-provider refactor.

Integration tests are gated on API-key env vars. If a key is missing, those
tests are skipped — not failed — so the suite stays green on machines that
don't have credentials.

Markers registered in pyproject.toml:
    integration — test hits a live API (skipped without creds).
    slow        — test takes more than a few seconds.
"""

import os

import pytest


# Load .env once at module import so individual tests see OPENAI_API_KEY etc.
# dotenv is already a judge optional-dep; this is a no-op if not installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _has_key(env_name: str) -> bool:
    return bool(os.environ.get(env_name))


# =============================================================================
# Skip-if-key-missing fixtures
# =============================================================================


@pytest.fixture
def openai_key():
    """Skip the test if OPENAI_API_KEY is not set."""
    if not _has_key("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    return os.environ["OPENAI_API_KEY"]


@pytest.fixture
def anthropic_key():
    """Skip the test if ANTHROPIC_API_KEY is not set."""
    if not _has_key("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    return os.environ["ANTHROPIC_API_KEY"]


@pytest.fixture
def any_judge_key(openai_key, anthropic_key):
    """Skip unless BOTH OpenAI and Anthropic keys are set (for cross-backend tests)."""
    return {"openai": openai_key, "anthropic": anthropic_key}


# =============================================================================
# Reusable prompts / scales
# =============================================================================


@pytest.fixture
def coherent_short_text():
    """A sentence that any reasonable judge will score > 70 on coherence."""
    return "The cat sat on the mat."


@pytest.fixture
def rate_on_1_7_messages():
    """Paper-faithful 1-7 valence rating prompt for 'happy'.

    Used by cross-backend score_on_scale tests. Expected outcome: score
    solidly in [5, 7] because 'happy' is paradigmatically high-valence.
    """
    return [
        {
            "role": "user",
            "content": (
                'Rate the emotional valence of "happy" on a scale from 1 to 7. '
                "1 = extremely unpleasant, 4 = neutral, 7 = extremely pleasant. "
                "Respond with just a single integer 1-7:"
            ),
        }
    ]
