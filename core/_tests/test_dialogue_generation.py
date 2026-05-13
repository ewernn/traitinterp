"""Tests for experiments/ant_emotion_concepts/scripts/dialogue_generation.py

Covers:
  - Lightweight mode (default) preserves legacy schema + paraphrased prompt.
  - Full mode reads paper-verbatim template, groups by (h_emo, a_emo, topic),
    parses [dialogue N] delimiters, post-hoc rewrites Person:/AI: → Human:/Assistant:.
  - _person_ai_to_human_assistant only rewrites speaker labels at line starts,
    not mentions inside dialogue body.

Run: pytest core/_tests/test_dialogue_generation.py -v
"""
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "experiments" / "ant_emotion_concepts" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from dialogue_generation import (  # noqa: E402
    _person_ai_to_human_assistant,
    generate_dialogues,
    _TWO_SPEAKER_TEMPLATE_PATH,
)


# =============================================================================
# Post-hoc Person/AI → Human/Assistant conversion
# =============================================================================

class TestPersonAiConversion:
    def test_simple_labels_rewritten(self):
        text = "Person: Hello.\n\nAI: Hi there.\n\nPerson: Bye.\n\nAI: Goodbye."
        out = _person_ai_to_human_assistant(text)
        assert "Person:" not in out
        assert "AI:" not in out
        assert "Human: Hello." in out
        assert "Assistant: Hi there." in out
        assert "Human: Bye." in out
        assert "Assistant: Goodbye." in out

    def test_body_mentions_not_rewritten(self):
        """A character saying "I saw a person at the AI lab" must not be touched."""
        text = (
            "Person: I saw a person at the AI lab yesterday.\n\n"
            "AI: That's interesting. The person you saw might work in AI safety."
        )
        out = _person_ai_to_human_assistant(text)
        # Speaker labels at line starts get rewritten:
        assert out.startswith("Human: I saw a person at the AI lab yesterday.")
        assert "\n\nAssistant: That's interesting." in out
        # Body text untouched:
        assert "a person at the AI lab" in out
        assert "The person you saw might work in AI safety." in out

    def test_leading_whitespace_preserved(self):
        text = "  Person: indented turn\n\n    AI: deeper indent"
        out = _person_ai_to_human_assistant(text)
        assert "  Human: indented turn" in out
        assert "    Assistant: deeper indent" in out

    def test_already_converted_passthrough(self):
        text = "Human: hi\n\nAssistant: hey"
        assert _person_ai_to_human_assistant(text) == text


# =============================================================================
# generate_dialogues full mode (mocked generation)
# =============================================================================

def _install_fake_generate_batch(monkeypatch, response_by_index):
    """Patch generate_batch so each call returns the i-th canned response."""
    calls = []

    def fake_generate_batch(model, tokenizer, prompts, max_new_tokens=None,
                            temperature=0.0, seed=None):
        calls.append({"prompts": list(prompts), "seed": seed,
                      "max_new_tokens": max_new_tokens})
        return [response_by_index[i] for i in range(len(prompts))]

    monkeypatch.setattr("dialogue_generation.generate_batch", fake_generate_batch)
    return calls


class TestGenerateDialoguesLightweight:
    def test_lightweight_default_preserves_legacy_schema(self, monkeypatch):
        """Lightweight returns one dialogue per call with current schema."""
        canned = [
            "Human: hi.\n\nAssistant: hey.",
            "Human: how are you?\n\nAssistant: fine.",
        ]
        calls = _install_fake_generate_batch(monkeypatch, canned)

        out = generate_dialogues(
            model=None, tokenizer=None,
            emotions=["happy", "sad"],
            n_dialogues=2,
            max_new_tokens=384, temperature=0.7, seed=42,
        )

        assert len(out) == 2
        assert set(out[0].keys()) == {"id", "human_emotion", "assistant_emotion",
                                      "text", "generation_prompt"}
        # Lightweight: paraphrased prompt content
        assert "Write a short conversation" in out[0]["generation_prompt"]
        # Single batched generate_batch call with N prompts
        assert len(calls) == 1
        assert len(calls[0]["prompts"]) == 2

    def test_invalid_replication_level_raises(self, monkeypatch):
        _install_fake_generate_batch(monkeypatch, [""])
        with pytest.raises(ValueError, match="replication_level"):
            generate_dialogues(
                model=None, tokenizer=None,
                emotions=["happy"],
                n_dialogues=1,
                replication_level="medium",
            )


class TestGenerateDialoguesFull:
    def test_full_mode_loads_paper_template(self, monkeypatch):
        """Full mode reads prompts/two_speaker_dialogue.txt and uses it as the prompt."""
        assert _TWO_SPEAKER_TEMPLATE_PATH.exists(), \
            "Paper template missing — should ship with ant_emotion_concepts"

        canned = [
            # 2 dialogues for cell 0
            "[dialogue 1]\n\nPerson: a1\n\nAI: a2\n\n[dialogue 2]\n\nPerson: b1\n\nAI: b2",
        ]
        calls = _install_fake_generate_batch(monkeypatch, canned)

        out = generate_dialogues(
            model=None, tokenizer=None,
            emotions=["happy"],  # only one emotion → all pairs are (happy, happy)
            n_dialogues=2,
            max_new_tokens=2048, temperature=0.7, seed=42,
            replication_level="full",
            topics=["a chance encounter"],
        )

        # Both dialogues group into one cell → one batched call
        assert len(calls) == 1
        prompt = calls[0]["prompts"][0]
        # Paper-verbatim phrases must appear in the rendered prompt
        assert "Write 2 different dialogues" in prompt
        assert "happy" in prompt  # both person_emotion and ai_emotion = "happy"
        assert "a chance encounter" in prompt

        # Both dialogues parsed
        assert len(out) == 2
        # Post-hoc conversion happened
        for d in out:
            assert "Person:" not in d["text"]
            assert "AI:" not in d["text"]
            assert "Human:" in d["text"]
            assert "Assistant:" in d["text"]

    def test_full_mode_groups_by_emotion_pair(self, monkeypatch):
        """Different (h_emo, a_emo) pairs → separate batched calls."""
        # With seed=0 + emotions=[A, B], the random.choice sequence will produce
        # some mix of (A,A), (A,B), (B,A), (B,B) — at least 2 distinct cells.
        canned = [
            "[dialogue 1]\n\nPerson: x\n\nAI: y",
        ] * 8  # 8 possible cells max
        calls = _install_fake_generate_batch(monkeypatch, canned)

        generate_dialogues(
            model=None, tokenizer=None,
            emotions=["happy", "sad"],
            n_dialogues=8,
            max_new_tokens=2048, temperature=0.7, seed=0,
            replication_level="full",
            topics=["a topic"],
        )

        # n_dialogues=8 across 2 emotions × 2 speakers = up to 4 distinct cells.
        # Single topic → at least 2 cells, at most 4.
        assert 1 < len(calls[0]["prompts"]) <= 4

    def test_full_mode_under_production_warning(self, monkeypatch, capsys):
        """When the model returns fewer dialogues than requested, log a warning."""
        # Request 4 in one cell but only get 1
        canned = ["[dialogue 1]\n\nPerson: only\n\nAI: one"]
        _install_fake_generate_batch(monkeypatch, canned)

        out = generate_dialogues(
            model=None, tokenizer=None,
            emotions=["happy"],
            n_dialogues=4,
            max_new_tokens=2048, temperature=0.7, seed=0,
            replication_level="full",
            topics=["one topic"],
        )

        captured = capsys.readouterr()
        assert "under-produced" in captured.out
        assert "1/4" in captured.out
        # Partial output kept
        assert len(out) == 1

    def test_full_mode_template_missing_raises(self, monkeypatch, tmp_path):
        """If the paper template file is missing, raise FileNotFoundError."""
        bogus_path = tmp_path / "nonexistent.txt"
        monkeypatch.setattr(
            "dialogue_generation._TWO_SPEAKER_TEMPLATE_PATH", bogus_path,
        )
        _install_fake_generate_batch(monkeypatch, [""])

        with pytest.raises(FileNotFoundError, match="Two-speaker paper template"):
            generate_dialogues(
                model=None, tokenizer=None,
                emotions=["happy"],
                n_dialogues=1,
                replication_level="full",
            )

    def test_full_mode_default_topic_when_none(self, monkeypatch):
        """topics=None falls through to a generic placeholder topic, no crash."""
        canned = ["[dialogue 1]\n\nPerson: a\n\nAI: b"]
        _install_fake_generate_batch(monkeypatch, canned)

        out = generate_dialogues(
            model=None, tokenizer=None,
            emotions=["happy"],
            n_dialogues=1,
            replication_level="full",
            topics=None,
        )
        assert len(out) == 1
