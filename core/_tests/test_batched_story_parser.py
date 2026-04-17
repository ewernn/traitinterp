"""
Tests for utils.extraction.parse_story_blocks.

Parser handles paper-style batched-generation responses where the model emits
[story N]-delimited blocks. Empirical failure modes from Llama 3.3 70B testing:
duplicate delimiters when the model restarts the list mid-response.

Run: pytest core/_tests/test_batched_story_parser.py -v
"""

import json

import pytest

from utils.extraction import parse_story_blocks


# =============================================================================
# Happy path — clean delimiter-separated responses
# =============================================================================

class TestCleanParse:
    def test_three_stories_parsed_in_order(self):
        response = (
            "[story 1]\nOnce upon a time there was a dog.\n\n"
            "[story 2]\nA second tale about a cat.\n\n"
            "[story 3]\nAnd finally a fish story."
        )
        stories = parse_story_blocks(response, expected_n=3)
        assert len(stories) == 3
        assert stories[0].startswith("Once upon a time")
        assert stories[1].startswith("A second tale")
        assert stories[2].startswith("And finally")

    def test_content_strips_leading_and_trailing_whitespace(self):
        response = "[story 1]\n\n\n   Hello.   \n\n\n"
        stories = parse_story_blocks(response, expected_n=1)
        assert stories == ["Hello."]

    def test_single_story(self):
        response = "[story 1] Just one story here."
        stories = parse_story_blocks(response, expected_n=1)
        assert stories == ["Just one story here."]

    def test_caps_at_expected_n(self):
        response = "\n".join(f"[story {i}] text {i}" for i in range(1, 6))
        stories = parse_story_blocks(response, expected_n=3)
        assert len(stories) == 3
        assert stories[0] == "text 1"
        assert stories[2] == "text 3"


# =============================================================================
# Duplicate-delimiter handling (the "calm" failure mode from Q-C3 test)
# =============================================================================

class TestDuplicateDelimiters:
    def test_model_restarts_list_first_pass_wins(self):
        """Model emits [story 1]..[story 2]..[story 1]..[story 2].. — keep first copy."""
        response = (
            "[story 1] first pass story one.\n"
            "[story 2] first pass story two.\n"
            "[story 1] second pass story one.\n"
            "[story 2] second pass story two."
        )
        stories = parse_story_blocks(response, expected_n=2)
        assert len(stories) == 2
        assert stories[0] == "first pass story one."
        assert stories[1] == "first pass story two."

    def test_calm_case_24_delimiters_expects_12(self):
        """Simulates the calm failure mode: 2 full restarts of a 12-batch."""
        lines = [f"[story {i}] first {i}" for i in range(1, 13)]
        lines += [f"[story {i}] restart {i}" for i in range(1, 13)]
        response = "\n".join(lines)
        stories = parse_story_blocks(response, expected_n=12)
        assert len(stories) == 12
        assert all(s.startswith("first") for s in stories), (
            f"Expected all 'first' pass stories, got: {[s[:10] for s in stories]}"
        )


# =============================================================================
# Under-production (model cuts off, failure case requiring caller retry)
# =============================================================================

class TestUnderProduction:
    def test_empty_response_returns_empty_list(self):
        assert parse_story_blocks("", expected_n=12) == []

    def test_no_delimiters_returns_empty_list(self):
        response = "Once upon a time there was a cat. She was a good cat."
        assert parse_story_blocks(response, expected_n=3) == []

    def test_fewer_stories_than_expected_returns_what_was_found(self):
        response = "[story 1] alpha\n[story 2] beta"
        stories = parse_story_blocks(response, expected_n=12)
        assert stories == ["alpha", "beta"]

    def test_gaps_in_numbering_still_parsed(self):
        """Model emits [story 1], [story 3], [story 7] — no [story 2,4-6]."""
        response = "[story 1] one\n[story 3] three\n[story 7] seven"
        stories = parse_story_blocks(response, expected_n=12)
        assert stories == ["one", "three", "seven"]


# =============================================================================
# Format tolerance (case, spacing, inline vs block)
# =============================================================================

class TestFormatTolerance:
    def test_case_insensitive(self):
        response = "[STORY 1] A\n[Story 2] B\n[story 3] C"
        stories = parse_story_blocks(response, expected_n=3)
        assert stories == ["A", "B", "C"]

    def test_internal_bracket_whitespace(self):
        response = "[ story 1 ] A\n[story  2] B\n[  STORY   3  ] C"
        stories = parse_story_blocks(response, expected_n=3)
        assert stories == ["A", "B", "C"]

    def test_inline_delimiter_not_on_own_line(self):
        """Delimiter can appear inline, not as its own line."""
        response = "Here are the stories. [story 1] First. [story 2] Second."
        stories = parse_story_blocks(response, expected_n=2)
        assert stories == ["First.", "Second."]

    def test_trailing_text_after_last_delimiter_included(self):
        """Last block captures through end of response."""
        response = (
            "[story 1] First story.\n"
            "[story 2] Second story, and trailing commentary follows."
        )
        stories = parse_story_blocks(response, expected_n=2)
        assert len(stories) == 2
        assert stories[1] == "Second story, and trailing commentary follows."


# =============================================================================
# Multi-paragraph stories (realistic paper-style)
# =============================================================================

class TestMultiParagraphStories:
    def test_multi_paragraph_story_preserved(self):
        response = (
            "[story 1]\n"
            "First paragraph of the story.\n\n"
            "Second paragraph with more detail.\n\n"
            "[story 2]\n"
            "Another story starts here."
        )
        stories = parse_story_blocks(response, expected_n=2)
        assert len(stories) == 2
        assert "First paragraph" in stories[0]
        assert "Second paragraph" in stories[0]
        assert stories[1].startswith("Another story")


# =============================================================================
# Real-Llama failure-mode tolerance (trailing colon, markdown bold, etc.)
# =============================================================================

class TestLlamaFormatQuirks:
    def test_trailing_colon_in_delimiter(self):
        """Llama frequently adds a trailing colon: `[story 1:]` — must match."""
        response = "[story 1:] First story content.\n[story 2:] Second."
        stories = parse_story_blocks(response, expected_n=2)
        assert stories == ["First story content.", "Second."]

    def test_markdown_bold_wrapped_delimiter(self):
        """`**[story 1]**` — asterisks must be consumed, not leaked into content."""
        response = "**[story 1]** First story.\n**[story 2]** Second story."
        stories = parse_story_blocks(response, expected_n=2)
        assert stories == ["First story.", "Second story."]
        assert "**" not in stories[0]
        assert "**" not in stories[1]

    def test_markdown_bold_and_trailing_colon_combined(self):
        """Real-world: `**[story 1:]**` combines both quirks."""
        response = "**[story 1:]** alpha.\n**[story 2:]** beta."
        stories = parse_story_blocks(response, expected_n=2)
        assert stories == ["alpha.", "beta."]


# =============================================================================
# Defensive input handling
# =============================================================================

class TestInputValidation:
    def test_non_str_raises_type_error(self):
        with pytest.raises(TypeError, match="response must be str"):
            parse_story_blocks(None, expected_n=3)
        with pytest.raises(TypeError, match="response must be str"):
            parse_story_blocks(b"[story 1] bytes", expected_n=3)
        with pytest.raises(TypeError, match="response must be str"):
            parse_story_blocks(["[story 1]"], expected_n=3)


# =============================================================================
# Edge cases (adjacent delimiters, empty content blocks)
# =============================================================================

class TestAdjacentDelimiters:
    def test_adjacent_delimiters_drop_empty_blocks(self):
        """Delimiters back-to-back — block between them is empty, skipped silently.

        Documents the `if block:` guard: empty first-occurrence blocks do not
        add entries to the returned list. Caller sees under-production and
        retries.
        """
        response = "[story 1][story 2] Only second has content."
        stories = parse_story_blocks(response, expected_n=2)
        # Story 1's block is empty (adjacent to story 2's delimiter), dropped.
        # Story 2's block contains the content.
        assert stories == ["Only second has content."]


# =============================================================================
# _generate_stories_batched_and_write (Increment 3 integration helper)
# =============================================================================

class TestGenerateStoriesBatched:
    """Tests the batched-generation helper with generate_batch + format_prompt
    monkeypatched to avoid loading a real model."""

    def _make_fake_response(self, n, prefix="story"):
        """Build a canned batched response with N story blocks."""
        return "\n\n".join(
            f"[story {i+1}] {prefix} {i+1} content."
            for i in range(n)
        )

    def _patch_generation(self, monkeypatch, canned_response):
        """Replace generate_batch + format_prompt with test doubles that capture
        arguments. Returns a dict with `generate_calls` and `format_calls` lists
        so tests can assert against either.
        """
        generate_calls = []
        format_calls = []

        def fake_generate_batch(model, tokenizer, prompts, max_new_tokens=None,
                                 temperature=0.0, seed=None):
            generate_calls.append({'prompts': list(prompts), 'seed': seed, 'temperature': temperature})
            return [canned_response for _ in prompts]

        def fake_format_prompt(prompt, tokenizer, use_chat_template=False, system_prompt=None):
            format_calls.append({
                'prompt': prompt,
                'system_prompt': system_prompt,
                'use_chat_template': use_chat_template,
            })
            # Produce a recognizable formatted string to distinguish from raw prompts
            return f"<formatted>sys={system_prompt!r}|user={prompt!r}"

        monkeypatch.setattr('utils.extraction.generate_batch', fake_generate_batch)
        monkeypatch.setattr('utils.extraction.format_prompt', fake_format_prompt)
        return {'generate_calls': generate_calls, 'format_calls': format_calls}

    def test_basic_flow_emits_one_record_per_story(self, tmp_path, monkeypatch):
        from utils.extraction import _generate_stories_batched_and_write

        canned = self._make_fake_response(3)
        calls = self._patch_generation(monkeypatch, canned)

        topics = ['topic A', 'topic B']
        out = tmp_path / 'pos.json'
        template = "Write {n_stories} stories on {topic} about {emotion}."

        n = _generate_stories_batched_and_write(
            topics=topics,
            output_path=out,
            model=None, tokenizer=None, use_chat_template=False,
            max_new_tokens=512, temperature=0.7, seed=42,
            batched_template=template, template_kwargs={'emotion': 'amazed'},
            stories_per_batch=3,
        )

        # 2 topics × 3 stories each = 6 records
        assert n == 6
        records = json.loads(out.read_text())
        assert len(records) == 6

        # Each record has the expected schema. Paper prompt lands in system_prompt
        # (role fidelity — paper labels it as a system prompt). User turn is empty.
        r = records[0]
        assert set(r.keys()) == {'prompt', 'response', 'system_prompt', 'story_idx', 'topic'}
        assert r['prompt'] == ''
        assert 'amazed' in r['system_prompt']
        assert r['topic'] == 'topic A'
        # story_idx is 0-based within each topic
        assert records[0]['story_idx'] == 0
        assert records[2]['story_idx'] == 2
        assert records[3]['story_idx'] == 0  # restart for next topic
        # topic appears contiguously (all story_idx for topic A before topic B)
        assert records[0]['topic'] == records[1]['topic'] == records[2]['topic']
        assert records[3]['topic'] == records[4]['topic'] == records[5]['topic']

    def test_per_topic_seed_increments(self, tmp_path, monkeypatch):
        from utils.extraction import _generate_stories_batched_and_write

        calls = self._patch_generation(monkeypatch, self._make_fake_response(2))

        _generate_stories_batched_and_write(
            topics=['t1', 't2', 't3'],
            output_path=tmp_path / 'pos.json',
            model=None, tokenizer=None, use_chat_template=False,
            max_new_tokens=512, temperature=0.7, seed=100,
            batched_template="{n_stories} {topic} {emotion}",
            template_kwargs={'emotion': 'sad'}, stories_per_batch=2,
        )

        # 3 topics → 3 generate_batch calls with seeds 100, 101, 102
        assert [c['seed'] for c in calls['generate_calls']] == [100, 101, 102]

    def test_seed_none_passes_through(self, tmp_path, monkeypatch):
        from utils.extraction import _generate_stories_batched_and_write

        calls = self._patch_generation(monkeypatch, self._make_fake_response(1))

        _generate_stories_batched_and_write(
            topics=['t1', 't2'],
            output_path=tmp_path / 'pos.json',
            model=None, tokenizer=None, use_chat_template=False,
            max_new_tokens=512, temperature=0.7, seed=None,
            batched_template="{n_stories} {topic} {emotion}",
            template_kwargs={'emotion': 'x'}, stories_per_batch=1,
        )
        assert [c['seed'] for c in calls['generate_calls']] == [None, None]

    def test_under_production_kept_partial(self, tmp_path, monkeypatch, capsys):
        from utils.extraction import _generate_stories_batched_and_write

        # Model produces only 2 stories when 5 are expected
        canned = self._make_fake_response(2)
        self._patch_generation(monkeypatch, canned)

        n = _generate_stories_batched_and_write(
            topics=['topic A'],
            output_path=tmp_path / 'pos.json',
            model=None, tokenizer=None, use_chat_template=False,
            max_new_tokens=512, temperature=0.7, seed=42,
            batched_template="{n_stories} {topic} {emotion}",
            template_kwargs={'emotion': 'x'}, stories_per_batch=5,
        )

        # Partial kept, no retry
        assert n == 2
        captured = capsys.readouterr()
        assert 'under-produced' in captured.out
        assert '2/5' in captured.out

    def test_zero_stories_still_writes_empty_list(self, tmp_path, monkeypatch):
        from utils.extraction import _generate_stories_batched_and_write

        # Model emits no delimiters at all
        self._patch_generation(monkeypatch, "Sorry, I cannot write those stories.")

        out = tmp_path / 'pos.json'
        n = _generate_stories_batched_and_write(
            topics=['topic A'],
            output_path=out,
            model=None, tokenizer=None, use_chat_template=False,
            max_new_tokens=512, temperature=0.7, seed=42,
            batched_template="{n_stories} {topic} {emotion}",
            template_kwargs={'emotion': 'x'}, stories_per_batch=3,
        )
        assert n == 0
        assert json.loads(out.read_text()) == []

    def test_generic_template_kwargs_multi_slot(self, tmp_path, monkeypatch):
        """Two-speaker template shape: {person_emotion}, {ai_emotion} — no emotion.

        Proves template_kwargs is generic enough for stages 3 + two-speaker
        without requiring another helper function.
        """
        from utils.extraction import _generate_stories_batched_and_write

        self._patch_generation(monkeypatch, self._make_fake_response(2))

        out = tmp_path / 'pos.json'
        template = "Write {n_stories} dialogues about {topic}. Person: {person_emotion}. AI: {ai_emotion}."
        n = _generate_stories_batched_and_write(
            topics=['t1'],
            output_path=out,
            model=None, tokenizer=None, use_chat_template=False,
            max_new_tokens=512, temperature=0.7, seed=42,
            batched_template=template,
            template_kwargs={'person_emotion': 'calm', 'ai_emotion': 'excited'},
            stories_per_batch=2,
        )
        assert n == 2
        rec = json.loads(out.read_text())[0]
        # Paper text (with template substitutions) lands in system_prompt, not prompt
        assert rec['prompt'] == ''
        assert 'calm' in rec['system_prompt']
        assert 'excited' in rec['system_prompt']

    def test_missing_placeholder_kwarg_raises_key_error(self, tmp_path, monkeypatch):
        """F3 failure mode: template references {emotion} but caller supplies {}.

        Python's native str.format() raises KeyError — fail loud, no retry.
        """
        from utils.extraction import _generate_stories_batched_and_write

        self._patch_generation(monkeypatch, self._make_fake_response(1))

        with pytest.raises(KeyError, match='emotion'):
            _generate_stories_batched_and_write(
                topics=['t1'],
                output_path=tmp_path / 'pos.json',
                model=None, tokenizer=None, use_chat_template=False,
                max_new_tokens=512, temperature=0.7, seed=42,
                batched_template="Write {n_stories} on {topic} about {emotion}.",
                template_kwargs={},  # missing 'emotion'
                stories_per_batch=1,
            )

    def test_extra_template_kwargs_ignored(self, tmp_path, monkeypatch):
        """Python's str.format() silently ignores extra kwargs — template_kwargs
        can carry more keys than the template references without error.
        """
        from utils.extraction import _generate_stories_batched_and_write

        self._patch_generation(monkeypatch, self._make_fake_response(1))

        n = _generate_stories_batched_and_write(
            topics=['t1'],
            output_path=tmp_path / 'pos.json',
            model=None, tokenizer=None, use_chat_template=False,
            max_new_tokens=512, temperature=0.7, seed=42,
            batched_template="Write {n_stories} on {topic}.",  # no {emotion}
            template_kwargs={'emotion': 'amazed', 'unused': 'extra'},
            stories_per_batch=1,
        )
        assert n == 1

    def test_role_fidelity_paper_text_lands_in_system_role(self, tmp_path, monkeypatch):
        """Paper labels the prompt as a SYSTEM prompt (line 1376). Verify
        format_prompt is invoked with prompt='' and the rendered template
        passed as system_prompt= kwarg — NOT as the user turn.
        """
        from utils.extraction import _generate_stories_batched_and_write

        calls = self._patch_generation(monkeypatch, self._make_fake_response(2))

        template = (
            "Write {n_stories} different stories based on {topic}. "
            "Character feeling {emotion}. Never say '{emotion}'."
        )
        _generate_stories_batched_and_write(
            topics=['topic A', 'topic B'],
            output_path=tmp_path / 'pos.json',
            model=None, tokenizer=None, use_chat_template=True,
            max_new_tokens=512, temperature=0.7, seed=42,
            batched_template=template,
            template_kwargs={'emotion': 'amazed'},
            stories_per_batch=2,
        )

        # 2 topics → 2 format_prompt calls
        fc = calls['format_calls']
        assert len(fc) == 2

        # Every call has EMPTY user prompt (role fidelity)
        assert all(c['prompt'] == '' for c in fc), (
            f"Expected prompt='' in all format_prompt calls, got: "
            f"{[c['prompt'][:40] for c in fc]}"
        )

        # System prompt carries the rendered paper template with substitutions
        assert all('amazed' in c['system_prompt'] for c in fc)
        assert 'topic A' in fc[0]['system_prompt']
        assert 'topic B' in fc[1]['system_prompt']
        # n_stories substituted
        assert '2 different' in fc[0]['system_prompt']

        # use_chat_template threaded through unchanged
        assert all(c['use_chat_template'] is True for c in fc)


# =============================================================================
# Full-mode pipeline integration (Increment 4c)
# =============================================================================

class TestFullModePipelineIntegration:
    """End-to-end full-mode pipeline branch in _generate_training_responses.

    Exercises F1 (missing config fields), F2 (file not found), and the happy
    path. F4 (CLI flag misuse in lightweight mode) is covered by a separate
    subprocess test further down.
    """

    def _setup_trait_fixture(self, tmp_path, monkeypatch, fields_present=None):
        """Build a fake trait dir with configurable extraction_config.yaml.

        `fields_present` is the set of required-field keys to write. Missing
        ones simulate F1 failures. Defaults to all three (happy path).
        """
        if fields_present is None:
            fields_present = {'batched_story_template_file', 'topics_file', 'stories_per_batch'}

        traits_base = tmp_path / 'traits'
        category_dir = traits_base / 'my_emotions'
        trait_dir = category_dir / 'happy'
        trait_dir.mkdir(parents=True, exist_ok=True)

        if 'batched_story_template_file' in fields_present:
            (category_dir / 'prompts').mkdir(exist_ok=True)
            (category_dir / 'prompts' / 'story.txt').write_text(
                "# comment line, should be stripped\n"
                "# another comment line\n"
                "\n"
                "Write {n_stories} different stories about {topic}. Character is {emotion}."
            )
        if 'topics_file' in fields_present:
            (category_dir / 'topics.json').write_text(
                json.dumps(['topic A', 'topic B', 'topic C'])
            )

        config_lines = []
        if 'batched_story_template_file' in fields_present:
            config_lines.append("batched_story_template_file: prompts/story.txt")
        if 'topics_file' in fields_present:
            config_lines.append("topics_file: topics.json")
        if 'stories_per_batch' in fields_present:
            config_lines.append("stories_per_batch: 2")
        (category_dir / 'extraction_config.yaml').write_text("\n".join(config_lines) + "\n")

        def fake_get_path(key, **kwargs):
            if key == 'datasets.trait':
                return traits_base / kwargs['trait']
            if key == 'datasets.traits':
                return traits_base
            raise KeyError(f'fake_get_path unexpected key: {key}')

        monkeypatch.setattr('utils.traits.get_path', fake_get_path)
        return 'my_emotions/happy', trait_dir

    def _fake_backend_and_patches(self, monkeypatch, canned_response):
        """Produce a fake backend object + monkey-patch generate_batch / format_prompt."""
        class FakeBackend:
            model = None
            tokenizer = None
        def fake_generate_batch(model, tokenizer, prompts, max_new_tokens=None,
                                temperature=0.0, seed=None):
            return [canned_response for _ in prompts]
        def fake_format_prompt(prompt, tokenizer, use_chat_template=False, system_prompt=None):
            return f"<sys={system_prompt!r}|user={prompt!r}>"
        monkeypatch.setattr('utils.extraction.generate_batch', fake_generate_batch)
        monkeypatch.setattr('utils.extraction.format_prompt', fake_format_prompt)
        return FakeBackend()

    def test_happy_path_writes_pos_json_with_parsed_stories(self, tmp_path, monkeypatch):
        from core.kwargs_configs import ExtractionConfig
        from utils.extraction import _generate_training_responses

        trait, trait_dir = self._setup_trait_fixture(tmp_path, monkeypatch)
        backend = self._fake_backend_and_patches(
            monkeypatch,
            "[story 1] alpha content.\n[story 2] beta content."
        )

        responses_path = tmp_path / 'responses'
        responses_path.mkdir()
        config = ExtractionConfig(experiment='x', replication_level='full', seed=42)

        _generate_training_responses(
            scenarios={'positive': [{'prompt': 'ignored_in_full_mode'}]},
            responses_path=responses_path,
            backend=backend,
            config=config,
            max_new_tokens=256,
            use_chat_template=True,
            trait=trait,
        )

        out = responses_path / 'pos.json'
        assert out.exists(), "expected pos.json to be written"
        records = json.loads(out.read_text())
        # 3 topics × 2 stories_per_batch = 6 records
        assert len(records) == 6
        # Emotion derived from trait dir name
        assert all('happy' in r['system_prompt'] for r in records)
        # Topics interpolated
        assert any('topic A' in r['system_prompt'] for r in records)

    def test_comment_stripping_prevents_leakage_into_prompt(self, tmp_path, monkeypatch):
        """Header comment lines in story.txt must not appear in the rendered prompt."""
        from core.kwargs_configs import ExtractionConfig
        from utils.extraction import _generate_training_responses

        trait, _ = self._setup_trait_fixture(tmp_path, monkeypatch)
        backend = self._fake_backend_and_patches(monkeypatch, "[story 1] content.\n[story 2] more.")

        responses_path = tmp_path / 'responses'
        responses_path.mkdir()
        config = ExtractionConfig(experiment='x', replication_level='full')

        _generate_training_responses(
            scenarios={'positive': [{'prompt': 'ignored'}]},
            responses_path=responses_path,
            backend=backend, config=config,
            max_new_tokens=256, use_chat_template=True, trait=trait,
        )

        records = json.loads((responses_path / 'pos.json').read_text())
        for r in records:
            assert 'comment line' not in r['system_prompt']
            assert 'should be stripped' not in r['system_prompt']
            # And real template content IS present
            assert 'Write' in r['system_prompt']

    def test_topics_limit_truncates(self, tmp_path, monkeypatch):
        from core.kwargs_configs import ExtractionConfig
        from utils.extraction import _generate_training_responses

        trait, _ = self._setup_trait_fixture(tmp_path, monkeypatch)
        backend = self._fake_backend_and_patches(monkeypatch, "[story 1] a.\n[story 2] b.")

        responses_path = tmp_path / 'responses'
        responses_path.mkdir()
        config = ExtractionConfig(experiment='x', replication_level='full', topics_limit=1)

        _generate_training_responses(
            scenarios={'positive': [{'prompt': 'x'}]},
            responses_path=responses_path,
            backend=backend, config=config,
            max_new_tokens=256, use_chat_template=True, trait=trait,
        )

        records = json.loads((responses_path / 'pos.json').read_text())
        # Only 1 topic × 2 stories = 2 records
        assert len(records) == 2
        topics_seen = {r['topic'] for r in records}
        assert topics_seen == {'topic A'}

    def test_stories_per_batch_override(self, tmp_path, monkeypatch):
        from core.kwargs_configs import ExtractionConfig
        from utils.extraction import _generate_training_responses

        trait, _ = self._setup_trait_fixture(tmp_path, monkeypatch)
        # Canned response has 5 stories; YAML default is 2, override to 5
        canned = "\n".join(f"[story {i+1}] s{i+1}" for i in range(5))
        backend = self._fake_backend_and_patches(monkeypatch, canned)

        responses_path = tmp_path / 'responses'
        responses_path.mkdir()
        config = ExtractionConfig(
            experiment='x', replication_level='full',
            stories_per_batch_override=5,
        )

        _generate_training_responses(
            scenarios={'positive': [{'prompt': 'x'}]},
            responses_path=responses_path,
            backend=backend, config=config,
            max_new_tokens=256, use_chat_template=True, trait=trait,
        )

        records = json.loads((responses_path / 'pos.json').read_text())
        # 3 topics × 5 stories (override) = 15
        assert len(records) == 15

    # ---- F1: missing required config fields ----
    @pytest.mark.parametrize('missing_field', [
        'batched_story_template_file',
        'topics_file',
        'stories_per_batch',
    ])
    def test_f1_missing_required_field_raises(self, tmp_path, monkeypatch, missing_field):
        from core.kwargs_configs import ExtractionConfig
        from utils.extraction import _generate_training_responses

        present = {'batched_story_template_file', 'topics_file', 'stories_per_batch'} - {missing_field}
        trait, _ = self._setup_trait_fixture(tmp_path, monkeypatch, fields_present=present)
        backend = self._fake_backend_and_patches(monkeypatch, "[story 1] x")

        responses_path = tmp_path / 'responses'
        responses_path.mkdir()
        config = ExtractionConfig(experiment='x', replication_level='full')

        with pytest.raises(ValueError, match=f"'{missing_field}'"):
            _generate_training_responses(
                scenarios={'positive': [{'prompt': 'x'}]},
                responses_path=responses_path,
                backend=backend, config=config,
                max_new_tokens=256, use_chat_template=True, trait=trait,
            )

    # ---- F2: referenced file missing ----
    def test_f2_template_file_missing_raises_file_not_found(self, tmp_path, monkeypatch):
        from core.kwargs_configs import ExtractionConfig
        from utils.extraction import _generate_training_responses

        trait, _ = self._setup_trait_fixture(tmp_path, monkeypatch)
        # Delete the template file AFTER fixture created it
        (tmp_path / 'traits' / 'my_emotions' / 'prompts' / 'story.txt').unlink()
        backend = self._fake_backend_and_patches(monkeypatch, "[story 1] x")

        responses_path = tmp_path / 'responses'
        responses_path.mkdir()
        config = ExtractionConfig(experiment='x', replication_level='full')

        with pytest.raises(FileNotFoundError):
            _generate_training_responses(
                scenarios={'positive': [{'prompt': 'x'}]},
                responses_path=responses_path,
                backend=backend, config=config,
                max_new_tokens=256, use_chat_template=True, trait=trait,
            )

    def test_lightweight_mode_still_uses_serial_path(self, tmp_path, monkeypatch):
        """Sanity: lightweight default still calls _generate_and_write, not the
        full-mode branch. No extraction_config.yaml changes required."""
        from core.kwargs_configs import ExtractionConfig

        # Monkeypatch _generate_and_write to detect it's called
        called = []
        def fake_gw(*args, **kwargs):
            called.append(True)
            return 0
        monkeypatch.setattr('utils.extraction._generate_and_write', fake_gw)

        from utils.extraction import _generate_training_responses

        class FakeBackend:
            model = None
            tokenizer = None

        config = ExtractionConfig(experiment='x', replication_level='lightweight')
        _generate_training_responses(
            scenarios={'positive': [{'prompt': 'x'}]},
            responses_path=tmp_path,
            backend=FakeBackend(), config=config,
            max_new_tokens=256, use_chat_template=False,
            trait='some/trait',
        )
        assert called == [True]

    def test_f2_topics_file_missing_raises_file_not_found(self, tmp_path, monkeypatch):
        """F2 sibling: topics file missing is a distinct failure mode from
        template file missing (different call site in _generate_training_responses_full)."""
        from core.kwargs_configs import ExtractionConfig
        from utils.extraction import _generate_training_responses

        trait, _ = self._setup_trait_fixture(tmp_path, monkeypatch)
        # Delete the topics file AFTER fixture created it
        (tmp_path / 'traits' / 'my_emotions' / 'topics.json').unlink()
        backend = self._fake_backend_and_patches(monkeypatch, "[story 1] x")

        responses_path = tmp_path / 'responses'
        responses_path.mkdir()
        config = ExtractionConfig(experiment='x', replication_level='full')

        with pytest.raises(FileNotFoundError):
            _generate_training_responses(
                scenarios={'positive': [{'prompt': 'x'}]},
                responses_path=responses_path,
                backend=backend, config=config,
                max_new_tokens=256, use_chat_template=True, trait=trait,
            )

    def test_trait_none_raises_informative_error(self, tmp_path, monkeypatch):
        """Full-mode branch needs trait= kwarg; None triggers an informative error."""
        from core.kwargs_configs import ExtractionConfig
        from utils.extraction import _generate_training_responses

        backend = self._fake_backend_and_patches(monkeypatch, "[story 1] x")
        config = ExtractionConfig(experiment='x', replication_level='full')

        with pytest.raises(ValueError, match='trait'):
            _generate_training_responses(
                scenarios={'positive': [{'prompt': 'x'}]},
                responses_path=tmp_path,
                backend=backend, config=config,
                max_new_tokens=256, use_chat_template=True,
                trait=None,
            )

    def test_skips_non_positive_label_in_full_mode(self, tmp_path, monkeypatch):
        """Full mode is single-polarity by design. 'negative' label is skipped
        with an informational message — pos.json is the only output."""
        from core.kwargs_configs import ExtractionConfig
        from utils.extraction import _generate_training_responses

        trait, _ = self._setup_trait_fixture(tmp_path, monkeypatch)
        backend = self._fake_backend_and_patches(
            monkeypatch, "[story 1] a.\n[story 2] b."
        )

        responses_path = tmp_path / 'responses'
        responses_path.mkdir()
        config = ExtractionConfig(experiment='x', replication_level='full')

        _generate_training_responses(
            scenarios={'positive': [{'prompt': 'x'}], 'negative': [{'prompt': 'y'}]},
            responses_path=responses_path,
            backend=backend, config=config,
            max_new_tokens=256, use_chat_template=True, trait=trait,
        )

        assert (responses_path / 'pos.json').exists()
        # Full mode must NOT write neg.json even when scenarios has a 'negative' key
        assert not (responses_path / 'neg.json').exists()

    def test_multi_word_trait_emotion_uses_hyphen(self, tmp_path, monkeypatch):
        """Critic-1 fix: trait 'grief_stricken' must render {emotion} as
        'grief-stricken' (paper convention), not 'grief_stricken'. Uses
        resolve_emotion_surface, not bare trait.split('/')[-1]."""
        from core.kwargs_configs import ExtractionConfig
        from utils.extraction import _generate_training_responses

        traits_base = tmp_path / 'traits'
        category_dir = traits_base / 'my_emotions'
        trait_dir = category_dir / 'grief_stricken'
        trait_dir.mkdir(parents=True, exist_ok=True)
        (category_dir / 'prompts').mkdir()
        (category_dir / 'prompts' / 'story.txt').write_text(
            "Write {n_stories} stories about {topic}. Feeling {emotion}."
        )
        (category_dir / 'topics.json').write_text(json.dumps(['a']))
        (category_dir / 'extraction_config.yaml').write_text(
            "batched_story_template_file: prompts/story.txt\n"
            "topics_file: topics.json\n"
            "stories_per_batch: 1\n"
        )

        def fake_get_path(key, **kwargs):
            if key == 'datasets.trait':
                return traits_base / kwargs['trait']
            if key == 'datasets.traits':
                return traits_base
            raise KeyError(key)
        monkeypatch.setattr('utils.traits.get_path', fake_get_path)

        backend = self._fake_backend_and_patches(monkeypatch, "[story 1] content.")

        responses_path = tmp_path / 'responses'
        responses_path.mkdir()
        config = ExtractionConfig(experiment='x', replication_level='full')

        _generate_training_responses(
            scenarios={'positive': [{'prompt': 'x'}]},
            responses_path=responses_path,
            backend=backend, config=config,
            max_new_tokens=256, use_chat_template=True,
            trait='my_emotions/grief_stricken',
        )

        rec = json.loads((responses_path / 'pos.json').read_text())[0]
        # Paper convention: hyphens, not underscores
        assert 'grief-stricken' in rec['system_prompt']
        assert 'grief_stricken' not in rec['system_prompt']

    def test_stories_per_batch_override_zero_fails_fast(self, tmp_path, monkeypatch):
        """Critic-1 fix: `is None` check (not `or`) so override=0 reaches the
        helper; helper rejects stories_per_batch <= 0 with ValueError. This is
        explicit 'no-fallbacks' behavior — the previous `or` pattern silently
        fell through to the YAML default, masking the nonsense input."""
        from core.kwargs_configs import ExtractionConfig
        from utils.extraction import _generate_training_responses

        trait, _ = self._setup_trait_fixture(tmp_path, monkeypatch)
        backend = self._fake_backend_and_patches(monkeypatch, "[story 1] x")

        responses_path = tmp_path / 'responses'
        responses_path.mkdir()
        config = ExtractionConfig(
            experiment='x', replication_level='full',
            stories_per_batch_override=0,
        )

        with pytest.raises(ValueError, match='stories_per_batch must be >= 1'):
            _generate_training_responses(
                scenarios={'positive': [{'prompt': 'x'}]},
                responses_path=responses_path,
                backend=backend, config=config,
                max_new_tokens=256, use_chat_template=True, trait=trait,
            )


# =============================================================================
# F4 CLI validation — unit tests (not subprocess)
# =============================================================================

class TestValidateFullModeFlags:
    """Pure-function F4 validator — all unit-testable without subprocess."""

    def _run(self, replication_level, topics_limit=None, stories_per_batch_override=None):
        from utils.extraction import validate_full_mode_flags
        validate_full_mode_flags(replication_level, topics_limit, stories_per_batch_override)

    def test_lightweight_with_no_full_flags_ok(self):
        self._run("lightweight")  # no-op, returns None

    def test_full_with_no_flags_ok(self):
        self._run("full")

    def test_full_with_topics_ok(self):
        self._run("full", topics_limit=10)

    def test_full_with_stories_per_batch_ok(self):
        self._run("full", stories_per_batch_override=5)

    def test_full_with_both_ok(self):
        self._run("full", topics_limit=10, stories_per_batch_override=5)

    def test_lightweight_with_topics_raises(self):
        with pytest.raises(ValueError, match="--topics is a full-mode-only flag"):
            self._run("lightweight", topics_limit=10)

    def test_lightweight_with_stories_per_batch_raises(self):
        with pytest.raises(ValueError, match="--stories-per-batch is a full-mode-only flag"):
            self._run("lightweight", stories_per_batch_override=5)


# =============================================================================
# _strip_leading_shell_comments — direct unit tests
# =============================================================================

class TestStripLeadingShellComments:
    def test_no_comments_passes_through(self):
        from utils.extraction import _strip_leading_shell_comments
        assert _strip_leading_shell_comments("Hello world.") == "Hello world."

    def test_strips_comment_block_with_blank_separator(self):
        from utils.extraction import _strip_leading_shell_comments
        text = "# comment 1\n# comment 2\n\nReal content here."
        assert _strip_leading_shell_comments(text) == "Real content here."

    def test_strips_comment_block_without_blank_separator(self):
        """If no blank line separates comments from content, content still returned."""
        from utils.extraction import _strip_leading_shell_comments
        text = "# comment\nReal content here."
        assert _strip_leading_shell_comments(text) == "Real content here."

    def test_only_comments_returns_empty(self):
        from utils.extraction import _strip_leading_shell_comments
        assert _strip_leading_shell_comments("# a\n# b\n") == ""

    def test_mid_file_hash_preserved(self):
        """A `#` that appears after real content is NOT a leading comment."""
        from utils.extraction import _strip_leading_shell_comments
        text = "Line one.\n# not a leading comment\nLine three."
        assert _strip_leading_shell_comments(text) == text

    def test_trailing_newline_preserved(self):
        from utils.extraction import _strip_leading_shell_comments
        text = "# c\n\nReal content.\n"
        result = _strip_leading_shell_comments(text)
        assert result == "Real content.\n"

    def test_no_trailing_newline_preserved(self):
        from utils.extraction import _strip_leading_shell_comments
        text = "# c\n\nReal content."
        result = _strip_leading_shell_comments(text)
        assert result == "Real content."

    def test_empty_input(self):
        from utils.extraction import _strip_leading_shell_comments
        assert _strip_leading_shell_comments("") == ""

    def test_leading_whitespace_before_hash(self):
        """`  # indented comment` still counts as a leading comment line."""
        from utils.extraction import _strip_leading_shell_comments
        text = "  # indented\n\nReal content."
        assert _strip_leading_shell_comments(text) == "Real content."
