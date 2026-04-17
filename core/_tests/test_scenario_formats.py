"""
Tests for utils/traits.py scenario-loading across .json / .jsonl / .txt formats.

Covers the cartesian .json format (prompts x system_prompts), format precedence
(.json > .jsonl > .txt), schema validation errors, and get_scenario_count across
all three formats.

Run: pytest core/_tests/test_scenario_formats.py -v
"""

import json
import pytest

from utils.traits import (
    _load_polarity,
    _load_polarity_json,
    content_hash_of_rendered_scenarios,
    load_extraction_config,
    load_scenarios,
    get_scenario_path,
    get_scenario_format,
    get_scenario_count,
    resolve_emotion_surface,
)


# =============================================================================
# Helpers
# =============================================================================

def _write_json(trait_dir, polarity, payload):
    """Write raw payload (may be non-dict) as JSON file."""
    trait_dir.mkdir(parents=True, exist_ok=True)
    (trait_dir / f'{polarity}.json').write_text(json.dumps(payload))


def _write_json_text(trait_dir, polarity, text):
    """Write raw text (e.g. malformed JSON) to the polarity.json path."""
    trait_dir.mkdir(parents=True, exist_ok=True)
    (trait_dir / f'{polarity}.json').write_text(text)


def _write_jsonl(trait_dir, polarity, lines):
    """Write a list of dicts as JSONL."""
    trait_dir.mkdir(parents=True, exist_ok=True)
    (trait_dir / f'{polarity}.jsonl').write_text(
        '\n'.join(json.dumps(line) for line in lines) + '\n'
    )


def _write_txt(trait_dir, polarity, lines):
    """Write raw lines as a .txt prompt file."""
    trait_dir.mkdir(parents=True, exist_ok=True)
    (trait_dir / f'{polarity}.txt').write_text('\n'.join(lines) + '\n')


@pytest.fixture
def patched_trait_dir(tmp_path, monkeypatch):
    """
    Redirect utils.traits.get_path('datasets.trait', trait=...) to tmp_path/traits/{trait}/
    and other dataset keys to their corresponding tmp locations.

    Returns a (trait_name, trait_dir) pair the tests can write to.
    """
    trait_name = 'test_cat/test_trait'
    trait_dir = tmp_path / 'traits' / trait_name
    traits_base = tmp_path / 'traits'

    def fake_get_path(key, **kwargs):
        if key == 'datasets.trait':
            return traits_base / kwargs['trait']
        if key == 'datasets.traits':
            return traits_base
        if key == 'datasets.trait_definition':
            return traits_base / kwargs['trait'] / 'definition.txt'
        raise KeyError(f'fake_get_path unexpected key: {key}')

    monkeypatch.setattr('utils.traits.get_path', fake_get_path)
    return trait_name, trait_dir


# =============================================================================
# Cartesian expansion (happy path)
# =============================================================================

class TestCartesianExpansion:
    """_load_polarity_json expands prompts x system_prompts in outer/inner order."""

    def test_3x2_expands_to_6(self, tmp_path):
        payload = {
            'prompts': ['p1', 'p2', 'p3'],
            'system_prompts': ['s1', 's2'],
        }
        _write_json(tmp_path, 'positive', payload)
        result = _load_polarity(tmp_path, 'positive')
        assert len(result) == 6

    def test_3x2_grouping_outer_is_system_prompt(self, tmp_path):
        """Entries 0-2 share system_prompts[0]; 3-5 share system_prompts[1]."""
        payload = {
            'prompts': ['p1', 'p2', 'p3'],
            'system_prompts': ['s1', 's2'],
        }
        _write_json(tmp_path, 'positive', payload)
        result = _load_polarity(tmp_path, 'positive')
        assert [e['system_prompt'] for e in result[:3]] == ['s1', 's1', 's1']
        assert [e['system_prompt'] for e in result[3:]] == ['s2', 's2', 's2']

    def test_3x2_inner_prompt_order_preserved(self, tmp_path):
        payload = {
            'prompts': ['p1', 'p2', 'p3'],
            'system_prompts': ['s1', 's2'],
        }
        _write_json(tmp_path, 'positive', payload)
        result = _load_polarity(tmp_path, 'positive')
        assert [e['prompt'] for e in result[:3]] == ['p1', 'p2', 'p3']
        assert [e['prompt'] for e in result[3:]] == ['p1', 'p2', 'p3']

    def test_each_entry_has_prompt_and_system_prompt_keys(self, tmp_path):
        payload = {'prompts': ['p'], 'system_prompts': ['s']}
        _write_json(tmp_path, 'positive', payload)
        result = _load_polarity(tmp_path, 'positive')
        entry = result[0]
        assert set(entry.keys()) == {'prompt', 'system_prompt'}
        assert entry['prompt'] == 'p'
        assert entry['system_prompt'] == 's'

    @pytest.mark.parametrize('n_prompts, n_systems', [(1, 1), (5, 1), (1, 5), (4, 3)])
    def test_shape_matches_product(self, tmp_path, n_prompts, n_systems):
        payload = {
            'prompts': [f'p{i}' for i in range(n_prompts)],
            'system_prompts': [f's{j}' for j in range(n_systems)],
        }
        _write_json(tmp_path, 'positive', payload)
        result = _load_polarity(tmp_path, 'positive')
        assert len(result) == n_prompts * n_systems


# =============================================================================
# Schema failures (each must raise ValueError)
# =============================================================================

class TestJsonSchemaFailures:
    """Malformed .json payloads must raise ValueError with a helpful message."""

    def test_missing_prompts_key(self, tmp_path):
        _write_json(tmp_path, 'positive', {'system_prompts': ['s']})
        with pytest.raises(ValueError, match='prompts'):
            _load_polarity(tmp_path, 'positive')

    def test_missing_system_prompts_key(self, tmp_path):
        _write_json(tmp_path, 'positive', {'prompts': ['p']})
        with pytest.raises(ValueError, match='system_prompts'):
            _load_polarity(tmp_path, 'positive')

    def test_empty_prompts_list(self, tmp_path):
        _write_json(tmp_path, 'positive', {'prompts': [], 'system_prompts': ['s']})
        with pytest.raises(ValueError, match='non-empty'):
            _load_polarity(tmp_path, 'positive')

    def test_empty_system_prompts_list(self, tmp_path):
        _write_json(tmp_path, 'positive', {'prompts': ['p'], 'system_prompts': []})
        with pytest.raises(ValueError, match='non-empty'):
            _load_polarity(tmp_path, 'positive')

    @pytest.mark.parametrize('bad_prompts', ['a string', {'a': 'b'}, 42])
    def test_prompts_not_a_list(self, tmp_path, bad_prompts):
        _write_json(tmp_path, 'positive', {'prompts': bad_prompts, 'system_prompts': ['s']})
        with pytest.raises(ValueError, match='must both be lists'):
            _load_polarity(tmp_path, 'positive')

    @pytest.mark.parametrize('bad_systems', ['a string', {'a': 'b'}, 42])
    def test_system_prompts_not_a_list(self, tmp_path, bad_systems):
        _write_json(tmp_path, 'positive', {'prompts': ['p'], 'system_prompts': bad_systems})
        with pytest.raises(ValueError, match='must both be lists'):
            _load_polarity(tmp_path, 'positive')

    def test_top_level_is_array(self, tmp_path):
        _write_json(tmp_path, 'positive', ['p1', 'p2'])
        with pytest.raises(ValueError, match='expected an object'):
            _load_polarity(tmp_path, 'positive')

    def test_top_level_is_string(self, tmp_path):
        _write_json(tmp_path, 'positive', 'just a string')
        with pytest.raises(ValueError, match='expected an object'):
            _load_polarity(tmp_path, 'positive')

    def test_mixed_types_in_prompts(self, tmp_path):
        _write_json(tmp_path, 'positive', {'prompts': [1, 'str'], 'system_prompts': ['s']})
        with pytest.raises(ValueError, match="'prompts' entries must all be strings"):
            _load_polarity(tmp_path, 'positive')

    def test_mixed_types_in_system_prompts(self, tmp_path):
        _write_json(tmp_path, 'positive', {'prompts': ['p'], 'system_prompts': ['s', 42]})
        with pytest.raises(ValueError, match="'system_prompts' entries must all be strings"):
            _load_polarity(tmp_path, 'positive')

    def test_invalid_json_malformed(self, tmp_path):
        _write_json_text(tmp_path, 'positive', '{"prompts": [')
        with pytest.raises(ValueError, match='invalid JSON'):
            _load_polarity(tmp_path, 'positive')


# =============================================================================
# Format collisions — fail fast (project convention: no silent precedence)
# =============================================================================

class TestFormatCollision:
    """Multiple formats for the same polarity must raise ValueError."""

    def test_json_and_jsonl_coexist_raises(self, tmp_path):
        _write_json(tmp_path, 'positive', {
            'prompts': ['a', 'b', 'c'],
            'system_prompts': ['s1', 's2'],
        })
        _write_jsonl(tmp_path, 'positive', [{'prompt': 'jsonl-only'}])
        with pytest.raises(ValueError, match='Multiple scenario formats'):
            _load_polarity(tmp_path, 'positive')

    def test_json_and_txt_coexist_raises(self, tmp_path):
        _write_json(tmp_path, 'positive', {
            'prompts': ['a', 'b'],
            'system_prompts': ['s1', 's2'],
        })
        _write_txt(tmp_path, 'positive', ['txt-one', 'txt-two'])
        with pytest.raises(ValueError, match='Multiple scenario formats'):
            _load_polarity(tmp_path, 'positive')

    def test_jsonl_and_txt_coexist_raises(self, tmp_path):
        _write_jsonl(tmp_path, 'positive', [{'prompt': 'a'}])
        _write_txt(tmp_path, 'positive', ['x', 'y'])
        with pytest.raises(ValueError, match='Multiple scenario formats'):
            _load_polarity(tmp_path, 'positive')

    def test_all_three_formats_coexist_raises(self, tmp_path):
        _write_json(tmp_path, 'positive', {
            'prompts': ['a'],
            'system_prompts': ['s'],
        })
        _write_jsonl(tmp_path, 'positive', [{'prompt': 'b'}])
        _write_txt(tmp_path, 'positive', ['c'])
        with pytest.raises(ValueError, match='Multiple scenario formats'):
            _load_polarity(tmp_path, 'positive')

    def test_txt_only_loads_as_prompt_list(self, tmp_path):
        _write_txt(tmp_path, 'positive', ['line1', 'line2', 'line3'])
        result = _load_polarity(tmp_path, 'positive')
        assert result == [
            {'prompt': 'line1'},
            {'prompt': 'line2'},
            {'prompt': 'line3'},
        ]

    def test_none_exist_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_polarity(tmp_path, 'positive')


# =============================================================================
# Existing formats behave as advertised
# =============================================================================

class TestExistingFormatsUnchanged:
    """.jsonl and .txt parsing match the docstring promise."""

    def test_jsonl_with_prompt_and_system_prompt(self, tmp_path):
        _write_jsonl(tmp_path, 'positive', [
            {'prompt': 'x', 'system_prompt': 'y'},
        ])
        result = _load_polarity(tmp_path, 'positive')
        assert result == [{'prompt': 'x', 'system_prompt': 'y'}]

    def test_jsonl_with_prompt_only_omits_system_prompt_key(self, tmp_path):
        """Current behavior: dict returned as-is (no system_prompt key injected)."""
        _write_jsonl(tmp_path, 'positive', [{'prompt': 'only-prompt'}])
        result = _load_polarity(tmp_path, 'positive')
        assert result == [{'prompt': 'only-prompt'}]
        assert 'system_prompt' not in result[0]

    def test_jsonl_missing_prompt_field_raises(self, tmp_path):
        _write_jsonl(tmp_path, 'positive', [{'system_prompt': 's'}])
        with pytest.raises(ValueError, match="missing 'prompt' field"):
            _load_polarity(tmp_path, 'positive')

    def test_jsonl_skips_blank_lines(self, tmp_path):
        trait_dir = tmp_path
        (trait_dir / 'positive.jsonl').write_text(
            json.dumps({'prompt': 'a'}) + '\n\n' + json.dumps({'prompt': 'b'}) + '\n'
        )
        result = _load_polarity(tmp_path, 'positive')
        assert len(result) == 2

    def test_txt_strips_whitespace_and_skips_empty(self, tmp_path):
        trait_dir = tmp_path
        (trait_dir / 'positive.txt').write_text('line1\n\n  line2  \n\nline3\n')
        result = _load_polarity(tmp_path, 'positive')
        assert result == [
            {'prompt': 'line1'},
            {'prompt': 'line2'},
            {'prompt': 'line3'},
        ]


# =============================================================================
# get_scenario_count across formats
# =============================================================================

class TestGetScenarioCount:
    """get_scenario_count returns N*M for .json, line count for .jsonl/.txt."""

    def test_json_returns_cartesian_product(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir
        _write_json(trait_dir, 'positive', {
            'prompts': ['a', 'b', 'c'],
            'system_prompts': ['s1', 's2'],
        })
        _write_json(trait_dir, 'negative', {
            'prompts': ['a', 'b', 'c'],
            'system_prompts': ['s1', 's2'],
        })
        counts = get_scenario_count(trait)
        assert counts['positive'] == 6
        assert counts['negative'] == 6

    def test_jsonl_returns_line_count(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir
        _write_jsonl(trait_dir, 'positive', [
            {'prompt': 'a'}, {'prompt': 'b'}, {'prompt': 'c'}, {'prompt': 'd'},
        ])
        _write_jsonl(trait_dir, 'negative', [
            {'prompt': 'a'}, {'prompt': 'b'}, {'prompt': 'c'}, {'prompt': 'd'},
        ])
        counts = get_scenario_count(trait)
        assert counts['positive'] == 4
        assert counts['negative'] == 4

    def test_txt_returns_line_count(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir
        _write_txt(trait_dir, 'positive', ['a', 'b', 'c'])
        _write_txt(trait_dir, 'negative', ['x', 'y', 'z'])
        counts = get_scenario_count(trait)
        assert counts['positive'] == 3
        assert counts['negative'] == 3

    def test_missing_returns_zero(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir
        trait_dir.mkdir(parents=True, exist_ok=True)
        counts = get_scenario_count(trait)
        assert counts == {'positive': 0, 'negative': 0}

    def test_mixed_formats_per_polarity(self, patched_trait_dir):
        """Polarities can use different formats independently."""
        trait, trait_dir = patched_trait_dir
        _write_json(trait_dir, 'positive', {
            'prompts': ['a', 'b'],
            'system_prompts': ['s1', 's2', 's3'],
        })
        _write_txt(trait_dir, 'negative', ['one', 'two', 'three', 'four', 'five', 'six'])
        counts = get_scenario_count(trait)
        assert counts['positive'] == 6
        assert counts['negative'] == 6


# =============================================================================
# get_scenario_path / get_scenario_format respect precedence
# =============================================================================

class TestScenarioPathAndFormat:
    """Path + format lookups agree with _load_polarity precedence."""

    def test_path_returns_json_when_present(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir
        _write_json(trait_dir, 'positive', {'prompts': ['p'], 'system_prompts': ['s']})
        _write_jsonl(trait_dir, 'positive', [{'prompt': 'p'}])
        _write_txt(trait_dir, 'positive', ['p'])
        assert get_scenario_path(trait, 'positive').suffix == '.json'

    def test_path_returns_jsonl_when_json_absent(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir
        _write_jsonl(trait_dir, 'positive', [{'prompt': 'p'}])
        _write_txt(trait_dir, 'positive', ['p'])
        assert get_scenario_path(trait, 'positive').suffix == '.jsonl'

    def test_path_returns_txt_when_others_absent(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir
        _write_txt(trait_dir, 'positive', ['p'])
        assert get_scenario_path(trait, 'positive').suffix == '.txt'

    def test_format_json(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir
        _write_json(trait_dir, 'positive', {'prompts': ['p'], 'system_prompts': ['s']})
        assert get_scenario_format(trait) == 'json'

    def test_format_jsonl(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir
        _write_jsonl(trait_dir, 'positive', [{'prompt': 'p'}])
        assert get_scenario_format(trait) == 'jsonl'

    def test_format_txt(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir
        _write_txt(trait_dir, 'positive', ['p'])
        assert get_scenario_format(trait) == 'txt'


# =============================================================================
# load_scenarios public API
# =============================================================================

class TestLoadScenariosPublicAPI:
    """High-level load_scenarios end-to-end for the .json cartesian path."""

    def test_json_both_polarities(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir
        _write_json(trait_dir, 'positive', {
            'prompts': ['p1', 'p2'],
            'system_prompts': ['s1', 's2'],
        })
        _write_json(trait_dir, 'negative', {
            'prompts': ['n1', 'n2'],
            'system_prompts': ['s1', 's2'],
        })
        result = load_scenarios(trait)
        assert len(result['positive']) == 4
        assert len(result['negative']) == 4
        assert result['positive'][0] == {'prompt': 'p1', 'system_prompt': 's1'}
        assert result['negative'][-1] == {'prompt': 'n2', 'system_prompt': 's2'}

    def test_mismatched_counts_raise(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir
        _write_json(trait_dir, 'positive', {
            'prompts': ['a', 'b', 'c'],
            'system_prompts': ['s1'],
        })
        _write_json(trait_dir, 'negative', {
            'prompts': ['x'],
            'system_prompts': ['s1'],
        })
        with pytest.raises(ValueError, match='Scenario count mismatch'):
            load_scenarios(trait)

    def test_polarity_arg_loads_only_requested(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir
        _write_json(trait_dir, 'positive', {
            'prompts': ['p'],
            'system_prompts': ['s'],
        })
        # No negative file on disk; requesting positive should still succeed.
        result = load_scenarios(trait, polarity='positive')
        assert set(result.keys()) == {'positive'}
        assert len(result['positive']) == 1


# =============================================================================
# prompt_template substitution + render-then-hash canonicalization
# =============================================================================

# Representative paper-style wrapper: the "amazed" emotion-concepts wrapper,
# condensed to a few lines for test readability while preserving structure.
AMAZED_WRAPPER_TEMPLATE = (
    "Write a story based on the following premise.\n\n"
    "Topic: {topic}\n\n"
    "The story should follow a character who is feeling {emotion}.\n\n"
    "IMPORTANT: Never name the emotion directly."
)

AMAZED_TOPICS = [
    "A person learns their childhood bully became a therapist",
    "A homeowner discovers previous residents left items in the attic",
    "Someone receives an apology letter years after the incident",
]


def _prebake(topic: str, emotion: str) -> str:
    """Produce the same rendered string that the template would produce for this (topic, emotion)."""
    return AMAZED_WRAPPER_TEMPLATE.format(topic=topic, emotion=emotion)


class TestResolveEmotionSurface:
    def test_default_converts_underscores_to_hyphens(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir  # trait == 'test_cat/test_trait'
        assert resolve_emotion_surface(trait) == 'test-trait'

    def test_single_word_trait_stays_as_is(self, patched_trait_dir, tmp_path):
        # Bare single-word trait name works too
        assert resolve_emotion_surface('amazed') == 'amazed'
        assert resolve_emotion_surface('some_cat/grief_stricken') == 'grief-stricken'

    def test_trait_yaml_override_wins(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir
        trait_dir.mkdir(parents=True, exist_ok=True)
        (trait_dir / 'trait.yaml').write_text("emotion_surface: amazed\n")
        assert resolve_emotion_surface(trait) == 'amazed'

    def test_blank_override_falls_back_to_default(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir
        trait_dir.mkdir(parents=True, exist_ok=True)
        (trait_dir / 'trait.yaml').write_text("emotion_surface: '   '\n")
        # Blank override is ignored; derivation from trait name used
        assert resolve_emotion_surface(trait) == 'test-trait'


class TestPromptTemplateSubstitution:
    """Template substitution renders {topic} and {emotion} into each scenario's prompt."""

    def _write_category_template(self, trait_dir, template: str):
        """Write extraction_config.yaml at the CATEGORY level (one dir up from trait)."""
        category_dir = trait_dir.parent
        category_dir.mkdir(parents=True, exist_ok=True)
        (category_dir / 'extraction_config.yaml').write_text(
            f"polarity: single\nprompt_template: |\n  "
            + template.replace('\n', '\n  ')
            + "\n"
        )

    def test_template_substitutes_topic_and_emotion(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir
        # Override emotion surface to match paper ("amazed" not "test-trait")
        trait_dir.mkdir(parents=True, exist_ok=True)
        (trait_dir / 'trait.yaml').write_text("emotion_surface: amazed\n")
        # Scenario file stores just topics as the `prompt` field, one per line.
        _write_jsonl(trait_dir, 'positive', [{'prompt': t} for t in AMAZED_TOPICS])
        self._write_category_template(trait_dir, AMAZED_WRAPPER_TEMPLATE)

        result = load_scenarios(trait, polarity='positive')
        assert len(result['positive']) == len(AMAZED_TOPICS)
        for i, entry in enumerate(result['positive']):
            assert entry['prompt'] == _prebake(AMAZED_TOPICS[i], 'amazed')

    def test_no_template_means_no_substitution(self, patched_trait_dir):
        trait, trait_dir = patched_trait_dir
        _write_jsonl(trait_dir, 'positive', [{'prompt': 'literal text'}])
        # No extraction_config.yaml, no template → prompt passes through unchanged.
        result = load_scenarios(trait, polarity='positive')
        assert result['positive'] == [{'prompt': 'literal text'}]

    def test_template_applies_to_json_cartesian_too(self, patched_trait_dir):
        """When .json cartesian is used with a template, the per-pair `prompt`
        (topic) is substituted; system_prompts pass through untouched.
        (Templates operate on each scenario's `prompt` field after the loader
        returns, so they compose cleanly with cartesian expansion.)"""
        trait, trait_dir = patched_trait_dir
        trait_dir.mkdir(parents=True, exist_ok=True)
        (trait_dir / 'trait.yaml').write_text("emotion_surface: amazed\n")
        _write_json(trait_dir, 'positive', {
            'prompts': AMAZED_TOPICS,
            'system_prompts': ['You are a skilled short-story writer.'],
        })
        self._write_category_template(trait_dir, AMAZED_WRAPPER_TEMPLATE)

        result = load_scenarios(trait, polarity='positive')
        assert len(result['positive']) == len(AMAZED_TOPICS)
        for i, entry in enumerate(result['positive']):
            assert entry['prompt'] == _prebake(AMAZED_TOPICS[i], 'amazed')
            assert entry['system_prompt'] == 'You are a skilled short-story writer.'


class TestRenderThenHashInvariance:
    """Hash must be identical between pre-baked .jsonl and template-form content
    that render to the same final prompts. This is the guardrail test — if the
    future ant_emotion_concepts migration to templates would change hashes, this
    fails and the migration isn't safe."""

    def _build_prebaked_trait(self, traits_base, emotion_surface):
        """Trait A: .jsonl with the fully pre-filled prompts (no template)."""
        trait = 'emotions_prebaked/' + emotion_surface
        trait_dir = traits_base / trait
        trait_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(trait_dir, 'positive', [
            {'prompt': _prebake(t, emotion_surface)} for t in AMAZED_TOPICS
        ])
        # Mark as single-polarity (no negative file)
        (trait_dir / 'trait.yaml').write_text("polarity: single\n")
        return trait

    def _build_templated_trait(self, traits_base, emotion_surface):
        """Trait B: .jsonl with JUST the topics + category extraction_config with the template."""
        trait = 'emotions_templated/' + emotion_surface
        category_dir = traits_base / 'emotions_templated'
        trait_dir = traits_base / trait
        category_dir.mkdir(parents=True, exist_ok=True)
        trait_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(trait_dir, 'positive', [{'prompt': t} for t in AMAZED_TOPICS])
        (trait_dir / 'trait.yaml').write_text(
            f"polarity: single\nemotion_surface: {emotion_surface}\n"
        )
        (category_dir / 'extraction_config.yaml').write_text(
            "polarity: single\nprompt_template: |\n  "
            + AMAZED_WRAPPER_TEMPLATE.replace('\n', '\n  ')
            + "\n"
        )
        return trait

    def test_prebaked_and_templated_render_identically(self, tmp_path, monkeypatch):
        """load_scenarios() must produce identical scenarios for both storage forms."""
        traits_base = tmp_path / 'traits'

        def fake_get_path(key, **kwargs):
            if key == 'datasets.trait':
                return traits_base / kwargs['trait']
            if key == 'datasets.traits':
                return traits_base
            if key == 'datasets.trait_definition':
                return traits_base / kwargs['trait'] / 'definition.txt'
            raise KeyError(f'fake_get_path unexpected key: {key}')

        monkeypatch.setattr('utils.traits.get_path', fake_get_path)

        emotion = 'amazed'
        trait_a = self._build_prebaked_trait(traits_base, emotion)
        trait_b = self._build_templated_trait(traits_base, emotion)

        a = load_scenarios(trait_a, polarity='positive')['positive']
        b = load_scenarios(trait_b, polarity='positive')['positive']

        assert a == b, (
            f"Rendered scenarios differ:\n"
            f"  prebaked[0]['prompt'] = {a[0]['prompt']!r}\n"
            f"  templated[0]['prompt'] = {b[0]['prompt']!r}"
        )

    def test_hash_invariance_across_storage_forms(self, tmp_path, monkeypatch):
        """content_hash_of_rendered_scenarios must be byte-identical for both forms.

        This is the test that gates whether a future ant_emotion_concepts
        migration would preserve vector-cache validity (via input_hashes).
        """
        traits_base = tmp_path / 'traits'

        def fake_get_path(key, **kwargs):
            if key == 'datasets.trait':
                return traits_base / kwargs['trait']
            if key == 'datasets.traits':
                return traits_base
            if key == 'datasets.trait_definition':
                return traits_base / kwargs['trait'] / 'definition.txt'
            raise KeyError(f'fake_get_path unexpected key: {key}')

        monkeypatch.setattr('utils.traits.get_path', fake_get_path)

        emotion = 'amazed'
        trait_a = self._build_prebaked_trait(traits_base, emotion)
        trait_b = self._build_templated_trait(traits_base, emotion)

        hash_a = content_hash_of_rendered_scenarios(trait_a, 'positive')
        hash_b = content_hash_of_rendered_scenarios(trait_b, 'positive')

        assert hash_a == hash_b, (
            f"Hash drift between storage forms — future migration would "
            f"trigger spurious re-extraction.\n"
            f"  prebaked .jsonl hash: {hash_a}\n"
            f"  templated hash:       {hash_b}"
        )

    def test_different_emotion_produces_different_hash(self, tmp_path, monkeypatch):
        """Sanity check: the canonicalization is not hashing to a constant."""
        traits_base = tmp_path / 'traits'

        def fake_get_path(key, **kwargs):
            if key == 'datasets.trait':
                return traits_base / kwargs['trait']
            if key == 'datasets.traits':
                return traits_base
            if key == 'datasets.trait_definition':
                return traits_base / kwargs['trait'] / 'definition.txt'
            raise KeyError(f'fake_get_path unexpected key: {key}')

        monkeypatch.setattr('utils.traits.get_path', fake_get_path)

        trait_amazed = self._build_prebaked_trait(traits_base, 'amazed')
        trait_sad = self._build_prebaked_trait(traits_base, 'sad')
        # Need separate dirs per emotion — "amazed" and "sad" get different subdirs
        # via the emotion-suffix pattern in _build_prebaked_trait
        hash_amazed = content_hash_of_rendered_scenarios(trait_amazed, 'positive')
        hash_sad = content_hash_of_rendered_scenarios(trait_sad, 'positive')
        assert hash_amazed != hash_sad


# =============================================================================
# extraction_config.yaml *_file path resolution (Increment 4a)
# =============================================================================

class TestExtractionConfigFileRefs:
    """load_extraction_config resolves *_file keys to absolute Paths eagerly,
    relative to each YAML's own parent directory, BEFORE the cascade merge."""

    def _setup_fake_paths(self, tmp_path, monkeypatch):
        traits_base = tmp_path / 'traits'
        traits_base.mkdir(parents=True, exist_ok=True)

        def fake_get_path(key, **kwargs):
            if key == 'datasets.trait':
                return traits_base / kwargs['trait']
            if key == 'datasets.traits':
                return traits_base
            raise KeyError(f'fake_get_path unexpected key: {key}')

        monkeypatch.setattr('utils.traits.get_path', fake_get_path)
        return traits_base

    def test_category_file_ref_resolves_against_category_dir(self, tmp_path, monkeypatch):
        """Category config's `prompts/story.txt` resolves to the category dir."""
        from pathlib import Path
        traits_base = self._setup_fake_paths(tmp_path, monkeypatch)

        category_dir = traits_base / 'my_cat'
        trait_dir = category_dir / 'my_trait'
        category_dir.mkdir(parents=True, exist_ok=True)
        trait_dir.mkdir(parents=True, exist_ok=True)
        (category_dir / 'extraction_config.yaml').write_text(
            "batched_story_template_file: prompts/story.txt\n"
        )

        merged = load_extraction_config('my_cat/my_trait')
        assert isinstance(merged['batched_story_template_file'], Path)
        assert merged['batched_story_template_file'].is_absolute()
        # Resolved against category dir (where the YAML lives)
        assert merged['batched_story_template_file'] == (category_dir / 'prompts' / 'story.txt').resolve()

    def test_trait_file_ref_resolves_against_trait_dir(self, tmp_path, monkeypatch):
        """Per-trait config's `prompts/bar.txt` resolves to the trait dir, NOT the category."""
        from pathlib import Path
        traits_base = self._setup_fake_paths(tmp_path, monkeypatch)

        category_dir = traits_base / 'my_cat'
        trait_dir = category_dir / 'my_trait'
        trait_dir.mkdir(parents=True, exist_ok=True)
        (trait_dir / 'extraction_config.yaml').write_text(
            "batched_story_template_file: prompts/bar.txt\n"
        )

        merged = load_extraction_config('my_cat/my_trait')
        assert merged['batched_story_template_file'] == (trait_dir / 'prompts' / 'bar.txt').resolve()

    def test_trait_override_beats_category_with_correct_base_dir(self, tmp_path, monkeypatch):
        """Category sets `prompts/foo.txt`; trait sets `prompts/bar.txt`. Trait wins,
        AND its path is anchored to the trait dir (not the category dir)."""
        from pathlib import Path
        traits_base = self._setup_fake_paths(tmp_path, monkeypatch)

        category_dir = traits_base / 'my_cat'
        trait_dir = category_dir / 'my_trait'
        trait_dir.mkdir(parents=True, exist_ok=True)
        (category_dir / 'extraction_config.yaml').write_text(
            "batched_story_template_file: prompts/foo.txt\n"
        )
        (trait_dir / 'extraction_config.yaml').write_text(
            "batched_story_template_file: prompts/bar.txt\n"
        )

        merged = load_extraction_config('my_cat/my_trait')
        # Per-trait wins, and uses trait-dir as base.
        assert merged['batched_story_template_file'] == (trait_dir / 'prompts' / 'bar.txt').resolve()

    def test_absolute_path_preserved_unchanged(self, tmp_path, monkeypatch):
        """Absolute paths in the YAML are not rewritten."""
        from pathlib import Path
        traits_base = self._setup_fake_paths(tmp_path, monkeypatch)

        category_dir = traits_base / 'my_cat'
        trait_dir = category_dir / 'my_trait'
        trait_dir.mkdir(parents=True, exist_ok=True)
        abs_path = tmp_path / 'somewhere' / 'else' / 'story.txt'
        (category_dir / 'extraction_config.yaml').write_text(
            f"batched_story_template_file: {abs_path}\n"
        )

        merged = load_extraction_config('my_cat/my_trait')
        assert merged['batched_story_template_file'] == abs_path

    def test_non_file_suffixed_fields_unchanged(self, tmp_path, monkeypatch):
        """Only `*_file` keys are resolved; ordinary fields pass through."""
        traits_base = self._setup_fake_paths(tmp_path, monkeypatch)

        category_dir = traits_base / 'my_cat'
        trait_dir = category_dir / 'my_trait'
        trait_dir.mkdir(parents=True, exist_ok=True)
        (category_dir / 'extraction_config.yaml').write_text(
            "stories_per_batch: 12\n"
            "prompt_template: |\n"
            "  Write {n_stories} about {topic}\n"
            "temperature: 0.7\n"
        )

        merged = load_extraction_config('my_cat/my_trait')
        assert merged['stories_per_batch'] == 12
        assert merged['temperature'] == 0.7
        assert isinstance(merged['prompt_template'], str)

    def test_file_ref_not_required_to_exist_at_load_time(self, tmp_path, monkeypatch):
        """Loader resolves path but does NOT check existence. Callers open() and
        get FileNotFoundError there — this keeps lightweight runs free of disk-I/O
        overhead for template files they never use."""
        traits_base = self._setup_fake_paths(tmp_path, monkeypatch)

        category_dir = traits_base / 'my_cat'
        trait_dir = category_dir / 'my_trait'
        trait_dir.mkdir(parents=True, exist_ok=True)
        (category_dir / 'extraction_config.yaml').write_text(
            "batched_story_template_file: prompts/does_not_exist.txt\n"
        )

        # Loader succeeds (no existence check)
        merged = load_extraction_config('my_cat/my_trait')
        assert 'batched_story_template_file' in merged
        # File doesn't exist — confirmed by direct check
        assert not merged['batched_story_template_file'].exists()
