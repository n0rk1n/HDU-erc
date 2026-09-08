import json
from pathlib import Path
import pytest
from chatbot.core.errors import ConfigError


GOEMOTIONS_LABELS = (
    'admiration amusement anger annoyance approval caring confusion curiosity desire '
    'disappointment disapproval disgust embarrassment excitement fear gratitude grief '
    'joy love nervousness optimism pride realization relief remorse sadness surprise neutral'
).split()


@pytest.mark.parametrize('root', [Path('config'), Path('data/config')])
def test_taxonomy_uses_official_goemotions_labels_and_ekman_families(root):
    from chatbot.emotion.config import load_taxonomy
    taxonomy = load_taxonomy(root / 'emotion_labels.json', root / 'emotion_families.json')
    assert list(taxonomy.labels) == GOEMOTIONS_LABELS
    assert set(taxonomy.families.values()) == {'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral'}
    assert taxonomy.families['anger'] == taxonomy.families['annoyance'] == taxonomy.families['disapproval'] == 'anger'
    assert taxonomy.families['fear'] == taxonomy.families['nervousness'] == 'fear'
    assert taxonomy.families['sadness'] == taxonomy.families['grief'] == taxonomy.families['disappointment'] == 'sadness'
    assert taxonomy.families['embarrassment'] == taxonomy.families['remorse'] == 'sadness'
    assert taxonomy.families['confusion'] == taxonomy.families['curiosity'] == taxonomy.families['realization'] == 'surprise'
    assert taxonomy.families['neutral'] == 'neutral'


@pytest.mark.parametrize('filename', ['emotion_labels.json', 'emotion_families.json', 'emotion_examples.json'])
def test_builtin_and_environment_template_configs_stay_in_sync(filename):
    assert (Path('config') / filename).read_bytes() == (Path('data/config') / filename).read_bytes()


@pytest.mark.parametrize('labels,families', [
    ('{"sad":"one","sad":"two"}', '{"sad":"loss"}'),
    ('{"sad":""}', '{"sad":"loss"}'),
    ('{"sad":"sad"}', '{}'),
    ('{"sad":"sad"}', '{"sad":"loss","unknown":"loss"}'),
])
def test_invalid_taxonomy_rejected(tmp_path, labels, families):
    from chatbot.emotion.config import load_taxonomy
    a, b = tmp_path/'labels.json', tmp_path/'families.json'
    a.write_text(labels); b.write_text(families)
    with pytest.raises(ConfigError):
        load_taxonomy(a,b)


def test_retrieval_settings_from_environment(monkeypatch):
    from chatbot.core.config import AppConfig
    from chatbot.emotion.config import load_emotion_settings
    monkeypatch.setenv('LLM_API_KEY','offline')
    monkeypatch.setenv('EMOTION_CONTEXT_TOKENS','10000')
    monkeypatch.setenv('EMOTION_TOKENIZER_MODEL','gpt-4o-mini')
    monkeypatch.setenv('EMOTION_EXAMPLE_LIMIT','0')
    monkeypatch.setenv('EMOTION_PRIOR_BOOST','4.5')
    monkeypatch.setenv('EMOTION_RECENT_LABEL_LIMIT','0')
    result=load_emotion_settings(AppConfig.from_env())
    assert result.retrieval.example_limit==0 and result.retrieval.prior_boost==4.5
    assert result.retrieval.recent_label_limit==0
    monkeypatch.setenv('EMOTION_PRIOR_BOOST','NaN')
    from chatbot.core.errors import ConfigError
    import pytest
    with pytest.raises(ConfigError):load_emotion_settings(AppConfig.from_env())


def test_nested_display_name_preserves_model_description_and_hash(tmp_path):
    from chatbot.emotion.config import load_taxonomy
    labels, families = tmp_path / 'labels.json', tmp_path / 'families.json'
    families.write_text('{"sad": "loss"}')
    labels.write_text('{"sad": "sorrow"}')
    legacy = load_taxonomy(labels, families)
    labels.write_text(json.dumps({'sad': {'description': 'sorrow', 'display_name': '低落'}}))
    configured = load_taxonomy(labels, families)
    assert configured.labels == {'sad': 'sorrow'}
    assert configured.content_hash == legacy.content_hash


@pytest.mark.parametrize('entry', [
    {}, {'display_name': '低落'}, {'description': ' '},
    {'description': 'sorrow', 'display_name': 1},
    {'description': 'sorrow', 'display_name': ''},
    {'description': 'sorrow', 'display_nam': 'typo'},
])
def test_invalid_nested_label_rejected(tmp_path, entry):
    from chatbot.emotion.config import load_taxonomy
    labels, families = tmp_path / 'labels.json', tmp_path / 'families.json'
    labels.write_text(json.dumps({'sad': entry}))
    families.write_text('{"sad": "loss"}')
    with pytest.raises(ConfigError):
        load_taxonomy(labels, families)
