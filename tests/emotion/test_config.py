import json
from pathlib import Path
import pytest
from chatbot.core.errors import ConfigError


def test_taxonomy_loads_and_separates_absence_from_neutral():
    from chatbot.emotion.config import load_taxonomy
    taxonomy = load_taxonomy(Path('config/emotion_labels.json'), Path('config/emotion_families.json'))
    assert len(taxonomy.labels) == 34
    assert taxonomy.families['sad'] == taxonomy.families['lonely'] == 'sadness_loss'
    assert taxonomy.families['no_emotion'] != taxonomy.families['neutral']
    assert taxonomy.labels['no_emotion'] != taxonomy.labels['neutral']


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
