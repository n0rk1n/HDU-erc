from chatbot.emotion.config import load_taxonomy, CONFIG_ROOT

def test_examples_cover_goemotions_and_selection_is_auditable():
    from collections import Counter
    from chatbot.emotion.prompt import load_examples
    from chatbot.emotion.retrieval import select_examples
    from tests.emotion.test_config import GOEMOTIONS_LABELS
    taxonomy=load_taxonomy()
    examples=load_examples(CONFIG_ROOT/'emotion_examples.json',taxonomy)
    assert Counter(e['emotion'] for e in examples) == {label: 2 for label in GOEMOTIONS_LABELS}
    assert all(e['id'].startswith('goemotions-train-') for e in examples)
    assert len({' '.join(e['dialogue'].casefold().split()) for e in examples}) == len(examples)
    selected=select_examples(examples,'convert file PDF',[],limit=4)
    assert len(selected)==4
    assert all('score' in e and 'reason' in e for e in selected)


def test_seed_examples_match_their_source_manifest():
    import hashlib
    import json
    from pathlib import Path
    from tests.emotion.test_config import GOEMOTIONS_LABELS
    manifest = json.loads(Path('data/config/emotion_examples.metadata.json').read_text())
    examples = json.loads(Path('data/config/emotion_examples.json').read_text())
    assert manifest['source_split'] == 'train'
    assert manifest['source_commit'] == '5594ac0ee7a13c77eb1e0b98f0f305a2ca65b6c4'
    for filename, expected_hash in manifest['config_sha256'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected_hash
    samples = manifest['samples']
    assert len(samples) == len(examples) == 56
    assert len({sample['comment_id'] for sample in samples}) == len(samples)
    for sample, example in zip(samples, examples):
        assert example['id'] == f"goemotions-train-{sample['comment_id']}"
        assert sample['label_ids'] == [GOEMOTIONS_LABELS.index(example['emotion'])]
        assert sample['train_line'] > 0


def test_retrieval_limit_and_prior_are_configurable():
    from chatbot.emotion.retrieval import select_examples
    examples=[{'dialogue':'apple','emotion':'happy'},{'dialogue':'banana','emotion':'sad'}]
    assert select_examples(examples,'apple',['sad'],limit=0,prior_boost=0)==[]
    assert select_examples(examples,'apple',['sad'],limit=1,prior_boost=0)[0]['emotion']=='happy'
    assert select_examples(examples,'apple',['sad'],limit=1,prior_boost=10)[0]['emotion']=='sad'
