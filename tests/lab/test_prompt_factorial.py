"""Catch confounded inputs and incorrect factorial statistics before paid runs."""
from copy import deepcopy
import importlib
import json

import pytest


def module():
    return importlib.import_module('scripts.prompt_factorial_stats')


def configs():
    base = {'method': 'semantic', 'model': {'name': 'fixed', 'max_tokens': 100},
            'evaluation_set_id': 'same-dev', 'embedding': {'index_id': 'i', 'queries_artifact_id': 'q'},
            'k': 4, 'candidate_count': 50}
    return {a: {**deepcopy(base), 'name': a, 'prompt_version': p, 'example_policy': e}
            for a, p, e in [('A', 'native-labels-v1', 'ranked'), ('B', 'native-labels-v3', 'ranked'),
                            ('C', 'native-labels-v1', 'contrastive-v1'), ('D', 'native-labels-v3', 'contrastive-v1')]}


@pytest.mark.parametrize('field,value', [('evaluation_set_id', 'different-dev'), ('model', {'name': 'changed'}),
                                         ('embedding', {'index_id': 'different'}), ('k', 8)])
def test_rejects_confounding_configuration(field, value):
    cfg = configs(); cfg['D'][field] = value
    with pytest.raises(ValueError, match='common|factorial'):
        module().validate_factorial(cfg)


def requests():
    from emotion_lab.taxonomy import messages_for, load_taxonomy
    out = {}
    for arm, cfg in configs().items():
        examples = [{'raw_text': 'That is nice.', 'labels': ['approval']}]
        if arm in 'CD':
            examples = [{'raw_text': 'That is annoying.', 'labels': ['annoyance']}]
        out[arm] = {'model': 'fixed', 'temperature': 0, 'messages': messages_for(
            'query', examples, load_taxonomy(), version=cfg['prompt_version'])}
    return out


def test_prompt_effect_keeps_examples_fixed_and_verifies_actual_instruction():
    data = requests()
    module().validate_requests(data)
    changed = deepcopy(data)
    changed['B']['messages'][0]['content'] = changed['B']['messages'][0]['content'].replace('That is nice.', 'Changed example.')
    with pytest.raises(ValueError, match='example'):
        module().validate_requests(changed)
    changed = deepcopy(data)
    changed['D']['messages'][0]['content'] = changed['D']['messages'][0]['content'].replace('Classify the emotion', 'Wrong instruction')
    with pytest.raises(ValueError, match='instruction'):
        module().validate_requests(changed)


def test_interaction_uses_all_four_arms_and_shared_resampling():
    # A/B/C always wrong, D always right: interaction is exactly +1,
    # regardless of the common bootstrap sample indices.
    rows = [{'sample_id': str(i), 'truth': ['joy'], **{a: {'predicted_labels': ['joy'] if a == 'D' else ['anger']} for a in 'ABCD'}} for i in range(3)]
    result = module().factorial_bootstrap(rows, seed=42, repeats=20)
    assert result['interaction']['value'] == 1.0
    assert result['interaction']['ci_low'] == 1.0 == result['interaction']['ci_high']
    assert result['bootstrap_draws'][0]['interaction'] == 1.0
    assert len(result['bootstrap_draws']) == 20
    assert all(len(x['sample_ids']) == 3 for x in result['bootstrap_draws'])
    assert module().factorial_bootstrap(rows, seed=42, repeats=20) == result


def test_best_combined_score_does_not_automatically_imply_synergy():
    # A=0, B=.5, C=.5, D=1: combined improvement is additive, interaction=0.
    rows = [{'sample_id': '1', 'truth': ['joy'], **{a: {'predicted_labels': ['joy'] if a in 'BD' else ['anger']} for a in 'ABCD'}},
            {'sample_id': '2', 'truth': ['joy'], **{a: {'predicted_labels': ['joy'] if a in 'CD' else ['anger']} for a in 'ABCD'}}]
    result = module().factorial_bootstrap(rows, repeats=20)
    assert result['interaction']['value'] == 0.0
    assert result['effects']['D-A']['value'] == 1.0
    assert result['effects']['B-A']['value'] == 0.5
    assert result['effects']['D-C']['value'] == 0.5
    assert all(x['interaction'] == 0 for x in result['bootstrap_draws'])


@pytest.mark.parametrize('records', [[], [{'sample_id': 'x', 'truth': ['joy'], 'A': {'predicted_labels': []}}]])
def test_empty_or_unpaired_records_are_rejected(records):
    with pytest.raises(ValueError, match='paired|empty'):
        module().factorial_bootstrap(records, repeats=20)


def test_factorial_storage_creates_its_own_frozen_comparison(configured):
    from emotion_lab.preparation import prepare_run
    from emotion_lab.runner import execute_run
    from emotion_lab.evaluation import evaluate, compare
    from test_runner import Counter, response
    store, cfg = configured
    run = prepare_run(store, cfg, counter=Counter())
    execute_run(store, run, transport=lambda request: response(('joy',)), counter=Counter())
    evaluations = {arm: evaluate(store, run, {}) for arm in 'ABCD'}
    original = compare(store, evaluations['A'], evaluations['D'], repeats=20)
    before = store.one('SELECT * FROM comparisons WHERE comparison_id=?', (original,))
    ids = [r['sample_id'] for r in store.rows('SELECT sample_id FROM run_items WHERE run_id=? ORDER BY ordinal', (run,))]
    rows = []
    for sid in ids:
        truth = [r['label_name'] for r in store.rows('SELECT d.label_name FROM sample_labels l JOIN label_definitions d ON d.dataset_version_id=l.dataset_version_id AND d.label_id=l.label_id WHERE l.sample_id=?', (sid,))]
        rows.append({'sample_id': sid, 'truth': truth, **{a: {'predicted_labels': ['joy']} for a in 'ABCD'}})
    result = module().factorial_bootstrap(rows, repeats=20)
    saved = module().save_factorial(store, evaluations, result, ids)
    assert saved['comparison_id'] != original
    assert store.one('SELECT * FROM comparisons WHERE comparison_id=?', (original,)) == before
    row = store.one('SELECT * FROM comparisons WHERE comparison_id=?', (saved['comparison_id'],))
    assert row['status'] == 'completed'
    assert store.json(row['analysis_config_artifact_id'])['evaluation_ids'] == evaluations
    metric = store.one('SELECT * FROM metric_values WHERE comparison_id=?', (saved['comparison_id'],))
    assert metric['metric_name'] == 'interaction_micro_f1' and metric['value'] == 0.0
    assert not store.rows("SELECT * FROM metric_values WHERE comparison_id=? AND metric_name='interaction_micro_f1'", (original,))
