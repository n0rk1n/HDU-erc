"""Protocol guards and paired four-arm statistics; no model calls."""
import json
import random
from pathlib import Path

from emotion_lab.storage import now
from emotion_lab.taxonomy import prompt_instruction

ARMS = {'A': ('native-labels-v1', 'ranked'), 'B': ('native-labels-v3', 'ranked'),
        'C': ('native-labels-v1', 'contrastive-v1'), 'D': ('native-labels-v3', 'contrastive-v1')}
COMPARISONS = {'B-A': ('A', 'B'), 'D-C': ('C', 'D'), 'C-A': ('A', 'C'),
               'D-B': ('B', 'D'), 'D-A': ('A', 'D')}


def validate_factorial(configs):
    if set(configs) != set(ARMS):
        raise ValueError('factorial requires A/B/C/D')
    common = None
    for arm, (version, policy) in ARMS.items():
        cfg = configs[arm]
        if (cfg.get('prompt_version'), cfg.get('example_policy')) != (version, policy):
            raise ValueError('factorial arm assignment differs')
        if cfg.get('method') != 'semantic' or cfg.get('k') != 4 or cfg.get('candidate_count') != 50:
            raise ValueError('factorial retrieval protocol differs')
        fixed = {k: v for k, v in cfg.items() if k not in {'name', 'prompt_version', 'example_policy'}}
        if common is not None and fixed != common:
            raise ValueError('common configuration differs across arms')
        common = fixed


def validate_requests(requests):
    if set(requests) != set(ARMS):
        raise ValueError('four paired requests required')
    examples, common = {}, None
    for arm, request in requests.items():
        messages = request['messages']
        if len(messages) != 2 or messages[0]['role'] != 'system' or messages[1]['role'] != 'user':
            raise ValueError('request message structure differs')
        system, example_text = messages[0]['content'].split('\nTraining examples:\n')
        instruction, definitions = system.split('\nLabel definitions:\n')
        if instruction != prompt_instruction(ARMS[arm][0], False):
            raise ValueError('actual instruction differs from frozen version')
        examples[arm] = json.loads(example_text)
        fixed = {**request, 'messages': [{'role': 'system', 'content': definitions}, messages[1]]}
        if common is not None and fixed != common:
            raise ValueError('request parameters, definitions or query differ')
        common = fixed
    if examples['A'] != examples['B'] or examples['C'] != examples['D']:
        raise ValueError('instruction change also changed selected examples')


def factorial_bootstrap(records, seed=42, repeats=1000):
    if not records:
        raise ValueError('empty paired records')
    if type(repeats) is not int or repeats < 20:
        raise ValueError('at least 20 bootstrap repeats required')
    if any(not all(a in r for a in ARMS) for r in records):
        raise ValueError('unpaired records')
    by_id = {r['sample_id']: r for r in records}
    if len(by_id) != len(records):
        raise ValueError('duplicate paired sample IDs')
    ids = sorted(by_id)
    counts = {}
    for arm in ARMS:
        counts[arm] = {}
        for sid, row in by_id.items():
            truth, predicted = set(row['truth']), set(row[arm]['predicted_labels'])
            counts[arm][sid] = (len(truth & predicted), len(predicted - truth), len(truth - predicted))

    def effects(selected):
        values = {}
        for arm in ARMS:
            tp, fp, fn = [sum(counts[arm][sid][i] for sid in selected) for i in range(3)]
            values[arm] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        result = {name: values[right] - values[left] for name, (left, right) in COMPARISONS.items()}
        result['interaction'] = result['D-C'] - result['B-A']
        return result

    point = effects(ids)
    rng = random.Random(seed)
    draws = []
    for _ in range(repeats):
        selected = rng.choices(ids, k=len(ids))
        draws.append({'sample_ids': selected, **effects(selected)})

    def interval(name):
        ordered = sorted(r[name] for r in draws)
        return {'value': point[name], 'ci_low': ordered[int(.025 * (repeats - 1))],
                'ci_high': ordered[int(.975 * (repeats - 1))]}

    return {'samples': len(ids), 'seed': seed, 'repeats': repeats, 'confidence': .95,
            'method': 'paired-percentile; common sample draws for all four arms; no multiplicity adjustment',
            'interaction_formula': '(D-C)-(B-A), on Micro-F1 scale',
            'interaction': interval('interaction'),
            'effects': {name: interval(name) for name in COMPARISONS}, 'bootstrap_draws': draws}


def save_factorial(store, evaluation_ids, result, sample_ids):
    """Create a distinct analysis record; never append to a frozen pairwise result."""
    if set(evaluation_ids) != set(ARMS) or len(set(sample_ids)) != result['samples']:
        raise ValueError('factorial members or evaluations differ')
    reference = None
    for eid in evaluation_ids.values():
        row = store.one('SELECT * FROM evaluations WHERE evaluation_id=?', (eid,))
        if row['status'] != 'completed':
            raise ValueError('factorial requires completed evaluations')
        identity = (row['sample_set_id'], row['ground_truth_manifest_sha256'], store.json(row['evaluation_config_artifact_id']))
        if reference is not None and identity != reference:
            raise ValueError('factorial evaluation definitions differ')
        reference = identity
        actual = [r['sample_id'] for r in store.rows('SELECT i.sample_id FROM evaluation_items e JOIN run_items i USING(run_item_id) WHERE e.evaluation_id=?', (eid,))]
        if sorted(actual) != sorted(sample_ids):
            raise ValueError('factorial paired samples differ')
    with store.transaction():
        aid = store.put({**result, 'evaluation_ids': evaluation_ids, 'analysis_source': Path(__file__).read_text()},
                        'prompt_factorial_bootstrap', inline=False)
        # A and D anchor the overall experiment. The analysis configuration names
        # all four evaluations and explicitly identifies this as an interaction.
        cid = store.insert('comparisons', baseline_evaluation_id=evaluation_ids['A'], candidate_evaluation_id=evaluation_ids['D'],
                           analysis_config_artifact_id=store.put({'version': 'paired-factorial-v1', 'kind': 'four-arm-interaction',
                               'evaluation_ids': evaluation_ids, 'formula': result['interaction_formula'],
                               'seed': result['seed'], 'repeats': result['repeats'], 'confidence': result['confidence']}, 'comparison_config'),
                           paired_members_artifact_id=store.put(sorted(sample_ids), 'paired_members'),
                           result_artifact_id=aid, status='building')
        effect = result['interaction']
        store.insert('metric_values', comparison_id=cid, metric_name='interaction_micro_f1', scope_key='factorial/A,B,C,D',
                     value=effect['value'], ci_low=effect['ci_low'], ci_high=effect['ci_high'], support=result['samples'])
        store.transition('comparisons', 'comparison_id', cid, 'completed', completed_at=now())
    return {'comparison_id': cid, 'artifact_id': aid}
