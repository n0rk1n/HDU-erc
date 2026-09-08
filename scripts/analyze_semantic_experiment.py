"""Reconcile four arms against source TSV, raw responses and exact vector scores."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
import statistics

import numpy as np

from emotion_lab.datasets import normalize
from emotion_lab.embeddings import load_index, load_query_bundle
from emotion_lab.storage import Store, digest, now

ROOT = Path('data/research')
ARMS = ['A', 'B', 'C', 'D']
LABELS = Path('data/benchmarks/goemotions/emotions.txt').read_text().splitlines()


def source(split):
    path = Path('data/benchmarks/goemotions') / f'{split}.tsv'
    manifest = json.loads(path.with_name('manifest.json').read_text())
    assert digest(path.read_bytes()) == manifest['files'][path.name]['sha256']
    with path.open() as f:
        return {r[2]: {'text': r[0], 'labels': {LABELS[int(x)] for x in r[1].split(',')}}
                for r in csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)}


def metrics(records, arm, family):
    counts = {label: [0, 0, 0] for label in LABELS}
    exact = hit = family_hit = family_exact = ftp = ffp = ffn = 0
    for r in records:
        true, pred = set(r['truth']), set(r[arm]['predicted_labels'])
        exact += true == pred; hit += bool(true & pred)
        for label in true | pred:
            counts[label][0] += label in true & pred
            counts[label][1] += label in pred - true
            counts[label][2] += label in true - pred
        tf, pf = {family[x] for x in true}, {family[x] for x in pred}
        family_hit += bool(tf & pf); family_exact += tf == pf
        ftp += len(tf & pf); ffp += len(pf - tf); ffn += len(tf - pf)
    per = {label: {'tp': tp, 'fp': fp, 'fn': fn, 'support': tp + fn,
                   'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}
           for label, (tp, fp, fn) in counts.items()}
    tp, fp, fn = [sum(v[i] for v in counts.values()) for i in range(3)]
    return {'n': len(records), 'tp': tp, 'fp': fp, 'fn': fn,
            'micro_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'macro_f1': sum(v['f1'] for v in per.values()) / 28,
            'precision': tp / (tp + fp) if tp + fp else 0., 'recall': tp / (tp + fn) if tp + fn else 0.,
            'exact_match': exact / len(records), 'any_hit': hit / len(records),
            'family_any_hit': family_hit / len(records), 'family_exact_match': family_exact / len(records),
            'family_micro_f1': 2 * ftp / (2 * ftp + ffp + ffn) if 2 * ftp + ffp + ffn else 0.,
            'mean_predicted_labels': (tp + fp) / len(records), 'per_label': per}


def main():
    state = json.loads((ROOT / 'semantic_experiment_state.json').read_text())
    assert state.get('completed_at')
    train, dev = source('train'), source('dev')
    records = {}; info = {}; audits = Counter()
    shared_request = {}; shared_config = None
    with Store(ROOT) as store:
        build = state['build']
        index = load_index(store, build['index_id'], build['corpus_id'])
        queries = load_query_bundle(store, build['queries_artifact_id'], index)
        corpus = index['records']
        for row in corpus:
            assert row['raw_text'] == train[row['source_id']]['text']
        audits['original_train_texts'] = len(corpus)
        lexical_tokens = [set(__import__('re').findall(r'\w+', r['raw_text'].casefold())) for r in corpus]
        score_cache = {}
        excluded = set()
        for sid in build['excluded_evaluation_set_ids'] + [build['corpus_id']]:
            excluded.update(r['normalized_text_sha256'] for r in store.rows(
                'SELECT s.normalized_text_sha256 FROM sample_set_members m JOIN samples s USING(sample_id) WHERE m.sample_set_id=?', (sid,)))
        for arm in ARMS:
            rid = state['runs'][arm]
            run = store.one('SELECT * FROM experiment_runs WHERE run_id=?', (rid,))
            assert run['status'] in ('completed', 'completed_with_errors')
            cfg = store.json(run['method_config_artifact_id'])
            common = {k: v for k, v in cfg.items() if k not in {'method', 'name', 'example_policy', 'embedding'}}
            if shared_config is None:
                shared_config = common
            assert common == shared_config
            assert run['index_id'] == (build['index_id'] if cfg['method'] == 'semantic' else None)
            frozen = store.json(run['prompt_config_artifact_id'])
            family = store.json(run['taxonomy_config_artifact_id'])['families']
            items = store.rows('SELECT i.*,s.source_id,s.raw_text,s.normalized_text_sha256,s.text_sha256 FROM run_items i JOIN samples s USING(sample_id) WHERE i.run_id=? ORDER BY i.ordinal', (rid,))
            assert len(items) == 200
            for item in items:
                gold = dev[item['source_id']]
                assert gold['text'] == item['raw_text'] and item['normalized_text_sha256'] not in excluded
                truth = {r['label_name'] for r in store.rows('SELECT d.label_name FROM sample_labels l JOIN label_definitions d ON d.dataset_version_id=l.dataset_version_id AND d.label_id=l.label_id WHERE l.sample_id=?', (item['sample_id'],))}
                assert truth == gold['labels']; audits['truths'] += 1
                record = records.setdefault(item['source_id'], {'source_id': item['source_id'], 'sample_id': item['sample_id'], 'text': gold['text'], 'truth': sorted(truth)})
                attempts = store.rows('SELECT a.* FROM execution_steps s JOIN call_attempts a USING(step_id) WHERE s.run_item_id=? ORDER BY a.attempt_no', (item['run_item_id'],))
                assert attempts and len({a['request_sha256'] for a in attempts}) == 1
                for a in attempts:
                    assert digest(store.read(a['request_artifact_id'])) == a['request_sha256']
                request = store.json(attempts[0]['request_artifact_id'])
                assert request['messages'][-1] == {'role': 'user', 'content': gold['text']}
                system, examples_text = request['messages'][0]['content'].split('\nTraining examples:\n')
                assert system.split('\nLabel definitions:')[0] == frozen['instruction']
                reduced = {**request, 'messages': [{'role': 'system', 'content': system}, request['messages'][-1]]}
                if item['source_id'] in shared_request:
                    assert shared_request[item['source_id']] == reduced
                else:
                    shared_request[item['source_id']] = reduced
                audits['requests'] += 1
                step = store.one("SELECT * FROM execution_steps WHERE run_item_id=? AND kind='retrieve'", (item['run_item_id'],))
                trace = store.json(step['output_artifact_id'])
                assert trace['corpus_sample_ids'] == [r['sample_id'] for r in corpus]
                stored_scores = np.asarray(trace['full_scores'])
                key = (cfg['method'], item['sample_id'])
                if key not in score_cache:
                    if cfg['method'] == 'semantic':
                        query = queries[item['sample_id']]
                        assert query['text_sha256'] == item['text_sha256']
                        expected = np.einsum('ij,j->i', index['vectors'], query['vector'], dtype=np.float64, optimize=False)
                    else:
                        q = set(__import__('re').findall(r'\w+', item['raw_text'].casefold()))
                        expected = np.asarray([len(q & t) / len(q | t) if q | t else 0. for t in lexical_tokens])
                    assert np.allclose(stored_scores, expected, atol=2e-6, rtol=0)
                    score_cache[key] = stored_scores
                else:
                    assert np.array_equal(stored_scores, score_cache[key])
                order = sorted(range(len(corpus)), key=lambda n: (-stored_scores[n], corpus[n]['source_id']))
                assert trace['candidate_order'] == [corpus[n]['sample_id'] for n in order]
                candidates = store.rows('SELECT r.*,s.source_id,s.raw_text,s.split FROM retrieval_items r JOIN samples s USING(sample_id) WHERE retrieval_step_id=? ORDER BY candidate_rank', (step['step_id'],))
                assert len(candidates) == 50 and {r['sample_id'] for r in candidates} == {corpus[n]['sample_id'] for n in order[:50]}
                ranks = {corpus[n]['sample_id']: (rank, n) for rank, n in enumerate(order[:50], 1)}
                for c in candidates:
                    rank, n = ranks[c['sample_id']]
                    assert c['split'] == 'train' and c['raw_text'] == train[c['source_id']]['text']
                    assert json.loads(c['score_components_json'])['original_rank'] == rank
                    assert math.isclose(c['similarity_score'], stored_scores[n], abs_tol=1e-12)
                audits['candidate_records'] += len(candidates)
                audits['full_score_vectors'] += 1
                selected = sorted([r for r in candidates if r['decision'] == 'selected'], key=lambda r: r['selected_rank'])
                examples = json.loads(examples_text)
                assert len(selected) == len(examples) == 4
                assert len({normalize(x['text']) for x in examples}) == 4
                evidence = []
                for selected_row, ex in zip(selected, examples):
                    original = train[selected_row['source_id']]
                    assert ex['text'] == original['text'] and set(ex['labels']) == original['labels']
                    assert normalize(ex['text']) != normalize(gold['text'])
                    evidence.append({'sample_id': selected_row['sample_id'], 'source_id': selected_row['source_id'],
                                     'text': ex['text'], 'labels': ex['labels'], 'score': selected_row['similarity_score'],
                                     'components': json.loads(selected_row['score_components_json'])})
                audits['selected_examples'] += 4
                if cfg['method'] == 'semantic':
                    assert trace['index_id'] == build['index_id'] and trace['queries_artifact_id'] == build['queries_artifact_id']
                    assert trace['query_vector_sha256'] == queries[item['sample_id']]['vector_sha256']
                predicted = []
                if item['final_prediction_id']:
                    pred = store.one('SELECT * FROM predictions WHERE prediction_id=?', (item['final_prediction_id'],))
                    attempt = next(a for a in attempts if a['call_attempt_id'] == pred['call_attempt_id'])
                    raw = store.json(attempt['response_artifact_id'])
                    native = json.loads(raw['choices'][0]['message']['content'])['labels']
                    saved = {r['label_name'] for r in store.rows('SELECT d.label_name FROM prediction_labels p JOIN label_definitions d ON d.dataset_version_id=p.dataset_version_id AND d.label_id=p.label_id WHERE p.prediction_id=?', (item['final_prediction_id'],))}
                    assert set(native) == saved and saved <= set(LABELS)
                    predicted = sorted(saved); audits['raw_predictions'] += 1
                    for source_key, field in [('prompt_tokens', 'input_tokens'), ('completion_tokens', 'output_tokens'), ('total_tokens', 'total_tokens')]:
                        assert raw['usage'][source_key] == attempt[field]
                    assert attempt['input_tokens'] + cfg['model']['max_tokens'] + cfg['model']['safety_tokens'] <= cfg['model']['context_tokens']
                record[arm] = {'predicted_labels': predicted, 'status': item['status'], 'prediction_id': item['final_prediction_id'],
                               'run_item_id': item['run_item_id'], 'call_attempt_ids': [a['call_attempt_id'] for a in attempts],
                               'selected_examples': evidence, 'contrast_pair': trace['contrast_pair']}
            calls = store.rows('SELECT * FROM v_call_audit WHERE run_id=?', (rid,))
            retrieval_ms = [r['latency_ms'] for r in store.rows("SELECT e.latency_ms FROM execution_steps e JOIN run_items i USING(run_item_id) WHERE i.run_id=? AND e.kind='retrieve'", (rid,))]
            info[arm] = {'run_id': rid, 'evaluation_id': state['evaluations'][arm], 'config': cfg, 'code_commit': run['code_commit'],
                         'calls': len(calls), 'retries': sum(a['attempt_no'] > 1 for a in calls),
                         'succeeded': sum(i['status'] == 'succeeded' for i in items), 'failed': sum(i['status'] == 'failed' for i in items),
                         'resolved_models': dict(Counter(a['resolved_model'] for a in calls)),
                         'mean_call_latency_ms': statistics.mean(a['latency_ms'] for a in calls if a['latency_ms'] is not None),
                         'retrieval_total_ms': sum(retrieval_ms), 'retrieval_median_ms': statistics.median(retrieval_ms),
                         'unknown_cost_calls': sum(a['estimated_cost_micros'] is None for a in calls),
                         **{k: sum(a[k] for a in calls if a[k] is not None) for k in ['input_tokens', 'output_tokens', 'cached_input_tokens', 'reasoning_tokens']}}
            print(json.dumps({'verified_arm': arm, 'audits': dict(audits), 'at': now()}), flush=True)
        paired = sorted(records.values(), key=lambda r: r['source_id'])
        assert len(paired) == 200 and all(all(a in r for a in ARMS) for r in paired)
        scores = {a: metrics(paired, a, family) for a in ARMS}
        retrieval_diagnostics = {}
        for arm in ARMS:
            retrieval_diagnostics[arm] = {
                'contrast_found': sum(bool(r[arm]['contrast_pair']) for r in paired),
                'example_label_any_gold_coverage': sum(bool(set(r['truth']) & {label for ex in r[arm]['selected_examples'] for label in ex['labels']}) for r in paired) / len(paired),
                'mean_example_label_jaccard': statistics.mean(len(set(r['truth']) & set(ex['labels'])) / len(set(r['truth']) | set(ex['labels'])) for r in paired for ex in r[arm]['selected_examples']),
                'scope': 'post-evaluation label-overlap diagnostics, never used for example selection',
            }
        for arm in ARMS:
            db = {r['metric_name']: r['value'] for r in store.rows("SELECT metric_name,value FROM metric_values WHERE evaluation_id=? AND scope_key='overall'", (state['evaluations'][arm],))}
            for name in ['micro_f1', 'macro_f1', 'exact_match']:
                assert math.isclose(db[name], scores[arm][name], abs_tol=1e-12)
        comparisons = {}
        for name, cid in state['comparisons'].items():
            candidate, baseline = name.split('-')
            row = store.one('SELECT * FROM comparisons WHERE comparison_id=?', (cid,))
            result = store.json(row['result_artifact_id'])
            assert math.isclose(result['micro_f1_delta'], scores[candidate]['micro_f1'] - scores[baseline]['micro_f1'], abs_tol=1e-12)
            changed = [r for r in paired if [x['source_id'] for x in r[candidate]['selected_examples']] != [x['source_id'] for x in r[baseline]['selected_examples']]]
            comparisons[name] = {k: v for k, v in result.items() if k != 'bootstrap_deltas'}
            comparisons[name].update(comparison_id=cid, changed_examples=len(changed),
                                     exact_fixed=sum(set(r['truth']) == set(r[candidate]['predicted_labels']) and set(r['truth']) != set(r[baseline]['predicted_labels']) for r in paired),
                                     exact_regressed=sum(set(r['truth']) != set(r[candidate]['predicted_labels']) and set(r['truth']) == set(r[baseline]['predicted_labels']) for r in paired))
        summary = {'created_at': now(), 'state': state, 'runs': info, 'metrics': scores, 'comparisons': comparisons,
                   'retrieval_diagnostics': retrieval_diagnostics,
                   'independent_validation': dict(audits), 'test_split_used': False,
                   'zero_support_labels': [label for label in LABELS if scores['A']['per_label'][label]['support'] == 0],
                   'scope': 'new dev200 screening; four sequential arms; fixed v3; no unobserved-test generalization claim'}
        with store.transaction():
            details = store.put({'records': paired, 'analysis_source': Path(__file__).read_text()}, 'semantic_paired_cases', inline=False)
            summary['paired_details_artifact_id'] = details
            aid = store.put(summary, 'semantic_experiment_analysis', inline=False)
            for cid in state['comparisons'].values():
                store.event('comparison_id', cid, 'independently_validated', payload=aid)
        summary['analysis_artifact_id'] = aid
        (ROOT / 'semantic_experiment_analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        with (ROOT / 'exports/semantic-dev200-paired-cases-20260908.jsonl').open('w') as out:
            for r in paired:
                out.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(json.dumps({'analysis_artifact_id': aid, 'comparisons': comparisons, 'independent_validation': dict(audits)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
