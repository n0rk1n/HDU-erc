"""Semantic retrieval must remain reproducible, train-only and recoverable."""
import json

import pytest

from emotion_lab.config import validate_config
from emotion_lab.preparation import prepare_run
from emotion_lab.retrieval import Retriever
from emotion_lab.runner import execute_run
from emotion_lab.storage import digest
from emotion_lab.taxonomy import load_taxonomy
from test_runner import Counter, response
from test_prompt_examples import contrast_corpus


def test_fresh_dev_set_excludes_previous_texts(configured):
    from emotion_lab.datasets import make_set
    store, cfg = configured
    ds = store.one('SELECT dataset_version_id FROM sample_sets WHERE sample_set_id=?', (cfg['evaluation_set_id'],))['dataset_version_id']
    old = make_set(store, {'dataset_version_id': ds, 'name': 'old', 'purpose': 'dev_pilot', 'split': 'dev', 'limit': 1, 'seed': 17})
    fresh = make_set(store, {'dataset_version_id': ds, 'name': 'fresh', 'purpose': 'dev_pilot', 'split': 'dev', 'exclude_set_ids': [old]})
    old_ids = {r['sample_id'] for r in store.rows('SELECT sample_id FROM sample_set_members WHERE sample_set_id=?', (old,))}
    new_ids = {r['sample_id'] for r in store.rows('SELECT sample_id FROM sample_set_members WHERE sample_set_id=?', (fresh,))}
    assert len(new_ids) == 1 and old_ids.isdisjoint(new_ids)


def resources(configured):
    from emotion_lab.embeddings import build_index, create_query_bundle, corpus_rows
    store, cfg = configured
    records = corpus_rows(store, cfg['corpus_set_id'])
    # First sample is joy/love; second is anger/annoyance.
    idx = build_index(store, cfg['corpus_set_id'], records, [[1, 0], [0, 1]],
                      {'name': 'fixture', 'revision': 'fixed', 'pooling': 'cls'}, {})
    queries = store.rows("SELECT * FROM samples WHERE split='dev' ORDER BY source_line")
    aid = create_query_bundle(store, idx, queries, [[0, 1], [1, 0]], {})
    cfg = {**cfg, 'method': 'semantic', 'k': 2, 'candidate_count': 2,
           'embedding': {'index_id': idx, 'queries_artifact_id': aid},
           'prompt_version': 'native-labels-v3'}
    cfg['model']['context_tokens'] = 30000
    return store, cfg, records, queries


def test_semantic_config_accepts_both_selection_policies():
    for policy in ['ranked', 'contrastive-v1']:
        cfg = validate_config({'method': 'semantic', 'k': 4, 'example_policy': policy,
                               'embedding': {'index_id': 'i', 'queries_artifact_id': 'q'}})
        assert cfg['embedding']['index_id'] == 'i'


@pytest.mark.parametrize('embedding', [None, {}, {'index_id': 'i'},
                         {'index_id': 'i', 'queries_artifact_id': 'q', 'unknown': 1}])
def test_missing_or_unknown_semantic_resources_fail(embedding):
    with pytest.raises(ValueError):
        validate_config({'method': 'semantic', 'k': 4, 'embedding': embedding})


def test_semantic_neighbor_order_and_full_native_labels(configured):
    store, cfg, records, queries = resources(configured)
    retriever = Retriever(store, cfg['corpus_set_id'])
    selected, candidates, trace, messages = retriever.select(
        queries[0], validate_config(cfg), load_taxonomy(), Counter())
    assert [r['source_id'] for r in selected] == ['t2', 't1']
    assert selected[0]['labels'] == ['anger', 'annoyance']
    assert candidates[0]['similarity_score'] == pytest.approx(1.0)
    assert trace['index_id'] == cfg['embedding']['index_id']
    assert trace['queries_artifact_id'] == cfg['embedding']['queries_artifact_id']
    assert trace['score_dtype'] == 'float32'
    assert messages[-1]['content'] == queries[0]['raw_text']


def test_semantic_copy_filter_ties_and_budget(configured):
    from emotion_lab.embeddings import create_query_bundle
    store, cfg, records, queries = resources(configured)
    aid = create_query_bundle(store, cfg['embedding']['index_id'], [records[0], queries[0]], [[1, 0], [1, 1]], {})
    cfg['embedding']['queries_artifact_id'] = aid
    cfg = validate_config(cfg)
    retriever = Retriever(store, cfg['corpus_set_id'])
    copied = retriever.select(records[0], cfg, load_taxonomy(), Counter())
    assert [r['source_id'] for r in copied[0]] == ['t2']
    assert copied[1][0]['reason_code'] == 'duplicate_text'
    tied = retriever.select(queries[0], cfg, load_taxonomy(), Counter())
    assert [r['source_id'] for r in tied[0]] == ['t1', 't2']
    cfg['model']['context_tokens'] = Counter().count(tied[3]) - 1 + cfg['model']['max_tokens'] + cfg['model']['safety_tokens']
    limited = retriever.select(queries[0], cfg, load_taxonomy(), Counter())
    assert len(limited[0]) == 1
    assert Counter().count(limited[3]) <= cfg['model']['context_tokens'] - cfg['model']['max_tokens'] - cfg['model']['safety_tokens']


def test_semantic_contrast_selects_adjacent_label_counterexample(contrast_corpus):
    from emotion_lab.embeddings import build_index, create_query_bundle, corpus_rows
    store, corpus_id = contrast_corpus
    records = corpus_rows(store, corpus_id)
    idx = build_index(store, corpus_id, records, [[1, 0, 0], [.99, .1, 0], [.98, .1, 0], [.97, .1, 0], [.8, .6, 0], [.7, .7, 0], [0, 0, 1]], {'name': 'fixture'}, {})
    query = store.one("SELECT * FROM samples WHERE split='dev' ORDER BY source_line LIMIT 1")
    aid = create_query_bundle(store, idx, [query], [[1, 0, 0]], {})
    cfg = validate_config({'method': 'semantic', 'k': 4, 'candidate_count': 7,
                           'embedding': {'index_id': idx, 'queries_artifact_id': aid},
                           'example_policy': 'ranked', 'model': {'context_tokens': 30000, 'max_tokens': 100}})
    r = Retriever(store, corpus_id)
    assert [x['source_id'] for x in r.select(query, cfg, load_taxonomy(), Counter())[0]] == ['a1', 'a2', 'a3', 'a4']
    cfg['example_policy'] = 'contrastive-v1'
    selected, _, trace, _ = r.select(query, cfg, load_taxonomy(), Counter())
    assert [x['source_id'] for x in selected] == ['a1', 'b1', 'a2', 'a3']
    assert selected[1]['labels'] == ['annoyance', 'disapproval']
    assert trace['contrast_pair']['contrast_label'] == 'annoyance'


def test_query_cache_rejects_wrong_text_and_wrong_index(configured):
    from emotion_lab.embeddings import create_query_bundle
    store, cfg, records, queries = resources(configured)
    bad = {**queries[0], 'raw_text': 'changed', 'text_sha256': digest(b'changed')}
    with pytest.raises(ValueError, match='query|text'):
        Retriever(store, cfg['corpus_set_id']).select(bad, validate_config(cfg), load_taxonomy(), Counter())
    payload = store.json(cfg['embedding']['queries_artifact_id'])
    payload['index_id'] = 'wrong'
    cfg['embedding']['queries_artifact_id'] = store.put(payload, 'invalid-fixture')
    with pytest.raises(ValueError, match='index'):
        prepare_run(store, cfg, counter=Counter())


@pytest.mark.parametrize('vectors', [[[0, 0], [1, 0]], [[float('nan'), 1], [1, 0]], [[1, 0]]])
def test_invalid_index_vectors_leave_no_partial_index(configured, vectors):
    from emotion_lab.embeddings import build_index, corpus_rows
    store, cfg = configured
    with pytest.raises(ValueError):
        build_index(store, cfg['corpus_set_id'], corpus_rows(store, cfg['corpus_set_id']),
                    vectors, {'name': 'fixture'}, {})
    assert not store.rows('SELECT * FROM embedding_indexes')


def test_index_rejects_wrong_row_order_and_nontrain_corpus(configured):
    from emotion_lab.embeddings import build_index, corpus_rows
    store, cfg = configured
    records = corpus_rows(store, cfg['corpus_set_id'])
    with pytest.raises(ValueError, match='manifest|order'):
        build_index(store, cfg['corpus_set_id'], records[::-1], [[1, 0], [0, 1]], {}, {})
    with pytest.raises(ValueError, match='corpus|train'):
        corpus_rows(store, cfg['evaluation_set_id'])


def test_index_roundtrip_detects_tampered_vector_and_reuses_identical_build(configured):
    from emotion_lab.embeddings import load_index, build_index
    store, cfg, records, queries = resources(configured)
    idx = cfg['embedding']['index_id']
    assert build_index(store, cfg['corpus_set_id'], records, [[1, 0], [0, 1]],
                       {'name': 'fixture', 'revision': 'fixed', 'pooling': 'cls'}, {}) == idx
    loaded = load_index(store, idx, cfg['corpus_set_id'])
    assert loaded['vectors'].tolist() == [[1, 0], [0, 1]]
    row = store.one('SELECT a.* FROM embedding_indexes i JOIN artifacts a ON a.artifact_id=i.index_bundle_artifact_id WHERE i.index_id=?', (idx,))
    (store.artifact_root / row['relative_path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash'):
        load_index(store, idx, cfg['corpus_set_id'])


@pytest.mark.parametrize('policy', ['ranked', 'contrastive-v1'])
def test_semantic_run_freezes_resources_and_resumes_without_duplicate_calls(configured, policy):
    store, cfg, records, queries = resources(configured)
    cfg['example_policy'] = policy
    run = prepare_run(store, cfg, counter=Counter())
    assert store.one('SELECT index_id FROM experiment_runs WHERE run_id=?', (run,))['index_id'] == cfg['embedding']['index_id']
    class Crash(BaseException):
        pass
    def crash(stage, attempt):
        if stage == 'response_saved':
            raise Crash()
    with pytest.raises(Crash):
        execute_run(store, run, transport=lambda request: response(), counter=Counter(), checkpoint=crash)
    execute_run(store, run, transport=lambda request: response(), counter=Counter(), resume=True)
    assert len(store.rows('SELECT * FROM call_attempts')) == 2
    assert store.one('SELECT status FROM experiment_runs WHERE run_id=?', (run,))['status'] == 'completed'
    assert all(r['split'] == 'train' for r in store.rows('SELECT s.split FROM retrieval_items r JOIN samples s USING(sample_id)'))
    assert store.verify()['ok']
