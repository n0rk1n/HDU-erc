"""Frozen, content-verified train matrices and query vectors for exact retrieval."""
import io

import numpy as np

from .storage import digest, now


def corpus_rows(store, corpus_id):
    corpus = store.one('SELECT * FROM sample_sets WHERE sample_set_id=?', (corpus_id,))
    if corpus['purpose'] != 'corpus' or corpus['status'] != 'frozen' or corpus['source_split'] != 'train':
        raise ValueError('index requires frozen train corpus')
    rows = store.rows('SELECT s.* FROM sample_set_members m JOIN samples s USING(sample_id) WHERE m.sample_set_id=? ORDER BY m.ordinal', (corpus_id,))
    if len(rows) != corpus['sample_count'] or digest([r['sample_id'] for r in rows]) != corpus['members_sha256']:
        raise ValueError('corpus manifest mismatch')
    return rows


def _manifest(records):
    return [{'sample_id': r['sample_id'], 'text_sha256': r['text_sha256']} for r in records]


def _vectors(values, count, normalize=False):
    matrix = np.asarray(values, dtype='<f4')
    if matrix.ndim != 2 or matrix.shape[0] != count or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError('invalid vector shape or nonfinite values')
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms < 1e-12):
        raise ValueError('zero vector')
    if normalize:
        matrix = (matrix / norms[:, None]).astype('<f4')
    elif not np.allclose(norms, 1, atol=2e-6, rtol=0):
        raise ValueError('vectors must be normalized')
    return np.ascontiguousarray(matrix)


def build_index(store, corpus_id, records, vectors, model_config, metadata):
    expected = corpus_rows(store, corpus_id)
    manifest = _manifest(expected)
    if _manifest(records) != manifest or any(digest(r['raw_text'].encode()) != r['text_sha256'] for r in records):
        raise ValueError('input manifest/order mismatch')
    matrix = _vectors(vectors, len(records), normalize=True)
    config = {'implementation': 'exact-cosine-v1', 'normalize': True, 'dtype': '<f4',
              'matrix_sha256': digest(matrix.tobytes()), 'tie_break': 'source_id-ascending'}
    fingerprint = digest({'corpus_id': corpus_id, 'manifest': manifest, 'model': model_config, 'config': config})
    existing = store.rows('SELECT index_id FROM embedding_indexes WHERE fingerprint=?', (fingerprint,))
    if existing:
        load_index(store, existing[0]['index_id'], corpus_id)
        return existing[0]['index_id']
    output = io.BytesIO()
    np.save(output, matrix, allow_pickle=False)
    with store.transaction():
        idx = store.insert('embedding_indexes', corpus_set_id=corpus_id,
                           index_config_artifact_id=store.put(config, 'embedding_index_config'),
                           model_config_artifact_id=store.put(model_config, 'embedding_model_config'),
                           build_environment_artifact_id=store.put(metadata, 'embedding_build_metadata', inline=False),
                           input_manifest_artifact_id=store.put(manifest, 'embedding_input_manifest', inline=False),
                           metric='cosine', fingerprint=fingerprint, status='building', started_at=now())
        for offset, record in enumerate(records):
            store.insert('embedding_items', index_id=idx, sample_id=record['sample_id'], vector_offset=offset,
                         input_sha256=record['text_sha256'], vector_sha256=digest(matrix[offset].tobytes()))
        store.transition('embedding_indexes', 'index_id', idx, 'ready',
                         index_bundle_artifact_id=store.put(output.getvalue(), 'embedding_matrix', inline=False,
                                                           media_type='application/x-npy'),
                         dimension=matrix.shape[1], dtype='<f4', completed_at=now())
    return idx


def load_index(store, index_id, corpus_id):
    row = store.one('SELECT * FROM embedding_indexes WHERE index_id=?', (index_id,))
    if row['status'] != 'ready' or row['corpus_set_id'] != corpus_id or row['metric'] != 'cosine':
        raise ValueError('index/corpus mismatch or index not ready')
    records = corpus_rows(store, corpus_id)
    manifest = store.json(row['input_manifest_artifact_id'])
    if manifest != _manifest(records):
        raise ValueError('index input manifest mismatch')
    matrix = np.load(io.BytesIO(store.read(row['index_bundle_artifact_id'])), allow_pickle=False)
    if matrix.dtype.str != '<f4' or matrix.ndim != 2 or matrix.shape[1] != row['dimension'] or row['dtype'] != '<f4':
        raise ValueError('index matrix dimension/dtype mismatch')
    matrix = _vectors(matrix, len(records))
    config = store.json(row['index_config_artifact_id'])
    model = store.json(row['model_config_artifact_id'])
    if digest(matrix.tobytes()) != config['matrix_sha256'] or digest({'corpus_id': corpus_id, 'manifest': manifest, 'model': model, 'config': config}) != row['fingerprint']:
        raise ValueError('index fingerprint mismatch')
    items = store.rows('SELECT * FROM embedding_items WHERE index_id=? ORDER BY vector_offset', (index_id,))
    if len(items) != len(records):
        raise ValueError('index mapping count mismatch')
    for n, (item, record) in enumerate(zip(items, records)):
        if item['vector_offset'] != n or item['sample_id'] != record['sample_id'] or item['input_sha256'] != record['text_sha256'] or item['vector_sha256'] != digest(matrix[n].tobytes()):
            raise ValueError('index row mapping/hash mismatch')
    matrix.setflags(write=False)
    return {'row': row, 'records': records, 'vectors': matrix, 'model_config': model}


def create_query_bundle(store, index_id, records, vectors, metadata):
    index = store.one('SELECT * FROM embedding_indexes WHERE index_id=?', (index_id,))
    matrix = _vectors(vectors, len(records), normalize=True)
    if index['status'] != 'ready' or matrix.shape[1] != index['dimension']:
        raise ValueError('query/index dimension mismatch')
    entries = []
    for n, record in enumerate(records):
        actual = store.one('SELECT * FROM samples WHERE sample_id=?', (record['sample_id'],))
        if actual['raw_text'] != record['raw_text'] or digest(record['raw_text'].encode()) != actual['text_sha256'] or record['text_sha256'] != actual['text_sha256']:
            raise ValueError('query text mismatch')
        entries.append({'sample_id': record['sample_id'], 'text_sha256': actual['text_sha256'],
                        'vector': matrix[n].tolist(), 'vector_sha256': digest(matrix[n].tobytes())})
    if len({r['sample_id'] for r in entries}) != len(entries):
        raise ValueError('duplicate query sample')
    with store.transaction():
        aid = store.put({'index_id': index_id, 'index_fingerprint': index['fingerprint'],
                         'records': entries, 'encoding_metadata': metadata}, 'embedding_queries', inline=False)
        store.event('index_id', index_id, 'queries_frozen', payload=aid)
    return aid


def load_query_bundle(store, artifact_id, index):
    payload = store.json(artifact_id)
    if payload['index_id'] != index['row']['index_id'] or payload['index_fingerprint'] != index['row']['fingerprint']:
        raise ValueError('query bundle index mismatch')
    records = payload['records']
    matrix = _vectors([r['vector'] for r in records], len(records))
    if matrix.shape[1] != index['vectors'].shape[1]:
        raise ValueError('query vector dimension mismatch')
    result = {}
    for n, record in enumerate(records):
        actual = store.one('SELECT text_sha256 FROM samples WHERE sample_id=?', (record['sample_id'],))
        if actual['text_sha256'] != record['text_sha256'] or digest(matrix[n].tobytes()) != record['vector_sha256']:
            raise ValueError('query vector/text hash mismatch')
        if record['sample_id'] in result:
            raise ValueError('duplicate query sample')
        result[record['sample_id']] = {**record, 'vector': matrix[n]}
    return result
