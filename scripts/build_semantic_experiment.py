"""Build the approved BGE index and a new dev slice without model-label tuning."""
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from emotion_lab.datasets import make_set
from emotion_lab.embeddings import build_index, corpus_rows, create_query_bundle, load_index
from emotion_lab.storage import Store, digest, now

ROOT = Path('data/research')
MODEL = ROOT / 'models/bge-base-en-v1.5-a5beb1e'
REVISION = 'a5beb1e3e68b9ab74eb54cfd186867f64f240e1a'
STATE = ROOT / 'semantic_build_state.json'


def save(state):
    pending = STATE.with_suffix('.pending')
    pending.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    pending.replace(STATE)


def encode(records, tokenizer, model, device, label, config):
    cache = ROOT / 'semantic_cache'
    cache.mkdir(exist_ok=True)
    matrix_path = cache / (label + '.npy')
    metadata_path = cache / (label + '.json')
    fingerprint = digest({'inputs': [(r['sample_id'], r['text_sha256']) for r in records], 'config': config})
    if matrix_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata['input_fingerprint'] != fingerprint or digest(matrix_path.read_bytes()) != metadata['matrix_file_sha256']:
            raise ValueError('encoding cache differs from frozen inputs')
        return np.load(matrix_path, allow_pickle=False), metadata
    batches, row_info, vectors = [], [], []
    started = now()
    with (cache / (label + '-batches.jsonl')).open('w') as log:
        for start in range(0, len(records), 32):
            batch = records[start:start + 32]
            texts = [r['raw_text'] for r in batch]
            original = tokenizer(texts, truncation=False, add_special_tokens=True)['input_ids']
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            stamp = now(); tick = time.perf_counter()
            with torch.inference_mode():
                output = model(**{k: v.to(device) for k, v in encoded.items()})
                result = output.last_hidden_state[:, 0].float().cpu().numpy()
            elapsed = time.perf_counter() - tick
            vectors.append(result)
            for i, record in enumerate(batch):
                used = encoded['input_ids'][i][encoded['attention_mask'][i].bool()].tolist()
                row_info.append({'sample_id': record['sample_id'], 'text_sha256': record['text_sha256'],
                                 'original_tokens': len(original[i]), 'used_input_ids': used,
                                 'truncated': len(original[i]) > 512})
            item = {'offset': start, 'count': len(batch), 'started_at': stamp, 'seconds': elapsed,
                    'padded_length': int(encoded['input_ids'].shape[1]), 'raw_vectors_sha256': digest(result.tobytes())}
            batches.append(item); log.write(json.dumps(item) + '\n'); log.flush()
            if start == 0 or (start // 32 + 1) % 50 == 0 or start + len(batch) == len(records):
                print(json.dumps({'phase': label, 'encoded': start + len(batch), 'total': len(records), 'at': now()}), flush=True)
    matrix = np.concatenate(vectors).astype('<f4')
    np.save(matrix_path, matrix, allow_pickle=False)
    metadata = {'input_fingerprint': fingerprint, 'started_at': started, 'completed_at': now(),
                'matrix_file_sha256': digest(matrix_path.read_bytes()), 'device': device, 'batch_size': 32,
                'pooling': 'last_hidden_state[:,0]', 'normalize_after_encoding': True,
                'truncated_count': sum(r['truncated'] for r in row_info), 'rows': row_info, 'batches': batches}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False))
    return matrix, metadata


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {'started_at': now()}
    files = {str(p.relative_to(MODEL)): digest(p.read_bytes()) for p in MODEL.rglob('*')
             if p.is_file() and '.cache' not in p.parts}
    if 'model.safetensors' not in files:
        raise ValueError('fixed BGE weights missing')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_config = {'name': 'BAAI/bge-base-en-v1.5', 'revision': REVISION, 'files': files,
                    'pooling': 'cls', 'input': 'raw-text-only; symmetric; no instruction', 'max_length': 512,
                    'truncation': True, 'normalization': 'float64-l2-to-float32', 'dtype': 'float32',
                    'device': device, 'batch_size': 32, 'attention': 'eager', 'seed': 42}
    model_config['dependencies'] = {name: importlib.metadata.version(name) for name in ['numpy', 'torch', 'transformers', 'tokenizers']}
    environment = {'platform': platform.platform(), 'python': platform.python_version(),
                   'dependencies': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                   'source': Path(__file__).read_text()}
    if state.get('model_config') and state['model_config'] != model_config:
        raise ValueError('cannot change an existing encoding configuration')
    state['model_config'] = model_config; save(state)
    with Store(ROOT) as store:
        if 'model_checkpoint_artifact_id' not in state:
            bundle = io.BytesIO()
            with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_STORED) as archive:
                for name in sorted(files):
                    archive.write(MODEL / name, name)
            with store.transaction():
                state['model_checkpoint_artifact_id'] = store.put(bundle.getvalue(), 'embedding_model_checkpoint',
                                                                 inline=False, media_type='application/zip')
                store.event('artifact_id', state['model_checkpoint_artifact_id'], 'model_checkpoint_archived')
            save(state)
            del bundle
        environment['model_checkpoint_artifact_id'] = state['model_checkpoint_artifact_id']
        corpus_id = store.one("SELECT sample_set_id FROM sample_sets WHERE name='goemotions-train' AND status='frozen'")['sample_set_id']
        state['corpus_id'] = corpus_id
        if 'evaluation_set_id' not in state:
            # Exclude all previously run dev sets, plus exact normalized train copies.
            excluded = sorted({r['evaluation_set_id'] for r in store.rows(
                "SELECT DISTINCT e.evaluation_set_id FROM experiments e JOIN sample_sets s ON s.sample_set_id=e.evaluation_set_id WHERE s.source_split='dev'")})
            selection = {'name': 'goemotions-dev200-semantic-20260908', 'purpose': 'dev_pilot', 'split': 'dev',
                         'seed': 90908, 'limit': 200, 'deduplicate': True,
                         'exclude_set_ids': sorted(set(excluded + [corpus_id]))}
            previous = store.rows('SELECT sample_set_id,selection_config_artifact_id FROM sample_sets WHERE name=?', (selection['name'],))
            if previous:
                saved = store.json(previous[0]['selection_config_artifact_id'])
                assert all(saved[k] == v for k, v in selection.items())
                state['evaluation_set_id'] = previous[0]['sample_set_id']
            else:
                state['evaluation_set_id'] = make_set(store, selection)
            state['excluded_evaluation_set_ids'] = excluded
            save(state)
        records = corpus_rows(store, corpus_id)
        queries = store.rows('SELECT s.* FROM sample_set_members m JOIN samples s USING(sample_id) WHERE m.sample_set_id=? ORDER BY m.ordinal', (state['evaluation_set_id'],))
        if 'index_id' not in state or 'queries_artifact_id' not in state:
            tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
            model = AutoModel.from_pretrained(MODEL, local_files_only=True, dtype=torch.float32, attn_implementation='eager').to(device).eval()
            if 'index_id' not in state:
                matrix, metadata = encode(records, tokenizer, model, device, 'train', model_config)
                state['index_id'] = build_index(store, corpus_id, records, matrix, model_config, {**environment, 'encoding': metadata})
                save(state)
            matrix, metadata = encode(queries, tokenizer, model, device, 'fresh-dev200', model_config)
            if 'queries_artifact_id' not in state:
                state['queries_artifact_id'] = create_query_bundle(store, state['index_id'], queries, matrix, {**environment, 'encoding': metadata})
                save(state)
        index = load_index(store, state['index_id'], corpus_id)
        assert len(index['records']) == 43410 and len(queries) == 200
        state['completed_at'] = now(); save(state)
        with store.transaction():
            aid = store.put(state, 'semantic_build_manifest', inline=False)
            store.event('index_id', state['index_id'], 'build_verified', payload=aid)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
