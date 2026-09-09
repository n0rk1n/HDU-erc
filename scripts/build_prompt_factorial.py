"""Reuse the frozen BGE train index; encode only a new, unexposed dev200."""
import importlib.metadata
import json
from pathlib import Path
import platform

from emotion_lab.datasets import make_set
from emotion_lab.embeddings import create_query_bundle, load_index, load_query_bundle
from emotion_lab.storage import Store, digest, now

ROOT = Path('data/research')
STATE = ROOT / 'prompt_factorial_build_state.json'


def save(state):
    pending = STATE.with_suffix('.pending')
    pending.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    pending.replace(STATE)


def main():
    previous = json.loads((ROOT / 'semantic_build_state.json').read_text())
    assert previous.get('completed_at')
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        'started_at': now(), **{k: previous[k] for k in ['corpus_id', 'index_id', 'model_config', 'model_checkpoint_artifact_id']}}
    assert state['index_id'] == previous['index_id']
    model_path = ROOT / 'models/bge-base-en-v1.5-a5beb1e'
    files = {str(p.relative_to(model_path)): digest(p.read_bytes()) for p in model_path.rglob('*')
             if p.is_file() and '.cache' not in p.parts}
    assert files == state['model_config']['files'], 'fixed model files changed'
    assert {name: importlib.metadata.version(name) for name in state['model_config']['dependencies']} == state['model_config']['dependencies']
    with Store(ROOT) as store:
        index = load_index(store, state['index_id'], state['corpus_id'])
        assert len(index['vectors']) == 43410
        if 'evaluation_set_id' not in state:
            excluded = sorted({r['evaluation_set_id'] for r in store.rows(
                "SELECT DISTINCT e.evaluation_set_id FROM experiments e JOIN sample_sets s ON s.sample_set_id=e.evaluation_set_id WHERE s.source_split='dev'")})
            selection = {'name': 'goemotions-dev200-prompt-factorial-20260909', 'purpose': 'dev_pilot', 'split': 'dev',
                         'seed': 90909, 'limit': 200, 'deduplicate': True,
                         'exclude_set_ids': sorted(set(excluded + [state['corpus_id']]))}
            found = store.rows('SELECT sample_set_id,selection_config_artifact_id FROM sample_sets WHERE name=?', (selection['name'],))
            assert len(found) <= 1
            if found:
                frozen = store.json(found[0]['selection_config_artifact_id'])
                assert all(frozen[k] == v for k, v in selection.items())
                state['evaluation_set_id'] = found[0]['sample_set_id']
            else:
                state['evaluation_set_id'] = make_set(store, selection)
            state['excluded_evaluation_set_ids'] = excluded
            state['selection'] = selection
            save(state)
        queries = store.rows('SELECT s.* FROM sample_set_members m JOIN samples s USING(sample_id) WHERE m.sample_set_id=? ORDER BY m.ordinal', (state['evaluation_set_id'],))
        assert len(queries) == 200
        if 'queries_artifact_id' not in state:
            import torch
            from transformers import AutoModel, AutoTokenizer
            from scripts.build_semantic_experiment import encode
            torch.set_num_threads(4); torch.manual_seed(42)
            device = state['model_config']['device']
            if device == 'mps' and not torch.backends.mps.is_available():
                raise RuntimeError('frozen encoding protocol requires MPS access')
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModel.from_pretrained(model_path, local_files_only=True, dtype=torch.float32,
                                               attn_implementation='eager').to(device).eval()
            matrix, encoding = encode(queries, tokenizer, model, device, 'prompt-factorial-dev200-20260909', state['model_config'])
            environment = {'platform': platform.platform(), 'python': platform.python_version(),
                           'dependencies': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                           'source': Path(__file__).read_text(),
                           'encoder_source': Path('scripts/build_semantic_experiment.py').read_text(),
                           'model_config': state['model_config'], 'model_checkpoint_artifact_id': state['model_checkpoint_artifact_id'],
                           'encoding': encoding}
            state['queries_artifact_id'] = create_query_bundle(store, state['index_id'], queries, matrix, environment)
            save(state)
        assert len(load_query_bundle(store, state['queries_artifact_id'], index)) == 200
        state['completed_at'] = now(); save(state)
        with store.transaction():
            aid = store.put(state, 'prompt_factorial_build_manifest', inline=False)
            store.event('index_id', state['index_id'], 'prompt_factorial_queries_verified', payload=aid)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
