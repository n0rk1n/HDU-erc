"""Verify frozen code, all exports, database and portable final backup."""
import json
from pathlib import Path
import re

from emotion_lab.embeddings import load_index, load_query_bundle
from emotion_lab.preparation import application_sources
from emotion_lab.storage import Store, digest, now


def main():
    root = Path('data/research')
    state = json.loads((root / 'semantic_experiment_state.json').read_text())
    analysis = json.loads((root / 'semantic_experiment_analysis.json').read_text())
    assert state.get('completed_at')
    tests = (root / 'semantic_final_tests.log').read_text()
    match = re.search(r'(\d+) passed', tests)
    assert match and not re.search(r'\d+ (failed|error)', tests)
    checks = {'created_at': now(), 'software_tests': match.group(0), 'runs': {}, 'exports': {},
              'independent_validation': analysis['independent_validation'], 'official_test_used': False,
              'verification_source': Path(__file__).read_text()}
    assert checks['independent_validation']['requests'] == 800
    assert checks['independent_validation']['selected_examples'] == 3200
    assert checks['independent_validation']['candidate_records'] == 40000
    with Store(root) as store:
        for arm, rid in state['runs'].items():
            run = store.one('SELECT * FROM experiment_runs WHERE run_id=?', (rid,))
            assert run['status'] in ['completed', 'completed_with_errors']
            assert store.json(run['working_diff_artifact_id'])['application_sources'] == application_sources()
            counts = store.rows('SELECT status,count(*) n FROM run_items WHERE run_id=? GROUP BY status', (rid,))
            assert sum(r['n'] for r in counts) == 200 and all(r['status'] in ['succeeded', 'failed'] for r in counts)
            checks['runs'][arm] = {'run_id': rid, 'counts': counts, 'code_commit': run['code_commit']}
            export = root / 'exports' / f'semantic-dev200-{arm}-20260908'
            manifest = json.loads((export / 'manifest.json').read_text())
            assert manifest['run_id'] == rid and manifest['sample_count'] == 200
            for name, sha in manifest['files'].items():
                assert digest((export / name).read_bytes()) == sha
            checks['exports'][arm] = len(manifest['files'])
        index = load_index(store, state['build']['index_id'], state['build']['corpus_id'])
        queries = load_query_bundle(store, state['build']['queries_artifact_id'], index)
        checks['train_vectors'] = len(index['vectors']); checks['query_vectors'] = len(queries)
        assert checks['train_vectors'] == 43410 and checks['query_vectors'] == 200
        checks['database'] = store.verify()
        assert checks['database']['ok'] and not checks['database']['orphan_files']
        with store.transaction():
            sources = {str(p): p.read_text() for p in [root / 'semantic_execution.log', root / 'semantic_build.log',
                       root / 'semantic_experiment_state.json', root / 'semantic_final_tests.log',
                       Path('scripts/analyze_semantic_experiment.py'), Path('scripts/report_semantic_experiment.py'),
                       Path('docs/research/semantic-retrieval-results-20260908.md')]}
            checks['final_sources_artifact_id'] = store.put(sources, 'semantic_final_sources', inline=False)
            aid = store.put(checks, 'verification_report', inline=False)
            for cid in state['comparisons'].values():
                store.event('comparison_id', cid, 'completion_verified', payload=aid)
        backup = root / 'backups/semantic-retrieval-final-20260908'
        store.backup(backup)
    with Store(backup) as restored:
        checked = restored.verify()
        assert checked['ok'] and not checked['orphan_files']
        restored_index = load_index(restored, state['build']['index_id'], state['build']['corpus_id'])
        assert len(load_query_bundle(restored, state['build']['queries_artifact_id'], restored_index)) == 200
        for rid in state['runs'].values():
            assert restored.one('SELECT count(*) n FROM run_items WHERE run_id=?', (rid,))['n'] == 200
    print(json.dumps({'verification_artifact_id': aid, 'backup': str(backup), 'backup_verification': checked,
                      'checks': {k: v for k, v in checks.items() if k != 'verification_source'}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
