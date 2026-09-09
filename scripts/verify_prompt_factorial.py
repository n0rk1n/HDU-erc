"""Verify frozen code, all exports, database and portable final backup."""
import json
from pathlib import Path
import re

from emotion_lab.embeddings import load_index, load_query_bundle
from emotion_lab.preparation import application_sources
from emotion_lab.storage import Store, digest, now


def main():
    root = Path('data/research')
    state = json.loads((root / 'prompt_factorial_state.json').read_text())
    analysis = json.loads((root / 'prompt_factorial_analysis.json').read_text())
    assert state.get('completed_at')
    tests = (root / 'prompt_factorial_final_tests.log').read_text()
    match = re.search(r'(\d+) passed', tests)
    assert match and not re.search(r'\d+ (failed|error)', tests)
    checks = {'created_at': now(), 'software_tests': match.group(0), 'runs': {}, 'exports': {},
              'independent_validation': analysis['independent_validation'], 'official_test_used': False,
              'verification_source': Path(__file__).read_text()}
    assert checks['independent_validation']['factorial_request_pairs'] == 200
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
            export = root / 'exports' / f'prompt-factorial-dev200-{arm}-20260909'
            manifest = json.loads((export / 'manifest.json').read_text())
            assert manifest['run_id'] == rid and manifest['sample_count'] == 200
            for name, sha in manifest['files'].items():
                assert digest((export / name).read_bytes()) == sha
            checks['exports'][arm] = len(manifest['files'])
        index = load_index(store, state['build']['index_id'], state['build']['corpus_id'])
        queries = load_query_bundle(store, state['build']['queries_artifact_id'], index)
        checks['train_vectors'] = len(index['vectors']); checks['query_vectors'] = len(queries)
        assert checks['train_vectors'] == 43410 and checks['query_vectors'] == 200
        assert len(store.json(analysis['factorial_artifact_id'])['bootstrap_draws']) == 1000
        assert analysis['comparisons']['B-A']['changed_examples'] == 0
        assert analysis['comparisons']['D-C']['changed_examples'] == 0
        checks['factorial_artifact_id'] = analysis['factorial_artifact_id']
        factorial_row = store.one('SELECT * FROM comparisons WHERE comparison_id=?', (analysis['factorial_comparison_id'],))
        assert factorial_row['status'] == 'completed' and factorial_row['result_artifact_id'] == analysis['factorial_artifact_id']
        assert store.json(factorial_row['analysis_config_artifact_id'])['evaluation_ids'] == state['evaluations']
        checks['factorial_comparison_id'] = analysis['factorial_comparison_id']
        checks['database'] = store.verify()
        assert checks['database']['ok'] and not checks['database']['orphan_files']
        with store.transaction():
            sources = {str(p): p.read_text() for p in [root / 'prompt_factorial_execution.log', root / 'prompt_factorial_build.log',
                       root / 'prompt_factorial_state.json', root / 'prompt_factorial_final_tests.log',
                       Path('scripts/analyze_prompt_factorial.py'), Path('scripts/report_prompt_factorial.py'),
                       Path('docs/research/prompt-factorial-results-20260909.md'),
                       Path('scripts/prompt_factorial_stats.py'), root / 'prompt_factorial_provenance_check.json',
                       root / 'prompt_factorial_original_workspace.json', root / 'prompt_factorial_history_check.json']}
            checks['final_sources_artifact_id'] = store.put(sources, 'prompt_factorial_final_sources', inline=False)
            aid = store.put(checks, 'verification_report', inline=False)
            for cid in [*state['comparisons'].values(), analysis['factorial_comparison_id']]:
                store.event('comparison_id', cid, 'completion_verified', payload=aid)
        backup = root / 'backups/prompt-factorial-final-20260909'
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
