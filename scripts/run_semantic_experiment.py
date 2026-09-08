"""Four frozen retrieval arms, each 200 items, with resumable raw evidence."""
import argparse
import json
from pathlib import Path
import random

from emotion_lab.config import credentials, load_environment, validate_config
from emotion_lab.evaluation import compare, evaluate, export_evaluation
from emotion_lab.preparation import prepare_run
from emotion_lab.runner import execute_run
from emotion_lab.storage import Store, digest, now

ROOT = Path('data/research')
ARMS = {'A': ('lexical', 'ranked'), 'B': ('semantic', 'ranked'),
        'C': ('lexical', 'contrastive-v1'), 'D': ('semantic', 'contrastive-v1')}
COMPARISONS = {'B-A': ('A', 'B'), 'D-C': ('C', 'D'), 'D-B': ('B', 'D'), 'C-A': ('A', 'C')}


def config_path(arm):
    return Path(f'config/experiments/dev200-semantic-{arm}.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-file')
    parser.add_argument('--write-configs', action='store_true')
    args = parser.parse_args()
    build = json.loads((ROOT / 'semantic_build_state.json').read_text())
    assert build.get('completed_at')
    if args.write_configs:
        base = json.loads(Path('config/experiments/dev200-prompt-v4-control.json').read_text())
        base.pop('evaluation_set')
        base.update(evaluation_set_id=build['evaluation_set_id'], prompt_version='native-labels-v3',
                    research_question='冻结模型和指令，比较词面/语义检索与普通/对比示例选择的四组效果；新的 dev200 初步验证',
                    comparison_group='goemotions-semantic-dev200-20260908')
        for arm, (method, policy) in ARMS.items():
            cfg = {**base, 'name': f'GoEmotions semantic retrieval {arm}', 'method': method, 'example_policy': policy}
            if method == 'semantic':
                cfg['embedding'] = {'index_id': build['index_id'], 'queries_artifact_id': build['queries_artifact_id']}
            validate_config(cfg)
            path = config_path(arm)
            if path.exists() and json.loads(path.read_text()) != cfg:
                raise ValueError('existing experiment config differs')
            path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
        return
    if not args.env_file:
        parser.error('--env-file is required for real model calls')
    load_environment(args.env_file); credentials()
    configs = {arm: validate_config(json.loads(config_path(arm).read_text())) for arm in ARMS}
    common = [{k: v for k, v in cfg.items() if k not in {'name', 'method', 'example_policy', 'embedding'}} for cfg in configs.values()]
    assert all(c == common[0] for c in common)
    assert all(c['k'] == 4 and c['candidate_count'] == 50 and c['prompt_version'] == 'native-labels-v3' for c in configs.values())
    state_path = ROOT / 'semantic_experiment_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state:
        assert state['config_hash'] == digest(configs)
    else:
        state = {'started_at': now(), 'config_hash': digest(configs), 'build': build,
                 'run_order': random.Random(90908).sample(list(ARMS), len(ARMS)), 'runs': {}, 'evaluations': {}, 'comparisons': {}}
    def save():
        pending = state_path.with_suffix('.pending')
        pending.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        pending.replace(state_path)
    with Store(ROOT) as store:
        if 'driver_artifact_id' not in state:
            with store.transaction():
                aid = store.put({'source': Path(__file__).read_text(), 'configs': configs, 'run_order': state['run_order'],
                                 'primary_comparisons': ['B-A', 'D-C'], 'secondary_comparisons': ['D-B', 'C-A'],
                                 'scope': 'new fixed dev200; all four arms finish before scoring; no prompt tuning; test unused',
                                 'authorization': 'User approved the semantic retrieval six-step implementation and four-arm experiment.'},
                                'semantic_experiment_driver', inline=False)
                store.event('artifact_id', aid, 'semantic_experiment_authorized')
            state['driver_artifact_id'] = aid; save()
        for arm in ARMS:
            if arm not in state['runs']:
                print(json.dumps({'phase': 'prepare', 'arm': arm, 'at': now()}), flush=True)
                # Recover a prepared run if interruption occurred before the state file was saved.
                found = store.rows('SELECT r.run_id FROM experiment_runs r JOIN experiments e USING(experiment_id) WHERE e.name=? AND e.comparison_group=?',
                                   (configs[arm]['name'], configs[arm]['comparison_group']))
                assert len(found) <= 1
                if found:
                    row = store.one('SELECT method_config_artifact_id FROM experiment_runs WHERE run_id=?', (found[0]['run_id'],))
                    frozen_config = store.json(row['method_config_artifact_id'])
                    assert all(frozen_config.get(k) == v for k, v in configs[arm].items()), 'recovered run configuration differs'
                state['runs'][arm] = found[0]['run_id'] if found else prepare_run(store, configs[arm])
                save()
                print(json.dumps({'phase': 'frozen', 'arm': arm, 'run_id': state['runs'][arm], 'at': now()}), flush=True)
        for arm in state['run_order']:
            rid = state['runs'][arm]
            status = store.one('SELECT status FROM experiment_runs WHERE run_id=?', (rid,))['status']
            if status in ('completed', 'completed_with_errors'):
                continue
            def checkpoint(stage, attempt_id):
                if stage == 'response_saved':
                    code = store.one('SELECT http_status FROM call_attempts WHERE call_attempt_id=?', (attempt_id,))['http_status']
                    if code in (400, 401, 403, 404):
                        raise RuntimeError(f'paused after HTTP {code}; response retained')
                if stage == 'parsed':
                    n = store.one('SELECT count(*) n FROM v_call_audit WHERE run_id=? AND response_artifact_id IS NOT NULL', (rid,))['n']
                    if n <= 2 or n % 20 == 0:
                        print(json.dumps({'phase': 'predict', 'arm': arm, 'responses_saved': n, 'at': now()}), flush=True)
            execute_run(store, rid, resume=status != 'ready', checkpoint=checkpoint)
            assert store.one('SELECT status FROM experiment_runs WHERE run_id=?', (rid,))['status'] in ('completed', 'completed_with_errors')
            print(json.dumps({'phase': 'completed', 'arm': arm, 'at': now()}), flush=True)
        for arm in ARMS:
            if arm not in state['evaluations']:
                found = store.rows("SELECT evaluation_id FROM evaluations WHERE run_id=? AND status='completed'", (state['runs'][arm],))
                state['evaluations'][arm] = found[0]['evaluation_id'] if found else evaluate(store, state['runs'][arm], {})
                save()
            output = ROOT / 'exports' / f'semantic-dev200-{arm}-20260908'
            if not output.exists():
                export_evaluation(store, state['evaluations'][arm], output)
        for name, (baseline, candidate) in COMPARISONS.items():
            if name not in state['comparisons']:
                state['comparisons'][name] = compare(store, state['evaluations'][baseline], state['evaluations'][candidate], seed=42, repeats=1000)
                save()
        state['completed_at'] = now(); save()
        with store.transaction():
            aid = store.put(state, 'semantic_execution_manifest', inline=False)
            store.event('artifact_id', aid, 'semantic_experiment_executed')
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
