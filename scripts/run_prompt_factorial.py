"""Four frozen instruction/example arms with resumable raw evidence."""
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
from scripts.prompt_factorial_stats import ARMS, COMPARISONS, validate_factorial



def config_path(arm):
    return Path(f'config/experiments/dev200-prompt-factorial-{arm}.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-file')
    parser.add_argument('--write-configs', action='store_true')
    args = parser.parse_args()
    build = json.loads((ROOT / 'prompt_factorial_build_state.json').read_text())
    assert build.get('completed_at')
    if args.write_configs:
        base = json.loads(Path('config/experiments/dev200-prompt-v4-control.json').read_text())
        base.pop('evaluation_set')
        base.update(evaluation_set_id=build['evaluation_set_id'], method='semantic',
                    research_question='固定语义检索器，比较 v1/v3 指令与普通/对比示例的独立作用和交互效应；新的 dev200',
                    comparison_group='goemotions-prompt-factorial-dev200-20260909')
        for arm, (version, policy) in ARMS.items():
            cfg = {**base, 'name': f'GoEmotions prompt factorial {arm}', 'prompt_version': version, 'example_policy': policy}
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
    validate_factorial(configs)
    state_path = ROOT / 'prompt_factorial_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state:
        assert state['config_hash'] == digest(configs)
    else:
        state = {'started_at': now(), 'config_hash': digest(configs), 'build': build,
                 'run_order': random.Random(90909).sample(list(ARMS), len(ARMS)), 'runs': {}, 'evaluations': {}, 'comparisons': {}}
    def save():
        pending = state_path.with_suffix('.pending')
        pending.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        pending.replace(state_path)
    with Store(ROOT) as store:
        if 'driver_artifact_id' not in state:
            with store.transaction():
                aid = store.put({'source': Path(__file__).read_text(), 'configs': configs, 'run_order': state['run_order'],
                                 'primary_comparisons': ['B-A', 'D-C'], 'secondary_comparisons': ['C-A', 'D-B', 'D-A'],
                                 'interaction': '(D-C)-(B-A); paired common bootstrap seed=42,repeats=1000',
                                 'scope': 'new fixed dev200; all four arms finish before scoring; existing v1/v3; semantic index fixed; test unused',
                                 'authorization': '用户批准继续执行原指令/优化指令与普通/对比示例的四组准确性实验，沿用已授权真实模型和存证方式。'},
                                'prompt_factorial_driver', inline=False)
                store.event('artifact_id', aid, 'prompt_factorial_authorized')
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
            output = ROOT / 'exports' / f'prompt-factorial-dev200-{arm}-20260909'
            if not output.exists():
                export_evaluation(store, state['evaluations'][arm], output)
        for name, (baseline, candidate) in COMPARISONS.items():
            if name not in state['comparisons']:
                state['comparisons'][name] = compare(store, state['evaluations'][baseline], state['evaluations'][candidate], seed=42, repeats=1000)
                save()
        state['completed_at'] = now(); save()
        with store.transaction():
            aid = store.put(state, 'prompt_factorial_execution_manifest', inline=False)
            store.event('artifact_id', aid, 'prompt_factorial_executed')
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
