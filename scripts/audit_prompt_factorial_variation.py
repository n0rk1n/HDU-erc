"""Reproduce the posthoc same-request diagnostic without additional model calls."""
import json
from pathlib import Path

from emotion_lab.storage import Store, now


def main():
    root = Path('data/research')
    analysis = json.loads((root / 'prompt_factorial_analysis.json').read_text())
    output = root / 'prompt_factorial_request_variation.json'
    with Store(root) as store:
        cases = store.json(analysis['paired_details_artifact_id'])['records']
        counts, details = {}, []
        for name, left, right in [('C-A', 'A', 'C'), ('D-B', 'B', 'D')]:
            counts[name] = {}
            for row in cases:
                hashes = [store.one('SELECT request_sha256 FROM call_attempts WHERE call_attempt_id=?',
                                    (row[arm]['call_attempt_ids'][0],))['request_sha256'] for arm in [left, right]]
                same = hashes[0] == hashes[1]
                group = 'same_request' if same else 'changed_request'
                item = counts[name].setdefault(group, {'samples': 0, 'different_predictions': 0,
                                                       'baseline_exact': 0, 'candidate_exact': 0})
                truth = set(row['truth'])
                before, after = set(row[left]['predicted_labels']), set(row[right]['predicted_labels'])
                item['samples'] += 1
                item['different_predictions'] += before != after
                item['baseline_exact'] += before == truth
                item['candidate_exact'] += after == truth
                details.append({'comparison': name, 'sample_id': row['sample_id'], 'same_request': same,
                                'request_hashes': hashes, 'different_predictions': before != after,
                                'baseline_exact': before == truth, 'candidate_exact': after == truth})
        if output.exists():
            saved = json.loads(output.read_text())
            assert saved['analysis_artifact_id'] == analysis['analysis_artifact_id']
            assert saved['counts'] == counts and saved['records'] == details
            artifact = store.json(saved['artifact_id'])
            assert artifact['counts'] == counts and artifact['records'] == details
            print(json.dumps({'existing_diagnostic_verified': True, 'records': len(details), 'counts': counts}, ensure_ascii=False))
            return
        result = {'created_at': now(), 'scope': 'posthoc request-repeat variability diagnostic; no additional model calls',
                  'analysis_artifact_id': analysis['analysis_artifact_id'], 'counts': counts, 'records': details,
                  'source': Path(__file__).read_text()}
        with store.transaction():
            aid = store.put(result, 'factorial_request_variation', inline=False)
            store.event('comparison_id', analysis['factorial_comparison_id'], 'request_variation_audited', payload=aid)
        result['artifact_id'] = aid
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'artifact_id': aid, 'counts': counts}, ensure_ascii=False))


if __name__ == '__main__':
    main()
