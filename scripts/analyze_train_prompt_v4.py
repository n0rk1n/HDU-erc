"""Independently reconcile real requests/predictions with original TSV files."""
import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from emotion_lab.storage import Store,digest,now
from emotion_lab.datasets import normalize
from emotion_lab.retrieval import CONTRAST_PAIRS

root=Path('data/research')
state=json.loads((root/'train_prompt_v4_state.json').read_text())
assert state.get('comparison_id'),'wait for both model runs and frozen evaluations'
labels=(Path('data/benchmarks/goemotions/emotions.txt').read_text().splitlines())

def source(split):
    with Path(f'data/benchmarks/goemotions/{split}.tsv').open() as f:
        return {r[2]:{'text':r[0],'labels':{labels[int(x)] for x in r[1].split(',')}} for r in csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE)}

train=source('train'); dev=source('dev')

def scores(records,field):
    counts={name:[0,0,0] for name in labels}
    exact=0
    for item in records:
        true=set(item['truth']); pred=set(item[field]['predicted_labels'])
        exact += true==pred
        for name in true|pred:
            counts[name][0]+=int(name in true and name in pred)
            counts[name][1]+=int(name in pred and name not in true)
            counts[name][2]+=int(name in true and name not in pred)
    per={name:{'tp':tp,'fp':fp,'fn':fn,'support':tp+fn,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.} for name,(tp,fp,fn) in counts.items()}
    tp=sum(x[0] for x in counts.values()); fp=sum(x[1] for x in counts.values()); fn=sum(x[2] for x in counts.values())
    return {'n':len(records),'micro_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'macro_f1':sum(p['f1'] for p in per.values())/28,'exact_match':exact/len(records) if records else 0.,'exact_count':exact,'tp':tp,'fp':fp,'fn':fn,'per_label':per}

with Store(root) as store:
    by_source={}
    run_info={}
    audits={'requests_verified':0,'native_truths_verified':0,'selected_train_examples_verified':0,'raw_predictions_verified':0,'all_gold_from_original_tsv':True,'test_split_used':False}
    for arm in ('control','candidate'):
        run_id=state[arm+'_run_id']; eid=state[arm+'_evaluation_id']
        run=store.one('SELECT * FROM experiment_runs WHERE run_id=?',(run_id,))
        assert run['status'] in ('completed','completed_with_errors')
        cfg=store.json(run['method_config_artifact_id'])
        items=store.rows('SELECT i.*,s.source_id,s.raw_text,s.normalized_text_sha256 FROM run_items i JOIN samples s USING(sample_id) WHERE run_id=? ORDER BY i.ordinal',(run_id,))
        assert len(items)==200
        for item in items:
            gold=dev[item['source_id']]
            assert item['raw_text']==gold['text']
            dbtruth={r['label_name'] for r in store.rows('SELECT d.label_name FROM sample_labels l JOIN label_definitions d ON d.dataset_version_id=l.dataset_version_id AND d.label_id=l.label_id WHERE l.sample_id=?',(item['sample_id'],))}
            assert dbtruth==gold['labels']
            audits['native_truths_verified']+=1
            entry=by_source.setdefault(item['source_id'],{'source_id':item['source_id'],'sample_id':item['sample_id'],'text':gold['text'],'truth':sorted(gold['labels'])})
            attempts=store.rows('SELECT a.* FROM call_attempts a JOIN execution_steps s USING(step_id) WHERE s.run_item_id=? ORDER BY a.attempt_no',(item['run_item_id'],))
            assert attempts
            assert len({a['request_sha256'] for a in attempts})==1
            for attempt in attempts:
                assert digest(store.read(attempt['request_artifact_id']))==attempt['request_sha256']
            request=store.json(attempts[0]['request_artifact_id'])
            assert request['messages'][-1]=={'role':'user','content':gold['text']}
            assert request['model']=='ZHIPU/GLM-5.3' and request['enable_thinking'] is True and request['reasoning_effort']=='low'
            assert request['temperature']==0 and request['max_tokens']==8192
            audits['requests_verified']+=1
            selected=store.rows("SELECT r.*,s.source_id,s.raw_text,s.normalized_text_sha256,s.split FROM retrieval_items r JOIN samples s USING(sample_id) JOIN execution_steps step ON step.step_id=r.retrieval_step_id WHERE step.run_item_id=? AND r.decision='selected' ORDER BY r.selected_rank",(item['run_item_id'],))
            assert len(selected)==4
            prompt_examples=json.loads(request['messages'][0]['content'].split('\nTraining examples:\n')[1])
            example_records=[]
            for r,ex in zip(selected,prompt_examples):
                original=train[r['source_id']]
                assert r['split']=='train' and ex['text']==original['text']==r['raw_text']
                assert set(ex['labels'])==original['labels']
                assert normalize(ex['text'])!=normalize(gold['text'])
                example_records.append({'source_id':r['source_id'],'sample_id':r['sample_id'],'text':ex['text'],'labels':ex['labels'],'score':r['similarity_score'],'details':json.loads(r['score_components_json'])})
                audits['selected_train_examples_verified']+=1
            assert len({normalize(ex['text']) for ex in prompt_examples})==4
            pred_labels=[]
            if item['final_prediction_id']:
                prediction=store.one('SELECT * FROM predictions WHERE prediction_id=?',(item['final_prediction_id'],))
                a=next(a for a in attempts if a['call_attempt_id']==prediction['call_attempt_id'])
                assert a['http_status']==200
                assert a['input_tokens']+cfg['model']['max_tokens']+cfg['model']['safety_tokens']<=cfg['model']['context_tokens']
                raw=store.json(a['response_artifact_id'])
                parsed=json.loads(raw['choices'][0]['message']['content'])
                saved={r['label_name'] for r in store.rows('SELECT d.label_name FROM prediction_labels l JOIN label_definitions d ON d.dataset_version_id=l.dataset_version_id AND d.label_id=l.label_id WHERE l.prediction_id=?',(item['final_prediction_id'],))}
                assert saved==set(parsed['labels']) and saved<=set(labels) and saved
                pred_labels=sorted(saved)
                for source_key,field in [('prompt_tokens','input_tokens'),('completion_tokens','output_tokens'),('total_tokens','total_tokens')]:
                    assert raw['usage'][source_key]==a[field]
                audits['raw_predictions_verified']+=1
            entry[arm]={'predicted_labels':pred_labels,'status':item['status'],'run_item_id':item['run_item_id'],'prediction_id':item['final_prediction_id'],'selected_examples':example_records,'call_attempt_ids':[a['call_attempt_id'] for a in attempts]}
        calls=store.rows('SELECT * FROM v_call_audit WHERE run_id=? ORDER BY prepared_at',(run_id,))
        def total(key):
            return sum(a[key] for a in calls if a[key] is not None)
        elapsed=(datetime.fromisoformat(run['completed_at'].replace('Z','+00:00'))-datetime.fromisoformat(run['started_at'].replace('Z','+00:00'))).total_seconds()
        run_info[arm]={'run_id':run_id,'evaluation_id':eid,'config':cfg,'code_commit':run['code_commit'],'status':run['status'],'succeeded':sum(i['status']=='succeeded' for i in items),'failed':sum(i['status']=='failed' for i in items),'calls':len(calls),'retries':sum(a['attempt_no']>1 for a in calls),'http_statuses':dict(Counter(str(a['http_status']) for a in calls)),'resolved_models':dict(Counter(a['resolved_model'] for a in calls)),'input_tokens':total('input_tokens'),'output_tokens':total('output_tokens'),'total_tokens':total('total_tokens'),'cached_input_tokens':total('cached_input_tokens'),'reasoning_tokens':total('reasoning_tokens'),'reasoning_responses':sum(a['reasoning_artifact_id'] is not None for a in calls),'unknown_cost_calls':sum(a['estimated_cost_micros'] is None for a in calls),'mean_call_latency_ms':statistics.mean(a['latency_ms'] for a in calls if a['latency_ms'] is not None),'run_wall_seconds':elapsed}
    records=sorted(by_source.values(),key=lambda r:r['source_id'])
    assert len(records)==200 and all('control' in r and 'candidate' in r for r in records)
    # Only the intended method switches and human-readable arm name may differ.
    ca={k:v for k,v in run_info['control']['config'].items() if k not in ('name','prompt_version')}
    cb={k:v for k,v in run_info['candidate']['config'].items() if k not in ('name','prompt_version')}
    assert ca==cb
    assert all(r['control']['selected_examples'] == r['candidate']['selected_examples'] for r in records), 'prompt-only comparison changed actual examples'
    audits['paired_examples_identical'] = len(records)
    # Verify snapshots against actual messages, including all retries.
    paired_requests = {}
    for arm in ('control','candidate'):
        run = store.one('SELECT * FROM experiment_runs WHERE run_id=?',(state[arm+'_run_id'],))
        frozen_instruction = store.json(run['prompt_config_artifact_id'])['instruction']
        for r in records:
            for aid in r[arm]['call_attempt_ids']:
                attempt = store.one('SELECT request_artifact_id FROM call_attempts WHERE call_attempt_id=?',(aid,))
                req = store.json(attempt['request_artifact_id'])
                instruction, rest = req['messages'][0]['content'].split('\nLabel definitions:',1)
                assert instruction == frozen_instruction
                req['messages'][0]['content'] = rest
                if r['source_id'] in paired_requests:
                    assert paired_requests[r['source_id']] == req
                else:
                    paired_requests[r['source_id']] = req
    audits['paired_requests_only_instruction_differs'] = len(paired_requests)
    metrics={arm:scores(records,arm) for arm in ('control','candidate')}
    for arm in metrics:
        persisted={r['metric_name']:r['value'] for r in store.rows("SELECT metric_name,value FROM metric_values WHERE evaluation_id=? AND scope_key='overall'",(state[arm+'_evaluation_id'],))}
        for name in ('micro_f1','macro_f1','exact_match'):
            assert math.isclose(persisted[name],metrics[arm][name],abs_tol=1e-12)
    comparison=store.one('SELECT * FROM comparisons WHERE comparison_id=?',(state['comparison_id'],))
    bootstrap=store.json(comparison['result_artifact_id'])
    assert math.isclose(bootstrap['micro_f1_delta'],metrics['candidate']['micro_f1']-metrics['control']['micro_f1'],abs_tol=1e-12)
    slices={}
    for name,selected in {
        'single_label':[r for r in records if len(r['truth'])==1],
        'multi_label':[r for r in records if len(r['truth'])>1],
        'neutral_cooccurrence':[r for r in records if 'neutral' in r['truth'] and len(r['truth'])>1],
        'contrast_found':[r for r in records if any(e['details']['selection_role']=='contrast' for e in r['candidate']['selected_examples'])],
        'examples_membership_changed':[r for r in records if {e['source_id'] for e in r['control']['selected_examples']}!={e['source_id'] for e in r['candidate']['selected_examples']}],
    }.items():
        slices[name]={arm:{k:v for k,v in scores(selected,arm).items() if k!='per_label'} for arm in ('control','candidate')}
    pairs=[]
    for a,b in CONTRAST_PAIRS:
        item={'pair':[a,b]}
        for arm in ('control','candidate'):
            item[arm]=sum((a in r['truth'] and b not in r['truth'] and b in r[arm]['predicted_labels'] and a not in r[arm]['predicted_labels']) or (b in r['truth'] and a not in r['truth'] and a in r[arm]['predicted_labels'] and b not in r[arm]['predicted_labels']) for r in records)
        pairs.append(item)
    fixed=[r for r in records if set(r['truth'])!=set(r['control']['predicted_labels']) and set(r['truth'])==set(r['candidate']['predicted_labels'])]
    regressed=[r for r in records if set(r['truth'])==set(r['control']['predicted_labels']) and set(r['truth'])!=set(r['candidate']['predicted_labels'])]
    summary={'created_at':now(),'state':state,'runs':run_info,'metrics':metrics,'slices':slices,'bootstrap':{k:v for k,v in bootstrap.items() if k!='bootstrap_deltas'},'pair_substitution_errors':pairs,'independent_validation':audits,'exact_fixed':len(fixed),'exact_regressed':len(regressed),'zero_support_labels':[l for l in labels if metrics['control']['per_label'][l]['support']==0],'example_cases':{'fixed':[{'source_id':r['source_id'],'text':r['text'],'truth':r['truth'],'control':r['control']['predicted_labels'],'candidate':r['candidate']['predicted_labels']} for r in fixed[:5]],'regressed':[{'source_id':r['source_id'],'text':r['text'],'truth':r['truth'],'control':r['control']['predicted_labels'],'candidate':r['candidate']['predicted_labels']} for r in regressed[:5]]}}
    family = store.json(store.one('SELECT * FROM experiment_runs WHERE run_id=?',(state['control_run_id'],))['taxonomy_config_artifact_id'])['families']
    supplements = {}
    for arm in ('control','candidate'):
        sets = [(set(r['truth']),set(r[arm]['predicted_labels'])) for r in records]
        fs = [({family[x] for x in t},{family[x] for x in p}) for t,p in sets]
        ftp=sum(len(t&p) for t,p in fs); ffp=sum(len(p-t) for t,p in fs); ffn=sum(len(t-p) for t,p in fs)
        supplements[arm] = {
            'native_any_hit':sum(bool(t&p) for t,p in sets)/len(sets),
            'family_any_hit':sum(bool(t&p) for t,p in fs)/len(fs),
            'family_exact_match':sum(t==p for t,p in fs)/len(fs),
            'family_micro_f1':2*ftp/(2*ftp+ffp+ffn) if 2*ftp+ffp+ffn else 0.,
            'mean_predicted_labels':sum(len(p) for t,p in sets)/len(sets),
            'neutral_only_support':sum(t=={'neutral'} for t,p in sets),
            'neutral_only_missing':sum(t=={'neutral'} and 'neutral' not in p for t,p in sets),
            'native_no_overlap':sum(not(t&p) for t,p in sets),
            'native_partial_overlap':sum(bool(t&p) and t!=p for t,p in sets),
        }
    summary['supplemental_metrics'] = supplements
    summary['development_basis_artifact_id'] = state['development_basis_artifact_id']
    summary['scope'] = 'Development iteration on previously inspected dev200; only v4 frozen candidate; untouched official test not used.'
    summary['corrected_training_audit_artifact_id'] = 'df0b445f-fe1b-4bea-bcb1-961f99778c41'
    with store.transaction():
        details_id=store.put({'records':records,'analysis_source':Path(__file__).read_text()},'train_prompt_v4_paired_case_details',inline=False)
        summary['paired_details_artifact_id']=details_id
        summary_id=store.put(summary,'train_prompt_v4_analysis',inline=False)
        store.event('comparison_id',state['comparison_id'],'independently_validated',payload=summary_id)
    summary['analysis_artifact_id']=summary_id
    (root/'train_prompt_v4_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (root/'exports'/'glm-dev200-prompt-v4-paired-cases-20260908.jsonl').open('w') as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k in ('bootstrap','independent_validation','exact_fixed','exact_regressed','zero_support_labels','analysis_artifact_id')},ensure_ascii=False,indent=2))
    for arm in ('control','candidate'):
        print(arm,json.dumps({k:v for k,v in metrics[arm].items() if k!='per_label'},ensure_ascii=False))
