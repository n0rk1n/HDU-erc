"""Render a balanced v4 report from verified artifacts, including repeat variation."""
import json
from pathlib import Path
from emotion_lab.storage import Store, now

root=Path('data/research')
a=json.loads((root/'train_prompt_v4_analysis.json').read_text())
basis=json.loads((root/'prompt_v4_development_basis.json').read_text())
prior=json.loads((root/'train_prompt_v3_analysis.json').read_text())
s=a['state'];m=a['metrics'];u=a['runs'];b=a['bootstrap'];extra=a['supplemental_metrics']
pct=lambda v:f'{v*100:.2f}%'
pp=lambda v:f'{v*100:+.2f}'
with Store(root) as store:
 def requests(rid):
  return {r['sample_id']:r['request_sha256'] for r in store.rows('SELECT i.sample_id,c.request_sha256 FROM run_items i JOIN execution_steps e USING(run_item_id) JOIN call_attempts c USING(step_id) WHERE i.run_id=? AND c.attempt_no=1',(rid,))}
 old_requests=requests(prior['state']['candidate_run_id']);new_requests=requests(s['control_run_id'])
 assert len(old_requests)==200 and old_requests==new_requests
 repeat={'previous_v3_run_id':prior['state']['candidate_run_id'],'current_v3_run_id':s['control_run_id'],'identical_request_hashes':200,'previous_micro_f1':prior['metrics']['candidate']['micro_f1'],'current_micro_f1':m['control']['micro_f1'],'scope':'same request bodies; no verified explanation of provider variability'}
 with store.transaction():
  repeat_id=store.put(repeat,'prompt_v4_control_repeat_variation')
  store.event('comparison_id',s['comparison_id'],'control_repeat_checked',payload=repeat_id)

lines=[
 '# 历史问题复盘后的 Prompt v4 对照结果','',
 f'生成时间（UTC）：{now()}。实际运行代码：`{u["control"]["code_commit"]}` / `{u["candidate"]["code_commit"]}`。','',
 '## 执行范围','',
 '固定同一 dev200，重新运行 v3 控制组和 v4 候选组，各 200 条。模型、标签定义、families、contrastive-v1 检索、候选 50、入选 4 条示例、解析器和原生多标签评分不变。两组全部完成后才评分；本轮只冻结一个 v4，不按当前输出再调参。','',
 '历史错误已用于本轮开发，不能称为独立未见验证。新增规则同时参考 train 的原始边界样本；prompt 不包含 dev 原文/真值对，官方 test 未使用。','',
 '## 主指标','',
 '| 指标 | 本轮 v3 重跑 | v4 | 差值（百分点） |','|---|---:|---:|---:|',
]
for title,key in [('Micro-F1','micro_f1'),('Macro-F1（28 类）','macro_f1'),('标签集合完全匹配率','exact_match')]:
 lines.append(f'| {title} | {pct(m["control"][key])} | {pct(m["candidate"][key])} | {pp(m["candidate"][key]-m["control"][key])} |')
for title,kind in [('Micro-precision','precision'),('Micro-recall','recall')]:
 vals={arm:m[arm]['tp']/(m[arm]['tp']+m[arm]['fp' if kind=='precision' else 'fn']) for arm in m}
 lines.append(f'| {title} | {pct(vals["control"])} | {pct(vals["candidate"])} | {pp(vals["candidate"]-vals["control"])} |')
lines += ['',f'Micro-F1 差值的 95% 配对 bootstrap 区间：[{pp(b["ci_low"])}, {pp(b["ci_high"])}] 个百分点。200 对样本，seed=42，1000 次重采样。','']
if b['micro_f1_delta']>0:
 lines.append('本轮 v4 的 Micro-F1 上升。'+('区间包含 0，不能确认改进稳定存在。' if b['ci_low']<=0 else '配对区间高于 0，但该区间不包含重复开发集选择、服务变异和多次独立运行的不确定性，仍需独立验证。'))
else:
 lines.append('本轮 v4 未提高主指标 Micro-F1；保留负面结果，不将新增规则或测试通过当作模型改进。')
lines += ['', '本轮候选不提升为默认版本。realization、caring 和 disappointment 的召回局部回升，但总体误报增幅更大；这只能解释整个 v4 组合的观测效果，不能分离每条提示规则的因果作用。决策工件：`df39aaf3-9235-4922-b79f-dee193d2f121`。', '']
lines += ['',f'标签集合不完全匹配→完全匹配 {a["exact_fixed"]} 条，完全匹配→不完全匹配 {a["exact_regressed"]} 条。',
 f'标签级计数：v3 TP/FP/FN = {m["control"]["tp"]}/{m["control"]["fp"]}/{m["control"]["fn"]}；v4 = {m["candidate"]["tp"]}/{m["candidate"]["fp"]}/{m["candidate"]["fn"]}。各指标共同解释结果，不只看其中上升的一项。','',
 '## 前轮问题与本轮改动','',
 '完整复盘见 [低分问题总复盘](low-accuracy-review-and-prompt-v4.md)，包括实际误报/漏报、检索局限、评价口径、类别不均衡、运行波动、开发集重复使用，以及已排除的接口/存储/引号读取因素。','',
 '- 撤掉 v3 的训练频率和最小标签集合倾向，同时检查多报与漏报。',
 '- 补足 disappointment、realization、approval、caring 的隐含或普通表达，不要求显式情绪词或独立句子。',
 '- 保留对 disapproval、neutral、幽默/讽刺误读的限制，但不把所有负面反应或不确定情绪归为某个兜底标签。',
 '- 先扫描输入，再参考例子；实际检索与入选例子保持一致，尚未证明检索问题已解决。','',
 '## 历史结果与运行变化','',
 '| 历史组 | Micro-F1 | Macro-F1 | 完全匹配率 |','|---|---:|---:|---:|',
]
for name,v in basis['historical_metrics'].items():
 lines.append(f'| {name} | {pct(v["micro_f1"])} | {pct(v["macro_f1"])} | {pct(v["exact_match"])} |')
lines += ['',f'前轮 v3 与本轮 v3 的 200 个请求体哈希全部相同；Micro-F1 为 {pct(repeat["previous_micro_f1"])} 与 {pct(repeat["current_micro_f1"])}。同参请求不保证确定性输出，不臆测供应商内部变化原因。变异核验工件：`{repeat_id}`。',
 '前轮两次 v2 也有 77/200 条标签集合不同。本轮以重新运行的 v3 为控制组，历史各轮全部保留，不挑选最有利的历史基线。','',
 f'离线常量 neutral 参照：完全匹配率 {pct(basis["always_neutral_reference"]["exact_match"])}，Micro-F1 {pct(basis["always_neutral_reference"]["micro_f1"])}。它不调用模型，不是候选方法；用于说明只看完全匹配率容易被类别比例误导。','',
 '## 补充指标与错误分布','',
 '| 指标 | v3 | v4 |','|---|---:|---:|',
]
for title,key in [('至少一个细类命中','native_any_hit'),('大类集合完全匹配','family_exact_match'),('至少一个大类命中','family_any_hit'),('大类 Micro-F1','family_micro_f1')]:
 lines.append(f'| {title} | {pct(extra["control"][key])} | {pct(extra["candidate"][key])} |')
for title,key in [('平均预测标签数','mean_predicted_labels'),('60 条仅 neutral 中漏判条数','neutral_only_missing'),('细类无交集条数','native_no_overlap')]:
 lines.append(f'| {title} | {extra["control"][key]} | {extra["candidate"][key]} |')
lines += ['', '| 标签 | 真值数 | v3 TP/FP/FN | v4 TP/FP/FN | v3 F1 | v4 F1 |','|---|---:|---|---|---:|---:|']
for label,l in m['control']['per_label'].items():
 r=m['candidate']['per_label'][label]
 lines.append(f'| {label} | {l["support"]} | {l["tp"]}/{l["fp"]}/{l["fn"]} | {r["tp"]}/{r["fp"]}/{r["fn"]} | {pct(l["f1"])} | {pct(r["f1"])} |')
lines += ['', '无真值支持的类别：'+', '.join(a['zero_support_labels'])+'。Macro-F1 的零分母按 0；本切片无法稳定估计稀有类。','',
 '## 调用代价','', '| 项目 | v3 | v4 |','|---|---:|---:|']
for title,key in [('成功样本','succeeded'),('失败样本','failed'),('调用数（含重试）','calls'),('重试','retries'),('输入 tokens','input_tokens'),('输出 tokens','output_tokens'),('reasoning tokens','reasoning_tokens'),('缓存输入 tokens','cached_input_tokens'),('运行秒数（不含冻结）','run_wall_seconds'),('费用未知调用数','unknown_cost_calls')]:
 lines.append(f'| {title} | {u["control"][key]} | {u["candidate"][key]} |')
lines += ['',f'输入 tokens 相对变化：{(u["candidate"]["input_tokens"]/u["control"]["input_tokens"]-1)*100:+.2f}%。未有核实的部署单价，费用为 NULL，不当作零。',
 '固定 GLM-5.3、temperature=0、thinking enabled、reasoning_effort low、max_tokens=8192、context_tokens=32768、并发4。服务未提供可核实的权重修订号。实际返回模型：'+json.dumps({k:v['resolved_models'] for k,v in u.items()},ensure_ascii=False)+'。两组顺序执行，缓存/耗时不可直接归因为算法收益。','',
 '## 案例','']
for key,title in [('fixed','由错变对'),('regressed','由对变错')]:
 lines += [f'### {title}','','| source_id | 原文 | 真值 | v3 | v4 |','|---|---|---|---|---|']
 for r in a['example_cases'][key]:
  vals=[r['source_id'],r['text'],', '.join(r['truth']),', '.join(r['control']),', '.join(r['candidate'])]
  lines.append('| '+' | '.join(v.replace('|','\\|').replace('\n',' ') for v in vals)+' |')
 lines.append('')
lines += ['## 数据与核验','',
 '- SQLite：`data/research/experiments.sqlite3`；完整请求/响应/检索轨迹在 `data/research/artifacts/`。',
 '- 成对案例：`data/research/exports/glm-dev200-prompt-v4-paired-cases-20260908.jsonl`。',
 '- 标准导出：`data/research/exports/glm-dev200-prompt-v4-control-20260908/` 和 `glm-dev200-prompt-v4-candidate-20260908/`。',
 '- 完整分析：`data/research/train_prompt_v4_analysis.json`；执行状态/日志：`train_prompt_v4_state.json` / `train_prompt_v4_execution.log`。',
 '- 可恢复备份：`data/research/backups/train-prompt-v4-final-20260908/`；完整性验证另存 verification_report 工件。',
 '', '独立核验：`'+json.dumps(a['independent_validation'],ensure_ascii=False)+'`。200 对实际请求除 instruction 外相同；200 对入选示例的 ID、顺序、全文、标签和检索分数一致。','',
 '| 记录 | ID |','|---|---|']
for key in ['control_run_id','candidate_run_id','control_evaluation_id','candidate_evaluation_id','comparison_id']:
 lines.append(f'| {key} | `{s[key]}` |')
for key in ['analysis_artifact_id','paired_details_artifact_id','development_basis_artifact_id']:
 lines.append(f'| {key} | `{a[key]}` |')
text='\n'.join(lines)+'\n'
target=Path('docs/research/glm-dev200-prompt-v4-results-20260908.md')
target.write_text(text)
with Store(root) as store, store.transaction():
 aid=store.put({'report':text,'generator_source':Path(__file__).read_text(),'analysis_artifact_id':a['analysis_artifact_id'],'repeat_variation_artifact_id':repeat_id},'prompt_v4_result_report',inline=False)
 store.event('comparison_id',s['comparison_id'],'result_reported',payload=aid)
print(json.dumps({'report':str(target),'artifact_id':aid}))
