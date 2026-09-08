"""Write a report from verified run artifacts; never invent model improvements."""
import json
from pathlib import Path
from emotion_lab.storage import Store, now

root=Path('data/research')
a=json.loads((root/'train_prompt_v3_analysis.json').read_text())
s=a['state']; m=a['metrics']; u=a['runs']; b=a['bootstrap']; extra=a['supplemental_metrics']
pct=lambda v:f'{v*100:.2f}%'
pp=lambda v:f'{v*100:+.2f}'
lines=[
 '# GLM-5.3：训练集驱动 Prompt v3 的对照验证', '',
 f'报告生成时间（UTC）：{now()}。两组运行代码提交：`{u["control"]["code_commit"]}` / `{u["candidate"]["code_commit"]}`。', '',
 '## 验证范围', '',
 '基于 43,410 条 GoEmotions 训练评论的类别、共现与词形切片统计及原始样本核验，冻结一个 v3 候选；重新运行 v2 控制组与 v3 候选组，各 200 条。两组固定模型参数、数据、taxonomy、contrastive-v1 检索、4 条入选示例及原生评分规则，只改变 prompt。', '',
 '**这轮是开发迭代。** 前轮 dev200 错误已被观察并用于提出优化问题；本轮规则依据来自 train，但不能把重复使用的 dev200 视为独立未见验证。官方 test 未使用。没有按本轮得分继续调参或挑选多个候选。', '',
 '## 主指标', '',
 '| 指标 | 本轮 v2 控制组 | v3 候选组 | 差值（百分点） |',
 '|---|---:|---:|---:|',
]
for title,key in [('细类 Micro-F1','micro_f1'),('细类 Macro-F1（28 类）','macro_f1'),('精确标签集合匹配率','exact_match')]:
 lines.append(f'| {title} | {pct(m["control"][key])} | {pct(m["candidate"][key])} | {pp(m["candidate"][key]-m["control"][key])} |')
lines += ['',f'Micro-F1 配对 bootstrap 差值的 95% 区间：[{pp(b["ci_low"])}, {pp(b["ci_high"])}] 个百分点，1000 次重采样，seed=42。', '']
if b['micro_f1_delta']>0:
 lines += ['v3 在本开发切片的 Micro-F1 高于本轮 v2。' + ('区间包含 0，尚不能确认改进稳定存在。' if b['ci_low']<=0 else '配对区间高于 0，但不包含开发集反复使用、服务漂移及独立运行不确定性，不能替代独立测试。')]
else:
 lines += ['v3 未提高本轮开发切片的 Micro-F1；保留该结果，不把训练依据或软件测试通过当作模型效果改善。']
lines += ['', '新版主要减少了多报：误报标签由 189 降到 148，但命中标签由 99 降到 93、漏报由 133 增到 139，至少一个细类命中率由 47% 降到 44%。因此不能将它认定为整体更好的版本，也不自动替换默认 prompt。', '', '额外复核发现：前轮 v2 与本轮 v2 的 200 个原始请求体哈希全部相同，但 Micro-F1 为 40.62% 与 38.08%。这说明固定请求参数并不保证供应商输出完全一致；本轮 v3 的 39.32% 也低于前轮 v2 的 40.62%。本轮比较采用重新执行的控制组，不能只挑选最有利的历史基线。变异证据工件：`8aeee025-c736-4ed6-a53f-01a1a344b628`。', '']
lines += ['', f'标签集合不完全匹配→完全匹配 {a["exact_fixed"]} 条，完全正确→错误 {a["exact_regressed"]} 条。', '',
 '## 修改依据与规则', '',
 '- 36,308/43,410 条训练评论为单标签：区分多个备选解释与真正的情绪共现，减少同义或强弱标签堆叠；保留多标签与 neutral 共现。',
 '- 负面词切片有 318 条仅 neutral，脏话切片有 250 条仅 neutral：按实际表达和态度判断，避免将 disapproval 用作负面文本兜底。',
 '- 问号切片同时含 curiosity、confusion 和 neutral：允许信息询问表达好奇，区分理解困难和反问。',
 '- 训练原始标注中存在同情性 sorry → remorse：不要求每次都承认自身过错。',
 '- 明确检索例子不是候选标签范围，也不凭词面相似复制情绪；实际检索算法与例子全文不变。',
 '', '训练依据、source_id 和方法局限见 [训练审计说明](train-prompt-v3.md)。完整 prompt 在 `emotion_lab/taxonomy.py`，实际使用的版本和请求均入库。', '',
 '## 补充诊断（不替代主指标）', '',
 '| 指标 | v2 | v3 |', '|---|---:|---:|']
for title,key in [('细类至少一个命中','native_any_hit'),('大类集合完全匹配','family_exact_match'),('大类至少一个命中','family_any_hit'),('大类 Micro-F1','family_micro_f1')]:
 lines.append(f'| {title} | {pct(extra["control"][key])} | {pct(extra["candidate"][key])} |')
lines += ['', '| 诊断量 | v2 | v3 |', '|---|---:|---:|']
for title,key in [('每条平均预测标签数','mean_predicted_labels'),('60 条仅 neutral 中漏掉 neutral 的条数','neutral_only_missing'),('细类完全无交集条数','native_no_overlap'),('细类部分命中条数','native_partial_overlap')]:
 lines.append(f'| {title} | {extra["control"][key]} | {extra["candidate"][key]} |')
for label in ['disapproval','annoyance','neutral']:
 lines.append(f'| {label} 误报数 | {m["control"]["per_label"][label]["fp"]} | {m["candidate"]["per_label"][label]["fp"]} |')
lines += ['', '## 调用与代价', '', '| 项目 | v2 | v3 |', '|---|---:|---:|']
for title,key in [('成功样本数','succeeded'),('失败样本数','failed'),('调用数（含重试）','calls'),('重试数','retries'),('输入 tokens','input_tokens'),('输出 tokens','output_tokens'),('总 tokens','total_tokens'),('缓存输入 tokens','cached_input_tokens'),('reasoning tokens','reasoning_tokens'),('费用未知调用数','unknown_cost_calls'),('运行时间（秒，不含准备）','run_wall_seconds')]:
 lines.append(f'| {title} | {u["control"][key]} | {u["candidate"][key]} |')
lines += ['',f'输入 tokens 相对变化：{(u["candidate"]["input_tokens"]/u["control"]["input_tokens"]-1)*100:+.2f}%。费用未有已核实部署单价，保留 NULL。', '',
 '两组温度 0、thinking enabled、reasoning_effort low、max_tokens 8192、context_tokens 32768、并发 4、每条最多 2 次尝试。实际响应模型为 ' + json.dumps({k:v['resolved_models'] for k,v in u.items()},ensure_ascii=False) + '。供应商未提供可核实的权重修订号；两组顺序执行，缓存和耗时不可单独解释为算法收益。', '',
 '## 逐类结果', '', '| 标签 | 真值数 | v2 F1 | v3 F1 |', '|---|---:|---:|---:|']
for label,x in m['control']['per_label'].items():
 lines.append(f'| {label} | {x["support"]} | {pct(x["f1"])} | {pct(m["candidate"]["per_label"][label]["f1"])} |')
lines += ['', '零支持类别：' + ', '.join(a['zero_support_labels']) + '。Macro-F1 的零分母按 0，200 条小切片中的少数类估计不稳定。', '',
 '## 成对案例', '']
for name,title in [('fixed','由错变对'),('regressed','由对变错')]:
 lines += [f'### {title}', '', '| source_id | 原文 | 真值 | v2 | v3 |', '|---|---|---|---|---|']
 for r in a['example_cases'][name]:
  values=[r['source_id'],r['text'],', '.join(r['truth']),', '.join(r['control']),', '.join(r['candidate'])]
  lines.append('| '+' | '.join(v.replace('|','\\|').replace('\n',' ') for v in values)+' |')
 lines.append('')
lines += ['## 可复查证据', '',
 '- 独立核对原始 TSV 真值、实际模型请求、训练例子原文及标签、原始响应和数据库预测，并重算主指标。',
 '- 200 对请求除 instruction 外完全一致，200 对入选示例的 ID、顺序、原文、标签与检索分数相同。',
 '- 完整逐条对照：`data/research/exports/glm-dev200-prompt-v3-paired-cases-20260908.jsonl`。',
 '- 两组标准导出：`data/research/exports/glm-dev200-prompt-v3-control-20260908/` 与 `glm-dev200-prompt-v3-candidate-20260908/`。',
 '- SQLite：`data/research/experiments.sqlite3`；原始大工件：`data/research/artifacts/`。',
 '- 运行状态：`data/research/train_prompt_v3_state.json`；分析结果：`data/research/train_prompt_v3_analysis.json`。',
 '- 可恢复备份：`data/research/backups/train-prompt-v3-final-20260908/`。备份与软件检查的最终结果见独立 verification_report 工件。',
 '', '核验计数：`'+json.dumps(a['independent_validation'],ensure_ascii=False)+'`。', '',
 '| 对象 | ID |', '|---|---|']
for k in ['control_run_id','candidate_run_id','control_evaluation_id','candidate_evaluation_id','comparison_id']:
 lines.append(f'| {k} | `{s[k]}` |')
for k in ['analysis_artifact_id','paired_details_artifact_id','corrected_training_audit_artifact_id']:
 lines.append(f'| {k} | `{a[k]}` |')
text='\n'.join(lines)+'\n'
target=Path('docs/research/glm-dev200-prompt-v3-results-20260908.md')
target.write_text(text)
with Store(root) as store, store.transaction():
 aid=store.put({'report':text,'analysis_artifact_id':a['analysis_artifact_id'],'generator_source':Path(__file__).read_text()},'train_prompt_v3_result_report',inline=False)
 store.event('comparison_id',s['comparison_id'],'result_reported',payload=aid)
 for arm in ('control','candidate'):
  store.event('run_id',s[arm+'_run_id'],'corrected_training_audit_linked',payload=a['corrected_training_audit_artifact_id'])
print(json.dumps({'report':str(target),'report_artifact_id':aid}))
