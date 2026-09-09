"""Render all predeclared factorial results, including unfavorable evidence."""
import json
from pathlib import Path

from emotion_lab.storage import Store, now

ROOT = Path('data/research')


def main():
    analysis = json.loads((ROOT / 'prompt_factorial_analysis.json').read_text())
    state, metrics, runs = analysis['state'], analysis['metrics'], analysis['runs']
    with Store(ROOT) as store:
        cases = store.json(analysis['paired_details_artifact_id'])['records']
        query = store.json(state['build']['queries_artifact_id'])
    pct = lambda x: f'{100*x:.2f}%'
    pp = lambda x: f'{100*x:+.2f}'
    lines = ['# 提示词与对比示例四组准确性实验', '', f'生成时间（UTC）：{now()}。', '',
             '## 研究问题与固定条件', '',
             '检验此前的指令优化和对比示例选择是否分别有收益，以及二者是否存在正向交互。v1 是原指令；v3 是此前基于训练集优化的版本，名称“优化”不预设准确性更好。本轮不再改写指令。', '',
             '| 组 | 指令 | 示例选择 |', '|---|---|---|',
             '| A | native-labels-v1 | ranked：普通排序 |',
             '| B | native-labels-v3 | ranked：普通排序 |',
             '| C | native-labels-v1 | contrastive-v1：现有对比规则 |',
             '| D | native-labels-v3 | contrastive-v1：现有对比规则 |', '',
             '四组统一使用 BGE 语义检索、同一冻结 train 索引、候选 50、入选 4 条完整标签示例。GLM-5.3、temperature=0、thinking=enabled、reasoning_effort=low、输出上限 8192、上下文预算 32768、安全预留 256、并发 4 均固定。重试最多 2 次尝试/样本、400 次尝试/组；正式 test 未用于选择、调参或推理。', '',
             'v3 相比 v1 增加了 GoEmotions 标注惯例、neutral 判断、最小充分标签集合、易混情绪边界及使用示例的规则；标签定义、families、输出 JSON 解析方式完全相同。两版实际指令全文随每次请求留存。', '',
             f'新评估集 `{state["build"]["evaluation_set_id"]}`：seed=90909，从 dev 排除先前运行集合的同文本记录与 train 文本复制后，确定性抽取 200 条。四组都重新调用模型，先全部完成再评分；顺序为 {state["run_order"]}。不能与旧 dev200 的分数直接相减。', '',
             '## 四组主结果', '', '| 指标 | A | B | C | D |', '|---|---:|---:|---:|---:|']
    for title, key in [('Micro-F1', 'micro_f1'), ('Macro-F1（28 类）', 'macro_f1'), ('完全匹配率', 'exact_match'),
                       ('Precision', 'precision'), ('Recall', 'recall'), ('至少一个细类命中', 'any_hit'),
                       ('大类 Micro-F1', 'family_micro_f1'), ('大类完全匹配', 'family_exact_match'), ('至少一个大类命中', 'family_any_hit')]:
        lines.append('| ' + title + ' | ' + ' | '.join(pct(metrics[a][key]) for a in 'ABCD') + ' |')
    for title, key in [('TP', 'tp'), ('FP', 'fp'), ('FN', 'fn'), ('平均预测标签数', 'mean_predicted_labels')]:
        lines.append('| ' + title + ' | ' + ' | '.join(str(metrics[a][key]) for a in 'ABCD') + ' |')
    lines += ['', '## 指令作用、示例作用及组合变化', '',
              '| 比较 | 含义 | Micro-F1 差值（百分点） | 95% 区间 | 示例成员或顺序改变 | 错→全对 / 全对→错 |',
              '|---|---|---:|---|---:|---|']
    titles = {'B-A': '普通示例下的指令作用', 'D-C': '对比示例下的指令作用',
              'C-A': 'v1 下的示例作用', 'D-B': 'v3 下的示例作用', 'D-A': '组合相对原基线的变化'}
    for name, c in analysis['comparisons'].items():
        lines.append(f'| {name} | {titles[name]} | {pp(c["micro_f1_delta"])} | [{pp(c["ci_low"])}, {pp(c["ci_high"])}] | {c["changed_examples"]} | {c["exact_fixed"]} / {c["exact_regressed"]} |')
    interaction = analysis['factorial']['interaction']
    lines += ['', '## 是否存在协同收益', '',
              '在 Micro-F1 尺度上定义交互效应：`(D-C)-(B-A)`，即使用对比示例后，指令优化带来的增益是否更大。D 组分数最高本身不能证明协同。', '',
              f'交互效应：**{pp(interaction["value"])} 个百分点**；95% 配对区间 **[{pp(interaction["ci_low"])}, {pp(interaction["ci_high"])}]**。', '']
    if interaction['ci_low'] > 0:
        text = '本轮新切片观察到正向交互，区间未跨零；仍需独立样本和重复运行确认，不能据此宣称通用稳定收益。'
    elif interaction['ci_high'] < 0:
        text = '本轮交互方向为负，区间未跨零，未支持正向协同；仍需独立样本与重复运行确认。'
    else:
        text = '交互区间包含零，本轮不足以确认两种改动存在稳定协同。'
    lines += [text, '',
              '所有区间使用同一批样本 ID 配对、seed=42、1000 次 percentile bootstrap；交互计算每次对四组使用相同抽样。区间未经多重比较校正，且不包含服务端重复运行波动。点估计采用整体 Micro-F1，交互结论依赖这一指标尺度。', '',
              '本轮保留全部结果，不自动修改默认配置。实验中的 dev200 属于开发验证切片；若据此继续改方法，该切片不能再视为独立验证集。', '',
              '## 每类 TP/FP/FN 与 F1', '', '| 标签 | 支持数 | A | B | C | D |', '|---|---:|---|---|---|---|']
    for label, row in metrics['A']['per_label'].items():
        values = []
        for arm in 'ABCD':
            p = metrics[arm]['per_label'][label]
            values.append(f'{p["tp"]}/{p["fp"]}/{p["fn"]}；{pct(p["f1"])}')
        lines.append(f'| {label} | {row["support"]} | ' + ' | '.join(values) + ' |')
    lines += ['', '没有真值支持的类别：' + (', '.join(analysis['zero_support_labels']) or '无') + '。Macro-F1 对零分母记 0，小样本稀有类不稳定。', '',
              '## 示例及执行诊断', '', '| 项目 | A | B | C | D |', '|---|---:|---:|---:|---:|']
    for title, key in [('找到对比样本的查询数', 'contrast_found'), ('示例标签包含至少一个真值的比例', 'example_label_any_gold_coverage'), ('示例标签与真值平均 Jaccard', 'mean_example_label_jaccard')]:
        lines.append('| ' + title + ' | ' + ' | '.join(str(analysis['retrieval_diagnostics'][a][key]) if key == 'contrast_found' else pct(analysis['retrieval_diagnostics'][a][key]) for a in 'ABCD') + ' |')
    lines += ['', '标签覆盖率在预测完成后计算，仅用于诊断，不参与候选或示例选择。实际 A/B 的示例全文及顺序一致、C/D 一致，指令变化没有同时改变示例。', '',
              '| 项目 | A | B | C | D |', '|---|---:|---:|---:|---:|']
    for title, key in [('成功样本', 'succeeded'), ('失败样本', 'failed'), ('调用次数', 'calls'), ('重试次数', 'retries'),
                       ('输入 tokens', 'input_tokens'), ('输出 tokens', 'output_tokens'), ('推理 tokens', 'reasoning_tokens'),
                       ('缓存输入 tokens', 'cached_input_tokens'), ('平均调用毫秒', 'mean_call_latency_ms'),
                       ('检索总毫秒', 'retrieval_total_ms'), ('检索中位毫秒', 'retrieval_median_ms'), ('费用未知调用', 'unknown_cost_calls')]:
        lines.append('| ' + title + ' | ' + ' | '.join(str(runs[a][key]) for a in 'ABCD') + ' |')
    encoding = query['encoding_metadata']['encoding']
    lines += ['', 'v3 比 v1 长，实际输入 tokens 不相等；比较的是完整指令版本效果，不能把差异归因于某一句话。推理 tokens 是输出用量的子项，不重复相加计费。价格未经核实，费用保存为 NULL。检索耗时包括排序、示例筛选、token 预算计算及首次索引校验；缓存和服务负载也影响调用耗时。', '',
              '服务返回模型标识：`' + json.dumps({a: runs[a]['resolved_models'] for a in 'ABCD'}, ensure_ascii=False) + '`。服务端权重修订未提供，不能保证相同别名始终对应完全相同权重。', '',
              '## 编码与独立核验', '',
              f'复用 43,410 条训练向量，索引 `{state["build"]["index_id"]}`；模型修订 `{state["build"]["model_config"]["revision"]}`。本轮仅编码 200 条新查询，设备 {encoding["device"]}，截断 {encoding["truncated_count"]} 条，各批推理耗时之和 {sum(b["seconds"] for b in encoding["batches"]):.3f} 秒。CLS pooling、归一化 float32、768 维；完整模型文件哈希及权重工件保留。', '',
              '核验计数：`' + json.dumps(analysis['independent_validation'], ensure_ascii=False) + '`。', '',
              '核验原始 train/dev TSV、查询排除集合、全部候选顺序及相似度、示例原文与完整标签、每次请求与原始响应、数据库预测、独立重算评分及配对区间。', '']
    for fixed, title in [(True, 'D 相对 A：由不完全匹配变为完全匹配'), (False, 'D 相对 A：由完全匹配变为不完全匹配')]:
        selected = [r for r in cases if (set(r['truth']) == set(r['D']['predicted_labels'])) == fixed and (set(r['truth']) == set(r['A']['predicted_labels'])) != fixed]
        lines += [f'### {title}', '', '| source_id | 原文 | 真值 | A | D |', '|---|---|---|---|---|']
        for r in selected[:5]:
            values = [r['source_id'], r['text'], ', '.join(r['truth']), ', '.join(r['A']['predicted_labels']), ', '.join(r['D']['predicted_labels'])]
            lines.append('| ' + ' | '.join(v.replace('|', '\\|').replace('\n', ' ') for v in values) + ' |')
        if not selected:
            lines.append('| — | 本轮无此类案例 | — | — | — |')
        lines.append('')
    lines += ['## 数据定位与复现', '', '- 数据库：`data/research/experiments.sqlite3`，完整关联工件：`data/research/artifacts/`。',
              '- 标准导出：`data/research/exports/prompt-factorial-dev200-{A,B,C,D}-20260909/`。',
              '- 四组逐条明细：`data/research/exports/prompt-factorial-dev200-paired-cases-20260909.jsonl`。',
              '- 最终备份：`data/research/backups/prompt-factorial-final-20260909/`，含本轮与此前历史实验、模型权重和向量。',
              '- 状态、分析和核验：`prompt_factorial_state.json`、`prompt_factorial_analysis.json`、`prompt_factorial_final_verification.json`。', '',
              '| 组 | Run ID | Evaluation ID | 代码提交 |', '|---|---|---|---|']
    for arm in 'ABCD':
        lines.append(f'| {arm} | `{state["runs"][arm]}` | `{state["evaluations"][arm]}` | `{runs[arm]["code_commit"]}` |')
    lines += ['', f'分析工件：`{analysis["analysis_artifact_id"]}`；四组明细：`{analysis["paired_details_artifact_id"]}`；共同抽样及交互统计：`{analysis["factorial_artifact_id"]}`。', '']
    report = '\n'.join(lines)
    path = Path('docs/research/prompt-factorial-results-20260909.md')
    path.write_text(report)
    with Store(ROOT) as store, store.transaction():
        aid = store.put({'report': report, 'generator_source': Path(__file__).read_text(),
                         'analysis_artifact_id': analysis['analysis_artifact_id']}, 'prompt_factorial_report', inline=False)
        store.event('artifact_id', aid, 'prompt_factorial_reported')
    print(json.dumps({'report': str(path), 'report_artifact_id': aid}, ensure_ascii=False))


if __name__ == '__main__':
    main()
