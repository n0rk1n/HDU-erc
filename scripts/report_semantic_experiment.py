"""Render all four arms from reconciled evidence, without cherry-picking a baseline."""
import json
from pathlib import Path

from emotion_lab.storage import Store, now

ROOT = Path('data/research')


def main():
    analysis = json.loads((ROOT / 'semantic_experiment_analysis.json').read_text())
    state, metrics, runs = analysis['state'], analysis['metrics'], analysis['runs']
    with Store(ROOT) as store:
        cases = store.json(analysis['paired_details_artifact_id'])['records']
        index = store.one('SELECT * FROM embedding_indexes WHERE index_id=?', (state['build']['index_id'],))
        build = store.json(index['build_environment_artifact_id'])
        query = store.json(state['build']['queries_artifact_id'])
    pct = lambda value: f'{value * 100:.2f}%'
    pp = lambda value: f'{value * 100:+.2f}'
    lines = ['# 语义检索四组对照实验', '', f'生成时间（UTC）：{now()}。', '',
             '## 实验范围', '',
             '固定 GLM-5.3、native-labels-v3 指令、28 类、候选 50、入选 4 条示例、相同输出预算及评分方式。使用新的固定 dev200；排除先前评估集合的同文本记录和训练集文本复制。四组完成后统一评分，官方 test 未使用。', '',
             '| 组 | 相似度 | 示例政策 |', '|---|---|---|',
             '| A | 词面 Jaccard | ranked |', '| B | BGE 语义余弦 | ranked |',
             '| C | 词面 Jaccard | contrastive-v1 |', '| D | BGE 语义余弦 | contrastive-v1 |', '',
             'A/B/C/D 的系统指令、标签定义、查询和模型参数相同，只有入选示例允许变化。对比规则和混淆关系沿用旧版。本轮没有加入专用分类器、重排模型或新的提示词。', '',
             '## 主结果', '', '| 指标 | A | B | C | D |', '|---|---:|---:|---:|---:|']
    for title, key in [('Micro-F1', 'micro_f1'), ('Macro-F1（28 类）', 'macro_f1'), ('完全匹配率', 'exact_match'),
                       ('Precision', 'precision'), ('Recall', 'recall'), ('至少一个细类命中', 'any_hit'),
                       ('大类 Micro-F1', 'family_micro_f1'), ('大类完全匹配', 'family_exact_match'), ('至少一个大类命中', 'family_any_hit')]:
        lines.append('| ' + title + ' | ' + ' | '.join(pct(metrics[a][key]) for a in 'ABCD') + ' |')
    for title, key in [('TP', 'tp'), ('FP', 'fp'), ('FN', 'fn'), ('平均预测标签数', 'mean_predicted_labels')]:
        lines.append('| ' + title + ' | ' + ' | '.join(str(metrics[a][key]) for a in 'ABCD') + ' |')
    lines += ['', '## 配对比较', '', '| 比较 | 用途 | Micro-F1 差值（百分点） | 95% 区间（百分点） | 示例顺序或成员改变 | 错→全对 / 全对→错 |', '|---|---|---:|---|---:|---|']
    for name, c in analysis['comparisons'].items():
        primary = '主比较：更换相似度' if name in ['B-A', 'D-C'] else '次比较：对比规则消融'
        lines.append(f'| {name} | {primary} | {pp(c["micro_f1_delta"])} | [{pp(c["ci_low"])}, {pp(c["ci_high"])}] | {c["changed_examples"]} | {c["exact_fixed"]} / {c["exact_regressed"]} |')
    lines += ['', '配对 percentile bootstrap：seed=42、1000 次重采样。区间未经多重比较校正，不覆盖服务重复运行波动；不能只挑最高的一组宣称稳定改进。', '']
    for name in ['B-A', 'D-C']:
        c = analysis['comparisons'][name]
        interpretation = '本轮观察到上升' if c['micro_f1_delta'] > 0 else '本轮未观察到提升'
        uncertainty = '区间包含零，证据不足以确认稳定差异' if c['ci_low'] <= 0 <= c['ci_high'] else '本轮区间未跨零，仍需独立重复和更大样本确认'
        lines.append(f'- {name}：{interpretation}（{pp(c["micro_f1_delta"])} 个百分点）；{uncertainty}。')
    lines += ['', '本轮保留全部实验配置和结果，不自动替换默认方法。该切片用于初步筛选；后续若据此调整方法，它也成为开发数据。旧 dev200 的分数不能与本轮新切片直接相减来估计方法收益。', '',
             '## 每类 TP/FP/FN 与 F1', '', '| 标签 | 支持数 | A | B | C | D |', '|---|---:|---|---|---|---|']
    for label, row in metrics['A']['per_label'].items():
        values = []
        for arm in 'ABCD':
            p = metrics[arm]['per_label'][label]
            values.append(f'{p["tp"]}/{p["fp"]}/{p["fn"]}；{pct(p["f1"])}')
        lines.append(f'| {label} | {row["support"]} | ' + ' | '.join(values) + ' |')
    lines += ['', '本切片没有真值支持的类别：' + (', '.join(analysis['zero_support_labels']) or '无') + '。Macro-F1 对零分母按 0，稀有类估计不稳定。', '',
              '## 示例诊断', '', '| 项目 | A | B | C | D |', '|---|---:|---:|---:|---:|']
    for title, key in [('找到对比样本的查询数', 'contrast_found'), ('4 条示例标签包含至少一个真值的比例', 'example_label_any_gold_coverage'), ('示例标签与真值的平均 Jaccard', 'mean_example_label_jaccard')]:
        values = [analysis['retrieval_diagnostics'][a][key] for a in 'ABCD']
        lines.append('| ' + title + ' | ' + ' | '.join(str(v) if key == 'contrast_found' else pct(v) for v in values) + ' |')
    lines += ['', '这些是完成预测之后计算的标签重合诊断，不参与检索选择，也不等同于最终分类准确性。', '',
              '## 用量与耗时', '', '| 项目 | A | B | C | D |', '|---|---:|---:|---:|---:|']
    for title, key in [('成功样本', 'succeeded'), ('最终失败', 'failed'), ('调用次数', 'calls'), ('重试', 'retries'),
                       ('输入 tokens', 'input_tokens'), ('输出 tokens', 'output_tokens'), ('推理 tokens', 'reasoning_tokens'),
                       ('缓存输入 tokens', 'cached_input_tokens'), ('平均模型调用毫秒', 'mean_call_latency_ms'),
                       ('检索总毫秒', 'retrieval_total_ms'), ('检索中位毫秒', 'retrieval_median_ms'), ('费用未知调用', 'unknown_cost_calls')]:
        lines.append('| ' + title + ' | ' + ' | '.join(str(runs[a][key]) for a in 'ABCD') + ' |')
    encoding = build['encoding']
    qencoding = query['encoding_metadata']['encoding']
    lines += ['', f'预先固定的顺序执行次序：{state["run_order"]}。实际输入 tokens 因示例长度不同而变化；缓存和服务负载会影响耗时，不能把总耗时差全部归因于检索算法。未获得可核实单价，费用保持 NULL。', '',
              '## 向量构建与存储', '',
              f'- 模型：`BAAI/bge-base-en-v1.5`，修订 `{state["build"]["model_config"]["revision"]}`。权重及配置全文已存工件 `{state["build"]["model_checkpoint_artifact_id"]}`。',
              f'- train：{len(encoding["rows"])} 条；查询：{len(qencoding["rows"])} 条。CLS pooling、无额外文本指令、归一化 float32、维度 {index["dimension"]}，精确余弦检索。',
              f'- 编码设备：{encoding["device"]}；batch_size=32，max_length=512。train 截断 {encoding["truncated_count"]} 条，查询截断 {qencoding["truncated_count"]} 条；逐条编码 token IDs、长度和批次耗时全部保留。',
              f'- 编码各批推理秒数之和：train {sum(b["seconds"] for b in encoding["batches"]):.3f}，query {sum(b["seconds"] for b in qencoding["batches"]):.3f}。这是可复用的一次性编码开销，另有加载、分词和写盘开销。',
              f'- 冻结训练索引 `{state["build"]["index_id"]}`；查询工件 `{state["build"]["queries_artifact_id"]}`。查询向量不进入训练索引，正式分类与恢复运行都读取同一向量工件。', '',
              '## 独立核验与案例', '', '核验计数：`' + json.dumps(analysis['independent_validation'], ensure_ascii=False) + '`。', '',
              '核验包含原始 train/dev TSV、全部候选和分数、入选示例全文及完整标签、实际请求、原始响应、数据库预测和独立重算的评分。', '']
    for name in ['B-A', 'D-C']:
        right, left = name.split('-')
        for fixed, label in [(True, '由不完全匹配变为完全匹配'), (False, '由完全匹配变为不完全匹配')]:
            selected = [r for r in cases if (set(r['truth']) == set(r[right]['predicted_labels'])) == fixed and (set(r['truth']) == set(r[left]['predicted_labels'])) != fixed]
            lines += [f'### {name}：{label}', '', '| source_id | 原文 | 真值 | 原方法 | 新方法 |', '|---|---|---|---|---|']
            for r in selected[:5]:
                values = [r['source_id'], r['text'], ', '.join(r['truth']), ', '.join(r[left]['predicted_labels']), ', '.join(r[right]['predicted_labels'])]
                lines.append('| ' + ' | '.join(v.replace('|', '\\|').replace('\n', ' ') for v in values) + ' |')
            if not selected:
                lines.append('| — | 本轮无此类案例 | — | — | — |')
            lines.append('')
    lines += ['## 复现定位', '', '- 数据库：`data/research/experiments.sqlite3`；全文工件：`data/research/artifacts/`。',
              '- 四组配对明细：`data/research/exports/semantic-dev200-paired-cases-20260908.jsonl`。',
              '- 各组标准导出：`data/research/exports/semantic-dev200-{A,B,C,D}-20260908/`。',
              '- 状态与分析：`semantic_build_state.json`、`semantic_experiment_state.json`、`semantic_experiment_analysis.json`。',
              '- 最终可恢复备份：`data/research/backups/semantic-retrieval-final-20260908/`；完整性核验记录单独保存。', '',
              '| 组 | Run ID | Evaluation ID | 代码提交 |', '|---|---|---|---|']
    for arm in 'ABCD':
        lines.append(f'| {arm} | `{state["runs"][arm]}` | `{state["evaluations"][arm]}` | `{runs[arm]["code_commit"]}` |')
    lines += ['', f'分析工件：`{analysis["analysis_artifact_id"]}`；逐条明细工件：`{analysis["paired_details_artifact_id"]}`。', '',
              '公开模型说明：[BGE-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)。本报告只使用本次真实运行数据，不以模型卡成绩代替本项目实验结果。', '']
    text = '\n'.join(lines)
    path = Path('docs/research/semantic-retrieval-results-20260908.md')
    path.write_text(text)
    with Store(ROOT) as store, store.transaction():
        aid = store.put({'report': text, 'generator_source': Path(__file__).read_text(), 'analysis_artifact_id': analysis['analysis_artifact_id']}, 'semantic_result_report', inline=False)
        store.event('artifact_id', aid, 'semantic_results_reported')
    print(json.dumps({'report': str(path), 'report_artifact_id': aid}, ensure_ascii=False))


if __name__ == '__main__':
    main()
