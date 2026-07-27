# EmpatheticDialogues V1

本目录把公开的 EmpatheticDialogues 官方测试集转换为项目现有的情绪消融格式，作为默认的真实数据基准。源数据由众包参与者撰写，不是生成式 AI 合成数据。

## 为什么选择它

| 数据集 | 规模与语言 | 标签 | 与本项目的关系 | 许可/使用边界 |
| --- | --- | --- | --- | --- |
| **[EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues)** | 约 2.5 万段英文对话 | 32 类 | 与项目现有 32 类逐项一致；多轮对话可直接复用上下文消融 | CC BY-NC 4.0，仅非商业使用 |
| [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) | 约 5.8 万条英文 Reddit 评论 | 27 类 + neutral，多标签 | 人工标注、规模大，但需重新映射标签且不是多轮对话 | 官方代码 Apache 2.0；原始 Reddit 内容仍需注意平台内容权利 |
| [CPED](https://github.com/scutcyr/CPED) | 约 1.2 万段中文对话、13.3 万句 | 13 类 | 中文、多轮，适合后续跨语言外部验证；13→32 映射会损失粒度 | 官方仓库 Apache 2.0，语料源自电视剧，使用前仍应复核内容权利 |
| [MELD](https://github.com/declare-lab/MELD) | 1,433 段英文多方对话、13,708 句 | 7 类 | 有逐句标签，但类别过粗，且源自《Friends》片段 | 再分发和影视内容权利边界不如本项目首选清晰 |
| [DailyDialog](https://aclanthology.org/I17-1099/) | 13,118 段英文对话 | 7 类 | 人工标注、多轮，但绝大多数句子为 neutral，且不能直接覆盖 32 类 | 数据说明限制商业使用 |

因此首轮接入 EmpatheticDialogues：它是唯一无需人为标签映射、还能保留多轮上下文结构的成熟候选。CPED 建议作为下一阶段的中文外部验证集单独报告，不应与 32 类主指标直接混算。

## 已纳入的数据

- `release/test.jsonl`：官方 `test.csv` 转换得到的 2,542 段对话，覆盖全部 32 类。
- `release/balanced_seed.jsonl`：从测试集按源文件顺序为每类取 2 条，共 64 条，便于沿用原来的低成本 Codex CLI 五组消融。
- `reports/*.csv`：完整测试集的标签、语言、上下文依赖和场景分布。
- `metadata.json`：来源 URL、论文、许可、原包 SHA-256 和转换版本。

`expected` 使用 `label_provenance=human_authored_emotion_grounding`。这表示众包作者按给定情绪类别回忆情境并展开对话；它不是 AI 生成标签，也不等同于对每条最终 utterance 进行独立事后复标和仲裁。论文或汇报中应使用“人工撰写、情绪情境锚定”这一准确表述。

## 转换规则

每段对话的第一位说话者是目标用户：

1. 取该用户最后一次非空发言作为 `current_input`；
2. 将此前发言按目标用户=`human`、另一位参与者=`ai` 写入 `history`；
3. 使用源数据的 conversation-level `context` 作为 `expected`；
4. 不把含有情绪情境描述的 `prompt` 放进模型输入，避免直接泄漏标签；仅保留为 `source_situation` 供追溯；
5. 恢复源数据中的 `_comma_` 占位符。

## 可复现生成与校验

已有官方压缩包时：

```bash
python scripts/benchmark/prepare_empathetic_dialogues.py \
  --archive /path/to/empatheticdialogues.tar.gz
```

省略 `--archive` 时脚本从 Meta 官方地址下载临时副本，并在转换前校验固定 SHA-256。随后执行原有校验与导出：

```bash
python scripts/benchmark/validate_emotion_benchmark.py \
  --input data/benchmarks/empathetic_dialogues_v1/release/test.jsonl

python scripts/benchmark/export_emotion_ablation_v2.py \
  --input data/benchmarks/empathetic_dialogues_v1/release/balanced_seed.jsonl \
  --output-dir data/records/empathetic_dialogues_seed_export
```

导出后的 `dialogues.jsonl` 和 `labels.jsonl` 可不改评估代码，直接交给 `run_emotion_ablation.py`、`run_codex_cli_emotion_ablation.py`、`evaluate_emotion_ablation.py` 或 `report_codex_cli_emotion_ablation.py`。指标仍为 Accuracy、Macro F1、失败数、语言/上下文切片和标签混淆。

## 已完成的原方法 pilot

仓库保留了 `reports/codex_pilot10/`：使用 Codex CLI 0.142.4、`gpt-5.6-sol`，对平衡 seed 前 10 条运行原有 5 组消融，共 50 次调用，调用失败为 0。

| Run | Accuracy | Macro F1 | Prompt treatment |
| --- | ---: | ---: | --- |
| `full` | 30.00% | 14.67% | baseline |
| `no_dynamic_examples` | 40.00% | 21.67% | effective |
| `no_emotion_history` | 30.00% | 15.00% | **10/10 与 full 相同，no-op** |
| `short_context` | 30.00% | 15.00% | 2/10 发生变化 |
| `zero_shot` | 20.00% | 10.00% | effective |

这 10 条仅覆盖 `afraid`、`angry`、`annoyed`、`anticipating`、`anxious` 五类，只能证明真实数据已贯通原验证链路并提供初步错误样例，不能用于 32 类总体结论，也不能据此断言静态示例优于动态示例。`no_emotion_history` 在这些样本没有可供移除的历史情绪字段，因此是无效消融；其指标波动不能归因于该组件。

## 研究限制

- 当前数据只有英文，不能支撑中文效果结论。
- 标签针对整段情绪情境，不是每个目标 utterance 的独立复标结果。
- 为公平评测只使用官方 test split；不要用 train split 构造 few-shot 示例后再把同源样本泄漏回测试输入。
- CC BY-NC 4.0 不允许把数据用于以商业利益为主要目的的场景。
