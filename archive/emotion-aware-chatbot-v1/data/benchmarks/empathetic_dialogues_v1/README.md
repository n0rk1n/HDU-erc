# EmpatheticDialogues V1

本目录把公开的 EmpatheticDialogues 官方测试集转换为项目现有的情绪消融格式，作为唯一正式基准。源数据由众包参与者人工撰写。

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

- `release/test.jsonl`：正式主基准。官方 `test.csv` 的 2,542 段对话，每段只取首位说话者的第一条情境表达，覆盖全部 32 类。
- `release/balanced_seed.jsonl`：从正式主基准为每类取 2 条，共 64 条；按轮询顺序排列，因此前 32 条就覆盖全部 32 类。
- `release/context_diagnostic.jsonl`：保留旧方法的“最后一条目标用户发言 + 历史”转换，仅作为弱标签诊断集，不计入正式准确率结论。
- `few_shot/train_examples.jsonl`：从官方 **train split** 每类取 2 条人工撰写示例，共 64 条；只用于 Prompt，不与 test 样本重叠。
- `reports/*.csv`：完整测试集的标签、语言、上下文依赖和场景分布。
- `metadata.json`：来源 URL、论文、许可、原包 SHA-256 和转换版本。

`expected` 使用 `label_provenance=human_authored_emotion_grounding`。这表示众包作者按给定情绪类别回忆情境并展开对话；它不等同于对每条最终 utterance 进行独立事后复标和仲裁。论文或汇报中应使用“人工撰写、情绪情境锚定”这一准确表述。

## 转换规则

EmpatheticDialogues 的 `context` 是整段对话的情绪情境标签，不是每一句话的独立标注。正式转换因此遵循：

1. 取目标用户第一条非空发言作为 `current_input`，该发言是情境展开的起点；
2. 正式样本的 `history=[]`，避免把整段标签错误地当作后续句子的当前情绪标签；
3. 使用源数据的 conversation-level `context` 作为 `expected`；
4. 不把含有情绪情境描述的 `prompt` 放进模型输入，避免直接泄漏标签；仅保留为 `source_situation` 供追溯；
5. 恢复源数据中的 `_comma_` 占位符。

旧转换仍写入 `context_diagnostic.jsonl`，并显式标记 `ground_truth_alignment=weak_conversation_label_on_later_turn` 与 `emotion_evidence_weak`。它可用于观察对话末段的标签漂移，但不能作为“当前句情绪识别”的正式 ground truth。

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

python scripts/benchmark/export_emotion_benchmark.py \
  --input data/benchmarks/empathetic_dialogues_v1/release/balanced_seed.jsonl \
  --output-dir data/records/empathetic_dialogues_seed_export
```

导出后的 `dialogues.jsonl` 和 `labels.jsonl` 可不改评估链路。正式主指标仍为 exact Accuracy 和 32 类 Macro F1；另报告 Family Accuracy / Family Macro F1，仅用于区分 `afraid→terrified` 一类的近邻边界错误，不能替代主指标。

正式集没有历史，因此只运行三组实际会改变 Prompt 的对照：

```bash
python -m scripts.ablation.run_emotion_ablation \
  --dialogues-file data/records/empathetic_dialogues_seed_export/dialogues.jsonl \
  --output-dir data/records/ablation \
  --run full \
  --run no_dynamic_examples \
  --run zero_shot
```

`no_emotion_history` 与 `short_context` 在该数据上都与 `full` 完全同构，不应调用、更不应把随机波动解释成组件贡献。历史相关消融需换用带**逐句人工标签**的数据集（例如 MELD/CPED）另做实验。

## 可用 Prompt 变体

`full` 之外提供 4 个固定 Prompt 变体（`prompt_no_label_guidance`、
`prompt_concise_direct`、`prompt_coarse_to_fine`、`prompt_contrastive_check`），
共享动态示例、情绪历史与默认上下文窗口，仅模板不同。应在独立 validation split 上选择模板，再在 test split 上做一次最终评测，避免测试集反复调参造成结果偏乐观。

早期标签错位问题及 32 条对齐 pilot 的诊断过程保留在 `reports/remediation_report.md` 和 `reports/aligned_pilot32/`。

## 研究限制

- 当前数据只有英文，不能支撑中文效果结论。
- 标签针对整段情绪情境，不是每个目标 utterance 的独立复标结果，因此后续句子只作为弱标签诊断。
- test split 只用于评测；few-shot 只来自 train split，并保留 split 和样本 ID 供泄漏检查。
- CC BY-NC 4.0 不允许把数据用于以商业利益为主要目的的场景。
