# EmpatheticDialogues 64 条正式消融结论

## 实验边界

- 数据：EmpatheticDialogues 公开平衡 seed，64 条，32 类每类 2 条。
- 模型：`gpt-5.6-sol`；Codex CLI：`0.146.0`。
- 配置：`full`、`no_dynamic_examples`、`zero_shot`。
- 有效性：三组均为 64/64 有效预测、0 调用失败，共 192 次有效调用。

## 主结果

| Run | Correct | Accuracy | Macro F1 | Family Accuracy |
| --- | ---: | ---: | ---: | ---: |
| `full` | 36/64 | 56.25% | 54.91% | 76.56% |
| `no_dynamic_examples` | 37/64 | 57.81% | 54.06% | 78.12% |
| `zero_shot` | 37/64 | 57.81% | 54.06% | 78.12% |

两个 treatment 相对 `full` 均只多正确 1 条：Accuracy 差值为 +1.56%，配对
bootstrap 95% 区间为 -4.69%～+9.38%；Macro F1 差值为 -0.85%，配对
bootstrap 95% 区间为 -5.00%～+5.65%。两组的精确 McNemar 检验均为
`p=1.000`（`full` 独有正确 2 条，treatment 独有正确 3 条）。

## 可以与不可以得出的结论

本次 64 条结果**没有证据表明动态示例能提高 Accuracy 或 Macro F1**。两个差值区间均
包含 0，McNemar 检验也不拒绝准确率相同假设。32 条 pilot 中 `full` 多正确 1 条的轻微
正向趋势在 64 条上没有保持，说明 pilot 只适合验证链路，不适合证明组件效果。

`no_dynamic_examples` 与 `zero_shot` 的 64/64 Prompt 都不同，但 64/64 最终预测完全
一致。这只能说明在本模型、本数据和本次确定性运行中，两种示例策略没有产生可检测的
预测差异；不能外推为“few-shot 在所有场景都无效”。

EmpatheticDialogues 正式记录没有逐句历史标签，因此仍不能验证
`no_emotion_history` 和 `short_context`。

## 证据

- 逐条预测：`data/records/codex_cli_ablation/empathetic_dialogues_seed64_gpt56sol/`
- 指标：`metrics.csv`
- 完整报告：`report-zh.md`
- 简要摘要：`summary.md`

下一阶段应在完整 2,542 条 test 上复核总体结论，并引入带逐句人工标签的多轮数据集，
单独验证情绪历史和上下文组件。
