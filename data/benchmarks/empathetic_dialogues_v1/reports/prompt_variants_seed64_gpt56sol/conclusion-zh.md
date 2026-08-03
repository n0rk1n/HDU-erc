# Prompt 变体实验结论

## 第二阶段候选 Prompt

- 候选 Prompt：`prompt_no_label_guidance`，Accuracy 59.38%，相对 `full` 点估计 +3.12%，配对 95% 区间 -3.12%–+9.38%。
- 候选 Prompt：`prompt_coarse_to_fine`，Accuracy 59.38%，相对 `full` 点估计 +3.12%，配对 95% 区间 -3.12%–+9.38%。

## 边界声明

- 64 条 seed 结果只用于筛选，不能作为最终结论。
- 当配对差值区间包含 0 时，不能表述为已经证明提升。
- 完整 2,542 条测试集尚未运行；第二阶段调用量需要用户再次确认。
