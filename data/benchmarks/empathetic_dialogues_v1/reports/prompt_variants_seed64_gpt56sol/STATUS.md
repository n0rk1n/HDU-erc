# Prompt 变体 seed64 实验状态

- 状态：`exploratory_pilot_frozen`
- 冻结日期：2026-08-03
- 数据：EmpatheticDialogues 官方 test 中的 64 条平衡 seed
- 范围：`full` 与 4 个 Prompt 变体，共 320 次有效预测

## 冻结决定

本实验已经使用 64 条 test seed 比较并筛选 Prompt，因此这些样本已经承担开发集
用途，不再属于未触碰的最终测试。现有预测、指标、报告和运行元数据继续保留，
仅用于复现实验过程与记录探索性趋势。

从冻结日期起：

- 不再基于这 64 条样本继续修改或筛选 Prompt；
- 不把 `prompt_no_label_guidance` 或 `prompt_coarse_to_fine` 称为已确认的第二阶段候选；
- 不根据本次结果直接启动完整 2,542 条 test；
- 下一步先完成输入—标签对齐审计，再把 Prompt 选择迁移到 validation split。

冻结只改变实验结论边界，不改变已经生成的原始预测和数值指标。
