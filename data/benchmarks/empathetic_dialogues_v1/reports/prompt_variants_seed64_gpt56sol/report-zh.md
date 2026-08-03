# Codex CLI 情绪识别 Prompt 多版本实验报告

> **状态：探索性 pilot 已于 2026-08-03 冻结。** 本实验使用官方 test 的 64 条
> 平衡 seed 进行 Prompt 筛选，不属于未触碰的最终测试。保留本报告和原始记录只为
> 追溯；不再基于这 64 条调参，也不据此直接启动完整 2,542 条 test。

## 实验元数据

- branch: `codex/prompt-variant-ablation-20260803`
- codex_version: `codex-cli 0.146.0`
- commit: `c69e2287767cec78922444f63a84515f85fc8a3b`
- ended_at: `2026-08-03T18:54:26+08:00`
- execution_note: `Each run began with a one-case smoke snapshot; matching successful provenance was reused, then the remaining 63 cases completed without failures.`
- model: `gpt-5.6-sol`
- started_at: `2026-08-03T18:19:28+08:00`

## 整体结果

| Run | Samples | Valid predictions | 调用失败 | Correct | Accuracy (95% CI) | Macro F1 | Family Accuracy* | Family Macro F1* | Δ Accuracy vs full (paired 95% CI) | Δ Macro F1 vs full (paired 95% CI) | McNemar exact p | Prompt identical/full | Treatment status | Provenance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| full | 64 | 64 | 0 | 36 | 56.25% (44.09%–67.71%) | 55.31% | 76.56% | 75.17% | +0.00% (+0.00%–+0.00%) | +0.00% (+0.00%–+0.00%) | 1.0000 | 64/64 | baseline | record_input_vs_full_by_case_id |
| prompt_coarse_to_fine | 64 | 64 | 0 | 38 | 59.38% (47.15%–70.54%) | 56.56% | 78.12% | 75.16% | +3.12% (-3.12%–+9.38%) | +1.25% (-3.10%–+7.05%) | 0.6250 | 0/64 | effective_prompt_change | record_input_vs_full_by_case_id |
| prompt_concise_direct | 64 | 64 | 0 | 37 | 57.81% (45.61%–69.13%) | 54.91% | 75.00% | 73.94% | +1.56% (-3.12%–+7.81%) | -0.40% (-3.59%–+5.00%) | 1.0000 | 0/64 | effective_prompt_change | record_input_vs_full_by_case_id |
| prompt_contrastive_check | 64 | 64 | 0 | 36 | 56.25% (44.09%–67.71%) | 53.65% | 75.00% | 73.01% | +0.00% (-6.25%–+6.25%) | -1.67% (-5.97%–+4.55%) | 1.0000 | 0/64 | effective_prompt_change | record_input_vs_full_by_case_id |
| prompt_no_label_guidance | 64 | 64 | 0 | 38 | 59.38% (47.15%–70.54%) | 57.92% | 76.56% | 75.07% | +3.12% (-3.12%–+9.38%) | +2.60% (-1.31%–+7.39%) | 0.6250 | 0/64 | effective_prompt_change | record_input_vs_full_by_case_id |

*Family 指标只用于诊断相邻标签（例如 afraid/terrified、annoyed/angry）的边界错误，
不能替代 32 类 exact Accuracy 与 Macro F1。

## 配对统计推断

| Run | full only correct | treatment only correct | McNemar exact p | Δ Accuracy paired 95% CI | Δ Macro F1 paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 0 | 0 | 1.0000 | +0.00%–+0.00% | +0.00%–+0.00% |
| prompt_coarse_to_fine | 1 | 3 | 0.6250 | -3.12%–+9.38% | -3.10%–+7.05% |
| prompt_concise_direct | 1 | 2 | 1.0000 | -3.12%–+7.81% | -3.59%–+5.00% |
| prompt_contrastive_check | 2 | 2 | 1.0000 | -6.25%–+6.25% | -5.97%–+4.55% |
| prompt_no_label_guidance | 1 | 3 | 0.6250 | -3.12%–+9.38% | -1.31%–+7.39% |

Accuracy 采用精确 McNemar 检验；差值区间使用按 case_id 配对的 percentile bootstrap（10000 次，固定种子 20260803）。
p < 0.05 才可视为拒绝“两配置准确率相同”的初步证据；Macro F1 差值仅根据配对 bootstrap 区间解读。

## 语言切片

| Run | Slice | Samples | Correct | Accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| full | zh | 0 | 0 | 0.00% | 0.00% |
| full | en | 64 | 36 | 56.25% | 55.31% |
| prompt_coarse_to_fine | zh | 0 | 0 | 0.00% | 0.00% |
| prompt_coarse_to_fine | en | 64 | 38 | 59.38% | 56.56% |
| prompt_concise_direct | zh | 0 | 0 | 0.00% | 0.00% |
| prompt_concise_direct | en | 64 | 37 | 57.81% | 54.91% |
| prompt_contrastive_check | zh | 0 | 0 | 0.00% | 0.00% |
| prompt_contrastive_check | en | 64 | 36 | 56.25% | 53.65% |
| prompt_no_label_guidance | zh | 0 | 0 | 0.00% | 0.00% |
| prompt_no_label_guidance | en | 64 | 38 | 59.38% | 57.92% |

## 上下文依赖切片

| Run | Slice | Samples | Correct | Accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| full | none | 64 | 36 | 56.25% | 55.31% |
| full | low | 0 | 0 | 0.00% | 0.00% |
| full | medium | 0 | 0 | 0.00% | 0.00% |
| full | high | 0 | 0 | 0.00% | 0.00% |
| prompt_coarse_to_fine | none | 64 | 38 | 59.38% | 56.56% |
| prompt_coarse_to_fine | low | 0 | 0 | 0.00% | 0.00% |
| prompt_coarse_to_fine | medium | 0 | 0 | 0.00% | 0.00% |
| prompt_coarse_to_fine | high | 0 | 0 | 0.00% | 0.00% |
| prompt_concise_direct | none | 64 | 37 | 57.81% | 54.91% |
| prompt_concise_direct | low | 0 | 0 | 0.00% | 0.00% |
| prompt_concise_direct | medium | 0 | 0 | 0.00% | 0.00% |
| prompt_concise_direct | high | 0 | 0 | 0.00% | 0.00% |
| prompt_contrastive_check | none | 64 | 36 | 56.25% | 53.65% |
| prompt_contrastive_check | low | 0 | 0 | 0.00% | 0.00% |
| prompt_contrastive_check | medium | 0 | 0 | 0.00% | 0.00% |
| prompt_contrastive_check | high | 0 | 0 | 0.00% | 0.00% |
| prompt_no_label_guidance | none | 64 | 38 | 59.38% | 57.92% |
| prompt_no_label_guidance | low | 0 | 0 | 0.00% | 0.00% |
| prompt_no_label_guidance | medium | 0 | 0 | 0.00% | 0.00% |
| prompt_no_label_guidance | high | 0 | 0 | 0.00% | 0.00% |

## 标签混淆

| Run | Expected | Predicted | Count |
| --- | --- | --- | ---: |
| full | afraid | terrified | 1 |
| full | angry | terrified | 1 |
| full | anticipating | excited | 2 |
| full | anxious | guilty | 1 |
| full | apprehensive | afraid | 1 |
| full | apprehensive | anxious | 1 |
| full | ashamed | guilty | 1 |
| full | caring | surprised | 1 |
| full | confident | anticipating | 1 |
| full | content | joyful | 1 |
| full | disappointed | anticipating | 1 |
| full | faithful | guilty | 1 |
| full | furious | angry | 1 |
| full | furious | disgusted | 1 |
| full | guilty | embarrassed | 1 |
| full | guilty | terrified | 1 |
| full | jealous | disappointed | 1 |
| full | joyful | excited | 2 |
| full | lonely | devastated | 1 |
| full | prepared | confident | 1 |
| full | sad | devastated | 1 |
| full | sentimental | guilty | 1 |
| full | sentimental | nostalgic | 1 |
| full | surprised | joyful | 1 |
| full | terrified | anticipating | 1 |
| full | trusting | grateful | 1 |
| prompt_coarse_to_fine | afraid | terrified | 1 |
| prompt_coarse_to_fine | angry | afraid | 1 |
| prompt_coarse_to_fine | anticipating | excited | 2 |
| prompt_coarse_to_fine | apprehensive | afraid | 1 |
| prompt_coarse_to_fine | apprehensive | anxious | 1 |
| prompt_coarse_to_fine | ashamed | guilty | 1 |
| prompt_coarse_to_fine | caring | surprised | 1 |
| prompt_coarse_to_fine | confident | anticipating | 1 |
| prompt_coarse_to_fine | content | joyful | 1 |
| prompt_coarse_to_fine | disappointed | anticipating | 1 |
| prompt_coarse_to_fine | disgusted | annoyed | 1 |
| prompt_coarse_to_fine | furious | angry | 1 |
| prompt_coarse_to_fine | furious | disgusted | 1 |
| prompt_coarse_to_fine | guilty | embarrassed | 1 |
| prompt_coarse_to_fine | guilty | terrified | 1 |
| prompt_coarse_to_fine | jealous | disappointed | 1 |
| prompt_coarse_to_fine | joyful | excited | 2 |
| prompt_coarse_to_fine | lonely | devastated | 1 |
| prompt_coarse_to_fine | prepared | confident | 1 |
| prompt_coarse_to_fine | sentimental | guilty | 1 |
| prompt_coarse_to_fine | sentimental | nostalgic | 1 |
| prompt_coarse_to_fine | surprised | joyful | 1 |
| prompt_coarse_to_fine | terrified | trusting | 1 |
| prompt_coarse_to_fine | trusting | grateful | 1 |
| prompt_concise_direct | afraid | terrified | 1 |
| prompt_concise_direct | angry | afraid | 1 |
| prompt_concise_direct | anticipating | excited | 1 |
| prompt_concise_direct | anticipating | prepared | 1 |
| prompt_concise_direct | apprehensive | afraid | 2 |
| prompt_concise_direct | ashamed | guilty | 1 |
| prompt_concise_direct | caring | surprised | 1 |
| prompt_concise_direct | confident | anticipating | 1 |
| prompt_concise_direct | content | joyful | 1 |
| prompt_concise_direct | disappointed | anticipating | 1 |
| prompt_concise_direct | furious | angry | 1 |
| prompt_concise_direct | furious | disgusted | 1 |
| prompt_concise_direct | guilty | embarrassed | 1 |
| prompt_concise_direct | guilty | terrified | 1 |
| prompt_concise_direct | jealous | disappointed | 1 |
| prompt_concise_direct | joyful | excited | 2 |
| prompt_concise_direct | lonely | devastated | 1 |
| prompt_concise_direct | prepared | confident | 1 |
| prompt_concise_direct | sad | devastated | 1 |
| prompt_concise_direct | sentimental | guilty | 1 |
| prompt_concise_direct | sentimental | nostalgic | 1 |
| prompt_concise_direct | surprised | joyful | 1 |
| prompt_concise_direct | terrified | trusting | 1 |
| prompt_concise_direct | trusting | confident | 1 |
| prompt_concise_direct | trusting | grateful | 1 |
| prompt_contrastive_check | afraid | terrified | 1 |
| prompt_contrastive_check | angry | afraid | 1 |
| prompt_contrastive_check | anticipating | excited | 2 |
| prompt_contrastive_check | apprehensive | afraid | 1 |
| prompt_contrastive_check | apprehensive | anxious | 1 |
| prompt_contrastive_check | ashamed | guilty | 1 |
| prompt_contrastive_check | caring | surprised | 1 |
| prompt_contrastive_check | confident | anticipating | 1 |
| prompt_contrastive_check | content | joyful | 1 |
| prompt_contrastive_check | disappointed | anticipating | 1 |
| prompt_contrastive_check | disgusted | annoyed | 1 |
| prompt_contrastive_check | faithful | guilty | 1 |
| prompt_contrastive_check | furious | angry | 1 |
| prompt_contrastive_check | furious | disgusted | 1 |
| prompt_contrastive_check | guilty | embarrassed | 1 |
| prompt_contrastive_check | guilty | terrified | 1 |
| prompt_contrastive_check | jealous | disappointed | 1 |
| prompt_contrastive_check | joyful | excited | 2 |
| prompt_contrastive_check | lonely | devastated | 1 |
| prompt_contrastive_check | prepared | confident | 1 |
| prompt_contrastive_check | sentimental | guilty | 1 |
| prompt_contrastive_check | sentimental | nostalgic | 1 |
| prompt_contrastive_check | surprised | joyful | 1 |
| prompt_contrastive_check | terrified | trusting | 1 |
| prompt_contrastive_check | trusting | confident | 1 |
| prompt_contrastive_check | trusting | grateful | 1 |
| prompt_no_label_guidance | angry | terrified | 1 |
| prompt_no_label_guidance | anticipating | excited | 1 |
| prompt_no_label_guidance | anxious | guilty | 1 |
| prompt_no_label_guidance | apprehensive | afraid | 1 |
| prompt_no_label_guidance | apprehensive | anxious | 1 |
| prompt_no_label_guidance | ashamed | guilty | 1 |
| prompt_no_label_guidance | caring | surprised | 1 |
| prompt_no_label_guidance | confident | anticipating | 1 |
| prompt_no_label_guidance | content | joyful | 1 |
| prompt_no_label_guidance | disappointed | anticipating | 1 |
| prompt_no_label_guidance | excited | anticipating | 1 |
| prompt_no_label_guidance | faithful | guilty | 1 |
| prompt_no_label_guidance | furious | angry | 1 |
| prompt_no_label_guidance | furious | disgusted | 1 |
| prompt_no_label_guidance | guilty | embarrassed | 1 |
| prompt_no_label_guidance | guilty | terrified | 1 |
| prompt_no_label_guidance | jealous | disappointed | 1 |
| prompt_no_label_guidance | joyful | excited | 2 |
| prompt_no_label_guidance | lonely | devastated | 1 |
| prompt_no_label_guidance | prepared | confident | 1 |
| prompt_no_label_guidance | sentimental | guilty | 1 |
| prompt_no_label_guidance | sentimental | nostalgic | 1 |
| prompt_no_label_guidance | surprised | joyful | 1 |
| prompt_no_label_guidance | terrified | trusting | 1 |
| prompt_no_label_guidance | trusting | grateful | 1 |

## 错误样例

- `full` / `ed-test-hit-104-conv-208`: expected=`angry`, predicted=`terrified`，跨情绪族错误; 输入：i just moved to this neighborhood and some dumb criminals shot one of my neighbors and ran into the woods!
- `full` / `ed-test-hit-346-conv-692`: expected=`anticipating`, predicted=`excited`，同情绪族边界错误; 输入：I am looking forward to going on vacation in a few weeks! We have a condo reserved on the beach, with fantastic ocean views. I'm ready!
- `full` / `ed-test-hit-656-conv-1313`: expected=`apprehensive`, predicted=`anxious`，同情绪族边界错误; 输入：we were 20 mins from the airport and the car broke down, overheated and would not start !
- `prompt_coarse_to_fine` / `ed-test-hit-104-conv-208`: expected=`angry`, predicted=`afraid`，跨情绪族错误; 输入：i just moved to this neighborhood and some dumb criminals shot one of my neighbors and ran into the woods!
- `prompt_coarse_to_fine` / `ed-test-hit-346-conv-692`: expected=`anticipating`, predicted=`excited`，同情绪族边界错误; 输入：I am looking forward to going on vacation in a few weeks! We have a condo reserved on the beach, with fantastic ocean views. I'm ready!
- `prompt_coarse_to_fine` / `ed-test-hit-656-conv-1313`: expected=`apprehensive`, predicted=`anxious`，同情绪族边界错误; 输入：we were 20 mins from the airport and the car broke down, overheated and would not start !
- `prompt_concise_direct` / `ed-test-hit-104-conv-208`: expected=`angry`, predicted=`afraid`，跨情绪族错误; 输入：i just moved to this neighborhood and some dumb criminals shot one of my neighbors and ran into the woods!
- `prompt_concise_direct` / `ed-test-hit-346-conv-692`: expected=`anticipating`, predicted=`prepared`，跨情绪族错误; 输入：I am looking forward to going on vacation in a few weeks! We have a condo reserved on the beach, with fantastic ocean views. I'm ready!
- `prompt_concise_direct` / `ed-test-hit-656-conv-1313`: expected=`apprehensive`, predicted=`afraid`，跨情绪族错误; 输入：we were 20 mins from the airport and the car broke down, overheated and would not start !
- `prompt_contrastive_check` / `ed-test-hit-104-conv-208`: expected=`angry`, predicted=`afraid`，跨情绪族错误; 输入：i just moved to this neighborhood and some dumb criminals shot one of my neighbors and ran into the woods!
- `prompt_contrastive_check` / `ed-test-hit-346-conv-692`: expected=`anticipating`, predicted=`excited`，同情绪族边界错误; 输入：I am looking forward to going on vacation in a few weeks! We have a condo reserved on the beach, with fantastic ocean views. I'm ready!
- `prompt_contrastive_check` / `ed-test-hit-656-conv-1313`: expected=`apprehensive`, predicted=`anxious`，同情绪族边界错误; 输入：we were 20 mins from the airport and the car broke down, overheated and would not start !
- `prompt_no_label_guidance` / `ed-test-hit-104-conv-208`: expected=`angry`, predicted=`terrified`，跨情绪族错误; 输入：i just moved to this neighborhood and some dumb criminals shot one of my neighbors and ran into the woods!
- `prompt_no_label_guidance` / `ed-test-hit-656-conv-1313`: expected=`apprehensive`, predicted=`anxious`，同情绪族边界错误; 输入：we were 20 mins from the airport and the car broke down, overheated and would not start !
- `prompt_no_label_guidance` / `ed-test-hit-34-conv-69`: expected=`caring`, predicted=`surprised`，跨情绪族错误; 输入：Well, can you tell me about your experience? I think we swapped places

## 方法

- 使用现有 `evaluate_records` 进行 case_id 匹配、Accuracy 和 Macro F1 计算。
- 配对统计按同一 case_id 重采样：精确 McNemar 检验 Accuracy，10000 次固定种子 percentile bootstrap 估计 Accuracy 和 Macro F1 差值区间。
- treatment 有效性来自逐条记录的 `input` Prompt 与同 `case_id` 的 `full` Prompt 比对；全部相同时标记为 `no_op_identical_to_full`。
- 按 `language` 和 `context_dependency` 的预定义枚举值切片，空切片显式记为 0。
- `success is not True` 单独计为调用失败，失败预测仍保留在标注分母中。

## 局限性

- 本次基准包含 64 条记录，标签来源为人工撰写情绪情境标签；高上下文依赖样本为 0 条。
- 本实验评估的是 Codex CLI Agent 执行链路，不是裸模型 API 评测；结果包含 Codex 系统指令和 Agent 运行环境的影响。
- 切片样本较少时，Accuracy 和 Macro F1 波动较大，不应单独解读。
- 调用失败同时会拉低指标，需与分类错误分开观察。
- 64 条结果只能作为探索性观察；下一步应先审计输入—标签对齐，并把 Prompt 选择迁移到 validation split。
- 执行过程说明：Each run began with a one-case smoke snapshot; matching successful provenance was reused, then the remaining 63 cases completed without failures.
