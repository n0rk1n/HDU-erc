# Codex CLI 情绪识别消融实验报告

## 实验元数据

- branch: `codex/public-seed64-ablation-20260803`
- codex_version: `codex-cli 0.146.0`
- commit: `working-tree based on 0706ebc`
- ended_at: `2026-08-03T15:58:55+08:00`
- execution_note: `CLI 0.142.4 was rejected before valid inference because gpt-5.6-sol required a newer Codex version. After upgrading to 0.146.0, a one-case smoke run succeeded; the three effective configurations then ran in parallel with 64/64 valid predictions and zero failures each. Failed pre-upgrade snapshots were overwritten by provenance-aware reruns and are excluded.`
- model: `gpt-5.6-sol`

## 整体结果

| Run | Samples | Valid predictions | 调用失败 | Correct | Accuracy (95% CI) | Macro F1 | Family Accuracy* | Family Macro F1* | Δ Accuracy vs full (paired 95% CI) | Δ Macro F1 vs full (paired 95% CI) | McNemar exact p | Prompt identical/full | Treatment status | Provenance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| full | 64 | 64 | 0 | 36 | 56.25% (44.09%–67.71%) | 54.91% | 76.56% | 73.73% | +0.00% (+0.00%–+0.00%) | +0.00% (+0.00%–+0.00%) | 1.0000 | 64/64 | baseline | record_input_vs_full_by_case_id |
| no_dynamic_examples | 64 | 64 | 0 | 37 | 57.81% (45.61%–69.13%) | 54.06% | 78.12% | 75.13% | +1.56% (-4.69%–+9.38%) | -0.85% (-5.00%–+5.65%) | 1.0000 | 0/64 | effective_prompt_change | record_input_vs_full_by_case_id |
| zero_shot | 64 | 64 | 0 | 37 | 57.81% (45.61%–69.13%) | 54.06% | 78.12% | 75.13% | +1.56% (-4.69%–+9.38%) | -0.85% (-5.00%–+5.65%) | 1.0000 | 0/64 | effective_prompt_change | record_input_vs_full_by_case_id |

*Family 指标只用于诊断相邻标签（例如 afraid/terrified、annoyed/angry）的边界错误，
不能替代 32 类 exact Accuracy 与 Macro F1。

## 配对统计推断

| Run | full only correct | treatment only correct | McNemar exact p | Δ Accuracy paired 95% CI | Δ Macro F1 paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 0 | 0 | 1.0000 | +0.00%–+0.00% | +0.00%–+0.00% |
| no_dynamic_examples | 2 | 3 | 1.0000 | -4.69%–+9.38% | -5.00%–+5.65% |
| zero_shot | 2 | 3 | 1.0000 | -4.69%–+9.38% | -5.00%–+5.65% |

Accuracy 采用精确 McNemar 检验；差值区间使用按 case_id 配对的 percentile bootstrap（10000 次，固定种子 20260803）。
p < 0.05 才可视为拒绝“两配置准确率相同”的初步证据；Macro F1 差值仅根据配对 bootstrap 区间解读。

## 语言切片

| Run | Slice | Samples | Correct | Accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| full | zh | 0 | 0 | 0.00% | 0.00% |
| full | en | 64 | 36 | 56.25% | 54.91% |
| no_dynamic_examples | zh | 0 | 0 | 0.00% | 0.00% |
| no_dynamic_examples | en | 64 | 37 | 57.81% | 54.06% |
| zero_shot | zh | 0 | 0 | 0.00% | 0.00% |
| zero_shot | en | 64 | 37 | 57.81% | 54.06% |

## 上下文依赖切片

| Run | Slice | Samples | Correct | Accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| full | none | 64 | 36 | 56.25% | 54.91% |
| full | low | 0 | 0 | 0.00% | 0.00% |
| full | medium | 0 | 0 | 0.00% | 0.00% |
| full | high | 0 | 0 | 0.00% | 0.00% |
| no_dynamic_examples | none | 64 | 37 | 57.81% | 54.06% |
| no_dynamic_examples | low | 0 | 0 | 0.00% | 0.00% |
| no_dynamic_examples | medium | 0 | 0 | 0.00% | 0.00% |
| no_dynamic_examples | high | 0 | 0 | 0.00% | 0.00% |
| zero_shot | none | 64 | 37 | 57.81% | 54.06% |
| zero_shot | low | 0 | 0 | 0.00% | 0.00% |
| zero_shot | medium | 0 | 0 | 0.00% | 0.00% |
| zero_shot | high | 0 | 0 | 0.00% | 0.00% |

## 标签混淆

| Run | Expected | Predicted | Count |
| --- | --- | --- | ---: |
| full | afraid | terrified | 1 |
| full | angry | afraid | 1 |
| full | anticipating | excited | 1 |
| full | anxious | guilty | 1 |
| full | apprehensive | afraid | 1 |
| full | apprehensive | anxious | 1 |
| full | ashamed | guilty | 1 |
| full | caring | surprised | 1 |
| full | confident | anticipating | 1 |
| full | content | joyful | 1 |
| full | disappointed | anticipating | 1 |
| full | disgusted | annoyed | 1 |
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
| full | terrified | afraid | 1 |
| full | trusting | grateful | 1 |
| no_dynamic_examples | afraid | terrified | 1 |
| no_dynamic_examples | angry | afraid | 1 |
| no_dynamic_examples | anticipating | excited | 2 |
| no_dynamic_examples | apprehensive | afraid | 1 |
| no_dynamic_examples | apprehensive | anxious | 1 |
| no_dynamic_examples | ashamed | guilty | 1 |
| no_dynamic_examples | caring | surprised | 1 |
| no_dynamic_examples | content | joyful | 1 |
| no_dynamic_examples | disappointed | anticipating | 1 |
| no_dynamic_examples | disgusted | annoyed | 1 |
| no_dynamic_examples | furious | angry | 1 |
| no_dynamic_examples | furious | disgusted | 1 |
| no_dynamic_examples | guilty | embarrassed | 1 |
| no_dynamic_examples | guilty | terrified | 1 |
| no_dynamic_examples | jealous | disappointed | 1 |
| no_dynamic_examples | joyful | excited | 2 |
| no_dynamic_examples | lonely | devastated | 1 |
| no_dynamic_examples | prepared | confident | 1 |
| no_dynamic_examples | sad | devastated | 1 |
| no_dynamic_examples | sentimental | guilty | 1 |
| no_dynamic_examples | sentimental | nostalgic | 1 |
| no_dynamic_examples | surprised | joyful | 1 |
| no_dynamic_examples | terrified | anticipating | 1 |
| no_dynamic_examples | trusting | confident | 1 |
| no_dynamic_examples | trusting | grateful | 1 |
| zero_shot | afraid | terrified | 1 |
| zero_shot | angry | afraid | 1 |
| zero_shot | anticipating | excited | 2 |
| zero_shot | apprehensive | afraid | 1 |
| zero_shot | apprehensive | anxious | 1 |
| zero_shot | ashamed | guilty | 1 |
| zero_shot | caring | surprised | 1 |
| zero_shot | content | joyful | 1 |
| zero_shot | disappointed | anticipating | 1 |
| zero_shot | disgusted | annoyed | 1 |
| zero_shot | furious | angry | 1 |
| zero_shot | furious | disgusted | 1 |
| zero_shot | guilty | embarrassed | 1 |
| zero_shot | guilty | terrified | 1 |
| zero_shot | jealous | disappointed | 1 |
| zero_shot | joyful | excited | 2 |
| zero_shot | lonely | devastated | 1 |
| zero_shot | prepared | confident | 1 |
| zero_shot | sad | devastated | 1 |
| zero_shot | sentimental | guilty | 1 |
| zero_shot | sentimental | nostalgic | 1 |
| zero_shot | surprised | joyful | 1 |
| zero_shot | terrified | anticipating | 1 |
| zero_shot | trusting | confident | 1 |
| zero_shot | trusting | grateful | 1 |

## 错误样例

- `full` / `ed-test-hit-104-conv-208`: expected=`angry`, predicted=`afraid`，跨情绪族错误; 输入：i just moved to this neighborhood and some dumb criminals shot one of my neighbors and ran into the woods!
- `full` / `ed-test-hit-656-conv-1313`: expected=`apprehensive`, predicted=`anxious`，同情绪族边界错误; 输入：we were 20 mins from the airport and the car broke down, overheated and would not start !
- `full` / `ed-test-hit-34-conv-69`: expected=`caring`, predicted=`surprised`，跨情绪族错误; 输入：Well, can you tell me about your experience? I think we swapped places
- `no_dynamic_examples` / `ed-test-hit-104-conv-208`: expected=`angry`, predicted=`afraid`，跨情绪族错误; 输入：i just moved to this neighborhood and some dumb criminals shot one of my neighbors and ran into the woods!
- `no_dynamic_examples` / `ed-test-hit-346-conv-692`: expected=`anticipating`, predicted=`excited`，同情绪族边界错误; 输入：I am looking forward to going on vacation in a few weeks! We have a condo reserved on the beach, with fantastic ocean views. I'm ready!
- `no_dynamic_examples` / `ed-test-hit-656-conv-1313`: expected=`apprehensive`, predicted=`anxious`，同情绪族边界错误; 输入：we were 20 mins from the airport and the car broke down, overheated and would not start !
- `zero_shot` / `ed-test-hit-104-conv-208`: expected=`angry`, predicted=`afraid`，跨情绪族错误; 输入：i just moved to this neighborhood and some dumb criminals shot one of my neighbors and ran into the woods!
- `zero_shot` / `ed-test-hit-346-conv-692`: expected=`anticipating`, predicted=`excited`，同情绪族边界错误; 输入：I am looking forward to going on vacation in a few weeks! We have a condo reserved on the beach, with fantastic ocean views. I'm ready!
- `zero_shot` / `ed-test-hit-656-conv-1313`: expected=`apprehensive`, predicted=`anxious`，同情绪族边界错误; 输入：we were 20 mins from the airport and the car broke down, overheated and would not start !

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
- `zero_shot` 同时禁用 few-shot 示例和情绪历史先验，因此属于组合消融；其相对 `full` 的指标差值不能单独归因于任一组件。
- 执行过程说明：CLI 0.142.4 was rejected before valid inference because gpt-5.6-sol required a newer Codex version. After upgrading to 0.146.0, a one-case smoke run succeeded; the three effective configurations then ran in parallel with 64/64 valid predictions and zero failures each. Failed pre-upgrade snapshots were overwritten by provenance-aware reruns and are excluded.
