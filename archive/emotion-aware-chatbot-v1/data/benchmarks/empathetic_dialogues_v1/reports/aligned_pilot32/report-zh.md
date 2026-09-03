# 情绪识别消融实验报告

## 实验元数据

- commit: `working-tree based on 95117d3`
- ended_at: `2026-07-27`
- execution_note: `首次受限环境导致 96/96 初始化失败；确认未进入模型后，以相同参数在可正常初始化的环境重跑。只执行实际改变 Prompt 的 3 组；no_emotion_history 和 short_context 经预检均与 full 相同，故未调用。`
- model: `gpt-5.6-sol`
- started_at: `2026-07-27`

## 整体结果

| Run | Samples | Valid predictions | 调用失败 | Correct | Accuracy (95% CI) | Macro F1 | Family Accuracy* | Family Macro F1* | Δ Accuracy vs full | Δ Macro F1 vs full | Prompt identical/full | Treatment status | Provenance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| full | 32 | 32 | 0 | 19 | 59.38% (42.26%–74.48%) | 50.52% | 75.00% | 68.83% | +0.00% | +0.00% | 32/32 | baseline | record_input_vs_full_by_case_id |
| no_dynamic_examples | 32 | 32 | 0 | 18 | 56.25% (39.33%–71.83%) | 46.88% | 71.88% | 60.74% | -3.12% | -3.65% | 0/32 | effective_prompt_change | record_input_vs_full_by_case_id |
| zero_shot | 32 | 32 | 0 | 18 | 56.25% (39.33%–71.83%) | 46.35% | 71.88% | 60.74% | -3.12% | -4.17% | 0/32 | effective_prompt_change | record_input_vs_full_by_case_id |

*Family 指标只用于诊断相邻标签（例如 afraid/terrified、annoyed/angry）的边界错误，
不能替代 32 类 exact Accuracy 与 Macro F1。

## 语言切片

| Run | Slice | Samples | Correct | Accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| full | zh | 0 | 0 | 0.00% | 0.00% |
| full | en | 32 | 19 | 59.38% | 50.52% |
| no_dynamic_examples | zh | 0 | 0 | 0.00% | 0.00% |
| no_dynamic_examples | en | 32 | 18 | 56.25% | 46.88% |
| zero_shot | zh | 0 | 0 | 0.00% | 0.00% |
| zero_shot | en | 32 | 18 | 56.25% | 46.35% |

## 上下文依赖切片

| Run | Slice | Samples | Correct | Accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| full | none | 32 | 19 | 59.38% | 50.52% |
| full | low | 0 | 0 | 0.00% | 0.00% |
| full | medium | 0 | 0 | 0.00% | 0.00% |
| full | high | 0 | 0 | 0.00% | 0.00% |
| no_dynamic_examples | none | 32 | 18 | 56.25% | 46.88% |
| no_dynamic_examples | low | 0 | 0 | 0.00% | 0.00% |
| no_dynamic_examples | medium | 0 | 0 | 0.00% | 0.00% |
| no_dynamic_examples | high | 0 | 0 | 0.00% | 0.00% |
| zero_shot | none | 32 | 18 | 56.25% | 46.35% |
| zero_shot | low | 0 | 0 | 0.00% | 0.00% |
| zero_shot | medium | 0 | 0 | 0.00% | 0.00% |
| zero_shot | high | 0 | 0 | 0.00% | 0.00% |

## 标签混淆

| Run | Expected | Predicted | Count |
| --- | --- | --- | ---: |
| full | angry | afraid | 1 |
| full | anticipating | excited | 1 |
| full | apprehensive | anxious | 1 |
| full | caring | surprised | 1 |
| full | faithful | guilty | 1 |
| full | furious | angry | 1 |
| full | guilty | terrified | 1 |
| full | jealous | disappointed | 1 |
| full | joyful | excited | 1 |
| full | lonely | devastated | 1 |
| full | sentimental | nostalgic | 1 |
| full | surprised | joyful | 1 |
| full | trusting | confident | 1 |
| no_dynamic_examples | angry | terrified | 1 |
| no_dynamic_examples | anticipating | excited | 1 |
| no_dynamic_examples | apprehensive | anxious | 1 |
| no_dynamic_examples | caring | surprised | 1 |
| no_dynamic_examples | disgusted | annoyed | 1 |
| no_dynamic_examples | faithful | guilty | 1 |
| no_dynamic_examples | furious | angry | 1 |
| no_dynamic_examples | guilty | terrified | 1 |
| no_dynamic_examples | jealous | disappointed | 1 |
| no_dynamic_examples | joyful | excited | 1 |
| no_dynamic_examples | lonely | devastated | 1 |
| no_dynamic_examples | sentimental | nostalgic | 1 |
| no_dynamic_examples | surprised | joyful | 1 |
| no_dynamic_examples | trusting | confident | 1 |
| zero_shot | angry | afraid | 1 |
| zero_shot | anticipating | excited | 1 |
| zero_shot | apprehensive | anxious | 1 |
| zero_shot | caring | surprised | 1 |
| zero_shot | disgusted | annoyed | 1 |
| zero_shot | faithful | guilty | 1 |
| zero_shot | furious | angry | 1 |
| zero_shot | guilty | terrified | 1 |
| zero_shot | jealous | disappointed | 1 |
| zero_shot | joyful | excited | 1 |
| zero_shot | lonely | devastated | 1 |
| zero_shot | sentimental | nostalgic | 1 |
| zero_shot | surprised | joyful | 1 |
| zero_shot | trusting | confident | 1 |

## 错误样例

- `full` / `ed-test-hit-104-conv-208`: expected=`angry`, predicted=`afraid`，跨情绪族错误; 输入：i just moved to this neighborhood and some dumb criminals shot one of my neighbors and ran into the woods!
- `full` / `ed-test-hit-346-conv-692`: expected=`anticipating`, predicted=`excited`，同情绪族边界错误; 输入：I am looking forward to going on vacation in a few weeks! We have a condo reserved on the beach, with fantastic ocean views. I'm ready!
- `full` / `ed-test-hit-656-conv-1313`: expected=`apprehensive`, predicted=`anxious`，同情绪族边界错误; 输入：we were 20 mins from the airport and the car broke down, overheated and would not start !
- `no_dynamic_examples` / `ed-test-hit-104-conv-208`: expected=`angry`, predicted=`terrified`，跨情绪族错误; 输入：i just moved to this neighborhood and some dumb criminals shot one of my neighbors and ran into the woods!
- `no_dynamic_examples` / `ed-test-hit-346-conv-692`: expected=`anticipating`, predicted=`excited`，同情绪族边界错误; 输入：I am looking forward to going on vacation in a few weeks! We have a condo reserved on the beach, with fantastic ocean views. I'm ready!
- `no_dynamic_examples` / `ed-test-hit-656-conv-1313`: expected=`apprehensive`, predicted=`anxious`，同情绪族边界错误; 输入：we were 20 mins from the airport and the car broke down, overheated and would not start !
- `zero_shot` / `ed-test-hit-104-conv-208`: expected=`angry`, predicted=`afraid`，跨情绪族错误; 输入：i just moved to this neighborhood and some dumb criminals shot one of my neighbors and ran into the woods!
- `zero_shot` / `ed-test-hit-346-conv-692`: expected=`anticipating`, predicted=`excited`，同情绪族边界错误; 输入：I am looking forward to going on vacation in a few weeks! We have a condo reserved on the beach, with fantastic ocean views. I'm ready!
- `zero_shot` / `ed-test-hit-656-conv-1313`: expected=`apprehensive`, predicted=`anxious`，同情绪族边界错误; 输入：we were 20 mins from the airport and the car broke down, overheated and would not start !

## 方法

- 使用现有 `evaluate_records` 进行 case_id 匹配、Accuracy 和 Macro F1 计算。
- treatment 有效性来自逐条记录的 `input` Prompt 与同 `case_id` 的 `full` Prompt 比对；全部相同时标记为 `no_op_identical_to_full`。
- 按 `language` 和 `context_dependency` 的预定义枚举值切片，空切片显式记为 0。
- `success is not True` 单独计为调用失败，失败预测仍保留在标注分母中。

## 局限性

- 本次基准包含 32 条记录，标签来源为人工撰写情绪情境标签；高上下文依赖样本为 0 条。
- 本实验评估的是隔离 Agent 执行链路，不是裸模型 API 评测；结果包含 Agent 系统指令和运行环境的影响。
- 切片样本较少时，Accuracy 和 Macro F1 波动较大，不应单独解读。
- 调用失败同时会拉低指标，需与分类错误分开观察。
- `zero_shot` 同时禁用 few-shot 示例和情绪历史先验，因此属于组合消融；其相对 `full` 的指标差值不能单独归因于任一组件。
- 执行过程说明：首次受限环境导致 96/96 初始化失败；确认未进入模型后，以相同参数在可正常初始化的环境重跑。只执行实际改变 Prompt 的 3 组；no_emotion_history 和 short_context 经预检均与 full 相同，故未调用。
