# Codex CLI 情绪识别消融实验报告

## 实验元数据

- branch: `codex/real-emotion-dataset-20260727`
- codex_version: `0.142.4`
- commit: `b594f37bcba63386a67236549e989b1996e3dd92`
- execution_note: `EmpatheticDialogues human-authored balanced seed prefix pilot: 10 records covering 5 labels; all five ablation configurations completed with no invocation failures.`
- model: `gpt-5.6-sol`

## 结论有效性警告

`no_emotion_history` 的输入 Prompt 均为 10/10 与 `full` 完全相同。
该运行是 no-op 重复对照，其指标差异是重复调用波动，不是消融效果；不能据此归因情绪历史或上下文长度的组件贡献。

## 整体结果

| Run | Samples | Valid predictions | 调用失败 | Correct | Accuracy | Macro F1 | Δ Accuracy vs full | Δ Macro F1 vs full | Prompt identical/full | Treatment status | Provenance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| full | 10 | 10 | 0 | 3 | 30.00% | 14.67% | +0.00% | +0.00% | 10/10 | baseline | record_input_vs_full_by_case_id |
| no_dynamic_examples | 10 | 10 | 0 | 4 | 40.00% | 21.67% | +10.00% | +7.00% | 0/10 | effective_prompt_change | record_input_vs_full_by_case_id |
| no_emotion_history | 10 | 10 | 0 | 3 | 30.00% | 15.00% | +0.00% | +0.33% | 10/10 | no_op_identical_to_full | record_input_vs_full_by_case_id |
| short_context | 10 | 10 | 0 | 3 | 30.00% | 15.00% | +0.00% | +0.33% | 8/10 | effective_prompt_change | record_input_vs_full_by_case_id |
| zero_shot | 10 | 10 | 0 | 2 | 20.00% | 10.00% | -10.00% | -4.67% | 0/10 | effective_prompt_change | record_input_vs_full_by_case_id |

## 语言切片

| Run | Slice | Samples | Correct | Accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| full | zh | 0 | 0 | 0.00% | 0.00% |
| full | en | 10 | 3 | 30.00% | 14.67% |
| no_dynamic_examples | zh | 0 | 0 | 0.00% | 0.00% |
| no_dynamic_examples | en | 10 | 4 | 40.00% | 21.67% |
| no_emotion_history | zh | 0 | 0 | 0.00% | 0.00% |
| no_emotion_history | en | 10 | 3 | 30.00% | 15.00% |
| short_context | zh | 0 | 0 | 0.00% | 0.00% |
| short_context | en | 10 | 3 | 30.00% | 15.00% |
| zero_shot | zh | 0 | 0 | 0.00% | 0.00% |
| zero_shot | en | 10 | 2 | 20.00% | 10.00% |

## 上下文依赖切片

| Run | Slice | Samples | Correct | Accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| full | none | 0 | 0 | 0.00% | 0.00% |
| full | low | 0 | 0 | 0.00% | 0.00% |
| full | medium | 8 | 3 | 37.50% | 18.52% |
| full | high | 2 | 0 | 0.00% | 0.00% |
| no_dynamic_examples | none | 0 | 0 | 0.00% | 0.00% |
| no_dynamic_examples | low | 0 | 0 | 0.00% | 0.00% |
| no_dynamic_examples | medium | 8 | 4 | 50.00% | 27.08% |
| no_dynamic_examples | high | 2 | 0 | 0.00% | 0.00% |
| no_emotion_history | none | 0 | 0 | 0.00% | 0.00% |
| no_emotion_history | low | 0 | 0 | 0.00% | 0.00% |
| no_emotion_history | medium | 8 | 3 | 37.50% | 18.75% |
| no_emotion_history | high | 2 | 0 | 0.00% | 0.00% |
| short_context | none | 0 | 0 | 0.00% | 0.00% |
| short_context | low | 0 | 0 | 0.00% | 0.00% |
| short_context | medium | 8 | 3 | 37.50% | 18.75% |
| short_context | high | 2 | 0 | 0.00% | 0.00% |
| zero_shot | none | 0 | 0 | 0.00% | 0.00% |
| zero_shot | low | 0 | 0 | 0.00% | 0.00% |
| zero_shot | medium | 8 | 2 | 25.00% | 11.11% |
| zero_shot | high | 2 | 0 | 0.00% | 0.00% |

## 标签混淆

| Run | Expected | Predicted | Count |
| --- | --- | --- | ---: |
| full | afraid | nostalgic | 1 |
| full | afraid | terrified | 1 |
| full | angry | afraid | 1 |
| full | angry | annoyed | 1 |
| full | anticipating | excited | 1 |
| full | anxious | caring | 1 |
| full | anxious | guilty | 1 |
| no_dynamic_examples | afraid | nostalgic | 1 |
| no_dynamic_examples | angry | afraid | 1 |
| no_dynamic_examples | angry | content | 1 |
| no_dynamic_examples | anticipating | excited | 1 |
| no_dynamic_examples | anxious | caring | 1 |
| no_dynamic_examples | anxious | guilty | 1 |
| no_emotion_history | afraid | terrified | 1 |
| no_emotion_history | angry | afraid | 1 |
| no_emotion_history | angry | apprehensive | 1 |
| no_emotion_history | anticipating | excited | 2 |
| no_emotion_history | anxious | caring | 1 |
| no_emotion_history | anxious | guilty | 1 |
| short_context | afraid | terrified | 1 |
| short_context | angry | afraid | 1 |
| short_context | angry | content | 1 |
| short_context | anticipating | excited | 2 |
| short_context | anxious | caring | 1 |
| short_context | anxious | guilty | 1 |
| zero_shot | afraid | apprehensive | 1 |
| zero_shot | afraid | terrified | 1 |
| zero_shot | angry | afraid | 1 |
| zero_shot | angry | apprehensive | 1 |
| zero_shot | anticipating | excited | 2 |
| zero_shot | anxious | caring | 1 |
| zero_shot | anxious | guilty | 1 |

## 错误样例

- `full` / `ed-test-hit-284-conv-569`: expected=`afraid`, predicted=`terrified`; 输入：Well I knew he was following us and so I asked the manager to walk us out to the car and when he did we saw the guy come out again from between the two buildings b/c he was waiting on us.  so scary!
- `full` / `ed-test-hit-396-conv-792`: expected=`afraid`, predicted=`nostalgic`; 输入：And yet I still felt as if I had to see the sequels when they came out too.
- `full` / `ed-test-hit-104-conv-208`: expected=`angry`, predicted=`afraid`; 输入：I do! I want to be able to protect my son
- `no_dynamic_examples` / `ed-test-hit-396-conv-792`: expected=`afraid`, predicted=`nostalgic`; 输入：And yet I still felt as if I had to see the sequels when they came out too.
- `no_dynamic_examples` / `ed-test-hit-104-conv-208`: expected=`angry`, predicted=`afraid`; 输入：I do! I want to be able to protect my son
- `no_dynamic_examples` / `ed-test-hit-134-conv-269`: expected=`angry`, predicted=`content`; 输入：Good idea. i don't like social media that much. It can get you in to trouble fast, even when expressing your opinions.
- `no_emotion_history` / `ed-test-hit-284-conv-569`: expected=`afraid`, predicted=`terrified`; 输入：Well I knew he was following us and so I asked the manager to walk us out to the car and when he did we saw the guy come out again from between the two buildings b/c he was waiting on us.  so scary!
- `no_emotion_history` / `ed-test-hit-104-conv-208`: expected=`angry`, predicted=`afraid`; 输入：I do! I want to be able to protect my son
- `no_emotion_history` / `ed-test-hit-134-conv-269`: expected=`angry`, predicted=`apprehensive`; 输入：Good idea. i don't like social media that much. It can get you in to trouble fast, even when expressing your opinions.
- `short_context` / `ed-test-hit-284-conv-569`: expected=`afraid`, predicted=`terrified`; 输入：Well I knew he was following us and so I asked the manager to walk us out to the car and when he did we saw the guy come out again from between the two buildings b/c he was waiting on us.  so scary!
- `short_context` / `ed-test-hit-104-conv-208`: expected=`angry`, predicted=`afraid`; 输入：I do! I want to be able to protect my son
- `short_context` / `ed-test-hit-134-conv-269`: expected=`angry`, predicted=`content`; 输入：Good idea. i don't like social media that much. It can get you in to trouble fast, even when expressing your opinions.
- `zero_shot` / `ed-test-hit-284-conv-569`: expected=`afraid`, predicted=`terrified`; 输入：Well I knew he was following us and so I asked the manager to walk us out to the car and when he did we saw the guy come out again from between the two buildings b/c he was waiting on us.  so scary!
- `zero_shot` / `ed-test-hit-396-conv-792`: expected=`afraid`, predicted=`apprehensive`; 输入：And yet I still felt as if I had to see the sequels when they came out too.
- `zero_shot` / `ed-test-hit-104-conv-208`: expected=`angry`, predicted=`afraid`; 输入：I do! I want to be able to protect my son

## 方法

- 使用现有 `evaluate_records` 进行 case_id 匹配、Accuracy 和 Macro F1 计算。
- treatment 有效性来自逐条记录的 `input` Prompt 与同 `case_id` 的 `full` Prompt 比对；全部相同时标记为 `no_op_identical_to_full`。
- 按 `language` 和 `context_dependency` 的预定义枚举值切片，空切片显式记为 0。
- `success is not True` 单独计为调用失败，失败预测仍保留在标注分母中。

## 局限性

- 本次基准包含 10 条记录，标签来源为人工撰写情绪情境标签；高上下文依赖样本为 2 条。
- 本实验评估的是 Codex CLI Agent 执行链路，不是裸模型 API 评测；结果包含 Codex 系统指令和 Agent 运行环境的影响。
- 切片样本较少时，Accuracy 和 Macro F1 波动较大，不应单独解读。
- 调用失败同时会拉低指标，需与分类错误分开观察。
- `zero_shot` 同时禁用 few-shot 示例和情绪历史先验，因此属于组合消融；其相对 `full` 的指标差值不能单独归因于任一组件。
- 执行过程说明：EmpatheticDialogues human-authored balanced seed prefix pilot: 10 records covering 5 labels; all five ablation configurations completed with no invocation failures.
