# Codex CLI 情绪识别消融摘要

⚠️ treatment 有效性警告：`no_emotion_history` 的输入 Prompt 均为 10/10 与 `full` 完全相同。
这些运行属于 no-op 重复对照；指标差异只能视为重复调用波动，不能归因于消融组件。

| Run | Samples | Valid predictions | 调用失败 | Correct | Accuracy | Macro F1 | Δ Accuracy vs full | Δ Macro F1 vs full | Prompt identical/full | Treatment status | Provenance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| full | 10 | 10 | 0 | 3 | 30.00% | 14.67% | +0.00% | +0.00% | 10/10 | baseline | record_input_vs_full_by_case_id |
| no_dynamic_examples | 10 | 10 | 0 | 4 | 40.00% | 21.67% | +10.00% | +7.00% | 0/10 | effective_prompt_change | record_input_vs_full_by_case_id |
| no_emotion_history | 10 | 10 | 0 | 3 | 30.00% | 15.00% | +0.00% | +0.33% | 10/10 | no_op_identical_to_full | record_input_vs_full_by_case_id |
| short_context | 10 | 10 | 0 | 3 | 30.00% | 15.00% | +0.00% | +0.33% | 8/10 | effective_prompt_change | record_input_vs_full_by_case_id |
| zero_shot | 10 | 10 | 0 | 2 | 20.00% | 10.00% | -10.00% | -4.67% | 0/10 | effective_prompt_change | record_input_vs_full_by_case_id |
