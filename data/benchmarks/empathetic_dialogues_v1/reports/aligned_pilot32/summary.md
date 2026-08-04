# 情绪识别消融摘要

| Run | Samples | Valid predictions | 调用失败 | Correct | Accuracy (95% CI) | Macro F1 | Family Accuracy* | Family Macro F1* | Δ Accuracy vs full | Δ Macro F1 vs full | Prompt identical/full | Treatment status | Provenance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| full | 32 | 32 | 0 | 19 | 59.38% (42.26%–74.48%) | 50.52% | 75.00% | 68.83% | +0.00% | +0.00% | 32/32 | baseline | record_input_vs_full_by_case_id |
| no_dynamic_examples | 32 | 32 | 0 | 18 | 56.25% (39.33%–71.83%) | 46.88% | 71.88% | 60.74% | -3.12% | -3.65% | 0/32 | effective_prompt_change | record_input_vs_full_by_case_id |
| zero_shot | 32 | 32 | 0 | 18 | 56.25% (39.33%–71.83%) | 46.35% | 71.88% | 60.74% | -3.12% | -4.17% | 0/32 | effective_prompt_change | record_input_vs_full_by_case_id |
