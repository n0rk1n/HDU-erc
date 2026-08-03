# Codex CLI 情绪识别消融摘要

| Run | Samples | Valid predictions | 调用失败 | Correct | Accuracy (95% CI) | Macro F1 | Family Accuracy* | Family Macro F1* | Δ Accuracy vs full (paired 95% CI) | Δ Macro F1 vs full (paired 95% CI) | McNemar exact p | Prompt identical/full | Treatment status | Provenance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| full | 64 | 64 | 0 | 36 | 56.25% (44.09%–67.71%) | 54.91% | 76.56% | 73.73% | +0.00% (+0.00%–+0.00%) | +0.00% (+0.00%–+0.00%) | 1.0000 | 64/64 | baseline | record_input_vs_full_by_case_id |
| no_dynamic_examples | 64 | 64 | 0 | 37 | 57.81% (45.61%–69.13%) | 54.06% | 78.12% | 75.13% | +1.56% (-4.69%–+9.38%) | -0.85% (-5.00%–+5.65%) | 1.0000 | 0/64 | effective_prompt_change | record_input_vs_full_by_case_id |
| zero_shot | 64 | 64 | 0 | 37 | 57.81% (45.61%–69.13%) | 54.06% | 78.12% | 75.13% | +1.56% (-4.69%–+9.38%) | -0.85% (-5.00%–+5.65%) | 1.0000 | 0/64 | effective_prompt_change | record_input_vs_full_by_case_id |
