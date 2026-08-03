# Codex CLI 情绪识别 Prompt 多版本实验摘要

| Run | Samples | Valid predictions | 调用失败 | Correct | Accuracy (95% CI) | Macro F1 | Family Accuracy* | Family Macro F1* | Δ Accuracy vs full (paired 95% CI) | Δ Macro F1 vs full (paired 95% CI) | McNemar exact p | Prompt identical/full | Treatment status | Provenance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| full | 64 | 64 | 0 | 36 | 56.25% (44.09%–67.71%) | 55.31% | 76.56% | 75.17% | +0.00% (+0.00%–+0.00%) | +0.00% (+0.00%–+0.00%) | 1.0000 | 64/64 | baseline | record_input_vs_full_by_case_id |
| prompt_coarse_to_fine | 64 | 64 | 0 | 38 | 59.38% (47.15%–70.54%) | 56.56% | 78.12% | 75.16% | +3.12% (-3.12%–+9.38%) | +1.25% (-3.10%–+7.05%) | 0.6250 | 0/64 | effective_prompt_change | record_input_vs_full_by_case_id |
| prompt_concise_direct | 64 | 64 | 0 | 37 | 57.81% (45.61%–69.13%) | 54.91% | 75.00% | 73.94% | +1.56% (-3.12%–+7.81%) | -0.40% (-3.59%–+5.00%) | 1.0000 | 0/64 | effective_prompt_change | record_input_vs_full_by_case_id |
| prompt_contrastive_check | 64 | 64 | 0 | 36 | 56.25% (44.09%–67.71%) | 53.65% | 75.00% | 73.01% | +0.00% (-6.25%–+6.25%) | -1.67% (-5.97%–+4.55%) | 1.0000 | 0/64 | effective_prompt_change | record_input_vs_full_by_case_id |
| prompt_no_label_guidance | 64 | 64 | 0 | 38 | 59.38% (47.15%–70.54%) | 57.92% | 76.56% | 75.07% | +3.12% (-3.12%–+9.38%) | +2.60% (-1.31%–+7.39%) | 0.6250 | 0/64 | effective_prompt_change | record_input_vs_full_by_case_id |
