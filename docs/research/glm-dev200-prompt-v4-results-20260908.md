# 历史问题复盘后的 Prompt v4 对照结果

生成时间（UTC）：2026-09-08T08:50:36.758369Z。实际运行代码：`bb72adc226c5e8b4af75d002b70cda25d83a3a64` / `bb72adc226c5e8b4af75d002b70cda25d83a3a64`。

## 执行范围

固定同一 dev200，重新运行 v3 控制组和 v4 候选组，各 200 条。模型、标签定义、families、contrastive-v1 检索、候选 50、入选 4 条示例、解析器和原生多标签评分不变。两组全部完成后才评分；本轮只冻结一个 v4，不按当前输出再调参。

历史错误已用于本轮开发，不能称为独立未见验证。新增规则同时参考 train 的原始边界样本；prompt 不包含 dev 原文/真值对，官方 test 未使用。

## 主指标

| 指标 | 本轮 v3 重跑 | v4 | 差值（百分点） |
|---|---:|---:|---:|
| Micro-F1 | 42.55% | 39.39% | -3.17 |
| Macro-F1（28 类） | 31.19% | 30.71% | -0.48 |
| 标签集合完全匹配率 | 32.50% | 21.50% | -11.00 |
| Micro-precision | 42.02% | 35.40% | -6.62 |
| Micro-recall | 43.10% | 44.40% | +1.29 |

Micro-F1 差值的 95% 配对 bootstrap 区间：[-7.96, +1.54] 个百分点。200 对样本，seed=42，1000 次重采样。

本轮 v4 未提高主指标 Micro-F1；保留负面结果，不将新增规则或测试通过当作模型改进。

本轮候选不提升为默认版本。realization、caring 和 disappointment 的召回局部回升，但总体误报增幅更大；这只能解释整个 v4 组合的观测效果，不能分离每条提示规则的因果作用。决策工件：`df39aaf3-9235-4922-b79f-dee193d2f121`。


标签集合不完全匹配→完全匹配 4 条，完全匹配→不完全匹配 26 条。
标签级计数：v3 TP/FP/FN = 100/138/132；v4 = 103/188/129。各指标共同解释结果，不只看其中上升的一项。

## 前轮问题与本轮改动

完整复盘见 [低分问题总复盘](low-accuracy-review-and-prompt-v4.md)，包括实际误报/漏报、检索局限、评价口径、类别不均衡、运行波动、开发集重复使用，以及已排除的接口/存储/引号读取因素。

- 撤掉 v3 的训练频率和最小标签集合倾向，同时检查多报与漏报。
- 补足 disappointment、realization、approval、caring 的隐含或普通表达，不要求显式情绪词或独立句子。
- 保留对 disapproval、neutral、幽默/讽刺误读的限制，但不把所有负面反应或不确定情绪归为某个兜底标签。
- 先扫描输入，再参考例子；实际检索与入选例子保持一致，尚未证明检索问题已解决。

## 历史结果与运行变化

| 历史组 | Micro-F1 | Macro-F1 | 完全匹配率 |
|---|---:|---:|---:|
| v1_ranked | 39.53% | 28.88% | 25.50% |
| v2_first | 40.62% | 30.22% | 24.50% |
| v2_repeat | 38.08% | 29.05% | 20.50% |
| v3 | 39.32% | 28.60% | 27.00% |

前轮 v3 与本轮 v3 的 200 个请求体哈希全部相同；Micro-F1 为 39.32% 与 42.55%。同参请求不保证确定性输出，不臆测供应商内部变化原因。变异核验工件：`468141e5-9e2c-4e72-b990-a9bf1f395a2f`。
前轮两次 v2 也有 77/200 条标签集合不同。本轮以重新运行的 v3 为控制组，历史各轮全部保留，不挑选最有利的历史基线。

离线常量 neutral 参照：完全匹配率 30.00%，Micro-F1 28.24%。它不调用模型，不是候选方法；用于说明只看完全匹配率容易被类别比例误导。

## 补充指标与错误分布

| 指标 | v3 | v4 |
|---|---:|---:|
| 至少一个细类命中 | 47.50% | 47.50% |
| 大类集合完全匹配 | 55.00% | 44.50% |
| 至少一个大类命中 | 62.00% | 61.50% |
| 大类 Micro-F1 | 59.29% | 55.63% |
| 平均预测标签数 | 1.19 | 1.455 |
| 60 条仅 neutral 中漏判条数 | 31 | 39 |
| 细类无交集条数 | 105 | 105 |

| 标签 | 真值数 | v3 TP/FP/FN | v4 TP/FP/FN | v3 F1 | v4 F1 |
|---|---:|---|---|---:|---:|
| admiration | 29 | 11/3/18 | 14/3/15 | 51.16% | 60.87% |
| amusement | 10 | 7/13/3 | 8/19/2 | 46.67% | 43.24% |
| anger | 9 | 2/1/7 | 2/1/7 | 33.33% | 33.33% |
| annoyance | 13 | 5/13/8 | 4/18/9 | 32.26% | 22.86% |
| approval | 17 | 8/10/9 | 8/20/9 | 45.71% | 35.56% |
| caring | 7 | 2/4/5 | 5/9/2 | 30.77% | 47.62% |
| confusion | 3 | 0/0/3 | 0/0/3 | 0.00% | 0.00% |
| curiosity | 7 | 3/3/4 | 3/3/4 | 46.15% | 46.15% |
| desire | 2 | 2/1/0 | 2/3/0 | 80.00% | 57.14% |
| disappointment | 9 | 2/0/7 | 3/2/6 | 36.36% | 42.86% |
| disapproval | 7 | 2/35/5 | 2/45/5 | 9.09% | 7.41% |
| disgust | 3 | 1/3/2 | 2/3/1 | 28.57% | 50.00% |
| embarrassment | 0 | 0/0/0 | 0/0/0 | 0.00% | 0.00% |
| excitement | 1 | 0/2/1 | 0/2/1 | 0.00% | 0.00% |
| fear | 4 | 1/0/3 | 0/0/4 | 40.00% | 0.00% |
| gratitude | 13 | 12/1/1 | 11/1/2 | 92.31% | 88.00% |
| grief | 2 | 0/0/2 | 0/1/2 | 0.00% | 0.00% |
| joy | 6 | 3/4/3 | 3/3/3 | 46.15% | 50.00% |
| love | 4 | 1/5/3 | 2/9/2 | 20.00% | 26.67% |
| nervousness | 0 | 0/0/0 | 0/4/0 | 0.00% | 0.00% |
| optimism | 9 | 3/1/6 | 3/2/6 | 46.15% | 42.86% |
| pride | 0 | 0/1/0 | 0/1/0 | 0.00% | 0.00% |
| realization | 5 | 0/0/5 | 3/4/2 | 0.00% | 50.00% |
| relief | 1 | 0/1/1 | 0/2/1 | 0.00% | 0.00% |
| remorse | 2 | 1/0/1 | 1/2/1 | 66.67% | 40.00% |
| sadness | 4 | 2/4/2 | 2/6/2 | 40.00% | 33.33% |
| surprise | 4 | 2/7/2 | 3/9/1 | 30.77% | 37.50% |
| neutral | 61 | 30/26/31 | 22/16/39 | 51.28% | 44.44% |

无真值支持的类别：embarrassment, nervousness, pride。Macro-F1 的零分母按 0；本切片无法稳定估计稀有类。

## 调用代价

| 项目 | v3 | v4 |
|---|---:|---:|
| 成功样本 | 200 | 200 |
| 失败样本 | 0 | 0 |
| 调用数（含重试） | 200 | 200 |
| 重试 | 0 | 0 |
| 输入 tokens | 424373 | 414173 |
| 输出 tokens | 3607 | 6346 |
| reasoning tokens | 1924 | 4434 |
| 缓存输入 tokens | 356864 | 356608 |
| 运行秒数（不含冻结） | 216.179511 | 231.161238 |
| 费用未知调用数 | 200 | 200 |

输入 tokens 相对变化：-2.40%。未有核实的部署单价，费用为 NULL，不当作零。
固定 GLM-5.3、temperature=0、thinking enabled、reasoning_effort low、max_tokens=8192、context_tokens=32768、并发4。服务未提供可核实的权重修订号。实际返回模型：{"control": {"zhipu/glm-5.3": 200}, "candidate": {"zhipu/glm-5.3": 200}}。两组顺序执行，缓存/耗时不可直接归因为算法收益。

## 案例

### 由错变对

| source_id | 原文 | 真值 | v3 | v4 |
|---|---|---|---|---|
| edv4zav | [NAME] just stopped getting the ball in the 4th quarter and stood in the corner | neutral | annoyance | neutral |
| eebg32y | didn't work anyway | disappointment | annoyance | disappointment |
| eez9zmx | The stress of electoral politics seems to be incredibly bad for one’s health. | realization | neutral | realization |
| ef5cueg | I f a picture and not realizing it until it bursts. [NAME]. | realization | neutral | realization |

### 由对变错

| source_id | 原文 | 真值 | v3 | v4 |
|---|---|---|---|---|
| ed01s0u | Another way to do it without screaming at your teammates! | neutral | neutral | approval, caring |
| ed04ap3 | We'll worry about it tomorrow | neutral | neutral | caring |
| ed3a4kd | These shows are scripted | neutral | neutral | realization |
| ed3kols | Thanks made feel a bit better | gratitude | gratitude | gratitude, relief |
| ed6gmuo | Really? Thanks for the correction. | gratitude | gratitude | gratitude, surprise |

## 数据与核验

- SQLite：`data/research/experiments.sqlite3`；完整请求/响应/检索轨迹在 `data/research/artifacts/`。
- 成对案例：`data/research/exports/glm-dev200-prompt-v4-paired-cases-20260908.jsonl`。
- 标准导出：`data/research/exports/glm-dev200-prompt-v4-control-20260908/` 和 `glm-dev200-prompt-v4-candidate-20260908/`。
- 完整分析：`data/research/train_prompt_v4_analysis.json`；执行状态/日志：`train_prompt_v4_state.json` / `train_prompt_v4_execution.log`。
- 可恢复备份：`data/research/backups/train-prompt-v4-final-20260908/`；完整性验证另存 verification_report 工件。

独立核验：`{"requests_verified": 400, "native_truths_verified": 400, "selected_train_examples_verified": 1600, "raw_predictions_verified": 400, "all_gold_from_original_tsv": true, "test_split_used": false, "paired_examples_identical": 200, "paired_requests_only_instruction_differs": 200}`。200 对实际请求除 instruction 外相同；200 对入选示例的 ID、顺序、全文、标签和检索分数一致。

| 记录 | ID |
|---|---|
| control_run_id | `6f0e9a2d-cc6d-4afb-b7e3-99a2aff24bf8` |
| candidate_run_id | `f9d0802a-c0fe-4327-ac23-6119c8838b0f` |
| control_evaluation_id | `82acf19b-0214-47df-8754-253ba5056da5` |
| candidate_evaluation_id | `d038126f-7b6c-476e-b486-588d81f64570` |
| comparison_id | `bfdebf7a-2e39-4cae-bb4c-a897701c60cc` |
| analysis_artifact_id | `41ce6d6b-951a-40fe-82b1-0903fc22161d` |
| paired_details_artifact_id | `c7bafd22-71dc-4403-bd36-d3bfcae62f5b` |
| development_basis_artifact_id | `a519b66c-e87e-4271-baa4-29cb06c8adea` |
