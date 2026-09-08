# GLM-5.3：训练集驱动 Prompt v3 的对照验证

报告生成时间（UTC）：2026-09-08T07:09:36.438330Z。两组运行代码提交：`879a6f674eeff55fe673b594b9668cccb709b423` / `879a6f674eeff55fe673b594b9668cccb709b423`。

## 验证范围

基于 43,410 条 GoEmotions 训练评论的类别、共现与词形切片统计及原始样本核验，冻结一个 v3 候选；重新运行 v2 控制组与 v3 候选组，各 200 条。两组固定模型参数、数据、taxonomy、contrastive-v1 检索、4 条入选示例及原生评分规则，只改变 prompt。

**这轮是开发迭代。** 前轮 dev200 错误已被观察并用于提出优化问题；本轮规则依据来自 train，但不能把重复使用的 dev200 视为独立未见验证。官方 test 未使用。没有按本轮得分继续调参或挑选多个候选。

## 主指标

| 指标 | 本轮 v2 控制组 | v3 候选组 | 差值（百分点） |
|---|---:|---:|---:|
| 细类 Micro-F1 | 38.08% | 39.32% | +1.25 |
| 细类 Macro-F1（28 类） | 29.05% | 28.60% | -0.45 |
| 精确标签集合匹配率 | 20.50% | 27.00% | +6.50 |

Micro-F1 配对 bootstrap 差值的 95% 区间：[-3.31, +5.59] 个百分点，1000 次重采样，seed=42。

v3 在本开发切片的 Micro-F1 高于本轮 v2。区间包含 0，尚不能确认改进稳定存在。

新版主要减少了多报：误报标签由 189 降到 148，但命中标签由 99 降到 93、漏报由 133 增到 139，至少一个细类命中率由 47% 降到 44%。因此不能将它认定为整体更好的版本，也不自动替换默认 prompt。

额外复核发现：前轮 v2 与本轮 v2 的 200 个原始请求体哈希全部相同，但 Micro-F1 为 40.62% 与 38.08%。这说明固定请求参数并不保证供应商输出完全一致；本轮 v3 的 39.32% 也低于前轮 v2 的 40.62%。本轮比较采用重新执行的控制组，不能只挑选最有利的历史基线。变异证据工件：`8aeee025-c736-4ed6-a53f-01a1a344b628`。


标签集合不完全匹配→完全匹配 21 条，完全正确→错误 8 条。

## 修改依据与规则

- 36,308/43,410 条训练评论为单标签：区分多个备选解释与真正的情绪共现，减少同义或强弱标签堆叠；保留多标签与 neutral 共现。
- 负面词切片有 318 条仅 neutral，脏话切片有 250 条仅 neutral：按实际表达和态度判断，避免将 disapproval 用作负面文本兜底。
- 问号切片同时含 curiosity、confusion 和 neutral：允许信息询问表达好奇，区分理解困难和反问。
- 训练原始标注中存在同情性 sorry → remorse：不要求每次都承认自身过错。
- 明确检索例子不是候选标签范围，也不凭词面相似复制情绪；实际检索算法与例子全文不变。

训练依据、source_id 和方法局限见 [训练审计说明](train-prompt-v3.md)。完整 prompt 在 `emotion_lab/taxonomy.py`，实际使用的版本和请求均入库。

## 补充诊断（不替代主指标）

| 指标 | v2 | v3 |
|---|---:|---:|
| 细类至少一个命中 | 47.00% | 44.00% |
| 大类集合完全匹配 | 46.00% | 47.50% |
| 大类至少一个命中 | 61.00% | 57.50% |
| 大类 Micro-F1 | 55.08% | 53.95% |

| 诊断量 | v2 | v3 |
|---|---:|---:|
| 每条平均预测标签数 | 1.44 | 1.205 |
| 60 条仅 neutral 中漏掉 neutral 的条数 | 38 | 36 |
| 细类完全无交集条数 | 106 | 112 |
| 细类部分命中条数 | 53 | 34 |
| disapproval 误报数 | 55 | 33 |
| annoyance 误报数 | 23 | 18 |
| neutral 误报数 | 18 | 26 |

## 调用与代价

| 项目 | v2 | v3 |
|---|---:|---:|
| 成功样本数 | 200 | 200 |
| 失败样本数 | 0 | 0 |
| 调用数（含重试） | 200 | 200 |
| 重试数 | 0 | 0 |
| 输入 tokens | 349773 | 424373 |
| 输出 tokens | 4285 | 4254 |
| 总 tokens | 354058 | 428627 |
| 缓存输入 tokens | 305664 | 356864 |
| reasoning tokens | 2385 | 2548 |
| 费用未知调用数 | 200 | 200 |
| 运行时间（秒，不含准备） | 193.88757 | 215.46865 |

输入 tokens 相对变化：+21.33%。费用未有已核实部署单价，保留 NULL。

两组温度 0、thinking enabled、reasoning_effort low、max_tokens 8192、context_tokens 32768、并发 4、每条最多 2 次尝试。实际响应模型为 {"control": {"zhipu/glm-5.3": 200}, "candidate": {"zhipu/glm-5.3": 200}}。供应商未提供可核实的权重修订号；两组顺序执行，缓存和耗时不可单独解释为算法收益。

## 逐类结果

| 标签 | 真值数 | v2 F1 | v3 F1 |
|---|---:|---:|---:|
| admiration | 29 | 51.06% | 60.87% |
| amusement | 10 | 46.67% | 52.94% |
| anger | 9 | 23.53% | 28.57% |
| annoyance | 13 | 24.39% | 22.86% |
| approval | 17 | 42.11% | 31.25% |
| caring | 7 | 40.00% | 28.57% |
| confusion | 3 | 0.00% | 0.00% |
| curiosity | 7 | 33.33% | 33.33% |
| desire | 2 | 80.00% | 100.00% |
| disappointment | 9 | 33.33% | 0.00% |
| disapproval | 7 | 12.12% | 9.52% |
| disgust | 3 | 25.00% | 28.57% |
| embarrassment | 0 | 0.00% | 0.00% |
| excitement | 1 | 0.00% | 0.00% |
| fear | 4 | 40.00% | 33.33% |
| gratitude | 13 | 92.31% | 88.89% |
| grief | 2 | 0.00% | 0.00% |
| joy | 6 | 44.44% | 46.15% |
| love | 4 | 44.44% | 40.00% |
| nervousness | 0 | 0.00% | 0.00% |
| optimism | 9 | 50.00% | 36.36% |
| pride | 0 | 0.00% | 0.00% |
| realization | 5 | 28.57% | 0.00% |
| relief | 1 | 0.00% | 0.00% |
| remorse | 2 | 0.00% | 50.00% |
| sadness | 4 | 33.33% | 36.36% |
| surprise | 4 | 23.53% | 28.57% |
| neutral | 61 | 45.10% | 44.64% |

零支持类别：embarrassment, nervousness, pride。Macro-F1 的零分母按 0，200 条小切片中的少数类估计不稳定。

## 成对案例

### 由错变对

| source_id | 原文 | 真值 | v2 | v3 |
|---|---|---|---|---|
| ed3kols | Thanks made feel a bit better | gratitude | gratitude, relief | gratitude |
| ed3vmna | You know he only follows NSFW subs | neutral | disapproval | neutral |
| eddjzwg | "I guess ""hungasfuck"" or whatever is not doing gif recipe vids anymore lol" | amusement | neutral | amusement |
| edic2u6 | I'm impressed. A 3 year old account and this is your only comment. | admiration | disapproval, surprise | admiration |
| edlriuk | This is cute AF. For 5.00 - 10.00? That's not bad yo. Some people at fairs would pay for 25.00 for this. | admiration, approval | relief | admiration, approval |

### 由对变错

| source_id | 原文 | 真值 | v2 | v3 |
|---|---|---|---|---|
| ed04ap3 | We'll worry about it tomorrow | neutral | neutral | caring |
| edac9qm | *sees title* hello there! | neutral | neutral | amusement |
| eddcj0a | Well I mean Australia is a good example of this working isn’t it ? | approval | approval | neutral |
| ee05gul | This is perfect | admiration, approval | admiration, approval | admiration |
| eea7dqp | Erm... Maybe he wants you to use those toys... On him. | neutral | neutral | amusement |

## 可复查证据

- 独立核对原始 TSV 真值、实际模型请求、训练例子原文及标签、原始响应和数据库预测，并重算主指标。
- 200 对请求除 instruction 外完全一致，200 对入选示例的 ID、顺序、原文、标签与检索分数相同。
- 完整逐条对照：`data/research/exports/glm-dev200-prompt-v3-paired-cases-20260908.jsonl`。
- 两组标准导出：`data/research/exports/glm-dev200-prompt-v3-control-20260908/` 与 `glm-dev200-prompt-v3-candidate-20260908/`。
- SQLite：`data/research/experiments.sqlite3`；原始大工件：`data/research/artifacts/`。
- 运行状态：`data/research/train_prompt_v3_state.json`；分析结果：`data/research/train_prompt_v3_analysis.json`。
- 可恢复备份：`data/research/backups/train-prompt-v3-final-20260908/`。备份与软件检查的最终结果见独立 verification_report 工件。

核验计数：`{"requests_verified": 400, "native_truths_verified": 400, "selected_train_examples_verified": 1600, "raw_predictions_verified": 400, "all_gold_from_original_tsv": true, "test_split_used": false, "paired_examples_identical": 200, "paired_requests_only_instruction_differs": 200}`。

| 对象 | ID |
|---|---|
| control_run_id | `e1acdace-49ed-4ee7-b848-4a8ecd4366ed` |
| candidate_run_id | `4c37317a-8916-47de-9a3e-d72386f522e9` |
| control_evaluation_id | `7a168578-fa0a-441c-97d9-400afb321d2d` |
| candidate_evaluation_id | `2409b417-fcb9-4546-a751-236a3cd95ccc` |
| comparison_id | `ee07589f-e9ca-444e-8c19-ac4d210c5669` |
| analysis_artifact_id | `38ae9a78-0256-4fbf-a1a8-3b86a1f1bf6b` |
| paired_details_artifact_id | `e8e5e99a-a27c-4242-bb8f-da889a7f2ee1` |
| corrected_training_audit_artifact_id | `df0b445f-fe1b-4bea-bcb1-961f99778c41` |
