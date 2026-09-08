# GLM-5.3：200 条开发集 prompt / example 真实对照结果

报告生成时间（UTC）：2026-09-08T06:16:47.661866Z。正式执行代码：`cc903b60a0bbf2040e8b051d2912bc38be50de1c`。

## 完成范围

两组均完成同一固定开发集的 200 条真实预测，共 400 条正式预测、400 条评分明细。额外 2 条接口检查独立保存，不纳入正式比较。已完成评分、1000 次配对 bootstrap、逐条来源核验、结果导出与 SQLite 入库。

基线调用 201 次，新版调用 200 次；正式样本最终失败均为 0。基线重试 1 次，新版重试 0 次。该次无效输出包含 Markdown JSON 代码围栏，严格解析器拒绝后重试成功；首次响应与重试均原样保留，没有在评分时静默清洗。

## 主要结果

| 指标 | 原版 v1 + ranked | 新版 v2 + contrastive-v1 | 新版减原版 |
|---|---:|---:|---:|
| Micro-F1 | 39.53% | 40.62% | +1.09 个百分点 |
| Macro-F1（28 类） | 28.88% | 30.22% | +1.34 个百分点 |
| 精确标签集合匹配率 | 25.50% | 24.50% | -1.00 个百分点 |

Micro-F1 差值的 95% 配对 percentile bootstrap 区间：[-2.71, +5.12] 个百分点，seed=42、重采样 1000 次、每次配对抽取 200 条。

Micro-F1 差值的配对区间包含 0，本轮尚不能确认新版组合优于原版；应保留结果，继续做消融和更大样本验证。

按整条标签集合判断，新版修正了 10 条原版错误，也使原本正确的 12 条变为错误。

## 实验约束

- 数据集：GoEmotions，官方固定版本 `5594ac0ee7a13c77eb1e0b98f0f305a2ca65b6c4`；原生 28 类多标签，包含 neutral 共现。
- 训练集：43,410 条，仅从冻结 train 语料检索，原文与标签均不改写。
- 评价：固定 `goemotions-dev200`，seed=42；官方 test 未用于本轮。开发集真值在两组方法冻结及预测完成后用于评分。
- 请求模型：`ZHIPU/GLM-5.3`；供应商实际返回的 model 字段见调用记录。服务未返回可验证的权重版本或模型修订号，因此不宣称已冻结供应商内部权重。
- 参数：temperature=0、enable_thinking=true、reasoning_effort=low、max_tokens=8192、实验 context_tokens=32768、safety_tokens=256、并发4；每条最多2次、每组最多400次尝试。
- 两组均 k=4、candidate_count=50，使用相同训练集、开发集、分词器、解析器及评分规则。唯一方法差异是 prompt 版本与示例选择策略。
- 锚点与对照标签关系属于语义先验，不是从本轮开发集错误事后生成的混淆矩阵。
- 本轮同时更改 prompt 和 example，只能解释组合效果；尚未进行“只改 prompt / 只改 example”的消融。
- F1 按多标签集合统计；28 类 Macro-F1 的零分母按 0。失败按空预测纳入分母，本轮最终无失败。

该 200 条切片中没有真值支持的标签：embarrassment, nervousness, pride。少数类和零支持类限制了 Macro-F1 的解释范围；区间反映样本抽样不确定性，不包括模型服务漂移或多次独立运行的不确定性。

## 调用与资源记录

| 项目 | 原版 | 新版 |
|---|---:|---:|
| 正式调用数（含重试） | 201 | 200 |
| 供应商输入 tokens | 265,039 | 349,773 |
| 供应商输出 tokens | 2,719 | 4,602 |
| 供应商缓存命中输入 tokens | 206,592 | 305,664 |
| 供应商报告的 reasoning tokens | 849 | 2,714 |
| 返回 reasoning 内容的响应数 | 50 | 98 |
| 费用未知调用数 | 201 | 200 |
| 平均调用延迟（秒） | 2.481 | 2.344 |
| 运行耗时（秒，含逐条检索及落库） | 183.39 | 192.92 |

正式调用输入 tokens 变化：+31.97%（相对变化）。该统计包含重试；两组依次运行且服务缓存存在，耗时和缓存差异不能单独归因于检索算法。运行耗时不含冻结阶段对全体样本的预算预检。

原版响应模型字段：`{'zhipu/glm-5.3': 201}`；新版响应模型字段：`{'zhipu/glm-5.3': 200}`。所有 HTTP 状态和 usage 原文已保留。

没有已核实的该部署计费单价，因此费用字段保留 NULL，不能把它解释为免费或零费用。请求中的 thinking / effort 字段已记录；供应商未返回的内部推理或版本信息没有补造。

## 分层诊断（事后分析，不用于本轮调参）

| 子集 | 样本数 | 原版 Micro-F1 | 新版 Micro-F1 |
|---|---:|---:|---:|
| 单标签 | 172 | 36.63% | 38.93% |
| 多标签 | 28 | 50.47% | 47.17% |
| neutral 共现 | 1 | 66.67% | 66.67% |
| 找到对照示例 | 77 | 36.63% | 34.78% |
| 实际更换示例成员 | 62 | 39.51% | 34.73% |

这些子集可能重叠，不能将行数相加视为独立总体。对照示例选择由输入文本与训练标签决定，不读取查询真值。

实际更换示例的 62 条中，Micro-F1 从 39.51% 降至 34.73%；这是事后子集描述，不能单独分离 prompt 与检索的作用。下一步应优先做两者的独立消融，检查弱相关对照示例是否引入干扰，不宜直接把本版组合当作已验证的研究改进。

## 逐标签 F1

| 标签 | 真值支持数 | 原版 F1 | 新版 F1 |
|---|---:|---:|---:|
| admiration | 29 | 62.50% | 54.17% |
| amusement | 10 | 47.37% | 51.61% |
| anger | 9 | 26.67% | 26.67% |
| annoyance | 13 | 25.81% | 31.58% |
| approval | 17 | 37.84% | 42.11% |
| caring | 7 | 26.67% | 37.50% |
| confusion | 3 | 0.00% | 0.00% |
| curiosity | 7 | 53.33% | 40.00% |
| desire | 2 | 66.67% | 57.14% |
| disappointment | 9 | 20.00% | 53.33% |
| disapproval | 7 | 10.71% | 9.68% |
| disgust | 3 | 25.00% | 22.22% |
| embarrassment | 0 | 0.00% | 0.00% |
| excitement | 1 | 0.00% | 0.00% |
| fear | 4 | 66.67% | 66.67% |
| gratitude | 13 | 88.89% | 88.00% |
| grief | 2 | 0.00% | 0.00% |
| joy | 6 | 50.00% | 40.00% |
| love | 4 | 36.36% | 40.00% |
| nervousness | 0 | 0.00% | 0.00% |
| optimism | 9 | 42.86% | 47.06% |
| pride | 0 | 0.00% | 0.00% |
| realization | 5 | 25.00% | 28.57% |
| relief | 1 | 0.00% | 0.00% |
| remorse | 2 | 0.00% | 0.00% |
| sadness | 4 | 26.67% | 33.33% |
| surprise | 4 | 23.53% | 25.00% |
| neutral | 61 | 46.15% | 51.49% |

## 相邻标签替代错误

计数定义：真值含 A 且不含 B，预测含 B 且漏掉 A，或反向情况。同一文本可能涉及多个标签对，本表不可直接相加为总错误率。

| 标签对 | 原版次数 | 新版次数 |
|---|---:|---:|
| anger / annoyance | 2 | 3 |
| annoyance / disapproval | 4 | 5 |
| admiration / approval | 2 | 3 |
| joy / excitement | 1 | 0 |
| joy / relief | 0 | 0 |
| sadness / disappointment | 3 | 1 |
| sadness / grief | 0 | 0 |
| fear / nervousness | 0 | 0 |
| curiosity / confusion | 0 | 1 |
| realization / surprise | 1 | 1 |
| remorse / embarrassment | 0 | 0 |

## 由错变对的案例

按 source_id 排序列出至多 5 条；这里的“对/错”是相对于公开数据集标注。

| source_id | 原文 | 真值 | 原版 | 新版 |
|---|---|---|---|---|
| ed3kols | Thanks made feel a bit better | gratitude | gratitude, relief | gratitude |
| edac9qm | *sees title* hello there! | neutral | amusement | neutral |
| edatc4w | I'll kill you if you do that again, *honey* | anger | amusement | anger |
| eddcj0a | Well I mean Australia is a good example of this working isn’t it ? | approval | neutral | approval |
| eddjzwg | "I guess ""hungasfuck"" or whatever is not doing gif recipe vids anymore lol" | amusement | neutral | amusement |

## 由对变错的案例

按 source_id 排序列出至多 5 条；这里的“对/错”是相对于公开数据集标注。

| source_id | 原文 | 真值 | 原版 | 新版 |
|---|---|---|---|---|
| ed04ap3 | We'll worry about it tomorrow | neutral | neutral | caring, optimism |
| ed3vmna | You know he only follows NSFW subs | neutral | neutral | disapproval, neutral |
| ed787tj | Could also mean that mutated versions of these plants could make excellent super weeds that destroy our infrastructure. Tree roots and weeds are already a huge problem. | neutral | neutral | disapproval |
| edeh53r | Not even going to open the link. Already know I agree with Sloss. | approval | approval | annoyance, approval |
| edlriuk | This is cute AF. For 5.00 - 10.00? That's not bad yo. Some people at fairs would pay for 25.00 for this. | admiration, approval | admiration, approval | approval, love |

## 数据位置与可复查记录

研究工作区相对路径：

- SQLite：`data/research/experiments.sqlite3`。
- 原始请求、响应、检索全量得分、配置和代码快照：SQLite artifacts 及 `data/research/artifacts/`。
- 原版导出：`data/research/exports/glm-dev200-baseline-20260908/`。
- 新版导出：`data/research/exports/glm-dev200-revised-20260908/`。
- 含可读标签、原文和两组完整入选示例的逐条对照：`data/research/exports/glm-dev200-paired-cases-20260908.jsonl`。
- 最终可恢复备份：`data/research/backups/glm-dev200-completed-20260908/`。
- 执行日志及驱动脚本：`data/research/glm_prompt_comparison_execution.log`、`data/research/run_glm_prompt_comparison.py`，内容同时入库。

| 记录 | ID |
|---|---|
| 原版 run | `9276ae31-7e55-4192-a577-2a0f4b660ecc` |
| 新版 run | `7070ade6-34a7-44a2-8074-d2f9607d8a4f` |
| 原版 evaluation | `fd1fc9b7-d409-4876-ae46-404d2643d62e` |
| 新版 evaluation | `a3af7f0e-f653-424f-9a01-759b33809361` |
| comparison | `1c64ca9b-d140-4cee-b53b-212ffb5866b6` |
| 独立核验及分析 artifact | `eadb171a-9f92-4492-a8f0-a584d463bce9` |
| 逐样本详细对照 artifact | `d948609d-d337-400c-a572-5effb0171bbd` |

## 独立核验

核对原始 TSV 真值 400 次、实际模型请求 400 条、入选训练示例 1600 次、原始响应与数据库有效预测 400 条；并独立重算三项主指标，与数据库评分一致。

软件测试 59 项通过。数据库、所有工件及可恢复备份的最终完整性核验结果另存于 verification_report / backup_manifest 工件。正式分析中没有重写旧记录，也没有把离线测试伪装为模型预测。
