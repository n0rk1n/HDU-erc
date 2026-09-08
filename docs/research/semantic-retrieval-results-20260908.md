# 语义检索四组对照实验

生成时间（UTC）：2026-09-08T09:59:52.073789Z。

## 实验范围

固定 GLM-5.3、native-labels-v3 指令、28 类、候选 50、入选 4 条示例、相同输出预算及评分方式。使用新的固定 dev200；排除先前评估集合的同文本记录和训练集文本复制。四组完成后统一评分，官方 test 未使用。

| 组 | 相似度 | 示例政策 |
|---|---|---|
| A | 词面 Jaccard | ranked |
| B | BGE 语义余弦 | ranked |
| C | 词面 Jaccard | contrastive-v1 |
| D | BGE 语义余弦 | contrastive-v1 |

A/B/C/D 的系统指令、标签定义、查询和模型参数相同，只有入选示例允许变化。对比规则和混淆关系沿用旧版。本轮没有加入专用分类器、重排模型或新的提示词。

## 主结果

| 指标 | A | B | C | D |
|---|---:|---:|---:|---:|
| Micro-F1 | 41.67% | 40.83% | 41.93% | 42.65% |
| Macro-F1（28 类） | 37.83% | 36.92% | 40.52% | 38.31% |
| 完全匹配率 | 31.50% | 31.00% | 30.50% | 32.50% |
| Precision | 41.15% | 40.33% | 41.67% | 41.87% |
| Recall | 42.19% | 41.35% | 42.19% | 43.46% |
| 至少一个细类命中 | 49.00% | 47.50% | 48.50% | 50.00% |
| 大类 Micro-F1 | 54.30% | 49.89% | 54.18% | 51.80% |
| 大类完全匹配 | 47.00% | 43.50% | 46.50% | 44.00% |
| 至少一个大类命中 | 59.00% | 54.00% | 59.00% | 56.50% |
| TP | 100 | 98 | 100 | 103 |
| FP | 143 | 145 | 140 | 143 |
| FN | 137 | 139 | 137 | 134 |
| 平均预测标签数 | 1.215 | 1.215 | 1.2 | 1.23 |

## 配对比较

| 比较 | 用途 | Micro-F1 差值（百分点） | 95% 区间（百分点） | 示例顺序或成员改变 | 错→全对 / 全对→错 |
|---|---|---:|---|---:|---|
| B-A | 主比较：更换相似度 | -0.83 | [-5.85, +4.26] | 200 | 18 / 19 |
| D-C | 主比较：更换相似度 | +0.72 | [-3.79, +4.87] | 200 | 16 / 12 |
| D-B | 次比较：对比规则消融 | +1.82 | [-2.58, +6.07] | 74 | 14 / 11 |
| C-A | 次比较：对比规则消融 | +0.26 | [-2.97, +3.77] | 63 | 12 / 14 |

配对 percentile bootstrap：seed=42、1000 次重采样。区间未经多重比较校正，不覆盖服务重复运行波动；不能只挑最高的一组宣称稳定改进。

- B-A：本轮未观察到提升（-0.83 个百分点）；区间包含零，证据不足以确认稳定差异。
- D-C：本轮观察到上升（+0.72 个百分点）；区间包含零，证据不足以确认稳定差异。

本轮保留全部实验配置和结果，不自动替换默认方法。该切片用于初步筛选；后续若据此调整方法，它也成为开发数据。旧 dev200 的分数不能与本轮新切片直接相减来估计方法收益。

## 每类 TP/FP/FN 与 F1

| 标签 | 支持数 | A | B | C | D |
|---|---:|---|---|---|---|
| admiration | 12 | 8/9/4；55.17% | 12/7/0；77.42% | 9/8/3；62.07% | 10/5/2；74.07% |
| amusement | 11 | 9/21/2；43.90% | 8/27/3；34.78% | 8/22/3；39.02% | 10/25/1；43.48% |
| anger | 4 | 0/4/4；0.00% | 0/4/4；0.00% | 0/0/4；0.00% | 0/3/4；0.00% |
| annoyance | 10 | 5/9/5；41.67% | 5/9/5；41.67% | 4/7/6；38.10% | 6/10/4；46.15% |
| approval | 18 | 3/5/15；23.08% | 3/5/15；23.08% | 3/3/15；25.00% | 3/4/15；24.00% |
| caring | 5 | 1/4/4；20.00% | 1/5/4；18.18% | 1/5/4；18.18% | 1/5/4；18.18% |
| confusion | 5 | 1/2/4；25.00% | 2/1/3；50.00% | 2/1/3；50.00% | 3/1/2；66.67% |
| curiosity | 12 | 3/5/9；30.00% | 2/8/10；18.18% | 4/8/8；33.33% | 2/9/10；17.39% |
| desire | 4 | 2/3/2；44.44% | 2/1/2；57.14% | 2/2/2；50.00% | 2/2/2；50.00% |
| disappointment | 6 | 0/2/6；0.00% | 0/3/6；0.00% | 0/2/6；0.00% | 0/2/6；0.00% |
| disapproval | 6 | 1/36/5；4.65% | 2/29/4；10.81% | 1/43/5；4.00% | 2/35/4；9.30% |
| disgust | 5 | 2/2/3；44.44% | 1/4/4；20.00% | 1/3/4；22.22% | 1/3/4；22.22% |
| embarrassment | 3 | 1/0/2；50.00% | 2/0/1；80.00% | 1/0/2；50.00% | 1/0/2；50.00% |
| excitement | 5 | 1/2/4；25.00% | 1/1/4；28.57% | 1/0/4；33.33% | 1/1/4；28.57% |
| fear | 5 | 1/0/4；33.33% | 1/0/4；33.33% | 1/0/4；33.33% | 1/0/4；33.33% |
| gratitude | 10 | 9/0/1；94.74% | 10/0/0；100.00% | 10/0/0；100.00% | 10/1/0；95.24% |
| grief | 1 | 0/0/1；0.00% | 1/0/0；100.00% | 0/0/1；0.00% | 1/0/0；100.00% |
| joy | 7 | 3/3/4；46.15% | 3/2/4；50.00% | 4/3/3；57.14% | 4/3/3；57.14% |
| love | 7 | 5/3/2；66.67% | 4/3/3；57.14% | 5/4/2；62.50% | 5/2/2；71.43% |
| nervousness | 1 | 1/1/0；66.67% | 1/2/0；50.00% | 1/0/0；100.00% | 1/2/0；50.00% |
| optimism | 11 | 4/1/7；50.00% | 4/2/7；47.06% | 5/2/6；55.56% | 5/1/6；58.82% |
| pride | 0 | 0/0/0；0.00% | 0/1/0；0.00% | 0/0/0；0.00% | 0/0/0；0.00% |
| realization | 2 | 0/0/2；0.00% | 0/2/2；0.00% | 0/2/2；0.00% | 0/2/2；0.00% |
| relief | 2 | 1/2/1；40.00% | 0/1/2；0.00% | 1/2/1；40.00% | 0/1/2；0.00% |
| remorse | 1 | 1/0/0；100.00% | 0/0/1；0.00% | 1/0/0；100.00% | 0/0/1；0.00% |
| sadness | 3 | 2/4/1；44.44% | 2/6/1；36.36% | 2/5/1；40.00% | 2/4/1；44.44% |
| surprise | 5 | 3/2/2；60.00% | 3/3/2；54.55% | 4/2/1；72.73% | 4/3/1；66.67% |
| neutral | 76 | 33/23/43；50.00% | 28/19/48；45.53% | 29/16/47；47.93% | 28/19/48；45.53% |

本切片没有真值支持的类别：pride。Macro-F1 对零分母按 0，稀有类估计不稳定。

## 示例诊断

| 项目 | A | B | C | D |
|---|---:|---:|---:|---:|
| 找到对比样本的查询数 | 0 | 0 | 70 | 84 |
| 4 条示例标签包含至少一个真值的比例 | 61.00% | 71.50% | 61.00% | 72.50% |
| 示例标签与真值的平均 Jaccard | 24.22% | 29.39% | 23.32% | 27.73% |

这些是完成预测之后计算的标签重合诊断，不参与检索选择，也不等同于最终分类准确性。

### 本轮能支持的判断

普通排序下，示例标签覆盖率由 61.00% 变为 71.50%；最终 Micro-F1 却由 41.67% 变为 40.83%。因此，示例中出现正确标签的比例上升，并不足以保证模型做出正确判定。

本轮尚未证明“换成通用语义向量”或“沿用当前对比规则”能稳定提高准确性。语义候选是否在主题相近时仍混淆情绪、模型是否过度跟随个别示例，都是后续可检验的假设，不能由这一次实验直接认定为原因。现有索引和逐条证据可用于研究候选的情绪判别性；若据本轮错误调整规则，须在独立数据上重新验证。

## 用量与耗时

| 项目 | A | B | C | D |
|---|---:|---:|---:|---:|
| 成功样本 | 200 | 200 | 200 | 200 |
| 最终失败 | 0 | 0 | 0 | 0 |
| 调用次数 | 200 | 200 | 200 | 200 |
| 重试 | 0 | 0 | 0 | 0 |
| 输入 tokens | 424990 | 426452 | 425126 | 426505 |
| 输出 tokens | 4159 | 4274 | 4279 | 4070 |
| 推理 tokens | 2447 | 2546 | 2570 | 2331 |
| 缓存输入 tokens | 394240 | 391680 | 358400 | 356608 |
| 平均模型调用毫秒 | 2898.26 | 3276.705 | 2638.505 | 2527.49 |
| 检索总毫秒 | 160989 | 147813 | 162588 | 149101 |
| 检索中位毫秒 | 801.5 | 735.0 | 812.5 | 739.0 |
| 费用未知调用 | 200 | 200 | 200 | 200 |

预先固定的顺序执行次序：['D', 'B', 'C', 'A']。实际输入 tokens 因示例长度不同而变化；缓存和服务负载会影响耗时，不能把总耗时差全部归因于检索算法。未获得可核实单价，费用保持 NULL。

检索耗时覆盖相似度计算、排序、示例筛选及提示词 token 预算计算；语义组首次调用还包含冻结索引加载与校验。因此该耗时不是单独的向量搜索基准。

各组服务端返回的模型标识及次数：`{"A": {"zhipu/glm-5.3": 200}, "B": {"zhipu/glm-5.3": 200}, "C": {"zhipu/glm-5.3": 200}, "D": {"zhipu/glm-5.3": 200}}`。固定 API 模型标识不等于固定服务端权重修订，本轮无法核实服务端权重版本。

## 向量构建与存储

- 模型：`BAAI/bge-base-en-v1.5`，修订 `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`。权重及配置全文已存工件 `51fea567-cf11-44bc-954a-b1af11f8b2a5`。
- train：43410 条；查询：200 条。CLS pooling、无额外文本指令、归一化 float32、维度 768，精确余弦检索。
- 编码设备：mps；batch_size=32，max_length=512。train 截断 0 条，查询截断 0 条；逐条编码 token IDs、长度和批次耗时全部保留。
- 编码各批推理秒数之和：train 60.849，query 0.560。这是可复用的一次性编码开销，另有加载、分词和写盘开销。
- 冻结训练索引 `ce150a15-47f3-4046-891e-144aa575d09b`；查询工件 `8bd61959-6b1b-4a26-8a5c-868bc2374762`。查询向量不进入训练索引，正式分类与恢复运行都读取同一向量工件。

## 独立核验与案例

核验计数：`{"original_train_texts": 43410, "truths": 800, "requests": 800, "candidate_records": 40000, "full_score_vectors": 800, "selected_examples": 3200, "raw_predictions": 800}`。

核验包含原始 train/dev TSV、全部候选和分数、入选示例全文及完整标签、实际请求、原始响应、数据库预测和独立重算的评分。

### B-A：由不完全匹配变为完全匹配

| source_id | 原文 | 真值 | 原方法 | 新方法 |
|---|---|---|---|---|
| edd659r | At least they have picked up some good humanist notions along the way. | approval | admiration | approval |
| eddlyx6 | Agreed. He always sounded like he was forcing to much air out of his lungs when he talked to try to have a more intimidating voice. | approval | approval, disapproval | approval |
| edei86k | You’re* such a nice person! | admiration | annoyance, disapproval | admiration |
| edenjv6 | Damn mods are just trying to offend everyone! | annoyance | annoyance, disapproval | annoyance |
| edfozau | In my eyes 6 La Liga’s in a span of 18 years is just embarrassing | embarrassment | disapproval | embarrassment |

### B-A：由完全匹配变为不完全匹配

| source_id | 原文 | 真值 | 原方法 | 新方法 |
|---|---|---|---|---|
| ed22yca | Are you a twink? | neutral | neutral | curiosity |
| edbo4lx | Dude just wanted to say more words. | neutral | neutral | disapproval |
| edezz31 | That just sounds like a show that would piss me off tbh | annoyance | annoyance | anger, annoyance |
| edf0tjj | Check out my mémés. (That word is from japanese pop culture right?) | curiosity | curiosity | neutral |
| edn1s50 | * [NAME]: 17 million * [NAME]: 90 billion * [NAME]: 4 trillion I think that one question sums up the DLB Show perfectly. | neutral | neutral | amusement |

### D-C：由不完全匹配变为完全匹配

| source_id | 原文 | 真值 | 原方法 | 新方法 |
|---|---|---|---|---|
| ed4825m | Juul is heavy vapor. Idk how to describe it but i didnt feel like that when i used other juices | confusion | neutral | confusion |
| edayhd8 | Brilliant, thanks so much! | admiration, gratitude | gratitude | admiration, gratitude |
| edd659r | At least they have picked up some good humanist notions along the way. | approval | admiration | approval |
| eddlyx6 | Agreed. He always sounded like he was forcing to much air out of his lungs when he talked to try to have a more intimidating voice. | approval | approval, disapproval | approval |
| edezz31 | That just sounds like a show that would piss me off tbh | annoyance | disapproval | annoyance |

### D-C：由完全匹配变为不完全匹配

| source_id | 原文 | 真值 | 原方法 | 新方法 |
|---|---|---|---|---|
| eczfe9l | Lol Yes Most people arent 100% either (especially women). For a variety of reasons...finding beauty in other people, curiosity, jealousy, etc... | amusement | amusement | amusement, neutral |
| ed22yca | Are you a twink? | neutral | neutral | curiosity |
| ed4do1y | AT doing some work! | neutral | neutral | admiration |
| edbo4lx | Dude just wanted to say more words. | neutral | neutral | disapproval |
| edf0tjj | Check out my mémés. (That word is from japanese pop culture right?) | curiosity | curiosity | neutral |

## 复现定位

- 数据库：`data/research/experiments.sqlite3`；全文工件：`data/research/artifacts/`。
- 四组配对明细：`data/research/exports/semantic-dev200-paired-cases-20260908.jsonl`。
- 各组标准导出：`data/research/exports/semantic-dev200-{A,B,C,D}-20260908/`。
- 状态与分析：`semantic_build_state.json`、`semantic_experiment_state.json`、`semantic_experiment_analysis.json`。
- 最终可恢复备份：`data/research/backups/semantic-retrieval-final-20260908/`；完整性核验记录单独保存。

| 组 | Run ID | Evaluation ID | 代码提交 |
|---|---|---|---|
| A | `0ec3c464-aa95-4a2d-9db6-ffd4fee4b0a2` | `d4d6740c-fb7c-4813-a041-8c7f41805eba` | `e991d09760c112d5dd8b742f95d9ce4c3a9f5e82` |
| B | `1e2dd3e6-e154-452f-83c9-54a0dcf93295` | `615cd26b-1884-4d7e-a947-20b754bbd58f` | `e991d09760c112d5dd8b742f95d9ce4c3a9f5e82` |
| C | `3198eb6b-ba12-47f3-b7b2-bb2b223fd7fd` | `f3175c15-6e61-4ed3-a8dc-6832bc45ef93` | `e991d09760c112d5dd8b742f95d9ce4c3a9f5e82` |
| D | `c938a0be-03b8-4b4f-ad51-808cd7f0fcf7` | `3815f92d-2e2c-4489-b777-8755b504190b` | `e991d09760c112d5dd8b742f95d9ce4c3a9f5e82` |

分析工件：`27fd525e-d281-4047-927f-3cf5dcdd9ecb`；逐条明细工件：`8bf90542-0145-44b1-86be-9dbcb596d7a4`。

公开模型说明：[BGE-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)。本报告只使用本次真实运行数据，不以模型卡成绩代替本项目实验结果。
