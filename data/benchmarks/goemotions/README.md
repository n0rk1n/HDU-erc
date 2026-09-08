# GoEmotions：易混淆情绪对比检索的主实验数据集

本研究选择 GoEmotions 官方按标注者一致性筛选的版本，开展**英文单条文本、多标签情绪识别**实验。固定已有语言模型与嵌入模型，研究示例检索方法，不训练或微调模型。当前研究分支提供独立 SQLite 实验运行器及 zero-shot、random、lexical 基线；对比检索和正式模型评测尚未开展。

## 选择依据

- Google Research 在 [ACL 2020 论文](https://aclanthology.org/2020.acl-main.372/)中发布；具有[官方源码与数据](https://github.com/google-research/google-research/tree/master/goemotions)、[Hugging Face 数据页](https://huggingface.co/datasets/google-research-datasets/go_emotions)及 [TensorFlow Datasets 支持](https://www.tensorflow.org/datasets/catalog/goemotions)，便于查阅研究与复现。
- 人工标注，27 种情绪加 `neutral`，能够研究 anger/annoyance、sadness/disappointment、fear/nervousness 等细粒度差异。这些只是候选类别对，是否实际易混需要由开发集基线结果确认。
- 提供固定 train/dev/test 划分，适合将训练集作为带标签的检索库，验证集用于方法选择，测试集用于最终报告。
- 不需要另行招募标注人员；英文标签、现成模型与公开资料适合以公共资源为主的研究条件。

原始语料有 58,009 条评论；这里使用官方筛选版本的 **54,263 条**。两者不要混写。`full` 版本中的多条标注记录也不等于多条独立文本。

## 数据与用途

| 文件 | 样本数 | 后续实验用途 |
| --- | ---: | --- |
| `train.tsv` | 43,410 | 固定带标签的示例检索库，不用于模型参数训练 |
| `dev.tsv` | 5,426 | 调整检索策略、示例数、提示词及识别易混类别 |
| `test.tsv` | 5,427 | 方法和参数冻结后进行最终比较 |
| `emotions.txt` | 28 个标签 | 标签 ID 按行号从 0 开始对应 |

三个 TSV **没有表头**，每行依次为 `text<TAB>label_ids<TAB>comment_id`。`label_ids` 是逗号分隔的整数列表。保留原始文本、全部标签和评论 ID，不翻译、不重新划分、不截断成 30 个 token。

标签依次为：admiration、amusement、anger、annoyance、approval、caring、confusion、curiosity、desire、disappointment、disapproval、disgust、embarrassment、excitement、fear、gratitude、grief、joy、love、nervousness、optimism、pride、realization、relief、remorse、sadness、surprise、neutral。

## 与当前项目的衔接边界

实验程序直接使用原生 28 类**标签集合**，配置集中在 `config/emotion_labels.json` 与 `config/emotion_families.json`。完整多标签参与评分，不使用主次情绪排序。

输入只有评论正文；不将真实标签或评论 ID 放进待测输入。数据没有可直接用于本实验的连续对话历史，因此不使用上一样本的预测情绪作为本样本先验，也不将数据集行序模拟成聊天轮次。每条文本均独立识别。

本数据适合验证文本情绪的区分能力；不能单独支撑中文效果、多轮情绪迁移或聊天回复满意度结论。第一阶段采用英文原文；中文和真实多轮场景需要另外的数据验证。

## 后续实验约定

研究问题：**在模型与示例预算固定时，针对候选易混情绪选择有区分信息的对照样本，是否比普通相似检索更准确？** 数据集本身及使用大模型不构成创新，提升尚待实验验证。

1. 主比较使用同一冻结版本的识别模型、标签描述、输出协议和生成设置。检索方法共享同一个冻结嵌入模型与 train 检索库。记录版本、参数、随机种子和实际请求。
2. 至少比较零示例、随机示例、普通相似检索、对比检索。带示例的方法使用相同示例数及相同输入 token 上限，并报告实际 token；增加“相似检索加类别多样性”的消融，区分对比机制与单纯增加类别覆盖的作用。
3. 候选标签只来自待测文本的模型预测或训练库检索；易混类别对及权重只在 train/dev 上确定。测试标签只供评分，不参与候选选择、示例检索、提示词或参数修改。如果增加初步预测/复核调用，需要计入完整成本，并设置同调用预算的对照。
4. 对照示例用于显示情绪边界，不表示两种情绪必然互斥。例如真实标签同时含 anger 和 annoyance 时，两者都应保留。类别对错误分析应区分“漏掉正确标签”和“额外预测了不在真值中的标签”；单标签样本切片只能作为补充诊断。
5. 主报告同时给出 28 类 Macro-F1 和 Micro-F1（包含 neutral），并列出各类 F1/支持数、预先在 dev 确定的易混类别切片、完整调用次数、token 与耗时。精确集合匹配率可作辅助指标。对多个随机种子报告波动；最终方法差异按测试样本进行配对置信区间估计。任何预测阈值在 dev 上冻结。
6. 先从 dev 固定抽取约 200 条做流程与费用试跑，再用 dev 比较方案；随机种子和样本 ID 落盘。最终冻结方案后评测完整 test，不能拿 test 试跑、挑提示词或筛选优势子集。解析失败应统计，重试策略保持一致且计入成本；不能丢弃失败样本美化指标。

## 数据完整性与重复文本

所有上游文件均保持下载时的原始字节。`manifest.json` 固定官方 Git commit、下载时间、URL、文件长度与 SHA-256；`audit.json` 是本地结构校验结果。`verify.py` 仅用 Python 标准库，可离线复核哈希、行数、标签和跨划分重复：

```bash
# 从项目根目录执行；不会调用模型，也不会修改数据。
python3 data/benchmarks/goemotions/verify.py
```

本次核验结果：

- 54,263 个评论 ID 在三个划分之间互不重复；结构与官方行数一致。
- 多标签样本数：train 7,102，dev 878，test 837。官方标签中还存在 neutral 与其他情绪共现（分别 1,396、174、181 条），必须按原样保留，不能假定 neutral 总是互斥。
- 与 train 存在完全相同正文的 dev 样本 43 条、test 样本 37 条。忽略大小写并压缩空白后分别为 49 条、42 条。
- dev/test 之间也有正文重复，详情见 `audit.json`。这些数字来自不涉及模型预测的结构审计，不用于选择有利的测试样本。

**检索器执行查询级重复过滤：** 对每条待测文本，排除 train 中 ID 相同或 NFKC 标准化后再 casefold/空白合并的文本 相同的示例。所有带示例的方法都执行同一规则，不依赖待测标签；同时约束选中示例之间的重复文本。具体规则与版本保存于运行配置和检索步骤工件；统计口径若与原始审计不同，以其记录的标准化版本为准。

保留官方 test 全集作为主报告。另报告按上述固定规则与 train/dev 均无重复的 test 切片，明确实际样本数；不能把自定义去重结果冒称官方划分。该检查只覆盖 ID 和文字重复，不代表已排除语义近似、同主题内容或模型预训练接触过数据的影响。

## 来源与许可

- 固定源版本：[`5594ac0ee7a13c77eb1e0b98f0f305a2ca65b6c4`](https://github.com/google-research/google-research/tree/5594ac0ee7a13c77eb1e0b98f0f305a2ca65b6c4/goemotions)。精确下载地址见 `manifest.json`。
- `upstream_README.md` 为该版本的上游说明原文；`LICENSE` 为该仓库的 Apache License 2.0 原文。Hugging Face 数据页亦标注 Apache-2.0；来源归属和原始数据说明一并保留。
- 论文引用：Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi. 2020. *GoEmotions: A Dataset of Fine-Grained Emotions*. Proceedings of ACL 2020, pages 4040–4054. DOI: [10.18653/v1/2020.acl-main.372](https://doi.org/10.18653/v1/2020.acl-main.372).

如需重新下载，按 `manifest.json` 中的固定 URL 获取文件，并运行 `verify.py` 确认 SHA-256；不要改为会随时间变化的 `master` 地址后覆盖基准。
