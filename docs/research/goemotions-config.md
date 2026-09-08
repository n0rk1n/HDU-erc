# GoEmotions 情绪配置

本次将当前应用的默认标签及参考示例改为 GoEmotions。`config/` 的内置配置与 `.env.example` 指向的 `data/config/` 配置同步，均包含 28 个官方标签，保留英文标签 ID、英文识别描述及中文显示名。

## 标签与分组

标签顺序对应官方 `emotions.txt` 的 ID 0～27，不保留旧标签别名。`description` 是本项目依据标签含义编写的识别说明，强调易混概念的区别，并非官方标注指南原文。

`emotion_families.json` 将官方 `ekman_mapping.json` 的“family → 标签列表”反转为当前加载器使用的“标签 → family”。官方文件覆盖 27 种情绪，项目另将 `neutral` 映射到 `neutral`。

| family | 标签 |
| --- | --- |
| anger | anger、annoyance、disapproval |
| disgust | disgust |
| fear | fear、nervousness |
| joy | admiration、amusement、approval、caring、desire、excitement、gratitude、joy、love、optimism、pride、relief |
| sadness | disappointment、embarrassment、grief、remorse、sadness |
| surprise | confusion、curiosity、realization、surprise |
| neutral | neutral |

family 是粗粒度分组，不是情绪强弱，也不能直接当作“所有易混情绪对”。后续研究仍应在开发集确认具体混淆关系，28 类指标为主，family 指标为辅。

## 示例库需要同步更换

旧示例使用 afraid、sad、angry 等旧标签，并有 GoEmotions 不存在的类别。只修改标签文件会使 `load_examples` 报未知标签；简单改英文单词也不能保证原始标注语义对齐。

两套 `emotion_examples.json` 已替换为 **56 条 GoEmotions train 原始样本，每类 2 条**。ID 格式为 `goemotions-train-<comment_id>`。所有文本和人工标签保持原样，包括原有拼写、标点和空白。

- 当前示例加载器只接受一个 `emotion` 字符串，所以仅选原本就有一个标注标签的训练样本，没有从多标签记录中任取一个标签。
- 候选通过类别相关词检索，再逐条阅读，优先选择能表达该情绪的短文本；不是随机抽样或总体分布的代表。固定 ID、源文件行号与原始标签 ID 记录在 `data/config/emotion_examples.metadata.json`。
- 文本规范化规则为 `" ".join(text.casefold().split())`。所选文本在 train 中只出现一次，在 dev/test 中没有同规则的重复；这里只核对留出集的文本和 ID，不使用其标签来选示例。
- 示例按官方标签顺序轮询两遍。当前检索方法仍是加权词面重叠与历史先验，默认最多选 4 条。这次更换数据没有实现对比检索算法。
- 这是一份供应用使用的经逐条审阅的种子示例库，不用于评测，也不替代后续研究中的完整 43,410 条训练集检索库。正式比较应固定公共候选库、避免当前候选筛选方式偏向某一方法，并保留原始多标签。

## 提示词、输出和历史兼容

`data/config/prompts/emotion_prompts.json` 更新为 `goemotions-v1`，去掉额外的 `no_emotion` 类，将纯事实、任务请求及中性反应纳入 `neutral` 的定义。识别失败继续由失败状态表示。

沿用 `primary_emotion` 与 `secondary_emotions`，不强制只输出一个情绪，也不禁止 neutral 与其他标签共现。主次顺序只用于应用展示；GoEmotions 本身没有主次排名。后续多标签评分应对两者取集合，不能只评主情绪。此配置修改尚未实现正式实验运行器或评分器。

既有数据库结果、历史消息、快照与 family 不会重写。旧标签不在新显示名配置中时，界面回退到原始英文标签；新分析只接受新标签，旧历史标签不会作为有效检索先验。模型仍可查看正常聊天历史，因此单条文本基准实验需要独立样本，不能沿用真实聊天中的跨样本历史。

合入配置后重启服务生效。如使用自定义路径，应确保标签、分组、示例及提示词来自同一套配置；不要让模型继续收到旧的 `no_emotion` 提示。没有为此修改 `.env`。

## 来源与复核

- [GoEmotions 论文](https://aclanthology.org/2020.acl-main.372/)。
- 固定上游 commit：`5594ac0ee7a13c77eb1e0b98f0f305a2ca65b6c4`。
- [官方标签](https://github.com/google-research/google-research/blob/5594ac0ee7a13c77eb1e0b98f0f305a2ca65b6c4/goemotions/data/emotions.txt)、[官方 Ekman 映射](https://github.com/google-research/google-research/blob/5594ac0ee7a13c77eb1e0b98f0f305a2ca65b6c4/goemotions/data/ekman_mapping.json)、[官方训练数据](https://github.com/google-research/google-research/blob/5594ac0ee7a13c77eb1e0b98f0f305a2ca65b6c4/goemotions/data/train.tsv)。
- 源文件地址、SHA-256、样本 ID/行号及配置文件 SHA-256 见 `data/config/emotion_examples.metadata.json`；保留上游 Apache-2.0 许可于 `data/config/goemotions_LICENSE.txt`。
- 上次下载的完整 benchmark 位于单独的数据准备分支，本配置分支从 main 创建，只读取了已校验的官方快照，不依赖另一个 worktree 才能启动应用或运行离线测试。

复现示例：先按 metadata 的固定 URL 与 SHA-256 取得 `train.tsv` 和 `emotions.txt`，按 `samples` 顺序读取 `train_line`（从 1 起），核对第三列 ID 与第二列标签列表，以第一列作为 `dialogue`、原生标签作为 `emotion`。禁止改用 dev/test 示例或重标原文。所有生成的 JSON 使用 UTF-8、`ensure_ascii=False`、`indent=2` 和文件末尾换行。

验证包括：两套配置一致、官方标签及 family 覆盖、56 条源样本核对、多标签和 neutral 共现解析、旧标签拒绝、新标签在图/API/历史展示中流转，以及旧记录不重标。不发起真实模型请求；通过测试不等于识别准确率已经提升。
