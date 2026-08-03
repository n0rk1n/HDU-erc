# 情绪识别 Prompt 多版本消融实验设计

## 背景

当前项目已经在 EmpatheticDialogues 公开平衡 seed 上完成 `full`、
`no_dynamic_examples` 和 `zero_shot` 三组实验。结果没有显示动态示例相对两个对照带来
稳定提升，但这只能说明示例策略在当前模型和数据上的效果有限，不能回答标签解释、指令
长度和分类步骤等 Prompt 设计是否影响 32 类情绪识别。

本轮实验在不改变模型、数据、示例检索、输出 Schema 和评分程序的前提下，比较多种
Prompt 结构。第一阶段只使用 64 条平衡 seed 进行筛选，避免直接在完整 2,542 条测试集
上产生过高调用量。

## 目标

1. 比较当前完整 Prompt 与四种可解释的新 Prompt 设计。
2. 判断标签语义说明、指令长度、分层分类和相邻标签对比检查是否影响识别结果。
3. 保存每条实际 Prompt、Prompt 哈希、逐条预测和运行环境信息，使实验可以复查和续跑。
4. 形成中文指标报告和结论，为是否扩展到完整测试集提供依据。

## 非目标

- 本轮不改变聊天回复 Prompt，不改变线上默认情绪识别行为。
- 本轮不比较模型，不调整温度、超时或重试策略。
- 本轮不重新设计动态示例检索算法。
- 第一阶段不自动运行完整 2,542 条测试集。
- 64 条筛选结果不用于宣称某种 Prompt 已获得普遍有效的性能提升。

## 实验配置

第一阶段固定运行以下五组，每组 64 条，共 320 次预测。`full` 是当前基准，其余四组
均保留动态示例、情绪历史开关和默认上下文窗口，只改变 Prompt 模板。

| 组别 | 设计 | 唯一主动变化 | 研究问题 |
| --- | --- | --- | --- |
| `full` | 当前完整 Prompt | 无 | 提供基准 |
| `prompt_no_label_guidance` | 删除详细标签语义和边界说明 | 不渲染 `label_guidance` | 标签解释是否有帮助 |
| `prompt_concise_direct` | 压缩任务说明 | 用简洁直接的分类指令替换说明正文 | 长指令是否必要 |
| `prompt_coarse_to_fine` | 分层分类 | 要求先确定情绪族，再选择具体标签 | 是否减少跨情绪族错误 |
| `prompt_contrastive_check` | 对比检查 | 要求最终选择前比较最可能的两个相邻标签 | 是否减少相邻标签混淆 |

### `full`

完全复用 `DEFAULT_EMOTION_ANALYSIS_PROMPT`，不修改当前基准文本。它继续包含任务说明、
32 类标签、标签定义、动态示例、近期情绪候选、对话上下文和当前结构化响应说明。

### `prompt_no_label_guidance`

保留标签名称列表、动态示例、近期情绪候选、对话上下文和响应说明，只删除
“Label definitions”区块。该组只回答标签语义与相邻边界说明是否提供增益。

### `prompt_concise_direct`

使用简洁指令明确以下要求：从给定标签中选择一个最符合目标用户当前输入的情绪；参考
提供的动态示例；不得输出标签集合之外的值。标签定义、动态示例、近期情绪候选、对话
上下文和响应说明仍然保留，避免把“简化说明”和“删除实验信息”混为同一处理。

### `prompt_coarse_to_fine`

在完整信息基础上加入确定性的两步决策要求：先在内部判断宽泛情绪族，再从该情绪族附近
选择一个精确的 32 类标签。只返回既有结构化结果，不要求输出隐藏推理过程。该组主要
观察 Family Accuracy 和跨情绪族错误是否改善。

### `prompt_contrastive_check`

在完整信息基础上要求模型在内部找出最可能的两个候选标签，依据标签定义和对话证据检查
二者边界，然后只返回最终选择。不得输出候选比较过程。该组主要观察相邻标签混淆和 exact
Accuracy 是否改善。

## 公平性控制

所有配置固定使用：

- `data/benchmarks/empathetic_dialogues_v1/release/balanced_seed.jsonl` 的同一 64 条记录；
- 同一模型名称和 Codex CLI 版本；
- 同一动态示例库和检索结果；
- 同一调用超时、失败重试次数和隔离运行方式；
- 同一 `data/config/codex_emotion_result.schema.json`；
- 同一标签解析、Accuracy、Macro F1、Family Accuracy 和配对统计代码。

新 Prompt 不改变响应字段和标签集合。调用失败与解析失败必须保留在正式分母中，并在报告
中与分类错误分开列出。

## 代码设计

### Prompt 变体边界

新增一个专门的 Prompt 变体模块，公开受限的变体名称集合，并根据变体名称返回模板或
渲染选项。默认值始终为 `full`。未知变体立即报错，不静默回退。

`build_emotion_analysis_prompt()` 和 `build_emotion_prompt()` 增加显式的
`prompt_variant` 参数。现有调用者不传参数时保持原有 Prompt 字节级行为，确保应用默认
行为不变。

### 实验配置

`AblationRunConfig` 增加 `prompt_variant` 字段。四个新运行配置统一使用：

- `example_mode="dynamic"`；
- `include_emotion_history=True`；
- 默认上下文窗口；
- 各自的 `prompt_variant`。

现有 `full`、`no_dynamic_examples`、`no_emotion_history`、`short_context` 和
`zero_shot` 的行为保持兼容。

### 结果与续跑

逐条结果继续保存完整 `input`，并在 provenance 中记录 `prompt_sha256`、模型、输出
Schema、Codex CLI 版本和隔离运行版本。结果文件中的 `run` 必须与配置名称一致；任一
provenance 不匹配时重新调用，匹配的成功记录可以安全续跑。

Prompt 实验输出使用独立目录，不能覆盖既有 seed64 消融结果。报告以 `full` 为配对基准，
保留每个 case_id 的逐条预测。

## 数据流

1. 从平衡 seed 导出 64 条既有消融输入。
2. 对同一条输入分别构造五个配置的 Prompt。
3. 在调用前验证四个 treatment 与 `full` 的 Prompt 均存在实际差异。
4. 使用隔离的 Codex CLI 进程执行分类，并在每条任务后原子保存快照。
5. 按 case_id 对齐五组结果，计算指标、置信区间和配对检验。
6. 生成 CSV、Markdown 中文报告、简明结论和运行元数据。

## 统计与判读规则

主指标按以下优先级报告，不根据结果临时更改：

1. exact Accuracy；
2. Macro F1；
3. Family Accuracy 和 Family Macro F1；
4. 调用失败数与解析失败数。

每个 treatment 相对 `full` 计算：

- exact McNemar 检验；
- 按 case_id 配对、固定随机种子的 10,000 次 percentile bootstrap；
- Accuracy 和 Macro F1 差值及其 95% 区间；
- `full` 独有正确数与 treatment 独有正确数；
- 相邻标签与跨情绪族混淆样例。

64 条结果仅作为筛选证据。即使某组点估计更高，只要差值区间包含 0，也只能表述为
“出现候选趋势”，不能表述为“已经证明提升”。

## 第二阶段决策

第一阶段完成后不自动发起完整测试。报告按照 exact Accuracy、Macro F1、Family
Accuracy 的固定顺序列出最多两个候选 Prompt，并同时列出统计不确定性和错误变化。

是否在完整 2,542 条测试集上运行 `full` 与候选组，由用户在查看 64 条报告和预计调用量
后确认。完整阶段开始前冻结 Prompt 文本、模型版本和统计方法，不根据测试结果继续改词。

## 测试策略

实施采用 TDD，至少覆盖：

- 默认 `full` Prompt 与改动前文本完全一致；
- 每个新变体包含其专属指令，且保留统一标签、示例、上下文和响应约束；
- `prompt_no_label_guidance` 不包含标签定义正文；
- 未知 Prompt 变体会明确报错；
- 五个 Prompt 对同一 case 的哈希均可区分；
- 四个 Prompt treatment 都不会被 no-op 检查误判为与 `full` 相同；
- 续跑 provenance 能区分 Prompt 变体；
- 报告器能够处理全部五组并以 `full` 为配对基准；
- 完整测试套件通过。

## 错误处理

- 未知变体、模板缺少必需占位符或渲染失败时，在调用模型前终止。
- 单条调用超时或无效 JSON 继续使用现有重试和快照机制，不丢失已完成记录。
- 任一配置不足 64 条或 case_id 集合不一致时，不生成正式对比结论。
- Prompt 与 `full` 完全相同时，将该组标记为 no-op 并停止调用。

## 交付物

- 可审计的五组 Prompt 定义与测试；
- 五组各 64 条逐条预测；
- `metrics.csv`、中文完整报告、简明摘要、结论和运行元数据；
- README 中的实验入口、配置说明和结果边界；
- 独立任务提交，不自动合并或推送到 `main`。

## 完成标准

1. 五组 Prompt 均通过自动化差异验证。
2. 五组各获得 64 条结果，失败和无效输出均被准确计数。
3. 报告包含预先约定的全部指标和配对统计。
4. 完整测试套件通过，Git diff 只包含本任务内容。
5. 最终结论明确区分 64 条筛选趋势与完整测试证据。
