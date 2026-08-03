# Codex CLI 情绪识别消融实验运行设计

## 目标

使用本机已登录的 Codex CLI 执行情绪识别消融实验，先完成 10 条样本的连通性试跑，再对 Emotion Ablation V2 的 64 条种子集运行 5 个既有实验配置，最终生成一份可追溯的中文 Markdown 报告。

本次结果衡量的是 Codex CLI Agent 在统一运行约束下的情绪识别表现，不把结果表述为裸模型 API 的性能。

## 实验范围

保持现有 5 个实验配置不变：

| 实验 | 示例策略 | 历史情绪先验 | 上下文窗口 |
| --- | --- | --- | --- |
| `full` | 动态示例 | 开启 | 默认窗口 |
| `no_dynamic_examples` | 静态示例 | 开启 | 默认窗口 |
| `no_emotion_history` | 动态示例 | 关闭 | 默认窗口 |
| `short_context` | 动态示例 | 开启 | 最近 1 轮 |
| `zero_shot` | 无示例 | 关闭 | 默认窗口 |

正式实验使用 `data/benchmarks/emotion_ablation_v2/release/seed.jsonl` 的 64 条记录，共计 320 个“样本 × 配置”任务。试跑从同一数据集中稳定选取前 10 条记录，共计 50 个任务。

## 运行架构

新增一个独立的 Codex CLI 运行器，不改变 Web 应用和现有 OpenAI-compatible LLM 路径。运行器复用现有情绪 Prompt 构建逻辑，对每个任务启动隔离的 `codex exec` 临时会话：

1. 读取种子集记录。
2. 按实验配置调用现有 `build_emotion_prompt`。
3. 将生成的 Prompt 包装成只做情绪分类的 Codex 指令。
4. 使用 `--ephemeral`、`--sandbox read-only` 和固定 JSON Schema 调用 Codex CLI。
5. 从最终 JSON 中读取情绪标签并校验其属于项目支持的标签集合。
6. 为每个任务立即写入可恢复的 JSONL 结果，避免中途退出后丢失已完成任务。

每个样本使用独立临时会话，防止上一条样本污染下一条样本。5 组实验使用相同的 Codex 模型和运行参数。

## 输出文件

实验文件写入独立目录，不覆盖既有运行记录：

```text
data/records/codex_cli_ablation/
  pilot/
    full.json
    no_dynamic_examples.json
    no_emotion_history.json
    short_context.json
    zero_shot.json
  seed64/
    full.json
    no_dynamic_examples.json
    no_emotion_history.json
    short_context.json
    zero_shot.json
    metrics.csv
    summary.md
    report-zh.md
  raw/
    *.jsonl
```

最终中文报告包含：

- 实验环境、分支、提交、Codex CLI 版本和运行时间。
- 5 组实验的样本数、成功数、Accuracy 和 Macro F1。
- 相对 `full` 的绝对指标变化。
- 中文与英文样本分组指标。
- 高/中/低上下文依赖样本表现。
- 各情绪标签的主要混淆情况。
- 典型错误案例与可能原因。
- 实验限制，包括 Codex Agent 系统指令影响、种子集规模和 `zero_shot` 的组合消融性质。

## 失败处理与恢复

- 单个 Codex 调用超时或返回无效 JSON 时记录失败，不终止整批任务。
- 每个任务最多重试一次；重试仍失败则保留错误信息并继续。
- 重新运行时默认跳过已经成功完成的“样本 × 配置”任务。
- 试跑阶段如输出解析失败、无效标签过多或 CLI 无法联网，则不进入 64 条正式运行。
- 报告同时展示总样本数、有效预测数和调用失败数，失败预测按错误计入正式 Accuracy 和 Macro F1。

## 安全与资源约束

- 不读取或写出 Codex 登录凭据。
- Codex 子进程使用只读沙箱，且不允许执行项目修改任务。
- 默认顺序执行，避免并发触发速率限制或造成不可控配额消耗。
- 先完成 50 个任务的试跑，只有试跑通过才继续 320 个正式任务。
- 所有生成物保存在实验分支工作树内，便于审计与清理。

## 验证标准

试跑通过需满足：

- 50 个任务均产生可读取的结果记录。
- 至少 95% 的任务返回合法 JSON 和项目支持的情绪标签。
- 相同输入不会继承其他样本的会话上下文。

正式运行完成需满足：

- 5 个实验组分别覆盖全部 64 条种子样本。
- 每个结果可以通过 `case_id` 回溯到原始样本。
- 指标由项目现有评估逻辑计算。
- 报告中的表格数值与 `metrics.csv`、原始结果一致。
- 运行器测试、现有消融相关测试和报告一致性检查通过。

## 非目标

- 本次不运行 500 条正式集。
- 本次不修改聊天机器人线上情绪识别行为。
- 本次不把 Codex CLI 结果与其他模型结果混为同一实验。
- 本次不声称种子集结果具有统计显著性或代表生产流量。
