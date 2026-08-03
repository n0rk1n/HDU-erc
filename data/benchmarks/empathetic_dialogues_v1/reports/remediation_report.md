# EmpatheticDialogues 低准确率问题修复复盘

## 结论

原 pilot 的低准确率不能直接归因于模型能力。主要问题是评测构造把“整段对话的情绪情境标签”贴到了目标用户最后一句话上，而项目 Prompt 要判断“当前情绪”；对话经过回应后，最后一句的情绪可能已转移。另有示例覆盖不足、检索受停用词干扰、近邻标签无边界说明、样本前缀不具代表性和无效消融等问题。

本次已修正正式数据、Prompt、示例库、检索、指标和实验范围，并保留旧转换作为弱标签诊断集，防止丢失研究线索。

## 问题证据

原 `codex_pilot10/` 有以下可复核现象：

- 10 条只覆盖 5 类，每类连续 2 条，不能代表 32 类总体表现；
- 目标标签在 10 条 `current_input` 中逐字出现 0 次；
- 典型错误包括 `afraid→terrified`、`angry→annoyed`、`anticipating→excited`，部分是强度或近邻边界；
- `no_emotion_history` 的 10/10 Prompt 与 `full` 相同；`short_context` 只有 2/10 发生变化；
- 原动态示例库只有 8 条，只覆盖 8/32 类，且按原始词集合重叠排序，常被通用词影响。

## 根因与处理

| 根因 | 处理 | 可复核位置 |
| --- | --- | --- |
| conversation-level 标签与最后一句 current emotion 错位 | 正式集改为首位说话者第一条情境表达；旧转换单列弱标签诊断集 | `release/test.jsonl`、`release/context_diagnostic.jsonl` |
| few-shot 仅覆盖 8/32 类 | 从官方 train split 生成每类 2 条、共 64 条人工示例；test 不进入 Prompt | `few_shot/train_examples.jsonl` |
| 检索被停用词和高频词干扰 | 去除英文停用词，使用示例库 IDF 加权，并保留可审计的选择原因 | `chatbot/emotion_retrieval.py` |
| 32 个细粒度标签缺少边界 | 为 32 类加入操作性定义，明确 afraid/terrified、annoyed/angry 等边界 | `chatbot/emotion_labels.py` |
| exact 指标无法区分近邻错与跨族错 | 保留 exact Accuracy/Macro F1 为主指标，新增 Family 指标作为诊断 | `scripts/evaluate_emotion_analysis.py` |
| seed 前缀集中在少数类别 | 改为按轮次穿插 32 类；前 32 条恰好每类 1 条 | `release/balanced_seed.jsonl` |
| 在无历史数据上运行历史消融 | Prompt 预检确认 no-op，本次只运行 3 组有效对照 | `reports/aligned_pilot32/` |

## 数据边界

- 正式 test：2,542 条，32 类，`ground_truth_alignment=aligned_first_speaker_grounding`，全部 `history=[]`。
- 弱标签诊断：2,542 条，显式标记 `weak_conversation_label_on_later_turn` 与 `emotion_evidence_weak`，不用于主准确率。
- 平衡 seed：64 条，每类 2 条；前 32 条每类 1 条。
- few-shot：64 条，来自官方 train split；与 test case ID 不重叠。
- `source_situation` 只保留作来源追溯，不导出到模型输入，避免泄漏。

## 实验设计修正

正式集没有历史，所以：

- `full`：动态检索 4 条 train few-shot；
- `no_dynamic_examples`：固定静态 few-shot；
- `zero_shot`：不使用 few-shot；
- `no_emotion_history`、`short_context`：逐 Prompt 与 `full` 相同，判定为 no-op，不执行模型调用。

修复后的 pilot 使用前 32 条（32 类各 1 条）和 `gpt-5.6-sol`。最终指标见 `aligned_pilot32/metrics.csv`；三组均要求 32/32 有效预测、0 调用失败后才允许解读。

## 修复后结果

三组最终均为 32/32 有效预测、0 调用失败：

| Run | Exact Accuracy（95% CI） | Macro F1 | Family Accuracy | Family Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| `full` | 59.38%（42.26%–74.48%） | 50.52% | 75.00% | 68.83% |
| `no_dynamic_examples` | 56.25%（39.33%–71.83%） | 46.88% | 71.88% | 60.74% |
| `zero_shot` | 56.25%（39.33%–71.83%） | 46.35% | 71.88% | 60.74% |

可以得出的结论只有：修复后的真实数据链路不再呈现原 pilot 的 20%–40% 低值，`full` 在这 32 条上比两个对照多正确 1 条；但置信区间高度重叠，不能据此宣称动态示例有统计显著优势。

`full` 的 13 个 exact 错误中，有 5 个落在同一预定义情绪族，包括 `anticipating→excited`、`apprehensive→anxious`、`furious→angry`、`lonely→devastated`、`sentimental→nostalgic`。其余错误中也存在文本本身支持多种解释的情况，例如邻居遭枪击的 `angry→afraid`，说明剩余误差同时包含细粒度边界、单标签语料的多情绪性和模型判断偏差。

修复前后不能做严格数值提升比较：旧 pilot10 使用错位的最后一句且只覆盖 5 类，新 pilot32 使用对齐后的首轮且覆盖 32 类。旧结果的作用是暴露评测问题，新结果的作用是验证修复后的链路。

## 复现命令

```bash
python scripts/benchmark/prepare_empathetic_dialogues.py \
  --archive /private/tmp/empatheticdialogues.tar.gz

python scripts/benchmark/export_emotion_benchmark.py \
  --input data/benchmarks/empathetic_dialogues_v1/release/balanced_seed.jsonl \
  --output-dir data/records/empathetic_dialogues_aligned_seed_export

python scripts/run_codex_cli_emotion_ablation.py \
  --dialogues-file data/records/empathetic_dialogues_aligned_seed_export/dialogues.jsonl \
  --output-dir data/records/codex_cli_ablation/empathetic_dialogues_aligned_pilot32 \
  --limit 32 --run full --model gpt-5.6-sol
```

将 `--run` 分别替换为 `no_dynamic_examples` 和 `zero_shot`。原始模型输出在 `data/records/`（按项目规则不提交）；确定性报告在 `reports/aligned_pilot32/`（提交）。

## 执行过程与异常处理

首次启动三组实验时，Codex CLI 因受限沙箱不能写入自身状态数据库，三组均产生 32 条失败快照，错误核心为 `attempt to write a readonly database` / `failed to initialize in-process app-server client`。这些记录没有进入模型，不能记为 0% 准确率。

处理步骤：

1. 检查每个结果文件的 `success`、`error`，确认是 96/96 环境初始化失败而非分类失败；
2. 保留同一 Prompt、模型、Schema、超时和重试参数，只调整 CLI 初始化所需的执行权限；
3. 利用现有 provenance 机制覆盖失败快照；
4. 重新生成报告前强制检查每组 `valid_predictions=32`、`failures=0`。

这段过程保留在复盘中，是为了避免后续把基础设施故障误读成模型准确率。

## 如何解释结果

- 不能把修复前 pilot10 与修复后 pilot32 直接做数值提升比较：两者输入目标和类别覆盖不同。
- exact Accuracy/Macro F1 是正式结论；Family 指标只表示预测是否落在预先固定的相邻情绪族内。
- 每类仅 1 条的 pilot32 只能用于链路与方向性验证。2026-08-03 已补充完整 64 条 seed 及配对统计，结果见 `seed64_gpt56sol/`；理想情况下仍应运行完整 2,542 条 test。
- EmpatheticDialogues 仍不适合验证历史截断和情绪历史组件。该问题需要引入带逐句人工标签的 MELD/CPED，并独立报告标签映射和许可边界。

## 防回归检查

- 转换测试断言正式集取首轮、诊断集标弱标签；
- 数据测试断言 2,542/2,542/64/64 条数、32 类覆盖、train/test 分离；
- Prompt 测试断言标签定义进入默认模板；
- 检索测试断言停用词不会压过内容词；
- 评估测试断言 exact 错误仍为错误，同时可标记 family near miss；
- 全量测试和数据校验结果在最终提交说明中记录。
