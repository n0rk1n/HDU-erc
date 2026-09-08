# GoEmotions 情绪识别实验工程

研究目标：使用固定现成模型，在原生 28 类多标签情绪识别上比较示例检索方法，重点分析易混情绪的漏报和误报。当前提供 zero-shot、随机示例、词面 Jaccard 三种基线，以及词面候选中的对照示例选择策略。向量检索尚未实现。

该研究分支已移除聊天 Web/API、会话、回复生成、点赞、情绪门控和历史聊天归档。原聊天工程保留在 Git 历史与 `main`。本工程不训练模型。

## 离线快速开始

在**当前研究工作区**执行，使用 Python 3.12 或更高版本：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock
.venv/bin/python -m emotion_lab db init
.venv/bin/python -m emotion_lab dataset import-goemotions
.venv/bin/python -m emotion_lab dataset make-set --config config/datasets/train.json
.venv/bin/python -m emotion_lab dataset make-set --config config/datasets/dev200.json
.venv/bin/python -m emotion_lab run --config config/experiments/zero-shot.json --dry-run
.venv/bin/python -m emotion_lab storage verify
.venv/bin/python -m pytest -q
```

初始化、导入、dry-run 和测试都不需要 API Key，也不会产生模型费用。数据集已随分支保存，无需重新下载。导入 54,263 条英文评论：train 43,410、dev 5,426、test 5,427，保留全部原生标签及 neutral 共现。来源、许可证与固定 SHA-256 见 [数据说明](data/benchmarks/goemotions/README.md)。

默认新库为 `data/research/experiments.sqlite3`。数据库、WAL、完整调用证据、向量和导出均不进入 Git。`.env` 仅从当前项目读取；不会自动寻找主工作区或父目录的凭据。旧 `SQLITE_DB_PATH` 不参与新库路径解析，已有聊天库或未知数据库会被拒绝打开。

`dataset make-set` 每次生成一个新的冻结集合。已有集合不要重复创建；配置引用名称时要求唯一，也可改用输出的 `sample_set_id`。`db status` 可以检查当前统计；样本集合详情可直接查询 `sample_sets`。

## 实际调用模型

1. 将 `.env.example` 复制为当前工作区的 `.env`，填写 `EMOTION_LLM_API_KEY`（兼容回退到 `LLM_API_KEY`）。
2. 检查 `config/experiments/*.json`：模型名称、供应商 URL、thinking、温度、输入/输出预算、示例数和尝试次数均会冻结保存。JSON 明确配置优先于环境参数。
3. 根据供应商实际部署核对模型和分词器。示例中的 32,768 是本次实验自定预算，不是宣称模型的最大窗口。远端模型别名可能变化，响应实际返回的名称另行记录；服务未返回的版本记为未知。
4. 固定开发集试跑和费用预算后，显式执行模型运行命令：

```bash
.venv/bin/python -m emotion_lab run --config config/experiments/zero-shot.json
.venv/bin/python -m emotion_lab resume --run-id RUN_ID
.venv/bin/python -m emotion_lab evaluate --run-id RUN_ID
.venv/bin/python -m emotion_lab export --evaluation-id EVALUATION_ID
```

`random.json`、`lexical.json` 使用相同模型、训练集、开发集和 4 个示例上限。输入相同文本及重复示例会被排除，预算不足时记录裁剪原因。lexical 是词面 Jaccard，不是向量检索。完整词面得分、候选顺序、筛选结果与实际请求分别留档。

本轮训练集驱动的调整见 [示例与提示词说明](docs/research/train-prompt-examples.md)。`dev200-prompt-baseline.json` 保留原版提示词和排序；`dev200-prompt-revised.json` 使用 `prompt_version=native-labels-v2`、`example_policy=contrastive-v1`。后者在同一候选池中优先加入一个不同相邻标签的训练例，无适合对照则保留原排序。默认配置仍为 v1/ranked；两个开关可单独配置，以供后续消融实验。旧静态示例文件只是测试夹具，不参与当前实验推理。

一次独立重复创建新的 run；`resume` 仅恢复同一 run，成功样本跳过，失败/中断样本保留原调用。`max_attempts` 是每条样本整个 run 的累计尝试上限，resume 不重置；要改变预算需新建 run。`max_total_attempts` 是运行级请求上限。没有价格配置时费用是 NULL，不是零。可在运行配置中设置 `pricing`，含 `currency`、`input_per_million`、`output_per_million` 和可选 `cached_input_per_million`，价格及计算输入均保存；估算不代表供应商账单。

按 Ctrl+C 暂停后可恢复。`cancel --run-id RUN_ID` 用于没有活跃调度器的已准备或暂停运行。一个实验目录只允许一个调度进程，HTTP 请求可以并发，SQLite 写入在主调度线程串行完成。超时或派发后进程退出可能导致远端结果未知；重试不承诺供应商只执行或计费一次。

## 评分与比较

```bash
.venv/bin/python -m emotion_lab reparse --run-id RUN_ID --parser-version labels-v2
.venv/bin/python -m emotion_lab evaluate --run-id RUN_ID --config config/evaluation.json
.venv/bin/python -m emotion_lab compare --baseline EVAL_A --candidate EVAL_B --seed 42 --repeats 1000
```

评分配置可指定 `zero_division`（0 或 1）、`parser_version`、`scope`。重解析用当前代码重新解析已保存的响应并登记新版本；应随解析代码变更修改版本号，不能把相同解析器的不同名字当成方法创新。默认评分引用运行原先选定的有效预测；指定 parser_version 时按调用顺序选择该解析版本的首个有效结果。旧评分不改变。

主指标为固定 28 类（含 neutral）的 Macro-F1、Micro-F1，另保存逐类 P/R/F1/support、family 指标、精确标签集合匹配、执行失败率及缺失预测率。失败/缺失预测按空集合评分，不丢弃分母。200 条开发集属于诊断切片；未完成运行只能输出 partial 诊断。官方完整评分必须覆盖完整 split。

比较严格按相同数据版本与样本 ID 配对，并核对评分配置；当前提供 Micro-F1 差值的配对 percentile bootstrap 区间。不同样本集不会自动取交集。正式 test 前应冻结协议；仅在 dev 上确定混淆对和参数。

## 数据查看、备份与恢复

23 张表的字段、主外键、状态与约束见 [数据库设计](docs/superpowers/specs/2026-09-08-emotion-experiment-platform-design.md)，实际 SQL 见 [001_initial.sql](emotion_lab/storage/001_initial.sql)。SQLite 工具可以直接查询：

- `v_run_summary`：运行进度、已知 Token 和未知用量/费用调用数。
- `v_call_audit`：每次实际调用、原始请求与响应工件位置。
- `v_evaluation_errors`：原文、真值、预测、漏报和误报。

较小工件存于 `artifacts.inline_bytes`，较大工件存于 `data/research/artifacts/` 的 gzip 文件，关联压缩前后哈希。供应商实际返回的 reasoning 才会保存，隐藏的内部推理不会被补造。

```bash
.venv/bin/python -m emotion_lab db status
.venv/bin/python -m emotion_lab storage verify
.venv/bin/python -m emotion_lab storage backup --output data/research/backups/snapshot-01
.venv/bin/python -m emotion_lab --data-dir data/research/backups/snapshot-01 storage verify
```

备份通过 SQLite Backup API 取得快照，再收集该快照引用的工件，附带哈希清单。备份目录自动以只读方式打开，校验会同时检查 `backup_manifest.json` 中的文件哈希。恢复时先验证备份，再把其中的 `experiments.sqlite3` 和 `artifacts/` 复制到一个新的工作数据目录（不把 `backup_manifest.json` 放入工作目录），然后以该目录作为 `--data-dir` 继续；保留原始备份，不覆盖唯一原库。CSV/JSONL 导出用于分析，完整证据迁移使用 backup。校验报告会列出损坏、缺失和孤儿文件；不会自动清理证据。

[本次实施与验收记录](docs/research/implementation-verification.md)。未运行真实模型的离线测试结果不能当作情绪识别效果。

[离线阶段记录](docs/research/prompt-examples-validation-20260908.md)与[GLM-5.3 真实对照结果](docs/research/glm-dev200-results-20260908.md)。两组各 200 条真实预测、评分与详细数据入库已完成；Micro-F1 从 39.53% 到 40.62%，95% 差值区间包含 0，目前不能确认新版更优。

## 语义检索实验

`method=semantic` 使用固定 BGE 编码的归一化 float32 矩阵做精确余弦检索。配置必须明确 `embedding.index_id` 与 `embedding.queries_artifact_id`；查询向量提前冻结，正式运行和恢复不会调用编码器。`ranked` 与 `contrastive-v1` 均可使用语义检索，词面方法继续保留。情绪标签不参与文本向量编码；对比示例选择仍读取训练样本的完整标签。

训练索引保存在 `embedding_indexes`、`embedding_items` 和矩阵工件中，读取时核对语料顺序、原文指纹、向量形状、归一化状态与逐行哈希。模型权重和配置也归档为工件。`make_set` 的 `exclude_set_ids` 可按规范化文本排除既有冻结集合，用于构造不与旧评估文本重复的开发切片。

本次四组协议：A=词面/普通排序，B=语义/普通排序，C=词面/现有对比规则，D=语义/现有对比规则。固定 v3、GLM-5.3、候选 50、示例 4；新 dev200 从未用于旧实验的开发记录中取样，并排除训练集文本复制。首次准备需要把固定 BGE 修订 `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` 的权重与配置下载到 `data/research/models/bge-base-en-v1.5-a5beb1e/`。

```bash
PYTHONPATH=. .venv/bin/python scripts/build_semantic_experiment.py
PYTHONPATH=. .venv/bin/python scripts/run_semantic_experiment.py --write-configs
# 核验配置、运行测试并提交代码后，再冻结四组并实际调用模型。
PYTHONPATH=. .venv/bin/python scripts/run_semantic_experiment.py --env-file .env
PYTHONPATH=. .venv/bin/python scripts/analyze_semantic_experiment.py
PYTHONPATH=. .venv/bin/python scripts/report_semantic_experiment.py
```

运行脚本保留 `semantic_experiment_state.json`，重复执行会恢复已有运行，不覆盖历史预测。按同一评估切片比较 B-A、D-C，并用 D-B、C-A 分析对比规则；全部组完成后才评分。真实模型结果与软件测试分别报告，见 [实施计划](docs/superpowers/plans/2026-09-08-semantic-retrieval.md)。
