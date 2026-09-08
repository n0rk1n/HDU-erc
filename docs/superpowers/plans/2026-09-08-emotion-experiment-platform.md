# 情绪实验工程实施计划

> 使用 executing-plans 在本会话顺序执行；用户已批准架构和继续实施，不启用子代理。

**目标：** 将研究分支改为可以导入 GoEmotions、运行现成模型、记录完整证据并复算评测的命令行工程。

**架构：** 独立 SQLite、不可变哈希文件、逐条执行租约。调用之前持久化请求，响应在解析前落盘；所有评分使用冻结预测快照。

**技术：** Python 3.12、sqlite3、httpx、现有官方 tokenizer；pytest 离线验证。删除聊天服务依赖。

**设计：** `docs/superpowers/specs/2026-09-08-emotion-experiment-platform-design.md`（用户已批准）。

## 全局约束

- 继续 `codex/goemotions-config-20260908` 隔离工作区；不合并、推送或触碰主工作区数据。
- 23 张表、真实多标签、train-only corpus、固定样本集合、未知费用 NULL。
- GoEmotions 固定源版本，数据导入幂等；每次重试独立记录。
- 先离线验收，初始化不能触发付费调用；dense/contrastive 尚未实现必须明确报错。

## 任务 1：存储和数据集

文件：`emotion_lab/storage/{database.py,artifacts.py,001_initial.sql}`、`emotion_lab/datasets.py`、`tests/lab/test_storage.py`。
接口：`Store(root, db_path=None, initialize=False)`；`put(data, kind)` / `read(artifact_id)`；`import_goemotions(store, source)` 返回版本 ID；`make_set(store, config)` 返回集合 ID。

- [ ] 写真实临时 SQLite 测试，验证旧库拒绝初始化、工件损坏检测、23 表、跨版本标签拒绝、冻结集合拒绝、train-only。

```python
with pytest.raises(sqlite3.IntegrityError):
    store.db.execute("UPDATE samples SET raw_text='changed' WHERE sample_id=?", (sample_id,))
assert store.read(artifact_id) == b'original bytes'
```

- [ ] 运行 `python -m pytest tests/lab/test_storage.py -q`，先确认缺失功能失败。
- [ ] 按设计创建注释完整的 SQL 表、约束、触发器及视图；实现带校验的初始化、事务事件、工件、校验、备份。
- [ ] 从数据提交 `3fa3928` 只恢复 `data/benchmarks/goemotions/`，实现来源校验、导入与集合冻结。
- [ ] 测试通过后提交存储阶段。

## 任务 2：模型与推理执行

文件：`emotion_lab/{config.py,taxonomy.py,retrieval.py,runner.py}`、`emotion_lab/llm/`、`tests/lab/test_runner.py`。
接口：`prepare_run(store, config, dry_run=False)`；`execute_run(store, run_id, transport=None)`；`parse_attempt(store, attempt_id, parser_version)`；模型传输接收实际序列化 request bytes，返回 status、headers 和 raw bytes。

- [ ] 写离线 HTTP 传输测试，返回固定多标签、非法 JSON、HTTP 错误、超时、无 usage；断点注入到派发与响应落盘边界。

```python
assert item['status'] == 'succeeded'
assert attempt['total_tokens'] is None
assert len(store.rows('SELECT * FROM call_attempts')) == 2
```

- [ ] 运行 `python -m pytest tests/lab/test_runner.py -q` 确认失败。
- [ ] 提取现有供应商参数、脱敏与官方 tokenizer；用 httpx 直接捕获完整响应，关闭隐式重试。
- [ ] 实现 labels 数组验证，zero-shot/random/lexical，完整候选与预算裁剪记录；不读取待测标签。
- [ ] 实现冻结配置、租约、调用审计、故障恢复、重解析，成功项恢复时跳过。
- [ ] 通过持久化证据断言后提交执行阶段。

## 任务 3：评测、比较和 CLI

文件：`emotion_lab/{evaluation.py,cli.py,__main__.py}`、`tests/lab/test_evaluation.py`、`tests/lab/test_cli.py`。
接口：`evaluate(store, run_id, config)`；`compare(store, baseline, candidate, seed, repeats)`；`export_evaluation(store, evaluation_id, output)`。

- [ ] 手算夹具验证真值 `{0,1}` 与预测 `{0}` 的 TP=1、FP=0、FN=1；空预测保留分母；配对比较必须严格对齐。
- [ ] 运行对应测试确认失败，再实现固定标签全集 Macro/Micro-F1、逐类与 family 指标、快照、配对 bootstrap。
- [ ] 实现 init/import/make-set/run/dry-run/resume/evaluate/compare/export/verify/backup，增加 reparse/cancel/status 操作。
- [ ] CLI subprocess 离线流程验证返回码、数据行数与不发生调用。

## 任务 4：清理、初始化和交付

文件：README、依赖、环境示例、启动脚本、实验配置、验收报告；删除设计所列旧 tracked 文件。

- [ ] 保存仍有价值的 tokenizer/脱敏测试与有来源的 56 条烟测夹具。
- [ ] 仅删除 Git 跟踪的旧聊天/归档/文档/测试，保留新模块和本次文档。
- [ ] 运行完整新测试、官方数据验证、CLI init/import、200 dev dry-run、备份还原校验。
- [ ] 核对主工作区 Git 状态和原数据文件哈希未变；记录实际库统计与测试结果。
- [ ] `git diff --check`、审阅改动、任务提交。最终说明仅离线验证，真实模型实验尚未执行。
