# 语义检索四组对照实施计划

**目标：** 完成已批准的语义检索改造，并在新的固定 dev200 上运行 A/B/C/D 共 800 条正式预测。

**设计依据：** 本任务对话已批准的六步方案。复用当前隔离研究分支和 SQLite；模型参数、v3 指令、28 类、k=4、候选 50、评分和重试预算固定。A=lexical/ranked，B=semantic/ranked，C=lexical/contrastive-v1，D=semantic/contrastive-v1。不修改情绪定义或混淆对。

**实现：** 使用固定修订的 BGE-base-en-v1.5 编码训练原文，归一化 float32 矩阵做精确余弦检索。索引与样本映射通过现有 embedding_indexes / embedding_items 保存，完整矩阵作为不可变工件。查询向量预先编码并保存，正式预测只读取冻结工件，支持复现和恢复。

**执行方式：** 按计划逐项本地完成，不使用子代理。已获实施授权，不重复询问。

## 约束与验收

- 原工作区 main 保持不变；研究分支不合并、不推送。
- 冻结已有官方 train；新评估集从未用于旧实验的 dev 文本中确定性抽样，排除与旧评估文本重复的记录。不得读取 test 进行选择或优化。
- 编码输入只有原文；保存模型提交、文件哈希、编码参数、长度及截断记录、设备、依赖、耗时和向量指纹。
- 查询向量不得进入训练索引；检索继续排除查询复制和重复文本，同分按 source_id 排序。
- 主比较为 B-A 与 D-C；D-B 与 C-A 为对比规则消融。先完成全部组再评分，不根据中间结果改方法；配对 bootstrap 仅作为本轮开发切片证据。
- 每组 200 条、并发 4、每条最多 2 次尝试、每组最多 400 次尝试；失败计入分母，费用未知时保留 NULL。

## 任务 1：向量索引与冻结查询

文件：新增 emotion_lab/embeddings.py、scripts/build_semantic_experiment.py、tests/lab/test_semantic_retrieval.py；更新 requirements。

- [x] 核验当前研究工作树干净，基线 64 tests passed。
- [x] 安装并锁定本地推理依赖，取得并固定 BGE 的实际提交和权重。
- [x] 先写失败测试：索引必须覆盖冻结 train，拒绝错序/错文本/零向量/NaN，加载时校验矩阵与每行哈希；query 文本不匹配时拒绝读取缓存。
- [x] 实现 build_index(store, corpus_id, records, vectors, model_config, metadata) -> index_id；load_index(store, index_id, corpus_id) 返回经过核验的行序和矩阵。
- [x] 实现 create_query_bundle / load_query_bundle，以 text_sha256 关联编码原文和向量，不依赖待测标签。
- [x] 构建并核验真实训练矩阵、新 dev200 查询矩阵，冻结输入清单与模型文件指纹。

## 任务 2：接入检索与真实运行

文件：emotion_lab/config.py、retrieval.py、preparation.py、runner.py；新增四组 JSON 配置及 scripts/run_semantic_experiment.py。

- [x] 先写失败测试：支持 semantic/ranked 与 semantic/contrastive-v1，缺索引配置必须失败；语义邻居改变但完整训练标签不变；复制过滤和预算继续生效。
- [x] Retriever 接受可选冻结语义资源；select 使用归一化向量点积并记录 index_id、query 工件、全量分数和实际选择。
- [x] prepare_run 与 execute_run 读取同一冻结索引和查询工件，experiment_runs.index_id 与配置相符；恢复不重新编码查询。
- [x] 使用假传输完成有意义的整条流程测试，断点恢复不产生重复请求；旧检索和旧指令回归通过。
- [x] 运行完整测试、diff 检查，提交冻结代码，保存四组驱动源与配置后开始真实调用。
- [x] 四组各 200 条完成后统一评分、导出与四个配对比较；运行过程持续保存进度和每次调用。

## 任务 3：独立验证与交付

文件：新增 scripts/analyze_semantic_experiment.py、docs/research/semantic-retrieval-results-20260908.md。

- [x] 从原始 train/dev TSV 核对文本和真值；逐条核对原始响应、数据库最终标签、4 条示例及 50 个候选。
- [x] 独立重算四组指标与向量排序；检查各组除检索方法和示例政策外条件一致，实际 instruction 完全相同。
- [x] 报告 Micro/Macro-F1、exact match、precision/recall、family 指标、每类变化、示例变化、token 与检索/模型耗时，并明确小样本和重复调用局限。
- [x] 完整性核验 SQLite、全部工件、导出及可恢复备份；保留负面结果；提交报告并检查工作树干净。


## 实际完成记录

- 实现提交：`eb75829`；四组运行冻结提交：`e991d09`。实际 `emotion_lab` 源码快照一致，runner 复用原逻辑，无需修改。
- BGE 固定修订：`a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`。43,410 条 train、200 条新 dev 查询均已编码，截断数均为 0；11 个模型文件哈希、全部训练向量与原始编码输出核验通过。
- 新 dev 集：`d39139c5-548a-47f8-bbd1-2762b43d1cd2`，从 5,177 条合格 dev 文本以 seed=90908 确定性选择。索引：`ce150a15-47f3-4046-891e-144aa575d09b`。
- 四组实际共 800 次调用，800 条成功、0 条失败、0 次重试。A/B/C/D Micro-F1 为 41.67% / 40.83% / 41.93% / 42.65%；B-A 与 D-C 的 95% 差值区间均包含 0，未确认稳定收益。详见 [完整结果](../../research/semantic-retrieval-results-20260908.md)。
- 独立核验：800 份请求及原始预测、40,000 条候选、3,200 条入选示例、800 份完整分数向量；原始 TSV、冻结文本、完整标签和数据库评分一致。软件测试 81 passed，`git diff --check` 通过。
- 最终核验工件：`fc967def-3a20-4b25-8896-079047fc0081`。活跃库校验 21,102 个工件通过；加入最终证据后，可恢复备份校验 21,104 个工件通过，均无错误或孤儿文件。
- 备份：`data/research/backups/semantic-retrieval-final-20260908/`。已只读重新打开并验证 train/query 向量与四组运行，可携带 SQLite 和关联工件恢复；保留此前所有实验。
- 原工作区 main 保持 `1af9d06` 且干净；研究分支保留，不合并、不推送。最终报告提交包含完成记录，工作树状态在提交后检查。
