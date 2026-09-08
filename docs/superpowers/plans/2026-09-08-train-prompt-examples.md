# 训练集示例与提示词对照实验实施计划

**目标：** 保留原版词面检索和提示词，增加训练集对照示例及标签区分说明，在已冻结的 dev200 上完成真实模型对照并入库。

**实现范围：** 沿用 emotion_lab 的配置、检索、提示词与审计接口，不修改数据库结构。继续当前隔离研究分支，按用户要求不合并；本任务直接执行，不启用子代理。

**依据：** 用户本轮请求及 `docs/superpowers/specs/2026-09-08-emotion-experiment-platform-design.md`。

## 实验约束

- 训练来源固定为 `goemotions-train`；评价集合固定为 `goemotions-dev200`，seed=42。构建方法期间不查看开发集真值，官方 test 不用于本轮。
- 基线为原始 native-labels-v1 + lexical ranked；候选为 native-labels-v2 + lexical contrastive-v1。两组均 k=4、candidate_count=50、相同模型/温度/推理档位/预算/重试上限。
- 对照策略只在词面前 50 条中，从第一个可用锚点的语义相邻标签寻找一条不同标签示例，要求正词面得分、原文去重且能放入预算。无适合示例则回退原排序。其余位置保持词面顺序。
- 语义相邻关系是人工方法先验，不把标签共现当作模型混淆，也不手工更改训练标签。
- 本轮两项一起修改，只报告组合效果，不单独归因、不声称论文创新已验证。
- 凭据仅从用户选定服务对应的本地环境读取；当前 GLM 与 DeepSeek 模板的差异必须等用户选择后再发请求。未核实价格时成本保留 NULL。

## 执行步骤

- [x] 新增 `tests/lab/test_prompt_examples.py`：真实临时数据库验证对照示例选择、原标签保留、重复与零相关回退、预算保护、配置拒绝错误组合、请求与提示词快照一致。先执行 `.venv/bin/python -m pytest -q tests/lab/test_prompt_examples.py`，确认新增功能缺失导致失败。
- [x] 修改 `emotion_lab/config.py`：验证 prompt_version 和 example_policy；默认保留 v1/ranked。修改 `emotion_lab/taxonomy.py`：按版本构建提示词并提供完整可审计快照。修改 `emotion_lab/retrieval.py`：加入确定性对照排序及原排名、相邻标签、锚点等审计字段。修改 `emotion_lab/preparation.py`：快照记录完整提示词规则与选择策略。
- [x] 从 SQLite train 生成类别统计和真实示例检查记录，写入 `docs/research/train-prompt-examples.md`；创建本轮两组配置。执行全套 pytest、两组 dry-run、git diff --check；核对改动后提交代码。
- [ ] 模型确认后，以固定代码创建并运行两个各 200 条的实验。逐步检查 HTTP 和解析结果；错误请求与重试全部保留。出现服务或凭据系统性错误及时暂停，不伪造预测。
- [ ] 对实际完成结果调用 evaluate、compare（配对 bootstrap，seed=42，1000 次）、export。将训练审计及中文结果报告作为 artifacts 存入 SQLite，报告 Micro/Macro F1、准确集合匹配率、失败数、调用数、token、置信区间和限制。
- [ ] 核验存储完整性，制作可恢复备份；提交结果文档。最终提供数据库、报告、run/evaluation ID、验证结果及尚未合并的分支信息。

## 当前执行状态

已完成代码、训练审计、两组 200 条离线验证和本地提交。真实模型、评分与比较步骤仍未执行：等待用户回复已发出的模型选择问题。离线验收及数据库内工件位置见 `docs/research/prompt-examples-validation-20260908.md`。
