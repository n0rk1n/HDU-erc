# 训练集驱动的 Prompt v3 优化计划

**目标：** 根据 GoEmotions train 的统计与原始标注边界改写 prompt，在同一 dev200 上独立验证 prompt 效果，保存完整证据。

**实现：** 继续现有隔离研究分支；添加 `native-labels-v3`，保持 v1/v2 原文、taxonomy、检索、示例预算、模型参数、解析器及评分器不变。沿用独立 SQLite 的不可变工件与实验表。

**技术：** Python、pytest、SQLite、现有 GLM-5.3 API。

**需求来源：** 用户要求回到训练集优化 prompt，提高准确率；沿用已经授权的模型和 200 条开发集对照。

## 约束

- 训练审计脚本只读取 train.tsv、emotions.txt、manifest.json；不把 dev/test 文本或真值放进 prompt。
- 前轮 dev200 错误已被观察，因此本轮属于开发迭代；不能声称是未见过的独立验证。官方 test 保留。
- 本轮只冻结一个 v3 候选，不根据本轮 dev 得分反复挑选版本。
- 两组固定 contrastive-v1、k=4、GLM 参数；重新运行 v2 控制组，以减少跨轮服务变化的影响。
- 保留全部失败、重试、usage、原始请求响应、训练来源、代码与 prompt 快照；费用未知仍为 NULL。
- 不合并、不推送，不改写已完成实验。

## 步骤

- [x] 确认原工作区及现有研究工作树干净，基线 59 项测试通过。
- [x] 新建 `scripts/audit_train_prompt_v3.py`，输出完整类别/共现/词形切片统计及可复查分层训练样本；保存审计工件。
- [x] 用训练审计写明规则依据；先扩展 `tests/lab/test_prompt_examples.py`，验证 v3 实际请求及冻结快照一致、示例选择不受影响。
- [x] 修改 `emotion_lab/taxonomy.py` 和 `emotion_lab/config.py`；新建两份 `config/experiments/dev200-prompt-v3-*.json`，保持其他配置相同。
- [ ] 运行完整 pytest 和 diff 检查，提交冻结实现。
- [ ] 冻结并执行两组各 200 条真实预测；逐条核验入选示例相同，再按原规则评价、配对 bootstrap、导出并入库。
- [ ] 写明实际增益或退步、调用代价与开发集适用边界；核验数据库及可恢复备份，提交报告，保留工作树。
