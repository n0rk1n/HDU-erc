# 提示词与对比示例四组实验实施计划

**目标：** 在当前隔离研究分支及独立 SQLite 中，完成用户批准的四组真实准确性实验，共 800 条预测。

**设计依据：** 本任务上一轮已明确的“原指令＋普通示例、优化指令＋普通示例、原指令＋对比示例、优化指令＋对比示例”。复用现有实现，不新增提示词版本、不训练模型。

**执行方式：** 当前会话按 writing-plans / executing-plans 逐项执行；遵守用户限制，不使用子代理。复用同一研究任务工作树，不合并、不推送。

## 固定协议

- A=v1/ranked；B=v3/ranked；C=v1/contrastive-v1；D=v3/contrastive-v1。v3 是此前按 train 优化的指令版本，“优化”表示来源，不预设效果更好。
- 四组统一 semantic 方法，复用 BGE-base-en-v1.5 固定修订 `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` 与 train 索引 `ce150a15-47f3-4046-891e-144aa575d09b`。候选 50，示例 4，完整原生 28 类标签，原有混淆对与去重政策不变。
- 新 dev200：seed=90909，排除所有此前已运行 dev 集合与规范化 train 文本；test 不用于抽样、调参或推理。新查询单独编码，不重建训练向量。
- 固定 `ZHIPU/GLM-5.3`、既有供应商地址、temperature=0、thinking=enabled、reasoning_effort=low、max_tokens=8192、context_tokens=32768、safety_tokens=256、timeout=120 秒、concurrency=4；每条最多 2 次尝试、每组最多 400 次尝试。价格未知保留 NULL。
- 先冻结四组全部配置与代码，按 seed=90909 打乱组顺序执行，全部完成后统一评分；每组均重新请求，不挪用旧响应。原始请求、响应、用量、完整检索得分、50 个候选及 4 条入选示例全部入库。
- 主要指标为 Micro-F1，另报告 28 类 Macro-F1、完全匹配、P/R、至少一个标签命中、大类指标与逐类误差；失败按空预测计入分母。
- 分别比较 B-A、D-C（指令作用），C-A、D-B（示例作用），D-A（组合变化）。配对 bootstrap seed=42、1000 次、95% 未校正区间。四组同一抽样计算交互效应 `(D-C)-(B-A)`，避免把 D 组最高误称为协同。
- 实际 A/B 入选示例及顺序必须相同，C/D 同样；同一指令在两种政策下内容一致。实际输入长度不同是指令因素的一部分，token 上限相同；报告真实用量，不宣称等 token。
- 结论限于本轮新切片与已冻结版本；单次服务调用不能覆盖重复波动，旧 dev200 分数不能直接相减。

## 任务 1：准备与验证协议

文件：新增 `scripts/build_prompt_factorial.py`、`scripts/prompt_factorial_stats.py`、`tests/lab/test_prompt_factorial.py`。

- [x] 核验当前研究树干净、无未完成运行；保存原工作区未提交删除清单，原样保留。基线 81 tests passed。
- [ ] 先写并观察失败测试：交互差值手算校验、同样本配对、空样本拒绝、四组共享 bootstrap 抽样；拒绝不同模型/样本/索引/预算或因指令造成的示例变化。
- [ ] 实现 `validate_factorial(configs)`、`validate_requests(requests)`、`factorial_bootstrap(records, seed=42, repeats=1000)`，所有统计不调用模型。
- [ ] build 脚本复用已有固定模型文件与 `encode`，校验 11 个文件哈希及依赖，确定性新抽 dev200，创建独立查询工件；保存选择、编码与源代码证据，保持旧工件不变。

## 任务 2：冻结与运行

文件：新增 `scripts/run_prompt_factorial.py`、`config/experiments/dev200-prompt-factorial-{A,B,C,D}.json`。

- [ ] 复用 prepare_run / execute_run / evaluate / compare 的恢复和存证机制，仅改变实验名、指令维度与样本配置；已有状态及配置必须匹配才能恢复。
- [ ] 四组配置通过协议校验；运行全套测试与 diff 检查，提交代码和配置后冻结运行。
- [ ] 运行命令：`PYTHONPATH=. .venv/bin/python scripts/run_prompt_factorial.py --env-file /Users/oriki/Database/HDU-erc/.env`。凭据只载入进程，不进入日志或工件。
- [ ] 800 条真实预测结束后统一评分，生成 5 个配对比较和每组标准导出；进度写入 `data/research/prompt_factorial_execution.log`。

## 任务 3：独立分析与交付

文件：新增 `scripts/analyze_prompt_factorial.py`、`scripts/report_prompt_factorial.py`、`scripts/verify_prompt_factorial.py`、`docs/research/prompt-factorial-results-20260909.md`。

- [ ] 基于原始 train/dev TSV、响应 JSON 和数据库逐条重算评分；独立重算余弦分数，核对 40,000 条候选、3,200 条示例及全部实际指令。
- [ ] 计算配对交互效应与区间，保存四组配对明细、1000 次共同抽样差值、输入配置和统计源码；指标关联原始 evaluation IDs。
- [ ] 报告真实四组指标、5 项差值、交互效应、逐类变化、用量、成功失败与检索诊断，不自动更改默认配置。
- [ ] 核验 SQLite 与完整工件；创建 `data/research/backups/prompt-factorial-final-20260909/` 并只读恢复核验。验证软件测试、源代码快照及导出哈希。
- [ ] 提交报告和完成记录；检查研究树干净、原工作区删除状态未改变。交付分支、工作树、未合并状态及报告链接。
