# 项目结构整理与无用内容清理设计

## 背景

当前项目已经形成 Web 应用、情绪识别、长期记忆、公开数据基准、消融实验和导师汇报的完整闭环，但目录逐步积累了以下问题：

- `chatbot/` 下二十余个模块平铺，领域边界主要依赖文件名和 README 说明；
- `scripts/` 同时放置单组评估、消融运行、Codex CLI 运行和报告脚本；
- `tests/` 下三十余个测试文件平铺，无法从目录直接看出所属领域；
- `docs/superpowers/` 混合当前设计、已经完成的实施计划和已退役方案；
- README 的项目结构仍提到已删除的 Emotion Ablation V2 合成基准；
- 工作区存在可再生成缓存、macOS 元数据和历史 worktree 副本。

本次整理只改善结构、文档与仓库卫生，不改变运行逻辑、接口语义、数据格式或实验口径。

## 目标

1. 让应用、实验脚本、测试和文档的目录直接体现职责边界。
2. 删除可证明可再生成、已被替代或已经失效的内容。
3. 保留公开基准的复现能力、历史实验结果和用户运行数据。
4. 保持 `uvicorn chatbot.web:app` 启动入口及现有 HTTP/SSE 行为不变。
5. 保持完整测试基线，不通过删除测试或弱化断言完成整理。

## 非目标

- 不采用 `src/` 布局，不引入 `pyproject.toml`，不更换依赖管理方式。
- 不重写大文件内部逻辑，不修复当前 FastAPI/Starlette 弃用警告。
- 不修改情绪标签、Prompt、消融配置、Accuracy 或 Macro F1 计算方式。
- 不迁移、删除或清空 `data/records/` 中的运行数据库和实验输出。
- 不合并、推送或删除尚未合并的 Git 分支。

## 当前盘点结论

### 应保留

- `chatbot/` 下现有 Python 模块均被应用、脚本或测试引用，没有可直接判定为死代码的业务模块。
- `data/benchmarks/empathetic_dialogues_v1/` 包含正式基准、平衡 seed、few-shot、许可和报告，是实验复现链路的一部分。
- `deliverables/advisor_full_summary_20260728/` 是当前完整导师汇报成品。
- `data/records/` 是忽略提交的用户运行数据与实验输出，不属于可再生成缓存的统一清理范围。
- `codex/advisor-report-ppt-20260727` 含尚未进入 `main` 的独立提交，必须保留分支和对应 worktree 内容。

### 可清理

- `.DS_Store`、`__pycache__/` 和 `.pytest_cache/` 等可再生成文件。
- 未跟踪且已失效的 `docs/superpowers/plans/2026-07-02-emotion-ablation-v2.md`。
- 只描述已退役合成 benchmark 且引用已删除文件的旧设计或实施计划；这些内容仍可从 Git 历史恢复。
- 已合并、干净且路径登记已经失效的 `codex/regenerate-readme-20260724` worktree 副本和登记。

### 暂不删除

- `.idea/`：它是本地 IDE 配置，不影响仓库提交，但不能证明对用户无用。
- 仍与当前功能对应的历史设计文档：移动到清晰目录，不当作垃圾删除。
- 旧导师汇报设计材料：与研究过程有关，统一归档而不是直接删除。

## 目标结构

```text
chatbot/
  web.py                         FastAPI 稳定入口
  chat_service.py                聊天业务编排
  core/
    config.py                    应用与模型配置
    llm.py                       Prompt、chain 和会话装配
    llm_adapter.py               OpenAI-compatible 适配器
    prompt_config.py             可覆盖 Prompt 配置
    runtime_store.py             通用 SQLite 运行时存储
    history.py                   消息、反馈和重新生成记录
  emotion/
    analysis.py                  情绪调用与分析记录
    state.py                     结构化情绪状态
    labels.py                    32 类标签和情绪族
    prompt.py                    情绪 Prompt 组装
    examples.py                  静态示例
    retrieval.py                 动态示例选择
    feedback.py                  情绪正确性反馈
    safety.py                    安全级别与回复提示
  memory/
    models.py                    长期记忆协议、模型和配置
    sqlite.py                    SQLite 记忆实现
    extractor.py                 单轮记忆抽取
    consolidation.py             周期性记忆提炼
  profile/
    repository.py                用户画像持久化
    onboarding.py                首次画像问题和草稿
  static/                        无构建前端

scripts/
  benchmark/                     公开数据转换、校验、统计和导出
  ablation/                      消融运行、评估和报告

tests/
  app/                           Web 与聊天编排
  core/                          配置、LLM、Prompt 和运行时存储
  emotion/                       情绪、安全和反馈
  memory/                        记忆抽取、存储和提炼
  profile/                       用户画像与首次录入
  scripts/                       benchmark 与 ablation CLI

docs/
  design/                        仍对应当前系统的设计文档
  archive/                       已完成计划和历史研究材料
```

根目录的 `.env.example`、`requirements.txt`、`pytest.ini`、`README.md`、`AGENTS.md` 和工具配置保持原位。`data/` 与 `deliverables/` 已有边界明确，不做无意义搬迁。

## 兼容性设计

### 应用入口

`chatbot.web` 和 `chatbot.chat_service` 保持在包根目录，避免改变 Uvicorn 启动命令和主要服务入口。内部领域模块改用新的包路径。

### 包公开接口

`chatbot.emotion` 与 `chatbot.memory` 由单文件改为包。各包的 `__init__.py` 只导出目前确有跨领域调用的稳定符号，不为未使用内部实现创建兼容别名。

### 脚本入口

消融脚本移动到 `scripts/ablation/`，文件名和 CLI 参数保持不变。README、测试中的子进程路径和模块导入同时更新。benchmark 脚本已经分组，不再改名。

### 数据和输出

所有默认数据路径保持不变：配置继续位于 `data/config/`，样例位于 `data/examples/`，公开基准位于 `data/benchmarks/`，运行输出位于 `data/records/`。

## 文档整理

- 把仍与当前实现一致的设计说明移动到 `docs/design/`。
- 把已完成但仍有研究过程价值的计划、旧导师汇报设计移动到 `docs/archive/`。
- 删除只服务于已退役 Emotion Ablation V2 合成数据、且引用对象已经从仓库移除的专用设计与实施计划。
- README 更新启动命令、实验脚本路径、项目结构和 benchmark 描述，不再声称仓库保留合成诊断集。
- 新增简短的 `docs/README.md`，说明 `design/` 与 `archive/` 的边界，避免历史材料被误当成当前操作指南。

## 本地垃圾与 Git 状态清理

本次只处理经过明确验证的目标，不使用覆盖面过大的 `git clean -X`：

1. 精确删除可再生成的 `.DS_Store`、`__pycache__/`、`.pytest_cache/`。
2. 保留 `.idea/` 和 `data/records/`。
3. 对历史 worktree 先确认工作区干净、分支合并状态和提交可达性。
4. 只移除已经合并的 `regenerate-readme` worktree 副本及其失效登记。
5. 保留未合并的 `advisor-report-ppt-20260727` 分支、提交和 worktree 内容。

## 实施顺序

1. 建立路径映射清单，先移动应用模块并更新内部导入。
2. 更新应用相关测试路径和导入，运行应用与领域测试。
3. 移动消融脚本并更新脚本间导入、README 命令和 CLI 测试。
4. 按当前/历史边界整理文档，删除已退役材料，修正 README。
5. 精确清理可再生成文件和已合并的旧 worktree。
6. 运行完整验证，检查 Git diff、状态和提交范围。

每一步都必须保持可独立验证；出现导入错误或行为测试失败时，在进入下一步前修复。

## 验证标准

基线命令：

```bash
uv run --with-requirements requirements.txt pytest -q
```

当前基线为 380 个测试通过、99 条既有弃用警告。最终必须满足：

- 全量 pytest 不少于原有 380 个测试，且无失败；
- `uvicorn chatbot.web:app` 的导入检查成功；
- 所有保留的 benchmark CLI 和移动后的 ablation CLI 的 `--help` 成功；
- README 引用的仓库内路径全部存在；
- `rg` 不再发现对已删除 Emotion Ablation V2 文件或旧脚本路径的当前操作指引；
- `git diff --check` 通过；
- 任务分支只包含本次结构整理、文档更新和无用内容删除。

既有 FastAPI/Starlette 弃用警告记录为基线，不在本任务中扩展为行为修复。

## 风险与控制

### 导入路径迁移

风险最高的部分是 Python 模块移动。控制方式是按领域逐组移动，每组移动后先运行对应测试，再运行全量测试；同时通过包公开接口减少跨领域模块对内部文件的依赖。

### CLI 路径变化

脚本移动会影响 README 命令和子进程测试。所有 `scripts/*.py` 引用必须一次性检索并更新，CLI 参数和输出格式保持原样。

### 历史材料误删

只删除已经被正式 benchmark 退役且引用对象不存在的材料。仍能解释当前系统设计或研究演进的内容移动到归档目录。所有版本化删除都可从 Git 历史恢复。

### 本地数据误删

禁止使用会覆盖 `data/records/` 的批量清理命令。运行数据库、Codex CLI 输出和实验报告不纳入垃圾清理。

## 完成定义

当目标目录结构落地、无用内容按上述边界清理、README 与实际路径一致、完整测试和 CLI smoke test 通过，并且任务改动已经提交到独立分支时，本任务完成。任务分支在用户明确要求合并前保持独立，不自动推送、合并或删除 worktree。
