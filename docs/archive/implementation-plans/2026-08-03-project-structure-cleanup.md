# 项目结构整理与无用内容清理实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变应用行为、数据格式和实验口径的前提下，把应用模块、实验脚本、测试与文档整理为按职责分组的清晰结构，并删除已确认无用的内容。

**Architecture:** 保留 `chatbot.web`、`chatbot.chat_service` 和 `chatbot.main` 三个稳定入口，把内部模块迁入 `core`、`emotion`、`memory`、`profile` 四个领域包。消融脚本迁入 `scripts.ablation` 并统一使用 `python -m` 调用；测试按对应领域分组。数据集、运行数据和最终交付物保持原位。

**Tech Stack:** Python 3.10+、FastAPI、LangChain、SQLite、pytest、uv、Git worktree。

## Global Constraints

- 以本地 `main` 的 `9d339d3` 为基线，在 `codex/organize-project-structure-20260803` 独立 worktree 中完成。
- 不改变 HTTP/SSE 行为、情绪标签、Prompt、消融配置、Accuracy、Macro F1 或数据 Schema。
- 保持 `uvicorn chatbot.web:app` 启动入口。
- 保留 `data/records/`、`.idea/`、EmpatheticDialogues 基准、最终导师汇报和未合并的 `codex/advisor-report-ppt-20260727`。
- 不引入 `src/`、`pyproject.toml` 或新依赖，不顺带修复既有弃用警告。
- 当前基线必须保持：`uv run --with-requirements requirements.txt pytest -q` 为 380 passed；99 条既有弃用警告允许继续存在。
- 所有纯移动都使用 Git 感知的重命名；所有内容修改使用 `apply_patch`。
- 每个任务完成后先运行聚焦测试，再创建原子提交。

---

## 文件结构映射

### 应用模块

| 旧路径 | 新路径 |
| --- | --- |
| `chatbot/config.py` | `chatbot/core/config.py` |
| `chatbot/llm.py` | `chatbot/core/llm.py` |
| `chatbot/llm_adapter.py` | `chatbot/core/llm_adapter.py` |
| `chatbot/prompt_config.py` | `chatbot/core/prompt_config.py` |
| `chatbot/runtime_store.py` | `chatbot/core/runtime_store.py` |
| `chatbot/history.py` | `chatbot/core/history.py` |
| `chatbot/emotion.py` | `chatbot/emotion/analysis.py` |
| `chatbot/emotion_state.py` | `chatbot/emotion/state.py` |
| `chatbot/emotion_labels.py` | `chatbot/emotion/labels.py` |
| `chatbot/emotion_prompt.py` | `chatbot/emotion/prompt.py` |
| `chatbot/emotion_examples.py` | `chatbot/emotion/examples.py` |
| `chatbot/emotion_retrieval.py` | `chatbot/emotion/retrieval.py` |
| `chatbot/emotion_feedback.py` | `chatbot/emotion/feedback.py` |
| `chatbot/safety.py` | `chatbot/emotion/safety.py` |
| `chatbot/memory.py` | `chatbot/memory/models.py` |
| `chatbot/local_memory.py` | `chatbot/memory/sqlite.py` |
| `chatbot/memory_extractor.py` | `chatbot/memory/extractor.py` |
| `chatbot/memory_consolidation.py` | `chatbot/memory/consolidation.py` |
| `chatbot/profile.py` | `chatbot/profile/repository.py` |
| `chatbot/profile_onboarding.py` | `chatbot/profile/onboarding.py` |

### 消融脚本

以下文件原名不变，全部从 `scripts/` 移入 `scripts/ablation/`：

- `evaluate_emotion_analysis.py`
- `run_emotion_ablation.py`
- `evaluate_emotion_ablation.py`
- `run_codex_cli_emotion_ablation.py`
- `report_codex_cli_emotion_ablation.py`

### 测试目录

- `tests/app/`：`test_web.py`、`test_chat_service.py`、`test_main.py`
- `tests/core/`：`test_config.py`、`test_llm.py`、`test_llm_adapter.py`、`test_prompt_config.py`、`test_runtime_store.py`、`test_history.py`
- `tests/emotion/`：所有 `test_emotion*.py`、`test_safety.py`
- `tests/memory/`：`test_memory.py`、`test_local_memory.py`、`test_memory_extractor.py`、`test_memory_consolidation.py`
- `tests/profile/`：`test_profile.py`、`test_profile_onboarding.py`
- `tests/scripts/`：benchmark、evaluate、run、report 相关测试
- `tests/project/`：`test_readme.py`
- `tests/conftest.py` 保持在测试根目录。

---

### Task 1: 建立 core 包并迁移基础设施模块

**Files:**
- Create: `chatbot/core/__init__.py`
- Move: `chatbot/{config,llm,llm_adapter,prompt_config,runtime_store,history}.py` → `chatbot/core/`
- Modify: `chatbot/main.py`
- Modify: `chatbot/web.py`
- Modify: `chatbot/chat_service.py`
- Modify: 所有引用旧 `chatbot.config`、`chatbot.llm`、`chatbot.llm_adapter`、`chatbot.prompt_config`、`chatbot.runtime_store`、`chatbot.history` 的 Python 文件
- Move tests: 对应文件 → `tests/core/`

**Interfaces:**
- Consumes: 现有 `ChatConfig`、`LlmConfig`、`RuntimeStore`、history 数据结果类型与 LLM 构建函数。
- Produces: `chatbot.core.config`、`chatbot.core.llm`、`chatbot.core.llm_adapter`、`chatbot.core.prompt_config`、`chatbot.core.runtime_store`、`chatbot.core.history`；函数和类型签名保持不变。

- [ ] **Step 1: 创建目录并执行纯移动**

```bash
mkdir -p chatbot/core tests/core
git mv chatbot/config.py chatbot/llm.py chatbot/llm_adapter.py chatbot/prompt_config.py chatbot/runtime_store.py chatbot/history.py chatbot/core/
git mv tests/test_config.py tests/test_llm.py tests/test_llm_adapter.py tests/test_prompt_config.py tests/test_runtime_store.py tests/test_history.py tests/core/
```

- [ ] **Step 2: 增加空包入口并替换全部 core 导入**

创建空的 `chatbot/core/__init__.py`，将旧前缀逐项替换为新前缀；例如：

```python
from chatbot.core.config import ChatConfig
from chatbot.core.runtime_store import RuntimeStore
```

运行 `rg -n "chatbot\.(config|llm|llm_adapter|prompt_config|runtime_store|history)" chatbot scripts tests`，输出必须只包含新 `chatbot.core.*` 路径。

- [ ] **Step 3: 运行 core 与应用导入测试**

```bash
uv run --with-requirements requirements.txt pytest -q tests/core tests/test_main.py tests/test_chat_service.py tests/test_web.py
```

Expected: 所有收集到的测试通过，无 import error。

- [ ] **Step 4: 提交 core 迁移**

```bash
git add chatbot tests scripts
git commit -m "refactor: group core infrastructure modules"
```

---

### Task 2: 建立 emotion 包并迁移情绪领域

**Files:**
- Create: `chatbot/emotion/__init__.py`
- Move: `chatbot/emotion.py` → `chatbot/emotion/analysis.py`
- Move: 其余七个情绪与安全模块 → `chatbot/emotion/`
- Modify: `chatbot/chat_service.py`、`chatbot/web.py`、`scripts/**/*.py`
- Move tests: `tests/test_emotion*.py`、`tests/test_safety.py` → `tests/emotion/`

**Interfaces:**
- Consumes: Task 1 的 `chatbot.core.runtime_store` 和 `chatbot.core.prompt_config`。
- Produces: `chatbot.emotion.analysis`、`state`、`labels`、`prompt`、`examples`、`retrieval`、`feedback`、`safety`；原有函数和类型名不变。

- [ ] **Step 1: 移动情绪模块和测试**

```bash
mkdir -p chatbot/emotion tests/emotion
git mv chatbot/emotion.py chatbot/emotion/analysis.py
git mv chatbot/emotion_state.py chatbot/emotion/state.py
git mv chatbot/emotion_labels.py chatbot/emotion/labels.py
git mv chatbot/emotion_prompt.py chatbot/emotion/prompt.py
git mv chatbot/emotion_examples.py chatbot/emotion/examples.py
git mv chatbot/emotion_retrieval.py chatbot/emotion/retrieval.py
git mv chatbot/emotion_feedback.py chatbot/emotion/feedback.py
git mv chatbot/safety.py chatbot/emotion/safety.py
```

把 `tests/test_emotion.py`、`test_emotion_labels.py`、`test_emotion_prompt.py`、`test_emotion_retrieval.py`、`test_emotion_state.py`、`test_emotion_feedback.py` 和 `test_safety.py` 移入 `tests/emotion/`。

- [ ] **Step 2: 建立包接口并更新导入**

`chatbot/emotion/__init__.py` 只从 `analysis.py` 导出跨领域使用的分析函数，从 `labels.py` 导出标签集合；测试需要 monkeypatch 模块全局量时直接导入 `chatbot.emotion.analysis`，避免 patch 包级副本。

所有旧模块名替换为：

```text
emotion.py            -> emotion.analysis
emotion_state.py      -> emotion.state
emotion_labels.py     -> emotion.labels
emotion_prompt.py     -> emotion.prompt
emotion_examples.py   -> emotion.examples
emotion_retrieval.py  -> emotion.retrieval
emotion_feedback.py   -> emotion.feedback
safety.py             -> emotion.safety
```

- [ ] **Step 3: 检查旧导入并运行聚焦测试**

```bash
rg -n "chatbot\.(emotion_state|emotion_labels|emotion_prompt|emotion_examples|emotion_retrieval|emotion_feedback|safety)" chatbot scripts tests
uv run --with-requirements requirements.txt pytest -q tests/emotion tests/core tests/test_main.py tests/test_chat_service.py tests/test_web.py
```

Expected: `rg` 无匹配；测试全部通过。

- [ ] **Step 4: 提交 emotion 迁移**

```bash
git add chatbot scripts tests
git commit -m "refactor: group emotion domain modules"
```

---

### Task 3: 建立 memory 与 profile 包

**Files:**
- Create: `chatbot/memory/__init__.py`
- Create: `chatbot/profile/__init__.py`
- Move: memory 与 profile 映射表中的六个模块
- Modify: `chatbot/chat_service.py`、`chatbot/web.py` 及相关模块
- Move tests: 对应文件 → `tests/memory/`、`tests/profile/`

**Interfaces:**
- Consumes: Task 1 的 core 包与 Task 2 的 `EmotionState`。
- Produces: `chatbot.memory.models`、`sqlite`、`extractor`、`consolidation`，以及 `chatbot.profile.repository`、`onboarding`。

- [ ] **Step 1: 移动模块与测试**

```bash
mkdir -p chatbot/memory chatbot/profile tests/memory tests/profile
git mv chatbot/memory.py chatbot/memory/models.py
git mv chatbot/local_memory.py chatbot/memory/sqlite.py
git mv chatbot/memory_extractor.py chatbot/memory/extractor.py
git mv chatbot/memory_consolidation.py chatbot/memory/consolidation.py
git mv chatbot/profile.py chatbot/profile/repository.py
git mv chatbot/profile_onboarding.py chatbot/profile/onboarding.py
```

将 `test_memory.py`、`test_local_memory.py`、`test_memory_extractor.py`、`test_memory_consolidation.py` 移入 `tests/memory/`；将 `test_profile.py`、`test_profile_onboarding.py` 移入 `tests/profile/`。

- [ ] **Step 2: 更新包接口和导入**

`chatbot/memory/__init__.py` 从 `models.py` 导出当前跨领域使用的 Protocol、配置和数据类；`chatbot/profile/__init__.py` 不复制可变状态，只保留显式模块路径。

逐项替换旧路径：

```text
chatbot.memory                -> chatbot.memory.models
chatbot.local_memory          -> chatbot.memory.sqlite
chatbot.memory_extractor      -> chatbot.memory.extractor
chatbot.memory_consolidation  -> chatbot.memory.consolidation
chatbot.profile               -> chatbot.profile.repository
chatbot.profile_onboarding    -> chatbot.profile.onboarding
```

- [ ] **Step 3: 运行领域和应用测试**

```bash
uv run --with-requirements requirements.txt pytest -q tests/memory tests/profile tests/test_main.py tests/test_chat_service.py tests/test_web.py
```

Expected: 全部通过。

- [ ] **Step 4: 提交 memory/profile 迁移**

```bash
git add chatbot tests
git commit -m "refactor: group memory and profile modules"
```

---

### Task 4: 整理消融脚本与剩余测试

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/ablation/__init__.py`
- Move: 五个消融脚本 → `scripts/ablation/`
- Move: app、scripts、project 类测试到对应目录
- Modify: README、测试中的脚本路径和 Python 模块导入

**Interfaces:**
- Consumes: Tasks 1–3 的新应用包路径。
- Produces: `python -m scripts.ablation.<module>` CLI；命令参数和输出格式不变。

- [ ] **Step 1: 移动脚本和测试**

```bash
mkdir -p scripts/ablation tests/app tests/scripts tests/project
git mv scripts/evaluate_emotion_analysis.py scripts/run_emotion_ablation.py scripts/evaluate_emotion_ablation.py scripts/run_codex_cli_emotion_ablation.py scripts/report_codex_cli_emotion_ablation.py scripts/ablation/
git mv tests/test_web.py tests/test_chat_service.py tests/test_main.py tests/app/
git mv tests/test_readme.py tests/project/
```

将以下测试移动到 `tests/scripts/`：

```text
tests/test_emotion_benchmark.py
tests/test_emotion_benchmark_cli.py
tests/test_prepare_empathetic_dialogues.py
tests/test_evaluate_emotion_analysis.py
tests/test_evaluate_emotion_ablation.py
tests/test_run_emotion_ablation.py
tests/test_run_codex_cli_emotion_ablation.py
tests/test_report_codex_cli_emotion_ablation.py
```

- [ ] **Step 2: 增加包文件并更新脚本间导入**

创建空的 `scripts/__init__.py` 与 `scripts/ablation/__init__.py`。将：

```python
from scripts.evaluate_emotion_analysis import ...
from scripts.run_emotion_ablation import ...
```

改为：

```python
from scripts.ablation.evaluate_emotion_analysis import ...
from scripts.ablation.run_emotion_ablation import ...
```

子进程测试统一使用 `sys.executable, "-m", "scripts.ablation.<module>", "--help"`。

- [ ] **Step 3: 运行 CLI smoke test 和脚本测试**

```bash
uv run --with-requirements requirements.txt python -m scripts.ablation.evaluate_emotion_analysis --help
uv run --with-requirements requirements.txt python -m scripts.ablation.run_emotion_ablation --help
uv run --with-requirements requirements.txt python -m scripts.ablation.evaluate_emotion_ablation --help
uv run --with-requirements requirements.txt python -m scripts.ablation.run_codex_cli_emotion_ablation --help
uv run --with-requirements requirements.txt python -m scripts.ablation.report_codex_cli_emotion_ablation --help
uv run --with-requirements requirements.txt pytest -q tests/scripts tests/project
```

Expected: 五个命令退出码为 0；脚本与 README 测试全部通过。

- [ ] **Step 4: 提交脚本与测试整理**

```bash
git add scripts tests README.md
git commit -m "refactor: group ablation scripts and tests"
```

---

### Task 5: 整理文档并修正 README

**Files:**
- Create: `docs/README.md`
- Move: 当前设计 → `docs/design/`
- Move: 已完成计划与旧汇报材料 → `docs/archive/`
- Delete: 只描述已退役合成 benchmark 的专用设计和计划
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 4 的最终应用、脚本和测试路径。
- Produces: 与实际目录一致的项目结构、启动命令、实验命令和文档导航。

- [ ] **Step 1: 分类文档**

保留并移动当前系统设计到 `docs/design/`；移动已完成实施计划和旧导师汇报设计到 `docs/archive/implementation-plans/`、`docs/archive/research-history/`。

删除：

```text
docs/superpowers/specs/2026-07-02-emotion-ablation-v2-design.md
docs/superpowers/plans/2026-07-02-emotion-ablation-v2-500-release.md
docs/superpowers/plans/2026-07-02-emotion-ablation-v2.md
```

前两项为版本化删除；第三项是忽略提交的本地旧计划。

- [ ] **Step 2: 新增文档导航并更新 README**

`docs/README.md` 明确：`design/` 是当前设计，`archive/` 仅供历史参考。README 完成以下机械更新：

- 所有消融命令改为 `python -m scripts.ablation.<module>`；
- 项目结构改为最终目录树；
- 删除“Emotion Ablation V2 合成诊断集”表述；
- benchmark、数据许可和最终交付物表述保持不变。

- [ ] **Step 3: 检查失效引用并运行文档测试**

```bash
rg -n "scripts/(evaluate_emotion|run_emotion|run_codex|report_codex).*\.py|emotion_ablation_v2|ablation_dialogues\.jsonl|ablation_labels\.jsonl" README.md docs/design docs/README.md
uv run --with-requirements requirements.txt pytest -q tests/project
```

Expected: 当前文档无旧命令或已删除数据引用；归档目录允许保留历史表述；README 测试通过。

- [ ] **Step 4: 提交文档整理**

```bash
git add -A README.md docs
git commit -m "docs: align documentation with project structure"
```

---

### Task 6: 精确清理并完成验证

**Files:**
- Delete local generated files: 精确列出的 `.DS_Store`、`.pytest_cache/`、`__pycache__/`
- Remove merged stale worktree copy: `.worktrees/regenerate-readme-20260724`
- Preserve: `.idea/`、`data/records/`、`.worktrees/advisor-report-ppt-20260727`

**Interfaces:**
- Consumes: Tasks 1–5 的完整结构。
- Produces: 干净任务分支、可复现验证结果和明确的剩余 Git 状态。

- [ ] **Step 1: 验证清理目标边界**

```bash
git -C "/Users/oriki/Library/Mobile Documents/com~apple~CloudDocs/Quan's Database(iCloud)/与杭电相关的仓库/HDU-erc" branch --merged main
git --git-dir="/Users/oriki/Library/Mobile Documents/com~apple~CloudDocs/Quan's Database(iCloud)/与杭电相关的仓库/HDU-erc/.git/worktrees/regenerate-readme-20260724" --work-tree="/Users/oriki/Library/Mobile Documents/com~apple~CloudDocs/Quan's Database(iCloud)/与杭电相关的仓库/HDU-erc/.worktrees/regenerate-readme-20260724" status --short --branch
git --git-dir="/Users/oriki/Library/Mobile Documents/com~apple~CloudDocs/Quan's Database(iCloud)/与杭电相关的仓库/HDU-erc/.git/worktrees/advisor-report-ppt-20260727" --work-tree="/Users/oriki/Library/Mobile Documents/com~apple~CloudDocs/Quan's Database(iCloud)/与杭电相关的仓库/HDU-erc/.worktrees/advisor-report-ppt-20260727" status --short --branch
git -C "/Users/oriki/Library/Mobile Documents/com~apple~CloudDocs/Quan's Database(iCloud)/与杭电相关的仓库/HDU-erc" clean -ndX
```

Expected: regenerate-readme 分支已合并且副本干净；advisor 分支未合并但副本干净；预览中明确排除 `.idea/`、`data/records/` 和 advisor worktree。

- [ ] **Step 2: 只删除已确认目标**

精确删除缓存、macOS 元数据、旧的未跟踪合成计划和已合并的 regenerate-readme 物理副本；对失效 Git worktree 登记使用 `git worktree prune`。不删除任何未合并分支。

- [ ] **Step 3: 运行最终完整验证**

```bash
uv run --with-requirements requirements.txt pytest -q
uv run --with-requirements requirements.txt python -c "from chatbot.web import app; assert app is not None"
uv run --with-requirements requirements.txt python -m scripts.benchmark.validate_emotion_benchmark --input data/benchmarks/empathetic_dialogues_v1/release/balanced_seed.jsonl
git diff --check main...HEAD
```

Expected: 不少于 380 tests passed、Web 导入成功、64 条 benchmark 验证成功、diff check 无输出。

- [ ] **Step 4: Python 结构与范围复核**

对所有移动后 Python 文件统计纯代码行数；本任务不新增业务逻辑，既有超长文件记录为基线，不顺带重写。检查：

```bash
rg -n "chatbot\.(config|llm|llm_adapter|prompt_config|runtime_store|history|emotion_state|emotion_labels|emotion_prompt|emotion_examples|emotion_retrieval|emotion_feedback|safety|local_memory|memory_extractor|memory_consolidation|profile_onboarding)" chatbot scripts tests
git status --short
git diff --stat main...HEAD
git log --oneline main..HEAD
```

Expected: 无旧应用模块导入；只存在本任务改动；提交历史按 core、emotion、memory/profile、scripts/tests、docs 分组。

- [ ] **Step 5: 如有最终清理差异则提交**

只有存在版本化清理差异时创建提交：

```bash
git add -A
git commit -m "chore: remove obsolete project artifacts"
```

没有版本化差异时不得创建空提交。
