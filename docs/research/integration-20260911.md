# 研究分支合并与数据迁移记录

日期：2026-09-11。用户要求将本任务修改合入目标分支，随后删除任务分支和 worktree。操作仅在本地进行，没有推送或新增模型调用。

## 合并范围

- 目标：`main`，合并前为 `1af9d06c1bc0f8b2754b1066568ed56598ab411a`。
- 任务分支：`codex/goemotions-config-20260908`，研究总账提交为 `8d1ce6d`。
- `f46e266` 修正总账中的可点击链接和运行命令后，研究内容已快进合入 main，没有冲突。
- 本记录和总账当前状态说明随后通过同一任务分支提交并快进合入。最终 main 提交以 `git log -1` 为准。
- 仅清理本任务分支和对应 worktree；其他任务分支和 worktree 保留。

## 实验数据保全

原 worktree 的 `data/research/` 不受 Git 跟踪，删除 worktree 前必须单独保存。已将整个目录在同一文件系统中重命名迁移到：

`/Users/oriki/Database/HDU-erc/data/research/`

迁移前清单含 **30,214 个文件条目、21,911,830,807 字节逻辑大小**。迁移后逐项比较 inode、设备、文件大小、修改时间、权限与符号链接目标，差异为 0。此数字不包含迁移时新增的集成核验材料。

目录完整包含 SQLite、全部原始工件、历轮备份、模型文件、向量/编码缓存、导出、运行状态、日志及分析结果。没有修改旧工件中的原始路径字段或历史运行快照。

原主工作区 `.env` 与 `data/chatbot.sqlite3` 的 SHA-256 前后相同。旧主工作区 `.venv` 保存于 `data/research/integration-20260911/preserved-main-venv/`；研究虚拟环境迁移为主工作区 `.venv`，23 个脚本/激活入口中的旧绝对路径已经修正，原入口文本也保留。

主工作区合并前的 50 项未提交删除均为 archive 下的旧测试文件，与本任务分支已提交的删除完全一致。快进合并后不再显示这些工作区差异；原文件缺失状态保持不变，原始状态清单和 400,917 字节的补丁已另存，未执行 reset 或覆盖用户编辑。

## 验证

- 合并前研究分支：91 项测试通过。
- 合并后主工作区：91 项测试通过，耗时 6.57 秒；验证的是迁移后的研究运行环境。
- 主工作区 Python 前缀为 `/Users/oriki/Database/HDU-erc/.venv`；numpy 2.5.3、torch 2.14.0、transformers 5.16.1、tokenizers 0.23.2 与实验环境一致。
- 迁移后的 pytest 命令入口可执行；总账中的 36 个已有本地链接在主工作区存在，旧 worktree 不再作为链接目标。
- 活跃库全部 29,450 个工件校验通过，无错误或孤儿文件；15 个运行全部 completed，实际调用仍为 2,806。
- 最新完整备份全部 29,449 个工件校验通过，无错误或孤儿文件；重新只读打开后 15 个运行和 2,806 次调用齐全。
- 核验直接以 SQLite mode=ro 打开，检查前后数据库文件 SHA-256 一致，没有追加调用或实验记录。

## 证据与继续研究入口

所有集成操作材料位于 `data/research/integration-20260911/`，不进入 Git：

| 文件/目录 | 内容 |
|---|---|
| before.json | 合并前提交、保护文件哈希、数据规模与迁移源/目标 |
| main-before-status.z | 主工作区原始 Git 状态，NUL 分隔 |
| main-before.patch | 原主工作区 50 项删除的补丁 |
| data-inode-manifest.json | 迁移前逐文件标识、大小、权限与时间 |
| migration.json | 同文件系统迁移及运行环境处理结果 |
| verification.json | 活跃库及最新完整备份的全量核验 |
| original-research-venv-entrypoints/ | 修改路径前的研究虚拟环境脚本 |
| preserved-main-venv/ | 未覆盖的原主工作区运行环境 |
| cleanup.json | 合并后分支/worktree 清理的最终核验，清理完成后生成 |

实验数据库：[experiments.sqlite3](/Users/oriki/Database/HDU-erc/data/research/experiments.sqlite3)。

研究总账：[research-progress-tracker-20260909.md](/Users/oriki/Database/HDU-erc/docs/research/research-progress-tracker-20260909.md)。总账主体是 2026-09-09 的研究历史；本次集成没有将任何尚未执行的后续实验改写为已完成。

后续在主工作区继续使用 `.venv/bin/python` 和 `data/research/`；不要再进入已安排删除的 goemotions-config worktree。
