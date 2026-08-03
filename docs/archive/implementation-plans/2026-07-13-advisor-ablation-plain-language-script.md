# Advisor Ablation Plain-Language Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 PPT 所在目录生成一份可用于约 15 分钟导师汇报的中文大白话 TXT 讲稿。

**Architecture:** 讲稿只读取已验证的 Seed-64 中文报告和现有 PPT 设计说明，按“背景—目的—准备—结果—反思—下一步”组织。成品以纯 UTF-8 文本写入 `outputs/`，随后检查文件存在、关键数字和限制说明是否准确。

**Tech Stack:** Markdown/TXT、仓库已有实验报告、shell 文本核验。

## Global Constraints

- 输出文件固定为 `/Users/oriki/Documents/HDU-erc/outputs/情绪识别消融实验导师汇报大白话版.txt`。
- 使用自然中文口语和第一人称，不把探索性结果表达成最终因果结论。
- 必须包含背景、目的、准备、结果、反思、下一步六部分。
- 关键数据必须与 Seed-64 报告一致：full 71.88% / 65.76%，静态示例 79.69% / 74.60%，zero-shot 67.19% / 60.83%。
- 必须说明两个 no-op 组均有 64/64 Prompt 与 full 相同，zero-shot 是组合消融。
- 不改动既有 PPT，也不引入外部资料。

---

## File Structure

- Read: `/Users/oriki/Documents/HDU-erc/data/records/codex_cli_ablation/seed64/report-zh.md` — 指标、实验边界、错误模式和局限性。
- Read: `/Users/oriki/Documents/HDU-erc/docs/superpowers/specs/2026-07-13-advisor-ablation-plain-language-script-design.md` — 成品语气和结构。
- Create: `/Users/oriki/Documents/HDU-erc/outputs/情绪识别消融实验导师汇报大白话版.txt` — 最终口述稿。

### Task 1: 核对事实并写入口述稿

**Files:**
- Read: `/Users/oriki/Documents/HDU-erc/data/records/codex_cli_ablation/seed64/report-zh.md`
- Create: `/Users/oriki/Documents/HDU-erc/outputs/情绪识别消融实验导师汇报大白话版.txt`

**Interfaces:**
- Consumes: 本地 Seed-64 指标、Prompt 有效性记录和设计说明。
- Produces: UTF-8 纯文本讲稿，包含六个明确中文小节。

- [ ] **Step 1: 核验报告中的关键事实**

运行：

```bash
sed -n '1,220p' /Users/oriki/Documents/HDU-erc/data/records/codex_cli_ablation/seed64/report-zh.md
```

预期结果：确认 64 条样本、320 次成功调用、五组指标、Prompt 相同计数以及局限性。

- [ ] **Step 2: 写入口述稿**

使用 `apply_patch` 创建 UTF-8 文本。正文顺序固定为：开场、背景、目的、准备与做法、主要结果、反思、下一步与收束。正文包含以下准确表述：

```text
本次不是要证明某个方案已经最终胜出，而是先确认哪些组件真的值得继续研究。
no_emotion_history 和 short_context 的 64/64 条 Prompt 没有改变，因此不能从这两组指标解释组件贡献。
zero-shot 同时拿掉了示例和历史情绪，只能说明“整体都拿掉”后变弱，不能说明到底是哪一项更重要。
```

- [ ] **Step 3: 核验成品内容**

运行：

```bash
test -s /Users/oriki/Documents/HDU-erc/outputs/情绪识别消融实验导师汇报大白话版.txt
rg -n '背景|目的|准备|结果|反思|下一步|71.88%|79.69%|67.19%|64/64|组合消融' /Users/oriki/Documents/HDU-erc/outputs/情绪识别消融实验导师汇报大白话版.txt
```

预期结果：文件非空，六部分和所有关键数字/限制语句均可检索。

- [ ] **Step 4: 提交成品**

```bash
git add -- /Users/oriki/Documents/HDU-erc/outputs/情绪识别消融实验导师汇报大白话版.txt
git commit -m "docs: add advisor ablation speaking script"
```

预期结果：TXT 成品以单独提交记录，便于与 PPT 一起追踪。
