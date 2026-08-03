# Advisor Emotion Ablation Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一份适合约 15 分钟中文导师汇报的 11 页可编辑 PowerPoint，准确呈现 Seed-64 情绪识别消融结果、处理有效性问题和下一轮研究计划。

**Architecture:** 使用 `@oai/artifact-tool` 的 JavaScript ES module 在独立临时工作区生成整套 `.pptx`；所有图表、矩阵和流程图均由原生 PowerPoint 形状组成。生成后先渲染为逐页 PNG 和总览图，再执行逐页视觉检查与自动重叠/越界检查，修正后将唯一最终文件写入项目 `outputs/`。

**Tech Stack:** Node.js ES modules、`@oai/artifact-tool`、Codex Grid 布局库、LibreOffice/Poppler 渲染辅助脚本、Python 版式检查脚本。

## Global Constraints

- 比例固定为 16:9，页面数量固定为 11 页。
- 适配约 15 分钟汇报：主体约 13 分钟，预留约 2 分钟提问。
- 封面标题不低于 50 pt，页标题不低于 35 pt，正文不低于 16 pt。
- 深海军蓝为主色，青色表示有效证据，暖琥珀表示警告/局限，灰色表示无效处理。
- `full` 固定深蓝，`no_dynamic_examples` 固定青色，`zero_shot` 固定暖橙；两个 no-op 组固定浅灰并附警告标记。
- 所有指标仅来自 `data/records/codex_cli_ablation/seed64/`，不得引入外部研究数据。
- 必须明确标注 `no_emotion_history` 与 `short_context` 为 64/64 Prompt 相同的 no-op。
- 必须明确标注 `zero_shot` 为同时移除示例和情绪历史的组合消融。
- 只使用原生形状、图表、矩阵和流程图，不使用装饰性图片。
- 输出必须可编辑，并通过逐页视觉检查及自动重叠/越界检查。

---

## File Structure

- Create: `/private/tmp/codex-presentations/019f5987-fe01-7412-b6cc-e7ce3e421069/advisor-emotion-ablation/deck.mjs` — 主题、组件、图表数据和 11 页内容的唯一生成源文件。
- Create: `/private/tmp/codex-presentations/019f5987-fe01-7412-b6cc-e7ce3e421069/advisor-emotion-ablation/rendered/` — 逐页 PNG 渲染结果。
- Create: `/private/tmp/codex-presentations/019f5987-fe01-7412-b6cc-e7ce3e421069/advisor-emotion-ablation/montage.png` — 全套幻灯片总览图。
- Create: `/Users/oriki/Documents/HDU-erc/outputs/情绪识别消融实验导师汇报.pptx` — 最终交付文件。
- Read: `/Users/oriki/Documents/HDU-erc/data/records/codex_cli_ablation/seed64/report-zh.md` — 结论、切片和错误样例。
- Read: `/Users/oriki/Documents/HDU-erc/data/records/codex_cli_ablation/seed64/metrics.csv` — 主指标精确值。
- Read: `/Users/oriki/Documents/HDU-erc/data/records/codex_cli_ablation/seed64/run-metadata.json` — 运行环境和时间信息。

### Task 1: 准备演示文稿运行环境与设计基线

**Files:**
- Create: `/private/tmp/codex-presentations/019f5987-fe01-7412-b6cc-e7ce3e421069/advisor-emotion-ablation/`
- Read: Codex Grid 的 `ARTIFACT.md`、`design_tokens.json`、`template-registry.json` 与布局预览图
- Read: `artifact_tool/API_QUICK_START.md` 与 `artifact_tool/api/API_DOCS.md`

**Interfaces:**
- Consumes: 已确认的设计说明 `docs/superpowers/specs/2026-07-13-advisor-emotion-ablation-presentation-design.md`。
- Produces: 可供 `deck.mjs` 直接引用的 Node、Python、渲染脚本和 Codex Grid 模块绝对路径；选定的 16:9 布局组合。

- [ ] **Step 1: 获取工作区依赖路径**

调用 Codex 工作区依赖加载器，记录返回的 Node.js、Python、`@oai/artifact-tool`、演示文稿辅助脚本和 Codex Grid 资源路径。预期结果是全部路径在本机存在且可读取。

- [ ] **Step 2: 完整阅读必需文档**

完整阅读以下资源，不截断内容：

```text
presentations/references/content-rules.md
builtin_templates_support/codex-grid-layout-library/ARTIFACT.md
assets/builtin_templates/codex-grid-layout-library/design_tokens.json
assets/builtin_templates/codex-grid-layout-library/artifact-tool-compose/template-registry.json
artifact_tool/API_QUICK_START.md
artifact_tool/api/API_DOCS.md
```

预期结果：确认 `Presentation.create()`、`slide.compose()`、文本/形状/图表 API、导出 API、布局模块参数和渲染检查命令的准确签名。

- [ ] **Step 3: 选择布局并建立临时工作区**

从 Codex Grid 预览图中选择最接近以下用途的布局：封面、结论标题+三卡片、水平流程、五行矩阵、指标条形图、左右论证、四步路线图。仅阅读被选布局对应的 `slide-XX.mjs` 模块。

运行工作区初始化脚本：

```bash
/Users/oriki/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node /Users/oriki/.codex/plugins/cache/openai-primary-runtime/presentations/26.709.11516/skills/presentations/container_tools/setup_artifact_tool_workspace.mjs /private/tmp/codex-presentations/019f5987-fe01-7412-b6cc-e7ce3e421069/advisor-emotion-ablation
```

预期结果：临时目录包含可执行的 Node ES module 环境，并能解析 `@oai/artifact-tool`。

- [ ] **Step 4: 校验本地数据源**

运行：

```bash
sed -n '1,260p' /Users/oriki/Documents/HDU-erc/data/records/codex_cli_ablation/seed64/report-zh.md
sed -n '1,20p' /Users/oriki/Documents/HDU-erc/data/records/codex_cli_ablation/seed64/metrics.csv
sed -n '1,200p' /Users/oriki/Documents/HDU-erc/data/records/codex_cli_ablation/seed64/run-metadata.json
```

预期结果：确认 64 条样本、320 次成功调用、五组指标、语言切片、上下文切片、Prompt 相同计数和运行元数据均可读取。

### Task 2: 实现 11 页演示文稿生成源文件

**Files:**
- Create: `/private/tmp/codex-presentations/019f5987-fe01-7412-b6cc-e7ce3e421069/advisor-emotion-ablation/deck.mjs`
- Create: `/Users/oriki/Documents/HDU-erc/outputs/情绪识别消融实验导师汇报.pptx`

**Interfaces:**
- Consumes: Task 1 确认的 `@oai/artifact-tool` API 与 Codex Grid 布局函数。
- Produces: `deck.mjs` 顶层执行后生成 11 页 16:9 `.pptx`；内部提供 `addHeader(slide, title, kicker, pageNo)`、`addFooter(slide)`、`addMetricBar(slide, args)`、`addPill(slide, args)` 四个复用函数。

- [ ] **Step 1: 定义主题、数据和复用组件**

在 `deck.mjs` 中定义以下固定值：

```javascript
const C = {
  navy: "0B1F33",
  blue: "173F5F",
  cyan: "14B8A6",
  cyanSoft: "DDF7F3",
  amber: "F59E0B",
  amberSoft: "FFF2D6",
  orange: "E97732",
  gray: "A7B0BA",
  graySoft: "EEF1F4",
  ink: "17212B",
  white: "FFFFFF",
  muted: "5F6B76"
};

const results = [
  { run: "full", accuracy: 71.88, f1: 65.76, color: C.blue, status: "baseline" },
  { run: "no_dynamic_examples", accuracy: 79.69, f1: 74.60, color: C.cyan, status: "有效处理" },
  { run: "no_emotion_history", accuracy: 73.44, f1: 68.89, color: C.gray, status: "no-op" },
  { run: "short_context", accuracy: 73.44, f1: 67.71, color: C.gray, status: "no-op" },
  { run: "zero_shot", accuracy: 67.19, f1: 60.83, color: C.orange, status: "组合消融" }
];

const language = {
  full: { zh: [71.88, 65.10], en: [71.88, 64.79] },
  no_dynamic_examples: { zh: [81.25, 75.00], en: [78.12, 71.35] },
  zero_shot: { zh: [68.75, 61.67], en: [65.62, 57.50] }
};
```

主题字体优先使用 `Noto Sans CJK SC`，无该字体时使用系统中文无衬线字体；所有正文保持 16 pt 以上。

- [ ] **Step 2: 实现第 1–3 页**

按以下内容生成：

```text
1. 封面：面向对话情绪识别的 Prompt 组件消融实验；副标题为 Codex CLI Agent · Seed-64 合成基准 · 导师汇报；日期 2026-07-13。
2. 研究问题：三张卡片分别写“示例选择”“情绪历史”“上下文长度”，底部结论带写“实验首先要证明处理真的改变了输入”。
3. 实验对象：水平流程“对话输入 → 上下文/历史组织 → few-shot 示例 → Codex CLI Agent → 32 类情绪标签”，右上角标注“评估 Agent 链路，不是裸模型 API”。
```

每页标题必须是结论式表达，页脚包含简短来源“本地 Seed-64 实验记录”。

- [ ] **Step 3: 实现第 4–5 页**

第 4 页使用五行三组件矩阵，列为“动态/静态示例、情绪历史、长上下文”，用实心圆表示保留、空心圆表示移除、警告符号标注 `zero_shot` 组合消融。第 5 页使用四个数字卡片：64 条样本、32+32 双语、32 类标签、320/320 成功；下方显示上下文分布 `14 / 22 / 28 / 0` 和环境 `Codex CLI 0.142.4 · gpt-5.5`。

- [ ] **Step 4: 实现第 6 页主结果图**

使用五组水平条形图，每组同时呈现 Accuracy 和 Macro F1，数值直接标在条尾。`no_emotion_history` 与 `short_context` 使用灰色并显示“Prompt 64/64 相同，不作归因”；突出两个有效比较：

```text
no_dynamic_examples vs full: Accuracy +7.81pp, Macro F1 +8.83pp
zero_shot vs full: Accuracy -4.69pp, Macro F1 -4.93pp
```

页底结论带写“Seed-64 上，静态示例配置最好；zero-shot 组合配置最弱”。

- [ ] **Step 5: 实现第 7 页 treatment validation**

左侧画 Prompt 变化计数：`no_dynamic_examples 0/64 相同`、`zero_shot 0/64 相同`；右侧画红/琥珀警告框：`no_emotion_history 64/64 相同`、`short_context 64/64 相同`。底部用因果链表达：

```text
输入未变化 → 只剩调用随机性 → 指标差不能解释组件贡献
```

- [ ] **Step 6: 实现第 8–9 页切片与错误模式**

第 8 页使用两组小型条形图或哑铃图比较 full、no_dynamic_examples、zero_shot 的中文/英文 Accuracy，并以小字附 Macro F1；强调两种语言趋势一致。第 9 页将典型混淆排成 7 个胶囊标签，并放入中英平行例句：

```text
小雅装完最后一个箱子后，屋子一下子空了。
The room feels empty now that Maya packed the last box.
expected: sad · predicted: lonely
```

- [ ] **Step 7: 实现第 10–11 页结论与计划**

第 10 页左右分栏：“当前证据支持”三条、“当前证据不支持”四条，中央以细分隔线区分。第 11 页使用四步路线图：多轮/high-context 样本 → 非泄漏历史先验 → 拆分组合消融 → 多随机种子与裸 API；页底放导师讨论问题“优先扩充真实对话，还是先完善可控合成基准？”

- [ ] **Step 8: 导出可编辑 PPTX**

在 `deck.mjs` 末尾调用 artifact-tool 的实际导出 API，将文件直接写入：

```text
/Users/oriki/Documents/HDU-erc/outputs/情绪识别消融实验导师汇报.pptx
```

运行：

```bash
node /private/tmp/codex-presentations/019f5987-fe01-7412-b6cc-e7ce3e421069/advisor-emotion-ablation/deck.mjs
```

预期结果：命令退出码为 0，输出文件存在，幻灯片数量为 11。

### Task 3: 渲染、逐页检查并修正版式

**Files:**
- Read: `/Users/oriki/Documents/HDU-erc/outputs/情绪识别消融实验导师汇报.pptx`
- Create: `/private/tmp/codex-presentations/019f5987-fe01-7412-b6cc-e7ce3e421069/advisor-emotion-ablation/rendered/*.png`
- Create: `/private/tmp/codex-presentations/019f5987-fe01-7412-b6cc-e7ce3e421069/advisor-emotion-ablation/montage.png`
- Modify: `/private/tmp/codex-presentations/019f5987-fe01-7412-b6cc-e7ce3e421069/advisor-emotion-ablation/deck.mjs`

**Interfaces:**
- Consumes: Task 2 生成的 `.pptx`。
- Produces: 11 张可读 PNG、1 张总览图、无溢出/遮挡/越界的修正版 `.pptx`。

- [ ] **Step 1: 渲染全部页面**

运行演示文稿技能提供的渲染脚本：

```bash
/Users/oriki/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 /Users/oriki/.codex/plugins/cache/openai-primary-runtime/presentations/26.709.11516/skills/presentations/container_tools/render_slides.py /Users/oriki/Documents/HDU-erc/outputs/情绪识别消融实验导师汇报.pptx --output_dir /private/tmp/codex-presentations/019f5987-fe01-7412-b6cc-e7ce3e421069/advisor-emotion-ablation/rendered
```

预期结果：生成 11 张逐页 PNG，文件名按页码排序。

- [ ] **Step 2: 生成总览图**

运行演示文稿技能提供的 montage 脚本，将 11 张页面图排成 3–4 列总览：

```bash
/Users/oriki/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 /Users/oriki/.codex/plugins/cache/openai-primary-runtime/presentations/26.709.11516/skills/presentations/container_tools/create_montage.py --input_dir /private/tmp/codex-presentations/019f5987-fe01-7412-b6cc-e7ce3e421069/advisor-emotion-ablation/rendered --output_file /private/tmp/codex-presentations/019f5987-fe01-7412-b6cc-e7ce3e421069/advisor-emotion-ablation/montage.png --num_col 4 --label_mode number --fail_on_image_error
```

预期结果：总览中 11 页视觉风格一致，标题层级和颜色语义统一。

- [ ] **Step 3: 逐页视觉检查**

先查看总览图，再使用图像查看工具依次打开 11 张原始尺寸 PNG。逐页检查：

```text
标题是否完整且不换成三行；正文是否至少 16 pt；图表数字是否可读；颜色是否符合处理语义；no-op 与组合消融是否显眼；中文字体是否正确；页脚是否统一；是否存在裁切、遮挡、重叠或元素越界。
```

预期结果：记录所有需要修正的具体页码和元素。

- [ ] **Step 4: 执行自动版式检查**

运行：

```bash
/Users/oriki/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 /Users/oriki/.codex/plugins/cache/openai-primary-runtime/presentations/26.709.11516/skills/presentations/container_tools/slides_test.py /Users/oriki/Documents/HDU-erc/outputs/情绪识别消融实验导师汇报.pptx
```

预期结果：无意外的 overlapping elements、out-of-bounds elements、text overflow；仅允许背景形状与明确设计的包含关系。

- [ ] **Step 5: 修正并重新生成**

根据逐页检查和自动检查结果修改 `deck.mjs` 中的坐标、字号、行距、文本长度或图表宽度，然后重新执行 Task 2 Step 8 和本任务 Step 1–4。每次只修正已定位的问题，直到视觉检查和自动检查全部通过。

### Task 4: 最终内容核验与交付

**Files:**
- Read: `/Users/oriki/Documents/HDU-erc/outputs/情绪识别消融实验导师汇报.pptx`
- Read: `/private/tmp/codex-presentations/019f5987-fe01-7412-b6cc-e7ce3e421069/advisor-emotion-ablation/rendered/*.png`

**Interfaces:**
- Consumes: Task 3 通过版式检查的最终文件。
- Produces: 单个可交付 `.pptx` 链接。

- [ ] **Step 1: 核对关键数字与措辞**

逐项确认 PPT 与报告一致：

```text
full 71.88 / 65.76
no_dynamic_examples 79.69 / 74.60，+7.81pp / +8.83pp
no_emotion_history 73.44 / 68.89，Prompt 64/64 相同
short_context 73.44 / 67.71，Prompt 64/64 相同
zero_shot 67.19 / 60.83，-4.69pp / -4.93pp，组合消融
中文 32、英文 32、32 类标签、320/320 成功、高上下文样本 0
```

- [ ] **Step 2: 确认最终文件属性**

检查文件存在、非空、可被渲染脚本再次打开，并确认输出目录中只有一个本次交付用 PPTX。运行：

```bash
ls -lh /Users/oriki/Documents/HDU-erc/outputs/情绪识别消融实验导师汇报.pptx
/Users/oriki/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 /Users/oriki/.codex/plugins/cache/openai-primary-runtime/presentations/26.709.11516/skills/presentations/container_tools/slides_test.py /Users/oriki/Documents/HDU-erc/outputs/情绪识别消融实验导师汇报.pptx
```

预期结果：文件大小大于 0，自动检查通过。

- [ ] **Step 3: 交付**

最终回复只提供一个独立的 Markdown 链接，指向：

```text
/Users/oriki/Documents/HDU-erc/outputs/情绪识别消融实验导师汇报.pptx
```
