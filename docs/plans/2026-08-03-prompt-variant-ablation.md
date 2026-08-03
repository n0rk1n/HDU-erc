# 情绪识别 Prompt 多版本消融实验实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 在不改变应用默认行为的前提下，实现 1 个基准 Prompt 和 4 个新 Prompt 变体，在 EmpatheticDialogues 64 条平衡 seed 上完成可复现的配对实验并生成中文报告。

**架构：** Prompt 变体由独立模块声明，现有 Prompt 构造器通过显式 `prompt_variant` 参数选择模板；现有消融配置和 Codex CLI 运行器负责逐条执行、续跑与 provenance；现有报告器增加 Prompt 实验报告类型和固定候选排序。应用调用不传新参数时仍使用 `full`。

**技术栈：** Python 3.10+、pytest、FastAPI 项目现有 Prompt 构造器、Codex CLI、JSON/JSONL、Accuracy、Macro F1、Family Accuracy、精确 McNemar 检验、配对 bootstrap。

## 全局约束

- 所有修改只在 `codex/prompt-variant-ablation-20260803` 分支和对应 worktree 中完成。
- 实施采用 TDD：每项生产行为先写失败测试并确认失败原因，再写最小实现。
- `build_emotion_prompt()` 的默认输出必须与改动前字节级一致。
- 四个新配置均固定使用动态示例、情绪历史和默认上下文窗口，只允许 Prompt 模板不同。
- 正式运行固定使用同一模型、Codex CLI 版本、输出 Schema、64 条数据和调用参数。
- 第一阶段只运行 64 条 seed，不自动扩展到完整 2,542 条测试集。
- 不引入新的 Python 依赖。
- 不启用子代理或并行代理；正式模型调用可以用独立 CLI 进程并行执行。

---

### Task 1：建立 Prompt 变体注册表并保持默认行为兼容

**文件：**
- 新建：`chatbot/emotion/prompt_variants.py`
- 修改：`chatbot/emotion/prompt.py`
- 修改：`chatbot/emotion/analysis.py`
- 修改：`chatbot/emotion/__init__.py`
- 测试：`tests/emotion/test_emotion_prompt.py`
- 测试：`tests/emotion/test_emotion.py`

**接口：**
- 产生：`PROMPT_VARIANT_NAMES: frozenset[str]`
- 产生：`DEFAULT_PROMPT_VARIANT = "full"`
- 产生：`resolve_emotion_prompt_template(prompt_variant: str) -> str`
- 修改：`build_emotion_analysis_prompt(..., prompt_variant: str = "full") -> str`
- 修改：`build_emotion_prompt(..., prompt_variant: str = "full") -> str`
- 依赖：现有 `DEFAULT_EMOTION_ANALYSIS_PROMPT`、`format_emotion_label_guidance()`、动态示例渲染和自定义 `PROMPT_CONFIG_PATH`

- [ ] **Step 1：写默认行为和四个变体的失败测试**

在 `tests/emotion/test_emotion_prompt.py` 增加：

```python
import pytest


def test_full_prompt_variant_preserves_existing_default_prompt():
    kwargs = {
        "emotion_labels": ["anxious", "lonely"],
        "emotion_label_set": {"anxious", "lonely"},
        "dialogue_context": "I feel alone tonight",
        "current_input": "I feel alone tonight",
    }
    assert build_emotion_analysis_prompt(**kwargs) == build_emotion_analysis_prompt(
        **kwargs,
        prompt_variant="full",
    )


@pytest.mark.parametrize(
    ("variant", "required_text"),
    [
        ("prompt_concise_direct", "Select exactly one emotion label"),
        ("prompt_coarse_to_fine", "First identify the broad emotion family internally"),
        ("prompt_contrastive_check", "Compare the two most plausible labels internally"),
    ],
)
def test_prompt_variant_renders_distinct_instruction(variant, required_text):
    prompt = build_emotion_analysis_prompt(
        emotion_labels=["anxious", "lonely"],
        emotion_label_set={"anxious", "lonely"},
        dialogue_context="I feel alone tonight",
        current_input="I feel alone tonight",
        prompt_variant=variant,
    )
    assert required_text in prompt
    assert "Emotion labels: anxious, lonely" in prompt
    assert "Dialogue context: I feel alone tonight" in prompt
    assert "Return exactly one JSON object" in prompt


def test_no_label_guidance_variant_removes_definition_block_only():
    prompt = build_emotion_analysis_prompt(
        emotion_labels=["anxious", "lonely"],
        emotion_label_set={"anxious", "lonely"},
        dialogue_context="I feel alone tonight",
        current_input="I feel alone tonight",
        prompt_variant="prompt_no_label_guidance",
    )
    assert "Emotion labels: anxious, lonely" in prompt
    assert "Label definitions" not in prompt
    assert "Labeled examples:" in prompt
    assert "Dialogue context: I feel alone tonight" in prompt


def test_unknown_prompt_variant_is_rejected_before_rendering():
    with pytest.raises(ValueError, match="Unknown emotion prompt variant"):
        build_emotion_analysis_prompt(
            emotion_labels=["anxious"],
            emotion_label_set={"anxious"},
            dialogue_context="test",
            prompt_variant="unknown",
        )
```

在 `tests/emotion/test_emotion.py` 增加一个透传测试，断言
`build_emotion_prompt(..., prompt_variant="prompt_coarse_to_fine")` 包含分层指令。

- [ ] **Step 2：运行测试并确认 RED**

运行：

```bash
/Users/oriki/.local/bin/uv run --with-requirements requirements.txt \
  pytest -q tests/emotion/test_emotion_prompt.py tests/emotion/test_emotion.py
```

预期：新增测试因为函数尚不接受 `prompt_variant` 而失败；旧测试继续通过。

- [ ] **Step 3：实现最小 Prompt 变体注册表**

在 `chatbot/emotion/prompt_variants.py` 定义不可变名称集合和四个内置模板。四个模板共享现有占位符：

```python
DEFAULT_PROMPT_VARIANT = "full"
PROMPT_VARIANT_NAMES = frozenset({
    "full",
    "prompt_no_label_guidance",
    "prompt_concise_direct",
    "prompt_coarse_to_fine",
    "prompt_contrastive_check",
})


def resolve_emotion_prompt_template(prompt_variant: str) -> str:
    try:
        return BUILTIN_PROMPT_VARIANTS[prompt_variant]
    except KeyError as exc:
        raise ValueError(
            f"Unknown emotion prompt variant: {prompt_variant!r}"
        ) from exc
```

四个模板使用以下固定正文；`COMMON_RESPONSE_BLOCK` 复用现有响应字段和取值限制，不允许
各变体单独修改输出约束：

```python
COMMON_RESPONSE_BLOCK = """- Response Format: Return exactly one JSON object with these fields:
  {"primary_emotion": "anxious", "confidence": 0.0, "secondary_emotions": [], "evidence": "short phrase from the dialogue", "reply_strategy": "brief guidance for the next chatbot reply", "trajectory_note": "optional change from prior emotion", "safety_level": "normal"}
  Use primary_emotion and secondary_emotions only from the provided Emotion labels. Use safety_level as one of: normal, supportive, crisis."""

PROMPT_NO_LABEL_GUIDANCE = """Infer the emotion expressed by the target user in the described situation or current input.
- Dialogue context: The conversation history between user and assistant, with utterances separated by </s>.
- Emotion labels: {emotion_labels}
- Choose a single inferred emotion from the provided Emotion labels, not outside of them.
{example_block}
{response_block}{likely_line}

Dialogue context: {dialogue_context}"""

PROMPT_CONCISE_DIRECT = """Select exactly one emotion label that best matches the target user's current input.
- Emotion labels: {emotion_labels}
- Use the label definitions and examples as evidence. Do not choose a label outside the list.
- Label definitions:
{label_guidance}
{example_block}
{response_block}{likely_line}

Dialogue context: {dialogue_context}"""

PROMPT_COARSE_TO_FINE = """Infer the emotion expressed by the target user in the described situation or current input.
- Emotion labels: {emotion_labels}
- Label definitions:
{label_guidance}
- First identify the broad emotion family internally, then select the single most precise label from the provided list.
- Return only the final structured result. Do not reveal the intermediate family decision.
{example_block}
{response_block}{likely_line}

Dialogue context: {dialogue_context}"""

PROMPT_CONTRASTIVE_CHECK = """Infer the emotion expressed by the target user in the described situation or current input.
- Emotion labels: {emotion_labels}
- Label definitions:
{label_guidance}
- Compare the two most plausible labels internally against the dialogue evidence and label boundaries before choosing one.
- Return only the final structured result. Do not reveal the candidate comparison.
{example_block}
{response_block}{likely_line}

Dialogue context: {dialogue_context}"""

BUILTIN_PROMPT_VARIANTS = {
    "full": DEFAULT_EMOTION_ANALYSIS_PROMPT,
    "prompt_no_label_guidance": PROMPT_NO_LABEL_GUIDANCE,
    "prompt_concise_direct": PROMPT_CONCISE_DIRECT,
    "prompt_coarse_to_fine": PROMPT_COARSE_TO_FINE,
    "prompt_contrastive_check": PROMPT_CONTRASTIVE_CHECK,
}
```

Prompt 构造器把 `COMMON_RESPONSE_BLOCK` 作为 `response_block` 加入 `values`。现有 full
模板继续使用自己当前的响应区块，因此默认文本不发生变化。

在 `build_emotion_analysis_prompt()` 中，`full` 继续使用
`load_prompt_config().emotion_analysis`，新变体使用注册表内置模板。所有模板仍通过同一
`values` 字典渲染；任何占位符错误沿用现有内置 full 回退仅限 `full`，实验变体渲染错误
必须抛出 `ValueError`。

在 `build_emotion_prompt()` 中将参数原样传给 `build_emotion_analysis_prompt()`。

- [ ] **Step 4：运行聚焦测试并确认 GREEN**

运行同 Step 2 命令。预期：全部通过。

- [ ] **Step 5：提交 Prompt 变体原子提交**

```bash
git add chatbot/emotion/prompt_variants.py chatbot/emotion/prompt.py \
  chatbot/emotion/analysis.py chatbot/emotion/__init__.py \
  tests/emotion/test_emotion_prompt.py tests/emotion/test_emotion.py
git diff --cached --check
git commit -m "feat: add emotion prompt variants"
```

---

### Task 2：把 Prompt 变体接入消融配置、Codex CLI 和 provenance

**文件：**
- 修改：`scripts/ablation/run_emotion_ablation.py`
- 修改：`scripts/ablation/run_codex_cli_emotion_ablation.py`
- 测试：`tests/scripts/test_run_emotion_ablation.py`
- 测试：`tests/scripts/test_run_codex_cli_emotion_ablation.py`

**接口：**
- 修改：`AblationRunConfig.prompt_variant: str = "full"`
- 产生：四个新 `RUN_CONFIGS` 键，名称与 Prompt 变体名称一致
- 修改：`build_case_prompt()` 和 `run_cases()` 将 `config.prompt_variant` 传给 Prompt 构造器
- 保持：现有续跑键继续使用实际 `prompt_sha256`，不新增隐式缓存条件

- [ ] **Step 1：写配置、差异和续跑的失败测试**

在 `tests/scripts/test_run_emotion_ablation.py` 增加：

```python
@pytest.mark.parametrize(
    "run_name",
    [
        "prompt_no_label_guidance",
        "prompt_concise_direct",
        "prompt_coarse_to_fine",
        "prompt_contrastive_check",
    ],
)
def test_prompt_experiment_configs_change_only_prompt_variant(run_name):
    config = RUN_CONFIGS[run_name]
    assert config.example_mode == "dynamic"
    assert config.include_emotion_history is True
    assert config.max_turns is None
    assert config.prompt_variant == run_name
```

在 `tests/scripts/test_run_codex_cli_emotion_ablation.py` 增加：

```python
def test_prompt_variant_runs_are_not_noops_against_full():
    run_names = [
        "full",
        "prompt_no_label_guidance",
        "prompt_concise_direct",
        "prompt_coarse_to_fine",
        "prompt_contrastive_check",
    ]
    assert runner.noop_runs_against_full(
        CASES,
        run_names,
        emotion_interval=5,
    ) == []
    prompts = {
        name: runner.build_case_prompt(
            runner.RUN_CONFIGS[name], CASES[0], 1, emotion_interval=5
        )
        for name in run_names
    }
    assert len(set(prompts.values())) == 5


def test_resume_provenance_changes_with_prompt_variant(tmp_path):
    full = runner.build_case_prompt(
        runner.RUN_CONFIGS["full"], CASES[0], 1, emotion_interval=5
    )
    concise = runner.build_case_prompt(
        runner.RUN_CONFIGS["prompt_concise_direct"],
        CASES[0],
        1,
        emotion_interval=5,
    )
    full_provenance = runner.build_resume_provenance(
        full, schema_file=tmp_path / "schema.json", model="gpt-test",
        codex_cli_version="codex-cli test",
    )
    concise_provenance = runner.build_resume_provenance(
        concise, schema_file=tmp_path / "schema.json", model="gpt-test",
        codex_cli_version="codex-cli test",
    )
    assert full_provenance["prompt_sha256"] != concise_provenance["prompt_sha256"]
```

- [ ] **Step 2：运行新增测试并确认 RED**

```bash
/Users/oriki/.local/bin/uv run --with-requirements requirements.txt pytest -q \
  tests/scripts/test_run_emotion_ablation.py \
  tests/scripts/test_run_codex_cli_emotion_ablation.py
```

预期：`AblationRunConfig` 尚无 `prompt_variant`，新配置不存在，测试失败。

- [ ] **Step 3：实现配置透传和四个运行配置**

将数据类扩展为：

```python
@dataclass(frozen=True)
class AblationRunConfig:
    name: str
    example_mode: str
    include_emotion_history: bool
    max_turns: int | None = None
    prompt_variant: str = "full"
```

增加：

```python
"prompt_no_label_guidance": AblationRunConfig(
    "prompt_no_label_guidance", "dynamic", True,
    prompt_variant="prompt_no_label_guidance",
),
"prompt_concise_direct": AblationRunConfig(
    "prompt_concise_direct", "dynamic", True,
    prompt_variant="prompt_concise_direct",
),
"prompt_coarse_to_fine": AblationRunConfig(
    "prompt_coarse_to_fine", "dynamic", True,
    prompt_variant="prompt_coarse_to_fine",
),
"prompt_contrastive_check": AblationRunConfig(
    "prompt_contrastive_check", "dynamic", True,
    prompt_variant="prompt_contrastive_check",
),
```

两个 Prompt 构造调用都增加 `prompt_variant=config.prompt_variant`。不改变
`build_resume_provenance()`，因为它已经对实际 Prompt 计算 SHA-256。

- [ ] **Step 4：运行聚焦测试和 CLI help 测试并确认 GREEN**

```bash
/Users/oriki/.local/bin/uv run --with-requirements requirements.txt pytest -q \
  tests/scripts/test_run_emotion_ablation.py \
  tests/scripts/test_run_codex_cli_emotion_ablation.py
```

预期：全部通过，`--run` choices 包含四个新名称。

- [ ] **Step 5：提交运行器原子提交**

```bash
git add scripts/ablation/run_emotion_ablation.py \
  scripts/ablation/run_codex_cli_emotion_ablation.py \
  tests/scripts/test_run_emotion_ablation.py \
  tests/scripts/test_run_codex_cli_emotion_ablation.py
git diff --cached --check
git commit -m "feat: run prompt variant ablations"
```

---

### Task 3：增加 Prompt 实验专用报告语义和候选排序

**文件：**
- 修改：`scripts/ablation/report_codex_cli_emotion_ablation.py`
- 测试：`tests/scripts/test_report_codex_cli_emotion_ablation.py`

**接口：**
- 产生：`REPORT_KINDS = ("emotion_ablation", "prompt_variants")`
- 产生：`rank_prompt_candidates(report: dict[str, Any], limit: int = 2) -> list[str]`
- 产生：`render_prompt_variant_conclusion(report: dict[str, Any]) -> str`
- 修改：`render_summary(..., report_kind: str = "emotion_ablation") -> str`
- 修改：`render_chinese_report(..., report_kind: str = "emotion_ablation") -> str`
- 修改：CLI 增加 `--report-kind`，默认 `emotion_ablation`
- 产生：Prompt 实验时额外生成 `conclusion-zh.md`

- [ ] **Step 1：写报告标题、候选排序和限制说明的失败测试**

在测试文件中用以下 helper 构造包含 `full` 和四个 Prompt treatment 的小型配对数据：

```python
def _prompt_run(name, predictions):
    return [
        {
            "case_id": seed["case_id"],
            "run": name,
            "input": f"{name} prompt {index}",
            "emotion": prediction,
            "success": True,
        }
        for index, (seed, prediction) in enumerate(
            zip(SEED_RECORDS, predictions),
            start=1,
        )
    ]


PROMPT_RUNS = {
    "full": _prompt_run("full", ["anxious", "sad"]),
    "prompt_no_label_guidance": _prompt_run(
        "prompt_no_label_guidance", ["sad", "grateful"]
    ),
    "prompt_concise_direct": _prompt_run(
        "prompt_concise_direct", ["anxious", "grateful"]
    ),
    "prompt_coarse_to_fine": _prompt_run(
        "prompt_coarse_to_fine", ["anxious", "grateful"]
    ),
    "prompt_contrastive_check": _prompt_run(
        "prompt_contrastive_check", ["sad", "grateful"]
    ),
}


def test_prompt_variant_report_uses_specific_title_and_limitations():
    report = build_report_data(PROMPT_RUNS, SEED_RECORDS)
    text = render_chinese_report(report, report_kind="prompt_variants")
    assert text.startswith("# Codex CLI 情绪识别 Prompt 多版本实验报告")
    assert "64 条结果只用于筛选" in text
    assert "zero_shot` 同时禁用" not in text


def test_rank_prompt_candidates_uses_preregistered_metric_order():
    report = {
        "runs": {
            "full": {
                "overall": {"accuracy": 0.50, "macro_f1": 0.50, "family_accuracy": 0.50},
                "treatment": {"status": "baseline"},
            },
            "prompt_coarse_to_fine": {
                "overall": {"accuracy": 0.75, "macro_f1": 0.60, "family_accuracy": 0.80},
                "treatment": {"status": "effective_prompt_change"},
            },
            "prompt_concise_direct": {
                "overall": {"accuracy": 0.50, "macro_f1": 0.80, "family_accuracy": 0.90},
                "treatment": {"status": "effective_prompt_change"},
            },
            "prompt_contrastive_check": {
                "overall": {"accuracy": 0.50, "macro_f1": 0.80, "family_accuracy": 0.70},
                "treatment": {"status": "effective_prompt_change"},
            },
            "prompt_no_label_guidance": {
                "overall": {"accuracy": 1.00, "macro_f1": 1.00, "family_accuracy": 1.00},
                "treatment": {"status": "no_op_identical_to_full"},
            },
        }
    }
    assert rank_prompt_candidates(report, limit=2) == [
        "prompt_coarse_to_fine",
        "prompt_concise_direct",
    ]


def test_prompt_variant_main_writes_conclusion(tmp_path):
    seed_file = tmp_path / "seed.jsonl"
    seed_file.write_text(
        "".join(json.dumps(item) + "\n" for item in SEED_RECORDS),
        encoding="utf-8",
    )
    full_file = tmp_path / "full.json"
    full_file.write_text(json.dumps(PROMPT_RUNS["full"]), encoding="utf-8")
    concise_file = tmp_path / "prompt_concise_direct.json"
    concise_file.write_text(
        json.dumps(PROMPT_RUNS["prompt_concise_direct"]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "report"
    result = main([
        "--report-kind", "prompt_variants",
        "--seed-file", str(seed_file),
        "--run", f"full={full_file}",
        "--run", f"prompt_concise_direct={concise_file}",
        "--output-dir", str(output_dir),
    ])
    assert result == 0
    conclusion = (output_dir / "conclusion-zh.md").read_text(encoding="utf-8")
    assert "候选 Prompt" in conclusion
    assert "不能表述为已经证明提升" in conclusion
```

候选排序固定为：排除 `full`、no-op 和不完整 Prompt 证据后，依次按 Accuracy、Macro
F1、Family Accuracy 降序，再按 run name 升序稳定打破完全并列。

- [ ] **Step 2：运行报告测试并确认 RED**

```bash
/Users/oriki/.local/bin/uv run --with-requirements requirements.txt pytest -q \
  tests/scripts/test_report_codex_cli_emotion_ablation.py
```

预期：报告尚不接受 `report_kind`，候选排序函数和结论文件不存在。

- [ ] **Step 3：实现报告类型、候选排序和结论输出**

实现：

```python
def rank_prompt_candidates(report: dict[str, Any], limit: int = 2) -> list[str]:
    candidates = []
    for name, run in report["runs"].items():
        if name == "full" or run["treatment"]["status"] != "effective_prompt_change":
            continue
        metric = run["overall"]
        candidates.append((
            -metric["accuracy"],
            -metric["macro_f1"],
            -metric["family_accuracy"],
            name,
        ))
    return [item[-1] for item in sorted(candidates)[:limit]]
```

`prompt_variants` 报告使用专用标题和局限性；`zero_shot` 局限只在实际报告包含该 run 时
出现。`conclusion-zh.md` 列出最多两个候选、各自相对 `full` 的点估计和区间，并固定声明
64 条结果只是筛选、区间包含 0 时不得声称已证明提升。

- [ ] **Step 4：运行报告聚焦测试并确认 GREEN**

运行同 Step 2 命令。预期：全部通过。

- [ ] **Step 5：提交报告器原子提交**

```bash
git add scripts/ablation/report_codex_cli_emotion_ablation.py \
  tests/scripts/test_report_codex_cli_emotion_ablation.py
git diff --cached --check
git commit -m "feat: report prompt variant experiments"
```

---

### Task 4：补充实验文档并验证代码阶段

**文件：**
- 修改：`README.md`
- 修改：`data/benchmarks/empathetic_dialogues_v1/README.md`
- 测试：`tests/project/test_readme.py`

**接口：**
- 产生：五组 Prompt 实验的可复制运行和报告命令
- 保持：现有 seed64 正式实验结论不变

- [ ] **Step 1：写 README 路径和命令的失败测试**

在 `tests/project/test_readme.py` 增加断言：

```python
def test_readme_documents_prompt_variant_experiment_commands():
    text = README_PATH.read_text(encoding="utf-8")
    assert "prompt_no_label_guidance" in text
    assert "prompt_concise_direct" in text
    assert "prompt_coarse_to_fine" in text
    assert "prompt_contrastive_check" in text
    assert "--report-kind prompt_variants" in text
```

- [ ] **Step 2：运行测试并确认 RED**

```bash
/Users/oriki/.local/bin/uv run --with-requirements requirements.txt \
  pytest -q tests/project/test_readme.py
```

预期：README 尚未包含新配置和命令，新增测试失败。

- [ ] **Step 3：写最小实验文档**

README 增加：五组定义、公平性控制、64 条筛选边界、运行命令、报告命令和完整集需再次
确认的说明。命令统一使用模块入口：

```bash
python -m scripts.ablation.run_codex_cli_emotion_ablation \
  --dialogues-file data/records/empathetic_dialogues_seed_export/dialogues.jsonl \
  --output-dir data/records/codex_cli_ablation/empathetic_dialogues_prompt_variants_seed64_gpt56sol \
  --run full \
  --run prompt_no_label_guidance \
  --run prompt_concise_direct \
  --run prompt_coarse_to_fine \
  --run prompt_contrastive_check \
  --model gpt-5.6-sol
```

- [ ] **Step 4：运行聚焦测试、完整测试和静态检查**

```bash
/Users/oriki/.local/bin/uv run --with-requirements requirements.txt pytest -q
git diff --check
python -m scripts.ablation.run_codex_cli_emotion_ablation --help
python -m scripts.ablation.report_codex_cli_emotion_ablation --help
```

预期：完整测试 0 失败，两个 CLI help 返回 0，Git diff 无空白错误。既有 99 条依赖弃用
warning 作为基线记录，不把它们误报为本任务新增失败。

- [ ] **Step 5：提交文档和代码阶段收尾**

```bash
git add README.md data/benchmarks/empathetic_dialogues_v1/README.md \
  tests/project/test_readme.py
git diff --cached --check
git commit -m "docs: add prompt variant experiment workflow"
```

---

### Task 5：执行 64 条正式 Prompt 实验并提交证据

**文件：**
- 生成并提交：`data/records/codex_cli_ablation/empathetic_dialogues_prompt_variants_seed64_gpt56sol/*.json`
- 生成并提交：`data/benchmarks/empathetic_dialogues_v1/reports/prompt_variants_seed64_gpt56sol/metrics.csv`
- 生成并提交：`data/benchmarks/empathetic_dialogues_v1/reports/prompt_variants_seed64_gpt56sol/summary.md`
- 生成并提交：`data/benchmarks/empathetic_dialogues_v1/reports/prompt_variants_seed64_gpt56sol/report-zh.md`
- 生成并提交：`data/benchmarks/empathetic_dialogues_v1/reports/prompt_variants_seed64_gpt56sol/conclusion-zh.md`
- 生成并提交：`data/benchmarks/empathetic_dialogues_v1/reports/prompt_variants_seed64_gpt56sol/run-metadata.json`
- 修改：`README.md`
- 修改：`data/benchmarks/empathetic_dialogues_v1/README.md`

**接口：**
- 消耗：已测试的五个 `RUN_CONFIGS`
- 产生：每组 64 条结果、固定统计报告和最多两个第二阶段候选

- [ ] **Step 1：导出输入并验证 64 条覆盖**

```bash
python scripts/benchmark/export_emotion_benchmark.py \
  --input data/benchmarks/empathetic_dialogues_v1/release/balanced_seed.jsonl \
  --output-dir data/records/empathetic_dialogues_seed_export
python scripts/benchmark/validate_emotion_benchmark.py \
  --input data/benchmarks/empathetic_dialogues_v1/release/balanced_seed.jsonl
```

确认导出 64 条、32 类每类 2 条，记录 seed 文件 SHA-256。

- [ ] **Step 2：记录冻结环境并做每组 1 条 smoke**

记录：`git rev-parse HEAD`、`codex --version`、模型 `gpt-5.6-sol`、开始时间和输出 Schema
SHA-256。分别对五组使用 `--limit 1 --retries 0`，确认 5/5 有效、Prompt 哈希互异。

- [ ] **Step 3：并行执行五组 64 条正式运行**

每个 CLI 进程只写自己的 `<run>.json`，允许五组并行。固定参数：

```bash
--model gpt-5.6-sol --timeout 180 --retries 1 --emotion-interval 5
```

如进程中断，使用相同命令续跑；provenance 匹配的成功记录必须复用。不得删除或覆盖有效
快照以制造全新结果。

- [ ] **Step 4：验证覆盖、Prompt 差异和 provenance**

使用只读校验命令确认：

- 每个文件恰好 64 个唯一 case_id；
- 每组 `success=True` 数和失败数准确；
- 每条记录的 `run` 正确；
- 四个 treatment 对 `full` 的 Prompt 为 0/64 完全相同；
- 同组记录的 model、schema、CLI 和 runtime provenance 一致；
- 没有 `.tmp` 残留文件。

- [ ] **Step 5：生成 Prompt 专用正式报告**

```bash
python -m scripts.ablation.report_codex_cli_emotion_ablation \
  --report-kind prompt_variants \
  --seed-file data/benchmarks/empathetic_dialogues_v1/release/balanced_seed.jsonl \
  --run full=data/records/codex_cli_ablation/empathetic_dialogues_prompt_variants_seed64_gpt56sol/full.json \
  --run prompt_no_label_guidance=data/records/codex_cli_ablation/empathetic_dialogues_prompt_variants_seed64_gpt56sol/prompt_no_label_guidance.json \
  --run prompt_concise_direct=data/records/codex_cli_ablation/empathetic_dialogues_prompt_variants_seed64_gpt56sol/prompt_concise_direct.json \
  --run prompt_coarse_to_fine=data/records/codex_cli_ablation/empathetic_dialogues_prompt_variants_seed64_gpt56sol/prompt_coarse_to_fine.json \
  --run prompt_contrastive_check=data/records/codex_cli_ablation/empathetic_dialogues_prompt_variants_seed64_gpt56sol/prompt_contrastive_check.json \
  --output-dir data/benchmarks/empathetic_dialogues_v1/reports/prompt_variants_seed64_gpt56sol \
  --commit "$(git rev-parse HEAD)" \
  --codex-version "$(codex --version)" \
  --branch codex/prompt-variant-ablation-20260803 \
  --model gpt-5.6-sol
```

随后生成 `run-metadata.json`，记录数据哈希、五个结果文件哈希、每组成功/失败数、冻结
参数、开始/结束时间和任何中断续跑事实。

- [ ] **Step 6：更新 README 的真实结果边界**

只写报告中实际出现的数字。明确：64 条属于筛选；区间包含 0 时不得写成已证明提升；完整
2,542 条尚未运行；第二阶段调用量需要用户再次确认。

- [ ] **Step 7：执行完成前新鲜验证**

```bash
/Users/oriki/.local/bin/uv run --with-requirements requirements.txt pytest -q
git diff --check
git status --short
```

并重新运行报告命令一次，确认生成物确定性不变。检查 Git diff 只包含本任务代码、文档和
正式实验证据。

- [ ] **Step 8：提交正式实验结果**

`data/records` 和 `docs` 被 `.gitignore` 忽略，必须按精确路径 `git add -f`，不得使用
`git add -A`：

```bash
git add README.md data/benchmarks/empathetic_dialogues_v1/README.md
git add -f \
  data/records/codex_cli_ablation/empathetic_dialogues_prompt_variants_seed64_gpt56sol \
  data/benchmarks/empathetic_dialogues_v1/reports/prompt_variants_seed64_gpt56sol
git diff --cached --check
git diff --cached --stat
git commit -m "feat: complete prompt variant seed64 experiment"
```

提交后再次确认任务 worktree 干净，并报告全部提交、验证命令、实验结论和仍未合并状态。
