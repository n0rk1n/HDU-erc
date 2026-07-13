import csv
import json

import pytest

from scripts.report_codex_cli_emotion_ablation import (
    build_report_data,
    main,
    render_chinese_report,
    render_metrics_csv,
    render_summary,
)


SEED_RECORDS = [
    {
        "case_id": "case-001",
        "expected": "anxious",
        "language": "zh",
        "context_dependency": "high",
        "current_input": "我一直担心明天的汇报。",
    },
    {
        "case_id": "case-002",
        "expected": "grateful",
        "language": "en",
        "context_dependency": "none",
        "current_input": "I appreciate your help.",
    },
]

RUNS = {
    "full": [
        {"case_id": "case-001", "run": "full", "emotion": "anxious", "success": True},
        {"case_id": "case-002", "run": "full", "emotion": "sad", "success": True},
    ],
    "no_emotion_history": [
        {"case_id": "case-001", "run": "no_emotion_history", "emotion": "", "success": False},
        {"case_id": "case-002", "run": "no_emotion_history", "emotion": "grateful", "success": True},
    ],
}


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([{"case_id": "case-001", "run": "full", "emotion": "anxious", "success": True}], "missing case IDs"),
        ([
            {"case_id": "case-001", "run": "full", "emotion": "anxious", "success": True},
            {"case_id": "case-001", "run": "full", "emotion": "sad", "success": True},
            {"case_id": "case-002", "run": "full", "emotion": "grateful", "success": True},
        ], "duplicate case IDs"),
        ([
            {"case_id": "case-001", "run": "full", "emotion": "anxious", "success": True},
            {"case_id": " case-001 ", "run": "full", "emotion": "sad", "success": True},
            {"case_id": "case-002", "run": "full", "emotion": "grateful", "success": True},
        ], "duplicate case IDs"),
        ([
            {"case_id": "case-001", "run": "full", "emotion": "anxious", "success": True},
            {"case_id": "case-002", "run": "full", "emotion": "grateful", "success": True},
            {"case_id": "case-999", "run": "full", "emotion": "sad", "success": True},
        ], "unknown case IDs"),
        ([
            {"case_id": "case-001", "run": "wrong", "emotion": "anxious", "success": True},
            {"case_id": "case-002", "run": "full", "emotion": "grateful", "success": True},
        ], "wrong or missing run name"),
    ],
)
def test_build_report_data_rejects_invalid_run_coverage(records, message):
    with pytest.raises(ValueError, match=message):
        build_report_data({"full": records}, SEED_RECORDS)


def test_build_report_data_includes_global_language_and_context_slices():
    report = build_report_data(RUNS, SEED_RECORDS)

    assert report["runs"]["full"]["overall"]["accuracy"] == 0.5
    assert report["runs"]["full"]["languages"]["zh"]["total"] == 1
    assert report["runs"]["full"]["contexts"]["high"]["total"] == 1


def test_build_report_data_keeps_failed_predictions_in_denominator():
    report = build_report_data(RUNS, SEED_RECORDS)
    run = report["runs"]["no_emotion_history"]

    assert run["overall"]["total"] == 2
    assert run["overall"]["correct"] == 1
    assert run["failures"] == 1
    assert run["valid_predictions"] == 1
    assert run["overall"]["errors"][0]["case_id"] == "case-001"


def test_render_chinese_report_contains_metrics_failures_and_limitations():
    report = build_report_data(RUNS, SEED_RECORDS)

    text = render_chinese_report(
        report,
        metadata={"commit": "abc123", "codex_version": "0.142.4"},
    )

    assert "# Codex CLI 情绪识别消融实验报告" in text
    assert "Macro F1" in text
    assert "调用失败" in text
    assert (
        "`zero_shot` 同时禁用 few-shot 示例和情绪历史先验，因此属于组合消融；"
        "其相对 `full` 的指标差值不能单独归因于任一组件。"
    ) in text
    assert "## 整体结果" in text
    assert "## 语言切片" in text
    assert "## 上下文依赖切片" in text
    assert "## 标签混淆" in text
    assert "## 错误样例" in text
    assert "## 方法" in text
    assert "## 局限性" in text
    assert "abc123" in text
    assert "0.142.4" in text


def test_render_metrics_csv_has_required_columns_and_full_deltas():
    report = build_report_data(RUNS, SEED_RECORDS)

    rows = list(csv.DictReader(render_metrics_csv(report).splitlines()))

    assert list(rows[0]) == [
        "run",
        "samples",
        "valid_predictions",
        "failures",
        "correct",
        "accuracy",
        "macro_f1",
        "accuracy_delta_vs_full",
        "macro_f1_delta_vs_full",
        "prompt_identical_to_full",
        "prompt_compared_to_full",
        "treatment_status",
        "treatment_provenance",
    ]
    assert rows[0]["run"] == "full"
    assert rows[0]["accuracy_delta_vs_full"] == "0.000000"
    assert rows[0]["macro_f1_delta_vs_full"] == "0.000000"


def test_build_report_data_detects_noop_from_case_matched_prompt_identity():
    runs = {
        "full": [
            {"case_id": "case-001", "run": "full", "input": "prompt one", "emotion": "anxious", "success": True},
            {"case_id": "case-002", "run": "full", "input": "prompt two", "emotion": "grateful", "success": True},
        ],
        "no_emotion_history": [
            {"case_id": "case-002", "run": "no_emotion_history", "input": "prompt two", "emotion": "grateful", "success": True},
            {"case_id": "case-001", "run": "no_emotion_history", "input": "prompt one", "emotion": "sad", "success": True},
        ],
        "short_context": [
            {"case_id": "case-001", "run": "short_context", "input": "shortened prompt", "emotion": "anxious", "success": True},
            {"case_id": "case-002", "run": "short_context", "input": "prompt two", "emotion": "grateful", "success": True},
        ],
    }

    report = build_report_data(runs, SEED_RECORDS)

    no_history = report["runs"]["no_emotion_history"]["treatment"]
    assert no_history == {
        "prompt_identical_to_full": 2,
        "prompt_compared_to_full": 2,
        "status": "no_op_identical_to_full",
        "provenance": "record_input_vs_full_by_case_id",
    }
    short_context = report["runs"]["short_context"]["treatment"]
    assert short_context["prompt_identical_to_full"] == 1
    assert short_context["prompt_compared_to_full"] == 2
    assert short_context["status"] == "effective_prompt_change"


def test_noop_warning_is_prominent_in_summary_csv_and_chinese_report():
    runs = {
        "full": [
            {"case_id": "case-001", "run": "full", "input": "same one", "emotion": "anxious", "success": True},
            {"case_id": "case-002", "run": "full", "input": "same two", "emotion": "sad", "success": True},
        ],
        "no_emotion_history": [
            {"case_id": "case-001", "run": "no_emotion_history", "input": "same one", "emotion": "sad", "success": True},
            {"case_id": "case-002", "run": "no_emotion_history", "input": "same two", "emotion": "grateful", "success": True},
        ],
        "short_context": [
            {"case_id": "case-001", "run": "short_context", "input": "same one", "emotion": "anxious", "success": True},
            {"case_id": "case-002", "run": "short_context", "input": "same two", "emotion": "grateful", "success": True},
        ],
    }
    report = build_report_data(runs, SEED_RECORDS)

    summary = render_summary(report)
    csv_rows = list(csv.DictReader(render_metrics_csv(report).splitlines()))
    chinese = render_chinese_report(
        report,
        metadata={"execution_note": "曾因容量中断，随后从已有成功记录继续执行。"},
    )

    assert "⚠️ treatment 有效性警告" in summary
    assert "no_emotion_history` 与 `short_context` 的输入 Prompt 均为 2/2 与 `full` 完全相同" in summary
    assert "指标差异只能视为重复调用波动，不能归因于消融组件" in summary
    by_run = {row["run"]: row for row in csv_rows}
    assert by_run["no_emotion_history"]["prompt_identical_to_full"] == "2"
    assert by_run["no_emotion_history"]["prompt_compared_to_full"] == "2"
    assert by_run["no_emotion_history"]["treatment_status"] == "no_op_identical_to_full"
    assert (
        by_run["no_emotion_history"]["treatment_provenance"]
        == "record_input_vs_full_by_case_id"
    )
    assert "## 结论有效性警告" in chinese
    assert "`no_emotion_history` 与 `short_context` 的输入 Prompt 均为 2/2 与 `full` 完全相同" in chinese
    assert "这两组是 no-op 重复对照，其指标差异是重复调用波动，不是消融效果" in chinese
    assert "不能据此归因情绪历史或上下文长度的组件贡献" in chinese


def test_chinese_report_uses_singular_wording_for_one_discovered_noop_run():
    runs = {
        "full": [
            {
                    "case_id": "case-001",
                    "run": "full",
                "input": "same prompt",
                "emotion": "anxious",
                "success": True,
            },
        ],
        "no_emotion_history": [
            {
                    "case_id": "case-001",
                    "run": "no_emotion_history",
                "input": "same prompt",
                "emotion": "sad",
                "success": True,
            },
        ],
    }

    text = render_chinese_report(build_report_data(runs, SEED_RECORDS[:1]))

    assert "`no_emotion_history` 的输入 Prompt 均为 1/1 与 `full` 完全相同" in text
    assert "该运行是 no-op 重复对照" in text
    assert "这两组是 no-op 重复对照" not in text


def test_chinese_report_discloses_required_seed64_limitations():
    report = build_report_data(RUNS, SEED_RECORDS)

    text = render_chinese_report(
        report,
        metadata={"execution_note": "容量中断后续跑"},
    )

    assert "本次基准仅包含 2 条合成 seed 记录" in text
    assert "高上下文依赖样本为 1 条" in text
    assert "Codex CLI Agent 执行链路，不是裸模型 API 评测" in text
    assert "容量中断后续跑" in text
    assert "`zero_shot` 同时禁用 few-shot 示例和情绪历史先验" in text


def test_main_writes_deterministic_report_artifacts(tmp_path):
    seed_file = tmp_path / "seed.jsonl"
    seed_file.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in SEED_RECORDS),
        encoding="utf-8",
    )
    run_args = []
    for name, records in RUNS.items():
        run_file = tmp_path / f"{name}.json"
        run_file.write_text(json.dumps(records), encoding="utf-8")
        run_args.extend(["--run", f"{name}={run_file}"])
    output_dir = tmp_path / "reports"

    result = main([
        "--seed-file",
        str(seed_file),
        "--output-dir",
        str(output_dir),
        "--commit",
        "abc123",
        "--codex-version",
        "0.142.4",
        "--branch",
        "codex/test",
        "--started-at",
        "2026-07-13T13:44:03+08:00",
        "--ended-at",
        "2026-07-13T14:39:19+08:00",
        "--model",
        "gpt-test",
        *run_args,
    ])

    assert result == 0
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "summary.md").exists()
    report_path = output_dir / "report-zh.md"
    assert report_path.exists()
    assert "abc123" in report_path.read_text(encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")
    assert "codex/test" in report_text
    assert "2026-07-13T13:44:03+08:00" in report_text
    assert "2026-07-13T14:39:19+08:00" in report_text
    assert "gpt-test" in report_text


def test_seed_id_alias_is_normalized_without_losing_slice_metadata():
    records = [{"case_id": "legacy-id", "run": "full", "emotion": "sad", "success": True}]
    seed = [{
        "id": "legacy-id",
        "label": "sad",
        "language": "zh",
        "context_dependency": "medium",
    }]

    report = build_report_data({"full": records}, seed)

    assert report["runs"]["full"]["overall"]["correct"] == 1
    assert report["runs"]["full"]["languages"]["zh"]["correct"] == 1
    assert report["runs"]["full"]["contexts"]["medium"]["correct"] == 1


def test_seed_case_id_takes_precedence_over_legacy_id():
    records = [
        {"case_id": "canonical-id", "run": "full", "emotion": "sad", "success": True},
    ]
    seed = [{
        "case_id": "canonical-id",
        "id": "legacy-id",
        "expected": "sad",
        "language": "en",
        "context_dependency": "low",
    }]

    report = build_report_data({"full": records}, seed)

    assert report["runs"]["full"]["overall"]["correct"] == 1
