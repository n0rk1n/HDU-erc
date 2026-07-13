import csv
import json

from scripts.report_codex_cli_emotion_ablation import (
    build_report_data,
    main,
    render_chinese_report,
    render_metrics_csv,
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
        {"case_id": "case-001", "emotion": "anxious", "success": True},
        {"case_id": "case-002", "emotion": "sad", "success": True},
    ],
    "no_emotion_history": [
        {"case_id": "case-001", "emotion": "", "success": False},
        {"case_id": "case-002", "emotion": "grateful", "success": True},
    ],
}


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
    assert "组合消融" in text
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
    ]
    assert rows[0]["run"] == "full"
    assert rows[0]["accuracy_delta_vs_full"] == "0.000000"
    assert rows[0]["macro_f1_delta_vs_full"] == "0.000000"


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
        *run_args,
    ])

    assert result == 0
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "summary.md").exists()
    report_path = output_dir / "report-zh.md"
    assert report_path.exists()
    assert "abc123" in report_path.read_text(encoding="utf-8")


def test_seed_id_alias_is_normalized_without_losing_slice_metadata():
    records = [{"case_id": "legacy-id", "emotion": "sad", "success": True}]
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
        {"case_id": "legacy-id", "emotion": "anxious", "success": True},
        {"case_id": "canonical-id", "emotion": "sad", "success": True},
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
