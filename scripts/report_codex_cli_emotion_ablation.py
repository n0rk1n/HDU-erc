"""Build deterministic metrics and a Chinese report for Codex CLI ablations."""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_emotion_analysis import evaluate_records, load_records


LANGUAGES = ("zh", "en")
CONTEXT_LEVELS = ("none", "low", "medium", "high")
METRIC_FIELDS = (
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
)

TREATMENT_PROVENANCE = "record_input_vs_full_by_case_id"


def _identifier(item: dict[str, Any]) -> str:
    for key in ("case_id", "id"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _expected(item: dict[str, Any]) -> Any:
    for key in ("expected", "target_emotion", "emotion", "label"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_annotations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for record in records:
        annotation = dict(record)
        annotation["case_id"] = _identifier(record)
        annotation["id"] = annotation["case_id"]
        annotation["expected"] = _expected(record)
        normalized.append(annotation)
    return normalized


def _validate_run_inputs(
    runs: dict[str, list[dict[str, Any]]],
    annotations: list[dict[str, Any]],
) -> None:
    benchmark_ids = [_identifier(item) for item in annotations]
    if any(not case_id for case_id in benchmark_ids):
        raise ValueError("benchmark contains a missing case ID")
    duplicate_benchmark_ids = sorted(
        case_id for case_id, count in Counter(benchmark_ids).items() if count > 1
    )
    if duplicate_benchmark_ids:
        raise ValueError(f"benchmark contains duplicate case IDs: {duplicate_benchmark_ids}")
    expected_ids = set(benchmark_ids)

    for name, records in runs.items():
        ids = [
            record["case_id"].strip()
            if isinstance(record.get("case_id"), str)
            else ""
            for record in records
        ]
        if any(not case_id for case_id in ids):
            raise ValueError(f"run {name!r} contains a missing case ID")
        duplicate_ids = sorted(case_id for case_id, count in Counter(ids).items() if case_id and count > 1)
        if duplicate_ids:
            raise ValueError(f"run {name!r} contains duplicate case IDs: {duplicate_ids}")
        actual_ids = set(ids)
        unknown_ids = sorted(actual_ids - expected_ids)
        if unknown_ids:
            raise ValueError(f"run {name!r} contains unknown case IDs: {unknown_ids}")
        missing_ids = sorted(expected_ids - actual_ids)
        if missing_ids:
            raise ValueError(f"run {name!r} contains missing case IDs: {missing_ids}")
        if any(record.get("run") != name for record in records):
            raise ValueError(f"run {name!r} contains a wrong or missing run name")


def _slice(
    records: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    selected = [item for item in annotations if predicate(item)]
    selected_ids = {_identifier(item) for item in selected}
    predictions = [item for item in records if _identifier(item) in selected_ids]
    return evaluate_records(predictions, selected)


def build_report_data(
    runs: dict[str, list[dict[str, Any]]],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate every run globally and over fixed language/context slices."""
    normalized = _normalize_annotations(annotations)
    _validate_run_inputs(runs, normalized)
    output: dict[str, Any] = {"runs": {}, "annotations": normalized}
    full_by_id = {
        _identifier(record): record
        for record in runs.get("full", [])
        if _identifier(record)
    }
    for name in sorted(runs):
        records = runs[name]
        identical = 0
        compared = 0
        for record in records:
            full_record = full_by_id.get(_identifier(record))
            prompt = record.get("input")
            full_prompt = full_record.get("input") if full_record is not None else None
            if not isinstance(prompt, str) or not isinstance(full_prompt, str):
                continue
            compared += 1
            identical += prompt == full_prompt

        if name == "full":
            treatment_status = "baseline"
        elif compared == len(records) and compared > 0 and identical == compared:
            treatment_status = "no_op_identical_to_full"
        elif compared == len(records) and compared > 0:
            treatment_status = "effective_prompt_change"
        else:
            treatment_status = "incomplete_prompt_evidence"

        output["runs"][name] = {
            "overall": evaluate_records(records, normalized),
            "languages": {
                language: _slice(
                    records,
                    normalized,
                    lambda item, value=language: item.get("language") == value,
                )
                for language in LANGUAGES
            },
            "contexts": {
                level: _slice(
                    records,
                    normalized,
                    lambda item, value=level: item.get("context_dependency") == value,
                )
                for level in CONTEXT_LEVELS
            },
            "failures": sum(record.get("success") is not True for record in records),
            "valid_predictions": sum(
                record.get("success") is True and bool(str(record.get("emotion", "")).strip())
                for record in records
            ),
            "treatment": {
                "prompt_identical_to_full": identical,
                "prompt_compared_to_full": compared,
                "status": treatment_status,
                "provenance": TREATMENT_PROVENANCE,
            },
        }
    return output


def _metric_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    runs = report["runs"]
    full = runs.get("full", {}).get("overall")
    if full is None:
        raise ValueError("A full run is required to calculate deltas.")
    rows = []
    for name in sorted(runs):
        run = runs[name]
        overall = run["overall"]
        rows.append({
            "run": name,
            "samples": overall["total"],
            "valid_predictions": run["valid_predictions"],
            "failures": run["failures"],
            "correct": overall["correct"],
            "accuracy": overall["accuracy"],
            "macro_f1": overall["macro_f1"],
            "accuracy_delta_vs_full": overall["accuracy"] - full["accuracy"],
            "macro_f1_delta_vs_full": overall["macro_f1"] - full["macro_f1"],
            "prompt_identical_to_full": run["treatment"]["prompt_identical_to_full"],
            "prompt_compared_to_full": run["treatment"]["prompt_compared_to_full"],
            "treatment_status": run["treatment"]["status"],
            "treatment_provenance": run["treatment"]["provenance"],
        })
    return rows


def render_metrics_csv(report: dict[str, Any]) -> str:
    """Render the required overall metrics table as stable UTF-8 CSV text."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=METRIC_FIELDS, lineterminator="\n")
    writer.writeheader()
    for raw_row in _metric_rows(report):
        row = dict(raw_row)
        for field in (
            "accuracy",
            "macro_f1",
            "accuracy_delta_vs_full",
            "macro_f1_delta_vs_full",
        ):
            row[field] = f"{row[field]:.6f}"
        writer.writerow(row)
    return buffer.getvalue()


def _noop_warning_lines(report: dict[str, Any]) -> list[str]:
    noops = [
        (name, run["treatment"])
        for name, run in sorted(report["runs"].items())
        if name != "full" and run["treatment"]["status"] == "no_op_identical_to_full"
    ]
    if not noops:
        return []
    names = " 与 ".join(f"`{name}`" for name, _ in noops)
    counts = {
        (treatment["prompt_identical_to_full"], treatment["prompt_compared_to_full"])
        for _, treatment in noops
    }
    if len(counts) == 1:
        identical, compared = counts.pop()
        identity = f"{names} 的输入 Prompt 均为 {identical}/{compared} 与 `full` 完全相同"
    else:
        details = "；".join(
            f"`{name}` 为 {treatment['prompt_identical_to_full']}/"
            f"{treatment['prompt_compared_to_full']}"
            for name, treatment in noops
        )
        identity = f"{details} 的输入 Prompt 与 `full` 完全相同"
    return [
        "⚠️ treatment 有效性警告：" + identity + "。",
        "这些运行属于 no-op 重复对照；指标差异只能视为重复调用波动，不能归因于消融组件。",
    ]


def _summary_table_lines(report: dict[str, Any]) -> list[str]:
    lines = [
        "| Run | Samples | Valid predictions | 调用失败 | Correct | Accuracy | Macro F1 | Δ Accuracy vs full | Δ Macro F1 vs full | Prompt identical/full | Treatment status | Provenance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in _metric_rows(report):
        lines.append(
            f"| {row['run']} | {row['samples']} | {row['valid_predictions']} | "
            f"{row['failures']} | {row['correct']} | {row['accuracy']:.2%} | "
            f"{row['macro_f1']:.2%} | {row['accuracy_delta_vs_full']:+.2%} | "
            f"{row['macro_f1_delta_vs_full']:+.2%} | "
            f"{row['prompt_identical_to_full']}/{row['prompt_compared_to_full']} | "
            f"{row['treatment_status']} | {row['treatment_provenance']} |"
        )
    return lines


def render_summary(report: dict[str, Any]) -> str:
    lines = ["# Codex CLI 情绪识别消融摘要", ""]
    warning_lines = _noop_warning_lines(report)
    if warning_lines:
        lines.extend([*warning_lines, ""])
    lines.extend(_summary_table_lines(report))
    return "\n".join(lines) + "\n"


def _slice_table(report: dict[str, Any], key: str, values: tuple[str, ...]) -> list[str]:
    lines = [
        "| Run | Slice | Samples | Correct | Accuracy | Macro F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name in sorted(report["runs"]):
        for value in values:
            metric = report["runs"][name][key][value]
            lines.append(
                f"| {name} | {value} | {metric['total']} | {metric['correct']} | "
                f"{metric['accuracy']:.2%} | {metric['macro_f1']:.2%} |"
            )
    return lines


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _confusion_lines(report: dict[str, Any]) -> list[str]:
    lines = [
        "| Run | Expected | Predicted | Count |",
        "| --- | --- | --- | ---: |",
    ]
    found = False
    for name in sorted(report["runs"]):
        counts = Counter(
            (error["expected"], error["predicted"] or "<missing>")
            for error in report["runs"][name]["overall"]["errors"]
        )
        for (expected, predicted), count in sorted(counts.items()):
            found = True
            lines.append(f"| {name} | {expected} | {predicted} | {count} |")
    if not found:
        lines.append("| - | - | - | 0 |")
    return lines


def _error_example_lines(report: dict[str, Any]) -> list[str]:
    annotations = {
        annotation.get("case_id", ""): annotation
        for annotation in report.get("annotations", [])
    }
    lines = []
    for name in sorted(report["runs"]):
        for error in report["runs"][name]["overall"]["errors"][:3]:
            annotation = annotations.get(error.get("case_id", ""), {})
            text = _escape_cell(annotation.get("current_input", "<not available>"))
            predicted = error["predicted"] or "<missing>"
            lines.append(
                f"- `{name}` / `{error.get('case_id') or 'unknown'}`: "
                f"expected=`{error['expected']}`, predicted=`{predicted}`; 输入：{text}"
            )
    return lines or ["- 未发现分类错误。"]


def render_chinese_report(
    report: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Render a deterministic Chinese Markdown report without environment reads."""
    metadata = metadata or {}
    lines = ["# Codex CLI 情绪识别消融实验报告", ""]
    if metadata:
        lines.extend(["## 实验元数据", ""])
        for key in sorted(metadata):
            lines.append(f"- {key}: `{_escape_cell(metadata[key])}`")
        lines.append("")

    warning_lines = _noop_warning_lines(report)
    if warning_lines:
        noop_count = sum(
            name != "full"
            and run["treatment"]["status"] == "no_op_identical_to_full"
            for name, run in report["runs"].items()
        )
        if noop_count == 1:
            noop_subject = "该运行"
        elif noop_count == 2:
            noop_subject = "这两组"
        else:
            noop_subject = f"上述 {noop_count} 组运行"
        lines.extend([
            "## 结论有效性警告",
            "",
            warning_lines[0].removeprefix("⚠️ treatment 有效性警告："),
            (
                f"{noop_subject}是 no-op 重复对照，其指标差异是重复调用波动，不是消融效果；"
                "不能据此归因情绪历史或上下文长度的组件贡献。"
            ),
            "",
        ])

    lines.extend(["## 整体结果", "", *_summary_table_lines(report), ""])
    lines.extend(["## 语言切片", "", *_slice_table(report, "languages", LANGUAGES), ""])
    lines.extend([
        "## 上下文依赖切片",
        "",
        *_slice_table(report, "contexts", CONTEXT_LEVELS),
        "",
    ])
    lines.extend(["## 标签混淆", "", *_confusion_lines(report), ""])
    lines.extend(["## 错误样例", "", *_error_example_lines(report), ""])
    lines.extend([
        "## 方法",
        "",
        "- 使用现有 `evaluate_records` 进行 case_id 匹配、Accuracy 和 Macro F1 计算。",
        "- treatment 有效性来自逐条记录的 `input` Prompt 与同 `case_id` 的 `full` Prompt 比对；"
        "全部相同时标记为 `no_op_identical_to_full`。",
        "- 按 `language` 和 `context_dependency` 的预定义枚举值切片，空切片显式记为 0。",
        "- `success is not True` 单独计为调用失败，失败预测仍保留在标注分母中。",
        "",
        "## 局限性",
        "",
        f"- 本次基准仅包含 {len(report.get('annotations', []))} 条合成 seed 记录；"
        f"高上下文依赖样本为 {sum(item.get('context_dependency') == 'high' for item in report.get('annotations', []))} 条。",
        "- 本实验评估的是 Codex CLI Agent 执行链路，不是裸模型 API 评测；"
        "结果包含 Codex 系统指令和 Agent 运行环境的影响。",
        "- 切片样本较少时，Accuracy 和 Macro F1 波动较大，不应单独解读。",
        "- 调用失败同时会拉低指标，需与分类错误分开观察。",
        "- `zero_shot` 同时禁用 few-shot 示例和情绪历史先验，因此属于组合消融；"
        "其相对 `full` 的指标差值不能单独归因于任一组件。",
    ])
    execution_note = metadata.get("execution_note")
    if execution_note:
        lines.append(f"- 执行过程说明：{_escape_cell(execution_note)}")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Chinese reports for Codex emotion ablations."
    )
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--run", action="append", required=True, help="NAME=path/to/analysis.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--commit")
    parser.add_argument("--codex-version")
    parser.add_argument("--branch")
    parser.add_argument("--started-at")
    parser.add_argument("--ended-at")
    parser.add_argument("--model")
    parser.add_argument(
        "--execution-note",
        help="Explicit reproducibility note, such as capacity interruption and resume history.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = {}
    for item in args.run:
        if "=" not in item:
            raise ValueError("--run must use NAME=PATH")
        name, raw_path = item.split("=", 1)
        if not name or name in runs:
            raise ValueError(f"Invalid or duplicate run name: {name!r}")
        runs[name] = load_records(Path(raw_path))

    report = build_report_data(runs, load_records(Path(args.seed_file)))
    metadata = {
        key: value
        for key, value in {
            "commit": args.commit,
            "codex_version": args.codex_version,
            "branch": args.branch,
            "started_at": args.started_at,
            "ended_at": args.ended_at,
            "model": args.model,
            "execution_note": args.execution_note,
        }.items()
        if value is not None
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.csv").write_text(render_metrics_csv(report), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_summary(report), encoding="utf-8")
    (output_dir / "report-zh.md").write_text(
        render_chinese_report(report, metadata=metadata),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
