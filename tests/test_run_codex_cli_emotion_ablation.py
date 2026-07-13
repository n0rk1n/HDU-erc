import json
import subprocess
import sys

import pytest

from scripts import run_codex_cli_emotion_ablation as runner


CASES = [
    {
        "id": "case-001",
        "turn_count": 4,
        "history": [
            {"role": "human", "content": "First human turn.", "emotion": "sad"},
            {"role": "ai", "content": "First AI turn."},
            {"role": "human", "content": "Final history turn.", "emotion": "anxious"},
            {"role": "ai", "content": "Final AI response."},
        ],
        "current_input": "Current input.",
    },
    {
        "id": "case-002",
        "history": [],
        "current_input": "I appreciate your help.",
    },
]


def test_build_command_uses_isolated_structured_exec(tmp_path):
    command = runner.build_command(tmp_path / "schema.json", model="gpt-test")
    assert command == [
        "codex", "exec", "--ephemeral", "--sandbox", "read-only",
        "--skip-git-repo-check", "--output-schema", str(tmp_path / "schema.json"),
        "--model", "gpt-test", "-",
    ]


def test_parse_result_accepts_supported_label():
    result = runner.parse_result('{"emotion":"anxious"}')
    assert result.emotion == "anxious"
    assert result.success is True


def test_parse_result_rejects_unknown_label():
    result = runner.parse_result('{"emotion":"relieved"}')
    assert result.success is False
    assert "Unsupported emotion" in result.error


def test_invoke_codex_passes_instruction_to_subprocess(tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, '{"emotion":"anxious"}\n', "")

    schema_file = tmp_path / "schema.json"
    result = runner.invoke_codex(
        "Emotion prompt",
        schema_file=schema_file,
        model=None,
        timeout=45,
        run_subprocess=fake_run,
    )

    assert result.emotion == "anxious"
    assert result.success is True
    command, kwargs = calls[0]
    assert command == [
        "codex", "exec", "--ephemeral", "--sandbox", "read-only",
        "--skip-git-repo-check", "--output-schema", str(schema_file), "-",
    ]
    assert kwargs == {
        "input": (
            "Classify the emotion using only the supplied prompt. "
            "Return exactly the JSON object required by the output schema.\n\n"
            "Emotion prompt"
        ),
        "text": True,
        "capture_output": True,
        "timeout": 45,
        "check": False,
    }


def test_invoke_codex_returns_timeout_failure(tmp_path):
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    result = runner.invoke_codex(
        "Emotion prompt",
        schema_file=tmp_path / "schema.json",
        model="gpt-test",
        timeout=1,
        run_subprocess=fake_run,
    )

    assert result.emotion == ""
    assert result.output == ""
    assert result.success is False
    assert "timed out" in result.error


def test_run_ablation_skips_existing_success_and_preserves_case_order(tmp_path):
    output_file = tmp_path / "full.json"
    output_file.write_text(json.dumps([{
        "case_id": "case-001", "run": "full", "emotion": "anxious", "success": True
    }]), encoding="utf-8")
    calls = []

    def fake_invoke(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return runner.CodexResult("grateful", '{"emotion":"grateful"}', True)

    records = runner.run_ablation(
        runner.RUN_CONFIGS["full"], CASES, output_file,
        schema_file=tmp_path / "schema.json", model=None, timeout=60,
        retries=1, emotion_interval=7, invoke=fake_invoke,
    )

    assert [record["case_id"] for record in records] == ["case-001", "case-002"]
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 60
    assert records[1]["emotion_interval"] == 7
    assert json.loads(output_file.read_text(encoding="utf-8")) == records
    assert not output_file.with_suffix(".json.tmp").exists()


def test_run_ablation_reinvokes_success_from_another_run(tmp_path):
    output_file = tmp_path / "full.json"
    output_file.write_text(json.dumps([{
        "case_id": "case-001",
        "run": "zero_shot",
        "emotion": "sad",
        "success": True,
    }]), encoding="utf-8")
    calls = []

    def fake_invoke(prompt, **kwargs):
        calls.append(prompt)
        return runner.CodexResult("anxious", '{"emotion":"anxious"}', True)

    records = runner.run_ablation(
        runner.RUN_CONFIGS["full"], CASES[:1], output_file,
        schema_file=tmp_path / "schema.json", model=None, timeout=60,
        retries=0, emotion_interval=5, invoke=fake_invoke,
    )

    assert len(calls) == 1
    assert records[0]["run"] == "full"
    assert records[0]["emotion"] == "anxious"
    assert json.loads(output_file.read_text(encoding="utf-8")) == records


def test_run_ablation_retries_failure_and_snapshots_last_result(tmp_path):
    results = [
        runner.CodexResult("", "bad", False, "Invalid JSON"),
        runner.CodexResult("grateful", '{"emotion":"grateful"}', True),
    ]
    calls = []

    def fake_invoke(prompt, **kwargs):
        calls.append(prompt)
        return results.pop(0)

    output_file = tmp_path / "full.json"
    records = runner.run_ablation(
        runner.RUN_CONFIGS["full"], CASES[:1], output_file,
        schema_file=tmp_path / "schema.json", model="gpt-test", timeout=60,
        retries=1, emotion_interval=5, invoke=fake_invoke,
    )

    assert len(calls) == 2
    assert records[0]["success"] is True
    assert records[0]["emotion"] == "grateful"
    assert records[0]["error"] == ""


def test_run_ablation_rejects_more_than_one_retry_before_invocation(tmp_path):
    calls = []

    def fake_invoke(prompt, **kwargs):
        calls.append(prompt)
        return runner.CodexResult("anxious", '{"emotion":"anxious"}', True)

    with pytest.raises(ValueError, match="retries must be 0 or 1"):
        runner.run_ablation(
            runner.RUN_CONFIGS["full"], CASES[:1], tmp_path / "full.json",
            schema_file=tmp_path / "schema.json", model=None, timeout=60,
            retries=2, emotion_interval=5, invoke=fake_invoke,
        )

    assert calls == []


@pytest.mark.parametrize(
    ("run_name", "included", "omitted"),
    [
        ("no_emotion_history", ["First human turn."], ["More likely emotion labels:"]),
        (
            "short_context",
            ["Final history turn.", "Final AI response.", "Current input."],
            ["First human turn.", "First AI turn."],
        ),
        ("zero_shot", ["Dialogue context:"], ["Dynamic EICL examples:", "Labeled examples:"]),
    ],
)
def test_run_ablation_preserves_prompt_variants(tmp_path, run_name, included, omitted):
    prompts = []

    def fake_invoke(prompt, **kwargs):
        prompts.append(prompt)
        return runner.CodexResult("anxious", '{"emotion":"anxious"}', True)

    runner.run_ablation(
        runner.RUN_CONFIGS[run_name], CASES[:1], tmp_path / f"{run_name}.json",
        schema_file=tmp_path / "schema.json", model=None, timeout=60,
        retries=0, emotion_interval=5, invoke=fake_invoke,
    )

    for text in included:
        assert text in prompts[0]
    for text in omitted:
        assert text not in prompts[0]


def test_main_runs_selected_configs_with_limit_and_cli_options(tmp_path, monkeypatch):
    dialogues_file = tmp_path / "dialogues.jsonl"
    dialogues_file.write_text(
        "\n".join(json.dumps(case) for case in CASES) + "\n",
        encoding="utf-8",
    )
    calls = []

    def fake_run(config, cases, output_file, **kwargs):
        calls.append((config, cases, output_file, kwargs))
        return []

    monkeypatch.setattr(runner, "run_ablation", fake_run)

    result = runner.main([
        "--dialogues-file", str(dialogues_file),
        "--output-dir", str(tmp_path / "out"),
        "--schema-file", str(tmp_path / "schema.json"),
        "--limit", "1",
        "--run", "full",
        "--model", "gpt-test",
        "--timeout", "45",
        "--retries", "1",
        "--emotion-interval", "7",
    ])

    assert result == 0
    assert len(calls) == 1
    config, cases, output_file, kwargs = calls[0]
    assert config is runner.RUN_CONFIGS["full"]
    assert cases == CASES[:1]
    assert output_file == tmp_path / "out" / "full.json"
    assert kwargs == {
        "schema_file": tmp_path / "schema.json",
        "model": "gpt-test",
        "timeout": 45,
        "retries": 1,
        "emotion_interval": 7,
    }


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--limit", "0"),
        ("--timeout", "0"),
        ("--emotion-interval", "0"),
        ("--retries", "-1"),
        ("--retries", "2"),
    ],
)
def test_main_rejects_invalid_numeric_option_before_loading_dialogues(option, value, monkeypatch):
    def fail_load_dialogues(path):
        raise AssertionError("dialogues must not be loaded for invalid arguments")

    monkeypatch.setattr(runner, "load_dialogues", fail_load_dialogues)

    with pytest.raises(SystemExit) as exc_info:
        runner.main([
            "--dialogues-file", "dialogues.jsonl",
            "--output-dir", "out",
            option, value,
        ])

    assert exc_info.value.code == 2


def test_direct_cli_help_works():
    result = subprocess.run(
        [sys.executable, "scripts/run_codex_cli_emotion_ablation.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Run emotion-recognition ablations with Codex CLI" in result.stdout
