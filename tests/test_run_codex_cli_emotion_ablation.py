import subprocess
import sys

from scripts import run_codex_cli_emotion_ablation as runner


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


def test_direct_cli_help_works():
    result = subprocess.run(
        [sys.executable, "scripts/run_codex_cli_emotion_ablation.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Run emotion-recognition ablations with Codex CLI" in result.stdout
