import json
import subprocess
import sys

from scripts import run_emotion_ablation


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLlm:
    def __init__(self, outputs=None):
        self.outputs = list(outputs or ['{"primary_emotion":"anxious","confidence":0.9}'])
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.outputs[0], Exception):
            raise self.outputs.pop(0)
        return FakeResponse(self.outputs.pop(0))


def test_run_config_writes_successful_records(tmp_path):
    dialogues_file = tmp_path / "dialogues.jsonl"
    output_file = tmp_path / "nested" / "full.json"
    dialogues_file.write_text(
        '{"id":"case-001","turn_count":1,"history":[],"current_input":"I am nervous."}\n',
        encoding="utf-8",
    )
    llm = FakeLlm()

    result = run_emotion_ablation.run_config(
        run_emotion_ablation.RUN_CONFIGS["full"],
        dialogues_file,
        output_file,
        llm,
        emotion_interval=5,
    )

    assert result == 1
    records = json.loads(output_file.read_text(encoding="utf-8"))
    assert records[0]["case_id"] == "case-001"
    assert records[0]["run"] == "full"
    assert records[0]["emotion"] == "anxious"
    assert records[0]["success"] is True
    assert "Dialogue context:" in records[0]["input"]


def test_run_config_uses_history_emotion_metadata_in_prompt(tmp_path):
    dialogues_file = tmp_path / "dialogues.jsonl"
    output_file = tmp_path / "full.json"
    dialogues_file.write_text(
        json.dumps({
            "id": "case-001",
            "turn_count": 2,
            "history": [
                {"role": "human", "content": "I miss them.", "emotion": "sad"},
                {"role": "ai", "content": "That sounds heavy."},
            ],
            "current_input": "It still hurts.",
        })
        + "\n",
        encoding="utf-8",
    )
    llm = FakeLlm()

    run_emotion_ablation.run_config(
        run_emotion_ablation.RUN_CONFIGS["full"],
        dialogues_file,
        output_file,
        llm,
        emotion_interval=5,
    )

    records = json.loads(output_file.read_text(encoding="utf-8"))
    assert "More likely emotion labels: sad" in records[0]["input"]
    assert "More likely emotion labels: sad" in llm.prompts[0]


def test_run_config_records_failed_cases(tmp_path):
    dialogues_file = tmp_path / "dialogues.jsonl"
    output_file = tmp_path / "full.json"
    dialogues_file.write_text(
        '{"id":"case-001","turn_count":1,"history":[],"current_input":"I am nervous."}\n',
        encoding="utf-8",
    )
    llm = FakeLlm([RuntimeError("LLM request timed out")])

    run_emotion_ablation.run_config(
        run_emotion_ablation.RUN_CONFIGS["full"],
        dialogues_file,
        output_file,
        llm,
        emotion_interval=5,
    )

    records = json.loads(output_file.read_text(encoding="utf-8"))
    assert records[0]["case_id"] == "case-001"
    assert records[0]["emotion"] == ""
    assert records[0]["success"] is False
    assert records[0]["error"] == "LLM request timed out"


def test_run_config_rejects_invalid_jsonl(tmp_path):
    dialogues_file = tmp_path / "dialogues.jsonl"
    output_file = tmp_path / "full.json"
    dialogues_file.write_text("not json\n", encoding="utf-8")

    try:
        run_emotion_ablation.run_config(
            run_emotion_ablation.RUN_CONFIGS["full"],
            dialogues_file,
            output_file,
            FakeLlm(),
            emotion_interval=5,
        )
    except ValueError as exc:
        assert "Invalid JSONL" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_run_config_rejects_malformed_history_without_llm_call(tmp_path):
    dialogues_file = tmp_path / "dialogues.jsonl"
    output_file = tmp_path / "full.json"
    dialogues_file.write_text(
        '{"id":"case-001","turn_count":1,"history":["bad-entry"],"current_input":"I am nervous."}\n',
        encoding="utf-8",
    )
    llm = FakeLlm()

    try:
        run_emotion_ablation.run_config(
            run_emotion_ablation.RUN_CONFIGS["full"],
            dialogues_file,
            output_file,
            llm,
            emotion_interval=5,
        )
    except ValueError as exc:
        assert str(dialogues_file) in str(exc)
        assert ":1" in str(exc)
        assert "history[0]" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

    assert llm.prompts == []


def test_main_validates_dialogues_before_model_setup(tmp_path, monkeypatch):
    calls = []
    missing_dialogues_file = tmp_path / "missing.jsonl"

    def fail_load_config(argv):
        calls.append("load_config")
        raise AssertionError("load_config should not run before dialogue validation")

    def fail_build_chat_model(config):
        calls.append("build_chat_model")
        raise AssertionError("build_chat_model should not run before dialogue validation")

    monkeypatch.setattr(run_emotion_ablation, "load_config", fail_load_config)
    monkeypatch.setattr(run_emotion_ablation, "build_chat_model", fail_build_chat_model)

    try:
        run_emotion_ablation.main([
            "--dialogues-file",
            str(missing_dialogues_file),
            "--output-dir",
            str(tmp_path / "out"),
        ])
    except FileNotFoundError as exc:
        assert exc.filename == str(missing_dialogues_file) or str(missing_dialogues_file) in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")

    assert calls == []


def test_direct_cli_help_works():
    result = subprocess.run(
        [sys.executable, "scripts/run_emotion_ablation.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Run emotion-recognition ablation experiments" in result.stdout
