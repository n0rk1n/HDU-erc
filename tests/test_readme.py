from pathlib import Path


def test_readme_documents_current_stream_and_emotion_config_rules():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "/api/chat/streams" in readme
    assert "/api/chat/stream`" not in readme
    assert "任一 `EMOTION_LLM_*`" in readme
    assert "data/examples/emotion_labels_sample.json" in readme
    assert "data/examples/dynamic_eicl_sample.json" in readme
    assert "data/records/emotion_labels.json" not in readme


def test_documented_example_files_exist():
    assert Path("data/examples/emotion_labels_sample.json").exists()
    assert Path("data/examples/static_few_shot_sample.json").exists()
    assert Path("data/examples/dynamic_eicl_sample.json").exists()
