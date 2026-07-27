import csv
import io
from pathlib import Path

from chatbot.emotion_labels import EMOTION_LABEL_SET
from scripts.benchmark.emotion_benchmark import load_jsonl, validate_records
from scripts.benchmark.prepare_empathetic_dialogues import (
    build_benchmark_records,
    select_balanced_seed,
)


def _rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def test_build_benchmark_records_uses_last_target_speaker_turn_and_history():
    rows = _rows(
        "conv_id,utterance_idx,context,prompt,speaker_idx,utterance,selfeval,tags\n"
        "hit:1_conv:2,1,guilty,A situation,10,I caused a delay_comma_ and felt bad.,,\n"
        "hit:1_conv:2,2,guilty,A situation,20,What happened?,,\n"
        "hit:1_conv:2,3,guilty,A situation,10,It affected the whole team.,,\n"
        "hit:1_conv:2,4,guilty,A situation,20,That sounds difficult.,,\n"
    )

    records = build_benchmark_records(rows, split="test")

    assert records == [
        {
            "case_id": "ed-test-hit-1-conv-2",
            "language": "en",
            "subset": "empathetic_dialogues_test",
            "expected": "guilty",
            "turn_count": 3,
            "history": [
                {"role": "human", "content": "I caused a delay, and felt bad."},
                {"role": "ai", "content": "What happened?"},
            ],
            "current_input": "It affected the whole team.",
            "scenario": "open_domain_dialogue",
            "annotation_status": "released",
            "source_stage": "release",
            "label_provenance": "human_authored_emotion_grounding",
            "context_dependency": "medium",
            "quality_flags": [],
            "source_dataset": "EmpatheticDialogues",
            "source_split": "test",
            "source_conversation_id": "hit:1_conv:2",
            "source_utterance_idx": 3,
            "source_situation": "A situation",
            "source_license": "CC BY-NC 4.0",
            "rationale": (
                "The expected label is the conversation-level emotion grounding "
                "provided by EmpatheticDialogues, not a post-hoc utterance annotation."
            ),
        }
    ]


def test_balanced_seed_is_deterministic_and_covers_every_label():
    records = []
    for label in sorted(EMOTION_LABEL_SET):
        for index in range(3):
            records.append({"case_id": f"{label}-{index}", "expected": label})

    selected = select_balanced_seed(records, per_label=2)

    assert len(selected) == 64
    assert {record["expected"] for record in selected} == EMOTION_LABEL_SET
    assert [record["case_id"] for record in selected[:2]] == ["afraid-0", "afraid-1"]
    assert {record["subset"] for record in selected} == {
        "empathetic_dialogues_balanced_seed"
    }


def test_checked_in_empathetic_dialogues_release_is_real_and_valid():
    root = "data/benchmarks/empathetic_dialogues_v1/release"
    test_records = load_jsonl(Path(root) / "test.jsonl")
    seed_records = load_jsonl(Path(root) / "balanced_seed.jsonl")

    assert len(test_records) == 2542
    assert len(seed_records) == 64
    assert {record["expected"] for record in test_records} == EMOTION_LABEL_SET
    assert {record["expected"] for record in seed_records} == EMOTION_LABEL_SET
    assert {record["label_provenance"] for record in test_records} == {
        "human_authored_emotion_grounding"
    }
    assert validate_records(test_records) == []
    assert validate_records(seed_records) == []
