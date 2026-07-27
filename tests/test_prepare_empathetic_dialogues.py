import csv
import hashlib
import io
import json
from pathlib import Path

from chatbot.emotion_labels import EMOTION_LABEL_SET
from scripts.benchmark.emotion_benchmark import load_jsonl, validate_records
from scripts.benchmark.prepare_empathetic_dialogues import (
    build_benchmark_records,
    build_few_shot_examples,
    select_balanced_seed,
)


def _rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def test_build_benchmark_records_aligns_label_to_first_target_speaker_turn():
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
            "turn_count": 1,
            "history": [],
            "current_input": "I caused a delay, and felt bad.",
            "scenario": "open_domain_dialogue",
            "annotation_status": "released",
            "source_stage": "release",
            "label_provenance": "human_authored_emotion_grounding",
            "context_dependency": "none",
            "quality_flags": [],
            "source_dataset": "EmpatheticDialogues",
            "source_split": "test",
            "source_conversation_id": "hit:1_conv:2",
            "source_utterance_idx": 1,
            "source_situation": "A situation",
            "source_license": "CC BY-NC 4.0",
            "evaluation_target": "conversation_situation_emotion",
            "ground_truth_alignment": "aligned_first_speaker_grounding",
            "rationale": (
                "The first target-speaker utterance initiates the human-authored "
                "situation grounded by the conversation-level emotion label."
            ),
        }
    ]


def test_context_diagnostic_is_explicitly_marked_as_weak_later_turn_label():
    rows = _rows(
        "conv_id,utterance_idx,context,prompt,speaker_idx,utterance,selfeval,tags\n"
        "c1,1,guilty,A situation,10,I caused a delay.,,\n"
        "c1,2,guilty,A situation,20,What happened?,,\n"
        "c1,3,guilty,A situation,10,It affected the team.,,\n"
    )

    record = build_benchmark_records(
        rows, split="test", target="context_diagnostic"
    )[0]

    assert record["case_id"] == "ed-test-c1-context"
    assert record["current_input"] == "It affected the team."
    assert len(record["history"]) == 2
    assert record["ground_truth_alignment"] == "weak_conversation_label_on_later_turn"
    assert record["quality_flags"] == ["emotion_evidence_weak"]


def test_balanced_seed_is_deterministic_and_covers_every_label():
    records = []
    for label in sorted(EMOTION_LABEL_SET):
        for index in range(3):
            records.append({"case_id": f"{label}-{index}", "expected": label})

    selected = select_balanced_seed(records, per_label=2)

    assert len(selected) == 64
    assert {record["expected"] for record in selected} == EMOTION_LABEL_SET
    assert [record["case_id"] for record in selected[:2]] == ["afraid-0", "angry-0"]
    assert len({record["expected"] for record in selected[:32]}) == 32
    assert {record["subset"] for record in selected} == {
        "empathetic_dialogues_balanced_seed"
    }


def test_checked_in_empathetic_dialogues_release_is_real_and_valid():
    root = "data/benchmarks/empathetic_dialogues_v1/release"
    test_records = load_jsonl(Path(root) / "test.jsonl")
    seed_records = load_jsonl(Path(root) / "balanced_seed.jsonl")
    diagnostic_records = load_jsonl(Path(root) / "context_diagnostic.jsonl")
    examples = load_jsonl(
        Path("data/benchmarks/empathetic_dialogues_v1/few_shot/train_examples.jsonl")
    )

    assert len(test_records) == 2542
    assert len(seed_records) == 64
    assert len(diagnostic_records) == 2542
    assert len(examples) == 64
    assert {record["expected"] for record in test_records} == EMOTION_LABEL_SET
    assert {record["expected"] for record in seed_records} == EMOTION_LABEL_SET
    assert {record["label_provenance"] for record in test_records} == {
        "human_authored_emotion_grounding"
    }
    assert {record["context_dependency"] for record in test_records} == {"none"}
    assert {record["ground_truth_alignment"] for record in test_records} == {
        "aligned_first_speaker_grounding"
    }
    assert {record["source_split"] for record in examples} == {"train"}
    assert {record["emotion"] for record in examples} == EMOTION_LABEL_SET
    assert {record["case_id"] for record in test_records}.isdisjoint(
        {record["example_id"] for record in examples}
    )
    assert validate_records(test_records) == []
    assert validate_records(seed_records) == []
    assert validate_records(diagnostic_records) == []


def test_few_shot_examples_are_balanced_train_only_human_examples():
    rows = []
    for label in sorted(EMOTION_LABEL_SET):
        for index in range(2):
            rows.append({
                "conv_id": f"{label}_{index}",
                "utterance_idx": "1",
                "context": label,
                "prompt": "situation",
                "speaker_idx": "1",
                "utterance": f"Human example for {label} {index}",
            })

    examples = build_few_shot_examples(rows, per_label=2)

    assert len(examples) == 64
    assert {example["emotion"] for example in examples} == EMOTION_LABEL_SET
    assert {example["source_split"] for example in examples} == {"train"}


def test_checked_in_metadata_hashes_match_converted_artifacts():
    root = Path("data/benchmarks/empathetic_dialogues_v1")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    artifacts = {
        "converted_test_sha256": root / "release/test.jsonl",
        "converted_context_diagnostic_sha256": root / "release/context_diagnostic.jsonl",
        "converted_balanced_seed_sha256": root / "release/balanced_seed.jsonl",
        "train_few_shot_sha256": root / "few_shot/train_examples.jsonl",
    }

    for key, path in artifacts.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata[key]
