"""Download and convert EmpatheticDialogues into the project benchmark format."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import tarfile
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chatbot.emotion.labels import EMOTION_LABEL_SET
from scripts.benchmark.emotion_benchmark import validate_records, write_jsonl


SOURCE_URL = (
    "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/"
    "empatheticdialogues.tar.gz"
)
SOURCE_SHA256 = "56f234d77b7dd1f005fd365bb17769cfe346c3c84295b69bc069c8ccb83be03d"
SOURCE_MEMBER_TEMPLATE = "empatheticdialogues/{split}.csv"
DEFAULT_OUTPUT_ROOT = Path("data/benchmarks/empathetic_dialogues_v1")
LABEL_PROVENANCE = "human_authored_emotion_grounding"


def _restore_text(value: str) -> str:
    return value.replace("_comma_", ",").strip()


def _case_id(split: str, conversation_id: str) -> str:
    normalized = conversation_id.replace(":", "-").replace("_", "-")
    return f"ed-{split}-{normalized}"


def _context_dependency(history_length: int) -> str:
    if history_length == 0:
        return "none"
    if history_length == 1:
        return "low"
    if history_length == 2:
        return "medium"
    return "high"


def build_benchmark_records(
    rows: Iterable[dict[str, Any]], *, split: str, target: str = "situation"
) -> list[dict[str, Any]]:
    """Create one case per dialogue, separating aligned and weak-label targets."""
    if target not in {"situation", "context_diagnostic"}:
        raise ValueError("target must be situation or context_diagnostic")
    conversations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        conversation_id = str(row.get("conv_id", "")).strip()
        if not conversation_id:
            raise ValueError("EmpatheticDialogues row is missing conv_id")
        conversations[conversation_id].append(row)

    records = []
    for conversation_id, conversation in conversations.items():
        ordered = sorted(conversation, key=lambda row: int(str(row["utterance_idx"])))
        target_speaker = str(ordered[0].get("speaker_idx", "")).strip()
        target_turns = [
            (index, row)
            for index, row in enumerate(ordered)
            if str(row.get("speaker_idx", "")).strip() == target_speaker
            and _restore_text(str(row.get("utterance", "")))
        ]
        if not target_turns:
            raise ValueError(f"{conversation_id}: no non-empty target-speaker utterance")

        current_position, current_row = (
            target_turns[0] if target == "situation" else target_turns[-1]
        )
        label = str(current_row.get("context", "")).strip().lower()
        if label not in EMOTION_LABEL_SET:
            raise ValueError(f"{conversation_id}: unsupported emotion label {label!r}")

        history = [
            {
                "role": (
                    "human"
                    if str(row.get("speaker_idx", "")).strip() == target_speaker
                    else "ai"
                ),
                "content": _restore_text(str(row.get("utterance", ""))),
            }
            for row in ordered[:current_position]
            if _restore_text(str(row.get("utterance", "")))
        ]
        utterance_idx = int(str(current_row["utterance_idx"]))
        aligned = target == "situation"
        records.append(
            {
                "case_id": _case_id(split, conversation_id) + ("" if aligned else "-context"),
                "language": "en",
                "subset": (
                    f"empathetic_dialogues_{split}"
                    if aligned
                    else "empathetic_dialogues_context_diagnostic"
                ),
                "expected": label,
                "turn_count": utterance_idx,
                "history": history,
                "current_input": _restore_text(str(current_row["utterance"])),
                "scenario": "open_domain_dialogue",
                "annotation_status": "released",
                "source_stage": "release",
                "label_provenance": LABEL_PROVENANCE,
                "context_dependency": _context_dependency(len(history)),
                "quality_flags": [] if aligned else ["emotion_evidence_weak"],
                "source_dataset": "EmpatheticDialogues",
                "source_split": split,
                "source_conversation_id": conversation_id,
                "source_utterance_idx": utterance_idx,
                "source_situation": _restore_text(str(current_row.get("prompt", ""))),
                "source_license": "CC BY-NC 4.0",
                "evaluation_target": (
                    "conversation_situation_emotion"
                    if aligned
                    else "later_turn_current_emotion"
                ),
                "ground_truth_alignment": (
                    "aligned_first_speaker_grounding"
                    if aligned
                    else "weak_conversation_label_on_later_turn"
                ),
                "rationale": (
                    "The first target-speaker utterance initiates the human-authored "
                    "situation grounded by the conversation-level emotion label."
                    if aligned
                    else
                    "Diagnostic only: the conversation-level emotion label is not an "
                    "independent annotation of this later target-speaker utterance."
                ),
            }
        )
    return records


def select_balanced_seed(
    records: list[dict[str, Any]], *, per_label: int = 2
) -> list[dict[str, Any]]:
    if per_label <= 0:
        raise ValueError("per_label must be positive")
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[str(record.get("expected", ""))].append(record)

    missing = sorted(
        label
        for label in EMOTION_LABEL_SET
        if len(by_label[label]) < per_label
    )
    if missing:
        raise ValueError(
            f"Not enough records for {per_label} per label: {', '.join(missing)}"
        )

    selected = []
    # Round-robin ordering makes every prefix more representative: the first 32
    # records cover all labels once instead of concentrating on alphabetic labels.
    for occurrence in range(per_label):
        for label in sorted(EMOTION_LABEL_SET):
            record = by_label[label][occurrence]
            selected.append(
                {**record, "subset": "empathetic_dialogues_balanced_seed"}
            )
    return selected


def build_few_shot_examples(
    rows: Iterable[dict[str, Any]], *, per_label: int = 2
) -> list[dict[str, Any]]:
    """Select human-authored train examples only; test examples never enter prompts."""
    train_records = build_benchmark_records(rows, split="train", target="situation")
    selected = select_balanced_seed(train_records, per_label=per_label)
    return [
        {
            "example_id": record["case_id"],
            "dialogue": record["current_input"],
            "emotion": record["expected"],
            "source_dataset": "EmpatheticDialogues",
            "source_split": "train",
            "label_provenance": LABEL_PROVENANCE,
        }
        for record in selected
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path) -> None:
    actual = _sha256(path)
    if actual != SOURCE_SHA256:
        raise ValueError(
            f"Archive SHA-256 mismatch: expected {SOURCE_SHA256}, got {actual}"
        )


def load_source_rows(path: Path, *, split: str) -> list[dict[str, Any]]:
    member_name = SOURCE_MEMBER_TEMPLATE.format(split=split)
    with tarfile.open(path, "r:gz") as archive:
        member = archive.extractfile(member_name)
        if member is None:
            raise ValueError(f"Archive does not contain {member_name}")
        with io.TextIOWrapper(member, encoding="utf-8", newline="") as text_file:
            return list(csv.DictReader(text_file))


def prepare_dataset(
    archive_path: Path,
    output_root: Path,
    *,
    split: str = "test",
    per_label: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    verify_archive(archive_path)
    source_rows = load_source_rows(archive_path, split=split)
    records = build_benchmark_records(source_rows, split=split, target="situation")
    diagnostic_records = build_benchmark_records(
        source_rows, split=split, target="context_diagnostic"
    )
    seed_records = select_balanced_seed(records, per_label=per_label)
    few_shot_examples = build_few_shot_examples(
        load_source_rows(archive_path, split="train"), per_label=per_label
    )
    errors = (
        validate_records(records)
        + validate_records(diagnostic_records)
        + validate_records(seed_records)
    )
    if errors:
        raise ValueError("Converted records failed validation:\n" + "\n".join(errors))

    release_dir = output_root / "release"
    write_jsonl(release_dir / f"{split}.jsonl", records)
    write_jsonl(release_dir / "context_diagnostic.jsonl", diagnostic_records)
    write_jsonl(release_dir / "balanced_seed.jsonl", seed_records)
    write_jsonl(output_root / "few_shot" / "train_examples.jsonl", few_shot_examples)
    return records, seed_records, few_shot_examples


def download_archive(path: Path) -> None:
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "HDU-erc-dataset-preparer/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the official EmpatheticDialogues test split."
    )
    parser.add_argument(
        "--archive",
        help="Existing official empatheticdialogues.tar.gz; downloads a temporary copy when omitted.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--per-label", type=int, default=2)
    args = parser.parse_args(argv)
    if args.per_label <= 0:
        parser.error("--per-label must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.archive:
        archive_path = Path(args.archive)
        records, seed_records, few_shot_examples = prepare_dataset(
            archive_path,
            Path(args.output_root),
            per_label=args.per_label,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="empathetic-dialogues-") as directory:
            archive_path = Path(directory) / "empatheticdialogues.tar.gz"
            download_archive(archive_path)
            records, seed_records, few_shot_examples = prepare_dataset(
                archive_path,
                Path(args.output_root),
                per_label=args.per_label,
            )
    print(
        f"Prepared {len(records)} official test dialogues and "
        f"{len(seed_records)} balanced seed cases plus "
        f"{len(few_shot_examples)} train-only prompt examples in {args.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
