from pathlib import Path


BENCHMARK_ROOT = Path("data/benchmarks/emotion_ablation_v2")


def test_benchmark_v2_structure_exists():
    expected_paths = [
        "README.md",
        "metadata.json",
        "schema.json",
        "guidelines/annotation_guidelines.md",
        "guidelines/label_taxonomy.md",
        "guidelines/bilingual_parallel_policy.md",
        "guidelines/quality_control.md",
        "raw_candidates/generated_candidates.jsonl",
        "annotation/annotator_a.jsonl",
        "annotation/annotator_b.jsonl",
        "annotation/adjudication.jsonl",
        "release/core_parallel.jsonl",
        "release/extended_independent.jsonl",
        "release/challenge.jsonl",
        "reports/dataset_card.md",
        "reports/quality_report.md",
    ]

    missing = [
        relative_path
        for relative_path in expected_paths
        if not (BENCHMARK_ROOT / relative_path).exists()
    ]

    assert missing == []
