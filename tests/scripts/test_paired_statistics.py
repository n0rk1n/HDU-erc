import pytest

from scripts.ablation.paired_statistics import (
    InvalidPairedComparisonError,
    PairedObservation,
    compare_paired_predictions,
)


def test_compare_paired_predictions_reports_exact_treatment_disadvantage():
    observations = [
        PairedObservation(
            expected="anxious" if index % 2 == 0 else "sad",
            baseline_prediction="anxious" if index % 2 == 0 else "sad",
            treatment_prediction="sad" if index % 2 == 0 else "anxious",
        )
        for index in range(8)
    ]

    result = compare_paired_predictions(observations)

    assert result.accuracy_delta == -1.0
    assert result.accuracy_delta_ci95_low == -1.0
    assert result.accuracy_delta_ci95_high == -1.0
    assert result.macro_f1_delta == -1.0
    assert result.macro_f1_delta_ci95_low == -1.0
    assert result.macro_f1_delta_ci95_high == -1.0
    assert result.full_only_correct == 8
    assert result.treatment_only_correct == 0
    assert result.mcnemar_exact_p_value == pytest.approx(0.0078125)


@pytest.mark.parametrize(
    ("observations", "bootstrap_samples", "reason"),
    [
        ([], 10_000, "at least one observation"),
        ([PairedObservation("sad", "sad", "sad")], 0, "must be positive"),
    ],
)
def test_compare_paired_predictions_rejects_invalid_inference_inputs(
    observations,
    bootstrap_samples,
    reason,
):
    with pytest.raises(InvalidPairedComparisonError, match=reason):
        compare_paired_predictions(
            observations,
            bootstrap_samples=bootstrap_samples,
        )
