"""Paired statistical inference for emotion-ablation predictions."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Final, Sequence

BOOTSTRAP_SAMPLES: Final = 10_000
BOOTSTRAP_SEED: Final = 20_260_803


@dataclass(frozen=True, slots=True)
class PairedObservation:
    """One benchmark case predicted by the baseline and one treatment."""

    expected: str
    baseline_prediction: str
    treatment_prediction: str


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Paired deltas, confidence intervals, and exact McNemar evidence."""

    accuracy_delta: float
    accuracy_delta_ci95_low: float
    accuracy_delta_ci95_high: float
    macro_f1_delta: float
    macro_f1_delta_ci95_low: float
    macro_f1_delta_ci95_high: float
    full_only_correct: int
    treatment_only_correct: int
    mcnemar_exact_p_value: float
    bootstrap_samples: int
    bootstrap_seed: int


@dataclass(frozen=True, slots=True)
class InvalidPairedComparisonError(Exception):
    """The paired comparison inputs cannot define an inference result."""

    reason: str

    def __str__(self) -> str:
        return self.reason


def compare_paired_predictions(
    observations: Sequence[PairedObservation],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> PairedComparison:
    """Compare a treatment with its case-matched full baseline."""
    if not observations:
        raise InvalidPairedComparisonError(
            reason="paired comparison requires at least one observation"
        )
    if bootstrap_samples <= 0:
        raise InvalidPairedComparisonError(
            reason="bootstrap_samples must be positive"
        )

    labels = sorted({observation.expected for observation in observations})
    baseline_correct = [
        observation.baseline_prediction == observation.expected
        for observation in observations
    ]
    treatment_correct = [
        observation.treatment_prediction == observation.expected
        for observation in observations
    ]
    full_only = sum(
        baseline and not treatment
        for baseline, treatment in zip(baseline_correct, treatment_correct)
    )
    treatment_only = sum(
        treatment and not baseline
        for baseline, treatment in zip(baseline_correct, treatment_correct)
    )

    accuracy_delta = _accuracy_delta(observations)
    macro_f1_delta = _macro_f1_delta(observations, labels)
    accuracy_bootstrap, macro_f1_bootstrap = _paired_bootstrap(
        observations,
        labels,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    accuracy_low, accuracy_high = _percentile_interval(accuracy_bootstrap)
    macro_f1_low, macro_f1_high = _percentile_interval(macro_f1_bootstrap)
    return PairedComparison(
        accuracy_delta=accuracy_delta,
        accuracy_delta_ci95_low=accuracy_low,
        accuracy_delta_ci95_high=accuracy_high,
        macro_f1_delta=macro_f1_delta,
        macro_f1_delta_ci95_low=macro_f1_low,
        macro_f1_delta_ci95_high=macro_f1_high,
        full_only_correct=full_only,
        treatment_only_correct=treatment_only,
        mcnemar_exact_p_value=_exact_mcnemar_p_value(full_only, treatment_only),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def _accuracy_delta(observations: Sequence[PairedObservation]) -> float:
    delta = sum(
        (observation.treatment_prediction == observation.expected)
        - (observation.baseline_prediction == observation.expected)
        for observation in observations
    )
    return delta / len(observations)


def _macro_f1_delta(
    observations: Sequence[PairedObservation],
    labels: Sequence[str],
) -> float:
    baseline = _macro_f1(
        [(item.expected, item.baseline_prediction) for item in observations],
        labels,
    )
    treatment = _macro_f1(
        [(item.expected, item.treatment_prediction) for item in observations],
        labels,
    )
    return treatment - baseline


def _macro_f1(pairs: Sequence[tuple[str, str]], labels: Sequence[str]) -> float:
    expected_counts = Counter(expected for expected, _ in pairs)
    predicted_counts = Counter(predicted for _, predicted in pairs)
    true_positive = Counter(
        expected for expected, predicted in pairs if expected == predicted
    )
    scores: list[float] = []
    for label in labels:
        tp = true_positive[label]
        fp = predicted_counts[label] - tp
        fn = expected_counts[label] - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return sum(scores) / len(scores) if scores else 0.0


def _paired_bootstrap(
    observations: Sequence[PairedObservation],
    labels: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    generator = random.Random(seed)
    accuracy_deltas: list[float] = []
    macro_f1_deltas: list[float] = []
    for _ in range(samples):
        resampled = [
            observations[generator.randrange(len(observations))]
            for _ in observations
        ]
        accuracy_deltas.append(_accuracy_delta(resampled))
        macro_f1_deltas.append(_macro_f1_delta(resampled, labels))
    return accuracy_deltas, macro_f1_deltas


def _percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return _percentile(ordered, 0.025), _percentile(ordered, 0.975)


def _percentile(ordered: Sequence[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _exact_mcnemar_p_value(full_only: int, treatment_only: int) -> float:
    discordant = full_only + treatment_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, successes)
        for successes in range(min(full_only, treatment_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)
