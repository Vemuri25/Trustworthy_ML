import pandas as pd
import pytest

from src.trustworthy_ml.drift import calculate_psi, feature_drift


def test_identical_distributions_have_near_zero_psi():
    psi = calculate_psi([50, 50], [50, 50])
    assert psi == pytest.approx(0.0, abs=1e-9)


def test_different_distributions_have_positive_psi():
    psi = calculate_psi([90, 10], [10, 90])
    assert psi > 0.20


def test_psi_is_finite_when_a_bin_is_empty():
    psi = calculate_psi([100, 0], [0, 100])
    assert psi is not None
    assert psi > 0


def test_drift_detector_flags_simulated_shift():
    reference = {
        "type": "numeric",
        "edges": [20, 30, 40, 50, 60],
        "reference_counts": [25, 25, 25, 25],
        "reference_missing_rate": 0.0,
    }

    # Balanced baseline: 25 observations in each reference bin.
    baseline = pd.Series(([25] * 25) + ([35] * 25)
                         + ([45] * 25) + ([55] * 25))
    baseline_result = feature_drift(baseline, reference)
    assert baseline_result["status"] == "low"
    assert baseline_result["psi"] == pytest.approx(0.0, abs=1e-9)

    # Shift all observations beyond the reference range.
    shifted = pd.Series([75] * 100)
    shifted_result = feature_drift(shifted, reference)

    assert shifted_result["status"] == "critical"
    assert shifted_result["psi"] >= 0.20
