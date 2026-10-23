import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.trustworthy_ml.drift import calculate_psi

ROOT = Path(__file__).resolve().parents[1]


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


def test_drift_report_flags_simulated_shift():
    result = subprocess.run(
        [sys.executable, "-m", "src.trustworthy_ml.drift"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    report_path = ROOT / "reports" / "drift_report.json"
    report = json.loads(report_path.read_text())

    baseline = report["batches"]["baseline_test_batch"]
    shifted = report["batches"]["simulated_shift"]

    assert baseline["features_warning_or_critical"] == 0
    assert shifted["features"]["age"]["status"] == "critical"
    assert shifted["features"]["hours-per-week"]["status"] == "critical"
