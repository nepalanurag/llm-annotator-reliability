"""Tests for pipeline.monitoring: PSI and chi-square drift detection."""

import pytest

from pipeline.monitoring import check_drift, chi2_gof, psi


def test_psi_identical_distributions_near_zero():
    assert psi({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) < 1e-6


def test_psi_known_value():
    # Hand: (0.9-0.5)*ln(1.8) + (0.1-0.5)*ln(0.2) = 0.8789.
    assert psi({"a": 0.5, "b": 0.5}, {"a": 0.9, "b": 0.1}) == pytest.approx(
        0.8789, abs=1e-3
    )


def test_psi_handles_new_label():
    # A label absent from baseline must not blow up (epsilon smoothing).
    value = psi({"a": 1.0}, {"a": 0.9, "b": 0.1})
    assert value > 0.25  # significant shift, finite


def test_chi2_gof_no_drift():
    stat, p = chi2_gof({"a": 50, "b": 50}, {"a": 0.5, "b": 0.5})
    assert p > 0.05


def test_chi2_gof_detects_shift():
    stat, p = chi2_gof({"a": 90, "b": 10}, {"a": 0.5, "b": 0.5})
    assert p < 1e-6


def test_check_drift_ok_verdict():
    verdict = check_drift({"a": 0.5, "b": 0.5}, {"a": 52, "b": 48})
    assert verdict["verdict"] == "OK"
    assert verdict["alert"] is False


def test_check_drift_alert_verdict():
    verdict = check_drift({"a": 0.5, "b": 0.5}, {"a": 90, "b": 10})
    assert verdict["verdict"] == "ALERT"
    assert verdict["alert"] is True
    assert verdict["psi"] > 0.25


def test_check_drift_rejects_empty():
    with pytest.raises(ValueError):
        check_drift({"a": 0.5}, {})
