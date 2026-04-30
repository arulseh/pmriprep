"""Tests for `_warn_invalid_cpi_combo` — known-bad CPI inference + infer_cpt
combinations identified by the DRO matrix experiments.

Reference: tests/benchmarks/run_cpi_infer_cpt_map_vs_vb.py.
"""

from __future__ import annotations

import logging

import pytest

from pmriprep.interfaces.dsc_compute import _warn_invalid_cpi_combo


class TestWarnInvalidCPICombo:
    """Cover the four CPI inference × cpi_infer_cpt cells."""

    def test_nlls_without_infer_cpt_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="pmriprep"):
            _warn_invalid_cpi_combo("cpi", "nlls", False)
        assert any("2-3x larger CBF bias" in rec.message for rec in caplog.records)

    def test_map_with_infer_cpt_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="pmriprep"):
            _warn_invalid_cpi_combo("cpi", "map", True)
        assert any(
            "MAP point estimate fails" in rec.message for rec in caplog.records
        )

    def test_vi_with_infer_cpt_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="pmriprep"):
            _warn_invalid_cpi_combo("cpi", "vi", True)
        assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)

    def test_nlls_with_infer_cpt_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="pmriprep"):
            _warn_invalid_cpi_combo("cpi", "nlls", True)
        assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)

    def test_vi_without_infer_cpt_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        # VI + fixed CPs is the FABBER --infer-cpt false default; not unstable.
        with caplog.at_level(logging.WARNING, logger="pmriprep"):
            _warn_invalid_cpi_combo("cpi", "vi", False)
        assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)

    def test_map_without_infer_cpt_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        # MAP + fixed CPs is comparable to VI per the matrix; not unstable.
        with caplog.at_level(logging.WARNING, logger="pmriprep"):
            _warn_invalid_cpi_combo("cpi", "map", False)
        assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)

    def test_non_cpi_method_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        # Vascular and SVD methods don't have the CPI-specific failure modes.
        with caplog.at_level(logging.WARNING, logger="pmriprep"):
            _warn_invalid_cpi_combo("vascular", "nlls", False)
            _warn_invalid_cpi_combo("oSVD", "map", True)
        assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)
