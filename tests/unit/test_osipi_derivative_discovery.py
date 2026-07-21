"""Regression tests for entity-aware OSIPI derivative discovery."""

from pathlib import Path

import pytest

from tests.osipi import conftest as osipi


def _set_output_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(osipi, "OUTPUT_ROOT", tmp_path)
    perf = tmp_path / "test-retest" / "sub-0081" / "ses-0001" / "perf"
    perf.mkdir(parents=True)
    return perf


def test_resolves_bare_and_entity_preserving_derivatives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    perf = _set_output_root(monkeypatch, tmp_path)
    bare = perf / "sub-0081_ses-0001_desc-dceVe_stat.nii.gz"
    entity_rich = perf / (
        "sub-0081_ses-0001_acq-fl3d_ce-GAD_run-1_desc-dceKtrans_stat.nii.gz"
    )
    bare.touch()
    entity_rich.touch()

    assert osipi.get_map_path("sub-0081", "ses-0001", "ve") == bare
    assert osipi.get_map_path("sub-0081", "ses-0001", "Ktrans") == entity_rich


def test_exact_suffix_excludes_uncertainty_maps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    perf = _set_output_root(monkeypatch, tmp_path)
    expected = perf / (
        "sub-0081_ses-0001_run-1_desc-dceKtrans_stat.nii.gz"
    )
    uncertainty = perf / (
        "sub-0081_ses-0001_run-1_desc-dceStandardErrorKtrans_stat.nii.gz"
    )
    expected.touch()
    uncertainty.touch()

    assert osipi.get_map_path("sub-0081", "ses-0001", "Ktrans") == expected


def test_ambiguous_derivative_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    perf = _set_output_root(monkeypatch, tmp_path)
    (perf / "sub-0081_ses-0001_run-1_desc-dceKtrans_stat.nii.gz").touch()
    (perf / "sub-0081_ses-0001_run-2_desc-dceKtrans_stat.nii.gz").touch()

    with pytest.raises(RuntimeError, match="Ambiguous"):
        osipi.get_map_path("sub-0081", "ses-0001", "Ktrans")


def test_resolves_entity_preserving_dce_transform(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(osipi, "OUTPUT_ROOT", tmp_path)
    anat = tmp_path / "test-retest" / "sub-0081" / "ses-0001" / "anat"
    anat.mkdir(parents=True)
    expected = anat / (
        "sub-0081_ses-0001_acq-fl3d_run-1_"
        "from-dceref_to-T1w_mode-image_xfm.h5"
    )
    expected.touch()

    assert osipi._get_dce_to_anat_xfm("sub-0081", "ses-0001") == expected
