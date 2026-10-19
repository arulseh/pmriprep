"""Tests for ``pmriprep.utils.dsc_qc_plots`` selectors and helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pmriprep.utils.dsc_qc_plots import (
    _select_dsc_qc_voxel,
    generate_dsc_leakage_qc_plot,
    generate_dsc_preprocessing_qc_plot,
)


@pytest.fixture
def cbf_volume() -> np.ndarray:
    """Synthetic 3-D CBF map with a clear median in the brain interior.

    Z-axis is deliberately ≥7 slices so 3-iteration erosion (used by
    the selector) leaves a non-empty interior in z.
    """
    cbf = np.zeros((20, 20, 11), dtype=np.float64)
    # Brain interior fills most of the volume with a graded CBF
    cbf[2:18, 2:18, 1:10] = 1.0
    cbf[5:15, 5:15, 3:8] = 2.0
    cbf[8:12, 8:12, 4:7] = 3.0
    return cbf


@pytest.fixture
def brain_mask(cbf_volume: np.ndarray) -> np.ndarray:
    return cbf_volume > 0


@pytest.fixture
def enhancing_mask(cbf_volume: np.ndarray) -> np.ndarray:
    """Small enhancing region with cbf>0."""
    enh = np.zeros_like(cbf_volume)
    enh[9:11, 9:11, 5:6] = 1.0
    return enh


def _binary_erosion(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    from scipy.ndimage import binary_erosion as _erosion
    return _erosion(mask, iterations=iterations)


class TestSelectDSCQCVoxelTiers:
    """Tier ordering: enhancing → eroded brain → centre."""

    def test_returns_coord_and_tier_tuple(
        self,
        cbf_volume: np.ndarray,
        brain_mask: np.ndarray,
    ) -> None:
        result = _select_dsc_qc_voxel(cbf_volume, brain_mask, _binary_erosion)
        assert isinstance(result, tuple) and len(result) == 2
        coord, tier = result
        assert isinstance(coord, tuple) and len(coord) == 3
        assert tier in {"enhancing", "eroded_brain", "center"}

    def test_enhancing_tier_wins_when_available(
        self,
        cbf_volume: np.ndarray,
        brain_mask: np.ndarray,
        enhancing_mask: np.ndarray,
    ) -> None:
        """When the enhancing mask has cbf>0 voxels, tier 1 must fire and
        return ``"enhancing"`` — even though tier 2 (eroded brain) also
        has many candidates.
        """
        coord, tier = _select_dsc_qc_voxel(
            cbf_volume, brain_mask, _binary_erosion,
            enhancing_mask=enhancing_mask,
        )
        assert tier == "enhancing"
        assert enhancing_mask[coord] > 0.5

    def test_eroded_brain_tier_when_no_enhancing_mask(
        self,
        cbf_volume: np.ndarray,
        brain_mask: np.ndarray,
    ) -> None:
        coord, tier = _select_dsc_qc_voxel(
            cbf_volume, brain_mask, _binary_erosion,
        )
        assert tier == "eroded_brain"
        assert brain_mask[coord]

    def test_eroded_brain_tier_when_enhancing_mask_empty(
        self,
        cbf_volume: np.ndarray,
        brain_mask: np.ndarray,
    ) -> None:
        empty_enh = np.zeros_like(cbf_volume)
        coord, tier = _select_dsc_qc_voxel(
            cbf_volume, brain_mask, _binary_erosion,
            enhancing_mask=empty_enh,
        )
        assert tier == "eroded_brain"

    def test_center_fallback_when_cbf_all_zero(
        self,
        brain_mask: np.ndarray,
    ) -> None:
        """If no voxel has cbf>0 (degenerate), the centre fallback fires."""
        cbf_zero = np.zeros_like(brain_mask, dtype=np.float64)
        coord, tier = _select_dsc_qc_voxel(
            cbf_zero, brain_mask, _binary_erosion,
        )
        assert tier == "center"
        assert brain_mask[coord]


@pytest.mark.parametrize(
    "analysis_baseline_frames, expected_tsnr, expected_cnr, expected_coverage",
    [(None, 50.0, 10.0, 1.0), (5, 31.6, 6.3, 0.667)],
)
def test_preprocessing_qc_uses_baseline_tsnr_and_signal_drop_cnr(
    tmp_path, monkeypatch, analysis_baseline_frames, expected_tsnr,
    expected_cnr, expected_coverage,
) -> None:
    """QC metrics use the same analysis baseline as exported quality maps."""
    from matplotlib.axes import Axes

    from pmriprep.interfaces.dsc_compute import ComputeDSCMaps

    original_axvline = Axes.axvline
    markers = []

    def record_axvline(ax, x=0, **kwargs):
        markers.append((x, kwargs.get("label")))
        return original_axvline(ax, x, **kwargs)

    monkeypatch.setattr(Axes, "axvline", record_axvline)
    curve = np.array([100.0, 102.0, 98.0, 104.0, 96.0, 75.0, 85.0])
    signal = np.broadcast_to(curve, (3, 3, 3, curve.size)).copy()
    signal[0, ..., 5:] = [90.0, 95.0]
    mask = np.ones((3, 3, 3), dtype=bool)

    _, stats = generate_dsc_preprocessing_qc_plot(
        signal,
        mask,
        baseline_frames=3,
        volume_tr_s=1.5,
        output_path=str(tmp_path / "dsc_qc.png"),
        analysis_baseline_frames=analysis_baseline_frames,
    )

    assert stats["mean_tsnr"] == pytest.approx(expected_tsnr)
    assert stats["mean_cnr"] == pytest.approx(expected_cnr)
    assert stats["fraction_cnr_gt_4"] == pytest.approx(expected_coverage)
    maps = ComputeDSCMaps()._compute_quality_maps(
        signal, analysis_baseline_frames or 3, mask.shape, mask,
    )
    assert stats["mean_tsnr"] == round(float(np.nanmean(maps.tsnr[mask])), 1)
    assert stats["mean_cnr"] == round(float(np.nanmean(maps.cnr[mask])), 1)
    assert stats["fraction_cnr_gt_4"] == round(float(np.mean(maps.cnr[mask] > 4)), 3)
    assert markers == (
        [] if analysis_baseline_frames is None
        else [(6.75, "Analysis baseline end (5 frames)")]
    )


def test_leakage_qc_accepts_signed_k2(tmp_path) -> None:
    """The leakage montage supports finite coefficients on both sides of zero."""
    shape = (5, 5, 5)
    k2 = np.linspace(-0.1, 0.1, np.prod(shape)).reshape(shape)
    auc_raw = np.ones(shape)
    auc_diff = k2 * 2.0

    output = generate_dsc_leakage_qc_plot(
        k2_map=k2,
        auc_raw=auc_raw,
        auc_corr=auc_raw + auc_diff,
        auc_diff=auc_diff,
        brain_mask=np.ones(shape, dtype=bool),
        output_path=str(tmp_path / "signed_k2_qc.png"),
    )

    assert output == str(tmp_path / "signed_k2_qc.png")
    assert (tmp_path / "signed_k2_qc.png").is_file()


def test_leakage_qc_direct_api_handles_all_nan_k2(tmp_path) -> None:
    """The plotting utility remains total when called without its workflow guard."""
    shape = (5, 5, 5)
    timepoints = 12

    output = generate_dsc_leakage_qc_plot(
        k2_map=np.full(shape, np.nan),
        auc_raw=np.ones(shape),
        auc_corr=np.ones(shape),
        auc_diff=np.zeros(shape),
        brain_mask=np.ones(shape, dtype=bool),
        conc_raw_4d=np.zeros(shape + (timepoints,)),
        conc_corr_4d=np.zeros(shape + (timepoints,)),
        output_path=str(tmp_path / "undefined_k2_qc.png"),
    )

    assert output == str(tmp_path / "undefined_k2_qc.png")
    assert (tmp_path / "undefined_k2_qc.png").is_file()


@pytest.mark.parametrize("domain", ["signal", "concentration"])
def test_fit_plot_compares_observation_and_prediction_in_same_domain(tmp_path, monkeypatch, domain) -> None:
    """Original signal is the observation for a saved CPI signal prediction."""
    import matplotlib.pyplot as plt
    from pmriprep.utils.dsc_qc_plots import generate_dsc_fit_qc_plot

    original = plt.subplots
    captured = []

    def subplots(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(plt, "subplots", subplots)
    concentration = np.array([0., 0., 2., 1., .1])
    signal = 100*np.exp(-.03*concentration)
    observed = signal if domain == "signal" else concentration
    prediction = observed*.99
    output, _ = generate_dsc_fit_qc_plot(
        tissue_concentration=concentration, model_prediction=prediction,
        aif=concentration, time_points=np.arange(5.)*2,
        cbf_map=np.ones((3, 3, 3)), brain_mask=np.ones((3, 3, 3)),
        qc_voxel=(1, 1, 1), qc_voxel_cbf=1., raw_signal=signal,
        fit_domain=domain, output_path=str(tmp_path / (domain + ".png")),
    )
    ax = captured[0][1][0, 1]
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), observed)
    np.testing.assert_array_equal(ax.lines[1].get_ydata(), prediction)
    assert ("Signal" in ax.get_ylabel()) == (domain == "signal")
    assert output == str(tmp_path / (domain + ".png"))
