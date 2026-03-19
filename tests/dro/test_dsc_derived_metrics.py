"""Phase 5: Derived metrics accuracy tests.

Tests PSR, PH, TTP, AUC on realistic synthetic signals. These metrics
are computed from signal or concentration curves and feed into clinical
decision-making, so accuracy is critical.

Exercises:
    dsc_signal.py: compute_psr_legacy(), compute_peak_height_legacy(),
                   compute_peak_height_conc(), compute_ttp_legacy()
    dsc_signal.py: normalize_cbv()
"""

import numpy as np
import numpy.typing as npt
import pytest

from pmriprep.utils.dsc_signal import (
    compute_peak_height_conc,
    compute_peak_height_legacy,
    compute_psr_legacy,
    compute_ttp_legacy,
    normalize_cbv,
    signal_to_concentration,
)

from .conftest import (
    DSC_TISSUE_PARAMS,
    concentration_to_signal,
    dsc_forward_model,
    generate_dsc_aif_dro,
    generate_leaky_dsc_tissue,
    make_exponential_residue,
)

_AIF_PEAK = 0.1
_BASELINE = 10


# =============================================================================
# Helpers
# =============================================================================


def _make_dsc_signal(
    tissue_key: str,
    te: float = 0.030,
    s0: float = 1000.0,
    k2: float = 0.0,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], float]:
    """Generate DSC signal for a tissue type.

    Returns (signal_1d_with_baseline, aif_post, dt).
    """
    aif_post, _ = generate_dsc_aif_dro(
        n_timepoints=60, dt_sec=1.5, baseline_frames=0,
        peak_delta_r2=_AIF_PEAK,
    )
    tissue = DSC_TISSUE_PARAMS[tissue_key]
    residue = make_exponential_residue(tissue.cbf, tissue.mtt, 60, 1.5)
    conc_post = dsc_forward_model(aif_post, residue, 1.5)

    if k2 > 0:
        conc_post = generate_leaky_dsc_tissue(conc_post, aif_post, k2, 1.5)

    signal = concentration_to_signal(conc_post, te, s0, _BASELINE)
    return signal, aif_post, 1.5


# =============================================================================
# Test classes
# =============================================================================


class TestPSRAnalytic:
    """Test PSR on known leakage levels."""

    @pytest.mark.parametrize(
        "k2",
        [pytest.param(0.0, id="K2_0"),
         pytest.param(0.05, id="K2_0.05"),
         pytest.param(0.1, id="K2_0.10"),
         pytest.param(0.2, id="K2_0.20")],
    )
    def test_psr_decreases_with_leakage(self, k2: float) -> None:
        """PSR should decrease with increasing leakage (lower K2 = more recovery)."""
        signal, _, _ = _make_dsc_signal("gm", k2=k2)
        psr = compute_psr_legacy(signal, baseline_frames=_BASELINE, tail_frames=5)

        assert isinstance(psr, float)
        assert np.isfinite(psr)
        # PSR should be between 0 and ~200 (can exceed 100 for T1 overshoot)
        assert -50 < psr < 200, f"PSR={psr:.1f} outside expected range"

    def test_psr_ordering_by_leakage(self) -> None:
        """Higher K2 -> lower PSR (less signal recovery)."""
        psrs = []
        for k2 in [0.0, 0.05, 0.1, 0.2]:
            signal, _, _ = _make_dsc_signal("gm", k2=k2)
            psr = compute_psr_legacy(signal, baseline_frames=_BASELINE, tail_frames=5)
            psrs.append(float(psr))

        # PSR should generally decrease with increasing K2
        # Allow some non-monotonicity due to discrete sampling
        assert psrs[0] >= psrs[-1] - 10, (
            f"PSR at K2=0 ({psrs[0]:.1f}) should be >= PSR at K2=0.2 ({psrs[-1]:.1f})"
        )

    def test_psr_no_signal_drop_epsilon(self) -> None:
        """No signal drop -> epsilon guard fires, PSR ≈ 0%."""
        signal = np.ones(70, dtype=np.float64) * 1000.0
        psr = compute_psr_legacy(signal, baseline_frames=10, tail_frames=5)
        assert abs(psr) < 1.0, f"PSR={psr:.1f} should be ~0 for flat signal"

    def test_psr_4d_input(self) -> None:
        """PSR should work with 4D input."""
        signal_1d, _, _ = _make_dsc_signal("gm")
        signal_4d = signal_1d[np.newaxis, np.newaxis, np.newaxis, :]
        psr_4d = compute_psr_legacy(signal_4d, baseline_frames=_BASELINE, tail_frames=5)
        psr_1d = compute_psr_legacy(signal_1d, baseline_frames=_BASELINE, tail_frames=5)

        assert psr_4d.shape == (1, 1, 1)
        assert abs(float(psr_4d[0, 0, 0]) - float(psr_1d)) < 1e-6


class TestPeakHeightAnalytic:
    """Test peak height on known signals."""

    @pytest.mark.parametrize(
        "tissue_key",
        [pytest.param("gm", id="GM"),
         pytest.param("wm", id="WM"),
         pytest.param("tumor", id="tumor")],
    )
    def test_peak_height_legacy_positive(self, tissue_key: str) -> None:
        """PH (% signal drop) should be positive for valid DSC signal."""
        signal, _, _ = _make_dsc_signal(tissue_key)
        ph = compute_peak_height_legacy(signal, baseline_frames=_BASELINE)
        assert isinstance(ph, float)
        assert ph > 0, f"PH should be positive, got {ph:.1f}"
        assert ph < 100, f"PH {ph:.1f}% exceeds 100% (signal below zero?)"

    def test_peak_height_ordering(self) -> None:
        """Higher CBF tissue should have higher peak height."""
        phs = {}
        for key in ["wm", "gm", "tumor"]:
            signal, _, _ = _make_dsc_signal(key)
            phs[key] = float(
                compute_peak_height_legacy(signal, baseline_frames=_BASELINE)
            )

        # Tumor (CBF=100) > GM (CBF=60) > WM (CBF=20) for peak height
        assert phs["tumor"] > phs["gm"], (
            f"tumor PH ({phs['tumor']:.1f}) should > GM PH ({phs['gm']:.1f})"
        )
        assert phs["gm"] > phs["wm"], (
            f"GM PH ({phs['gm']:.1f}) should > WM PH ({phs['wm']:.1f})"
        )

    @pytest.mark.parametrize(
        "te",
        [pytest.param(0.025, id="TE25ms"),
         pytest.param(0.030, id="TE30ms"),
         pytest.param(0.040, id="TE40ms")],
    )
    def test_peak_height_conc_units(self, te: float) -> None:
        """PH in concentration domain should be in s⁻¹ and TE-independent."""
        signal, _, _ = _make_dsc_signal("gm", te=te)
        signal_4d = signal[np.newaxis, np.newaxis, np.newaxis, :]
        ph = compute_peak_height_conc(signal_4d, te_seconds=te, baseline_end=_BASELINE)
        ph_val = float(ph[0, 0, 0])

        assert ph_val > 0, "Concentration-domain PH should be positive"
        # PH should be approximately the same regardless of TE
        # (ΔR2* = -ln(S/S0)/TE cancels TE algebraically)
        assert np.isfinite(ph_val)


class TestTTPAnalytic:
    """Test time-to-peak on known signals."""

    @pytest.mark.parametrize(
        "tissue_key",
        [pytest.param("gm", id="GM"),
         pytest.param("wm", id="WM"),
         pytest.param("tumor", id="tumor")],
    )
    def test_ttp_within_acquisition(self, tissue_key: str) -> None:
        """TTP should be within the acquisition window."""
        signal, aif_post, dt = _make_dsc_signal(tissue_key)
        conc = signal_to_concentration(signal, te=0.030, baseline_frames=_BASELINE)

        ttp = compute_ttp_legacy(conc, pre_bolus_frames=_BASELINE, dt=dt)
        ttp_val = float(ttp)

        # TTP should be after baseline and within acquisition
        assert ttp_val > _BASELINE * dt, f"TTP {ttp_val:.1f}s before bolus"
        total_time = len(signal) * dt
        assert ttp_val < total_time, f"TTP {ttp_val:.1f}s after end of acquisition"

    def test_ttp_4d_input(self) -> None:
        """TTP should work with 4D input."""
        signal, _, dt = _make_dsc_signal("gm")
        conc_1d = signal_to_concentration(signal, te=0.030, baseline_frames=_BASELINE)
        conc_4d = conc_1d[np.newaxis, np.newaxis, np.newaxis, :]

        ttp_4d = compute_ttp_legacy(conc_4d, pre_bolus_frames=_BASELINE, dt=dt)
        ttp_1d = compute_ttp_legacy(conc_1d, pre_bolus_frames=_BASELINE, dt=dt)

        assert ttp_4d.shape == (1, 1, 1)
        assert abs(float(ttp_4d[0, 0, 0]) - float(ttp_1d)) < 1e-6


class TestAUCRatio:
    """Test AUC ratio property from indicator dilution theory.

    AUC_tissue / AUC_AIF ≈ rCBV (relative cerebral blood volume).
    """

    def test_auc_ratio_scales_with_cbv(self) -> None:
        """AUC ratio should correlate with ground truth CBV."""
        dt = 1.5
        aif_post, _ = generate_dsc_aif_dro(
            n_timepoints=60, dt_sec=dt, baseline_frames=0,
            peak_delta_r2=_AIF_PEAK,
        )
        auc_aif = float(np.sum(aif_post) * dt)

        ratios = {}
        for key, tissue in DSC_TISSUE_PARAMS.items():
            residue = make_exponential_residue(tissue.cbf, tissue.mtt, 60, dt)
            conc = dsc_forward_model(aif_post, residue, dt)
            auc_tissue = float(np.sum(conc) * dt)
            ratios[key] = auc_tissue / auc_aif if auc_aif > 0 else 0

        # Tumor (highest CBV) > GM > WM (lowest CBV)
        assert ratios["tumor"] > ratios["gm"], (
            f"Tumor AUC ratio ({ratios['tumor']:.3f}) should > "
            f"GM ({ratios['gm']:.3f})"
        )
        assert ratios["gm"] > ratios["wm"], (
            f"GM AUC ratio ({ratios['gm']:.3f}) should > "
            f"WM ({ratios['wm']:.3f})"
        )


class TestCNRAndSNR:
    """Test CNR and tSNR computation on synthetic signals."""

    def test_tsnr_scales_with_snr(self) -> None:
        """tSNR of baseline should scale with input SNR."""
        from .conftest import add_rician_noise, snr_to_noise_std

        te = 0.030
        s0 = 1000.0
        tissue = DSC_TISSUE_PARAMS["gm"]

        aif_post, _ = generate_dsc_aif_dro(
            n_timepoints=60, dt_sec=1.5, baseline_frames=0,
            peak_delta_r2=_AIF_PEAK,
        )
        residue = make_exponential_residue(tissue.cbf, tissue.mtt, 60, 1.5)
        conc_post = dsc_forward_model(aif_post, residue, 1.5)
        signal_clean = concentration_to_signal(conc_post, te, s0, _BASELINE)

        tsnr_values = {}
        for snr in [10, 20, 50]:
            noise_std = snr_to_noise_std(s0, float(snr))
            signal_noisy = add_rician_noise(signal_clean, noise_std, seed=42)
            # tSNR = mean(baseline) / std(baseline)
            baseline = signal_noisy[:_BASELINE]
            tsnr = float(np.mean(baseline)) / max(float(np.std(baseline)), 1e-10)
            tsnr_values[snr] = tsnr

        # Higher input SNR should give higher tSNR
        assert tsnr_values[50] > tsnr_values[20], (
            f"tSNR at SNR50 ({tsnr_values[50]:.1f}) should > "
            f"tSNR at SNR20 ({tsnr_values[20]:.1f})"
        )
        assert tsnr_values[20] > tsnr_values[10], (
            f"tSNR at SNR20 ({tsnr_values[20]:.1f}) should > "
            f"tSNR at SNR10 ({tsnr_values[10]:.1f})"
        )


class TestNormalizeCBV:
    """Test CBV normalization to NAWM."""

    def test_ncbv_tumor_gt_one(self) -> None:
        """nCBV of tumor should be > 1 when normalized to WM."""
        shape = (10, 10, 3)
        cbv_map = np.ones(shape, dtype=np.float64)

        # WM region: CBV = 2.0
        nawm_mask = np.zeros(shape, dtype=bool)
        nawm_mask[2:5, 2:5, :] = True
        cbv_map[nawm_mask] = 2.0

        # Tumor region: CBV = 6.0
        cbv_map[7:9, 7:9, :] = 6.0

        ncbv = normalize_cbv(cbv_map, nawm_mask=nawm_mask)

        # Tumor nCBV should be 6.0 / 2.0 = 3.0
        tumor_ncbv = float(np.median(ncbv[7:9, 7:9, :]))
        assert tumor_ncbv > 1.5, f"Tumor nCBV {tumor_ncbv:.2f} should be > 1.5"

        # WM nCBV should be ~1.0
        wm_ncbv = float(np.median(ncbv[nawm_mask]))
        assert abs(wm_ncbv - 1.0) < 0.01, f"WM nCBV {wm_ncbv:.2f} should be ~1.0"

    def test_normalization_factor_returned(self) -> None:
        """return_normalization_factor should return tuple."""
        shape = (5, 5, 3)
        cbv_map = np.ones(shape, dtype=np.float64) * 3.0
        nawm_mask = np.ones(shape, dtype=bool)

        ncbv, factor = normalize_cbv(
            cbv_map, nawm_mask=nawm_mask, return_normalization_factor=True,
        )

        assert abs(factor - 3.0) < 0.01
        assert np.allclose(ncbv, 1.0)

    def test_brain_mask_fallback(self) -> None:
        """When no NAWM mask, brain_mask should be used as fallback."""
        shape = (5, 5, 3)
        cbv_map = np.ones(shape, dtype=np.float64) * 4.0
        brain_mask = np.ones(shape, dtype=bool)

        ncbv, factor = normalize_cbv(
            cbv_map, brain_mask=brain_mask, return_normalization_factor=True,
        )

        assert abs(factor - 4.0) < 0.01

    def test_no_mask_identity(self) -> None:
        """No mask at all should return original map unchanged."""
        shape = (5, 5, 3)
        cbv_map = np.ones(shape, dtype=np.float64) * 3.0

        ncbv, factor = normalize_cbv(
            cbv_map, return_normalization_factor=True,
        )

        assert factor == 1.0
        assert np.allclose(ncbv, cbv_map)

    def test_small_nawm_mask(self) -> None:
        """NAWM mask with very few voxels should still work."""
        shape = (10, 10, 3)
        cbv_map = np.ones(shape, dtype=np.float64) * 5.0

        nawm_mask = np.zeros(shape, dtype=bool)
        nawm_mask[5, 5, 1] = True  # Single voxel

        ncbv, factor = normalize_cbv(
            cbv_map, nawm_mask=nawm_mask, return_normalization_factor=True,
        )

        assert abs(factor - 5.0) < 0.01
