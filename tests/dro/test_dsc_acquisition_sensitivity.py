"""Phase 6: Acquisition parameter sensitivity tests.

Validates that recovered perfusion parameters behave correctly across
TE, TR, and field strength variations. TE should cancel algebraically
in the deltaR2* formula at high SNR; TR determines dt for deconvolution.

Exercises:
    TE handling in signal_to_concentration(), dt handling in deconvolution,
    constants from dsc_constants.py
"""

import numpy as np
import pytest

from pmriprep.utils.dsc_deconvolution import svd_deconvolve_oscillation
from pmriprep.utils.dsc_signal import signal_to_concentration

from .conftest import (
    DSC_TISSUE_PARAMS,
    concentration_to_signal,
    dsc_forward_model,
    generate_dsc_aif_dro,
    make_exponential_residue,
)

_AIF_PEAK = 0.1
_BASELINE = 10


# =============================================================================
# Test classes
# =============================================================================


class TestTEIndependence:
    """CBF/MTT should be TE-independent at high SNR.

    ΔR2* = -ln(S/S0)/TE cancels TE algebraically, so the same tissue
    produces the same concentration regardless of TE (noiseless case).
    At low SNR, clipping threshold in ΔR2* space is -ln(ε)/TE which
    IS TE-dependent.
    """

    TES = [0.020, 0.025, 0.030, 0.035, 0.040, 0.050]

    def test_concentration_te_independence(self) -> None:
        """Noiseless concentration should be identical across TE values."""
        tissue = DSC_TISSUE_PARAMS["gm"]
        aif_post, _ = generate_dsc_aif_dro(
            n_timepoints=60, dt_sec=1.5, baseline_frames=0,
            peak_delta_r2=_AIF_PEAK,
        )
        residue = make_exponential_residue(tissue.cbf, tissue.mtt, 60, 1.5)
        conc_true = dsc_forward_model(aif_post, residue, 1.5)

        concs = {}
        for te in self.TES:
            signal = concentration_to_signal(conc_true, te, 1000.0, _BASELINE)
            conc = signal_to_concentration(signal, te, baseline_frames=_BASELINE)
            concs[te] = conc[_BASELINE:]

        # All should be identical (noiseless round-trip)
        ref = concs[self.TES[0]]
        for te in self.TES[1:]:
            nrmse = float(np.sqrt(np.mean((concs[te] - ref) ** 2))) / float(np.max(ref))
            assert nrmse < 0.01, (
                f"Concentration differs at TE={te*1000:.0f}ms: NRMSE={nrmse:.4f}"
            )

    @pytest.mark.parametrize(
        "te",
        [pytest.param(t, id=f"TE{int(t*1000)}ms") for t in [0.020, 0.030, 0.040, 0.050]],
    )
    def test_cbf_te_independence_noiseless(self, te: float) -> None:
        """CBF from noiseless signal should be TE-independent."""
        tissue = DSC_TISSUE_PARAMS["gm"]
        aif_post, _ = generate_dsc_aif_dro(
            n_timepoints=60, dt_sec=1.5, baseline_frames=0,
            peak_delta_r2=_AIF_PEAK,
        )
        residue = make_exponential_residue(tissue.cbf, tissue.mtt, 60, 1.5)
        conc_true = dsc_forward_model(aif_post, residue, 1.5)

        signal = concentration_to_signal(conc_true, te, 1000.0, _BASELINE)
        conc = signal_to_concentration(signal, te, baseline_frames=_BASELINE)

        result = svd_deconvolve_oscillation(conc[_BASELINE:], aif_post, 1.5)

        # All TE values should give the same CBF within deconvolution tolerance
        assert abs(result.cbf - tissue.cbf) / tissue.cbf < 0.25, (
            f"CBF={result.cbf:.1f} at TE={te*1000:.0f}ms, "
            f"expected ~{tissue.cbf:.1f}"
        )


class TestTRDependence:
    """Parameters should scale correctly with TR (dt for deconvolution)."""

    @pytest.mark.parametrize(
        "tr",
        [pytest.param(t, id=f"TR{t:.1f}s") for t in [1.0, 1.5, 2.0, 2.5]],
    )
    def test_cbf_with_different_tr(self, tr: float) -> None:
        """CBF should be recoverable across different TR values."""
        tissue = DSC_TISSUE_PARAMS["gm"]
        n_tp = int(90 / tr)  # ~90s total acquisition

        aif_post, _ = generate_dsc_aif_dro(
            n_timepoints=n_tp, dt_sec=tr, baseline_frames=0,
            peak_delta_r2=_AIF_PEAK,
        )
        residue = make_exponential_residue(tissue.cbf, tissue.mtt, n_tp, tr)
        conc = dsc_forward_model(aif_post, residue, tr)

        result = svd_deconvolve_oscillation(conc, aif_post, tr)

        # CBF recovery with wider tolerance for different temporal resolution
        assert abs(result.cbf - tissue.cbf) / tissue.cbf < 0.35, (
            f"CBF={result.cbf:.1f} at TR={tr}s, expected ~{tissue.cbf:.1f}"
        )

    def test_mtt_scales_with_time(self) -> None:
        """MTT in seconds should be consistent across TR values."""
        tissue = DSC_TISSUE_PARAMS["gm"]
        mtts = {}

        for tr in [1.0, 1.5, 2.0]:
            n_tp = int(90 / tr)
            aif_post, _ = generate_dsc_aif_dro(
                n_timepoints=n_tp, dt_sec=tr, baseline_frames=0,
                peak_delta_r2=_AIF_PEAK,
            )
            residue = make_exponential_residue(tissue.cbf, tissue.mtt, n_tp, tr)
            conc = dsc_forward_model(aif_post, residue, tr)
            result = svd_deconvolve_oscillation(conc, aif_post, tr)
            mtts[tr] = result.mtt

        # MTT should be reasonably consistent (in seconds)
        mtt_values = list(mtts.values())
        mtt_range = max(mtt_values) - min(mtt_values)
        mtt_mean = np.mean(mtt_values)
        assert mtt_range / mtt_mean < 0.40, (
            f"MTT range ({mtt_range:.2f}s) too large relative to "
            f"mean ({mtt_mean:.2f}s) across TRs"
        )


class TestFieldStrengthIndependence:
    """Relative CBF/CBV should be invariant to field strength.

    The pipeline does relative quantification only. Field strength affects
    r2* (s⁻¹ per mM), but since both AIF and tissue are affected equally,
    relative parameters should be unchanged.
    """

    def test_relative_cbf_field_independent(self) -> None:
        """Relative CBF (ratio of tissues) should not depend on field strength.

        This test uses concentration-domain data (not signal-domain), since
        the signal conversion itself is field-independent (ΔR2* = -ln(S/S0)/TE
        doesn't use r2*).
        """
        tissue_gm = DSC_TISSUE_PARAMS["gm"]
        tissue_wm = DSC_TISSUE_PARAMS["wm"]

        aif_post, _ = generate_dsc_aif_dro(
            n_timepoints=60, dt_sec=1.5, baseline_frames=0,
            peak_delta_r2=_AIF_PEAK,
        )

        # Compute CBF ratio in concentration domain (field-independent)
        for tissue, name in [(tissue_gm, "GM"), (tissue_wm, "WM")]:
            residue = make_exponential_residue(tissue.cbf, tissue.mtt, 60, 1.5)
            conc = dsc_forward_model(aif_post, residue, 1.5)
            result = svd_deconvolve_oscillation(conc, aif_post, 1.5)
            assert result.cbf > 0, f"{name} CBF should be positive"

        # The ratio GM/WM CBF should be ~3.0 (60/20)
        residue_gm = make_exponential_residue(tissue_gm.cbf, tissue_gm.mtt, 60, 1.5)
        residue_wm = make_exponential_residue(tissue_wm.cbf, tissue_wm.mtt, 60, 1.5)
        conc_gm = dsc_forward_model(aif_post, residue_gm, 1.5)
        conc_wm = dsc_forward_model(aif_post, residue_wm, 1.5)

        r_gm = svd_deconvolve_oscillation(conc_gm, aif_post, 1.5)
        r_wm = svd_deconvolve_oscillation(conc_wm, aif_post, 1.5)

        ratio = r_gm.cbf / r_wm.cbf if r_wm.cbf > 0 else 0
        expected_ratio = tissue_gm.cbf / tissue_wm.cbf

        assert abs(ratio - expected_ratio) / expected_ratio < 0.20, (
            f"GM/WM CBF ratio {ratio:.2f}, expected {expected_ratio:.2f}"
        )

    @pytest.mark.parametrize(
        "te",
        [pytest.param(0.025, id="TE25ms_1.5T"),
         pytest.param(0.030, id="TE30ms_3T"),
         pytest.param(0.040, id="TE40ms_3T")],
    )
    def test_signal_roundtrip_field_independent(self, te: float) -> None:
        """Signal round-trip should be field-independent (TE cancels)."""
        tissue = DSC_TISSUE_PARAMS["gm"]
        aif_post, _ = generate_dsc_aif_dro(
            n_timepoints=60, dt_sec=1.5, baseline_frames=0,
            peak_delta_r2=_AIF_PEAK,
        )
        residue = make_exponential_residue(tissue.cbf, tissue.mtt, 60, 1.5)
        conc_true = dsc_forward_model(aif_post, residue, 1.5)

        # Through signal domain
        signal = concentration_to_signal(conc_true, te, 1000.0, _BASELINE)
        conc_recovered = signal_to_concentration(signal, te, baseline_frames=_BASELINE)

        peak = float(np.max(conc_true))
        rmse = float(np.sqrt(np.mean((conc_recovered[_BASELINE:] - conc_true) ** 2)))
        assert rmse / peak < 0.01, (
            f"Signal round-trip NRMSE {rmse/peak:.4f} at TE={te*1000:.0f}ms"
        )
