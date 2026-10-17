"""TE cancellation in signal conversion and frame-interval sensitivity."""

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
        assert abs(result.cbf - tissue.cbf) / tissue.cbf < 0.35, (
            f"CBF={result.cbf:.1f} at TE={te*1000:.0f}ms, "
            f"expected ~{tissue.cbf:.1f}"
        )


class TestTRDependence:
    """CBF recovery and residue-MTT consistency across frame intervals."""

    def test_cbf_and_mtt_across_tr(self) -> None:
        tissue = DSC_TISSUE_PARAMS["gm"]
        mtts = []
        for tr in (1.0, 1.5, 2.0, 2.5):
            n_tp = int(90 / tr)
            aif, _ = generate_dsc_aif_dro(
                n_timepoints=n_tp, dt_sec=tr, baseline_frames=0,
                peak_delta_r2=_AIF_PEAK,
            )
            residue = make_exponential_residue(tissue.cbf, tissue.mtt, n_tp, tr)
            conc = dsc_forward_model(aif, residue, tr)
            result = svd_deconvolve_oscillation(conc, aif, tr)
            assert abs(result.cbf - tissue.cbf) / tissue.cbf < 0.45, (
                f"CBF={result.cbf:.1f} at TR={tr}s, expected ~{tissue.cbf:.1f}"
            )
            if tr <= 2.0:
                mtts.append(result.mtt)
        mtt_range = max(mtts) - min(mtts)
        assert mtt_range / np.mean(mtts) < 0.40
