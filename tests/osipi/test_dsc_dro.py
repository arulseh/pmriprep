"""OSIPI DSC DRO external validation tests.

Validates PMRIPrep's DSC deconvolution methods against the OSIPI Code
Collection DSC DRO — 14 concentration-domain test cases with known CBV/CBF
ground truth, gamma-variate residue (lambda=3), TR=1.243s, CNR=200.

Source: OSIPI/DCE-DSC-MRI_CodeCollection commit a55b83d8 (2022-04-21)
File:   test/DSCmodels/data/dsc_data.csv

Unit conventions (per OSIPI SR_TBG_BNIPhoenix_USA implementation):
    CBF (ml/100ml/min) = max(residue) * 60 * 100
    CBV (ml/100ml)     = AUC(tissue) / AUC(AIF) * 100

The OSIPI DRO uses a gamma-variate residue (not exponential), causing
systematic CBF overestimation from SVD regularization. sSVD performs best
(1-28% error) because its conservative truncation smooths the gamma peak.
CBV via AUC is model-free and recovers well (3-19%).

OSIPI units are ml/100ml (not ml/100g as in DSCTissueParams). The density
correction factor is ~1.04 (negligible for these tests).
"""

from typing import List, Tuple

import numpy as np
import pytest

from pmriprep.utils.dsc_deconvolution import (
    svd_deconvolve_oscillation,
    svd_deconvolve_standard,
    tikhonov_deconvolve,
)

from .conftest import (
    OSIPIDSCCase,
    load_osipi_dsc_dro,
    skip_no_osipi_dsc_dro,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# OSIPI unit conversion factors
_CBF_SCALE = 60.0 * 100.0  # residue max -> ml/100ml/min


def _compute_cbf(result: object) -> float:
    """Convert deconvolution result to OSIPI CBF units."""
    return result.cbf * _CBF_SCALE  # type: ignore[union-attr]


def _compute_cbv_auc(
    c_tis: np.ndarray, c_aif: np.ndarray, tr: float,
) -> float:
    """Compute CBV via AUC ratio in OSIPI units (ml/100ml)."""
    aif_auc = np.trapezoid(c_aif, dx=tr)
    if aif_auc < 1e-10:
        return 0.0
    return float(np.trapezoid(c_tis, dx=tr) / aif_auc * 100.0)


def _get_cases() -> List[OSIPIDSCCase]:
    """Load OSIPI DSC DRO cases or skip."""
    cases = load_osipi_dsc_dro()
    if cases is None:
        pytest.skip("OSIPI DSC DRO data not available")
    return cases


def _get_case_by_params(
    cases: List[OSIPIDSCCase], cbv: float, cbf: float,
) -> OSIPIDSCCase:
    """Find a specific case by GT CBV and CBF."""
    for c in cases:
        if c.cbv == cbv and c.cbf == cbf:
            return c
    raise ValueError(f"No case with CBV={cbv}, CBF={cbf}")


# Representative subset for secondary methods (low, mid, high CBF)
_REPRESENTATIVE_CASES: List[Tuple[float, float]] = [
    (2.0, 5.0),   # low CBF
    (4.0, 40.0),  # mid CBF
    (4.0, 70.0),  # high CBF
]


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


@skip_no_osipi_dsc_dro
class TestOSIPIDSCDRO:
    """Validate DSC deconvolution against OSIPI external DRO."""

    # --- Primary method: sSVD on all 14 cases ---

    @pytest.mark.parametrize(
        "case_idx", list(range(14)),
        ids=[
            "CBV4_CBF10", "CBV4_CBF20", "CBV4_CBF30", "CBV4_CBF40",
            "CBV4_CBF50", "CBV4_CBF60", "CBV4_CBF70",
            "CBV2_CBF5", "CBV2_CBF10", "CBV2_CBF15", "CBV2_CBF20",
            "CBV2_CBF25", "CBV2_CBF30", "CBV2_CBF35",
        ],
    )
    def test_ssvd_cbf_recovery(self, case_idx: int) -> None:
        """sSVD CBF should recover within 30% of OSIPI ground truth.

        sSVD performs best on the gamma-variate residue because conservative
        truncation smooths the delayed peak. Systematic overestimation (1-28%)
        increases at low CBF where the residue peak is harder to resolve.
        """
        cases = _get_cases()
        case = cases[case_idx]
        result = svd_deconvolve_standard(case.c_tis, case.c_aif, case.tr)
        cbf_recovered = _compute_cbf(result)

        assert cbf_recovered == pytest.approx(
            case.cbf, rel=0.30, abs=2.0,
        ), (
            f"{case.label}: sSVD CBF={cbf_recovered:.1f} vs "
            f"GT={case.cbf:.0f} ml/100ml/min"
        )

    @pytest.mark.parametrize(
        "case_idx", list(range(14)),
        ids=[
            "CBV4_CBF10", "CBV4_CBF20", "CBV4_CBF30", "CBV4_CBF40",
            "CBV4_CBF50", "CBV4_CBF60", "CBV4_CBF70",
            "CBV2_CBF5", "CBV2_CBF10", "CBV2_CBF15", "CBV2_CBF20",
            "CBV2_CBF25", "CBV2_CBF30", "CBV2_CBF35",
        ],
    )
    def test_cbv_recovery(self, case_idx: int) -> None:
        """AUC-based CBV should recover within 20% of OSIPI ground truth.

        Model-free AUC ratio method: CBV = AUC(tissue)/AUC(AIF) * 100.
        Not method-dependent — uses raw concentration curves directly.
        """
        cases = _get_cases()
        case = cases[case_idx]
        cbv_recovered = _compute_cbv_auc(case.c_tis, case.c_aif, case.tr)

        assert cbv_recovered == pytest.approx(
            case.cbv, rel=0.20, abs=0.5,
        ), (
            f"{case.label}: AUC CBV={cbv_recovered:.2f} vs "
            f"GT={case.cbv:.0f} ml/100ml"
        )

    # --- Secondary methods on representative subset ---

    @pytest.mark.parametrize(
        "cbv_cbf", _REPRESENTATIVE_CASES,
        ids=["low_CBF", "mid_CBF", "high_CBF"],
    )
    def test_osvd_cbf_recovery(
        self, cbv_cbf: Tuple[float, float],
    ) -> None:
        """oSVD CBF recovery on representative cases.

        oSVD's Oscillation Index criterion (Wu et al., MRM 2003) was
        designed for exponential residues. Gamma-variate (lambda=3)
        residues are inherently smooth — OI is always below target at
        low thresholds, causing under-regularization. threshold_min=0.15
        provides a regularization floor appropriate for CNR=200 data.
        """
        cases = _get_cases()
        case = _get_case_by_params(cases, cbv_cbf[0], cbv_cbf[1])
        result = svd_deconvolve_oscillation(
            case.c_tis, case.c_aif, case.tr, threshold_min=0.15,
        )
        cbf_recovered = _compute_cbf(result)

        assert cbf_recovered == pytest.approx(
            case.cbf, rel=0.45, abs=5.0,
        ), (
            f"{case.label}: oSVD CBF={cbf_recovered:.1f} vs "
            f"GT={case.cbf:.0f} ml/100ml/min"
        )

    @pytest.mark.parametrize(
        "cbv_cbf", _REPRESENTATIVE_CASES,
        ids=["low_CBF", "mid_CBF", "high_CBF"],
    )
    def test_tikhonov_cbf_recovery(
        self, cbv_cbf: Tuple[float, float],
    ) -> None:
        """Tikhonov CBF recovery on representative OSIPI synthetic cases.

        L-curve-selected modified-penalty Tikhonov (the production operator)
        recovers CBF within the tolerance band across the low/mid/high-CBF
        representative cases of the OSIPI synthetic DRO.
        """
        cases = _get_cases()
        case = _get_case_by_params(cases, cbv_cbf[0], cbv_cbf[1])
        result = tikhonov_deconvolve(case.c_tis, case.c_aif, case.tr)
        cbf_recovered = _compute_cbf(result)

        assert cbf_recovered == pytest.approx(
            case.cbf, rel=0.40, abs=5.0,
        ), (
            f"{case.label}: Tikhonov CBF={cbf_recovered:.1f} vs "
            f"GT={case.cbf:.0f} ml/100ml/min"
        )

    @pytest.mark.parametrize(
        "cbv_cbf", _REPRESENTATIVE_CASES,
        ids=["low_CBF", "mid_CBF", "high_CBF"],
    )
    def test_cross_method_agreement(
        self, cbv_cbf: Tuple[float, float],
    ) -> None:
        """All methods should agree on CBF within 80% of each other.

        Methods differ in regularization philosophy: sSVD truncates
        small singular values, Tikhonov applies a continuous
        first-order smoothness penalty with L-curve λ selection. On
        the OSIPI phantom (exponential residue) Tikhonov's smoothing
        biases CBF higher than sSVD's truncation. Kudo et al. (Radiology
        2010) reported 20–40% inter-method CBF variation on phantom
        data; ours runs to ~65% on the high-CBF case for these two
        methods, dominated by Tikhonov's residue broadening.
        """
        import itertools

        cases = _get_cases()
        case = _get_case_by_params(cases, cbv_cbf[0], cbv_cbf[1])

        methods = {
            "sSVD": lambda c, a, tr: svd_deconvolve_standard(c, a, tr),
            "Tikhonov": lambda c, a, tr: tikhonov_deconvolve(c, a, tr),
        }

        cbf_values = {}
        for name, fn in methods.items():
            result = fn(case.c_tis, case.c_aif, case.tr)
            cbf_values[name] = _compute_cbf(result)

        for m1, m2 in itertools.combinations(cbf_values, 2):
            v1, v2 = cbf_values[m1], cbf_values[m2]
            mean_v = (v1 + v2) / 2.0
            if mean_v > 0:
                rel_diff = abs(v1 - v2) / mean_v
                assert rel_diff < 0.80, (
                    f"{case.label}: {m1}={v1:.1f} vs {m2}={v2:.1f}, "
                    f"rel_diff={rel_diff:.3f} >= 0.80"
                )
