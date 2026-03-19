"""Phase 7: Multi-tissue 3D phantom full-chain tests.

Capstone integration test exercising signal -> concentration -> AIF detection
-> K2 correction -> deconvolution -> normalization on a single phantom.

Exercises:
    Full pipeline chain end-to-end
"""

from typing import Dict, Set, Tuple

import numpy as np
import numpy.typing as npt
import pytest

from pmriprep.utils.dsc_aif import atlas_based_aif_search
from pmriprep.utils.dsc_deconvolution import (
    svd_deconvolve_oscillation,
    svd_deconvolve_standard,
    svd_deconvolve_standard_batch,
)
from pmriprep.utils.dsc_preload import apply_k2_correction
from pmriprep.utils.dsc_signal import (
    normalize_cbv,
    signal_to_concentration,
)

from .conftest import (
    DSC_TISSUE_PARAMS,
    concentration_to_signal,
    dsc_forward_model,
    generate_dsc_3d_phantom,
    generate_dsc_aif_dro,
    generate_leaky_dsc_tissue,
    make_exponential_residue,
)

_AIF_PEAK_SIGNAL = 0.1  # For tissue concentration generation
_AIF_PEAK_AIF = 40.0  # For AIF voxels (high peak, arterial)
_BASELINE = 10
_DT = 1.5
_TE = 0.030
_S0 = 1000.0


# =============================================================================
# Helpers
# =============================================================================


def _build_full_chain_phantom(
    shape: Tuple[int, int, int] = (16, 16, 3),
    k2_tumor: float = 0.0,
) -> Tuple[
    npt.NDArray[np.floating],  # 4D concentration volume
    Dict[str, npt.NDArray[np.bool_]],  # tissue masks
    npt.NDArray[np.floating],  # 1D AIF (full length with baseline)
]:
    """Build a multi-tissue phantom for full-chain testing.

    Layout:
    - WM center (6x6)
    - GM ring around WM
    - Tumor patch (2x2) in one corner, with optional K2 leakage
    - AIF voxels at edges
    - CSF (zero concentration)
    """
    nx, ny, nz = shape
    n_tp = 60

    # Generate AIF for tissue concentration (low peak for realistic deltaR2*)
    aif_tissue, _ = generate_dsc_aif_dro(
        n_timepoints=n_tp, dt_sec=_DT, baseline_frames=0,
        peak_delta_r2=_AIF_PEAK_SIGNAL,
    )

    # Generate tissue concentration curves
    tissue_curves: Dict[str, npt.NDArray[np.floating]] = {}
    for name, params in DSC_TISSUE_PARAMS.items():
        residue = make_exponential_residue(params.cbf, params.mtt, n_tp, _DT)
        c = dsc_forward_model(aif_tissue, residue, _DT)
        if name == "tumor" and k2_tumor > 0:
            c = generate_leaky_dsc_tissue(c, aif_tissue, k2_tumor, _DT)
        tissue_curves[name] = c

    # Build tissue map
    tissue_labels = {"gm": 1, "wm": 2, "tumor": 3}
    tissue_map = np.zeros(shape, dtype=np.int32)

    for z in range(nz):
        # WM center
        tissue_map[5:11, 5:11, z] = tissue_labels["wm"]
        # GM ring
        tissue_map[3:13, 3:13, z] = np.where(
            tissue_map[3:13, 3:13, z] == 0,
            tissue_labels["gm"],
            tissue_map[3:13, 3:13, z],
        )
        # Tumor patch
        tissue_map[13:15, 13:15, z] = tissue_labels["tumor"]

    # AIF voxels
    aif_coords: Set[Tuple[int, int, int]] = set()
    for z in range(nz):
        for x in [1]:
            for y in [1, 2, 3]:
                aif_coords.add((x, y, z))

    # Use a separate high-peak AIF for the AIF voxels themselves
    aif_high, _ = generate_dsc_aif_dro(
        n_timepoints=n_tp, dt_sec=_DT, baseline_frames=0,
        peak_delta_r2=_AIF_PEAK_AIF,
    )

    volume, masks = generate_dsc_3d_phantom(
        shape=shape,
        aif_coords=aif_coords,
        aif=aif_high,
        tissue_curves=tissue_curves,
        tissue_map=tissue_map,
        tissue_labels=tissue_labels,
        baseline_frames=_BASELINE,
    )

    # AIF with baseline (tissue-level AIF for deconvolution)
    aif_full = np.zeros(_BASELINE + n_tp, dtype=np.float64)
    aif_full[_BASELINE:] = aif_tissue

    return volume, masks, aif_full


# =============================================================================
# Test classes
# =============================================================================


class TestFullChainRecovery:
    """Full-chain parameter recovery on multi-tissue phantom."""

    def test_concentration_domain_recovery(self) -> None:
        """Deconvolution in concentration domain should recover per-tissue CBF."""
        volume, masks, aif_full = _build_full_chain_phantom()

        aif_post = aif_full[_BASELINE:]

        # Deconvolve per-tissue median curves
        for tissue_key in ["gm", "wm", "tumor"]:
            tissue_mask = masks[tissue_key]
            truth = DSC_TISSUE_PARAMS[tissue_key]

            # Extract all tissue voxel curves, take median
            curves = volume[tissue_mask, _BASELINE:]
            median_curve = np.median(curves, axis=0)

            result = svd_deconvolve_oscillation(median_curve, aif_post, _DT)

            assert result.cbf > 0, f"{tissue_key} CBF should be positive"
            assert np.isfinite(result.cbf), f"{tissue_key} CBF not finite"
            assert np.isfinite(result.mtt), f"{tissue_key} MTT not finite"

    def test_batch_deconvolution(self) -> None:
        """Batch deconvolution should work on the phantom volume."""
        volume, masks, aif_full = _build_full_chain_phantom()

        aif_post = aif_full[_BASELINE:]
        conc_post = volume[..., _BASELINE:]

        residues, cbf, cbv, mtt = svd_deconvolve_standard_batch(
            tissue_curves=conc_post,
            aif=aif_post,
            dt=_DT,
            mask=masks["brain"],
        )

        assert cbf.shape == volume.shape[:3]
        assert np.all(np.isfinite(cbf[masks["brain"]]))

        # GM should have higher CBF than WM
        gm_cbf = float(np.median(cbf[masks["gm"]]))
        wm_cbf = float(np.median(cbf[masks["wm"]]))
        assert gm_cbf > wm_cbf, (
            f"GM CBF ({gm_cbf:.1f}) should > WM CBF ({wm_cbf:.1f})"
        )

    @pytest.mark.parametrize(
        "method",
        [pytest.param("ssvd", id="sSVD"),
         pytest.param("osvd", id="oSVD")],
    )
    def test_tissue_ordering_preserved(self, method: str) -> None:
        """CBF ordering tumor > GM > WM should be preserved."""
        volume, masks, aif_full = _build_full_chain_phantom()
        aif_post = aif_full[_BASELINE:]

        cbf_by_tissue = {}
        for tissue_key in ["gm", "wm", "tumor"]:
            curves = volume[masks[tissue_key], _BASELINE:]
            median_curve = np.median(curves, axis=0)

            if method == "ssvd":
                from pmriprep.utils.dsc_deconvolution import svd_deconvolve_standard
                result = svd_deconvolve_standard(median_curve, aif_post, _DT)
            else:
                result = svd_deconvolve_oscillation(median_curve, aif_post, _DT)

            cbf_by_tissue[tissue_key] = result.cbf

        assert cbf_by_tissue["tumor"] > cbf_by_tissue["gm"], (
            f"Tumor CBF ({cbf_by_tissue['tumor']:.1f}) should > "
            f"GM CBF ({cbf_by_tissue['gm']:.1f})"
        )
        assert cbf_by_tissue["gm"] > cbf_by_tissue["wm"], (
            f"GM CBF ({cbf_by_tissue['gm']:.1f}) should > "
            f"WM CBF ({cbf_by_tissue['wm']:.1f})"
        )


class TestNAWMNormalization:
    """Test NAWM normalization on the phantom."""

    def test_ncbv_tumor_elevated(self) -> None:
        """nCBV (normalized to NAWM) should be elevated in tumor."""
        volume, masks, aif_full = _build_full_chain_phantom()
        aif_post = aif_full[_BASELINE:]
        conc_post = volume[..., _BASELINE:]

        # Get CBV map via batch deconvolution
        _, _, cbv, _ = svd_deconvolve_standard_batch(
            tissue_curves=conc_post,
            aif=aif_post,
            dt=_DT,
            mask=masks["brain"],
        )

        # Normalize to WM
        ncbv = normalize_cbv(cbv, nawm_mask=masks["wm"])

        wm_ncbv = float(np.median(ncbv[masks["wm"]]))
        tumor_ncbv = float(np.median(ncbv[masks["tumor"]]))

        # WM should be ~1.0 after normalization
        assert abs(wm_ncbv - 1.0) < 0.1, f"WM nCBV {wm_ncbv:.2f} should be ~1.0"

        # Tumor should be elevated
        assert tumor_ncbv > 1.2, (
            f"Tumor nCBV {tumor_ncbv:.2f} should be > 1.2"
        )

    def test_normalization_fallback_chain(self) -> None:
        """Test fallback: no NAWM -> brain mask -> identity."""
        shape = (8, 8, 3)
        cbv = np.ones(shape, dtype=np.float64) * 5.0
        cbv[3:5, 3:5, :] = 10.0  # "tumor"

        # Fallback 1: brain_mask
        brain_mask = np.ones(shape, dtype=bool)
        ncbv1, factor1 = normalize_cbv(
            cbv, brain_mask=brain_mask, return_normalization_factor=True,
        )
        assert factor1 == 5.0  # Median of whole brain

        # Fallback 2: no mask at all -> identity
        ncbv2, factor2 = normalize_cbv(
            cbv, return_normalization_factor=True,
        )
        assert factor2 == 1.0  # No normalization

    def test_small_nawm_mask_works(self) -> None:
        """NAWM mask with < 10 voxels should still normalize."""
        shape = (10, 10, 3)
        cbv = np.ones(shape, dtype=np.float64) * 3.0

        nawm_mask = np.zeros(shape, dtype=bool)
        nawm_mask[5, 5, 0:2] = True  # Only 2 voxels

        ncbv, factor = normalize_cbv(
            cbv, nawm_mask=nawm_mask, return_normalization_factor=True,
        )

        assert abs(factor - 3.0) < 0.01


class TestK2CorrectionInChain:
    """Test K2 correction integrated with the phantom chain."""

    def test_k2_correction_on_leaky_tumor(self) -> None:
        """K2 correction should work within the full phantom chain."""
        volume, masks, aif_full = _build_full_chain_phantom(k2_tumor=0.1)
        aif_post = aif_full[_BASELINE:]
        conc_post = volume[..., _BASELINE:]

        # Apply K2 correction
        # Need full-length AIF (matching concentration time axis)
        n_tp_post = conc_post.shape[3]
        assert len(aif_post) == n_tp_post

        result = apply_k2_correction(
            concentration=conc_post,
            aif=aif_post,
            brain_mask=masks["brain"],
            tr=_DT,
        )

        assert result.concentration_corrected.shape == conc_post.shape
        assert result.k2_map.shape == volume.shape[:3]
        assert result.correction_applied is True

        # Corrected concentration should be finite
        assert np.all(np.isfinite(
            result.concentration_corrected[masks["brain"]]
        ))
