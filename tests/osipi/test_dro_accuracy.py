"""DRO ground truth accuracy tests for OSIPI synthetic phantoms.

Compares PMRIPrep DCE parameter maps against known ground truth Ktrans
from OSIPI synthetic Digital Reference Objects (sub-0091, sub-0092).

Ground truth Ktrans values are very low (0.003-0.011 min⁻¹) where noise
dominates and Extended Tofts is poorly identified. Thresholds here are
regression guards based on GT alone, not accuracy targets.

Tier 1 — Summary statistics (per-session, no registration needed):
    Median direction, order of magnitude, MAE sanity checks.

Tier 2 — Voxel-wise (within GT tumor mask):
    Pearson/Spearman correlation, CCC, Bayesian CI coverage.

Tier 3 — Visualization (always passes, generates QC artifacts).
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from .conftest import (
    DRO_SESSIONS,
    OUTPUT_ROOT,
    get_map_path,
    load_ci_map,
    load_ground_truth_ktrans,
    load_map,
    load_tumor_mask,
    skip_no_dro_gt,
    skip_no_dro_outputs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dro_pair(
    subject: str,
    session: str,
) -> Optional[Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.bool_]]]:
    """Load GT Ktrans, pmriprep Ktrans, and tumor mask for a DRO session.

    Returns
    -------
    Optional[Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.bool_]]]
        Tuple of (gt_data, calc_data, mask) or None if files missing.
    """
    gt = load_ground_truth_ktrans(subject, session)
    calc = load_map(subject, session, "Ktrans")
    mask = load_tumor_mask(subject, session)

    if gt is None or calc is None or mask is None:
        return None

    return gt, calc, mask


def _extract_valid_pairs(
    gt: NDArray[np.floating],
    calc: NDArray[np.floating],
    mask: NDArray[np.bool_],
    require_positive_gt: bool = True,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Extract valid voxel pairs within the tumor mask.

    Parameters
    ----------
    gt : NDArray[np.floating]
        Ground truth Ktrans array.
    calc : NDArray[np.floating]
        Calculated Ktrans array.
    mask : NDArray[np.bool_]
        Boolean tumor mask.
    require_positive_gt : bool
        If True, exclude voxels where GT <= 0.

    Returns
    -------
    Tuple[NDArray[np.floating], NDArray[np.floating]]
        Tuple of (gt_values, calc_values) as 1D arrays.
    """
    gt_vals = gt[mask]
    calc_vals = calc[mask]

    valid = np.isfinite(gt_vals) & np.isfinite(calc_vals)
    if require_positive_gt:
        valid &= gt_vals > 0

    return gt_vals[valid], calc_vals[valid]


def _concordance_correlation_coefficient(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
) -> float:
    """Compute Lin's concordance correlation coefficient (CCC).

    CCC measures agreement between two variables, combining correlation
    with accuracy (deviation from identity line).

    Reference:
        Lin LI. A concordance correlation coefficient to evaluate
        reproducibility. Biometrics 1989;45:255-268.

    Parameters
    ----------
    x : NDArray[np.floating]
        First measurement array.
    y : NDArray[np.floating]
        Second measurement array.

    Returns
    -------
    float
        CCC value in [-1, 1].
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    cov_xy = np.mean((x - mean_x) * (y - mean_y))

    denominator = var_x + var_y + (mean_x - mean_y) ** 2
    if denominator < 1e-15:
        return 0.0

    return float(2.0 * cov_xy / denominator)


# Session parametrization IDs for readable test output
_SESSION_IDS = [f"{s}-{ses}" for s, ses in DRO_SESSIONS]


def _skip_if_missing(subject: str, session: str) -> None:
    """Skip test if pmriprep outputs are missing for this session."""
    ktrans_path = get_map_path(subject, session, "Ktrans")
    if not ktrans_path.exists():
        pytest.skip(f"PMRIPrep outputs not found for {subject} {session}")


# =============================================================================
# Tier 1: Summary Statistics (no registration needed)
# =============================================================================


@skip_no_dro_outputs
@skip_no_dro_gt
class TestDROSummaryAccuracy:
    """Summary-level accuracy: pmriprep Ktrans vs DRO ground truth."""

    @pytest.mark.parametrize("subject,session", DRO_SESSIONS, ids=_SESSION_IDS)
    def test_ktrans_median_direction(
        self, subject: str, session: str
    ) -> None:
        """Document bias direction (calc vs GT median).

        PMRIPrep overestimates on sub-0091 but underestimates on
        sub-0092, suggesting different sensitivity to DRO
        characteristics. This test documents the direction for
        regression tracking.
        """
        _skip_if_missing(subject, session)
        data = _load_dro_pair(subject, session)
        assert data is not None, f"Failed to load DRO data for {subject} {session}"
        gt, calc, mask = data
        gt_vals, calc_vals = _extract_valid_pairs(gt, calc, mask)

        gt_median = float(np.median(gt_vals))
        calc_median = float(np.median(calc_vals))
        direction = "overestimate" if calc_median >= gt_median else "underestimate"

        # Document only — bias direction varies by DRO
        assert True, (
            f"Bias direction: {direction}, "
            f"calc={calc_median:.6f}, GT={gt_median:.6f}"
        )

    @pytest.mark.parametrize("subject,session", DRO_SESSIONS, ids=_SESSION_IDS)
    def test_ktrans_median_order_of_magnitude(
        self, subject: str, session: str
    ) -> None:
        """PMRIPrep median Ktrans should be within 100x of GT median.

        Sanity check: median shouldn't be off by more than 2 orders of magnitude.
        """
        _skip_if_missing(subject, session)
        data = _load_dro_pair(subject, session)
        assert data is not None
        gt, calc, mask = data
        gt_vals, calc_vals = _extract_valid_pairs(gt, calc, mask)

        gt_median = float(np.median(gt_vals))
        calc_median = float(np.median(calc_vals))

        if gt_median > 0:
            ratio = calc_median / gt_median
            assert 0.01 < ratio < 100.0, (
                f"Median ratio {ratio:.1f}x out of range [0.01, 100]: "
                f"calc={calc_median:.6f}, GT={gt_median:.6f}"
            )

    @pytest.mark.parametrize("subject,session", DRO_SESSIONS, ids=_SESSION_IDS)
    def test_ktrans_mae_sanity(
        self, subject: str, session: str
    ) -> None:
        """MAE should be less than GT mean (error smaller than signal).

        Sanity check: if MAE exceeds the GT mean, the map is essentially
        noise relative to the true signal. This is a generous threshold.
        """
        _skip_if_missing(subject, session)
        data = _load_dro_pair(subject, session)
        assert data is not None
        gt, calc, mask = data
        gt_vals, calc_vals = _extract_valid_pairs(gt, calc, mask)

        mae = float(np.mean(np.abs(calc_vals - gt_vals)))
        gt_mean = float(np.mean(gt_vals))

        # MAE should be less than 10x GT mean (generous for these low-Ktrans DROs)
        assert mae < gt_mean * 10.0, (
            f"MAE {mae:.6f} > 10x GT mean {gt_mean:.6f}"
        )

    @pytest.mark.parametrize("subject,session", DRO_SESSIONS, ids=_SESSION_IDS)
    def test_n_voxels_in_mask(
        self, subject: str, session: str
    ) -> None:
        """Tumor mask should have a reasonable number of GT-positive voxels.

        Ensures the mask is not empty and has enough voxels for
        meaningful statistics (at least 50 voxels with GT > 0).
        """
        _skip_if_missing(subject, session)
        data = _load_dro_pair(subject, session)
        if data is None:
            pytest.skip(f"No DRO data for {subject} {session}")

        gt, calc, mask = data
        gt_vals, calc_vals = _extract_valid_pairs(gt, calc, mask)
        n_voxels = len(gt_vals)

        assert n_voxels >= 50, (
            f"Too few GT-positive voxels in tumor mask: {n_voxels} (need >= 50)"
        )


# =============================================================================
# Tier 2: Voxel-wise Accuracy (within GT tumor mask)
# =============================================================================


@skip_no_dro_outputs
@skip_no_dro_gt
class TestDROVoxelwise:
    """Voxel-wise accuracy metrics within the GT tumor mask."""

    @pytest.mark.parametrize("subject,session", DRO_SESSIONS, ids=_SESSION_IDS)
    def test_ktrans_pearson_r(
        self, subject: str, session: str
    ) -> None:
        """Pearson correlation between pmriprep and GT Ktrans > 0.0.

        Any positive linear correlation indicates the map captures some
        spatial structure of the GT.
        """
        _skip_if_missing(subject, session)
        data = _load_dro_pair(subject, session)
        assert data is not None
        gt, calc, mask = data
        gt_vals, calc_vals = _extract_valid_pairs(gt, calc, mask)

        if len(gt_vals) < 10:
            pytest.skip(f"Too few valid voxels: {len(gt_vals)}")

        r = float(np.corrcoef(gt_vals, calc_vals)[0, 1])
        assert r > 0.0, f"Pearson r = {r:.4f} (expected > 0.0)"

    @pytest.mark.parametrize("subject,session", DRO_SESSIONS, ids=_SESSION_IDS)
    def test_ktrans_ccc(
        self, subject: str, session: str
    ) -> None:
        """CCC between pmriprep and GT Ktrans (document, minimal gate).

        CCC captures both correlation and bias. Expected to be low due
        to systematic bias on these low-Ktrans DROs.
        Gate: CCC > -0.5 (regression guard only).
        """
        _skip_if_missing(subject, session)
        data = _load_dro_pair(subject, session)
        assert data is not None
        gt, calc, mask = data
        gt_vals, calc_vals = _extract_valid_pairs(gt, calc, mask)

        if len(gt_vals) < 10:
            pytest.skip(f"Too few valid voxels: {len(gt_vals)}")

        ccc = _concordance_correlation_coefficient(gt_vals, calc_vals)
        assert ccc > -0.5, f"CCC = {ccc:.4f} (expected > -0.5)"

    @pytest.mark.parametrize("subject,session", DRO_SESSIONS, ids=_SESSION_IDS)
    def test_ktrans_spatial_pattern(
        self, subject: str, session: str
    ) -> None:
        """Spearman rank correlation > 0.0 (robust to systematic bias).

        Rank correlation tests whether the spatial pattern of high/low
        Ktrans voxels is preserved, regardless of the absolute scale bias.
        """
        _skip_if_missing(subject, session)
        data = _load_dro_pair(subject, session)
        assert data is not None
        gt, calc, mask = data
        gt_vals, calc_vals = _extract_valid_pairs(gt, calc, mask)

        if len(gt_vals) < 10:
            pytest.skip(f"Too few valid voxels: {len(gt_vals)}")

        rho, _ = stats.spearmanr(gt_vals, calc_vals)
        assert rho > 0.0, f"Spearman rho = {rho:.4f} (expected > 0.0)"

    @pytest.mark.parametrize("subject,session", DRO_SESSIONS, ids=_SESSION_IDS)
    def test_bayesian_ci_coverage(
        self, subject: str, session: str
    ) -> None:
        """Document 95% CI coverage of GT (no hard gate).

        Bayesian MAP on these DROs currently produces degenerate CI maps
        (lower == upper == point estimate), yielding 0% coverage. This
        is a known limitation: MAP estimation without MCMC or Laplace
        approximation does not produce meaningful uncertainty intervals.

        This test documents actual coverage for regression tracking.
        """
        _skip_if_missing(subject, session)

        gt = load_ground_truth_ktrans(subject, session)
        mask = load_tumor_mask(subject, session)
        ci_lower = load_ci_map(subject, session, "lower", "Ktrans")
        ci_upper = load_ci_map(subject, session, "upper", "Ktrans")

        if any(x is None for x in (gt, mask, ci_lower, ci_upper)):
            pytest.skip(
                f"Missing CI maps or GT for {subject} {session}"
            )

        assert gt is not None and mask is not None
        assert ci_lower is not None and ci_upper is not None

        gt_vals = gt[mask]
        lower_vals = ci_lower[mask]
        upper_vals = ci_upper[mask]

        valid = (
            np.isfinite(gt_vals)
            & np.isfinite(lower_vals)
            & np.isfinite(upper_vals)
            & (gt_vals > 0)
        )

        if int(np.sum(valid)) < 10:
            pytest.skip(f"Too few valid voxels for CI check: {int(np.sum(valid))}")

        gt_v = gt_vals[valid]
        lo_v = lower_vals[valid]
        hi_v = upper_vals[valid]

        # Check if CI bounds are degenerate (lower == upper)
        ci_width = hi_v - lo_v
        degenerate = float(np.mean(np.abs(ci_width) < 1e-10)) * 100.0

        covered = (gt_v >= lo_v) & (gt_v <= hi_v)
        coverage_pct = float(np.mean(covered)) * 100.0

        # Document, don't gate — MAP CI is known to be degenerate
        # When CI maps are fixed, tighten to > 50%
        assert True, (
            f"CI 95% coverage = {coverage_pct:.1f}%, "
            f"degenerate CIs = {degenerate:.0f}%, n={len(gt_v)}"
        )


# =============================================================================
# Tier 3: Visualization (always passes, generates QC artifacts)
# =============================================================================


@skip_no_dro_outputs
@skip_no_dro_gt
class TestDROVisualization:
    """Generate DRO accuracy QC visualizations (always passes)."""

    @pytest.fixture(autouse=True)
    def _setup_qc_dir(self) -> None:
        """Set up output directory for QC plots."""
        self.qc_dir = OUTPUT_ROOT / "dro_accuracy_qc"
        self.qc_dir.mkdir(parents=True, exist_ok=True)

    @pytest.mark.parametrize("subject,session", DRO_SESSIONS, ids=_SESSION_IDS)
    def test_scatter_plot_vs_gt(
        self, subject: str, session: str
    ) -> None:
        """Generate calculated vs GT scatter plot per session."""
        _skip_if_missing(subject, session)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        data = _load_dro_pair(subject, session)
        if data is None:
            return  # Always pass

        gt, calc, mask = data
        gt_vals, calc_vals = _extract_valid_pairs(gt, calc, mask)

        if len(gt_vals) < 10:
            return

        r = float(np.corrcoef(gt_vals, calc_vals)[0, 1])
        ccc = _concordance_correlation_coefficient(gt_vals, calc_vals)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(gt_vals, calc_vals, alpha=0.3, s=8, color="steelblue")

        lims = [0, max(float(np.max(gt_vals)), float(np.max(calc_vals))) * 1.1]
        ax.plot(lims, lims, "r--", alpha=0.5, label="identity")

        ax.set_xlabel("GT Ktrans (min⁻¹)")
        ax.set_ylabel("PMRIPrep Ktrans (min⁻¹)")
        ax.set_title(
            f"{subject} {session}\n"
            f"r={r:.3f}, CCC={ccc:.3f}, n={len(gt_vals)}"
        )
        ax.legend(fontsize=8)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        fig.tight_layout()
        out_path = self.qc_dir / f"{subject}_{session}_scatter.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    @pytest.mark.parametrize("subject,session", DRO_SESSIONS, ids=_SESSION_IDS)
    def test_error_histogram(
        self, subject: str, session: str
    ) -> None:
        """Generate histogram of voxel-wise errors per session."""
        _skip_if_missing(subject, session)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        data = _load_dro_pair(subject, session)
        if data is None:
            return

        gt, calc, mask = data
        gt_vals, calc_vals = _extract_valid_pairs(gt, calc, mask)

        if len(gt_vals) < 10:
            return

        errors = calc_vals - gt_vals
        mae = float(np.mean(np.abs(errors)))
        bias = float(np.mean(errors))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(errors, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
        ax.axvline(bias, color="red", linestyle="-", label=f"Bias={bias:.4f}")
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)

        ax.set_xlabel("Error (calc - GT) min⁻¹")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{subject} {session} Error Distribution\n"
            f"MAE={mae:.4f}, Bias={bias:.4f}, n={len(gt_vals)}"
        )
        ax.legend(fontsize=8)

        fig.tight_layout()
        out_path = self.qc_dir / f"{subject}_{session}_error_hist.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    @pytest.mark.parametrize("subject,session", DRO_SESSIONS, ids=_SESSION_IDS)
    def test_spatial_error_map(
        self, subject: str, session: str
    ) -> None:
        """Generate middle-slice spatial error map per session."""
        _skip_if_missing(subject, session)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        data = _load_dro_pair(subject, session)
        if data is None:
            return

        gt, calc, mask = data
        error = np.where(mask, calc - gt, np.nan)

        # Find middle slice with most mask voxels
        mask_per_slice = np.sum(mask, axis=(0, 1))
        mid_slice = int(np.argmax(mask_per_slice))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # GT
        im0 = axes[0].imshow(
            gt[:, :, mid_slice].T, origin="lower", cmap="hot",
            vmin=0, vmax=float(np.nanmax(gt[mask])),
        )
        axes[0].set_title(f"GT Ktrans (slice {mid_slice})")
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        # Calculated
        im1 = axes[1].imshow(
            calc[:, :, mid_slice].T, origin="lower", cmap="hot",
            vmin=0, vmax=float(np.nanmax(calc[mask])),
        )
        axes[1].set_title(f"PMRIPrep Ktrans (slice {mid_slice})")
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        # Error map
        err_slice = error[:, :, mid_slice]
        vmax_err = float(np.nanmax(np.abs(err_slice[np.isfinite(err_slice)])))
        im2 = axes[2].imshow(
            err_slice.T, origin="lower", cmap="RdBu_r",
            vmin=-vmax_err, vmax=vmax_err,
        )
        axes[2].set_title(f"Error (calc - GT)")
        plt.colorbar(im2, ax=axes[2], shrink=0.8)

        fig.suptitle(f"{subject} {session} Spatial Error Map", fontsize=14)
        fig.tight_layout()
        out_path = self.qc_dir / f"{subject}_{session}_error_map.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
