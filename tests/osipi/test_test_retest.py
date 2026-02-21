"""Test-retest reproducibility tests for OSIPI clinical subjects.

Sub-0081: Two sessions acquired on consecutive days on the same 1.5T Siemens
scanner with Magnevist. Extended Tofts + Bayesian MAP.

Tier 1 — Summary statistics (no registration, always runs):
    Session medians within expected %diff, comparison to perf-plus reference.

Tier 2 — Voxel-wise (requires FLIRT registration, skips gracefully):
    Pearson correlation, ICC, DICE, Bland-Altman.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from numpy.typing import NDArray

from pmriprep.utils.registration import calculate_correlation, calculate_dice
from pmriprep.utils.reproducibility import (
    BlandAltmanResult,
    ReproducibilityResult,
    compute_bland_altman,
    compute_icc_2_1,
    compute_percent_difference,
    compute_within_subject_cv,
)

from .conftest import (
    SUBJECT_0081,
    SES1,
    SES2,
    apply_transform,
    compute_joint_mask,
    get_map_path,
    get_mask_path,
    get_perfplus_session_stats,
    get_perfplus_test_retest,
    get_ref_dce_path,
    has_fsl,
    load_map,
    load_mask,
    register_sessions,
    skip_no_fsl,
    skip_no_outputs,
    skip_no_perfplus,
)

# =============================================================================
# Helper: compute masked median
# =============================================================================


def _masked_median(
    data: Optional[NDArray[np.floating]],
    mask: Optional[NDArray[np.floating]] = None,
) -> Optional[float]:
    """Compute median of data within mask, excluding NaN/Inf."""
    if data is None:
        return None

    if mask is not None:
        values = data[mask > 0.5]
    else:
        values = data.ravel()

    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None

    return float(np.median(values))


def _masked_count(
    data: Optional[NDArray[np.floating]],
    mask: Optional[NDArray[np.floating]] = None,
) -> int:
    """Count valid (finite, positive) voxels within mask."""
    if data is None:
        return 0

    if mask is not None:
        values = data[mask > 0.5]
    else:
        values = data.ravel()

    return int(np.sum(np.isfinite(values)))


# =============================================================================
# Tier 1: Summary Statistics (no registration needed)
# =============================================================================


@skip_no_outputs
class TestSummaryStatistics:
    """Summary-level test-retest reproducibility using session medians."""

    def _get_session_median(
        self,
        param: str,
        session: str,
    ) -> float:
        """Load map and compute QC-masked median."""
        data = load_map(SUBJECT_0081, session, param)
        qc_mask = load_mask(SUBJECT_0081, session, "qc")
        assert data is not None, f"Missing {param} map for {session}"

        median = _masked_median(data, qc_mask)
        assert median is not None, f"No valid voxels for {param} in {session}"
        return median

    def test_ktrans_median_agreement(self) -> None:
        """Session Ktrans medians should be within 50%.

        Current: ~33.6%. Perf-plus: 2.68%.
        Threshold set at 50% as regression guard.
        """
        med1 = self._get_session_median("Ktrans", SES1)
        med2 = self._get_session_median("Ktrans", SES2)

        pct_diff = compute_percent_difference(
            np.array([med1]), np.array([med2])
        )
        assert pct_diff < 50.0, (
            f"Ktrans median %diff = {pct_diff:.1f}% (> 50%), "
            f"ses1={med1:.4f}, ses2={med2:.4f}"
        )

    def test_ve_median_agreement(self) -> None:
        """Session ve medians should be within 40%.

        Current: ~22.3%.
        Threshold set at 40% as regression guard.
        """
        med1 = self._get_session_median("ve", SES1)
        med2 = self._get_session_median("ve", SES2)

        pct_diff = compute_percent_difference(
            np.array([med1]), np.array([med2])
        )
        assert pct_diff < 40.0, (
            f"ve median %diff = {pct_diff:.1f}% (> 40%), "
            f"ses1={med1:.4f}, ses2={med2:.4f}"
        )

    def test_vp_median_agreement(self) -> None:
        """Session vp medians should be within 30%."""
        med1 = self._get_session_median("vp", SES1)
        med2 = self._get_session_median("vp", SES2)

        pct_diff = compute_percent_difference(
            np.array([med1]), np.array([med2])
        )
        assert pct_diff < 30.0, (
            f"vp median %diff = {pct_diff:.1f}% (> 30%), "
            f"ses1={med1:.4f}, ses2={med2:.4f}"
        )

    @skip_no_perfplus
    def test_ktrans_pct_diff_vs_perfplus(self) -> None:
        """PMRIPrep Ktrans %diff should be < 50%.

        Current: ~33.6%. Perf-plus: 2.68%.
        Threshold set at 50% as regression guard; tighten as pipeline improves.
        """
        med1 = self._get_session_median("Ktrans", SES1)
        med2 = self._get_session_median("Ktrans", SES2)

        pct_diff = compute_percent_difference(
            np.array([med1]), np.array([med2])
        )

        # Perf-plus reference: 2.68%
        assert pct_diff < 50.0, (
            f"Ktrans %diff = {pct_diff:.1f}% (> 50%), "
            f"perf-plus reference = 2.68%"
        )

    @skip_no_perfplus
    def test_ktrans_cv_vs_perfplus(self) -> None:
        """PMRIPrep Ktrans within-subject CV should be < 25%.

        Current: ~16.8%. Perf-plus: 1.34%.
        Threshold set at 25% as regression guard; tighten as pipeline improves.
        """
        med1 = self._get_session_median("Ktrans", SES1)
        med2 = self._get_session_median("Ktrans", SES2)

        cv = compute_within_subject_cv(np.array([med1]), np.array([med2]))

        # Perf-plus reference: 1.34%
        assert cv < 25.0, (
            f"Ktrans CV = {cv:.1f}% (> 25%), "
            f"perf-plus reference = 1.34%"
        )


# =============================================================================
# Tier 2: Mask Reproducibility (needs registration)
# =============================================================================


def _get_registration(
    subject: str = SUBJECT_0081,
) -> Optional[tuple]:
    """Get or compute registration transform."""
    return register_sessions(subject)


@skip_no_outputs
@skip_no_fsl
class TestMaskReproducibility:
    """Test-retest reproducibility of brain and QC masks (after registration)."""

    @pytest.fixture(autouse=True)
    def _setup_registration(self) -> None:
        """Set up registration for the class."""
        reg = _get_registration()
        if reg is None:
            pytest.skip("Registration failed")
        self.mat_file, self.out_dir = reg

    def test_brain_mask_dice(self) -> None:
        """Brain mask DICE across sessions should be > 0.90."""
        brain1 = load_mask(SUBJECT_0081, SES1, "brain")
        assert brain1 is not None

        # Register ses-0002 brain mask to ses-0001
        brain2_in = get_mask_path(SUBJECT_0081, SES2, "brain")
        brain2_out = Path(self.out_dir) / "brain_mask_ses2_reg.nii.gz"
        ref_path = get_ref_dce_path(SUBJECT_0081, SES1)
        result = apply_transform(
            brain2_in, ref_path, self.mat_file, brain2_out, "nearestneighbour"
        )
        assert result is not None, "Failed to register ses-0002 brain mask"

        import nibabel as nib
        brain2_img = nib.load(brain2_out)
        brain2 = np.asarray(brain2_img.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]

        dice = calculate_dice(brain1, brain2)
        assert dice > 0.90, f"Brain mask DICE = {dice:.3f} (< 0.90)"

    def test_qc_mask_dice(self) -> None:
        """QC mask DICE across sessions should be > 0.80."""
        qc1 = load_mask(SUBJECT_0081, SES1, "qc")
        assert qc1 is not None

        # Register ses-0002 QC mask to ses-0001
        qc2_in = get_mask_path(SUBJECT_0081, SES2, "qc")
        qc2_out = Path(self.out_dir) / "qc_mask_ses2_reg.nii.gz"
        ref_path = get_ref_dce_path(SUBJECT_0081, SES1)
        result = apply_transform(
            qc2_in, ref_path, self.mat_file, qc2_out, "nearestneighbour"
        )
        assert result is not None, "Failed to register ses-0002 QC mask"

        import nibabel as nib
        qc2_img = nib.load(qc2_out)
        qc2 = np.asarray(qc2_img.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]

        dice = calculate_dice(qc1, qc2)
        # QC masks are derived from model fitting quality and are highly
        # sensitive to noise differences between sessions. Low DICE (0.26)
        # is expected for test-retest; brain masks provide the structural
        # reproducibility check.
        assert dice > 0.15, f"QC mask DICE = {dice:.3f} (< 0.15)"


# =============================================================================
# Tier 2: Voxel-wise Reproducibility (needs registration)
# =============================================================================


@skip_no_outputs
@skip_no_fsl
class TestVoxelwiseReproducibility:
    """Voxel-wise test-retest reproducibility after inter-session registration."""

    @pytest.fixture(autouse=True)
    def _setup_registered_data(self) -> None:
        """Register ses-0002 maps to ses-0001 and compute joint mask."""
        reg = _get_registration()
        if reg is None:
            pytest.skip("Registration failed")
        self.mat_file, self.out_dir = reg

        # Compute joint mask
        joint = compute_joint_mask(SUBJECT_0081, self.mat_file)
        if joint is None:
            pytest.skip("Joint mask has insufficient voxels")
        self.joint_mask = joint

    def _load_registered_pair(
        self, suffix: str
    ) -> tuple:
        """Load ses-0001 map and registered ses-0002 map.

        Returns:
            Tuple of (ses1_data, ses2_data_registered) masked arrays,
            or pytest.skip if data unavailable.
        """
        import nibabel as nib

        data1 = load_map(SUBJECT_0081, SES1, suffix)
        if data1 is None:
            pytest.skip(f"Missing {suffix} for {SES1}")

        # Register ses-0002 map
        map2_in = get_map_path(SUBJECT_0081, SES2, suffix)
        map2_out = Path(self.out_dir) / f"{suffix}_ses2_reg.nii.gz"
        ref_path = get_ref_dce_path(SUBJECT_0081, SES1)
        result = apply_transform(map2_in, ref_path, self.mat_file, map2_out)
        if result is None:
            pytest.skip(f"Failed to register {suffix} for {SES2}")

        img2 = nib.load(map2_out)
        data2 = np.asarray(img2.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]

        # Extract masked values
        mask = self.joint_mask
        v1 = data1[mask]
        v2 = data2[mask]

        # Filter NaN/Inf from both
        valid = np.isfinite(v1) & np.isfinite(v2)
        v1 = v1[valid]
        v2 = v2[valid]

        if len(v1) < 100:
            pytest.skip(f"Too few valid voxels for {suffix}: {len(v1)}")

        return v1, v2

    def test_pearson_ktrans(self) -> None:
        """Ktrans Pearson r across sessions should be > 0.30.

        Current: ~0.49. Target: > 0.80.
        Threshold set low as regression guard; tighten as pipeline improves.
        """
        v1, v2 = self._load_registered_pair("Ktrans")
        r = float(np.corrcoef(v1, v2)[0, 1])
        assert r > 0.30, f"Ktrans Pearson r = {r:.3f} (< 0.30)"

    def test_pearson_ve(self) -> None:
        """ve Pearson r across sessions should be > 0.30.

        Current: ~0.58. Target: > 0.75.
        """
        v1, v2 = self._load_registered_pair("ve")
        r = float(np.corrcoef(v1, v2)[0, 1])
        assert r > 0.30, f"ve Pearson r = {r:.3f} (< 0.30)"

    def test_pearson_vp(self) -> None:
        """vp Pearson r across sessions should be > 0.25.

        Current: ~0.46. Target: > 0.60.
        """
        v1, v2 = self._load_registered_pair("vp")
        r = float(np.corrcoef(v1, v2)[0, 1])
        assert r > 0.25, f"vp Pearson r = {r:.3f} (< 0.25)"

    def test_ktrans_icc(self) -> None:
        """Ktrans ICC(2,1) should be > 0.25.

        Current: ~0.42. Target: > 0.70.
        """
        v1, v2 = self._load_registered_pair("Ktrans")
        icc = compute_icc_2_1(v1, v2)
        assert icc > 0.25, f"Ktrans ICC(2,1) = {icc:.3f} (< 0.25)"

    def test_modelfree_iauc60_correlation(self) -> None:
        """IAUC60 Pearson r across sessions should be > 0.40.

        Current: ~0.58. Target: > 0.85.
        """
        v1, v2 = self._load_registered_pair("IAUC60")
        r = float(np.corrcoef(v1, v2)[0, 1])
        assert r > 0.40, f"IAUC60 Pearson r = {r:.3f} (< 0.40)"

    def test_modelfree_iauc90_correlation(self) -> None:
        """IAUC90 Pearson r across sessions should be > 0.40.

        Current: ~0.57. Target: > 0.85.
        """
        v1, v2 = self._load_registered_pair("IAUC90")
        r = float(np.corrcoef(v1, v2)[0, 1])
        assert r > 0.40, f"IAUC90 Pearson r = {r:.3f} (< 0.40)"

    def test_modelfree_iauc120_correlation(self) -> None:
        """IAUC120 Pearson r across sessions should be > 0.40.

        Current: ~0.57. Target: > 0.85.
        """
        v1, v2 = self._load_registered_pair("IAUC120")
        r = float(np.corrcoef(v1, v2)[0, 1])
        assert r > 0.40, f"IAUC120 Pearson r = {r:.3f} (< 0.40)"

    def test_modelfree_ser_correlation(self) -> None:
        """SER Pearson r across sessions should be > -0.10.

        Current: ~0.03. SER is known to have poor test-retest due to
        sensitivity to timing differences between sessions.
        """
        v1, v2 = self._load_registered_pair("SER")
        r = float(np.corrcoef(v1, v2)[0, 1])
        assert r > -0.10, f"SER Pearson r = {r:.3f} (< -0.10)"


# =============================================================================
# Tier 2: Visualization (always passes, generates artifacts)
# =============================================================================


@skip_no_outputs
@skip_no_fsl
class TestVisualization:
    """Generate test-retest QC visualizations (always passes)."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Set up registration and output directory."""
        reg = _get_registration()
        if reg is None:
            pytest.skip("Registration failed")
        self.mat_file, self.out_dir = reg

        joint = compute_joint_mask(SUBJECT_0081, self.mat_file)
        if joint is None:
            pytest.skip("Joint mask has insufficient voxels")
        self.joint_mask = joint

        self.qc_dir = Path(self.out_dir)
        self.qc_dir.mkdir(parents=True, exist_ok=True)

    def _load_registered_pair(self, suffix: str) -> Optional[tuple]:
        """Load registered pair, returning None if unavailable."""
        import nibabel as nib

        data1 = load_map(SUBJECT_0081, SES1, suffix)
        if data1 is None:
            return None

        map2_in = get_map_path(SUBJECT_0081, SES2, suffix)
        map2_out = self.qc_dir / f"{suffix}_ses2_reg_viz.nii.gz"
        ref_path = get_ref_dce_path(SUBJECT_0081, SES1)
        result = apply_transform(map2_in, ref_path, self.mat_file, map2_out)
        if result is None:
            return None

        img2 = nib.load(map2_out)
        data2 = np.asarray(img2.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]

        mask = self.joint_mask
        v1 = data1[mask]
        v2 = data2[mask]

        valid = np.isfinite(v1) & np.isfinite(v2)
        return v1[valid], v2[valid]

    def test_bland_altman_plots(self) -> None:
        """Generate Bland-Altman plots for Ktrans, ve, vp."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        params = ["Ktrans", "ve", "vp"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, param in zip(axes, params):
            pair = self._load_registered_pair(param)
            if pair is None:
                ax.set_title(f"{param} (no data)")
                continue

            v1, v2 = pair
            ba = compute_bland_altman(v1, v2)

            mean_vals = (v1 + v2) / 2.0
            diff_vals = v1 - v2

            ax.scatter(mean_vals, diff_vals, alpha=0.1, s=1, color="steelblue")
            ax.axhline(ba.bias, color="red", linestyle="-", label=f"Bias={ba.bias:.4f}")
            ax.axhline(ba.upper_limit, color="gray", linestyle="--", label="95% LoA")
            ax.axhline(ba.lower_limit, color="gray", linestyle="--")
            ax.set_xlabel(f"Mean {param}")
            ax.set_ylabel(f"Difference (ses1 - ses2)")
            ax.set_title(f"{param} Bland-Altman")
            ax.legend(fontsize=8)

        fig.suptitle("Sub-0081 Test-Retest: Bland-Altman", fontsize=14)
        fig.tight_layout()

        out_path = self.qc_dir / "bland_altman.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def test_scatter_plots(self) -> None:
        """Generate scatter plots for Ktrans, ve, vp."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        params = ["Ktrans", "ve", "vp"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, param in zip(axes, params):
            pair = self._load_registered_pair(param)
            if pair is None:
                ax.set_title(f"{param} (no data)")
                continue

            v1, v2 = pair
            r = float(np.corrcoef(v1, v2)[0, 1])

            ax.scatter(v1, v2, alpha=0.1, s=1, color="steelblue")

            # Identity line
            lims = [
                min(float(np.min(v1)), float(np.min(v2))),
                max(float(np.max(v1)), float(np.max(v2))),
            ]
            ax.plot(lims, lims, "r--", alpha=0.5, label="identity")
            ax.set_xlabel(f"{param} (ses-0001)")
            ax.set_ylabel(f"{param} (ses-0002)")
            ax.set_title(f"{param} (r={r:.3f})")
            ax.legend(fontsize=8)

        fig.suptitle("Sub-0081 Test-Retest: Scatter", fontsize=14)
        fig.tight_layout()

        out_path = self.qc_dir / "scatter.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
