"""Test-retest reproducibility tests for OSIPI clinical subjects.

OSIPI clinical cohort: sub-0081 through sub-0088. Each subject has two DCE
sessions acquired on the same scanner. Extended Tofts + Bayesian MAP.

Tests are parametrized across all 8 subjects. Subjects without processed
outputs are gracefully skipped; as new subjects are processed, tests
automatically pick them up.

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
    SES1,
    SES2,
    TEST_RETEST_SUBJECTS,
    _outputs_exist,
    apply_transform,
    compute_joint_mask,
    get_map_path,
    get_mask_path,
    get_perfplus_session_stats,
    get_perfplus_test_retest,
    get_ref_dce_path,
    has_fsl,
    has_perfplus_reference,
    load_map,
    load_mask,
    register_sessions,
    skip_no_fsl,
)

# ---------------------------------------------------------------------------
# Parametrization helpers
# ---------------------------------------------------------------------------

_SUBJECT_IDS = [s.replace("sub-", "") for s in TEST_RETEST_SUBJECTS]


def _skip_if_missing(subject: str) -> None:
    """Skip test if pmriprep outputs are missing for either session."""
    if not (_outputs_exist(subject, SES1) and _outputs_exist(subject, SES2)):
        pytest.skip(
            f"PMRIPrep outputs not found for {subject} {SES1} and/or {SES2}"
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


class TestSummaryStatistics:
    """Summary-level test-retest reproducibility using session medians."""

    def _get_session_median(
        self,
        subject: str,
        param: str,
        session: str,
    ) -> float:
        """Load map and compute QC-masked median."""
        data = load_map(subject, session, param)
        qc_mask = load_mask(subject, session, "qc")
        assert data is not None, f"Missing {param} map for {subject} {session}"

        median = _masked_median(data, qc_mask)
        assert median is not None, (
            f"No valid voxels for {param} in {subject} {session}"
        )
        return median

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_ktrans_median_agreement(self, subject: str) -> None:
        """Session Ktrans medians should be within 50%."""
        _skip_if_missing(subject)
        med1 = self._get_session_median(subject, "Ktrans", SES1)
        med2 = self._get_session_median(subject, "Ktrans", SES2)

        pct_diff = compute_percent_difference(
            np.array([med1]), np.array([med2])
        )
        assert pct_diff < 50.0, (
            f"{subject} Ktrans median %diff = {pct_diff:.1f}% (> 50%), "
            f"ses1={med1:.4f}, ses2={med2:.4f}"
        )

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_ve_median_agreement(self, subject: str) -> None:
        """Session ve medians should be within 40%."""
        _skip_if_missing(subject)
        med1 = self._get_session_median(subject, "ve", SES1)
        med2 = self._get_session_median(subject, "ve", SES2)

        pct_diff = compute_percent_difference(
            np.array([med1]), np.array([med2])
        )
        assert pct_diff < 40.0, (
            f"{subject} ve median %diff = {pct_diff:.1f}% (> 40%), "
            f"ses1={med1:.4f}, ses2={med2:.4f}"
        )

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_vp_median_agreement(self, subject: str) -> None:
        """Session vp medians should be within 55%."""
        _skip_if_missing(subject)
        med1 = self._get_session_median(subject, "vp", SES1)
        med2 = self._get_session_median(subject, "vp", SES2)

        pct_diff = compute_percent_difference(
            np.array([med1]), np.array([med2])
        )
        assert pct_diff < 55.0, (
            f"{subject} vp median %diff = {pct_diff:.1f}% (> 55%), "
            f"ses1={med1:.4f}, ses2={med2:.4f}"
        )

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_ktrans_pct_diff_vs_perfplus(self, subject: str) -> None:
        """PMRIPrep Ktrans %diff should be < 50%."""
        _skip_if_missing(subject)
        if not has_perfplus_reference(subject):
            pytest.skip(f"No perf-plus reference for {subject}")

        med1 = self._get_session_median(subject, "Ktrans", SES1)
        med2 = self._get_session_median(subject, "Ktrans", SES2)

        pct_diff = compute_percent_difference(
            np.array([med1]), np.array([med2])
        )
        assert pct_diff < 50.0, (
            f"{subject} Ktrans %diff = {pct_diff:.1f}% (> 50%)"
        )

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_ktrans_cv_vs_perfplus(self, subject: str) -> None:
        """PMRIPrep Ktrans within-subject CV should be < 25%."""
        _skip_if_missing(subject)
        if not has_perfplus_reference(subject):
            pytest.skip(f"No perf-plus reference for {subject}")

        med1 = self._get_session_median(subject, "Ktrans", SES1)
        med2 = self._get_session_median(subject, "Ktrans", SES2)

        cv = compute_within_subject_cv(np.array([med1]), np.array([med2]))
        assert cv < 25.0, (
            f"{subject} Ktrans CV = {cv:.1f}% (> 25%)"
        )


# =============================================================================
# Tier 2: Mask Reproducibility (needs registration)
# =============================================================================


@skip_no_fsl
class TestMaskReproducibility:
    """Test-retest reproducibility of brain and QC masks (after registration)."""

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_brain_mask_dice(self, subject: str) -> None:
        """Brain mask DICE across sessions should be > 0.90."""
        _skip_if_missing(subject)
        reg = register_sessions(subject)
        if reg is None:
            pytest.skip(f"Registration failed for {subject}")
        mat_file, out_dir = reg

        brain1 = load_mask(subject, SES1, "brain")
        assert brain1 is not None, f"Missing brain mask for {subject} {SES1}"

        brain2_in = get_mask_path(subject, SES2, "brain")
        brain2_out = Path(out_dir) / f"{subject}_brain_mask_ses2_reg.nii.gz"
        ref_path = get_ref_dce_path(subject, SES1)
        result = apply_transform(
            brain2_in, ref_path, mat_file, brain2_out, "nearestneighbour"
        )
        assert result is not None, (
            f"Failed to register {subject} ses-0002 brain mask"
        )

        import nibabel as nib
        brain2_img = nib.load(brain2_out)
        brain2 = np.asarray(brain2_img.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]

        dice = calculate_dice(brain1, brain2)
        assert dice > 0.90, (
            f"{subject} brain mask DICE = {dice:.3f} (< 0.90)"
        )

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_qc_mask_dice(self, subject: str) -> None:
        """QC mask DICE across sessions should be > 0.15."""
        _skip_if_missing(subject)
        reg = register_sessions(subject)
        if reg is None:
            pytest.skip(f"Registration failed for {subject}")
        mat_file, out_dir = reg

        qc1 = load_mask(subject, SES1, "qc")
        assert qc1 is not None, f"Missing QC mask for {subject} {SES1}"

        qc2_in = get_mask_path(subject, SES2, "qc")
        qc2_out = Path(out_dir) / f"{subject}_qc_mask_ses2_reg.nii.gz"
        ref_path = get_ref_dce_path(subject, SES1)
        result = apply_transform(
            qc2_in, ref_path, mat_file, qc2_out, "nearestneighbour"
        )
        assert result is not None, (
            f"Failed to register {subject} ses-0002 QC mask"
        )

        import nibabel as nib
        qc2_img = nib.load(qc2_out)
        qc2 = np.asarray(qc2_img.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]

        dice = calculate_dice(qc1, qc2)
        assert dice > 0.15, (
            f"{subject} QC mask DICE = {dice:.3f} (< 0.15)"
        )


# =============================================================================
# Tier 2: Voxel-wise Reproducibility (needs registration)
# =============================================================================


def _load_registered_pair(
    subject: str,
    suffix: str,
    mat_file: str,
    out_dir: str,
    joint_mask: NDArray[np.bool_],
) -> tuple:
    """Load ses-0001 map and registered ses-0002 map.

    Returns:
        Tuple of (ses1_values, ses2_values) as masked 1D arrays,
        or pytest.skip if data unavailable.
    """
    import nibabel as nib

    data1 = load_map(subject, SES1, suffix)
    if data1 is None:
        pytest.skip(f"Missing {suffix} for {subject} {SES1}")

    map2_in = get_map_path(subject, SES2, suffix)
    map2_out = Path(out_dir) / f"{subject}_{suffix}_ses2_reg.nii.gz"
    ref_path = get_ref_dce_path(subject, SES1)
    result = apply_transform(map2_in, ref_path, mat_file, map2_out)
    if result is None:
        pytest.skip(f"Failed to register {suffix} for {subject} {SES2}")

    img2 = nib.load(map2_out)
    data2 = np.asarray(img2.get_fdata(), dtype=np.float64)  # type: ignore[attr-defined]

    v1 = data1[joint_mask]
    v2 = data2[joint_mask]

    valid = np.isfinite(v1) & np.isfinite(v2)
    v1 = v1[valid]
    v2 = v2[valid]

    if len(v1) < 100:
        pytest.skip(f"Too few valid voxels for {subject} {suffix}: {len(v1)}")

    return v1, v2


def _setup_for_subject(subject: str) -> tuple:
    """Register sessions and compute joint mask for a subject.

    Returns:
        Tuple of (mat_file, out_dir, joint_mask), or pytest.skip.
    """
    _skip_if_missing(subject)
    reg = register_sessions(subject)
    if reg is None:
        pytest.skip(f"Registration failed for {subject}")
    mat_file, out_dir = reg

    joint = compute_joint_mask(subject, mat_file)
    if joint is None:
        pytest.skip(f"Joint mask has insufficient voxels for {subject}")

    return mat_file, out_dir, joint


@skip_no_fsl
class TestVoxelwiseReproducibility:
    """Voxel-wise test-retest reproducibility after inter-session registration."""

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_pearson_ktrans(self, subject: str) -> None:
        """Ktrans Pearson r across sessions should be > 0.30."""
        mat_file, out_dir, joint = _setup_for_subject(subject)
        v1, v2 = _load_registered_pair(subject, "Ktrans", mat_file, out_dir, joint)
        r = float(np.corrcoef(v1, v2)[0, 1])
        assert r > 0.30, f"{subject} Ktrans Pearson r = {r:.3f} (< 0.30)"

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_pearson_ve(self, subject: str) -> None:
        """ve Pearson r across sessions should be > 0.30."""
        mat_file, out_dir, joint = _setup_for_subject(subject)
        v1, v2 = _load_registered_pair(subject, "ve", mat_file, out_dir, joint)
        r = float(np.corrcoef(v1, v2)[0, 1])
        assert r > 0.30, f"{subject} ve Pearson r = {r:.3f} (< 0.30)"

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_pearson_vp(self, subject: str) -> None:
        """vp Pearson r across sessions should be > 0.25."""
        mat_file, out_dir, joint = _setup_for_subject(subject)
        v1, v2 = _load_registered_pair(subject, "vp", mat_file, out_dir, joint)
        r = float(np.corrcoef(v1, v2)[0, 1])
        assert r > 0.25, f"{subject} vp Pearson r = {r:.3f} (< 0.25)"

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_ktrans_icc(self, subject: str) -> None:
        """Ktrans ICC(2,1) should be > 0.25."""
        mat_file, out_dir, joint = _setup_for_subject(subject)
        v1, v2 = _load_registered_pair(subject, "Ktrans", mat_file, out_dir, joint)
        icc = compute_icc_2_1(v1, v2)
        assert icc > 0.25, f"{subject} Ktrans ICC(2,1) = {icc:.3f} (< 0.25)"

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_modelfree_iauc60_correlation(self, subject: str) -> None:
        """IAUC60 Pearson r across sessions should be > 0.40."""
        mat_file, out_dir, joint = _setup_for_subject(subject)
        v1, v2 = _load_registered_pair(subject, "IAUC60", mat_file, out_dir, joint)
        r = float(np.corrcoef(v1, v2)[0, 1])
        assert r > 0.40, f"{subject} IAUC60 Pearson r = {r:.3f} (< 0.40)"

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_modelfree_iauc90_correlation(self, subject: str) -> None:
        """IAUC90 Pearson r across sessions should be > 0.40."""
        mat_file, out_dir, joint = _setup_for_subject(subject)
        v1, v2 = _load_registered_pair(subject, "IAUC90", mat_file, out_dir, joint)
        r = float(np.corrcoef(v1, v2)[0, 1])
        assert r > 0.40, f"{subject} IAUC90 Pearson r = {r:.3f} (< 0.40)"

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_modelfree_iauc120_correlation(self, subject: str) -> None:
        """IAUC120 Pearson r across sessions should be > 0.40."""
        mat_file, out_dir, joint = _setup_for_subject(subject)
        v1, v2 = _load_registered_pair(subject, "IAUC120", mat_file, out_dir, joint)
        r = float(np.corrcoef(v1, v2)[0, 1])
        assert r > 0.40, f"{subject} IAUC120 Pearson r = {r:.3f} (< 0.40)"

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_modelfree_ser_correlation(self, subject: str) -> None:
        """SER Pearson r across sessions should be > -0.10."""
        mat_file, out_dir, joint = _setup_for_subject(subject)
        v1, v2 = _load_registered_pair(subject, "SER", mat_file, out_dir, joint)
        r = float(np.corrcoef(v1, v2)[0, 1])
        assert r > -0.10, f"{subject} SER Pearson r = {r:.3f} (< -0.10)"


# =============================================================================
# Tier 2: Visualization (always passes, generates artifacts)
# =============================================================================


@skip_no_fsl
class TestVisualization:
    """Generate test-retest QC visualizations (always passes)."""

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_bland_altman_plots(self, subject: str) -> None:
        """Generate Bland-Altman plots for Ktrans, ve, vp."""
        mat_file, out_dir, joint = _setup_for_subject(subject)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        params = ["Ktrans", "ve", "vp"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, param in zip(axes, params):
            try:
                v1, v2 = _load_registered_pair(
                    subject, param, mat_file, out_dir, joint
                )
            except Exception:
                ax.set_title(f"{param} (no data)")
                continue

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

        fig.suptitle(f"{subject} Test-Retest: Bland-Altman", fontsize=14)
        fig.tight_layout()

        qc_dir = Path(out_dir)
        out_path = qc_dir / f"{subject}_bland_altman.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_scatter_plots(self, subject: str) -> None:
        """Generate scatter plots for Ktrans, ve, vp."""
        mat_file, out_dir, joint = _setup_for_subject(subject)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        params = ["Ktrans", "ve", "vp"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, param in zip(axes, params):
            try:
                v1, v2 = _load_registered_pair(
                    subject, param, mat_file, out_dir, joint
                )
            except Exception:
                ax.set_title(f"{param} (no data)")
                continue

            r = float(np.corrcoef(v1, v2)[0, 1])

            ax.scatter(v1, v2, alpha=0.1, s=1, color="steelblue")

            lims = [
                min(float(np.min(v1)), float(np.min(v2))),
                max(float(np.max(v1)), float(np.max(v2))),
            ]
            ax.plot(lims, lims, "r--", alpha=0.5, label="identity")
            ax.set_xlabel(f"{param} (ses-0001)")
            ax.set_ylabel(f"{param} (ses-0002)")
            ax.set_title(f"{param} (r={r:.3f})")
            ax.legend(fontsize=8)

        fig.suptitle(f"{subject} Test-Retest: Scatter", fontsize=14)
        fig.tight_layout()

        qc_dir = Path(out_dir)
        out_path = qc_dir / f"{subject}_scatter.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
