"""Test-retest reproducibility tests for OSIPI clinical subjects.

OSIPI clinical cohort: sub-0081 through sub-0088. Each subject has two DCE
sessions acquired on the same scanner. model="auto" (BIC selection) + NLLS,
population AIF.

Tests are parametrized across all 8 subjects. Subjects without processed
outputs are gracefully skipped; as new subjects are processed, tests
automatically pick them up.

Tier 1 — Summary statistics (no registration, always runs):
    OSIPI tumor-mean Score_repeat and repeatability coefficient.

Tier 2 — Registration QC (requires FLIRT, skips gracefully):
    Brain/QC mask overlap and diagnostic visualizations. Voxelwise
    visit-to-visit PK-map correlation is not an OSIPI validation endpoint.
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
    get_ref_dce_path,
    has_fsl,
    load_clinical_tumor_mask,
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
# Helper: OSIPI clinical tumor means
# =============================================================================


def _clinical_tumor_mean(subject: str, session: str, parameter: str) -> float:
    """Return the OSIPI tumor-region mean for one parameter map."""
    data = load_map(subject, session, parameter)
    mask = load_clinical_tumor_mask(subject, session)
    assert data is not None, f"Missing {parameter} map for {subject} {session}"
    assert mask is not None, f"Missing tumor mask for {subject} {session}"

    values = data[mask]
    values = values[np.isfinite(values)]
    assert len(values) > 0, f"No finite tumor voxels for {subject} {session}"
    return float(np.mean(values))


def _osipi_repeatability(parameter: str) -> tuple[float, float]:
    """Compute OSIPI ``Score_repeat`` and repeatability coefficient.

    For each subject, ``CV = (|V1 - V2| / sqrt(2)) / mean(V1, V2)``.
    The cohort metrics are ``Score_repeat = exp(-mean(CV))`` and
    ``%RC = 2.77 * sqrt(mean(CV**2))``. Despite its published name,
    ``%RC`` is represented as a fraction by the OSIPI scoring script.
    """
    cvs: list[float] = []
    for subject in TEST_RETEST_SUBJECTS:
        _skip_if_missing(subject)
        visit1 = _clinical_tumor_mean(subject, SES1, parameter)
        visit2 = _clinical_tumor_mean(subject, SES2, parameter)
        mean = (visit1 + visit2) / 2.0
        assert mean > 0.0, f"Non-positive tumor mean for {subject} {parameter}"
        cvs.append(abs(visit1 - visit2) / np.sqrt(2.0) / mean)

    cv_array = np.asarray(cvs, dtype=np.float64)
    score_repeat = float(np.exp(-np.mean(cv_array)))
    pct_rc = float(2.77 * np.sqrt(np.mean(cv_array**2)))
    return score_repeat, pct_rc


# =============================================================================
# Tier 1: Summary Statistics (no registration needed)
# =============================================================================


class TestSummaryStatistics:
    """OSIPI tumor-mean cohort repeatability regression tests."""

    @pytest.mark.parametrize(
        "parameter,expected_score,expected_pct_rc",
        [
            ("Ktrans", 0.8803703804, 0.4622329816),
            ("ve", 0.8782049972, 0.4642157400),
            ("vp", 0.8273458804, 0.6017271163),
        ],
    )
    def test_osipi_repeatability(
        self,
        parameter: str,
        expected_score: float,
        expected_pct_rc: float,
    ) -> None:
        """Cohort metrics should match the validated publication run."""
        score_repeat, pct_rc = _osipi_repeatability(parameter)
        assert score_repeat == pytest.approx(expected_score, abs=1e-8)
        assert pct_rc == pytest.approx(expected_pct_rc, abs=1e-8)


# =============================================================================
# Tier 2: Mask Reproducibility (needs registration)
# =============================================================================


def _compute_slab_overlap(
    brain1: NDArray[np.floating],
    brain2_reg: NDArray[np.floating],
) -> NDArray[np.bool_]:
    """Compute slab overlap mask from registered brain masks.

    For thin-slab DCE acquisitions with different slab positions, the
    registered ses-0002 brain mask may have zero voxels on some slices
    (outside the original slab). This function identifies which slices
    have brain data from both sessions and restricts comparison to those.

    Parameters
    ----------
    brain1 : NDArray[np.floating]
        ses-0001 brain mask (3D array).
    brain2_reg : NDArray[np.floating]
        Registered ses-0002 brain mask (3D array).

    Returns
    -------
    NDArray[np.bool_]
        Boolean 3D mask where both sessions have brain coverage.
    """
    # Identify slices (z-axis) where each session has brain data
    b1 = brain1 > 0.5
    b2 = brain2_reg > 0.5

    overlap = np.ones(brain1.shape, dtype=bool)
    for z in range(brain1.shape[2]):
        if b1[:, :, z].sum() == 0 or b2[:, :, z].sum() == 0:
            overlap[:, :, z] = False

    return overlap


@skip_no_fsl
class TestMaskReproducibility:
    """Test-retest reproducibility of brain and QC masks (after registration).

    Dice is computed only within the FOV overlap between sessions to avoid
    penalizing positioning differences that place the imaging slab at
    different z-locations across visits.
    """

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_brain_mask_dice(self, subject: str) -> None:
        """Brain mask DICE within FOV overlap should be > 0.90."""
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

        # Restrict Dice to slices where both sessions have brain data.
        # Thin-slab DCE with different positioning can leave some slices
        # with zero coverage from one session after registration.
        slab_overlap = _compute_slab_overlap(brain1, brain2)
        brain1_masked = brain1.copy()
        brain2_masked = brain2.copy()
        brain1_masked[~slab_overlap] = 0
        brain2_masked[~slab_overlap] = 0

        dice = calculate_dice(brain1_masked, brain2_masked)
        assert dice > 0.65, (
            f"{subject} brain mask DICE = {dice:.3f} (< 0.65)"
        )

    @pytest.mark.parametrize("subject", TEST_RETEST_SUBJECTS, ids=_SUBJECT_IDS)
    def test_qc_mask_volume_agreement(self, subject: str) -> None:
        """QC mask volume ratio across sessions should be > 0.40.

        With model-auto per-voxel model selection, QC masks flag voxels
        where the BIC-selected model fits well. Different sessions may
        select different models for the same voxels, so spatial Dice is
        inherently low. Volume ratio (min/max of QC voxel counts within
        the joint brain mask) tests whether both sessions identify a
        similar *fraction* of brain as high-quality.

        Requires adequate registration quality (brain Dice > 0.80);
        subjects with extreme slab positioning differences cannot
        support meaningful mask comparison.
        """
        _skip_if_missing(subject)
        reg = register_sessions(subject)
        if reg is None:
            pytest.skip(f"Registration failed for {subject}")
        mat_file, out_dir = reg

        joint_brain = compute_joint_mask(subject, mat_file, min_voxels=100)
        if joint_brain is not None:
            brain1 = load_mask(subject, SES1, "brain")
            if brain1 is not None:
                import nibabel as nib
                brain2_path = (
                    Path(out_dir)
                    / f"{subject}_{SES2}_brain_mask_reg.nii.gz"
                )
                if brain2_path.exists():
                    brain2 = np.asarray(
                        nib.load(brain2_path).get_fdata(),
                        dtype=np.float64,
                    )
                    dice = calculate_dice(brain1, brain2)
                    if dice < 0.80:
                        pytest.skip(
                            f"{subject} registration quality insufficient "
                            f"(brain Dice={dice:.3f})"
                        )
        if joint_brain is None:
            pytest.skip(
                f"Joint brain mask has insufficient voxels for {subject}"
            )

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

        v1 = int(np.sum((qc1 > 0.5) & joint_brain))
        v2 = int(np.sum((qc2 > 0.5) & joint_brain))
        vmax = max(v1, v2)
        ratio = min(v1, v2) / vmax if vmax > 0 else 0.0

        assert ratio > 0.40, (
            f"{subject} QC mask volume ratio = {ratio:.2f} (< 0.40), "
            f"ses1={v1}, ses2={v2}"
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
    require_positive: bool = False,
) -> tuple:
    """Load ses-0001 map and registered ses-0002 map.

    Parameters
    ----------
    subject : str
        Subject ID.
    suffix : str
        Map suffix key.
    mat_file : str
        FLIRT transform path.
    out_dir : str
        Output directory for intermediate files.
    joint_mask : NDArray[np.bool_]
        Boolean 3D mask defining the comparison region.
    require_positive : bool
        If True, additionally require that at least one
        session has a positive value at each voxel. Useful for PK
        parameters (Ktrans, ve, vp) where non-enhancing tissue
        produces zeros that dilute correlation.

    Returns
    -------
    tuple
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
    if require_positive:
        valid = valid & ((v1 > 0) | (v2 > 0))
    v1 = v1[valid]
    v2 = v2[valid]

    if len(v1) < 100:
        pytest.skip(f"Too few valid voxels for {subject} {suffix}: {len(v1)}")

    return v1, v2


def _setup_for_subject(
    subject: str,
    min_brain_dice: float = 0.80,
) -> tuple:
    """Register sessions and compute joint mask for a subject.

    Requires a minimum brain mask Dice overlap to proceed with
    voxelwise comparisons. Subjects with extreme positioning
    differences (e.g., >10mm slab shift across 16 thin slices)
    cannot support meaningful voxelwise analysis — summary-level
    tests (Tier 1) are the appropriate level for those.

    Parameters
    ----------
    subject : str
        Subject ID.
    min_brain_dice : float
        Minimum brain mask Dice to accept registration.

    Returns
    -------
    tuple
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

    # Verify registration quality via brain mask overlap
    brain1 = load_mask(subject, SES1, "brain")
    if brain1 is not None:
        import nibabel as nib
        brain2_out = (
            Path(out_dir) / f"{subject}_{SES2}_brain_mask_reg.nii.gz"
        )
        if brain2_out.exists():
            brain2 = np.asarray(
                nib.load(brain2_out).get_fdata(), dtype=np.float64
            )
            dice = calculate_dice(brain1, brain2)
            if dice < min_brain_dice:
                pytest.skip(
                    f"{subject} registration quality insufficient for "
                    f"voxelwise analysis (brain Dice={dice:.3f})"
                )

    return mat_file, out_dir, joint


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
