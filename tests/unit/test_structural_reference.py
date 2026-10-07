"""Unit tests for confidence-based structural reference selection."""

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import pytest

from pmriprep.utils.structural_reference import (
    ImageType,
    StructuralCandidate,
    _apply_application_overrides,
    _apply_isotropy_adjustment,
    _compute_base_confidence,
    _determine_processing_mode_v2,
    _map_selection_to_anat_output,
    _score_candidate,
    build_candidates_from_session,
    classify_image_type,
    compute_fov_coverage,
    select_structural_reference_v2,
)


def _make_candidate(
    image_type: ImageType = ImageType.T1W_PRECON,
    geometric_mean: float = 1.0,
    isotropy_ratio: float = 1.0,
    is_high_res: bool = True,
    has_lesion_mask: bool = False,
    voxel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    path: str = "/tmp/test.nii.gz",
) -> StructuralCandidate:
    """Helper to create a StructuralCandidate for testing."""
    return StructuralCandidate(
        path=Path(path),
        image_type=image_type,
        voxel_sizes=voxel_sizes,
        geometric_mean=geometric_mean,
        isotropy_ratio=isotropy_ratio,
        is_high_res=is_high_res,
        has_lesion_mask=has_lesion_mask,
    )


def _create_nifti(
    path: Path,
    shape: Tuple[int, int, int] = (64, 64, 64),
    voxel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    affine: np.ndarray = None,
) -> None:
    """Create a minimal NIfTI file for testing."""
    if affine is None:
        affine = np.diag([*voxel_sizes, 1.0])
    data = np.random.randint(0, 100, shape, dtype=np.int16)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))


@pytest.mark.parametrize(
    ("image_type", "high_res", "lesion_mask", "expected"),
    [
        pytest.param(ImageType.T1W_PRECON, True, False, 1.0, id="precon-t1-high-res"),
        pytest.param(ImageType.T1W_PRECON, False, False, 0.4, id="precon-t1-low-res"),
        pytest.param(ImageType.T1W_POSTCON, True, False, 0.7, id="postcon-t1-high-res"),
        pytest.param(ImageType.T1W_POSTCON, False, False, 0.3, id="postcon-t1-low-res"),
        pytest.param(ImageType.T2W, True, False, 0.7, id="t2-high-res"),
        pytest.param(ImageType.T2W, False, False, 0.3, id="t2-low-res"),
        pytest.param(ImageType.FLAIR_PRECON, True, False, 0.65,
                     id="precon-flair-high-res"),
        pytest.param(ImageType.FLAIR_PRECON, False, False, 0.3,
                     id="precon-flair-low-res"),
        pytest.param(ImageType.FLAIR_POSTCON, True, False, 0.55,
                     id="postcon-flair-high-res"),
        pytest.param(ImageType.FLAIR_POSTCON, False, False, 0.2,
                     id="postcon-flair-low-res"),
        pytest.param(ImageType.T1W_POSTCON, True, True, 0.9,
                     id="postcon-t1-lesion-bonus"),
        pytest.param(ImageType.FLAIR_POSTCON, True, True, 0.70,
                     id="postcon-flair-lesion-bonus"),
        pytest.param(ImageType.T1W_PRECON, True, True, 1.0,
                     id="precon-t1-lesion-no-effect"),
    ],
)
def test_base_confidence(
    image_type: ImageType,
    high_res: bool,
    lesion_mask: bool,
    expected: float,
) -> None:
    candidate = _make_candidate(
        image_type,
        is_high_res=high_res,
        has_lesion_mask=lesion_mask,
    )
    assert _compute_base_confidence(candidate) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("confidence", "ratio", "expected"),
    [
        pytest.param(0.8, 1.0, 0.84, id="isotropic-bonus"),
        pytest.param(0.8, 1.2, 0.8, id="bonus-boundary-neutral"),
        pytest.param(0.8, 1.5, 0.8, id="neutral-range"),
        pytest.param(0.8, 2.5, 0.8, id="penalty-boundary-neutral"),
        pytest.param(0.8, 3.0, 0.72, id="anisotropic-penalty"),
        pytest.param(1.0, 1.0, 1.0, id="upper-clamp"),
        pytest.param(0.0, 3.0, 0.0, id="lower-clamp"),
    ],
)
def test_isotropy_adjustment(
    confidence: float,
    ratio: float,
    expected: float,
) -> None:
    assert _apply_isotropy_adjustment(confidence, ratio) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("confidences", "image_type", "application", "has_postcon", "expected"),
    [
        pytest.param((0.7,), ImageType.T1W_POSTCON, "fast_segmentation", False,
                     (0.0,),
                     id="fast-excludes-postcon"),
        pytest.param((0.4,), ImageType.T1W_PRECON, "fast_segmentation", False,
                     (0.7,),
                     id="fast-precon-floor"),
        pytest.param((0.9,), ImageType.T1W_PRECON, "fast_segmentation", False,
                     (0.9,),
                     id="fast-keeps-higher-precon"),
        pytest.param((0.7, 0.3), ImageType.T1W_POSTCON, "dce_registration", False,
                     (0.875, 0.375), id="dce-postcon-bonus"),
        pytest.param((1.0,), ImageType.T1W_PRECON, "dce_registration", True,
                     (0.8,),
                     id="dce-precon-penalty"),
        pytest.param((1.0,), ImageType.T1W_PRECON, "dce_registration", False,
                     (1.0,),
                     id="dce-precon-no-penalty"),
        pytest.param((0.8,), ImageType.T1W_PRECON, "dsc_registration", False,
                     (0.8,),
                     id="dsc-no-override"),
        pytest.param((0.8,), ImageType.T1W_PRECON, "default", False, (0.8,),
                     id="default-no-override"),
    ],
)
def test_application_overrides(
    confidences: tuple[float, ...],
    image_type: ImageType,
    application: str,
    has_postcon: bool,
    expected: tuple[float, ...],
) -> None:
    candidate = _make_candidate(image_type)
    for confidence, result_expected in zip(confidences, expected, strict=True):
        result = _apply_application_overrides(
            confidence,
            candidate,
            application,
            has_postcon_available=has_postcon,
        )
        assert result == pytest.approx(result_expected)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param("sub-01_T1w.nii.gz", ImageType.T1W_PRECON, id="t1-precon"),
        pytest.param("sub-01_ce-GAD_T1w.nii.gz", ImageType.T1W_POSTCON,
                     id="t1-postcon"),
        pytest.param("sub-01_T2w.nii.gz", ImageType.T2W, id="t2"),
        pytest.param("sub-01_FLAIR.nii.gz", ImageType.FLAIR_PRECON,
                     id="flair-precon"),
        pytest.param("sub-01_ce-GAD_FLAIR.nii.gz", ImageType.FLAIR_POSTCON,
                     id="flair-postcon"),
    ],
)
def test_classify_image_type(name: str, expected: ImageType) -> None:
    assert classify_image_type(Path("/data") / name) == expected


class TestFOVCoverage:
    """Tests for compute_fov_coverage."""

    def test_full_overlap(self, tmp_path: Path) -> None:
        """Identical FOV gives coverage = 1.0."""
        struct = tmp_path / "struct.nii.gz"
        perf = tmp_path / "perf.nii.gz"
        _create_nifti(struct, shape=(64, 64, 64), voxel_sizes=(1.0, 1.0, 1.0))
        _create_nifti(perf, shape=(64, 64, 64), voxel_sizes=(1.0, 1.0, 1.0))
        coverage = compute_fov_coverage(struct, perf)
        assert coverage == pytest.approx(1.0, abs=0.01)

    def test_partial_overlap(self, tmp_path: Path) -> None:
        """Smaller structural FOV gives coverage < 1.0."""
        struct = tmp_path / "struct.nii.gz"
        perf = tmp_path / "perf.nii.gz"
        _create_nifti(struct, shape=(32, 64, 64), voxel_sizes=(1.0, 1.0, 1.0))
        _create_nifti(perf, shape=(64, 64, 64), voxel_sizes=(1.0, 1.0, 1.0))
        coverage = compute_fov_coverage(struct, perf)
        assert 0.0 < coverage < 1.0

    def test_no_overlap(self, tmp_path: Path) -> None:
        """Completely non-overlapping FOVs give coverage = 0.0."""
        struct = tmp_path / "struct.nii.gz"
        perf = tmp_path / "perf.nii.gz"
        # Structural at origin
        _create_nifti(struct, shape=(10, 10, 10), voxel_sizes=(1.0, 1.0, 1.0))
        # Perfusion far away
        perf_affine = np.diag([1.0, 1.0, 1.0, 1.0])
        perf_affine[:3, 3] = [100.0, 100.0, 100.0]
        _create_nifti(perf, shape=(10, 10, 10), affine=perf_affine)
        coverage = compute_fov_coverage(struct, perf)
        assert coverage == pytest.approx(0.0, abs=0.01)


class TestEndToEndSelection:
    """Tests for select_structural_reference_v2."""

    def test_precon_wins_over_postcon_for_dsc(self) -> None:
        """For DSC registration, high-res precon T1w wins over postcon."""
        precon = _make_candidate(ImageType.T1W_PRECON, is_high_res=True)
        postcon = _make_candidate(ImageType.T1W_POSTCON, is_high_res=True)
        result = select_structural_reference_v2(
            [precon, postcon], application="dsc_registration",
        )
        assert result is not None
        assert result.candidate.image_type == ImageType.T1W_PRECON

    def test_postcon_wins_for_dce(self) -> None:
        """For DCE registration, postcon T1w wins (contrast-matched)."""
        precon = _make_candidate(ImageType.T1W_PRECON, is_high_res=True)
        postcon = _make_candidate(ImageType.T1W_POSTCON, is_high_res=True)
        result = select_structural_reference_v2(
            [precon, postcon], application="dce_registration",
        )
        assert result is not None
        assert result.candidate.image_type == ImageType.T1W_POSTCON
        assert result.confidence == pytest.approx(0.91875)

    def test_high_res_precon_wins_over_low_res_postcon_for_dce(self) -> None:
        """Contrast matching does not replace the image-quality score."""
        precon = _make_candidate(ImageType.T1W_PRECON, is_high_res=True)
        postcon = _make_candidate(ImageType.T1W_POSTCON, is_high_res=False)
        result = select_structural_reference_v2(
            [precon, postcon], application="dce_registration",
        )
        assert result is not None
        assert result.candidate.image_type == ImageType.T1W_PRECON

    def test_isotropic_precon_wins_over_anisotropic_postcon_for_dce(self) -> None:
        """Contrast matching cannot erase a strong anisotropy penalty."""
        precon = _make_candidate(ImageType.T1W_PRECON, is_high_res=True)
        postcon = _make_candidate(
            ImageType.T1W_POSTCON,
            is_high_res=True,
            isotropy_ratio=3.0,
        )
        result = select_structural_reference_v2(
            [precon, postcon], application="dce_registration",
        )
        assert result is not None
        assert result.candidate.image_type == ImageType.T1W_PRECON

    def test_precon_wins_for_fast(self) -> None:
        """For FAST, precon T1w wins (postcon excluded)."""
        precon = _make_candidate(ImageType.T1W_PRECON, is_high_res=True)
        postcon = _make_candidate(ImageType.T1W_POSTCON, is_high_res=True)
        result = select_structural_reference_v2(
            [precon, postcon], application="fast_segmentation",
        )
        assert result is not None
        assert result.candidate.image_type == ImageType.T1W_PRECON

    def test_empty_candidates(self) -> None:
        """Empty candidate list returns None."""
        result = select_structural_reference_v2([], application="default")
        assert result is None

    def test_all_excluded(self) -> None:
        """All candidates excluded returns None."""
        postcon = _make_candidate(ImageType.T1W_POSTCON, is_high_res=True)
        result = select_structural_reference_v2(
            [postcon], application="fast_segmentation",
        )
        assert result is None

    def test_single_candidate(self) -> None:
        """Single candidate is returned with correct confidence."""
        cand = _make_candidate(ImageType.T1W_PRECON, is_high_res=True)
        result = select_structural_reference_v2([cand], application="default")
        assert result is not None
        assert result.candidate is cand
        assert result.confidence > 0.9

    def test_low_res_precon_vs_high_res_postcon(self) -> None:
        """Low-res precon loses to high-res postcon for default."""
        precon = _make_candidate(
            ImageType.T1W_PRECON, geometric_mean=2.5, is_high_res=False,
        )
        postcon = _make_candidate(
            ImageType.T1W_POSTCON, geometric_mean=1.0, is_high_res=True,
        )
        result = select_structural_reference_v2(
            [precon, postcon], application="default",
        )
        assert result is not None
        assert result.candidate.image_type == ImageType.T1W_POSTCON

    def test_provenance_dict(self) -> None:
        """to_provenance_dict returns expected keys."""
        cand = _make_candidate(ImageType.T1W_PRECON, is_high_res=True)
        result = select_structural_reference_v2([cand], application="default")
        assert result is not None
        prov = result.to_provenance_dict()
        expected_keys = {
            "selected_image", "image_type", "confidence", "application",
            "geometric_mean_mm", "isotropy_ratio",
            "is_high_res", "has_lesion_mask", "coverage_fraction",
            "selection_reason", "n_candidates",
        }
        assert set(prov.keys()) == expected_keys

class TestDetermineProcessingModeV2:
    """Tests for _determine_processing_mode_v2."""

    def test_precon_only(self, tmp_path: Path) -> None:
        """Only precon: normal mode, FAST on precon."""
        precon = tmp_path / "sub-01_T1w.nii.gz"
        _create_nifti(precon, voxel_sizes=(1.0, 1.0, 1.0))
        ref, is_postcon, fast = _determine_processing_mode_v2(str(precon), None)
        assert ref == str(precon)
        assert is_postcon is False
        assert fast == str(precon)

    def test_postcon_only(self, tmp_path: Path) -> None:
        """Only postcon: postcon reference, no FAST."""
        postcon = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        _create_nifti(postcon, voxel_sizes=(1.0, 1.0, 1.0))
        ref, is_postcon, fast = _determine_processing_mode_v2(None, str(postcon))
        assert ref == str(postcon)
        assert is_postcon is True
        assert fast is None

    def test_comparable_quality_precon_wins(self, tmp_path: Path) -> None:
        """Both 1mm isotropic: precon wins (penalty makes postcon lose)."""
        precon = tmp_path / "sub-01_T1w.nii.gz"
        postcon = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        _create_nifti(precon, voxel_sizes=(1.0, 1.0, 1.0))
        _create_nifti(postcon, voxel_sizes=(1.0, 1.0, 1.0))
        ref, is_postcon, fast = _determine_processing_mode_v2(
            str(precon), str(postcon),
        )
        assert ref == str(precon)
        assert is_postcon is False

    def test_poor_precon_postcon_wins(self, tmp_path: Path) -> None:
        """2mm SE precon vs 1mm postcon: postcon wins, no hybrid FAST."""
        precon = tmp_path / "sub-01_acq-se_T1w.nii.gz"
        postcon = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        _create_nifti(precon, voxel_sizes=(2.5, 2.5, 5.0))
        _create_nifti(postcon, voxel_sizes=(1.0, 1.0, 1.0))
        ref, is_postcon, fast = _determine_processing_mode_v2(
            str(precon), str(postcon),
        )
        assert ref == str(postcon)
        assert is_postcon is True
        # Precon too anisotropic (ratio=2.0) and poor res → may or may not qualify
        # The geometric mean is (2.5*2.5*5)^(1/3) ≈ 3.15 → POOR, but isotropy ≈ 2.0
        # is within 3.0 gate, but geometric_mean > 2.0 → fails FAST gate
        assert fast is None

    def test_no_images_raises(self) -> None:
        """No images at all raises ValueError."""
        with pytest.raises(ValueError, match="At least one T1w"):
            _determine_processing_mode_v2(None, None)

    def test_max_voxel_dim_cutoff(self, tmp_path: Path) -> None:
        """Voxel > 4mm in any dimension forces low-res classification."""
        precon = tmp_path / "sub-01_T1w.nii.gz"
        postcon = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        # Precon: 1x1x5mm (through-plane > 4mm)
        _create_nifti(precon, voxel_sizes=(1.0, 1.0, 5.0))
        _create_nifti(postcon, voxel_sizes=(1.0, 1.0, 1.0))
        ref, is_postcon, fast = _determine_processing_mode_v2(
            str(precon), str(postcon),
        )
        # Precon has max dim > 4mm, forced to low-res → postcon should win
        assert ref == str(postcon)
        assert is_postcon is True

    def test_hybrid_fast_with_acceptable_precon(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """1.8mm precon + 1mm postcon: postcon wins, hybrid FAST on precon."""
        precon = tmp_path / "sub-01_T1w.nii.gz"
        postcon = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        _create_nifti(precon, voxel_sizes=(1.8, 1.8, 1.8))
        _create_nifti(postcon, voxel_sizes=(1.0, 1.0, 1.0))
        with caplog.at_level("INFO", logger="pmriprep.utils.structural_reference"):
            ref, is_postcon, fast = _determine_processing_mode_v2(
                str(precon), str(postcon),
            )
        assert ref == str(postcon)
        assert is_postcon is True
        # 1.8mm precon: geometric_mean=1.8, isotropy=1.0
        # Meets FAST gate (<=2.0mm and isotropy <=3.0)
        assert fast == str(precon)
        assert "penalty=0.10" in caplog.text

    def test_good_resolution_boundary_uses_strong_penalty(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The inclusive 1.5 mm high-resolution boundary uses the 0.30 tier."""
        precon = tmp_path / "sub-01_T1w.nii.gz"
        postcon = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        _create_nifti(precon, voxel_sizes=(1.5, 1.5, 1.5))
        _create_nifti(postcon, voxel_sizes=(1.0, 1.0, 1.0))

        with caplog.at_level("INFO", logger="pmriprep.utils.structural_reference"):
            ref, is_postcon, _fast = _determine_processing_mode_v2(
                str(precon), str(postcon),
            )

        assert ref == str(precon)
        assert is_postcon is False
        assert "penalty=0.30" in caplog.text


@pytest.mark.parametrize(
    ("image_type", "expected"),
    [
        pytest.param(ImageType.T1W_PRECON, "outputnode.t1w_preproc", id="t1-precon"),
        pytest.param(ImageType.T1W_POSTCON, "outputnode.t1w_postcon_coreg",
                     id="t1-postcon"),
        pytest.param(ImageType.T2W, "outputnode.t2w_coreg", id="t2"),
        pytest.param(ImageType.FLAIR_PRECON, "outputnode.flair_coreg",
                     id="flair-precon"),
        pytest.param(ImageType.FLAIR_POSTCON, "outputnode.flair_postcon_coreg",
                     id="flair-postcon"),
    ],
)
def test_map_selection_to_anat_output(image_type: ImageType, expected: str) -> None:
    assert _map_selection_to_anat_output(image_type) == expected


@pytest.mark.parametrize(
    ("image_type", "isotropy", "application", "has_postcon", "expected"),
    [
        pytest.param(ImageType.T1W_PRECON, 1.0, "default", False, 1.0,
                     id="high-res-precon-default"),
        pytest.param(ImageType.T1W_PRECON, 3.0, "default", False, 0.9,
                     id="anisotropic-penalty"),
        pytest.param(ImageType.T1W_POSTCON, 1.0, "dce_registration", True,
                     0.91875, id="dce-postcon-bonus"),
    ],
)
def test_score_candidate(
    image_type: ImageType,
    isotropy: float,
    application: str,
    has_postcon: bool,
    expected: float,
) -> None:
    candidate = _make_candidate(
        image_type,
        is_high_res=True,
        isotropy_ratio=isotropy,
    )
    assert _score_candidate(
        candidate,
        application,
        has_postcon_available=has_postcon,
    ) == pytest.approx(expected)


class TestBuildCandidatesFromSession:
    """Tests for build_candidates_from_session."""

    def test_builds_from_modalities(self, tmp_path: Path) -> None:
        """Builds candidates from modality paths."""
        # Create mock modality object
        t1w = tmp_path / "sub-01_T1w.nii.gz"
        t1w_ce = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        _create_nifti(t1w, voxel_sizes=(1.0, 1.0, 1.0))
        _create_nifti(t1w_ce, voxel_sizes=(1.0, 1.0, 1.0))

        class MockModalities:
            t1w_pre = t1w
            t1w_ce_gad = t1w_ce
            t2w = None
            flair = None

        candidates = build_candidates_from_session(MockModalities())
        assert len(candidates) == 2
        types = {c.image_type for c in candidates}
        assert ImageType.T1W_PRECON in types
        assert ImageType.T1W_POSTCON in types

    def test_lesion_mask_propagated(self, tmp_path: Path) -> None:
        """has_lesion_mask flag is propagated to all candidates."""
        t1w = tmp_path / "sub-01_T1w.nii.gz"
        _create_nifti(t1w, voxel_sizes=(1.0, 1.0, 1.0))

        class MockModalities:
            t1w_pre = t1w
            t1w_ce_gad = None
            t2w = None
            flair = None

        candidates = build_candidates_from_session(
            MockModalities(), has_lesion_mask=True,
        )
        assert len(candidates) == 1
        assert candidates[0].has_lesion_mask is True

    def test_max_voxel_dim_forces_low_res(self, tmp_path: Path) -> None:
        """Any voxel dimension > 4mm forces is_high_res=False."""
        t1w = tmp_path / "sub-01_T1w.nii.gz"
        # 1x1x5mm: geometric mean ≈ 1.71 (<= 1.5 threshold: False)
        # But max dim = 5mm > 4mm cutoff → forced low-res
        _create_nifti(t1w, voxel_sizes=(1.0, 1.0, 5.0))

        class MockModalities:
            t1w_pre = t1w
            t1w_ce_gad = None
            t2w = None
            flair = None

        candidates = build_candidates_from_session(MockModalities())
        assert len(candidates) == 1
        assert candidates[0].is_high_res is False

    def test_flair_included(self, tmp_path: Path) -> None:
        """FLAIR modality is included in candidates."""
        flair_path = tmp_path / "sub-01_FLAIR.nii.gz"
        _create_nifti(flair_path, voxel_sizes=(1.0, 1.0, 1.0))

        class MockModalities:
            t1w_pre = None
            t1w_ce_gad = None
            t2w = None
            flair = flair_path

        candidates = build_candidates_from_session(MockModalities())
        assert len(candidates) == 1
        assert candidates[0].image_type == ImageType.FLAIR_PRECON

    def test_missing_files_skipped(self, tmp_path: Path) -> None:
        """Non-existent file paths are gracefully skipped."""
        class MockModalities:
            t1w_pre = tmp_path / "nonexistent.nii.gz"
            t1w_ce_gad = None
            t2w = None
            flair = None

        candidates = build_candidates_from_session(MockModalities())
        assert len(candidates) == 0
