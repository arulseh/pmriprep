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


class TestBaseConfidence:
    """Tests for _compute_base_confidence."""

    def test_t1w_precon_high_res(self) -> None:
        cand = _make_candidate(ImageType.T1W_PRECON, is_high_res=True)
        assert _compute_base_confidence(cand) == pytest.approx(1.0)

    def test_t1w_precon_low_res(self) -> None:
        cand = _make_candidate(ImageType.T1W_PRECON, is_high_res=False)
        assert _compute_base_confidence(cand) == pytest.approx(0.4)

    def test_t1w_postcon_high_res(self) -> None:
        cand = _make_candidate(ImageType.T1W_POSTCON, is_high_res=True)
        assert _compute_base_confidence(cand) == pytest.approx(0.7)

    def test_t1w_postcon_low_res(self) -> None:
        cand = _make_candidate(ImageType.T1W_POSTCON, is_high_res=False)
        assert _compute_base_confidence(cand) == pytest.approx(0.3)

    def test_t2w_high_res(self) -> None:
        cand = _make_candidate(ImageType.T2W, is_high_res=True)
        assert _compute_base_confidence(cand) == pytest.approx(0.7)

    def test_t2w_low_res(self) -> None:
        cand = _make_candidate(ImageType.T2W, is_high_res=False)
        assert _compute_base_confidence(cand) == pytest.approx(0.3)

    def test_flair_precon_high_res(self) -> None:
        cand = _make_candidate(ImageType.FLAIR_PRECON, is_high_res=True)
        assert _compute_base_confidence(cand) == pytest.approx(0.65)

    def test_flair_precon_low_res(self) -> None:
        cand = _make_candidate(ImageType.FLAIR_PRECON, is_high_res=False)
        assert _compute_base_confidence(cand) == pytest.approx(0.3)

    def test_flair_postcon_high_res(self) -> None:
        cand = _make_candidate(ImageType.FLAIR_POSTCON, is_high_res=True)
        assert _compute_base_confidence(cand) == pytest.approx(0.55)

    def test_flair_postcon_low_res(self) -> None:
        cand = _make_candidate(ImageType.FLAIR_POSTCON, is_high_res=False)
        assert _compute_base_confidence(cand) == pytest.approx(0.2)


class TestLesionMaskBonus:
    """Tests for lesion mask confidence overrides."""

    def test_postcon_t1w_with_lesion_mask(self) -> None:
        """Post-contrast T1w: 0.7 -> 0.9 with lesion mask."""
        cand = _make_candidate(
            ImageType.T1W_POSTCON, is_high_res=True, has_lesion_mask=True,
        )
        assert _compute_base_confidence(cand) == pytest.approx(0.9)

    def test_postcon_t1w_without_lesion_mask(self) -> None:
        cand = _make_candidate(
            ImageType.T1W_POSTCON, is_high_res=True, has_lesion_mask=False,
        )
        assert _compute_base_confidence(cand) == pytest.approx(0.7)

    def test_postcon_flair_with_lesion_mask(self) -> None:
        """Post-contrast FLAIR: 0.55 -> 0.70 with lesion mask."""
        cand = _make_candidate(
            ImageType.FLAIR_POSTCON, is_high_res=True, has_lesion_mask=True,
        )
        assert _compute_base_confidence(cand) == pytest.approx(0.70)

    def test_postcon_flair_without_lesion_mask(self) -> None:
        cand = _make_candidate(
            ImageType.FLAIR_POSTCON, is_high_res=True, has_lesion_mask=False,
        )
        assert _compute_base_confidence(cand) == pytest.approx(0.55)

    def test_precon_t1w_lesion_mask_no_effect(self) -> None:
        """Lesion mask has no effect on precon T1w confidence."""
        cand = _make_candidate(
            ImageType.T1W_PRECON, is_high_res=True, has_lesion_mask=True,
        )
        assert _compute_base_confidence(cand) == pytest.approx(1.0)


class TestIsotropyAdjustment:
    """Tests for _apply_isotropy_adjustment."""

    def test_isotropic_bonus(self) -> None:
        """Nearly isotropic (ratio < 1.2) gets +5% bonus."""
        result = _apply_isotropy_adjustment(0.8, isotropy_ratio=1.0)
        assert result == pytest.approx(0.84)

    def test_neutral_range(self) -> None:
        """Ratio between 1.2 and 2.5: no adjustment."""
        result = _apply_isotropy_adjustment(0.8, isotropy_ratio=1.5)
        assert result == pytest.approx(0.8)

    def test_anisotropic_penalty(self) -> None:
        """Highly anisotropic (ratio > 2.5) gets -10% penalty."""
        result = _apply_isotropy_adjustment(0.8, isotropy_ratio=3.0)
        assert result == pytest.approx(0.72)

    def test_clamp_to_1(self) -> None:
        """Result clamped to maximum 1.0."""
        result = _apply_isotropy_adjustment(1.0, isotropy_ratio=1.0)
        assert result <= 1.0

    def test_clamp_to_0(self) -> None:
        """Result clamped to minimum 0.0."""
        result = _apply_isotropy_adjustment(0.0, isotropy_ratio=3.0)
        assert result >= 0.0


class TestApplicationOverrides:
    """Tests for _apply_application_overrides."""

    def test_fast_excludes_postcon(self) -> None:
        """FAST segmentation excludes post-contrast T1w (score 0.0)."""
        cand = _make_candidate(ImageType.T1W_POSTCON)
        result = _apply_application_overrides(0.7, cand, "fast_segmentation")
        assert result == pytest.approx(0.0)

    def test_fast_minimum_precon(self) -> None:
        """FAST ensures precon T1w has minimum confidence 0.7."""
        cand = _make_candidate(ImageType.T1W_PRECON)
        result = _apply_application_overrides(0.4, cand, "fast_segmentation")
        assert result == pytest.approx(0.7)

    def test_fast_keeps_higher_precon(self) -> None:
        """FAST keeps precon confidence when already > 0.7."""
        cand = _make_candidate(ImageType.T1W_PRECON)
        result = _apply_application_overrides(0.9, cand, "fast_segmentation")
        assert result == pytest.approx(0.9)

    def test_dce_postcon_boost(self) -> None:
        """DCE registration boosts postcon T1w to 0.98."""
        cand = _make_candidate(ImageType.T1W_POSTCON)
        result = _apply_application_overrides(0.7, cand, "dce_registration")
        assert result == pytest.approx(0.98)

    def test_dce_precon_penalized_when_postcon_available(self) -> None:
        """DCE penalizes precon T1w by *0.8 when postcon is available."""
        cand = _make_candidate(ImageType.T1W_PRECON)
        result = _apply_application_overrides(
            1.0, cand, "dce_registration", has_postcon_available=True,
        )
        assert result == pytest.approx(0.8)

    def test_dce_precon_no_penalty_when_no_postcon(self) -> None:
        """DCE does not penalize precon when no postcon available."""
        cand = _make_candidate(ImageType.T1W_PRECON)
        result = _apply_application_overrides(
            1.0, cand, "dce_registration", has_postcon_available=False,
        )
        assert result == pytest.approx(1.0)

    def test_dsc_no_overrides(self) -> None:
        """DSC registration applies no overrides."""
        cand = _make_candidate(ImageType.T1W_PRECON)
        result = _apply_application_overrides(0.8, cand, "dsc_registration")
        assert result == pytest.approx(0.8)

    def test_default_no_overrides(self) -> None:
        """Default application applies no overrides."""
        cand = _make_candidate(ImageType.T1W_PRECON)
        result = _apply_application_overrides(0.8, cand, "default")
        assert result == pytest.approx(0.8)


class TestClassifyImageType:
    """Tests for classify_image_type."""

    def test_t1w_precon(self) -> None:
        path = Path("/data/sub-01_T1w.nii.gz")
        assert classify_image_type(path) == ImageType.T1W_PRECON

    def test_t1w_postcon(self) -> None:
        path = Path("/data/sub-01_ce-GAD_T1w.nii.gz")
        assert classify_image_type(path) == ImageType.T1W_POSTCON

    def test_t2w(self) -> None:
        path = Path("/data/sub-01_T2w.nii.gz")
        assert classify_image_type(path) == ImageType.T2W

    def test_flair_precon(self) -> None:
        path = Path("/data/sub-01_FLAIR.nii.gz")
        assert classify_image_type(path) == ImageType.FLAIR_PRECON

    def test_flair_postcon(self) -> None:
        path = Path("/data/sub-01_ce-GAD_FLAIR.nii.gz")
        assert classify_image_type(path) == ImageType.FLAIR_POSTCON


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
        assert result.confidence == pytest.approx(0.98, abs=0.05)

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

    def test_hybrid_fast_with_acceptable_precon(self, tmp_path: Path) -> None:
        """1.8mm precon + 1mm postcon: postcon wins, hybrid FAST on precon."""
        precon = tmp_path / "sub-01_T1w.nii.gz"
        postcon = tmp_path / "sub-01_ce-GAD_T1w.nii.gz"
        _create_nifti(precon, voxel_sizes=(1.8, 1.8, 1.8))
        _create_nifti(postcon, voxel_sizes=(1.0, 1.0, 1.0))
        ref, is_postcon, fast = _determine_processing_mode_v2(
            str(precon), str(postcon),
        )
        assert ref == str(postcon)
        assert is_postcon is True
        # 1.8mm precon: geometric_mean=1.8, isotropy=1.0
        # Meets FAST gate (<=2.0mm and isotropy <=3.0)
        assert fast == str(precon)


class TestMapSelectionToAnatOutput:
    """Tests for _map_selection_to_anat_output."""

    def test_precon_t1w(self) -> None:
        assert _map_selection_to_anat_output(ImageType.T1W_PRECON) == "outputnode.t1w_preproc"

    def test_postcon_t1w(self) -> None:
        assert _map_selection_to_anat_output(ImageType.T1W_POSTCON) == "outputnode.t1w_postcon_coreg"

    def test_t2w(self) -> None:
        assert _map_selection_to_anat_output(ImageType.T2W) == "outputnode.t2w_coreg"

    def test_flair_precon(self) -> None:
        assert _map_selection_to_anat_output(ImageType.FLAIR_PRECON) == "outputnode.flair_coreg"

    def test_flair_postcon(self) -> None:
        assert _map_selection_to_anat_output(ImageType.FLAIR_POSTCON) == "outputnode.flair_postcon_coreg"


class TestScoreCandidate:
    """Tests for the full scoring pipeline (_score_candidate)."""

    def test_high_res_precon_default(self) -> None:
        """1mm isotropic precon T1w scores ~1.0 for default application."""
        cand = _make_candidate(ImageType.T1W_PRECON, is_high_res=True)
        score = _score_candidate(cand, "default")
        # base=1.0, isotropy bonus *1.05 → clamped to 1.0
        assert score == pytest.approx(1.0)

    def test_anisotropic_penalty_propagates(self) -> None:
        """High anisotropy propagates through scoring pipeline."""
        cand = _make_candidate(
            ImageType.T1W_PRECON, is_high_res=True, isotropy_ratio=3.0,
        )
        score = _score_candidate(cand, "default")
        # base=1.0, isotropy penalty *0.9 → 0.9
        assert score == pytest.approx(0.9)

    def test_dce_postcon_overrides_dominate(self) -> None:
        """DCE application override sets postcon to 0.98."""
        cand = _make_candidate(ImageType.T1W_POSTCON, is_high_res=True)
        score = _score_candidate(cand, "dce_registration", has_postcon_available=True)
        assert score == pytest.approx(0.98)


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
