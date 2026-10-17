"""Integration tests for DSC sidecar AIF-provenance fields.

Exercises the full DSC datasinks factory write path with a controlled
``qc_metrics`` dict and asserts the resulting JSON sidecars carry the
expected ``PerfusionProcessing.AIF`` block plus the per-map ``AIFDependent`` flag.

Also verifies the drift-guard coverage check: every map emitted by the
factory must have a corresponding entry in ``DSC_AIF_MAP_KEYS``.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.utils.aif_provenance import (
    AIF_INDEPENDENT_MAPS,
    DSC_AIF_MAP_KEYS,
    apply_aif_sidecar,
    build_aif_provenance_block,
)
from pmriprep.utils.sidecar_builders import build_dsc_map_sidecar


def _write_dummy_nifti(path: Path, value: float = 0.0) -> str:
    """Write a constant-valued 4x4x4 NIfTI and return its path."""
    data = np.full((4, 4, 4), value, dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return str(path)


@pytest.fixture
def minimal_bids_dir(tmp_path):
    """Minimal BIDS dataset root with a stub source DSC NIfTI."""
    bids = tmp_path / "bids"
    perf = bids / "sub-test01" / "ses-01" / "perf"
    perf.mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "test", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    src = perf / "sub-test01_ses-01_dsc.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), src)
    return bids


@pytest.fixture
def minimal_inputs(tmp_path, minimal_bids_dir):
    """Dict of trait values that satisfies the DSC factory inputnode."""
    src_dsc = minimal_bids_dir / "sub-test01" / "ses-01" / "perf" / "sub-test01_ses-01_dsc.nii.gz"
    anat = minimal_bids_dir / "sub-test01" / "ses-01" / "anat"
    anat.mkdir(parents=True, exist_ok=True)
    sources = {}
    for attr, name in (
        ("source_t1w", "sub-test01_ses-01_acq-mprage_run-1_T1w.nii.gz"),
        ("source_t1w_postcon", "sub-test01_ses-01_acq-mprage_ce-GAD_run-2_T1w.nii.gz"),
        ("source_flair", "sub-test01_ses-01_acq-3d_run-1_FLAIR.nii.gz"),
        ("source_flair_postcon", "sub-test01_ses-01_acq-3d_ce-GAD_run-2_FLAIR.nii.gz"),
        ("source_t2w", "sub-test01_ses-01_acq-tse_run-1_T2w.nii.gz"),
    ):
        sources[attr] = _write_dummy_nifti(anat / name)
    return {
        "output_dir": str(tmp_path / "derivatives"),
        "subject_id": "test01",
        "session_id": "01",
        "bids_dir": str(minimal_bids_dir),
        "source_dsc": str(src_dsc),
        **sources,
        "mtt_map": _write_dummy_nifti(tmp_path / "mtt.nii.gz", 4.0),
        "tmax_map": _write_dummy_nifti(tmp_path / "tmax.nii.gz", 1.5),
        "psr_map": _write_dummy_nifti(tmp_path / "psr.nii.gz", 100.0),
        "rel_cbf_map": _write_dummy_nifti(tmp_path / "rcbf.nii.gz", 60.0),
        "rel_cbv_map": _write_dummy_nifti(tmp_path / "rcbv.nii.gz", 2.0),
        "rel_cbv_auc_map": _write_dummy_nifti(tmp_path / "rcbv_auc.nii.gz", 3.0),
        "ncbv_map": _write_dummy_nifti(tmp_path / "ncbv.nii.gz", 1.0),
        "ncbf_map": _write_dummy_nifti(tmp_path / "ncbf.nii.gz", 0.75),
    }


def _load_sidecar(derivatives_root: Path, suffix: str) -> dict:
    """Load the one sidecar matching the given BIDS desc suffix."""
    matches = list(derivatives_root.rglob(f"*{suffix}*.json"))
    assert len(matches) == 1, (
        f"Expected exactly one sidecar matching {suffix!r}; "
        f"found {len(matches)}: {matches}"
    )
    with open(matches[0]) as f:
        return json.load(f)


def _aif_block(sidecar: dict) -> dict:
    return sidecar["PerfusionProcessing"]["AIF"]


def _run_writer(inputs: dict, qc_metrics: dict | None, tmp_path: Path | None = None) -> Path:
    """Build and run the DSC datasinks factory workflow."""
    from pmriprep.workflows.dsc.datasinks import init_dsc_datasinks_wf

    # Forward every per-input key supplied by the test as ``present_inputs``
    # so the factory builds a sink for it.
    present = set(inputs.keys()) - {
        "output_dir", "bids_dir", "subject_id", "session_id", "qc_metrics",
        "smooth_fwhm", "smooth_hemodynamic",
    }
    has_smoothing = "smooth_hemodynamic" in inputs
    deconvolution = (
        qc_metrics.get("deconvolution", {})
        if isinstance(qc_metrics, dict)
        else {}
    )
    wf = init_dsc_datasinks_wf(
        output_dir=inputs["output_dir"],
        bids_dir=inputs["bids_dir"],
        subject_id=inputs["subject_id"],
        session_id=inputs["session_id"],
        deconvolution_method=deconvolution.get("method", "oSVD"),
        present_inputs=present,
        has_smoothing=has_smoothing,
    )
    if tmp_path is not None:
        wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    for k, v in inputs.items():
        if k in {"output_dir", "subject_id", "session_id", "bids_dir"}:
            continue
        setattr(inp, k, v)
    if qc_metrics is not None:
        inp.qc_metrics = qc_metrics
    wf.run()
    return Path(inputs["output_dir"])


def test_cbv_sidecars_report_statistics_from_their_own_images(
    tmp_path: Path, minimal_inputs: dict,
) -> None:
    """Lesion, user-lesion and ROI summaries retain distinct CBV estimators."""
    from pmriprep.workflows.dsc.base import init_dsc_lesion_stats_wf

    def save_image(name: str, data: np.ndarray) -> str:
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), path)
        return str(path)

    shape = (4, 4, 4)
    indices = np.arange(64, dtype=np.float32).reshape(shape)
    residue_cbv = indices + 1
    auc_cbv = 100 + indices**2
    residue_cbv[0, 0, 0] = np.nan
    auc_cbv[3, 3, 3] = np.nan
    nawm = np.zeros(shape, dtype=bool)
    nawm[2:] = True
    normalized_cbv = auc_cbv / np.nanmedian(auc_cbv[nawm])
    images = {"cbv": residue_cbv, "cbv_auc": auc_cbv, "ncbv": normalized_cbv}

    lesion = np.zeros(shape, dtype=np.float32)
    lesion[0] = 2
    lesion[1] = 1
    roi = np.zeros(shape, dtype=bool)
    roi[:, 1, :] = True
    user_lesion = np.zeros(shape, dtype=bool)
    user_lesion[:, 0, :] = True

    inputs = dict(minimal_inputs)
    for name, attr in (
        ("cbv", "rel_cbv_map"), ("cbv_auc", "rel_cbv_auc_map"), ("ncbv", "ncbv_map"),
    ):
        inputs[attr] = save_image(name, images[name])

    stats_wf = init_dsc_lesion_stats_wf(subject_id="test01", session_id="01")
    stats_wf.base_dir = str(tmp_path / "stats_work")
    stats_wf.inputs.inputnode.cbv_map = inputs["rel_cbv_map"]
    stats_wf.inputs.inputnode.cbv_auc_map = inputs["rel_cbv_auc_map"]
    stats_wf.inputs.inputnode.ncbv_map = inputs["ncbv_map"]
    stats_wf.inputs.inputnode.nawm_mask = save_image("nawm", nawm)
    stats_wf.inputs.inputnode.tumor_lesion_mask = save_image("lesion", lesion)
    stats_wf.inputs.inputnode.roi_mask = save_image("roi", roi)
    stats_wf.inputs.inputnode.user_mask = save_image("user_lesion", user_lesion)
    stats_graph = stats_wf.run(plugin="Linear")
    for node in stats_graph.nodes():
        inputs[f"{node.name}_json"] = node.result.outputs.stats_json

    derivatives = _run_writer(inputs, qc_metrics=None, tmp_path=tmp_path)
    selections = (
        ("LesionStats", "contrast_enhancing_tumor", lesion == 2),
        ("LesionStats", "non_enhancing_tumor", lesion == 1),
        ("ROIStats", "roi", roi),
        ("UserLesionStats", "roi", user_lesion),
    )
    for map_key, desc in (
        ("cbv", "dscRelCbv"), ("cbv_auc", "dscRelAUCCbv"), ("ncbv", "dscNawmNormCbv"),
    ):
        image = images[map_key]
        sidecar = _load_sidecar(derivatives, desc)
        for fragment_key, region, mask in selections:
            fragment = sidecar[fragment_key]
            stats = fragment["Regions"][region]["stats"]
            # Regional statistics are serialized to six decimal places.
            assert stats["median"] == round(float(np.nanmedian(image[mask])), 6)
            assert stats["n_valid"] == int(np.count_nonzero(np.isfinite(image[mask])))
            assert stats["n_total"] == int(np.count_nonzero(mask))
            reference = fragment["NAWMReference"]
            assert reference["median"] == round(float(np.nanmedian(image[nawm])), 6)
            assert reference["n_valid"] == int(np.count_nonzero(np.isfinite(image[nawm])))


@pytest.mark.parametrize("method,detection,source", [
    ("sSVD", "atlas_strict", "measured"),
    ("oSVD", "synthetic", "synthetic"),
    ("tikhonov", "atlas_strict", "measured"),
])
def test_svd_mtt_label_and_aif_source(method: str, detection: str, source: str) -> None:
    """The AUC formula applies to both measured and synthetic AIF sources."""
    sidecar = build_dsc_map_sidecar(
        base={}, description="Mean transit time", map_key="mtt", units="s",
        qc_metrics={
            "deconvolution": {"method": method},
            "aif_detection": {"detection_method": detection, "quality_score": 0.8},
        },
    )
    assert sidecar["MTTMethod"] == "full_bolus_auc_cbv_over_deconvolution_cbf"
    assert sidecar["PerfusionProcessing"]["AIF"]["Source"] == source


def test_cpi_mtt_sidecar_reports_residue_integral() -> None:
    """CPI MTT provenance describes its fitted residue integral."""
    cpi_sidecar = build_dsc_map_sidecar(
        base={},
        description="CPI mean transit time",
        map_key="mtt",
        qc_metrics={"deconvolution": {"method": "cpi"}},
        units="s",
    )
    assert cpi_sidecar["MTTMethod"] == "fitted_residue_integral"
    assert "FallbackMTTMethod" not in cpi_sidecar

    mixed_sidecar = build_dsc_map_sidecar(
        base={},
        description="CPI mean transit time",
        map_key="mtt",
        qc_metrics={
            "deconvolution": {"method": "cpi", "n_fallback_osvd": 3}
        },
        units="s",
    )
    assert mixed_sidecar["MTTMethod"] == "fitted_residue_integral"
    assert mixed_sidecar["FallbackMTTMethod"] == (
        "osvd_residue_integral_over_osvd_cbf"
    )
    assert mixed_sidecar["FallbackMask"] == "desc-dscFallback_mask (value 1)"


@pytest.mark.parametrize("nested", [False, True])
def test_tmax_sidecar_reports_fitted_delay_bound_occupancy(nested: bool) -> None:
    """Fitted-delay boundary contact supports flat and production QC layouts."""
    deconvolution = {
        "method": "cpi",
        "delay_at_bound_fraction": 0.125,
    }
    qc_metrics = (
        {"deconvolution": deconvolution} if nested else deconvolution
    )
    sidecar = build_dsc_map_sidecar(
        base={},
        description="Fitted CPI delay",
        map_key="tmax",
        qc_metrics=qc_metrics,
        units="s",
    )

    assert sidecar["FittedDelayAtBoundFraction"] == pytest.approx(0.125)
    assert sidecar["FittedDelayAtBoundFractionDenominator"] == (
        "accepted fitted-delay parametric fits"
    )


def test_arrival_delay_sidecar_reports_unestimated_cpi_timing() -> None:
    """CPI distinguishes unestimated arrival delay from measured zero."""
    sidecar = build_dsc_map_sidecar(
        base={},
        description="CPI arrival delay",
        map_key="arrival_delay",
        qc_metrics={
            "deconvolution": {
                "method": "cpi",
                "arrival_delay_method": "not_estimated",
            },
        },
        units="s",
    )

    assert sidecar["ArrivalDelayMethod"] == "not_estimated"
    assert "FittedDelayAtBoundFraction" not in sidecar


def test_arrival_delay_sidecar_reports_fitted_estimator_only() -> None:
    """Fallback perfusion does not introduce a second delay estimator."""
    sidecar = build_dsc_map_sidecar(
        base={},
        description="AIF-relative fitted arrival delay",
        map_key="arrival_delay",
        qc_metrics={
            "deconvolution": {
                "method": "cpi",
                "arrival_delay_method": "fitted_aif_relative_arrival_delay",
                "fallback_rate": 0.125,
                "delay_at_bound_fraction": 0.01,
                "arrival_delay_finite_voxels": 875,
                "arrival_delay_finite_fraction": 0.875,
            },
        },
        units="s",
    )
    assert sidecar["ArrivalDelayMethod"] == "fitted_aif_relative_arrival_delay"
    assert sidecar["ArrivalDelayFiniteVoxelCount"] == 875
    assert sidecar["ArrivalDelayFiniteVoxelFraction"] == pytest.approx(0.875)
    assert sidecar["ArrivalDelayCoverageDenominator"] == "DSC analysis mask"
    assert sidecar["FittedDelayAtBoundFraction"] == pytest.approx(0.01)
    assert "TmaxMethod" not in sidecar


def test_cpi_timing_and_uncertainty_derivatives(
    minimal_inputs: dict, tmp_path: Path,
) -> None:
    """CPI writes fitted arrival delay, posterior CBF SD and parametric fit quality."""
    minimal_inputs.update({
        "tmax_map": _write_dummy_nifti(tmp_path / "arrival_delay.nii.gz", -0.5),
        "cbf_std_map": _write_dummy_nifti(tmp_path / "cbf_std.nii.gz", 2.0),
        "parametric_r2_map": _write_dummy_nifti(tmp_path / "parametric_r2.nii.gz", 0.875),
        "config_dict": {"inference": "svb"},
    })
    derivatives = _run_writer(
        minimal_inputs,
        {
            "deconvolution": {
                "method": "cpi",
                "arrival_delay_method": "fitted_aif_relative_arrival_delay",
                "fallback_rate": 0.0,
            },
        },
        tmp_path,
    )
    assert len(list(derivatives.rglob("*dscArrivalDelay_stat.nii.gz"))) == 1
    assert not list(derivatives.rglob("*dscTmax_stat.nii.gz"))
    sidecar = _load_sidecar(derivatives, "dscArrivalDelay_stat")
    assert sidecar["ArrivalDelayMethod"] == "fitted_aif_relative_arrival_delay"
    assert "NaN" in sidecar["RejectedFitValues"]

    for desc, value, units in (
        ("dscArrivalDelay", -0.5, "s"),
        ("dscPosteriorSDCbf", 2.0, "1/min"),
        ("dscParametricRsquared", 0.875, "1"),
    ):
        image = next(derivatives.rglob(f"*desc-{desc}_stat.nii.gz"))
        np.testing.assert_array_equal(nib.load(image).get_fdata(), value)
        assert _load_sidecar(derivatives, f"{desc}_stat")["Units"] == units

    r2_meta = _load_sidecar(derivatives, "dscParametricRsquared_stat")
    assert r2_meta["AIFDependent"] is True

    uncertainty_meta = _load_sidecar(derivatives, "dscPosteriorSDCbf_stat")
    assert uncertainty_meta["EstimationMethod"] == "svb"
    assert uncertainty_meta["InferenceMethod"] == "spatial_variational_bayes"
    assert uncertainty_meta["SpatialPriorApplied"] is True
    assert uncertainty_meta["SpatialPriorModel"] == "Penny M-prior"
    assert (
        uncertainty_meta["SpatialUpdateScheme"]
        == "synchronous_batched_backtracking"
    )
    assert uncertainty_meta["SpatiallyRegularizedParameters"] == ["cbf"]

    assert uncertainty_meta["Description"] == "CBF posterior standard deviation"


@pytest.mark.parametrize("map_key,nested", [
    ("cbv", True), ("ncbv", True), ("rel_cbv", True), ("mtt", True),
    ("cbv", False),
])
def test_cpi_volume_and_transit_sidecars_report_spacing_bound_occupancy(
    nested: bool,
    map_key: str,
) -> None:
    """CPI spacing rails accompany residue-derived volume and transit maps."""
    deconvolution = {
        "method": "cpi",
        "cpi_spacing_at_bound_fraction": 0.125,
    }
    qc_metrics = (
        {"deconvolution": deconvolution} if nested else deconvolution
    )

    sidecar = build_dsc_map_sidecar(
        base={},
        description="CPI residue-derived map",
        map_key=map_key,
        qc_metrics=qc_metrics,
        units="s" if map_key == "mtt" else "arbitrary",
    )

    assert sidecar["CPIControlPointSpacingAtBoundFraction"] == pytest.approx(
        0.125
    )
    assert sidecar["CPIControlPointSpacingAtBoundFractionDenominator"] == (
        "accepted CPI fits with inferred control-point positions"
    )


@pytest.mark.parametrize("map_key", ["cbf", "cbv_auc", "tmax"])
def test_cpi_spacing_bound_occupancy_omits_unrelated_sidecars(
    map_key: str,
) -> None:
    """Do not imply that CPI spacing directly parameterizes unrelated maps."""
    sidecar = build_dsc_map_sidecar(
        base={},
        description="Unrelated CPI map",
        map_key=map_key,
        qc_metrics={
            "deconvolution": {
                "method": "cpi",
                "cpi_spacing_at_bound_fraction": 0.125,
            }
        },
        units="arbitrary",
    )

    assert "CPIControlPointSpacingAtBoundFraction" not in sidecar


def test_dsc_writer_recreates_deleted_cached_derivative(
    minimal_inputs: dict, tmp_path: Path,
) -> None:
    """External derivatives are rewritten even when the node cache survives."""
    from pmriprep.workflows.dsc.datasinks import init_dsc_datasinks_wf

    wf = init_dsc_datasinks_wf(
        output_dir=minimal_inputs["output_dir"],
        bids_dir=minimal_inputs["bids_dir"],
        subject_id=minimal_inputs["subject_id"],
        session_id=minimal_inputs["session_id"],
        present_inputs={"mtt_map"},
    )
    wf.base_dir = str(tmp_path / "work-recreate")
    inp = wf.inputs.inputnode
    inp.source_dsc = minimal_inputs["source_dsc"]
    for name in ("mtt_map", "tmax_map", "psr_map"):
        setattr(inp, name, minimal_inputs[name])
    inp.qc_metrics = {}

    wf.run()
    perf = (
        Path(minimal_inputs["output_dir"]) / "pmriprep"
        / "sub-test01" / "ses-01" / "perf"
    )
    nii = perf / "sub-test01_ses-01_desc-dscMtt_stat.nii.gz"
    sidecar = nii.with_suffix("").with_suffix(".json")
    assert nii.exists() and sidecar.exists()

    nii.unlink()
    sidecar.unlink()
    wf.run()

    assert nii.exists() and sidecar.exists()


def test_parametric_provenance_carries_fit_health_denominators() -> None:
    """Fallback sidecars distinguish requests, successes, and denominators."""
    from pmriprep.workflows.dsc.datasinks import _build_parametric_extra

    extra = _build_parametric_extra(
        {
            "deconvolution": {
                "method": "cpi",
                "inference": "vi",
                "n_mask_voxels": 100,
                "n_fit_admissible": 80,
                "n_fit_inadmissible": 20,
                "n_fallback_requested": 12,
                "n_fallback_attempted": 12,
                "n_fallback_osvd": 10,
                "n_fallback_failed": 2,
                "fallback_requested_rate": 0.15,
                "fallback_rate": 0.125,
            }
        }
    )

    assert extra["FitAdmissibleVoxelCount"] == 80
    assert extra["FitInadmissibleVoxelCount"] == 20
    assert extra["FallbackRequestedVoxelCount"] == 12
    assert extra["FallbackSucceededVoxelCount"] == 10
    assert extra["FallbackFailedVoxelCount"] == 2
    assert extra["FallbackRequestedFraction"] == pytest.approx(0.15)
    assert extra["FallbackSucceededFraction"] == pytest.approx(0.125)
    assert extra["FallbackMinimumRSquared"] == pytest.approx(0.0)
    assert extra["FallbackMinimumRSquaredInclusive"] is True


# =============================================================================
# AIF block derivation (pure function tests)
# =============================================================================


class TestBuildAifProvenanceBlock:
    def test_measured_atlas_strict(self) -> None:
        block = build_aif_provenance_block(
            {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.85}}
        )
        assert block["Source"] == "measured"
        assert block["DetectionLevel"] == "atlas_strict"
        assert block["QualityFlag"] == "pass"
        assert block["GammaFitApplied"] is False

    def test_synthetic_template(self) -> None:
        block = build_aif_provenance_block({
            "aif_detection": {"detection_method": "synthetic", "quality_score": 0.2},
            "synthetic_fallback_reason": "no_candidates",
        })
        assert block["Source"] == "synthetic"
        assert block["QualityFlag"] == "poor"
        assert block["SyntheticFallbackReason"] == "no_candidates"

    def test_external_aif(self) -> None:
        block = build_aif_provenance_block(
            {"aif_detection": {"detection_method": "external", "quality_score": 0.9}}
        )
        assert block["Source"] == "external"
        assert block["ExternalAIFVerified"] is False

    def test_gamma_fit_applied(self) -> None:
        block = build_aif_provenance_block(
            {"aif_detection": {"detection_method": "atlas_relaxed",
                                "quality_score": 0.72, "gamma_fit_applied": True}}
        )
        assert block["Source"] == "gamma_fit_measured"
        assert block["GammaFitApplied"] is True

    def test_quality_warn_for_permissive_tier(self) -> None:
        block = build_aif_provenance_block(
            {"aif_detection": {
                "detection_method": "atlas_permissive", "quality_score": 0.6,
            }}
        )
        assert block["QualityFlag"] == "warn"

    def test_strict_tier_not_downgraded_by_ranking_score(self) -> None:
        block = build_aif_provenance_block(
            {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.3}}
        )
        assert block["QualityFlag"] == "pass"

    def test_missing_qc_returns_empty(self) -> None:
        assert build_aif_provenance_block(None) == {}
        assert build_aif_provenance_block({}) == {}
        assert build_aif_provenance_block({"aif_detection": {}}) == {}

    def test_compound_union_two_tiers_accepted(self) -> None:
        """Cascade emits ``"atlas_relaxed+atlas_permissive"`` when the
        union-pool fallback accepts voxels from multiple tiers; the
        validator must accept compound strings whose every part is in
        the enum."""
        block = build_aif_provenance_block({
            "aif_detection": {
                "detection_method": "atlas_relaxed+atlas_permissive",
                "quality_score": 0.7,
            },
        })
        assert block != {}
        assert block["DetectionLevel"] == "atlas_relaxed+atlas_permissive"
        assert block["Source"] == "measured"

    def test_compound_with_unknown_part_rejected(self) -> None:
        """Compound strings are accepted only when EVERY part is a known
        cascade tier; an unknown part rejects the whole union."""
        block = build_aif_provenance_block({
            "aif_detection": {
                "detection_method": "atlas_relaxed+nonsense",
                "quality_score": 0.5,
            },
        })
        assert block == {}

    def test_compound_with_synthetic_part_accepted(self) -> None:
        """The cascade shouldn't produce ``"...+synthetic"`` (synthetic
        is a terminal fallback, not a contributing tier), but the
        validator should accept syntactically valid compounds defensively
        rather than skipping the whole provenance block."""
        block = build_aif_provenance_block({
            "aif_detection": {
                "detection_method": "atlas_relaxed+synthetic",
                "quality_score": 0.4,
            },
        })
        assert block != {}
        assert block["DetectionLevel"] == "atlas_relaxed+synthetic"


class TestApplyAifSidecar:
    def test_stamps_dependent_and_block(self) -> None:
        sidecar: dict = {}
        apply_aif_sidecar(
            sidecar, "cbf",
            {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.8}},
        )
        assert sidecar["AIFDependent"] is True
        assert sidecar["PerfusionProcessing"]["AIF"]["Source"] == "measured"

    def test_independent_map_marked(self) -> None:
        sidecar: dict = {}
        apply_aif_sidecar(
            sidecar, "psr",
            {"aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.9}},
        )
        assert sidecar["AIFDependent"] is False

    @pytest.mark.parametrize("map_key", ["parametric_r2", "fallback"])
    def test_parametric_diagnostics_are_aif_dependent(self, map_key: str) -> None:
        sidecar: dict = {}
        apply_aif_sidecar(
            sidecar,
            map_key,
            {"aif_detection": {"detection_method": "synthetic"}},
        )
        assert sidecar["AIFDependent"] is True

    def test_unknown_map_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown DSC map key"):
            apply_aif_sidecar({}, "not_a_real_map", None)

    def test_missing_qc_stamps_tags_only(self) -> None:
        sidecar: dict = {}
        apply_aif_sidecar(sidecar, "cbf", None)
        assert sidecar["AIFDependent"] is True
        assert "PerfusionProcessing" not in sidecar


# =============================================================================
# Drift guard
# =============================================================================


class TestAifMapRegistryCoverage:
    """Every map_key the factory might call apply_aif_sidecar with must be
    registered in DSC_AIF_MAP_KEYS; otherwise the workflow raises
    KeyError at runtime on a real pipeline run.
    """

    def test_factory_keys_subset_of_registered(self) -> None:
        from pmriprep.workflows.dsc.datasinks import _DSC_OUTPUT_KEYS

        missing = _DSC_OUTPUT_KEYS - DSC_AIF_MAP_KEYS
        assert not missing, (
            f"Factory keys not registered in DSC_AIF_MAP_KEYS: {missing}."
        )

    def test_aif_independent_subset_of_registered(self) -> None:
        extra = AIF_INDEPENDENT_MAPS - DSC_AIF_MAP_KEYS
        assert not extra, (
            f"AIF_INDEPENDENT_MAPS contains keys not in DSC_AIF_MAP_KEYS: "
            f"{extra}"
        )


# =============================================================================
# End-to-end factory integration
# =============================================================================


class TestWriterAifBlockPresentOnAllMaps:
    """Running the full factory with a measured-AIF qc_metrics dict stamps
    PerfusionProcessing.AIF and AIFDependent on every
    emitted sidecar."""

    def test_measured_maps_and_quantitative_metadata(
        self, minimal_inputs: dict, tmp_path: Path,
    ) -> None:
        """Measured-AIF outputs preserve map contents, units and estimator-specific QC."""
        minimal_inputs.update({
            "k2_map": _write_dummy_nifti(tmp_path / "k2.nii.gz", -0.03125),
            "cnr_map": _write_dummy_nifti(tmp_path / "cnr.nii.gz", 5.0),
            "tsnr_map": _write_dummy_nifti(tmp_path / "tsnr.nii.gz", 40.0),
        })
        qc = {
            "aif_detection": {"detection_method": "atlas_strict", "quality_score": 0.85},
            "normalization_factor": 0.023,
            "cbf_normalization_factor": 42.0,
            "deconvolution": {
                "method": "oSVD",
                "cbv_auc_finite_voxels": 100,
                "cbv_auc_negative_voxels": 7,
                "cbv_auc_negative_fraction": 0.07,
            },
        }
        derivatives = _run_writer(minimal_inputs, qc, tmp_path)
        outputs = {
            "dscMtt": (4.0, "s", True),
            "dscTmax": (1.5, "s", True),
            "dscPsr": (100.0, "%", False),
            "dscRelCbf": (60.0, "1/min", True),
            "dscRelCbv": (2.0, "arbitrary", True),
            "dscRelAUCCbv": (3.0, "arbitrary", True),
            "dscNawmNormCbv": (1.0, "1", True),
            "dscNawmNormCbf": (0.75, "1", True),
            "dscK2": (-0.03125, "1/s", False),
            "dscCnr": (5.0, "1", False),
            "dscTsnr": (40.0, "1", False),
        }
        for desc, (value, units, aif_dependent) in outputs.items():
            sidecar = _load_sidecar(derivatives, f"{desc}_stat")
            assert _aif_block(sidecar)["Source"] == "measured", desc
            assert _aif_block(sidecar)["QualityFlag"] == "pass", desc
            assert sidecar["AIFDependent"] is aif_dependent, desc
            assert sidecar["Units"] == units, desc
            image = next(derivatives.rglob(f"*desc-{desc}_stat.nii.gz"))
            np.testing.assert_array_equal(nib.load(image).get_fdata(), value)

        assert _load_sidecar(derivatives, "dscK2_stat")["SignedCoefficient"] is True
        assert _load_sidecar(derivatives, "dscCnr_stat")["Estimator"] == "(S0 - Smin) / sigma0"
        assert _load_sidecar(derivatives, "dscTsnr_stat")["TemporalWindow"] == (
            "detected pre-bolus volumes only"
        )

        tmax_meta = _load_sidecar(derivatives, "dscTmax_stat")
        assert tmax_meta["Clipping"] == "none"
        assert "oSVD" in tmax_meta["NegativeValues"]
        assert "parametric" not in tmax_meta["NegativeValues"]
        assert "unlike tissue concentration TTP" in tmax_meta["TimingConvention"]

        mtt_meta = _load_sidecar(derivatives, "dscMtt_stat")
        assert mtt_meta["PostFitClipping"] == "none"
        assert mtt_meta["MTTMethod"] == "full_bolus_auc_cbv_over_deconvolution_cbf"
        assert "approaches zero" in mtt_meta["LowFlowCaution"]
        for suffix in ("dscRelCbf_stat", "dscRelCbv_stat", "dscRelAUCCbv_stat"):
            meta = _load_sidecar(derivatives, suffix)
            assert meta["PostFitClipping"] == "none"
            assert "negative/non-finite" in meta["UndefinedValue"]

        ncbv = _load_sidecar(derivatives, "dscNawmNormCbv_stat")
        ncbf = _load_sidecar(derivatives, "dscNawmNormCbf_stat")
        raw_cbf = _load_sidecar(derivatives, "dscRelCbf_stat")

        cbv_ref = ncbv["PerfusionProcessing"]["NormalizationReference"]
        cbf_ref = ncbf["PerfusionProcessing"]["NormalizationReference"]
        assert cbv_ref["parameter"] == "CBV"
        assert cbv_ref["factor"] == pytest.approx(0.023)
        assert cbf_ref["parameter"] == "CBF"
        assert cbf_ref["factor"] == pytest.approx(42.0)
        assert "NormalizationReference" not in raw_cbf.get("PerfusionProcessing", {})

        auc_signed = _load_sidecar(
            derivatives, "dscRelAUCCbv_stat"
        )["SignedEstimator"]
        ncbv_signed = _load_sidecar(
            derivatives, "dscNawmNormCbv_stat"
        )["SignedEstimator"]
        assert "NormalizationReferenceDomain" not in auc_signed
        assert ncbv_signed["NormalizationReferenceDomain"] == (
            "finite_nonnegative_NAWM_with_positive_median_required"
        )

        for signed in (auc_signed, ncbv_signed):
            assert signed["ValueDomain"] == "nonnegative_or_NaN"
            assert signed["NegativeValuesRetained"] is False
            assert signed["NegativeValueHandling"] == "encoded_as_NaN"
            assert signed["FiniteInBrainVoxelCount"] == 100
            assert signed["NegativeInBrainVoxelCount"] == 7
            assert signed["NegativeInBrainVoxelFraction"] == pytest.approx(0.07)
            assert (
                signed["NegativeFractionDenominator"]
                == "finite_pre_sanitization_in_brain_voxels"
            )
            assert signed["NegativeFractionSource"] == (
                "pre_sanitization_AUC_CBV_estimator"
            )

    def test_k2_sink_is_conditional(self, minimal_inputs) -> None:
        from pmriprep.workflows.dsc.datasinks import init_dsc_datasinks_wf

        workflow = init_dsc_datasinks_wf(
            output_dir=minimal_inputs["output_dir"],
            bids_dir=minimal_inputs["bids_dir"],
            subject_id=minimal_inputs["subject_id"],
            session_id=minimal_inputs["session_id"],
            present_inputs=set(),
        )

        assert "k2_map_sink" not in workflow.list_node_names()

    def test_synthetic_template_flags(self, minimal_inputs, tmp_path) -> None:
        minimal_inputs["smooth_fwhm"] = 3.0
        minimal_inputs["smooth_hemodynamic"] = {
            "rel_cbv": _write_dummy_nifti(tmp_path / "rcbv_sm.nii.gz", 2.5),
            "ncbv": _write_dummy_nifti(tmp_path / "ncbv_sm.nii.gz", 1.25),
        }
        qc = {
            "aif_detection": {"detection_method": "synthetic", "quality_score": 0.3},
            "synthetic_fallback_reason": "no_candidates",
            "deconvolution": {
                "cbv_auc_finite_voxels": 100,
                "cbv_auc_negative_voxels": 7,
                "cbv_auc_negative_fraction": 0.07,
            },
        }
        derivatives = _run_writer(minimal_inputs, qc, tmp_path)
        cbv = _load_sidecar(derivatives, "dscRelAUCCbv_stat")
        assert _aif_block(cbv)["Source"] == "synthetic"
        assert _aif_block(cbv)["SyntheticFallbackReason"] == "no_candidates"
        assert cbv["AIFDependent"] is True

        for suffix in ("dscMtt_stat", "dscNawmNormCbf_stat"):
            sidecar = _load_sidecar(derivatives, suffix)
            assert sidecar["AIFDependent"] is True
            assert _aif_block(sidecar)["Source"] == "synthetic"

        assert cbv["SignedEstimator"]["NegativeInBrainVoxelFraction"] == pytest.approx(0.07)
        for desc, value in (("dscRelAUCSm3Cbv", 2.5), ("dscNawmNormSm3Cbv", 1.25)):
            sidecar = _load_sidecar(derivatives, f"{desc}_stat")
            assert _aif_block(sidecar)["Source"] == "synthetic"
            assert sidecar["SmoothingSupport"] == "finite source voxels within the brain mask"
            assert sidecar["InvalidSupportPreserved"] is True
            signed = sidecar["SignedEstimator"]
            assert signed["ValueDomain"] == "nonnegative_or_NaN"
            assert signed["NegativeValuesRetained"] is False
            assert "NegativeInBrainVoxelFraction" not in signed
            image = next(derivatives.rglob(f"*desc-{desc}_stat.nii.gz"))
            np.testing.assert_array_equal(nib.load(image).get_fdata(), value)

    def test_gamma_fit_measured_source(self, minimal_inputs, tmp_path) -> None:
        qc = {"aif_detection": {
            "detection_method": "atlas_relaxed",
            "quality_score": 0.72,
            "gamma_fit_applied": True,
        }}
        derivatives = _run_writer(minimal_inputs, qc, tmp_path)
        cbv = _load_sidecar(derivatives, "dscRelAUCCbv_stat")
        assert _aif_block(cbv)["Source"] == "gamma_fit_measured"
        assert _aif_block(cbv)["GammaFitApplied"] is True


class TestMissingAifQc:
    def test_missing_qc_stamps_tags_but_no_block(self, minimal_inputs, tmp_path) -> None:
        """Missing QC leaves the AIFDependent flag as the available AIF metadata."""
        derivatives = _run_writer(minimal_inputs, None, tmp_path)
        sidecar = _load_sidecar(derivatives, "dscRelAUCCbv_stat")
        assert sidecar["AIFDependent"] is True
        assert "AIF" not in sidecar.get("PerfusionProcessing", {})


# =============================================================================
# Structural underlays: T1w/FLAIR/T2w in DSC space must land in anat/, not perf/
# =============================================================================


class TestStructuralUnderlayDatatype:
    """Regression: T1w-in-DSC sink must write to anat/ (not perf/).

    niworkflows derives ``datatype`` from ``source_file``; with
    ``source_dsc`` (a perf-datatype file) the default would be ``perf``,
    which makes pybids reject the path because BIDS does not allow ``T1w``
    suffix in ``perf/``. ``_add_anat_sink`` must override datatype="anat".
    """

    def _build_and_run(
        self, minimal_inputs: dict, tmp_path: Path, work_subdir: str,
    ) -> Path:
        """Write optional underlays and return their session directory."""
        output = _run_writer(minimal_inputs, None, tmp_path / work_subdir)
        return output / "pmriprep" / "sub-test01" / "ses-01"

    def test_both_t1w_underlays_in_dsc(self, minimal_inputs, tmp_path) -> None:
        """Both underlays configured: two anat/ outputs differing by ce- entity."""
        minimal_inputs["t1w_in_dsc"] = _write_dummy_nifti(tmp_path / "t1w_in_dsc.nii.gz", 10.0)
        minimal_inputs["t1w_postcon_in_dsc"] = _write_dummy_nifti(
            tmp_path / "t1w_postcon_in_dsc.nii.gz", 20.0,
        )
        ses = self._build_and_run(minimal_inputs, tmp_path, "work-both")

        anat_dir = ses / "anat"
        precon = anat_dir / "sub-test01_ses-01_acq-mprage_run-1_space-dscref_T1w.nii.gz"
        postcon = anat_dir / "sub-test01_ses-01_acq-mprage_ce-GAD_run-2_space-dscref_T1w.nii.gz"
        assert precon.exists() and postcon.exists(), (
            f"both underlays expected; anat/ contains: "
            f"{sorted(p.name for p in anat_dir.iterdir())}"
        )
        for image, value, source_name, contrast in (
            (precon, 10.0, "sub-test01_ses-01_acq-mprage_run-1_T1w.nii.gz", "Pre-contrast"),
            (postcon, 20.0, "sub-test01_ses-01_acq-mprage_ce-GAD_run-2_T1w.nii.gz", "Post-contrast"),
        ):
            np.testing.assert_array_equal(nib.load(image).get_fdata(), value)
            meta = json.loads(image.with_suffix("").with_suffix(".json").read_text())
            assert contrast in meta["Description"]
            assert meta["Sources"] == [
                f"bids:raw:sub-test01/ses-01/anat/{source_name}",
                "bids:raw:sub-test01/ses-01/perf/sub-test01_ses-01_dsc.nii.gz",
            ]
        assert not list((ses / "perf").glob("*T1w.nii.gz"))

    def test_t1w_subtraction_in_dsc_lands_in_anat(self, minimal_inputs, tmp_path) -> None:
        """Subtraction underlay: anat/ with desc-subtraction, no ce- entity."""
        minimal_inputs["t1w_subtraction_in_dsc"] = _write_dummy_nifti(
            tmp_path / "t1w_subtraction_in_dsc.nii.gz",
        )
        ses = self._build_and_run(minimal_inputs, tmp_path, "work-subtraction")

        anat_dir = ses / "anat"
        expected = anat_dir / "sub-test01_ses-01_acq-mprage_run-1_space-dscref_desc-subtraction_T1w.nii.gz"
        assert expected.exists(), (
            f"subtraction underlay missing at {expected}; "
            f"anat/ contains: {sorted(p.name for p in anat_dir.iterdir())}"
        )
        # No ce- entity, and distinct from the plain precon underlay.
        assert not (anat_dir / "sub-test01_ses-01_ce-GAD_space-dscref_desc-subtraction_T1w.nii.gz").exists()
        assert not (anat_dir / "sub-test01_ses-01_space-dscref_T1w.nii.gz").exists()
        # Custom space => SpatialReference is required and present.
        meta = json.loads(expected.with_suffix("").with_suffix(".json").read_text())
        assert "subtraction" in meta["Description"].lower()
        assert "SpatialReference" in meta
        assert meta["IntensityNormalizationApplied"] is False
        assert meta["Quantitative"] is False
        assert meta["Units"] == "arbitrary"
        assert len(meta["Sources"]) == 3

    def test_optional_underlays_absent(self, minimal_inputs, tmp_path) -> None:
        """A writer run without registered anatomical inputs emits only perfusion maps."""
        ses = self._build_and_run(minimal_inputs, tmp_path, "work-sub-absent")
        anat_dir = ses / "anat"
        subtraction = (
            list(anat_dir.glob("*desc-subtraction*")) if anat_dir.exists() else []
        )
        assert not subtraction, f"subtraction must not be emitted; got {subtraction}"
        assert not list(anat_dir.glob("*.nii.gz"))

    def test_both_flair_underlays_in_dsc(self, minimal_inputs, tmp_path) -> None:
        """Both FLAIR underlays configured: two anat/ outputs differing by ce- entity."""
        minimal_inputs["flair_in_dsc"] = _write_dummy_nifti(tmp_path / "flair_in_dsc.nii.gz", 10.0)
        minimal_inputs["flair_postcon_in_dsc"] = _write_dummy_nifti(
            tmp_path / "flair_postcon_in_dsc.nii.gz", 20.0,
        )
        ses = self._build_and_run(minimal_inputs, tmp_path, "work-flair-both")

        anat_dir = ses / "anat"
        precon = anat_dir / "sub-test01_ses-01_acq-3d_run-1_space-dscref_FLAIR.nii.gz"
        postcon = anat_dir / "sub-test01_ses-01_acq-3d_ce-GAD_run-2_space-dscref_FLAIR.nii.gz"
        assert precon.exists() and postcon.exists(), (
            f"both FLAIR underlays expected; anat/ contains: "
            f"{sorted(p.name for p in anat_dir.iterdir())}"
        )

        for image, value, source_name, description in (
            (precon, 10.0, "sub-test01_ses-01_acq-3d_run-1_FLAIR.nii.gz",
             "FLAIR registered to DSC space"),
            (postcon, 20.0, "sub-test01_ses-01_acq-3d_ce-GAD_run-2_FLAIR.nii.gz",
             "Post-contrast FLAIR registered to DSC space"),
        ):
            np.testing.assert_array_equal(nib.load(image).get_fdata(), value)
            meta = json.loads(image.with_suffix("").with_suffix(".json").read_text())
            assert meta["Description"] == description
            assert meta["Sources"] == [
                f"bids:raw:sub-test01/ses-01/anat/{source_name}",
                "bids:raw:sub-test01/ses-01/perf/sub-test01_ses-01_dsc.nii.gz",
            ]
        assert not list((ses / "perf").glob("*FLAIR.nii.gz"))
