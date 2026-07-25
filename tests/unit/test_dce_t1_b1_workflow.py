"""Construction contracts for DCE T1/B1 preparation."""

from pathlib import Path

from pmriprep.config.dce import DCEConfig
from pmriprep.workflows.dce.base import (
    init_dce_anatomical_masks_wf,
    init_dce_brain_extraction_wf,
    init_dce_dsc_resampling_wf,
    init_dce_dsc_validation_wf,
    init_dce_lesion_masks_wf,
    init_dce_atlas_transform_wf,
    init_dce_native_input_wf,
    init_dce_preprocessing_wf,
    init_dce_qc_collection_wf,
    init_dce_registration_wf,
    init_dce_underlays_wf,
    init_dce_t1_b1_wf,
)


def _names(config: DCEConfig) -> set[str]:
    workflow = init_dce_t1_b1_wf(
        config=config, subject_id="0001", session_id="0001",
    )
    return set(workflow.list_node_names())


def test_t1_b1_workflow_vfa_mode() -> None:
    """VFA mode constructs the complete B1-corrected T1 chain."""
    assert _names(DCEConfig()) == {
        "inputnode", "outputnode", "b1_mapping", "vfa_reorient",
        "vfa_preprocess", "transform_b1_to_vfa", "t1_mapping",
    }


def test_t1_b1_workflow_external_t1_mode() -> None:
    """External T1 retains B1 preparation without constructing VFA fitting."""
    config = DCEConfig(t1_map_file=Path("external_t1map.nii.gz"))
    assert _names(config) == {"inputnode", "outputnode", "b1_mapping"}


def test_t1_b1_workflow_no_t1_mode() -> None:
    """No-T1 mode retains B1 correction for DCE signal conversion."""
    assert _names(DCEConfig(no_t1=True)) == {
        "inputnode", "outputnode", "b1_mapping",
    }


def test_t1_b1_workflow_no_t1_no_b1_mode() -> None:
    """Explicit B1 opt-out leaves the no-T1 preparation contract empty."""
    assert _names(DCEConfig(no_t1=True, b1_method="none")) == {
        "inputnode", "outputnode",
    }


def test_dce_native_input_workflow_contract() -> None:
    """Native input preparation retains reorientation and early-reference nodes."""
    workflow = init_dce_native_input_wf()
    assert set(workflow.list_node_names()) == {
        "inputnode", "outputnode", "dce_reorient", "dce_early_mean",
    }
    assert set(workflow.get_node("outputnode").interface._fields) == {
        "dce_file", "early_mean",
    }


def test_dce_brain_extraction_workflow_contract() -> None:
    """Brain extraction exposes the complete native-space result contract."""
    workflow = init_dce_brain_extraction_wf(config=DCEConfig(), mem_gb=2.0)
    assert set(workflow.list_node_names()) == {
        "inputnode", "outputnode", "dce_brain_extract",
    }
    assert set(workflow.get_node("outputnode").interface._fields) == {
        "out_brain", "out_mask", "method_used", "qc_metrics",
    }


def test_dce_preprocessing_workflow_contract() -> None:
    """Preprocessing exposes every motion/baseline result consumed downstream."""
    workflow = init_dce_preprocessing_wf(config=DCEConfig(), mem_gb=6.0)
    assert set(workflow.list_node_names()) == {
        "inputnode", "outputnode", "dce_preprocess",
    }
    assert set(workflow.get_node("outputnode").interface._fields) == {
        "out_file", "out_params", "mean_img", "baseline_mean_img",
        "baseline_frames", "bolus_arrival_frame", "fd", "qc_metrics",
    }


def test_dce_registration_workflow_contract() -> None:
    """Registration exposes transforms, registered image, and QC provenance."""
    workflow = init_dce_registration_wf(mem_gb=2.0)
    assert set(workflow.list_node_names()) == {
        "inputnode", "outputnode", "dce_registration",
    }
    assert set(workflow.get_node("outputnode").interface._fields) == {
        "out_file", "out_matrix", "out_h5", "qc_metrics",
    }


def test_dce_atlas_transform_workflow_contract() -> None:
    """Atlas preparation preserves the ordered transformed-mask list contract."""
    workflow = init_dce_atlas_transform_wf()
    assert set(workflow.list_node_names()) == {
        "inputnode", "outputnode", "transform_atlas_cascade",
    }
    assert workflow.get_node("outputnode").interface._fields == ["out_files"]


def test_dce_anatomical_masks_workflow_contract() -> None:
    """Anatomical masks retain distinct binary and probability transforms."""
    workflow = init_dce_anatomical_masks_wf(has_tissue_seg=True)
    assert set(workflow.list_node_names()) == {
        "inputnode", "outputnode", "transform_brain_mask",
        "transform_csf_probseg", "transform_wm_probseg", "transform_gm_probseg",
    }
    no_tissue = init_dce_anatomical_masks_wf(has_tissue_seg=False)
    assert set(no_tissue.list_node_names()) == {
        "inputnode", "outputnode", "transform_brain_mask",
    }


def test_dce_dsc_resampling_workflow_contract() -> None:
    """DSC resampling preserves distinct relative-CBV and nCBV paths."""
    workflow = init_dce_dsc_resampling_wf()
    assert set(workflow.list_node_names()) == {
        "inputnode", "outputnode", "resample_dsc_cbv", "resample_dsc_ncbv",
    }
    assert set(workflow.get_node("outputnode").interface._fields) == {
        "dsc_cbv_file", "dsc_ncbv_file",
    }


def test_dce_dsc_validation_workflow_contract() -> None:
    """Cross-validation exposes its quantitative result and plot."""
    workflow = init_dce_dsc_validation_wf(
        config=DCEConfig(), subject_id="0001", session_id="0001",
    )
    assert set(workflow.list_node_names()) == {
        "inputnode", "outputnode", "dsc_dce_validation", "dsc_dce_xval_plot",
    }
    assert set(workflow.get_node("outputnode").interface._fields) == {
        "validation_result", "plot_file",
    }


def test_dce_underlays_workflow_contract() -> None:
    """Underlay construction remains independently conditional per modality."""
    empty = init_dce_underlays_wf(config=DCEConfig())
    assert set(empty.list_node_names()) == {"inputnode", "outputnode"}
    full = init_dce_underlays_wf(
        config=DCEConfig(underlays=[
            "t1w_precon", "t1w_postcon", "t1w_subtraction",
            "flair_precon", "flair_postcon", "t2w",
        ])
    )
    assert set(full.list_node_names()) == {
        "inputnode", "outputnode", "transform_t1w_underlay",
        "transform_t1w_postcon_underlay", "transform_t1w_subtraction_underlay",
        "transform_flair_underlay", "transform_flair_postcon_underlay",
        "transform_t2w_underlay",
    }


def test_dce_qc_collection_workflow_contract() -> None:
    """QC aggregation exposes one collected metrics dictionary."""
    workflow = init_dce_qc_collection_wf(config=DCEConfig())
    assert set(workflow.list_node_names()) == {
        "inputnode", "outputnode", "collect_qc_metrics",
    }
    assert workflow.get_node("outputnode").interface._fields == ["qc_metrics"]


def test_dce_lesion_masks_workflow_contract() -> None:
    """Lesion preparation preserves validation, transform, and priority nodes."""
    workflow = init_dce_lesion_masks_wf()
    assert set(workflow.list_node_names()) == {
        "inputnode", "outputnode", "transform_nawm_mask", "validate_roi_mask",
        "validate_user_lesion_mask", "transform_non_enhancing_mask",
        "transform_enhancing_mask", "transform_lesion", "select_aif_exclusion",
        "combine_masks", "transform_stats_lesion",
    }
    assert set(workflow.get_node("outputnode").interface._fields) == {
        "nawm_mask", "roi_mask", "user_lesion_mask", "enhancing_mask",
        "non_enhancing_mask", "aif_exclusion_mask", "lesion_mask", "mask_origin",
    }
