"""Integration tests for synthetic-AIF SKIP plumbing.

Exercises the plumbing that carries the synthetic-AIF signal from the
DSC AIF cascade through to downstream consumers (DCE vp constraint,
DSC-DCE validation, sidecar writers). Complements the per-module unit
tests by asserting end-to-end flag propagation.
"""
from __future__ import annotations


def test_build_aif_block_measured() -> None:
    from pmriprep.utils.aif_provenance import build_aif_provenance_block

    block = build_aif_provenance_block(
        {"aif_detection": {"detection_method": "atlas_relaxed", "quality_score": 0.8}}
    )
    assert block["Source"] == "measured"


def test_build_aif_block_template_via_detection_level() -> None:
    from pmriprep.utils.aif_provenance import build_aif_provenance_block

    block = build_aif_provenance_block({"aif_detection": {"detection_method": "synthetic"}})
    assert block["Source"] == "synthetic"


def test_build_aif_block_accepts_flat_dict() -> None:
    """Flat pre-merge AIF QC (detection_level at top level) is accepted."""
    from pmriprep.utils.aif_provenance import build_aif_provenance_block

    block = build_aif_provenance_block(
        {"detection_method": "synthetic", "synthetic_fallback_reason": "low_snr"}
    )
    assert block["Source"] == "synthetic"
    assert block["SyntheticFallbackReason"] == "low_snr"


def test_build_aif_block_none_input() -> None:
    from pmriprep.utils.aif_provenance import build_aif_provenance_block

    assert build_aif_provenance_block(None) == {}


def test_build_aif_block_missing_aif_block() -> None:
    from pmriprep.utils.aif_provenance import build_aif_provenance_block

    assert build_aif_provenance_block({"preprocessing": {}}) == {}


def test_validation_node_synthetic_aif_short_circuits() -> None:
    """`_dsc_dce_validation_node` returns SKIP with synthetic_aif reason
    when the upstream constraint already SKIPped for that reason.

    This guards the round-trip: AIF QC → prepare_vp_constraint_from_data
    → constraint_qc.skip_reason='synthetic_aif' → validation node reads the
    reason and returns a matching SKIP status without running the xval.
    """
    from pmriprep.workflows.dce.base import _dsc_dce_validation_node
    from pmriprep.utils.dsc_cbv_constraint import SKIP_REASON_SYNTHETIC_AIF

    pk_qc = {
        "dsc_cbv_constraint": {
            "mode": "skip",
            "skip_reason": SKIP_REASON_SYNTHETIC_AIF,
        }
    }
    # vp_file present so we don't short-circuit on the "no vp_file" guard
    result = _dsc_dce_validation_node(
        vp_file="/tmp/dummy_vp.nii.gz",
        dsc_cbv_file=None,
        dsc_ncbv_file=None,
        brain_mask="/tmp/dummy_mask.nii.gz",
        no_dsc_validation=False,
        pk_qc=pk_qc,
        dsc_preload_configured=None,
    )
    assert result["skipped"] is True
    assert result["status"] == "SKIPPED"
    assert result["skip_reason"] == SKIP_REASON_SYNTHETIC_AIF


def test_validation_node_measured_aif_does_not_skip_on_synthetic_path() -> None:
    """Measured AIF + constraint that SKIPped for a different reason
    (e.g. magnitude gate) does not trigger the synthetic_aif path."""
    from pmriprep.workflows.dce.base import _dsc_dce_validation_node
    from pmriprep.utils.dsc_cbv_constraint import (
        SKIP_REASON_NO_NAWM_MASK, SKIP_REASON_SYNTHETIC_AIF,
    )

    pk_qc = {
        "dsc_cbv_constraint": {
            "mode": "skip",
            "skip_reason": SKIP_REASON_NO_NAWM_MASK,  # not synthetic
        }
    }
    result = _dsc_dce_validation_node(
        vp_file="/tmp/dummy_vp.nii.gz",
        dsc_cbv_file=None,
        dsc_ncbv_file=None,
        brain_mask="/tmp/dummy_mask.nii.gz",
        no_dsc_validation=False,
        pk_qc=pk_qc,
        dsc_preload_configured=None,
    )
    # Without dsc inputs at all, this hits the no-DSC path; the key is
    # that it does NOT emit synthetic_aif.
    assert result.get("skip_reason") != SKIP_REASON_SYNTHETIC_AIF
