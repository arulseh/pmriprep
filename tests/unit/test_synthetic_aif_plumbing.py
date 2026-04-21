"""Integration tests for v0.28.3 synthetic-AIF SKIP plumbing.

Exercises the plumbing that carries the synthetic-AIF signal from the
DSC AIF cascade through to downstream consumers (DCE vp constraint,
DSC-DCE validation, sidecar writers). Complements the per-module unit
tests by asserting end-to-end flag propagation.
"""
from __future__ import annotations


def test_extract_aif_provenance_measured() -> None:
    from pmriprep.interfaces.dsc_outputs import _extract_aif_provenance

    qc = {"aif": {"detection_level": "atlas_relaxed"}}
    source, comparable = _extract_aif_provenance(qc)
    assert source == "measured"
    assert comparable is True


def test_extract_aif_provenance_synthetic_via_detection_level() -> None:
    from pmriprep.interfaces.dsc_outputs import _extract_aif_provenance

    qc = {"aif": {"detection_level": "synthetic"}}
    source, comparable = _extract_aif_provenance(qc)
    assert source == "synthetic"
    assert comparable is False


def test_extract_aif_provenance_synthetic_via_legacy_flag() -> None:
    from pmriprep.interfaces.dsc_outputs import _extract_aif_provenance

    qc = {"aif": {"cbv_auc_denominator_is_synthetic": True}}
    source, comparable = _extract_aif_provenance(qc)
    assert source == "synthetic"
    assert comparable is False


def test_extract_aif_provenance_none_input() -> None:
    from pmriprep.interfaces.dsc_outputs import _extract_aif_provenance

    source, comparable = _extract_aif_provenance(None)
    assert source == "measured"
    assert comparable is True


def test_extract_aif_provenance_missing_aif_block() -> None:
    from pmriprep.interfaces.dsc_outputs import _extract_aif_provenance

    source, comparable = _extract_aif_provenance({"preprocessing": {}})
    assert source == "measured"
    assert comparable is True


def test_validation_node_synthetic_aif_short_circuits() -> None:
    """`_dsc_dce_validation_node` returns SKIP with synthetic_aif reason
    when the upstream constraint already SKIPped for that reason.

    This guards the v0.28.3 round-trip: AIF QC → prepare_vp_constraint_from_data
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
