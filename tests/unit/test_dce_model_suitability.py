"""Tests for non-fatal, reason-coded 2CXM suitability screening."""

import pytest

from pmriprep.utils.dce_model_suitability import assess_two_cxm_suitability


def _qc(
    *,
    model: str = "2cxm",
    tr: float = 4.0,
    sampling_eligible: bool = False,
    accepted: int = 20,
    aif_confidence: str = "standard",
) -> dict:
    return {
        "aif_detection": {"aif_confidence": aif_confidence},
        "t1_mapping": {"method": "vfa", "b1_correction": "afi"},
        "pk_modeling": {
            "model": model,
            "candidate_models": ["patlak", "extended_tofts", "2cxm"],
            "selection_tr_seconds": tr,
            "acquisition_duration_seconds": 320.0,
            "two_cxm_auto_sampling_eligible": sampling_eligible,
            "r2_median": 0.7,
            "dsc_cbv_constraint": {"mode": "soft"},
            "vp_at_bound_fractions": {"whole_brain": 0.12},
            "voxel_accounting": {
                "fitting_eligible": {"fraction_of_brain": 0.8},
                "optimizer_converged": {"fraction_of_attempted": 0.6},
                "retained_nondegenerate": {"fraction_of_attempted": 0.55},
                "quantitative_qc_accepted": {
                    "count": accepted,
                    "fraction_of_attempted": 0.2 if accepted else 0.0,
                    "fraction_of_retained": 0.36 if accepted else 0.0,
                },
            },
        },
    }


def test_explicit_slow_2cxm_warns_without_rejecting() -> None:
    """The established auto-sampling boundary becomes an explicit-model warning."""
    result = assess_two_cxm_suitability(_qc())

    assert result["status"] == "WARN"
    assert result["reasons"] == [
        "temporal_sampling_outside_auto_2cxm_eligibility"
    ]
    assert result["scope"] == "screening_only_not_biological_validation"
    assert result["metrics"]["volume_tr_seconds"] == 4.0
    assert result["metrics"]["quantitative_qc_accepted_fraction_of_retained"] == (
        pytest.approx(0.36)
    )
    assert result["metrics"]["vp_at_bound_fractions"] == {"whole_brain": 0.12}


def test_zero_accepted_and_reduced_aif_add_reasons() -> None:
    """Missing accepted output and reduced AIF confidence both request review."""
    result = assess_two_cxm_suitability(
        _qc(tr=2.5, sampling_eligible=True, accepted=0, aif_confidence="reduced")
    )

    assert result["status"] == "WARN"
    assert result["reasons"] == [
        "no_quantitatively_accepted_voxels",
        "reduced_aif_confidence",
    ]


def test_auto_candidate_with_no_established_concern_passes_screen_only() -> None:
    """An auto-eligible 2CXM candidate can pass without a validity claim."""
    result = assess_two_cxm_suitability(
        _qc(model="auto", tr=2.5, sampling_eligible=True)
    )

    assert result["status"] == "PASS"
    assert result["metrics"]["auto_candidate"] is True


def test_non_2cxm_model_is_not_applicable() -> None:
    """Fixed Tofts-family runs do not receive a 2CXM assessment."""
    qc = _qc(model="extended_tofts")
    qc["pk_modeling"]["candidate_models"] = []

    result = assess_two_cxm_suitability(qc)

    assert result["status"] == "NOT_APPLICABLE"
    assert result["metrics"] == {}


def test_missing_pk_qc_is_not_evaluated() -> None:
    """Missing model QC is distinguished from a passing assessment."""
    assert assess_two_cxm_suitability({})["status"] == "NOT_EVALUATED"


def test_applicable_model_without_accounting_is_not_evaluated() -> None:
    """Incomplete current-shape QC cannot accidentally produce a passing screen."""
    result = assess_two_cxm_suitability(
        {
            "pk_modeling": {
                "model": "2cxm",
                "two_cxm_auto_sampling_eligible": True,
            }
        }
    )

    assert result["status"] == "NOT_EVALUATED"
    assert result["reasons"] == ["voxel_accounting_unavailable"]


def test_qc_collection_runs_suitability_after_merging_inputs() -> None:
    """Complete AIF, fit, and support state reaches the collected assessment."""
    from pmriprep.workflows.dce._helpers import _collect_qc_metrics

    collected = _collect_qc_metrics(
        b1_qc=None,
        t1_qc=None,
        signal_conversion_qc=None,
        preprocess_qc=None,
        registration_qc=None,
        aif_qc={"aif_confidence": "reduced"},
        pk_qc=_qc()["pk_modeling"],
        validation_result=None,
        use_external_t1=True,
        b1_method="none",
    )

    assessment = collected["model_suitability"]["2cxm"]
    assert assessment["status"] == "WARN"
    assert assessment["reasons"] == [
        "temporal_sampling_outside_auto_2cxm_eligibility",
        "reduced_aif_confidence",
    ]
    assert assessment["metrics"]["t1_method"] == "external"


def test_qc_collection_discloses_assumed_t1_and_independent_b1() -> None:
    """No-T1 QC records the quantitative assumption and applied B1 method."""
    from pmriprep.workflows.dce._helpers import _collect_qc_metrics

    collected = _collect_qc_metrics(
        b1_qc={"method": "precomputed"},
        t1_qc=None,
        signal_conversion_qc={
            "assumed_t1_ms": 1200.0,
            "assumed_t1_field_strength_t": 3.0,
            "assumed_t1_policy": "uniform_field_strength_reference",
        },
        preprocess_qc=None,
        registration_qc=None,
        aif_qc=None,
        pk_qc=None,
        validation_result=None,
        no_t1=True,
        b1_method="precomputed",
    )

    assert collected["t1_mapping"] == {
        "method": "assumed",
        "t1_status": "Skipped by configuration",
        "skipped": True,
        "assumed_t1_ms": 1200.0,
        "assumed_t1_field_strength_t": 3.0,
        "assumed_t1_policy": "uniform_field_strength_reference",
        "b1_correction": "precomputed",
    }


def test_qc_collection_nests_vfa_motion_provenance_with_t1() -> None:
    """Inter-VFA assessment reaches the quantitative T1 provenance block."""
    from pmriprep.workflows.dce._helpers import _collect_qc_metrics

    motion = {"motion_assessment": {"qc_status": "PASS", "correction_applied": False}}
    collected = _collect_qc_metrics(
        b1_qc=None,
        t1_qc={"t1_method": "hybrid"},
        vfa_qc=motion,
        signal_conversion_qc=None,
        preprocess_qc=None,
        registration_qc=None,
        aif_qc=None,
        pk_qc=None,
        validation_result=None,
    )

    assert collected["t1_mapping"]["vfa_preprocessing"] == motion
