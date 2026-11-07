"""Unit tests for contrast agent utilities."""

import json
import math
from pathlib import Path
from typing import Any, Callable

import pytest

from pmriprep.utils.contrast_agent import (
    ParameterSource,
    normalize_agent_name,
    build_dce_provenance,
    build_dsc_provenance,
    calculate_dose_from_volume,
    calculate_volume_for_dose,
    classify_dose,
    collect_aif_fallback_warning,
    collect_default_warnings,
    get_agent_info,
    get_available_agents,
    get_brand_name,
    get_r1_from_sidecar,
    get_r1_from_sidecar_dict,
    lookup_gd_multiplicity,
    lookup_r1,
    lookup_r1_safe,
    lookup_standard_dose,
    parse_contrast_agent_string,
)


@pytest.mark.parametrize(
    ("raw", "agent", "volume"),
    [
        pytest.param("gadovist 3,5 ml", "gadovist", 3.5, id="comma-decimal-ml"),
        pytest.param("Magnevist 10ml", "Magnevist", 10.0, id="integer-ml"),
        pytest.param("Vueway 3,5", "Vueway", 3.5, id="volume-without-suffix"),
        pytest.param("gadobutrol", "gadobutrol", None, id="name-only"),
        pytest.param("", "", None, id="empty"),
    ],
)
def test_parse_contrast_agent_string(
    raw: str,
    agent: str,
    volume: float | None,
) -> None:
    """DICOM/BIDS agent strings retain names and optional volumes."""
    result = parse_contrast_agent_string(raw)
    assert result["agent_name"] == agent
    if volume is None:
        assert result["volume_ml"] is None
    else:
        assert result["volume_ml"] == pytest.approx(volume)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("Gadavist", "gadobutrol", id="brand-alias"),
        pytest.param("gd-dtpa", "gd-dtpa", id="generic-passthrough"),
    ],
)
def test_normalize_agent_name(raw: str, expected: str) -> None:
    assert normalize_agent_name(raw) == expected


@pytest.mark.parametrize(
    ("agent", "field_strength", "expected"),
    [
        pytest.param("gadobutrol", 3.0, 5.0, id="gadobutrol-3t"),
        pytest.param("gadobutrol", 1.5, 5.2, id="gadobutrol-1p5t"),
        pytest.param("gd-dtpa", 3.0, 3.7, id="gd-dtpa-3t"),
        pytest.param("gd-dtpa", 1.5, 4.1, id="gd-dtpa-1p5t"),
        pytest.param("gd-dtpa-bma", 1.5, 4.3, id="gadodiamide-1p5t"),
        pytest.param("gd-dtpa-bma", 3.0, 4.0, id="gadodiamide-3t"),
        pytest.param("gadoterate", 1.5, 3.6, id="gadoterate-1p5t"),
        pytest.param("gadoterate", 3.0, 3.5, id="gadoterate-3t"),
        pytest.param("gadoteridol", 1.5, 4.1, id="gadoteridol-1p5t"),
        pytest.param("gadoteridol", 3.0, 3.7, id="gadoteridol-3t"),
        pytest.param("gadobenate", 1.5, 6.3, id="gadobenate-1p5t"),
        pytest.param("gadobenate", 3.0, 5.5, id="gadobenate-3t"),
        pytest.param("gadoxetate", 1.5, 6.9, id="gadoxetate-1p5t"),
        pytest.param("gadoxetate", 3.0, 6.2, id="gadoxetate-3t"),
        pytest.param("gadopiclenol", 1.5, 12.8, id="gadopiclenol-1p5t"),
        pytest.param("gadopiclenol", 3.0, 11.6, id="gadopiclenol-3t"),
        pytest.param("gd-dtpa", 2.25, 4.1, id="low-bin-boundary"),
        pytest.param("gd-dtpa", (2.2501, 5.0), 3.7, id="mid-bin-range"),
        pytest.param("gadoquatrane", 3.0, 10.5, id="gadoquatrane-3t"),
        pytest.param("gadoquatrane", 1.5, 11.8, id="gadoquatrane-1p5t"),
    ],
)
def test_lookup_r1_values(
    agent: str,
    field_strength: float | tuple[float, ...],
    expected: float,
) -> None:
    """Published relaxivities and field-strength bin boundaries are preserved."""
    fields = field_strength if isinstance(field_strength, tuple) else (field_strength,)
    for value in fields:
        assert lookup_r1(agent, value) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("alias", "generic"),
    [
        pytest.param("Gadavist", "gadobutrol", id="gadavist"),
        pytest.param("gd-dota", "gadoterate", id="gd-dota"),
        pytest.param("gd-hp-do3a", "gadoteridol", id="gd-hp-do3a"),
        pytest.param("gd-bopta", "gadobenate", id="gd-bopta"),
        pytest.param("gadopentetate", "gd-dtpa", id="gadopentetate"),
        pytest.param("ambelvist", "gadoquatrane", id="ambelvist"),
    ],
)
def test_lookup_r1_aliases(alias: str, generic: str) -> None:
    assert lookup_r1(alias, 3.0) == pytest.approx(lookup_r1(generic, 3.0))


@pytest.mark.parametrize(
    ("agent", "field_strength", "message"),
    [
        *[
            pytest.param(agent, 7.0, "dce.r1_relaxivity", id=f"{agent}-7t")
            for agent in get_available_agents()
        ],
        pytest.param("gadopiclenol", 5.01, "dce.r1_relaxivity", id="unsupported-field-bin"),
        pytest.param("fakium", 3.0, "Available agents", id="unknown-agent"),
    ],
)
def test_lookup_r1_errors(agent: str, field_strength: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        lookup_r1(agent, field_strength)


@pytest.mark.parametrize(
    ("agent", "expected"),
    [
        pytest.param("fakium", None, id="unknown"),
        pytest.param("gadobutrol", 5.0, id="known"),
    ],
)
def test_lookup_r1_safe(agent: str, expected: float | None) -> None:
    result = lookup_r1_safe(agent, 3.0)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("agent", "expected"),
    [
        pytest.param("gadobutrol", 0.1, id="gadobutrol"),
        pytest.param("gadopiclenol", 0.05, id="gadopiclenol"),
        pytest.param("gadoquatrane", 0.04, id="gadoquatrane"),
        pytest.param("ambelvist", 0.04, id="ambelvist-alias"),
    ],
)
def test_lookup_standard_dose(agent: str, expected: float) -> None:
    assert lookup_standard_dose(agent) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("agents", "expected"),
    [
        pytest.param(("gadoquatrane",), 4, id="tetrameric"),
        pytest.param(("ambelvist",), 4, id="tetrameric-alias"),
        pytest.param(
            ("gadobutrol", "gd-dtpa", "gadopiclenol"), 1, id="monomeric",
        ),
        pytest.param(("unknown_agent",), 1, id="unknown-default"),
    ],
)
def test_lookup_gd_multiplicity(agents: tuple[str, ...], expected: int) -> None:
    for agent in agents:
        assert lookup_gd_multiplicity(agent) == expected


class TestGetAvailableAgents:
    """Tests for retrieving the available agent list."""

    def test_returns_sorted_list(self) -> None:
        """Returned list is sorted and includes key agents."""
        agents = get_available_agents()
        assert agents == sorted(agents)
        assert "gadobutrol" in agents
        assert "gd-dtpa" in agents


class TestGetAgentInfo:
    """Tests for detailed agent information retrieval."""

    def test_gadobutrol_info(self) -> None:
        """Gadobutrol info includes its r1 values and formulation concentration."""
        info = get_agent_info("gadobutrol")
        assert info["r1_1_5T"] == pytest.approx(5.2)
        assert info["r1_3_0T"] == pytest.approx(5.0)
        assert math.isnan(info["r1_7_0T"])
        assert info["concentration_mmol_per_ml"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("volume", "weight", "agent", "expected"),
    [
        pytest.param(10.0, 70.0, "gd-dtpa", 5.0 / 70.0, id="gd-dtpa"),
        pytest.param(5.0, 70.0, "gadobutrol", 5.0 / 70.0, id="gadobutrol"),
    ],
)
def test_calculate_dose_from_volume(
    volume: float,
    weight: float,
    agent: str,
    expected: float,
) -> None:
    assert calculate_dose_from_volume(volume, weight, agent) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("volume", "weight", "message"),
    [
        pytest.param(0.0, 70.0, "Volume must be positive", id="zero-volume"),
        pytest.param(10.0, -1.0, "Body weight must be positive", id="negative-weight"),
    ],
)
def test_calculate_dose_from_volume_rejects_invalid_input(
    volume: float,
    weight: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        calculate_dose_from_volume(volume, weight)


@pytest.mark.parametrize(
    ("agent", "expected"),
    [
        pytest.param("gd-dtpa", 14.0, id="gd-dtpa"),
        pytest.param("gadobutrol", 7.0, id="gadobutrol"),
    ],
)
def test_calculate_volume_for_dose(agent: str, expected: float) -> None:
    assert calculate_volume_for_dose(0.1, 70.0, agent) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("dose", "expected"),
    [
        pytest.param(0.005, "none", id="none"),
        pytest.param(0.05, "half", id="half"),
        pytest.param(0.1, "full", id="full"),
        pytest.param(0.2, "double", id="double"),
    ],
)
def test_classify_dose(dose: float, expected: str) -> None:
    assert classify_dose(dose) == expected


@pytest.mark.parametrize("field_strength", [None, 7.0])
def test_agent_identity_survives_unavailable_r1(
    tmp_path: Path, field_strength: float | None,
) -> None:
    sidecar: dict[str, Any] = {"ContrastBolusAgent": "Ambelvist 7ml"}
    if field_strength is not None:
        sidecar["MagneticFieldStrength"] = field_strength
    path = tmp_path / "contrast.json"
    path.write_text(json.dumps(sidecar))
    result = get_r1_from_sidecar(path)
    assert result == get_r1_from_sidecar_dict(sidecar)
    assert result["agent_name"] == "gadoquatrane"
    assert result["agent_string"] == "Ambelvist 7ml"
    assert result["volume_ml"] == 7.0
    assert result["r1"] is None
    assert result["source"] == (
        "no_field_strength" if field_strength is None else "unsupported_field"
    )


def test_gadoquatrane_volume_helpers_use_gd_units() -> None:
    assert calculate_dose_from_volume(7.0, 70.0, "Ambelvist") == pytest.approx(0.04)
    assert calculate_volume_for_dose(0.04, 70.0, "Ambelvist") == pytest.approx(7.0)


@pytest.mark.parametrize("builder", [build_dce_provenance, build_dsc_provenance])
def test_provenance_normalizes_brand_name(
    builder: Callable[..., dict[str, Any]],
) -> None:
    agent = builder(contrast_agent="Gadavist")["PerfusionProcessing"]["ContrastAgent"]
    assert agent["GenericName"] == "gadobutrol"
    assert agent["BrandName"] == "Gadavist"


def test_provenance_default_r1_uses_agent_and_field() -> None:
    perf = build_dce_provenance(
        contrast_agent="Gadavist", field_strength=3.0,
    )["PerfusionProcessing"]
    assert perf["Relaxivity"]["r1"] == pytest.approx(5.0)
    with pytest.raises(ValueError, match="dce.r1_relaxivity"):
        build_dce_provenance(contrast_agent="Ambelvist", field_strength=7.0)


class TestGetR1FromSidecarDict:
    """Tests for extracting r1 from a BIDS sidecar dictionary."""

    def test_valid_sidecar(self) -> None:
        """Dict with ContrastBolusAgent + MagneticFieldStrength returns correct r1."""
        sidecar = {
            "ContrastBolusAgent": "gadovist 3,5 ml",
            "MagneticFieldStrength": 3.0,
        }
        result = get_r1_from_sidecar_dict(sidecar)
        assert result["r1"] == pytest.approx(5.0)
        assert result["agent_name"] == "gadobutrol"
        assert result["source"] == "dicom_lookup"
        assert result["volume_ml"] == pytest.approx(3.5)


@pytest.mark.parametrize(
    "cases",
    [
        pytest.param(
            (("gadobutrol", "Gadavist"), ("gadopiclenol", "Vueway"),
             ("gd-dtpa", "Magnevist"), ("gadoterate", "Dotarem"),
             ("gadoxetate", "Eovist")),
            id="known-generics",
        ),
        pytest.param(
            (("unknown_agent", None), ("ferumoxytol", None)),
            id="unknown-generics",
        ),
        pytest.param(
            (("Gadobutrol", "Gadavist"), ("GD-DTPA", "Magnevist")),
            id="case-insensitive",
        ),
    ],
)
def test_get_brand_name(cases: tuple[tuple[str, str | None], ...]) -> None:
    for agent, expected in cases:
        assert get_brand_name(agent) == expected


class TestBuildDceProvenance:
    """Tests for build_dce_provenance."""

    def test_user_override_r1(self) -> None:
        """User-supplied r1 records user_override source."""
        prov = build_dce_provenance(
            relaxivity_source=ParameterSource.USER_OVERRIDE,
            r1_relaxivity=5.0,
            field_strength=3.0,
        )
        perf = prov["PerfusionProcessing"]
        assert perf["ContrastAgent"] == {"GenericName": "unspecified", "Source": "unspecified"}
        assert perf["Relaxivity"]["r1"] == 5.0
        assert perf["Relaxivity"]["Source"] == "user_override"

    def test_bids_detected_agent(self) -> None:
        """BIDS-detected agent includes brand name and original string."""
        prov = build_dce_provenance(
            contrast_agent="gadobutrol",
            agent_source=ParameterSource.BIDS_SIDECAR,
            field_strength=3.0,
            bids_agent_string="Gadavist 7ml",
            agent_standard_dose_source=ParameterSource.AGENT_LOOKUP,
        )
        agent = prov["PerfusionProcessing"]["ContrastAgent"]
        assert agent["GenericName"] == "gadobutrol"
        assert agent["BrandName"] == "Gadavist"
        assert agent["OriginalString"] == "Gadavist 7ml"
        assert agent["Source"] == "bids_sidecar"
        assert prov["PerfusionProcessing"]["Relaxivity"]["Source"] == "agent_lookup"
        assert prov["PerfusionProcessing"]["Relaxivity"]["r1"] == 5.0
        assert prov["PerfusionProcessing"]["ContrastDose"]["AgentStandardDoseSource"] == "agent_lookup"

    def test_unspecified_agent_requires_r1(self) -> None:
        with pytest.raises(ValueError, match="explicit r1_relaxivity"):
            build_dce_provenance()

    def test_dose_source_tracking(self) -> None:
        """Dose source is recorded correctly."""
        prov = build_dce_provenance(
            r1_relaxivity=3.7,
            dose_multiplier=0.5,
            dose_source=ParameterSource.USER_SPECIFIED,
        )
        dose = prov["PerfusionProcessing"]["ContrastDose"]
        assert dose["dose_multiplier"] == 0.5
        assert dose["Source"] == "user_specified"

    def test_values_are_python_scalars(self) -> None:
        """All numeric values are Python floats, not numpy."""
        prov = build_dce_provenance(r1_relaxivity=5.0, field_strength=3.0)
        perf = prov["PerfusionProcessing"]
        assert type(perf["Relaxivity"]["r1"]) is float
        assert type(perf["Relaxivity"]["field_strength_T"]) is float
        assert type(perf["ContrastDose"]["dose_multiplier"]) is float


class TestBuildDscProvenance:
    """Tests for build_dsc_provenance."""

    def test_default_values(self) -> None:
        """Default provenance has expected structure."""
        prov = build_dsc_provenance()
        perf = prov["PerfusionProcessing"]
        assert "ContrastDose" in perf
        assert "Deconvolution" in perf
        assert "LeakageCorrection" in perf
        assert "PreloadDetection" in perf

    def test_dose_tracking(self) -> None:
        """Dose source recorded correctly."""
        prov = build_dsc_provenance(
            dose_multiplier=0.5,
            dose_source=ParameterSource.USER_SPECIFIED,
            agent_standard_dose=0.05,
            agent_standard_dose_source=ParameterSource.USER_SPECIFIED,
        )
        dose = prov["PerfusionProcessing"]["ContrastDose"]
        assert dose["dose_multiplier"] == 0.5
        assert dose["standard_dose_mmol_kg"] == 0.05
        assert dose["Source"] == "user_specified"
        assert dose["AgentStandardDoseSource"] == "user_specified"

    def test_preload_unknown(self) -> None:
        """Unknown administered preload is the default."""
        prov = build_dsc_provenance()
        preload = prov["PerfusionProcessing"]["PreloadDetection"]
        assert preload == {"source": "unknown"}

    def test_preload_user_specified(self) -> None:
        """Preload user-specified recorded correctly."""
        prov = build_dsc_provenance(
            preload_source=ParameterSource.USER_SPECIFIED,
        )
        preload = prov["PerfusionProcessing"]["PreloadDetection"]
        assert preload == {"source": "user_specified"}

    def test_agent_identity_from_sidecar(self) -> None:
        """BIDS-detected agent includes generic name, brand, and original string."""
        prov = build_dsc_provenance(
            contrast_agent="gadopiclenol",
            agent_source=ParameterSource.BIDS_SIDECAR,
            bids_agent_string="Vueway",
        )
        agent = prov["PerfusionProcessing"]["ContrastAgent"]
        assert agent["GenericName"] == "gadopiclenol"
        assert agent["BrandName"] == "Vueway"
        assert agent["OriginalString"] == "Vueway"
        assert agent["Source"] == "bids_sidecar"

    def test_agent_identity_omitted_by_default(self) -> None:
        """No ContrastAgent block when agent is not provided."""
        prov = build_dsc_provenance()
        assert "ContrastAgent" not in prov["PerfusionProcessing"]

    def test_agent_without_bids_string(self) -> None:
        """Agent without bids_agent_string omits OriginalString."""
        prov = build_dsc_provenance(
            contrast_agent="gadobutrol",
            agent_source=ParameterSource.BIDS_SIDECAR,
        )
        agent = prov["PerfusionProcessing"]["ContrastAgent"]
        assert agent["GenericName"] == "gadobutrol"
        assert agent["BrandName"] == "Gadavist"
        assert "OriginalString" not in agent

    def test_backward_compat_no_new_params(self) -> None:
        """Omitted optional arguments preserve the default output."""
        prov = build_dsc_provenance(
            dose_multiplier=0.5,
            dose_source=ParameterSource.USER_SPECIFIED,
            agent_standard_dose=0.05,
            agent_standard_dose_source=ParameterSource.USER_SPECIFIED,
        )
        perf = prov["PerfusionProcessing"]
        assert "ContrastAgent" not in perf
        assert "ContrastDose" in perf
        assert "Deconvolution" in perf

    def test_gd_multiplicity_recorded(self) -> None:
        """Tetrameric agent records gd_ions_per_molecule and effective Gd dose."""
        prov = build_dsc_provenance(
            agent_standard_dose=0.04,
            gd_multiplicity=4,
        )
        dose = prov["PerfusionProcessing"]["ContrastDose"]
        assert dose["gd_ions_per_molecule"] == 4
        assert dose["effective_gd_dose_mmol_kg"] == pytest.approx(0.04)

    def test_gd_multiplicity_omitted_for_monomeric(self) -> None:
        """Monomeric agent does not record gd_ions_per_molecule."""
        prov = build_dsc_provenance(agent_standard_dose=0.1, gd_multiplicity=1)
        dose = prov["PerfusionProcessing"]["ContrastDose"]
        assert "gd_ions_per_molecule" not in dose

    def test_inference_cpi_included(self) -> None:
        """Static CPI provenance describes the executing signal-domain model."""
        prov = build_dsc_provenance(deconv_method="cpi", inference="vi")
        deconv = prov["PerfusionProcessing"]["Deconvolution"]
        assert deconv["method"] == "cpi"
        assert deconv["inference"] == "vi"
        assert deconv["inference_engine"] == "pmriprep"
        assert deconv["fit_domain"] == "signal"
        assert deconv["cpi_model"] == "continuous_cubic_residue"
        assert "cpi_parameterization" not in deconv
        assert "cpi_prior_profile" not in deconv
        assert "vb_max_iterations" not in deconv

    def test_inference_ignored_for_osvd(self) -> None:
        """Inference engine not recorded for non-parametric methods."""
        prov = build_dsc_provenance(deconv_method="oSVD", inference="vi")
        deconv = prov["PerfusionProcessing"]["Deconvolution"]
        assert deconv["method"] == "oSVD"
        assert "inference" not in deconv


class TestCollectDefaultWarnings:
    """Tests for collect_default_warnings."""

    def test_no_warnings_when_all_specified(self) -> None:
        """No warnings when all sources are user-specified."""
        prov = build_dce_provenance(
            contrast_agent="gadobutrol",
            agent_source=ParameterSource.BIDS_SIDECAR,
            r1_relaxivity=5.0,
            dose_source=ParameterSource.USER_SPECIFIED,
        )
        warnings = collect_default_warnings(prov, "DCE")
        assert warnings == []

    def test_dce_dose_default_warning(self) -> None:
        """Warning when DCE dose is defaulted."""
        prov = build_dce_provenance(
            agent_source=ParameterSource.BIDS_SIDECAR,
            contrast_agent="gadobutrol",
            dose_source=ParameterSource.DEFAULT_FALLBACK,
        )
        warnings = collect_default_warnings(prov, "DCE")
        assert any("dose" in w.lower() for w in warnings)

    def test_dsc_dose_default_warning(self) -> None:
        """Warning when DSC dose is defaulted."""
        prov = build_dsc_provenance(
            dose_source=ParameterSource.DEFAULT_FALLBACK,
            agent_standard_dose_source=ParameterSource.DEFAULT_FALLBACK,
        )
        warnings = collect_default_warnings(prov, "DSC")
        assert any("dose" in w.lower() for w in warnings)

    def test_dsc_no_preload_warning_when_specified(self) -> None:
        """No preload warning when user-specified."""
        prov = build_dsc_provenance(
            dose_source=ParameterSource.USER_SPECIFIED,
            agent_standard_dose_source=ParameterSource.USER_SPECIFIED,
            preload_source=ParameterSource.USER_SPECIFIED,
        )
        warnings = collect_default_warnings(prov, "DSC")
        assert warnings == []

    def test_dsc_no_agent_warning_when_sidecar(self) -> None:
        """No agent warning when DSC agent is BIDS-detected."""
        prov = build_dsc_provenance(
            contrast_agent="gadopiclenol",
            agent_source=ParameterSource.BIDS_SIDECAR,
            dose_source=ParameterSource.USER_SPECIFIED,
            agent_standard_dose_source=ParameterSource.BIDS_SIDECAR,
            preload_source=ParameterSource.USER_SPECIFIED,
        )
        warnings = collect_default_warnings(prov, "DSC")
        assert warnings == []

    def test_dsc_agent_default_warning(self) -> None:
        """Warning when DSC agent falls back to default."""
        prov = build_dsc_provenance(
            contrast_agent="gd-dtpa",
            agent_source=ParameterSource.DEFAULT_FALLBACK,
            dose_source=ParameterSource.USER_SPECIFIED,
            agent_standard_dose_source=ParameterSource.USER_SPECIFIED,
            preload_source=ParameterSource.USER_SPECIFIED,
        )
        warnings = collect_default_warnings(prov, "DSC")
        assert len(warnings) == 1
        assert "defaulted" in warnings[0].lower() or "gd-dtpa" in warnings[0].lower()

    def test_dsc_no_agent_block_no_agent_warning(self) -> None:
        """No agent-identity warning when ContrastAgent block is absent."""
        prov = build_dsc_provenance(
            dose_source=ParameterSource.USER_SPECIFIED,
            agent_standard_dose_source=ParameterSource.USER_SPECIFIED,
            preload_source=ParameterSource.USER_SPECIFIED,
        )
        warnings = collect_default_warnings(prov, "DSC")
        assert not any("agent" in w.lower() and "dose" not in w.lower() for w in warnings)


@pytest.mark.parametrize(
    ("qc", "modalities", "contains", "excludes"),
    [
        pytest.param(None, ("DSC",), (), (), id="no-qc"),
        pytest.param({}, ("DSC", "DCE"), (), (), id="empty-qc"),
        pytest.param(
            {"aif_detection": {"detection_method": "atlas_strict"}},
            ("DSC",), (), (), id="dsc-measured",
        ),
        pytest.param(
            {"aif_detection": {"detection_method": "unavailable", "detection_status": "failed"}},
            ("DSC",), ("DSC AIF detection failed", "AIF-independent maps"), (),
            id="dsc-unavailable",
        ),
        pytest.param({"preprocessing": {}}, ("DCE",), (), (),
                     id="dce-no-detection"),
        pytest.param({"aif_detection": {"aif_source": "atlas"}},
                     ("DCE",), (), (), id="dce-atlas"),
        pytest.param(
            {"aif_detection": {"aif_source": "population"}},
            ("DCE",), ("Parker population model", "user-selected"), (),
            id="dce-population",
        ),
        pytest.param(
            {"aif_detection": {"aif_source": "population",
                               "auto_detection_attempted": True,
                               "auto_fallback_reason":
                                   "0/5 candidates scored >= 0.40"}},
            ("DCE",),
            ("automatic detection found no acceptable arterial curve",),
            ("0/5 candidates",),
            id="dce-auto-fallback",
        ),
        pytest.param(
            {"aif_detection": {"detection_method": "unavailable"}},
            ("UNKNOWN",), (), (), id="unknown-modality",
        ),
    ],
)
def test_collect_aif_fallback_warning(
    qc: dict | None,
    modalities: tuple[str, ...],
    contains: tuple[str, ...],
    excludes: tuple[str, ...],
) -> None:
    """Fallback disclosure follows the AIF source and modality."""
    for modality in modalities:
        result = collect_aif_fallback_warning(qc, modality)
        if not contains:
            assert result is None
            continue
        assert result is not None
        assert all(text in result for text in contains)
        assert all(text not in result for text in excludes)
