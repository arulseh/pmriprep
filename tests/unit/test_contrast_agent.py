"""Unit tests for contrast agent utilities."""

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
    get_r1_from_sidecar_dict,
    lookup_gd_multiplicity,
    lookup_r1,
    lookup_r1_safe,
    lookup_standard_dose,
    parse_contrast_agent_string,
)


class TestParseContrastAgentString:
    """Tests for parsing DICOM/BIDS contrast agent strings."""

    def test_agent_with_european_volume(self) -> None:
        """Parse 'gadovist 3,5 ml' with comma decimal and ml suffix."""
        result = parse_contrast_agent_string("gadovist 3,5 ml")
        assert result["agent_name"] == "gadovist"
        assert result["volume_ml"] == pytest.approx(3.5)

    def test_agent_with_integer_volume(self) -> None:
        """Parse 'Magnevist 10ml' with integer volume."""
        result = parse_contrast_agent_string("Magnevist 10ml")
        assert result["agent_name"] == "Magnevist"
        assert result["volume_ml"] == pytest.approx(10.0)

    def test_agent_with_volume_no_ml_suffix(self) -> None:
        """Parse 'Vueway 3,5' without ml suffix extracts volume."""
        result = parse_contrast_agent_string("Vueway 3,5")
        assert result["agent_name"] == "Vueway"
        assert result["volume_ml"] == pytest.approx(3.5)

    def test_agent_name_only(self) -> None:
        """Parse 'gadobutrol' with no volume info."""
        result = parse_contrast_agent_string("gadobutrol")
        assert result["agent_name"] == "gadobutrol"
        assert result["volume_ml"] is None

    def test_empty_string(self) -> None:
        """Empty string returns empty agent name and None volume."""
        result = parse_contrast_agent_string("")
        assert result["agent_name"] == ""
        assert result["volume_ml"] is None


class TestNormalizeAgentName:
    """Tests for agent name normalization and alias resolution."""

    def test_brand_alias_resolves(self) -> None:
        """'Gadavist' resolves to generic 'gadobutrol' via alias."""
        assert normalize_agent_name("Gadavist") == "gadobutrol"

    def test_generic_name_passthrough(self) -> None:
        """Generic name 'gd-dtpa' passes through unchanged."""
        assert normalize_agent_name("gd-dtpa") == "gd-dtpa"


class TestLookupR1:
    """Tests for R1 relaxivity lookup."""

    def test_gadobutrol_3t(self) -> None:
        """Gadobutrol at 3T returns Szomolanyi 2019 plasma value."""
        assert lookup_r1("gadobutrol", 3.0) == pytest.approx(4.97)

    def test_gd_dtpa_3t(self) -> None:
        """Gd-DTPA at 3T returns 3.7 L/(mmol*s) (Rohrer 2005 plasma)."""
        assert lookup_r1("gd-dtpa", 3.0) == pytest.approx(3.7)

    def test_gd_dtpa_1_5t(self) -> None:
        """Gd-DTPA at 1.5T returns 4.1 L/(mmol*s) (Rohrer 2005 plasma)."""
        assert lookup_r1("gd-dtpa", 1.5) == pytest.approx(4.1)

    def test_gd_dtpa_7t(self) -> None:
        """Gd-DTPA at 7T returns Noebauer-Huhmann 2010 plasma value."""
        assert lookup_r1("gd-dtpa", 7.0) == pytest.approx(3.3)

    def test_gadobutrol_7t(self) -> None:
        """Gadobutrol at 7T returns Szomolanyi 2019 plasma value."""
        assert lookup_r1("gadobutrol", 7.0) == pytest.approx(3.83)

    def test_field_strength_bin_low(self) -> None:
        """B0 ≤ 2.25T snaps to 1.5T value (no interpolation)."""
        # Two-bin selection: 2.25T midpoint snaps low.
        assert lookup_r1("gd-dtpa", 2.25) == pytest.approx(4.1)

    def test_field_strength_bin_mid(self) -> None:
        """2.25 < B0 ≤ 5.0T uses 3T value."""
        assert lookup_r1("gd-dtpa", 2.5) == pytest.approx(3.7)
        assert lookup_r1("gd-dtpa", 5.0) == pytest.approx(3.7)

    def test_unsupported_7t_agent_raises(self) -> None:
        """Agent without peer-reviewed 7T value raises at B0 > 5.0T."""
        with pytest.raises(ValueError, match="No peer-reviewed 7T r1"):
            lookup_r1("gadoxetate", 7.0)

    def test_brand_name_resolves(self) -> None:
        """Brand name 'Gadavist' returns same r1 as 'gadobutrol'."""
        assert lookup_r1("Gadavist", 3.0) == pytest.approx(
            lookup_r1("gadobutrol", 3.0)
        )

    def test_gadoquatrane_3t(self) -> None:
        """Gadoquatrane at 3T returns 10.5 L/(mmol*s)."""
        assert lookup_r1("gadoquatrane", 3.0) == pytest.approx(10.5)

    def test_gadoquatrane_1_5t(self) -> None:
        """Gadoquatrane at 1.5T returns 11.8 L/(mmol*s)."""
        assert lookup_r1("gadoquatrane", 1.5) == pytest.approx(11.8)

    def test_ambelvist_alias_resolves(self) -> None:
        """Brand name 'Ambelvist' resolves to gadoquatrane."""
        assert lookup_r1("ambelvist", 3.0) == pytest.approx(
            lookup_r1("gadoquatrane", 3.0)
        )

    def test_unknown_agent_raises(self) -> None:
        """Unknown agent 'fakium' raises ValueError listing available agents."""
        with pytest.raises(ValueError, match="Available agents"):
            lookup_r1("fakium", 3.0)


class TestLookupR1Safe:
    """Tests for the safe (non-raising) R1 lookup."""

    def test_unknown_returns_none(self) -> None:
        """Unknown agent returns None instead of raising."""
        assert lookup_r1_safe("fakium", 3.0) is None

    def test_known_returns_value(self) -> None:
        """Known agent returns the same value as lookup_r1."""
        assert lookup_r1_safe("gadobutrol", 3.0) == pytest.approx(4.97)


class TestLookupStandardDose:
    """Tests for standard clinical dose lookup."""

    def test_gadobutrol_standard_dose(self) -> None:
        """Gadobutrol standard dose is 0.1 mmol/kg."""
        assert lookup_standard_dose("gadobutrol") == pytest.approx(0.1)

    def test_gadopiclenol_half_dose(self) -> None:
        """Gadopiclenol (Vueway/Elucirem) standard dose is 0.05 mmol/kg."""
        assert lookup_standard_dose("gadopiclenol") == pytest.approx(0.05)

    def test_gadoquatrane_dose(self) -> None:
        """Gadoquatrane (Ambelvist) standard dose is 0.04 mmol/kg (tetrameric)."""
        assert lookup_standard_dose("gadoquatrane") == pytest.approx(0.04)

    def test_ambelvist_alias_dose(self) -> None:
        """Brand name 'Ambelvist' resolves to gadoquatrane for dose lookup."""
        assert lookup_standard_dose("ambelvist") == pytest.approx(0.04)


class TestLookupGdMultiplicity:
    """Tests for Gd3+ ion count per molecule lookup."""

    def test_gadoquatrane_tetrameric(self) -> None:
        """Gadoquatrane has 4 Gd3+ ions per molecule."""
        assert lookup_gd_multiplicity("gadoquatrane") == 4

    def test_ambelvist_alias(self) -> None:
        """Brand name resolves to gadoquatrane multiplicity."""
        assert lookup_gd_multiplicity("ambelvist") == 4

    def test_standard_agents_monomeric(self) -> None:
        """Standard monomeric agents return 1."""
        assert lookup_gd_multiplicity("gadobutrol") == 1
        assert lookup_gd_multiplicity("gd-dtpa") == 1
        assert lookup_gd_multiplicity("gadopiclenol") == 1

    def test_unknown_agent_defaults_to_1(self) -> None:
        """Unknown agent returns 1 (monomeric default)."""
        assert lookup_gd_multiplicity("unknown_agent") == 1


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
        """Gadobutrol info contains Szomolanyi 2019 r1 + 1.0M concentration."""
        info = get_agent_info("gadobutrol")
        assert info["r1_1_5T"] == pytest.approx(4.78)
        assert info["r1_3_0T"] == pytest.approx(4.97)
        assert info["r1_7_0T"] == pytest.approx(3.83)
        assert info["concentration_mmol_per_ml"] == pytest.approx(1.0)


class TestCalculateDoseFromVolume:
    """Tests for dose calculation from injected volume."""

    def test_gd_dtpa_standard(self) -> None:
        """10 mL of 0.5M Gd-DTPA for 70 kg patient gives correct dose."""
        # 10 mL * 0.5 mmol/mL = 5 mmol; 5 mmol / 70 kg = 0.07143 mmol/kg
        dose = calculate_dose_from_volume(10.0, 70.0, "gd-dtpa")
        assert dose == pytest.approx(5.0 / 70.0)

    def test_gadobutrol_double_concentration(self) -> None:
        """Gadobutrol (1M) half volume gives same dose as 0.5M agent."""
        dose_dtpa = calculate_dose_from_volume(10.0, 70.0, "gd-dtpa")
        dose_gado = calculate_dose_from_volume(5.0, 70.0, "gadobutrol")
        assert dose_dtpa == pytest.approx(dose_gado)

    def test_zero_volume_raises(self) -> None:
        """Volume = 0 raises ValueError."""
        with pytest.raises(ValueError, match="Volume must be positive"):
            calculate_dose_from_volume(0.0, 70.0)

    def test_negative_weight_raises(self) -> None:
        """Negative weight raises ValueError."""
        with pytest.raises(ValueError, match="Body weight must be positive"):
            calculate_dose_from_volume(10.0, -1.0)


class TestCalculateVolumeForDose:
    """Tests for volume calculation from target dose."""

    def test_gd_dtpa_standard_dose(self) -> None:
        """0.1 mmol/kg for 70 kg with Gd-DTPA (0.5M) = 14 mL."""
        vol = calculate_volume_for_dose(0.1, 70.0, "gd-dtpa")
        assert vol == pytest.approx(14.0)

    def test_gadobutrol_half_volume(self) -> None:
        """0.1 mmol/kg for 70 kg with gadobutrol (1M) = 7 mL."""
        vol = calculate_volume_for_dose(0.1, 70.0, "gadobutrol")
        assert vol == pytest.approx(7.0)


class TestClassifyDose:
    """Tests for dose classification."""

    def test_none_dose(self) -> None:
        """Very low dose classified as 'none'."""
        assert classify_dose(0.005) == "none"

    def test_half_dose(self) -> None:
        """0.05 mmol/kg classified as 'half'."""
        assert classify_dose(0.05) == "half"

    def test_full_dose(self) -> None:
        """0.1 mmol/kg classified as 'full'."""
        assert classify_dose(0.1) == "full"

    def test_double_dose(self) -> None:
        """0.2 mmol/kg classified as 'double'."""
        assert classify_dose(0.2) == "double"


class TestGetR1FromSidecarDict:
    """Tests for extracting r1 from a BIDS sidecar dictionary."""

    def test_valid_sidecar(self) -> None:
        """Dict with ContrastBolusAgent + MagneticFieldStrength returns correct r1."""
        sidecar = {
            "ContrastBolusAgent": "gadovist 3,5 ml",
            "MagneticFieldStrength": 3.0,
        }
        result = get_r1_from_sidecar_dict(sidecar)
        # gadovist -> gadobutrol -> 4.97 at 3T (Szomolanyi 2019)
        assert result["r1"] == pytest.approx(4.97)
        assert result["source"] == "dicom_lookup"
        assert result["volume_ml"] == pytest.approx(3.5)


class TestGetBrandName:
    """Tests for get_brand_name reverse lookup."""

    def test_known_generics(self) -> None:
        """Known generic names return correct brand names."""
        assert get_brand_name("gadobutrol") == "Gadavist"
        assert get_brand_name("gadopiclenol") == "Vueway"
        assert get_brand_name("gd-dtpa") == "Magnevist"
        assert get_brand_name("gadoterate") == "Dotarem"
        assert get_brand_name("gadoxetate") == "Eovist"

    def test_unknown_generic(self) -> None:
        """Unknown generic name returns None."""
        assert get_brand_name("unknown_agent") is None
        assert get_brand_name("ferumoxytol") is None

    def test_case_insensitive(self) -> None:
        """Lookup is case-insensitive."""
        assert get_brand_name("Gadobutrol") == "Gadavist"
        assert get_brand_name("GD-DTPA") == "Magnevist"


class TestBuildDceProvenance:
    """Tests for build_dce_provenance."""

    def test_user_override_r1(self) -> None:
        """User-supplied r1 records user_override source."""
        prov = build_dce_provenance(
            agent_source=ParameterSource.USER_OVERRIDE,
            r1_relaxivity=5.0,
            field_strength=3.0,
        )
        perf = prov["PerfusionProcessing"]
        assert perf["ContrastAgent"]["Source"] == "user_override"
        assert perf["Relaxivity"]["r1"] == 5.0
        assert perf["Relaxivity"]["Source"] == "user_override"

    def test_bids_detected_agent(self) -> None:
        """BIDS-detected agent includes brand name and original string."""
        prov = build_dce_provenance(
            contrast_agent="gadobutrol",
            agent_source=ParameterSource.BIDS_SIDECAR,
            r1_relaxivity=5.0,
            field_strength=3.0,
            bids_agent_string="Gadavist 7ml",
        )
        agent = prov["PerfusionProcessing"]["ContrastAgent"]
        assert agent["GenericName"] == "gadobutrol"
        assert agent["BrandName"] == "Gadavist"
        assert agent["OriginalString"] == "Gadavist 7ml"
        assert agent["Source"] == "bids_sidecar"

    def test_default_fallback(self) -> None:
        """Default fallback uses Gd-DTPA with default_fallback source."""
        prov = build_dce_provenance()
        agent = prov["PerfusionProcessing"]["ContrastAgent"]
        assert agent["GenericName"] == "gd-dtpa"
        assert agent["BrandName"] == "Magnevist"
        assert agent["Source"] == "default_fallback"

    def test_dose_source_tracking(self) -> None:
        """Dose source is recorded correctly."""
        prov = build_dce_provenance(
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

    def test_preload_auto_detected(self) -> None:
        """Preload auto-detected is the default."""
        prov = build_dsc_provenance()
        preload = prov["PerfusionProcessing"]["PreloadDetection"]
        assert preload["Source"] == "auto_detected"

    def test_preload_user_specified(self) -> None:
        """Preload user-specified recorded correctly."""
        prov = build_dsc_provenance(
            preload_source=ParameterSource.USER_SPECIFIED,
        )
        preload = prov["PerfusionProcessing"]["PreloadDetection"]
        assert preload["Source"] == "user_specified"

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
        """Existing callers without new params get identical output."""
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
        assert dose["effective_gd_dose_mmol_kg"] == pytest.approx(0.16)

    def test_gd_multiplicity_omitted_for_monomeric(self) -> None:
        """Monomeric agent does not record gd_ions_per_molecule."""
        prov = build_dsc_provenance(agent_standard_dose=0.1, gd_multiplicity=1)
        dose = prov["PerfusionProcessing"]["ContrastDose"]
        assert "gd_ions_per_molecule" not in dose

    def test_inference_cpi_included(self) -> None:
        """Inference engine + FABBER prior profile recorded for CPI."""
        prov = build_dsc_provenance(deconv_method="cpi", inference="vi")
        deconv = prov["PerfusionProcessing"]["Deconvolution"]
        assert deconv["method"] == "cpi"
        assert deconv["inference"] == "vi"
        assert deconv["inference_engine"] == "fabber"
        assert deconv["cpi_prior_profile"] == "fabber"

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

    def test_dce_agent_default_warning(self) -> None:
        """Warning when DCE agent falls back to default."""
        prov = build_dce_provenance(
            agent_source=ParameterSource.DEFAULT_FALLBACK,
            r1_relaxivity=3.8,
            field_strength=3.0,
        )
        warnings = collect_default_warnings(prov, "DCE")
        assert len(warnings) >= 1
        assert "defaulted" in warnings[0].lower() or "Gd-DTPA" in warnings[0]

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


class TestCollectAifFallbackWarning:
    """Tests for collect_aif_fallback_warning."""

    def test_none_when_no_qc(self) -> None:
        assert collect_aif_fallback_warning(None, "DSC") is None

    def test_none_when_empty_dict(self) -> None:
        assert collect_aif_fallback_warning({}, "DSC") is None
        assert collect_aif_fallback_warning({}, "DCE") is None

    def test_dsc_none_when_measured(self) -> None:
        qc: dict = {"aif_detection": {"detection_method": "atlas_strict"}}
        assert collect_aif_fallback_warning(qc, "DSC") is None

    def test_dsc_warning_when_synthetic(self) -> None:
        qc: dict = {"aif_detection": {"detection_method": "synthetic"}}
        result = collect_aif_fallback_warning(qc, "DSC")
        assert result is not None
        assert "Parker population template" in result
        assert "cascade_exhausted" in result

    def test_dsc_uses_fallback_reason(self) -> None:
        qc: dict = {
            "aif_detection": {"detection_method": "synthetic"},
            "synthetic_fallback_reason": "no_candidates",
        }
        result = collect_aif_fallback_warning(qc, "DSC")
        assert result is not None
        assert "no_candidates" in result

    def test_dce_none_when_no_aif_detection(self) -> None:
        qc: dict = {"preprocessing": {}}
        assert collect_aif_fallback_warning(qc, "DCE") is None

    def test_dce_none_when_atlas(self) -> None:
        qc: dict = {"aif_detection": {"aif_source": "atlas"}}
        assert collect_aif_fallback_warning(qc, "DCE") is None

    def test_dce_warning_when_population(self) -> None:
        qc: dict = {"aif_detection": {"aif_source": "population"}}
        result = collect_aif_fallback_warning(qc, "DCE")
        assert result is not None
        assert "Parker population model" in result
        assert "user-selected" in result

    def test_dce_auto_fallback_includes_reason(self) -> None:
        qc: dict = {
            "aif_detection": {
                "aif_source": "population",
                "auto_detection_attempted": True,
                "auto_fallback_reason": "0/5 candidates scored >= 0.40",
            },
        }
        result = collect_aif_fallback_warning(qc, "DCE")
        assert result is not None
        assert "auto-detection failed" in result
        assert "0/5 candidates" in result

    def test_unknown_modality_returns_none(self) -> None:
        qc: dict = {"aif_detection": {"detection_method": "synthetic"}}
        assert collect_aif_fallback_warning(qc, "UNKNOWN") is None
