"""Unit tests for contrast agent utilities."""

import pytest

from pmriprep.utils.contrast_agent import (
    _normalize_agent_name,
    calculate_dose_from_volume,
    calculate_volume_for_dose,
    classify_dose,
    get_agent_info,
    get_available_agents,
    get_r1_from_sidecar_dict,
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
        assert _normalize_agent_name("Gadavist") == "gadobutrol"

    def test_generic_name_passthrough(self) -> None:
        """Generic name 'gd-dtpa' passes through unchanged."""
        assert _normalize_agent_name("gd-dtpa") == "gd-dtpa"


class TestLookupR1:
    """Tests for R1 relaxivity lookup."""

    def test_gadobutrol_3t(self) -> None:
        """Gadobutrol at 3T returns 5.0 L/(mmol*s)."""
        assert lookup_r1("gadobutrol", 3.0) == pytest.approx(5.0)

    def test_gd_dtpa_3t(self) -> None:
        """Gd-DTPA at 3T returns 3.7 L/(mmol*s)."""
        assert lookup_r1("gd-dtpa", 3.0) == pytest.approx(3.7)

    def test_gd_dtpa_1_5t(self) -> None:
        """Gd-DTPA at 1.5T returns 4.1 L/(mmol*s)."""
        assert lookup_r1("gd-dtpa", 1.5) == pytest.approx(4.1)

    def test_interpolation_midpoint(self) -> None:
        """Field strength 2.25T linearly interpolates between 1.5T and 3.0T values."""
        # 2.25T is midpoint between 2.0 and 2.5 (interpolation zone)
        # fraction = (2.25 - 1.5) / (3.0 - 1.5) = 0.5
        # gd-dtpa: 4.1 + 0.5 * (3.7 - 4.1) = 4.1 - 0.2 = 3.9
        expected = 4.1 + 0.5 * (3.7 - 4.1)
        assert lookup_r1("gd-dtpa", 2.25) == pytest.approx(expected)

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
        assert lookup_r1_safe("gadobutrol", 3.0) == pytest.approx(5.0)


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
        """Gadobutrol info contains r1 values and 1.0M concentration."""
        info = get_agent_info("gadobutrol")
        assert info["r1_1_5T"] == pytest.approx(5.2)
        assert info["r1_3_0T"] == pytest.approx(5.0)
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
        # gadovist -> gadobutrol -> 5.0 at 3T
        assert result["r1"] == pytest.approx(5.0)
        assert result["source"] == "dicom_lookup"
        assert result["volume_ml"] == pytest.approx(3.5)
