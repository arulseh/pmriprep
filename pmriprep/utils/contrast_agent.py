"""Contrast agent utilities for DCE-MRI processing.

Provides relaxivity lookups for common gadolinium-based contrast agents,
dose calculation utilities, and provenance tracking for output sidecars.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Final, NamedTuple, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Contrast Agent Relaxivity Data
# =============================================================================

class AgentRelaxivity(NamedTuple):
    """R1 relaxivity values at different field strengths.

    Values are in L/(mmol·s) (equivalently s^-1 per mM) per millimole of
    molecule (not per Gd3+ ion). r1_7_0T is None for agents without a
    peer-reviewed plasma-37°C measurement at 7T; lookups at B0 ≥ 5T for
    those agents raise rather than extrapolate.
    """

    r1_1_5T: float  # R1 relaxivity at 1.5T
    r1_3_0T: float  # R1 relaxivity at 3.0T
    r1_7_0T: Optional[float] = None  # R1 at 7T; None = no measured value
    gd_ions: int = 1  # Gd3+ ions per molecule (4 for tetrameric gadoquatrane)
    source: str = ""  # Primary citation for the values


#: R1 relaxivity values for common gadolinium-based contrast agents in
#: human plasma at 37°C, per Gd3+ ion unless gd_ions > 1.
#:
CONTRAST_AGENTS: Final[dict[str, AgentRelaxivity]] = {
    # Linear agents (lower relaxivity, higher stability)
    "gd-dtpa": AgentRelaxivity(  # Magnevist
        r1_1_5T=4.1, r1_3_0T=3.7, r1_7_0T=3.3,
        source="Rohrer 2005 (1.5T/3T plasma); Noebauer-Huhmann 2010 (7T)",
    ),
    "gadopentetate": AgentRelaxivity(  # Magnevist (generic)
        r1_1_5T=4.1, r1_3_0T=3.7, r1_7_0T=3.3,
        source="Rohrer 2005 (1.5T/3T plasma); Noebauer-Huhmann 2010 (7T)",
    ),
    "gd-dtpa-bma": AgentRelaxivity(  # Omniscan (NSF-restricted)
        r1_1_5T=4.3, r1_3_0T=4.0, r1_7_0T=None,
        source="Rohrer 2005 plasma; no peer-reviewed 7T plasma measurement",
    ),

    # Macrocyclic agents (higher stability, preferred clinically)
    "gadoterate": AgentRelaxivity(  # Dotarem
        r1_1_5T=3.32, r1_3_0T=3.00, r1_7_0T=2.84,
        source="Szomolanyi 2019 (human plasma 37°C, all 3 fields)",
    ),
    "gd-dota": AgentRelaxivity(  # Dotarem (generic)
        r1_1_5T=3.32, r1_3_0T=3.00, r1_7_0T=2.84,
        source="Szomolanyi 2019 (human plasma 37°C, all 3 fields)",
    ),
    "gadoteridol": AgentRelaxivity(  # ProHance
        r1_1_5T=3.80, r1_3_0T=3.28, r1_7_0T=3.21,
        source="Szomolanyi 2019 (human plasma 37°C, all 3 fields)",
    ),
    "gd-hp-do3a": AgentRelaxivity(  # ProHance (generic)
        r1_1_5T=3.80, r1_3_0T=3.28, r1_7_0T=3.21,
        source="Szomolanyi 2019 (human plasma 37°C, all 3 fields)",
    ),
    "gadobutrol": AgentRelaxivity(  # Gadavist/Gadovist
        r1_1_5T=4.78, r1_3_0T=4.97, r1_7_0T=3.83,
        source="Szomolanyi 2019 (human plasma 37°C, all 3 fields)",
    ),

    # High-relaxivity agents
    "gadopiclenol": AgentRelaxivity(  # Elucirem/Vueway
        r1_1_5T=12.8, r1_3_0T=11.6, r1_7_0T=9.6,
        source="Robic 2019 (human serum 37°C, 1.5T-7T)",
    ),
    "gadoquatrane": AgentRelaxivity(  # Ambelvist (tetrameric)
        r1_1_5T=11.8, r1_3_0T=10.5, r1_7_0T=8.0, gd_ions=4,
        source="Lohrke 2022 (per-molecule plasma; ±15% uncertainty)",
    ),
    "gadobenate": AgentRelaxivity(  # MultiHance — protein-binding
        r1_1_5T=6.3, r1_3_0T=5.5, r1_7_0T=None,
        source="Rohrer 2005 plasma; no clean 7T (protein-binding makes "
               "any single value approximate)",
    ),
    "gd-bopta": AgentRelaxivity(  # MultiHance (generic)
        r1_1_5T=6.3, r1_3_0T=5.5, r1_7_0T=None,
        source="Rohrer 2005 plasma; no clean 7T (protein-binding)",
    ),
    "gadoxetate": AgentRelaxivity(  # Eovist/Primovist — hepatobiliary
        r1_1_5T=6.9, r1_3_0T=6.2, r1_7_0T=None,
        source="Rohrer 2005 plasma; no clean 7T human-plasma measurement",
    ),
}

#: Agent name aliases for flexible lookup
AGENT_ALIASES: Final[dict[str, str]] = {
    # Brand names → generic
    "magnevist": "gd-dtpa",
    "omniscan": "gd-dtpa-bma",
    "dotarem": "gadoterate",
    "clariscan": "gadoterate",
    "prohance": "gadoteridol",
    "gadavist": "gadobutrol",
    "gadovist": "gadobutrol",
    "vueway": "gadopiclenol",
    "elucirem": "gadopiclenol",
    "ambelvist": "gadoquatrane",
    "multihance": "gadobenate",
    "eovist": "gadoxetate",
    "primovist": "gadoxetate",
    # Abbreviations
    "dtpa": "gd-dtpa",
    "dota": "gd-dota",
}

#: Default contrast agent if not specified
DEFAULT_AGENT: Final[str] = "gd-dtpa"


# =============================================================================
# Standard Clinical Doses
# =============================================================================

#: Standard clinical doses per agent (mmol/kg).
#: Most agents: 0.1 mmol/kg. High-relaxivity agents use lower doses.
#: Keyed by generic name; brand names resolved via AGENT_ALIASES + normalize_agent_name()
AGENT_STANDARD_DOSES: Final[dict[str, float]] = {
    "gd-dtpa": 0.1,
    "gadopentetate": 0.1,
    "gd-dtpa-bma": 0.1,
    "gadoterate": 0.1,
    "gd-dota": 0.1,
    "gadoteridol": 0.1,
    "gd-hp-do3a": 0.1,
    "gadobutrol": 0.1,
    "gadopiclenol": 0.05,    # Vueway/Elucirem — high relaxivity
    "gadoquatrane": 0.04,    # Ambelvist — tetrameric
    "gadobenate": 0.1,
    "gd-bopta": 0.1,
    "gadoxetate": 0.025,     # Eovist/Primovist — hepatobiliary
}


# =============================================================================
# Contrast Agent Concentrations
# =============================================================================

#: Standard concentrations for contrast agents in mmol/mL
#: Most agents are supplied at 0.5 M (0.5 mmol/mL)
#: Gadobutrol (Gadavist) is supplied at 1.0 M (1.0 mmol/mL)
AGENT_CONCENTRATIONS: Final[dict[str, float]] = {
    "gd-dtpa": 0.5,
    "gadopentetate": 0.5,
    "gd-dtpa-bma": 0.5,
    "gadoterate": 0.5,
    "gd-dota": 0.5,
    "gadoteridol": 0.5,
    "gd-hp-do3a": 0.5,
    "gadobutrol": 1.0,       # Higher concentration (1 M)
    "gadopiclenol": 0.5,
    "gadoquatrane": 0.5,
    "gadobenate": 0.5,
    "gd-bopta": 0.5,
    "gadoxetate": 0.25,      # Lower concentration (0.25 M)
}


# =============================================================================
# Contrast Agent String Parsing (BIDS/DICOM metadata)
# =============================================================================

def parse_contrast_agent_string(agent_string: str) -> dict[str, Any]:
    """Parse contrast agent string from BIDS/DICOM metadata.

    Handles ContrastBolusAgent strings that may include volume info,
    such as "gadovist 3,5 ml" or "Magnevist 10ml".

    Parameters
    ----------
    agent_string : str
        Raw contrast agent string from DICOM/BIDS sidecar

    Returns
    -------
    dict
        Dictionary with:
        - 'agent_name': Extracted agent name (e.g., "gadovist")
        - 'volume_ml': Volume in ml if found, else None
        - 'original': Original input string

    Examples
    --------
        >>> parse_contrast_agent_string("gadovist 3,5 ml")
        {'agent_name': 'gadovist', 'volume_ml': 3.5, 'original': 'gadovist 3,5 ml'}
        >>> parse_contrast_agent_string("Magnevist 10ml")
        {'agent_name': 'Magnevist', 'volume_ml': 10.0, 'original': 'Magnevist 10ml'}
    """
    result: dict[str, Any] = {
        "agent_name": agent_string.strip() if agent_string else "",
        "volume_ml": None,
        "original": agent_string,
    }

    if not agent_string:
        return result

    # Pattern to match volume: number (with comma or period decimal) followed by ml
    volume_pattern = r"(\d+[,.]?\d*)\s*ml"
    match = re.search(volume_pattern, agent_string, re.IGNORECASE)

    if match:
        # Extract volume, converting European comma to period
        volume_str = match.group(1).replace(",", ".")
        try:
            result["volume_ml"] = float(volume_str)
        except ValueError:
            pass

        # Extract agent name: everything before the volume pattern
        agent_name = re.sub(volume_pattern, "", agent_string, flags=re.IGNORECASE).strip()
        agent_name = agent_name.strip(" ,.-")

        if agent_name:
            result["agent_name"] = agent_name
    else:
        # No "ml" suffix found. Try to separate agent name from trailing numbers.
        # Handles European strings like "Vueway 3,5" or "Gadovist 7.5"
        name_num_pattern = r"^([a-zA-Z][a-zA-Z\s\-]*?)\s+(\d+[,.]?\d*)$"
        match2 = re.match(name_num_pattern, agent_string.strip())
        if match2:
            result["agent_name"] = match2.group(1).strip()
            volume_str = match2.group(2).replace(",", ".")
            try:
                result["volume_ml"] = float(volume_str)
            except ValueError:
                pass

    return result


def lookup_r1_safe(
    agent: str,
    field_strength: float = 3.0,
) -> Optional[float]:
    """Look up r1 relaxivity, returning None for unknown agents.

    Unlike lookup_r1(), this function does not raise ValueError for
    unknown agents - it returns None instead, allowing flexible fallback.

    Parameters
    ----------
    agent : str
        Contrast agent name (brand name, generic, or abbreviation)
    field_strength : float
        Magnetic field strength in Tesla

    Returns
    -------
    float or None
        r1 relaxivity in s^-1 per mM, or None if agent not recognized
    """
    try:
        return lookup_r1(agent, field_strength)
    except ValueError:
        return None


def get_r1_from_sidecar(
    sidecar_path: Union[str, Path],
    field_strength: Optional[float] = None,
) -> dict[str, Any]:
    """Extract r1 relaxivity from BIDS JSON sidecar.

    Reads ContrastBolusAgent from dcm2niix JSON sidecar and looks up
    appropriate r1 value. Returns a dict with all relevant information,
    allowing the caller to decide how to handle missing data.

    Parameters
    ----------
    sidecar_path : Path or str
        Path to JSON sidecar file
    field_strength : float, optional
        Field strength in Tesla. If None, reads from sidecar.

    Returns
    -------
    dict
        Dictionary with:
        - 'r1': float or None (r1 in s^-1 mM^-1)
        - 'agent_name': str or None (parsed agent name)
        - 'agent_string': str or None (original ContrastBolusAgent string)
        - 'field_strength': float or None
        - 'volume_ml': float or None (if volume in agent string)
        - 'source': str ('dicom_lookup', 'not_found', 'unknown_agent', 'file_error')

    Examples
    --------
        >>> result = get_r1_from_sidecar("sub-01_dce.json")  # doctest: +SKIP
        >>> if result['r1'] is not None:  # doctest: +SKIP
        ...     print(f"r1={result['r1']:.1f} for {result['agent_name']}")
        ... else:
        ...     print(f"Using default r1 (agent: {result['source']})")
    """
    result: dict[str, Any] = {
        "r1": None,
        "agent_name": None,
        "agent_string": None,
        "field_strength": field_strength,
        "volume_ml": None,
        "source": "not_found",
    }

    json_path = Path(sidecar_path)
    if not json_path.exists():
        result["source"] = "file_error"
        return result

    # Load JSON
    try:
        with open(json_path) as f:
            metadata = json.load(f)
    except Exception:
        result["source"] = "file_error"
        return result

    # Extract field strength if not provided
    if field_strength is None:
        field_strength = metadata.get("MagneticFieldStrength")
        result["field_strength"] = field_strength

    if field_strength is None:
        result["source"] = "no_field_strength"
        return result

    # Try to extract contrast agent name
    agent_string = metadata.get("ContrastBolusAgent", "")
    result["agent_string"] = agent_string

    if not agent_string or not agent_string.strip():
        result["source"] = "not_in_dicom"
        return result

    # Parse agent string
    parsed = parse_contrast_agent_string(agent_string)
    result["agent_name"] = parsed["agent_name"]
    result["volume_ml"] = parsed["volume_ml"]

    # Look up r1
    r1 = lookup_r1_safe(parsed["agent_name"], field_strength)

    if r1 is not None:
        result["r1"] = r1
        result["source"] = "dicom_lookup"
    else:
        result["source"] = "unknown_agent"

    return result


def get_r1_from_sidecar_dict(
    sidecar: dict[str, Any],
    field_strength: Optional[float] = None,
) -> dict[str, Any]:
    """Extract r1 relaxivity from BIDS sidecar dictionary.

    Like get_r1_from_sidecar() but takes an already-loaded dict.

    Parameters
    ----------
    sidecar : dict
        BIDS JSON sidecar as dictionary
    field_strength : float, optional
        Field strength in Tesla. If None, reads from sidecar.

    Returns
    -------
    dict
        Same dict format as get_r1_from_sidecar()
    """
    result: dict[str, Any] = {
        "r1": None,
        "agent_name": None,
        "agent_string": None,
        "field_strength": field_strength,
        "volume_ml": None,
        "source": "not_found",
    }

    # Extract field strength if not provided
    if field_strength is None:
        field_strength = sidecar.get("MagneticFieldStrength")
        result["field_strength"] = field_strength

    if field_strength is None:
        result["source"] = "no_field_strength"
        return result

    # Try to extract contrast agent name
    agent_string = sidecar.get("ContrastBolusAgent", "")
    result["agent_string"] = agent_string

    if not agent_string or not agent_string.strip():
        result["source"] = "not_in_dicom"
        return result

    # Parse agent string
    parsed = parse_contrast_agent_string(agent_string)
    result["agent_name"] = parsed["agent_name"]
    result["volume_ml"] = parsed["volume_ml"]

    # Look up r1
    r1 = lookup_r1_safe(parsed["agent_name"], field_strength)

    if r1 is not None:
        result["r1"] = r1
        result["source"] = "dicom_lookup"
    else:
        result["source"] = "unknown_agent"

    return result


# =============================================================================
# R1 Relaxivity Lookup Functions
# =============================================================================

def normalize_agent_name(agent: str) -> str:
    """Normalize agent name for lookup.

    Parameters
    ----------
    agent : str
        Agent name (brand name, generic, or abbreviation)

    Returns
    -------
    str
        Normalized lowercase agent name
    """
    name = agent.lower().strip()
    # Check aliases first
    if name in AGENT_ALIASES:
        return AGENT_ALIASES[name]
    return name


def lookup_standard_dose(agent: str) -> float:
    """Look up standard clinical dose for a contrast agent.

    Uses normalize_agent_name() to resolve brand names via AGENT_ALIASES.
    E.g., "Vueway" -> "gadopiclenol" -> 0.05 mmol/kg

    Parameters
    ----------
    agent : str
        Contrast agent name (brand name, generic, or abbreviation).

    Returns
    -------
    float
        Standard dose in mmol/kg (0.1 for most agents, 0.05 for Vueway, etc.).
    """
    name = normalize_agent_name(agent)
    return AGENT_STANDARD_DOSES.get(name, 0.1)


def lookup_gd_multiplicity(agent: str) -> int:
    """Look up Gd3+ ion count per molecule for a contrast agent.

    Most GBCAs are monomeric (1 Gd per molecule). Gadoquatrane is
    tetrameric (4 Gd per molecule). This affects DSC susceptibility
    scaling but NOT DCE (where per-molecule r1 already accounts for
    all Gd ions).

    Parameters
    ----------
    agent : str
        Contrast agent name (brand name, generic, or abbreviation).

    Returns
    -------
    int
        Number of Gd3+ ions per molecule (1 for most agents, 4 for
        gadoquatrane).
    """
    name = normalize_agent_name(agent)
    relaxivity = CONTRAST_AGENTS.get(name)
    return relaxivity.gd_ions if relaxivity else 1


def lookup_r1(
    agent: str = DEFAULT_AGENT,
    field_strength: float = 3.0,
) -> float:
    """Look up r1 relaxivity for a contrast agent at a given field strength.

    Parameters
    ----------
    agent : str
        Contrast agent name (brand name, generic, or abbreviation).
        Examples: "gadobutrol", "Gadavist", "gd-dtpa", "Magnevist"
    field_strength : float
        Magnetic field strength in Tesla (default 3.0T)

    Returns
    -------
    float
        r1 relaxivity in L/(mmol·s) or equivalently s^-1 per mM

    Raises
    ------
    ValueError
        If the agent name is not recognized

    Examples
    --------
        >>> lookup_r1("gadobutrol", 3.0)
        4.97
        >>> lookup_r1("gd-dtpa", 1.5)
        4.1
        >>> lookup_r1("Gadavist", 3.0)  # Brand name
        4.97
    """
    name = normalize_agent_name(agent)

    if name not in CONTRAST_AGENTS:
        available = sorted(set(CONTRAST_AGENTS.keys()) | set(AGENT_ALIASES.keys()))
        raise ValueError(
            f"Unknown contrast agent: '{agent}'. "
            f"Available agents: {', '.join(available)}"
        )

    relaxivity = CONTRAST_AGENTS[name]

    # Three field-strength bins, midpoints at 2.25T and 5.0T:
    #   B0 ≤ 2.25T → 1.5T value
    #   2.25T < B0 ≤ 5.0T → 3.0T value
    #   B0 > 5.0T → 7.0T value (raise if not measured for this agent)
    # No linear interpolation: r1(B0) is concave (NMRD profile), and the
    # 1.5T/3T/7T values come from different studies for some agents, so
    # interpolation can mix sources.
    if field_strength <= 2.25:
        return relaxivity.r1_1_5T
    if field_strength <= 5.0:
        return relaxivity.r1_3_0T
    if relaxivity.r1_7_0T is None:
        raise ValueError(
            f"No peer-reviewed 7T r1 measurement for agent '{name}' "
            f"(B0={field_strength}T). Verify the contrast agent and field "
            "strength against literature, or use a different agent. "
            f"Citation: {relaxivity.source}"
        )
    return relaxivity.r1_7_0T


def get_available_agents() -> list[str]:
    """Get list of all available contrast agent names.

    Returns
    -------
    list
        Sorted list of recognized agent names (generic names only)
    """
    return sorted(CONTRAST_AGENTS.keys())


def get_agent_info(agent: str) -> dict[str, float]:
    """Get detailed information about a contrast agent.

    Parameters
    ----------
    agent : str
        Contrast agent name

    Returns
    -------
    dict
        Dict with r1_1_5T, r1_3_0T, and concentration

    Raises
    ------
    ValueError
        If the agent name is not recognized
    """
    name = normalize_agent_name(agent)

    if name not in CONTRAST_AGENTS:
        raise ValueError(f"Unknown contrast agent: '{agent}'")

    relaxivity = CONTRAST_AGENTS[name]
    concentration = AGENT_CONCENTRATIONS.get(name, 0.5)

    return {
        "r1_1_5T": relaxivity.r1_1_5T,
        "r1_3_0T": relaxivity.r1_3_0T,
        "r1_7_0T": (
            relaxivity.r1_7_0T if relaxivity.r1_7_0T is not None else float("nan")
        ),
        "concentration_mmol_per_ml": concentration,
    }


# =============================================================================
# Dose Calculation Functions
# =============================================================================

def calculate_dose_from_volume(
    volume_ml: float,
    body_weight_kg: float,
    agent: str = DEFAULT_AGENT,
) -> float:
    """Calculate contrast agent dose in mmol/kg from injected volume.

    Standard dose is typically 0.1 mmol/kg (single dose) or 0.05 mmol/kg (half dose).

    Parameters
    ----------
    volume_ml : float
        Injected volume in milliliters
    body_weight_kg : float
        Patient body weight in kilograms
    agent : str
        Contrast agent name (affects concentration)

    Returns
    -------
    float
        Dose in mmol/kg

    Raises
    ------
    ValueError
        If volume or weight is non-positive, or agent is unknown

    Examples
    --------
        >>> # 10 mL of 0.5 M Gd-DTPA for 70 kg patient
        >>> calculate_dose_from_volume(10.0, 70.0, "gd-dtpa")
        0.0714...
        >>> # 5 mL of 1.0 M Gadobutrol for 70 kg patient (same dose)
        >>> calculate_dose_from_volume(5.0, 70.0, "gadobutrol")
        0.0714...
    """
    if volume_ml <= 0:
        raise ValueError(f"Volume must be positive, got {volume_ml}")
    if body_weight_kg <= 0:
        raise ValueError(f"Body weight must be positive, got {body_weight_kg}")

    name = normalize_agent_name(agent)

    if name not in AGENT_CONCENTRATIONS:
        # Check if it's at least a known alias
        if name not in CONTRAST_AGENTS:
            raise ValueError(f"Unknown contrast agent: '{agent}'")
        # Default concentration if not explicitly listed
        concentration = 0.5
    else:
        concentration = AGENT_CONCENTRATIONS[name]

    # Calculate: volume (mL) × concentration (mmol/mL) / weight (kg) = dose (mmol/kg)
    total_mmol = volume_ml * concentration
    dose_mmol_per_kg = total_mmol / body_weight_kg

    return dose_mmol_per_kg


def calculate_volume_for_dose(
    dose_mmol_per_kg: float,
    body_weight_kg: float,
    agent: str = DEFAULT_AGENT,
) -> float:
    """Calculate injection volume required for a target dose.

    Parameters
    ----------
    dose_mmol_per_kg : float
        Target dose in mmol/kg (0.1 for standard, 0.05 for half)
    body_weight_kg : float
        Patient body weight in kilograms
    agent : str
        Contrast agent name (affects concentration)

    Returns
    -------
    float
        Required volume in milliliters

    Raises
    ------
    ValueError
        If dose or weight is non-positive, or agent is unknown

    Examples
    --------
        >>> # Standard dose (0.1 mmol/kg) of Gd-DTPA for 70 kg patient
        >>> calculate_volume_for_dose(0.1, 70.0, "gd-dtpa")
        14.0
        >>> # Standard dose of Gadobutrol for 70 kg (half volume due to 1M conc)
        >>> calculate_volume_for_dose(0.1, 70.0, "gadobutrol")
        7.0
    """
    if dose_mmol_per_kg <= 0:
        raise ValueError(f"Dose must be positive, got {dose_mmol_per_kg}")
    if body_weight_kg <= 0:
        raise ValueError(f"Body weight must be positive, got {body_weight_kg}")

    name = normalize_agent_name(agent)

    if name not in AGENT_CONCENTRATIONS:
        if name not in CONTRAST_AGENTS:
            raise ValueError(f"Unknown contrast agent: '{agent}'")
        concentration = 0.5
    else:
        concentration = AGENT_CONCENTRATIONS[name]

    # Calculate: dose (mmol/kg) × weight (kg) / concentration (mmol/mL) = volume (mL)
    total_mmol = dose_mmol_per_kg * body_weight_kg
    volume_ml = total_mmol / concentration

    return volume_ml


def classify_dose(dose_mmol_per_kg: float) -> str:
    """Classify a contrast agent dose.

    Parameters
    ----------
    dose_mmol_per_kg : float
        Dose in mmol/kg

    Returns
    -------
    str
        Dose classification: "none", "quarter", "half", "full", "double", or "high"
    """
    if dose_mmol_per_kg < 0.01:
        return "none"
    elif dose_mmol_per_kg < 0.04:
        return "quarter"
    elif dose_mmol_per_kg < 0.08:
        return "half"
    elif dose_mmol_per_kg < 0.15:
        return "full"
    elif dose_mmol_per_kg < 0.25:
        return "double"
    else:
        return "high"


# =============================================================================
# Provenance Source Tracking
# =============================================================================


class ParameterSource:
    """Constants for tracking where processing parameters originated.

    Used in provenance dicts to record whether a parameter was explicitly
    provided by the user, auto-detected from BIDS metadata, or fell back
    to a pipeline default.
    """

    USER_OVERRIDE = "user_override"
    BIDS_SIDECAR = "bids_sidecar"
    DEFAULT_FALLBACK = "default_fallback"
    USER_SPECIFIED = "user_specified"
    AUTO_DETECTED = "auto_detected"


#: Canonical brand names for each generic agent (deterministic, not
#: dependent on AGENT_ALIASES iteration order).
_PRIMARY_BRANDS: Final[dict[str, str]] = {
    "gd-dtpa": "Magnevist",
    "gadopentetate": "Magnevist",
    "gd-dtpa-bma": "Omniscan",
    "gadoterate": "Dotarem",
    "gd-dota": "Dotarem",
    "gadoteridol": "ProHance",
    "gd-hp-do3a": "ProHance",
    "gadobutrol": "Gadavist",
    "gadopiclenol": "Vueway",
    "gadoquatrane": "Ambelvist",
    "gadobenate": "MultiHance",
    "gd-bopta": "MultiHance",
    "gadoxetate": "Eovist",
}


def get_brand_name(generic_name: str) -> Optional[str]:
    """Reverse lookup: generic agent name to primary brand name.

    Parameters
    ----------
    generic_name : str
        Generic or chemical agent name (e.g., "gadobutrol").

    Returns
    -------
    str or None
        Primary brand name (e.g., "Gadavist"), or None if unknown.
    """
    return _PRIMARY_BRANDS.get(generic_name.lower().strip())


def build_dce_provenance(
    *,
    contrast_agent: Optional[str] = None,
    agent_source: str = ParameterSource.DEFAULT_FALLBACK,
    r1_relaxivity: float = 3.8,
    field_strength: float = 3.0,
    dose_multiplier: float = 1.0,
    dose_source: str = ParameterSource.DEFAULT_FALLBACK,
    agent_standard_dose: float = 0.1,
    bids_agent_string: Optional[str] = None,
    scanner_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build DCE provenance dict for output JSON sidecars.

    All values are explicitly cast to Python scalars to ensure
    consistent nipype hashing and JSON serialization.

    Parameters
    ----------
    contrast_agent : str, optional
        Generic agent name (e.g., "gadobutrol").
    agent_source : str
        How the agent was determined (ParameterSource constant).
    r1_relaxivity : float
        R1 relaxivity in s^-1/mM.
    field_strength : float
        Magnetic field strength in Tesla.
    dose_multiplier : float
        Dose multiplier (0.5=half, 1.0=full, 2.0=double).
    dose_source : str
        How the dose was determined (ParameterSource constant).
    agent_standard_dose : float
        Standard clinical dose in mmol/kg.
    bids_agent_string : str, optional
        Raw ContrastBolusAgent string from BIDS sidecar.
    scanner_metadata : dict, optional
        Acquisition metadata from BIDS sidecar (TR/TE/
        FlipAngle/etc.) emitted under PerfusionProcessing.ScannerMetadata
        so downstream sidecars carry the acquisition context directly
        without requiring users to traverse Sources.

    Returns
    -------
    dict
        Dict with PerfusionProcessing namespace containing ContrastAgent,
        Relaxivity, and ContrastDose sub-dicts.
    """
    agent_name = str(contrast_agent) if contrast_agent else DEFAULT_AGENT
    brand = get_brand_name(agent_name)

    agent_dict: dict[str, Any] = {
        "GenericName": str(agent_name),
        "Source": str(agent_source),
    }
    if brand is not None:
        agent_dict["BrandName"] = str(brand)
    if bids_agent_string is not None:
        agent_dict["OriginalString"] = str(bids_agent_string)

    perf: dict[str, Any] = {
        "ContrastAgent": agent_dict,
        "Relaxivity": {
            "r1": float(r1_relaxivity),
            "units": "s-1.mM-1",
            "field_strength_T": float(field_strength),
            "Source": str(agent_source),
        },
        "ContrastDose": {
            "standard_dose_mmol_kg": float(agent_standard_dose),
            "dose_multiplier": float(dose_multiplier),
            "Source": str(dose_source),
        },
    }
    if scanner_metadata:
        block = _build_scanner_metadata_block(scanner_metadata)
        if block:
            perf["ScannerMetadata"] = block
    return {"PerfusionProcessing": perf}


def build_dsc_provenance(
    *,
    contrast_agent: Optional[str] = None,
    agent_source: str = ParameterSource.DEFAULT_FALLBACK,
    bids_agent_string: Optional[str] = None,
    dose_multiplier: float = 1.0,
    dose_source: str = ParameterSource.DEFAULT_FALLBACK,
    agent_standard_dose: float = 0.1,
    agent_standard_dose_source: str = ParameterSource.DEFAULT_FALLBACK,
    gd_multiplicity: int = 1,
    preload_source: str = "unknown",
    deconv_method: str = "oSVD",
    inference: Optional[str] = None,
    leakage_method: str = "bsw",
    scanner_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build DSC provenance dict for output JSON sidecars.

    DSC quantification uses relative metrics (r2* cancels in CBV/CBF),
    so contrast agent relaxivity is not included. Agent identity is
    recorded for reproducibility; dose and preload affect AIF detection
    thresholds and leakage correction.

    Parameters
    ----------
    contrast_agent : str, optional
        Generic agent name (e.g., "gadopiclenol").
    agent_source : str
        How the agent was determined (ParameterSource constant).
    bids_agent_string : str, optional
        Raw ContrastBolusAgent string from BIDS sidecar.
    dose_multiplier : float
        Dose multiplier (0.5=half, 1.0=full, 2.0=double).
    dose_source : str
        How the dose was determined (ParameterSource constant).
    agent_standard_dose : float
        Standard clinical molecular dose in mmol/kg.
    agent_standard_dose_source : str
        How agent std dose was determined.
    gd_multiplicity : int
        Gd3+ ions per molecule (1 for monomeric, 4 for
        gadoquatrane). Recorded in provenance when != 1.
    preload_source : str
        Whether the administered preload was user-specified or is unknown.
    deconv_method : str
        Deconvolution method name.
    inference : str, optional
        Inference method (nlls/map/vb/svb) for CPI/vascular.
    leakage_method : str
        Leakage correction method name.
    scanner_metadata : dict, optional
        Acquisition metadata from BIDS sidecar (TR/TE/
        FlipAngle/FieldStrength) emitted under
        PerfusionProcessing.ScannerMetadata.

    Returns
    -------
    dict
        Dict with PerfusionProcessing namespace containing ContrastAgent
        (when agent is known), ContrastDose, Deconvolution, and
        LeakageCorrection sub-dicts.

    Notes
    -----
        The ``PerfusionProcessing.AIF`` sub-block is assembled
        downstream at sidecar-write time by
        :func:`pmriprep.utils.aif_provenance.apply_aif_sidecar`, which
        reads post-detection ``qc_metrics['aif']``. This function runs
        at workflow construction, before AIF detection, so it cannot
        populate AIF fields itself.
    """
    deconv_block: dict[str, Any] = {"method": str(deconv_method)}
    if inference is not None and deconv_method in ("cpi", "vascular"):
        deconv_block["inference"] = str(inference)
        # Identify the executing implementation separately from the published
        # parameterization/prior lineage used by the CPI path.
        if deconv_method == "cpi":
            deconv_block["inference_engine"] = "pmriprep"
            deconv_block["cpi_parameterization"] = "fabber_derived"
            deconv_block["cpi_prior_profile"] = "fabber_derived"
            deconv_block["cpi_cbf_prior"] = {
                "transform": "identity",
                "mean": 1.0,
                "var": 1e6,
            }
            deconv_block["cpi_fraction_prior"] = {
                "transform": "sigmoid",
                "fitting_space_mean": 0.5,
                "fitting_space_var": 1e7,
            }
            deconv_block["vb_max_iterations"] = 10
            deconv_block["vb_noise_prior"] = {"c0": 1e-6, "s0": 1e6}

    perf: dict[str, Any] = {}

    if contrast_agent is not None:
        agent_name = str(contrast_agent)
        brand = get_brand_name(agent_name)
        agent_dict: dict[str, Any] = {
            "GenericName": str(agent_name),
            "Source": str(agent_source),
        }
        if brand is not None:
            agent_dict["BrandName"] = str(brand)
        if bids_agent_string is not None:
            agent_dict["OriginalString"] = str(bids_agent_string)
        perf["ContrastAgent"] = agent_dict

    dose_dict: dict[str, Any] = {
        "standard_dose_mmol_kg": float(agent_standard_dose),
        "dose_multiplier": float(dose_multiplier),
        "Source": str(dose_source),
        "AgentStandardDoseSource": str(agent_standard_dose_source),
    }
    if gd_multiplicity != 1:
        dose_dict["gd_ions_per_molecule"] = int(gd_multiplicity)
        dose_dict["effective_gd_dose_mmol_kg"] = float(
            agent_standard_dose * dose_multiplier * gd_multiplicity
        )
    perf["ContrastDose"] = dose_dict
    perf["Deconvolution"] = deconv_block
    perf["LeakageCorrection"] = {"method": str(leakage_method)}
    perf["PreloadDetection"] = {"source": str(preload_source)}
    if scanner_metadata:
        block = _build_scanner_metadata_block(scanner_metadata)
        if block:
            perf["ScannerMetadata"] = block

    return {"PerfusionProcessing": perf}


def _build_scanner_metadata_block(
    sidecar: dict[str, Any],
) -> dict[str, Any]:
    """Extract acquisition metadata for the ScannerMetadata sidecar block.

    Mirrors the field names returned by ``_read_bids_sidecar``: ``te``,
    ``volume_tr``, ``pulse_tr``, ``field_strength``, ``flip_angle``.
    Each is emitted under its BIDS-spec name with original units (TE/TR
    in seconds, FlipAngle in degrees, MagneticFieldStrength in Tesla).
    Returns an empty dict when no recognized fields are populated; the
    caller should skip emission rather than write an empty block.
    """
    bids_keys = {
        "te": "EchoTime",
        "volume_tr": "RepetitionTime",
        "pulse_tr": "RepetitionTimeExcitation",
        "field_strength": "MagneticFieldStrength",
        "flip_angle": "FlipAngle",
    }
    block: dict[str, Any] = {}
    for src_key, bids_key in bids_keys.items():
        if src_key in sidecar and sidecar[src_key] is not None:
            block[bids_key] = float(sidecar[src_key])
    return block


def collect_default_warnings(
    provenance_dict: dict[str, Any],
    modality: str,
) -> list[str]:
    """Generate human-readable warnings for defaulted parameters.

    Inspects Source fields in the provenance dict and returns warning
    strings for any parameters that fell back to pipeline defaults.

    Parameters
    ----------
    provenance_dict : dict
        Provenance dict from build_dce_provenance or
        build_dsc_provenance.
    modality : str
        "DCE" or "DSC".

    Returns
    -------
    list
        List of warning strings. Empty if all parameters were
        user-specified or BIDS-detected.
    """
    warnings: list[str] = []
    perf = provenance_dict.get("PerfusionProcessing", {})

    if modality == "DCE":
        # Check contrast agent source
        agent = perf.get("ContrastAgent", {})
        if agent.get("Source") == ParameterSource.DEFAULT_FALLBACK:
            r1_info = perf.get("Relaxivity", {})
            r1_val = r1_info.get("r1", "?")
            fs_val = r1_info.get("field_strength_T", "?")
            agent_name = agent.get("GenericName", DEFAULT_AGENT)
            brand = agent.get("BrandName", "")
            brand_str = f" ({brand})" if brand else ""
            warnings.append(
                f"Contrast agent defaulted to {agent_name}{brand_str} "
                f"(r1={r1_val} s-1/mM at {fs_val}T). "
                f"Specify ContrastBolusAgent in BIDS sidecar or use --dce-r1."
            )

        # Check dose source
        dose = perf.get("ContrastDose", {})
        if dose.get("Source") == ParameterSource.DEFAULT_FALLBACK:
            mult = dose.get("dose_multiplier", 1.0)
            std = dose.get("standard_dose_mmol_kg", 0.1)
            warnings.append(
                f"DCE contrast dose assumed standard "
                f"({mult}x, {std} mmol/kg). Use --dce-dose to specify."
            )

    elif modality == "DSC":
        # Check contrast agent source (DSC omits ContrastAgent when unknown,
        # unlike DCE which always includes it — guard against empty dict)
        agent = perf.get("ContrastAgent", {})
        if agent and agent.get("Source") == ParameterSource.DEFAULT_FALLBACK:
            agent_name = agent.get("GenericName", DEFAULT_AGENT)
            brand = agent.get("BrandName", "")
            brand_str = f" ({brand})" if brand else ""
            warnings.append(
                f"DSC contrast agent defaulted to {agent_name}{brand_str}. "
                f"Specify ContrastBolusAgent in BIDS sidecar."
            )

        # Check dose source
        dose = perf.get("ContrastDose", {})
        if dose.get("Source") == ParameterSource.DEFAULT_FALLBACK:
            mult = dose.get("dose_multiplier", 1.0)
            std = dose.get("standard_dose_mmol_kg", 0.1)
            warnings.append(
                f"DSC contrast dose assumed standard "
                f"({mult}x, {std} mmol/kg). Use --dsc-dose to specify."
            )

        # Check agent standard dose source
        if dose.get("AgentStandardDoseSource") == ParameterSource.DEFAULT_FALLBACK:
            std = dose.get("standard_dose_mmol_kg", 0.1)
            warnings.append(
                f"DSC agent standard dose assumed {std} mmol/kg. "
                f"Use --dsc-agent-standard-dose for non-standard agents "
                f"(e.g., 0.05 for Vueway)."
            )

    return warnings


def collect_aif_fallback_warning(
    qc_metrics: Optional[dict[str, Any]],
    modality: str,
) -> Optional[str]:
    """Return a warning string if AIF fell back to population/template.

    Parameters
    ----------
    qc_metrics : dict, optional
        QC metrics dict containing AIF detection results.
    modality : str
        ``"DSC"`` or ``"DCE"``.

    Returns
    -------
    str or None
        Warning string if AIF used population/template, else ``None``.
    """
    if not qc_metrics:
        return None

    if modality == "DSC":
        aif_qc = qc_metrics.get("aif_detection") or {}
        # A real AIF retained below the gamma-fit floor (synthetic fallback
        # disabled) is flagged but NOT swapped, so it wouldn't otherwise surface.
        if aif_qc.get("aif_gamma_fit_below_floor_retained"):
            r2 = aif_qc.get("gamma_r2_retained")
            r2_txt = ""
            if r2 is not None:
                try:
                    r2_txt = f" (R²={float(r2):.2f})"
                except (TypeError, ValueError):
                    pass
            return (
                f"DSC AIF retained despite a poor gamma-variate fit{r2_txt} "
                f"— synthetic fallback is disabled. CBF, MTT, and Tmax may "
                f"be unreliable; AUC-nCBV and PSR remain interpretable."
            )
        if aif_qc.get("detection_method") != "synthetic":
            return None
        reason_code = str(
            qc_metrics.get("synthetic_fallback_reason")
            or aif_qc.get("synthetic_fallback_reason")
            or "cascade_exhausted"
        )
        reason = {
            "cascade_exhausted": "automatic detection found no acceptable arterial curve",
            "no_candidates": "no arterial candidates survived automatic detection",
            "low_snr": "arterial candidates had insufficient signal quality",
            "gamma_r2_floor": "the measured curve had a poor gamma-variate fit",
        }.get(reason_code, "automatic detection found no acceptable arterial curve")
        return (
            f"DSC AIF fell back to Parker population template "
            f"\u2014 {reason}. "
            f"nCBV and nCBF remain usable (NAWM normalisation "
            f"cancels AIF scaling). K2-corrected rCBV inherits "
            f"template-AIF shape bias on the first-pass integral. "
            f"MTT / Tmax are qualitative only."
        )

    if modality == "DCE":
        aif_qc = qc_metrics.get("aif_detection") or {}
        if aif_qc.get("aif_source") != "population":
            return None
        if aif_qc.get("auto_detection_attempted"):
            detail = " — automatic detection found no acceptable arterial curve"
        else:
            detail = " (user-selected)"
        return (
            f"DCE AIF using Parker population model{detail}. "
            f"Individual arterial physiology is not captured; "
            f"Ktrans may be biased by AIF peak/shape mismatch."
        )

    return None
