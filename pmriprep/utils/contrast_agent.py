"""Contrast agent utilities for DCE-MRI processing.

Provides relaxivity lookups for common gadolinium-based contrast agents
and dose calculation utilities.
"""

import json
import re
from pathlib import Path
from typing import Any, Final, NamedTuple, Optional, Union


# =============================================================================
# Contrast Agent Relaxivity Data
# =============================================================================

class AgentRelaxivity(NamedTuple):
    """R1 relaxivity values at different field strengths.

    Values are in L/(mmol·s) or equivalently s^-1 per mM.
    """
    r1_1_5T: float  # R1 relaxivity at 1.5T
    r1_3_0T: float  # R1 relaxivity at 3.0T


#: R1 relaxivity values for common gadolinium-based contrast agents
#: References:
#:   - Rohrer et al. (2005) Invest Radiol
#:   - Shen et al. (2015) Invest Radiol
#:   - Frenzel et al. (2017) Invest Radiol
#:   - Aime et al. (2019) J Magn Reson Imaging
CONTRAST_AGENTS: Final[dict[str, AgentRelaxivity]] = {
    # Linear agents (lower relaxivity, higher stability)
    "gd-dtpa": AgentRelaxivity(r1_1_5T=4.1, r1_3_0T=3.7),        # Magnevist
    "gadopentetate": AgentRelaxivity(r1_1_5T=4.1, r1_3_0T=3.7),  # Magnevist (generic)
    "gd-dtpa-bma": AgentRelaxivity(r1_1_5T=4.0, r1_3_0T=3.6),    # Omniscan

    # Macrocyclic agents (higher stability, preferred clinically)
    "gadoterate": AgentRelaxivity(r1_1_5T=3.6, r1_3_0T=3.5),     # Dotarem
    "gd-dota": AgentRelaxivity(r1_1_5T=3.6, r1_3_0T=3.5),        # Dotarem (generic)
    "gadoteridol": AgentRelaxivity(r1_1_5T=4.1, r1_3_0T=3.7),    # ProHance
    "gd-hp-do3a": AgentRelaxivity(r1_1_5T=4.1, r1_3_0T=3.7),     # ProHance (generic)
    "gadobutrol": AgentRelaxivity(r1_1_5T=5.2, r1_3_0T=5.0),     # Gadavist/Gadovist

    # High-relaxivity agents
    "gadopiclenol": AgentRelaxivity(r1_1_5T=12.8, r1_3_0T=11.6), # Elucirem (newest)
    "gadobenate": AgentRelaxivity(r1_1_5T=6.3, r1_3_0T=5.5),     # MultiHance
    "gd-bopta": AgentRelaxivity(r1_1_5T=6.3, r1_3_0T=5.5),       # MultiHance (generic)
    "gadoxetate": AgentRelaxivity(r1_1_5T=6.9, r1_3_0T=6.2),     # Eovist/Primovist
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

    Args:
        agent_string: Raw contrast agent string from DICOM/BIDS sidecar

    Returns:
        Dictionary with:
        - 'agent_name': Extracted agent name (e.g., "gadovist")
        - 'volume_ml': Volume in ml if found, else None
        - 'original': Original input string

    Examples:
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

    return result


def lookup_r1_safe(
    agent: str,
    field_strength: float = 3.0,
) -> Optional[float]:
    """Look up r1 relaxivity, returning None for unknown agents.

    Unlike lookup_r1(), this function does not raise ValueError for
    unknown agents - it returns None instead, allowing flexible fallback.

    Args:
        agent: Contrast agent name (brand name, generic, or abbreviation)
        field_strength: Magnetic field strength in Tesla

    Returns:
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

    Args:
        sidecar_path: Path to JSON sidecar file
        field_strength: Field strength in Tesla. If None, reads from sidecar.

    Returns:
        Dictionary with:
        - 'r1': float or None (r1 in s^-1 mM^-1)
        - 'agent_name': str or None (parsed agent name)
        - 'agent_string': str or None (original ContrastBolusAgent string)
        - 'field_strength': float or None
        - 'volume_ml': float or None (if volume in agent string)
        - 'source': str ('dicom_lookup', 'not_found', 'unknown_agent', 'file_error')

    Examples:
        >>> result = get_r1_from_sidecar("sub-01_dce.json")
        >>> if result['r1'] is not None:
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

    Args:
        sidecar: BIDS JSON sidecar as dictionary
        field_strength: Field strength in Tesla. If None, reads from sidecar.

    Returns:
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

def _normalize_agent_name(agent: str) -> str:
    """Normalize agent name for lookup.

    Args:
        agent: Agent name (brand name, generic, or abbreviation)

    Returns:
        Normalized lowercase agent name
    """
    name = agent.lower().strip()
    # Check aliases first
    if name in AGENT_ALIASES:
        return AGENT_ALIASES[name]
    return name


def lookup_r1(
    agent: str = DEFAULT_AGENT,
    field_strength: float = 3.0,
) -> float:
    """Look up r1 relaxivity for a contrast agent at a given field strength.

    Args:
        agent: Contrast agent name (brand name, generic, or abbreviation).
               Examples: "gadobutrol", "Gadavist", "gd-dtpa", "Magnevist"
        field_strength: Magnetic field strength in Tesla (default 3.0T)

    Returns:
        r1 relaxivity in L/(mmol·s) or equivalently s^-1 per mM

    Raises:
        ValueError: If the agent name is not recognized

    Examples:
        >>> lookup_r1("gadobutrol", 3.0)
        5.0
        >>> lookup_r1("gd-dtpa", 1.5)
        4.1
        >>> lookup_r1("Gadavist", 3.0)  # Brand name
        5.0
    """
    name = _normalize_agent_name(agent)

    if name not in CONTRAST_AGENTS:
        available = sorted(set(CONTRAST_AGENTS.keys()) | set(AGENT_ALIASES.keys()))
        raise ValueError(
            f"Unknown contrast agent: '{agent}'. "
            f"Available agents: {', '.join(available)}"
        )

    relaxivity = CONTRAST_AGENTS[name]

    # Select appropriate field strength
    if field_strength <= 2.0:
        # Use 1.5T value for low field
        return relaxivity.r1_1_5T
    elif field_strength >= 2.5:
        # Use 3.0T value for high field
        return relaxivity.r1_3_0T
    else:
        # Linear interpolation between 1.5T and 3.0T for intermediate fields
        # Note: This is a rough approximation; relaxivity changes are not strictly linear
        fraction = (field_strength - 1.5) / (3.0 - 1.5)
        return relaxivity.r1_1_5T + fraction * (relaxivity.r1_3_0T - relaxivity.r1_1_5T)


def get_available_agents() -> list[str]:
    """Get list of all available contrast agent names.

    Returns:
        Sorted list of recognized agent names (generic names only)
    """
    return sorted(CONTRAST_AGENTS.keys())


def get_agent_info(agent: str) -> dict[str, float]:
    """Get detailed information about a contrast agent.

    Args:
        agent: Contrast agent name

    Returns:
        Dict with r1_1_5T, r1_3_0T, and concentration

    Raises:
        ValueError: If the agent name is not recognized
    """
    name = _normalize_agent_name(agent)

    if name not in CONTRAST_AGENTS:
        raise ValueError(f"Unknown contrast agent: '{agent}'")

    relaxivity = CONTRAST_AGENTS[name]
    concentration = AGENT_CONCENTRATIONS.get(name, 0.5)

    return {
        "r1_1_5T": relaxivity.r1_1_5T,
        "r1_3_0T": relaxivity.r1_3_0T,
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

    Args:
        volume_ml: Injected volume in milliliters
        body_weight_kg: Patient body weight in kilograms
        agent: Contrast agent name (affects concentration)

    Returns:
        Dose in mmol/kg

    Raises:
        ValueError: If volume or weight is non-positive, or agent is unknown

    Examples:
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

    name = _normalize_agent_name(agent)

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

    Args:
        dose_mmol_per_kg: Target dose in mmol/kg (0.1 for standard, 0.05 for half)
        body_weight_kg: Patient body weight in kilograms
        agent: Contrast agent name (affects concentration)

    Returns:
        Required volume in milliliters

    Raises:
        ValueError: If dose or weight is non-positive, or agent is unknown

    Examples:
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

    name = _normalize_agent_name(agent)

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

    Args:
        dose_mmol_per_kg: Dose in mmol/kg

    Returns:
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
