"""Pre-flight validation for PMRIPrep metadata completeness.

Validates BIDS sidecar metadata and acquisition parameters BEFORE
workflow initialization, catching common errors early with clear
diagnostic messages rather than cryptic failures deep in processing.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pmriprep.config.dce import DCEConfig
from pmriprep.config.dsc import DSCConfig
from pmriprep.utils.contrast_agent import (
    lookup_gd_multiplicity,
    lookup_r1_safe,
    lookup_standard_dose,
    normalize_agent_name,
)

logger = logging.getLogger("pmriprep.preflight")

HIGH_R1_THRESHOLD: float = 8.0  # s⁻¹/mM — above this, T1 effects dominate
LOW_GD_DOSE_THRESHOLD: float = 0.05  # mmol-Gd/kg — DSC deconvolution floor
LOW_DCE_DOSE_R1_THRESHOLD: float = 0.15  # kg⁻¹·s⁻¹ — DCE tissue enhancement floor

# Temporal sufficiency: baseline (~8-10) + first-pass (~10) + post-bolus (~10)
DSC_MIN_VOLUMES: int = 30
DSC_MARGINAL_VOLUMES: int = 40
DSC_MIN_DURATION_S: float = 40.0
DSC_MARGINAL_DURATION_S: float = 60.0

# Temporal sufficiency: baseline (~5) + wash-in (~3) + washout (~5+)
DCE_MIN_VOLUMES: int = 15
DCE_MARGINAL_VOLUMES: int = 20
DCE_MIN_DURATION_S: float = 60.0
DCE_MARGINAL_DURATION_S: float = 120.0


class PreflightError(RuntimeError):
    """Raised when pre-flight validation finds fatal errors."""


@dataclass
class PreflightMessage:
    """A single validation message.

    Attributes
    ----------
        level: Severity — "error", "warning", or "info".
        code: Machine-readable code (e.g. "DSC_MISSING_TE").
        message: Human-readable description.
    """

    level: str
    code: str
    message: str


@dataclass
class PreflightReport:
    """Aggregated validation report.

    Attributes
    ----------
        messages: All validation messages collected.
    """

    messages: list[PreflightMessage] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Return True if any error-level messages exist."""
        return any(m.level == "error" for m in self.messages)

    def log_all(self) -> None:
        """Log every message at its appropriate level."""
        level_map = {
            "error": logging.ERROR,
            "warning": logging.WARNING,
            "info": logging.INFO,
        }
        for msg in self.messages:
            logger.log(
                level_map.get(msg.level, logging.INFO),
                "[%s] %s",
                msg.code,
                msg.message,
            )

    def raise_if_errors(self) -> None:
        """Raise PreflightError listing all errors if any exist."""
        errors = [m for m in self.messages if m.level == "error"]
        if not errors:
            return
        lines = [f"Pre-flight validation failed with {len(errors)} error(s):"]
        for e in errors:
            lines.append(f"  [{e.code}] {e.message}")
        raise PreflightError("\n".join(lines))

    def warning_strings(self, prefix: str = "") -> list[str]:
        """Return formatted warning strings, optionally filtered by code prefix.

        Parameters
        ----------
        prefix : str, optional
            If non-empty, only return warnings whose code starts
            with this string (e.g. ``"DSC_"``).

        Returns
        -------
        list of str
            List of ``"[CODE] message"`` strings for warning-level messages.
        """
        return [
            f"[{m.code}] {m.message}"
            for m in self.messages
            if m.level == "warning" and m.code.startswith(prefix)
        ]


def validate_sidecar_exists(
    nifti_path: Path,
    modality: str,
) -> list[PreflightMessage]:
    """Check that a JSON sidecar exists for the given NIfTI file.

    Parameters
    ----------
    nifti_path : Path
        Path to the NIfTI file.
    modality : str
        Modality label for messages (e.g. "DSC", "DCE").

    Returns
    -------
    list of PreflightMessage
        List of validation messages.
    """
    msgs: list[PreflightMessage] = []
    name = nifti_path.name
    sidecar = nifti_path.with_name(
        name.replace(".nii.gz", ".json").replace(".nii", ".json")
    )
    if not sidecar.exists():
        severity = "error" if modality.upper() in ("DSC", "DCE") else "warning"
        msgs.append(PreflightMessage(
            level=severity,
            code=f"{modality.upper()}_NO_SIDECAR",
            message=(
                f"No JSON sidecar found for {name}. "
                f"All metadata will be missing — perfusion maps will use "
                f"fallback defaults and may be silently wrong."
            ),
        ))
    return msgs


def validate_dsc_metadata(
    sidecar: dict[str, Any],
    dsc_config: Optional[DSCConfig] = None,
) -> list[PreflightMessage]:
    """Validate DSC acquisition metadata.

    Parameters
    ----------
    sidecar : dict
        Dict returned by ``_read_bids_sidecar()``.
    dsc_config : DSCConfig, optional
        DSC configuration for preload and dose checks.

    Returns
    -------
    list of PreflightMessage
        List of validation messages.
    """
    msgs: list[PreflightMessage] = []

    # TE
    te = sidecar.get("te")
    echo_time_array = sidecar.get("echo_time_array") or isinstance(te, (list, tuple))
    if echo_time_array:
        pass
    elif sidecar.get("echo_time_invalid") or (
        te is not None and (
            isinstance(te, bool) or not isinstance(te, (int, float))
            or not math.isfinite(te)
        )
    ):
        msgs.append(PreflightMessage(
            level="error",
            code="DSC_INVALID_TE",
            message="EchoTime must be a finite scalar number in seconds.",
        ))
    elif te is None:
        msgs.append(PreflightMessage(
            level="error",
            code="DSC_MISSING_TE",
            message=(
                "EchoTime not found in DSC BIDS sidecar. "
                "TE is the divisor in the deltaR2* equation — "
                "no physically meaningful default exists."
            ),
        ))
    elif te <= 0:
        msgs.append(PreflightMessage(
            level="error",
            code="DSC_INVALID_TE",
            message=f"EchoTime={te:.6f}s is non-positive (physically impossible).",
        ))
    elif te < 0.005 or te > 0.100:
        msgs.append(PreflightMessage(
            level="error",
            code="DSC_TE_RANGE",
            message=(
                f"EchoTime={te:.6f}s is outside the expected range "
                f"[0.005, 0.100]s for DSC. BIDS requires EchoTime in seconds "
                f"— check for ms-vs-s confusion."
            ),
        ))

    # Volume TR
    volume_tr = sidecar.get("volume_tr")
    if volume_tr is None:
        msgs.append(PreflightMessage(
            level="error",
            code="DSC_MISSING_TR",
            message=(
                "RepetitionTime not found in DSC BIDS sidecar. "
                "TR is required for deconvolution timestep and CBV integration."
            ),
        ))
    elif volume_tr <= 0:
        msgs.append(PreflightMessage(
            level="error",
            code="DSC_INVALID_TR",
            message=(
                f"RepetitionTime={volume_tr:.6f}s is non-positive "
                "(physically impossible)."
            ),
        ))
    elif volume_tr > 3.0:
        msgs.append(PreflightMessage(
            level="warning",
            code="DSC_TR_COARSE",
            message=(
                f"RepetitionTime={volume_tr:.1f}s is unusually long for DSC. "
                f"Typical DSC TR is 1-2s for adequate first-pass bolus sampling."
            ),
        ))

    # Field strength
    field_strength = sidecar.get("field_strength")
    if field_strength is None:
        msgs.append(PreflightMessage(
            level="error",
            code="DSC_MISSING_FIELD_STRENGTH",
            message=(
                "MagneticFieldStrength not found in DSC sidecar. "
                "Required for field-strength-dependent constants. "
                "Add MagneticFieldStrength to your BIDS sidecar."
            ),
        ))

    # Multi-echo detection from filename or acquisition metadata.
    if sidecar.get("echo_entity") or echo_time_array:
        msgs.append(PreflightMessage(
            level="error",
            code="DSC_MULTI_ECHO_UNSUPPORTED",
            message=(
                "Multi-echo DSC detected (echo entity or EchoTime array). "
                "PMRIPrep supports single-echo GRE-EPI only."
            ),
        ))

    # Sequence type validation
    scanning_seq = sidecar.get("scanning_sequence")
    if scanning_seq is not None:
        has_se = "SE" in scanning_seq
        has_gr = "GR" in scanning_seq
        if has_se and not has_gr:
            msgs.append(PreflightMessage(
                level="error",
                code="DSC_SPIN_ECHO_UNSUPPORTED",
                message=(
                    f"Spin-echo DSC detected "
                    f"(ScanningSequence={'/'.join(sorted(scanning_seq))}). "
                    f"PMRIPrep assumes T2*-weighted (gradient-echo) "
                    f"signal conversion."
                ),
            ))
        elif has_se and has_gr:
            msgs.append(PreflightMessage(
                level="error",
                code="DSC_HYBRID_SEQUENCE",
                message=(
                    f"Combined gradient- and spin-echo DSC detected "
                    f"(ScanningSequence={'/'.join(sorted(scanning_seq))}). "
                    f"PMRIPrep supports single-echo gradient-echo DSC only."
                ),
            ))
    else:
        msgs.append(PreflightMessage(
            level="info",
            code="DSC_SEQUENCE_TYPE_UNKNOWN",
            message=(
                "ScanningSequence not found in sidecar. "
                "PMRIPrep assumes single-echo GRE-EPI."
            ),
        ))

    # --- Agent-specific checks ---
    agent_raw = sidecar.get("contrast_agent")
    agent_name: Optional[str] = None
    if agent_raw:
        agent_name = normalize_agent_name(agent_raw)

    if dsc_config is not None:
        preload = dsc_config.preload
        # Real-world preload doses are quantized (0.0, 0.25, 0.5, 1.0);
        # 0.01 threshold separates explicit no-preload from any real dose.
        no_confirmed_preload = (
            isinstance(preload, str) or float(preload) < 0.01
        )

        # DSC check 1: high-r1 agent without preload
        if agent_name and field_strength is not None:
            r1 = lookup_r1_safe(agent_name, field_strength)
            if r1 is not None and r1 >= HIGH_R1_THRESHOLD and no_confirmed_preload:
                if isinstance(preload, str):
                    detail = (
                        "Preload status is unknown. If no "
                        "preload is present, T1 shine-through will be "
                        "disproportionately severe."
                    )
                    lead = "without confirmed preload"
                    cta = " Use --dsc-preload to confirm preload status."
                else:
                    detail = (
                        "No-preload configuration with a high-r1 agent "
                        "produces severe T1 shine-through."
                    )
                    lead = "with no preload (user-specified)"
                    cta = ""
                if dsc_config.leakage_method == "bsw":
                    correction_detail = (
                        "Strong T1 shine-through can reduce the non-enhancing "
                        "reference support available to BSW leakage correction; "
                        "correction will be skipped if that reference cannot be "
                        "estimated reliably."
                    )
                elif dsc_config.leakage_method == "none":
                    correction_detail = (
                        "Leakage correction is disabled by configuration, so "
                        "this contamination is not corrected and may bias DSC "
                        "perfusion estimates."
                    )
                else:
                    raise ValueError(
                        "Unsupported DSC leakage_method="
                        f"{dsc_config.leakage_method!r}"
                    )
                msgs.append(PreflightMessage(
                    level="warning",
                    code="DSC_HIGH_R1_NO_PRELOAD",
                    message=(
                        f"High-r1 agent ({agent_raw}, r1={r1:.1f} s-1/mM) "
                        f"{lead}. {detail} "
                        f"{correction_detail}{cta}"
                    ),
                ))

        # DSC check 2: high FA without preload
        fa = sidecar.get("flip_angle")
        if fa is not None and fa >= 60 and no_confirmed_preload:
            if isinstance(preload, str):
                detail = (
                    "If no preload is present, this FA maximizes "
                    "T1 contamination."
                )
                fa_lead = "without confirmed preload"
                fa_cta = " Use --dsc-preload to confirm."
            else:
                detail = (
                    "No-preload DSC at this FA maximizes T1 contamination."
                )
                fa_lead = "with no preload (user-specified)"
                fa_cta = ""
            msgs.append(PreflightMessage(
                level="warning",
                code="DSC_HIGH_FA_NO_PRELOAD",
                message=(
                    f"DSC FlipAngle={fa:.0f} deg {fa_lead}. "
                    f"DSC consensus recommends 30 deg FA for no-preload "
                    f"protocols; 60 deg is for preloaded acquisitions. "
                    f"{detail}{fa_cta}"
                ),
            ))

        # DSC check 3: low effective Gd dose
        if agent_name:
            gd_mult = lookup_gd_multiplicity(agent_name)
            eff_gd_dose = (
                dsc_config.agent_standard_dose * dsc_config.dose * gd_mult
            )
            if eff_gd_dose < LOW_GD_DOSE_THRESHOLD:
                msgs.append(PreflightMessage(
                    level="warning",
                    code="DSC_LOW_EFFECTIVE_DOSE",
                    message=(
                        f"Effective DSC Gd dose is {eff_gd_dose:.3f} mmol-Gd/kg "
                        f"(< {LOW_GD_DOSE_THRESHOLD} threshold). At this dose, "
                        f"peak deltaR2* may be insufficient for robust "
                        f"deconvolution; CBF/MTT maps will have reduced "
                        f"precision. Consider higher dose or acknowledge "
                        f"reduced quality."
                    ),
                ))

    return msgs


def validate_dce_metadata(
    sidecar: dict[str, Any],
    dce_config: DCEConfig,
) -> list[PreflightMessage]:
    """Validate DCE acquisition metadata.

    Parameters
    ----------
    sidecar : dict
        Dict returned by ``_read_bids_sidecar()``.
    dce_config : DCEConfig
        DCE configuration for r1 override checks.

    Returns
    -------
    list of PreflightMessage
        List of validation messages.
    """
    msgs: list[PreflightMessage] = []

    # FlipAngle
    fa = sidecar.get("flip_angle")
    if fa is None:
        msgs.append(PreflightMessage(
            level="error",
            code="DCE_MISSING_FLIP_ANGLE",
            message=(
                "FlipAngle not found in DCE BIDS sidecar. "
                "Required for the Ernst/SPGR signal equation in "
                "signal-to-concentration conversion."
            ),
        ))
    elif fa <= 0 or fa > 90:
        msgs.append(PreflightMessage(
            level="error",
            code="DCE_INVALID_FLIP_ANGLE",
            message=(
                f"FlipAngle={fa:.1f} degrees is outside (0, 90]. "
                f"Check for radians-vs-degrees confusion."
            ),
        ))

    # TR checks
    volume_tr = sidecar.get("volume_tr")
    pulse_tr = sidecar.get("pulse_tr")

    if volume_tr is None and pulse_tr is None:
        msgs.append(PreflightMessage(
            level="error",
            code="DCE_MISSING_TR",
            message=(
                "Neither RepetitionTime nor RepetitionTimeExcitation found "
                "in DCE BIDS sidecar. At least one TR is required."
            ),
        ))
    else:
        if (volume_tr is not None and pulse_tr is not None
                and volume_tr < pulse_tr):
            msgs.append(PreflightMessage(
                level="error",
                code="DCE_TR_SWAPPED",
                message=(
                    f"RepetitionTime ({volume_tr:.4f}s) < "
                    f"RepetitionTimeExcitation ({pulse_tr:.4f}s). "
                    f"These fields appear to be swapped."
                ),
            ))

        # 3D DCE with only pulse TR — no volume TR for time vector
        if volume_tr is None and pulse_tr is not None and pulse_tr < 0.1:
            msgs.append(PreflightMessage(
                level="error",
                code="DCE_NO_VOLUME_TR",
                message=(
                    f"No volume TR found; pulse TR ({pulse_tr * 1000:.1f}ms) "
                    f"would produce a nonsensical time vector for PK fitting. "
                    f"Add RepetitionTime to the BIDS sidecar or provide a "
                    f"time_points_file."
                ),
            ))

        # Only volume TR, no pulse TR, and volume TR is too long for signal eq
        if (pulse_tr is None and volume_tr is not None
                and volume_tr > 1.0):
            # Check if pulse_tr would have been inferred from volume_tr
            # (in _read_bids_sidecar, pulse_tr falls back to volume_tr)
            # So this case only fires when volume_tr > 1.0 and there's
            # explicitly no RepetitionTimeExcitation
            msgs.append(PreflightMessage(
                level="error",
                code="DCE_NO_PULSE_TR",
                message=(
                    f"No pulse TR (RepetitionTimeExcitation) found; "
                    f"volume TR ({volume_tr:.3f}s) cannot be used in the "
                    f"SPGR signal equation. E1 = exp(-TR/T1) ≈ 0 at this "
                    f"TR, making concentration conversion undefined. "
                    f"Add RepetitionTimeExcitation to BIDS sidecar."
                ),
            ))

    # Field strength
    field_strength = sidecar.get("field_strength")
    if field_strength is None:
        msgs.append(PreflightMessage(
            level="error",
            code="DCE_MISSING_FIELD_STRENGTH",
            message=(
                "MagneticFieldStrength not found in DCE sidecar. "
                "Required for field-strength-dependent constants. "
                "Add MagneticFieldStrength to your BIDS sidecar."
            ),
        ))

    # Sequence type validation
    scanning_seq = sidecar.get("scanning_sequence")
    sequence_var = sidecar.get("sequence_variant")
    te = sidecar.get("te")
    if sidecar.get("echo_time_array"):
        msgs.append(PreflightMessage(
            level="error", code="DCE_MULTI_ECHO_UNSUPPORTED",
            message="Multi-echo DCE is unsupported; EchoTime must be a finite scalar.",
        ))
    elif sidecar.get("echo_time_invalid"):
        msgs.append(PreflightMessage(
            level="error", code="DCE_INVALID_TE",
            message="EchoTime must be a finite scalar number in seconds.",
        ))

    if scanning_seq is not None:
        has_se = "SE" in scanning_seq
        has_gr = "GR" in scanning_seq
        has_ir = "IR" in scanning_seq
        has_ep = "EP" in scanning_seq

        if has_se and not has_gr:
            msgs.append(PreflightMessage(
                level="error",
                code="DCE_SPIN_ECHO_UNSUPPORTED",
                message=(
                    f"Spin-echo DCE detected "
                    f"(ScanningSequence={'/'.join(sorted(scanning_seq))}). "
                    f"PMRIPrep uses the SPGR/Ernst signal equation, which "
                    f"requires spoiled gradient-echo acquisition."
                ),
            ))
        elif has_se and has_gr:
            msgs.append(PreflightMessage(
                level="error",
                code="DCE_HYBRID_SEQUENCE",
                message=(
                    f"Combined gradient- and spin-echo DCE detected "
                    f"(ScanningSequence={'/'.join(sorted(scanning_seq))}). "
                    f"PMRIPrep requires spoiled gradient-echo DCE for "
                    f"the SPGR signal equation."
                ),
            ))

        if has_ir:
            msgs.append(PreflightMessage(
                level="error",
                code="DCE_IR_UNSUPPORTED",
                message=(
                    f"Inversion-recovery prepared DCE detected "
                    f"(ScanningSequence={'/'.join(sorted(scanning_seq))}). "
                    f"The SPGR signal equation does not model inversion "
                    f"preparation."
                ),
            ))

        if sequence_var is not None and "SS" in sequence_var:
            is_bssfp = (
                te is not None
                and pulse_tr is not None
                and pulse_tr > 0
                and te > 0.4 * pulse_tr
            )
            if is_bssfp:
                msgs.append(PreflightMessage(
                    level="error",
                    code="DCE_BSSFP_UNSUPPORTED",
                    message=(
                        f"Balanced steady-state free precession (bSSFP) DCE "
                        f"detected (SequenceVariant="
                        f"{'/'.join(sorted(sequence_var))}, "
                        f"TE/TR={te:.4f}/{pulse_tr:.4f}s). The bSSFP signal "
                        f"equation differs fundamentally from SPGR."
                    ),
                ))
            elif "SP" not in sequence_var:
                msgs.append(PreflightMessage(
                    level="warning",
                    code="DCE_UNSPOILED_GR",
                    message=(
                        f"Non-spoiled gradient echo (SSFP-FID) detected "
                        f"(SequenceVariant="
                        f"{'/'.join(sorted(sequence_var))}). "
                        f"The SPGR signal model assumes complete spoiling. "
                        f"Results may contain systematic bias."
                    ),
                ))

        if has_ep:
            msgs.append(PreflightMessage(
                level="info",
                code="DCE_EPI_READOUT",
                message=(
                    "EPI-based DCE acquisition detected. The EPI readout "
                    "introduces T2* sensitivity that may confound T1-based "
                    "concentration estimates at high concentrations."
                ),
            ))
    else:
        msgs.append(PreflightMessage(
            level="info",
            code="DCE_SEQUENCE_TYPE_UNKNOWN",
            message=(
                "ScanningSequence not found in sidecar. "
                "PMRIPrep assumes spoiled gradient-echo (SPGR) acquisition."
            ),
        ))

    # r1 relaxivity warnings (skip if user supplied override)
    contrast_agent = sidecar.get("contrast_agent")
    if dce_config.r1_relaxivity is None:
        if contrast_agent is None and field_strength is not None:
            msgs.append(PreflightMessage(
                level="warning",
                code="DCE_UNKNOWN_AGENT",
                message=(
                    "Contrast agent not identified in DCE sidecar. "
                    "r1 varies up to 3x across agents. Using generic "
                    "Gd-DTPA r1. Supply --dce-r1 for known agents."
                ),
            ))
    else:
        # User supplied r1
        r1_val = dce_config.r1_relaxivity
        msgs.append(PreflightMessage(
            level="info",
            code="DCE_R1_OVERRIDE",
            message=(
                f"Using user-supplied r1={r1_val:.1f} s^-1/mM "
                f"(overrides sidecar lookup)."
            ),
        ))
        if r1_val < 1.0 or r1_val > 15.0:
            msgs.append(PreflightMessage(
                level="warning",
                code="DCE_R1_OUT_OF_RANGE",
                message=(
                    f"User-supplied r1={r1_val:.1f} s^-1/mM is outside "
                    f"[1.0, 15.0] — likely a typo. Typical Gd-based "
                    f"agents have r1 in 3-7 s^-1/mM at clinical fields."
                ),
            ))

    # --- Agent-specific checks ---
    agent_name: Optional[str] = None
    if contrast_agent:
        agent_name = normalize_agent_name(contrast_agent)

    # DCE agent checks (require agent + field strength for r1 lookup)
    r1: Optional[float] = None
    if agent_name and field_strength is not None:
        r1 = lookup_r1_safe(agent_name, field_strength)

    # DCE check 1: high-r1 agent with measured AIF
    if (r1 is not None
            and r1 >= HIGH_R1_THRESHOLD
            and dce_config.aif_method != "population"
            and dce_config.aif_file is None):
        msgs.append(PreflightMessage(
            level="warning",
            code="DCE_HIGH_R1_AIF_SATURATION",
            message=(
                f"High-r1 agent ({contrast_agent}, r1={r1:.1f} s-1/mM) "
                f"with measured AIF. Arterial peak signal approaches "
                f"the Ernst ceiling; noise amplification and E1 clipping "
                f"systematically underestimate peak AIF concentration. "
                f"Consider --dce-aif-method population or acknowledge "
                f"degraded AIF accuracy."
            ),
        ))

    # DCE check 2: low effective dose × r1
    if agent_name:
        if r1 is None and dce_config.r1_relaxivity is not None:
            r1 = dce_config.r1_relaxivity
        if r1 is not None:
            agent_std_dose = lookup_standard_dose(agent_name)
            eff_dose_r1 = agent_std_dose * dce_config.dose * r1
            if eff_dose_r1 < LOW_DCE_DOSE_R1_THRESHOLD:
                msgs.append(PreflightMessage(
                    level="warning",
                    code="DCE_LOW_EFFECTIVE_DOSE",
                    message=(
                        f"Effective DCE dose x r1 = {eff_dose_r1:.3f} "
                        f"kg-1 s-1 (< {LOW_DCE_DOSE_R1_THRESHOLD} "
                        f"threshold). Tissue enhancement may be too low "
                        f"for reliable Ktrans estimation."
                    ),
                ))

    return msgs


def validate_vfa_metadata(
    vfa_flip_angles: list[Optional[float]],
    vfa_trs: list[Optional[float]],
) -> list[PreflightMessage]:
    """Validate VFA T1 mapping metadata.

    Only called when VFA data IS present (VFA is optional).

    Parameters
    ----------
    vfa_flip_angles : list of float or None
        Position-preserving flip angles extracted from VFA sidecars.
    vfa_trs : list of float
        TR values extracted from VFA sidecars.

    Returns
    -------
    list of PreflightMessage
        List of validation messages.
    """
    msgs: list[PreflightMessage] = []

    missing_angles = sum(angle is None for angle in vfa_flip_angles)
    if missing_angles:
        msgs.append(PreflightMessage(
            level="error",
            code="VFA_MISSING_FLIP_ANGLE",
            message=(
                f"FlipAngle is missing from {missing_angles} VFA sidecar(s). "
                "Every VFA image requires a nominal flip angle."
            ),
        ))

    valid_angles = [float(angle) for angle in vfa_flip_angles if angle is not None]
    invalid_angles = [
        angle
        for angle in valid_angles
        if not math.isfinite(angle) or angle <= 0.0 or angle > 90.0
    ]
    if invalid_angles:
        msgs.append(PreflightMessage(
            level="error",
            code="VFA_INVALID_FLIP_ANGLE",
            message=(
                "VFA FlipAngle values must be finite and in (0, 90] degrees; "
                f"found {invalid_angles}."
            ),
        ))

    distinct_angles = set(valid_angles)
    if len(distinct_angles) < 2:
        msgs.append(PreflightMessage(
            level="error",
            code="VFA_FEW_ANGLES",
            message=(
                f"Only {len(distinct_angles)} distinct VFA flip angle(s) "
                f"found ({sorted(distinct_angles)}). VFA T1 mapping requires "
                f"at least 2 distinct angles."
            ),
        ))

    if any(tr is None for tr in vfa_trs):
        msgs.append(PreflightMessage(
            level="warning",
            code="VFA_MISSING_TR",
            message=(
                "RepetitionTime missing from one or more VFA sidecars. "
                "T1 mapping requires TR for the linearized fit."
            ),
        ))

    # Check flip angle conditioning (ratio of max to min)
    valid_angles = [angle for angle in valid_angles if angle > 0]
    if len(valid_angles) >= 2:
        ratio = max(valid_angles) / min(valid_angles)
        if ratio < 1.5:
            msgs.append(PreflightMessage(
                level="warning",
                code="VFA_POOR_CONDITIONING",
                message=(
                    f"VFA flip angle ratio ({ratio:.1f}) is below 1.5. "
                    f"Angles {sorted(valid_angles)} may be too similar for "
                    f"robust T1 estimation."
                ),
            ))

    return msgs


def validate_afi_metadata(
    tr1: Optional[float],
    tr2: Optional[float],
) -> list[PreflightMessage]:
    """Validate AFI B1+ mapping metadata.

    Only called when AFI data IS present (AFI is optional).

    Parameters
    ----------
    tr1 : float or None
        First AFI TR in ms, or None.
    tr2 : float or None
        Second AFI TR in ms, or None.

    Returns
    -------
    list of PreflightMessage
        List of validation messages.
    """
    msgs: list[PreflightMessage] = []

    if tr1 is None or tr2 is None:
        msgs.append(PreflightMessage(
            level="warning",
            code="AFI_MISSING_TR",
            message=(
                "AFI files present but TR1 or TR2 is missing. "
                "B1+ correction will fail without both TR values."
            ),
        ))
    elif tr1 >= tr2:
        msgs.append(PreflightMessage(
            level="warning",
            code="AFI_TR_ORDER",
            message=(
                f"AFI TR1 ({tr1:.1f}ms) >= TR2 ({tr2:.1f}ms). "
                f"AFI requires TR2 > TR1 for the dual-TR B1+ calculation."
            ),
        ))

    return msgs


def validate_dsc_temporal_sufficiency(
    n_volumes: int,
    tr: Optional[float],
) -> list[PreflightMessage]:
    """Validate DSC temporal sampling sufficiency.

    Parameters
    ----------
    n_volumes : int
        Number of temporal volumes (NIfTI shape[3]).
    tr : float or None
        Repetition time in seconds, or None if unavailable.

    Returns
    -------
    list of PreflightMessage
    """
    msgs: list[PreflightMessage] = []

    if n_volumes < DSC_MIN_VOLUMES:
        msgs.append(PreflightMessage(
            level="error",
            code="DSC_TOO_FEW_VOLUMES",
            message=(
                f"DSC acquisition has {n_volumes} volumes "
                f"(minimum: {DSC_MIN_VOLUMES}). Insufficient timepoints "
                f"for baseline estimation, bolus tracking, and "
                f"deconvolution."
            ),
        ))
    elif n_volumes < DSC_MARGINAL_VOLUMES:
        msgs.append(PreflightMessage(
            level="warning",
            code="DSC_MARGINAL_VOLUMES",
            message=(
                f"DSC acquisition has {n_volumes} volumes "
                f"({DSC_MIN_VOLUMES}-{DSC_MARGINAL_VOLUMES} is marginal). "
                f"Leakage correction and AIF quality scoring have "
                f"reduced reliability."
            ),
        ))

    if tr is not None and tr > 0:
        duration = n_volumes * tr
        if duration < DSC_MIN_DURATION_S:
            msgs.append(PreflightMessage(
                level="error",
                code="DSC_SHORT_DURATION",
                message=(
                    f"DSC acquisition duration is {duration:.1f}s "
                    f"({n_volumes} volumes x TR={tr:.3f}s), below the "
                    f"{DSC_MIN_DURATION_S:.0f}s minimum required for "
                    f"first-pass bolus capture and recirculation."
                ),
            ))
        elif duration < DSC_MARGINAL_DURATION_S:
            msgs.append(PreflightMessage(
                level="warning",
                code="DSC_MARGINAL_DURATION",
                message=(
                    f"DSC acquisition duration is {duration:.1f}s "
                    f"({n_volumes} volumes x TR={tr:.3f}s). "
                    f"Post-bolus sampling may be insufficient for "
                    f"leakage correction."
                ),
            ))

    return msgs


def validate_dce_temporal_sufficiency(
    n_volumes: int,
    volume_tr: Optional[float],
) -> list[PreflightMessage]:
    """Validate DCE temporal sampling sufficiency.

    Parameters
    ----------
    n_volumes : int
        Number of temporal volumes (NIfTI shape[3]).
    volume_tr : float or None
        Volume TR in seconds (temporal resolution), or None if unavailable.

    Returns
    -------
    list of PreflightMessage
    """
    msgs: list[PreflightMessage] = []

    if n_volumes < DCE_MIN_VOLUMES:
        msgs.append(PreflightMessage(
            level="error",
            code="DCE_TOO_FEW_VOLUMES",
            message=(
                f"DCE acquisition has {n_volumes} volumes "
                f"(minimum: {DCE_MIN_VOLUMES}). Insufficient timepoints "
                f"for pharmacokinetic model fitting."
            ),
        ))
    elif n_volumes < DCE_MARGINAL_VOLUMES:
        msgs.append(PreflightMessage(
            level="warning",
            code="DCE_MARGINAL_VOLUMES",
            message=(
                f"DCE acquisition has {n_volumes} volumes "
                f"({DCE_MIN_VOLUMES}-{DCE_MARGINAL_VOLUMES} is marginal). "
                f"Only Patlak model is recommended at this sampling."
            ),
        ))

    if volume_tr is not None and volume_tr > 0:
        duration = n_volumes * volume_tr
        if duration < DCE_MIN_DURATION_S:
            msgs.append(PreflightMessage(
                level="error",
                code="DCE_SHORT_DURATION",
                message=(
                    f"DCE acquisition duration is {duration:.1f}s "
                    f"({n_volumes} volumes x TR={volume_tr:.3f}s), below "
                    f"the {DCE_MIN_DURATION_S:.0f}s minimum required for "
                    f"contrast wash-in and washout characterization."
                ),
            ))
        elif duration < DCE_MARGINAL_DURATION_S:
            msgs.append(PreflightMessage(
                level="warning",
                code="DCE_MARGINAL_DURATION",
                message=(
                    f"DCE acquisition duration is {duration:.1f}s "
                    f"({n_volumes} volumes x TR={volume_tr:.3f}s). "
                    f"Washout phase sampling may be insufficient for "
                    f"Extended Tofts or 2CXM fitting."
                ),
            ))

    return msgs


def run_preflight(
    session_data: Any,
    decision: Any,
    dsc_sidecar: Optional[dict[str, Any]],
    dce_sidecar: Optional[dict[str, Any]],
    dce_config: Optional[DCEConfig],
    vfa_flip_angles: Optional[list[Optional[float]]] = None,
    vfa_trs: Optional[list[Optional[float]]] = None,
    dsc_config: Optional[DSCConfig] = None,
) -> PreflightReport:
    """Run all pre-flight validations based on pipeline decisions.

    Parameters
    ----------
    session_data : Any
        SessionData from discovery.
    decision : Any
        PipelineDecision from decide_pipelines().
    dsc_sidecar : dict or None
        DSC sidecar dict, or None if DSC not running.
    dce_sidecar : dict or None
        DCE sidecar dict, or None if DCE not running.
    dce_config : DCEConfig or None
        DCE configuration, or None if DCE not running.
    vfa_flip_angles : list of float, optional
        VFA flip angles, if VFA data present.
    vfa_trs : list of float or None, optional
        VFA TR values, if VFA data present.
    dsc_config : DSCConfig, optional
        DSC configuration, or None if DSC not running.

    Returns
    -------
    PreflightReport
        PreflightReport with all collected messages.
    """
    report = PreflightReport()

    # DSC candidate validation precedes metadata checks and respects skip flags.
    candidates = getattr(session_data.modalities, "dsc_candidates", [])
    if decision.run_dsc and len(candidates) > 1:
        multi_echo = any(re.search(r"_echo-\d+", path.name) for path in candidates)
        code = "DSC_MULTI_ECHO_UNSUPPORTED" if multi_echo else "DSC_AMBIGUOUS_INPUT"
        context = f"sub-{session_data.subject}"
        if session_data.session:
            context += f"/ses-{session_data.session}"
        names = "\n".join(f"  {path}" for path in candidates)
        detail = "Multi-echo DSC is unsupported. " if multi_echo else ""
        report.messages.append(PreflightMessage(
            level="error", code=code,
            message=(
                f"{detail}Multiple DSC candidates for {context}:\n{names}\n"
                "Provide one single-echo GRE DSC acquisition per subject/session."
            ),
        ))

    # DSC validation
    if decision.run_dsc and session_data.modalities.dsc is not None:
        report.messages.extend(
            validate_sidecar_exists(
                Path(session_data.modalities.dsc), "DSC",
            )
        )
        if dsc_sidecar is not None:
            report.messages.extend(
                validate_dsc_metadata(dsc_sidecar, dsc_config=dsc_config)
            )
        if session_data.modalities.dsc_n_volumes is not None:
            report.messages.extend(
                validate_dsc_temporal_sufficiency(
                    session_data.modalities.dsc_n_volumes,
                    tr=dsc_sidecar.get("volume_tr") if dsc_sidecar else None,
                )
            )

    # DCE validation
    if decision.run_dce and session_data.modalities.dce is not None:
        report.messages.extend(
            validate_sidecar_exists(
                Path(session_data.modalities.dce), "DCE",
            )
        )
        if dce_sidecar is not None and dce_config is not None:
            report.messages.extend(
                validate_dce_metadata(dce_sidecar, dce_config)
            )
        if session_data.modalities.dce_n_volumes is not None:
            report.messages.extend(
                validate_dce_temporal_sufficiency(
                    session_data.modalities.dce_n_volumes,
                    volume_tr=(
                        dce_sidecar.get("volume_tr")
                        if dce_sidecar else None
                    ),
                )
            )

        # VFA validation (only if VFA data present)
        vfa_t1_active = dce_config is None or dce_config.uses_vfa_t1
        if (session_data.modalities.vfa
                and vfa_t1_active
                and vfa_flip_angles is not None
                and vfa_trs is not None):
            report.messages.extend(
                validate_vfa_metadata(vfa_flip_angles, vfa_trs)
            )

        # AFI validation (only if AFI data present and method is AFI, not precomputed)
        if (session_data.modalities.afi
                and session_data.modalities.b1_method_detected != "precomputed"):
            report.messages.extend(
                validate_afi_metadata(
                    session_data.modalities.afi_tr1_ms,
                    session_data.modalities.afi_tr2_ms,
                )
            )

        if dce_config is not None and dce_config.b1_method == "precomputed":
            if not session_data.modalities.b1_precomputed_file:
                report.messages.append(PreflightMessage(
                    level="error",
                    code="DCE_B1_PRECOMPUTED_MISSING",
                    message=(
                        "b1_method='precomputed' was requested, but no supported "
                        "TB1map/TB1TFL input was discovered for this session. "
                        "Provide a precomputed B1+ map, choose b1_method='auto', "
                        "or explicitly disable B1+ correction with 'none'."
                    ),
                ))
        if dce_config is not None and dce_config.b1_method == "afi":
            if not (
                session_data.modalities.afi
                and session_data.modalities.afi_mode == "4d"
                and session_data.modalities.afi_4d_file
            ):
                report.messages.append(PreflightMessage(
                    level="error",
                    code="DCE_B1_AFI_MISSING",
                    message=(
                        "b1_method='afi' was requested, but no supported 4D "
                        "TB1AFI input was discovered for this session. Provide "
                        "a 4D AFI acquisition, choose b1_method='auto', or "
                        "explicitly disable B1+ correction with 'none'."
                    ),
                ))

    return report
