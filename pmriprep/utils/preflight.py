"""Pre-flight validation for PMRIPrep metadata completeness.

Validates BIDS sidecar metadata and acquisition parameters BEFORE
workflow initialization, catching common errors early with clear
diagnostic messages rather than cryptic failures deep in processing.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pmriprep.config.dce import DCEConfig

logger = logging.getLogger("pmriprep.preflight")


class PreflightError(RuntimeError):
    """Raised when pre-flight validation finds fatal errors."""


@dataclass
class PreflightMessage:
    """A single validation message.

    Attributes:
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

    Attributes:
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


def validate_sidecar_exists(
    nifti_path: Path,
    modality: str,
) -> list[PreflightMessage]:
    """Check that a JSON sidecar exists for the given NIfTI file.

    Args:
        nifti_path: Path to the NIfTI file.
        modality: Modality label for messages (e.g. "DSC", "DCE").

    Returns:
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


def validate_dsc_metadata(sidecar: dict[str, Any]) -> list[PreflightMessage]:
    """Validate DSC acquisition metadata.

    Args:
        sidecar: Dict returned by ``_read_bids_sidecar()``.

    Returns:
        List of validation messages.
    """
    msgs: list[PreflightMessage] = []

    # TE
    te = sidecar.get("te")
    if te is None:
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

    # TR
    tr = sidecar.get("tr")
    if tr is None:
        msgs.append(PreflightMessage(
            level="error",
            code="DSC_MISSING_TR",
            message=(
                "RepetitionTime not found in DSC BIDS sidecar. "
                "TR is required for deconvolution timestep and CBV integration."
            ),
        ))
    elif tr <= 0:
        msgs.append(PreflightMessage(
            level="error",
            code="DSC_INVALID_TR",
            message=f"RepetitionTime={tr:.6f}s is non-positive (physically impossible).",
        ))
    elif tr > 3.0:
        msgs.append(PreflightMessage(
            level="warning",
            code="DSC_TR_COARSE",
            message=(
                f"RepetitionTime={tr:.1f}s is unusually long for DSC. "
                f"Typical DSC TR is 1-2s for adequate first-pass bolus sampling."
            ),
        ))

    # Field strength
    if sidecar.get("field_strength") is None:
        msgs.append(PreflightMessage(
            level="warning",
            code="DSC_MISSING_FIELD_STRENGTH",
            message=(
                "MagneticFieldStrength not found in DSC sidecar. "
                "Defaulting to 3T for TE2-dependent constants."
            ),
        ))

    return msgs


def validate_dce_metadata(
    sidecar: dict[str, Any],
    dce_config: DCEConfig,
) -> list[PreflightMessage]:
    """Validate DCE acquisition metadata.

    Args:
        sidecar: Dict returned by ``_read_bids_sidecar()``.
        dce_config: DCE configuration for r1 override checks.

    Returns:
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

    # r1 relaxivity warnings (skip if user supplied override)
    if dce_config.r1_relaxivity is None:
        field_strength = sidecar.get("field_strength")
        contrast_agent = sidecar.get("contrast_agent")

        if field_strength is None and contrast_agent is not None:
            msgs.append(PreflightMessage(
                level="warning",
                code="DCE_R1_NO_FIELD_STRENGTH",
                message=(
                    f"Field strength missing; cannot look up agent-specific "
                    f"r1 for {contrast_agent}. Falling back to Gd-DTPA "
                    f"r1=3.8 at 3T. Supply --dce-r1 or add "
                    f"MagneticFieldStrength to sidecar."
                ),
            ))
        elif field_strength is None:
            msgs.append(PreflightMessage(
                level="warning",
                code="DCE_MISSING_FIELD_STRENGTH",
                message=(
                    "MagneticFieldStrength not found in DCE sidecar. "
                    "Using 3T defaults for r1 relaxivity."
                ),
            ))

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

    return msgs


def validate_vfa_metadata(
    vfa_flip_angles: list[float],
    vfa_trs: list[Optional[float]],
) -> list[PreflightMessage]:
    """Validate VFA T1 mapping metadata.

    Only called when VFA data IS present (VFA is optional).

    Args:
        vfa_flip_angles: Flip angles extracted from VFA sidecars.
        vfa_trs: TR values extracted from VFA sidecars.

    Returns:
        List of validation messages.
    """
    msgs: list[PreflightMessage] = []

    distinct_angles = set(vfa_flip_angles)
    if len(distinct_angles) < 2:
        msgs.append(PreflightMessage(
            level="warning",
            code="VFA_FEW_ANGLES",
            message=(
                f"Only {len(distinct_angles)} distinct VFA flip angle(s) "
                f"found ({sorted(distinct_angles)}). T1 mapping requires "
                f"at least 2 distinct angles for a reliable fit."
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
    valid_angles = [a for a in vfa_flip_angles if a > 0]
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

    Args:
        tr1: First AFI TR in ms, or None.
        tr2: Second AFI TR in ms, or None.

    Returns:
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


def run_preflight(
    session_data: Any,
    decision: Any,
    dsc_sidecar: Optional[dict[str, Any]],
    dce_sidecar: Optional[dict[str, Any]],
    dce_config: Optional[DCEConfig],
    vfa_flip_angles: Optional[list[float]] = None,
    vfa_trs: Optional[list[Optional[float]]] = None,
) -> PreflightReport:
    """Run all pre-flight validations based on pipeline decisions.

    Args:
        session_data: SessionData from discovery.
        decision: PipelineDecision from decide_pipelines().
        dsc_sidecar: DSC sidecar dict, or None if DSC not running.
        dce_sidecar: DCE sidecar dict, or None if DCE not running.
        dce_config: DCE configuration, or None if DCE not running.
        vfa_flip_angles: VFA flip angles, if VFA data present.
        vfa_trs: VFA TR values, if VFA data present.

    Returns:
        PreflightReport with all collected messages.
    """
    report = PreflightReport()

    # DSC validation
    if decision.run_dsc and session_data.modalities.dsc is not None:
        report.messages.extend(
            validate_sidecar_exists(
                Path(session_data.modalities.dsc), "DSC",
            )
        )
        if dsc_sidecar is not None:
            report.messages.extend(validate_dsc_metadata(dsc_sidecar))

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

        # VFA validation (only if VFA data present)
        if (session_data.modalities.vfa
                and vfa_flip_angles is not None
                and vfa_trs is not None):
            report.messages.extend(
                validate_vfa_metadata(vfa_flip_angles, vfa_trs)
            )

        # AFI validation (only if AFI data present)
        if session_data.modalities.afi:
            report.messages.extend(
                validate_afi_metadata(
                    session_data.modalities.afi_tr1_ms,
                    session_data.modalities.afi_tr2_ms,
                )
            )

    return report
