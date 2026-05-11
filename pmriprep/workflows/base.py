"""Master workflow for PMRIPrep BIDS-App.

This module implements the master workflow that orchestrates all pipeline
components (anatomical, tumor segmentation, DSC, DCE) based on available data
and user-specified skip flags.
"""

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import TYPE_CHECKING, Any, Optional, Union

import nipype.interfaces.utility as niu  # type: ignore[import-untyped]
import nipype.pipeline.engine as pe  # type: ignore[import-untyped]

from pmriprep.config.anat import AnatConfig
from pmriprep.config.dce import DCEConfig
from pmriprep.config.dsc import DSCConfig
from pmriprep.config.tumor_seg import TumorSegConfig
from pmriprep.workflows.discovery import discover_session_data, SessionData

if TYPE_CHECKING:
    from bids.layout import BIDSLayout


@dataclass
class PipelineSkipFlags:
    """Flags to control which pipeline components to skip.

    These flags allow users to explicitly disable pipeline components
    even when the required data is available.

    Attributes:
        skip_anat: Skip anatomical preprocessing pipeline.
        skip_tumor_seg: Skip tumor segmentation pipeline.
        skip_dsc: Skip DSC perfusion pipeline.
        skip_dce: Skip DCE perfusion pipeline.
    """

    skip_anat: bool = False
    skip_tumor_seg: bool = False
    skip_dsc: bool = False
    skip_dce: bool = False


@dataclass
class PipelineDecision:
    """Decision about which pipelines to run.

    Attributes:
        run_anat: Whether to run anatomical preprocessing.
        run_tumor_seg: Whether to run tumor segmentation.
        run_dsc: Whether to run DSC perfusion processing.
        run_dce: Whether to run DCE perfusion processing.
        anat_reason: Reason for anat decision if not running.
        tumor_seg_reason: Reason for tumor_seg decision if not running.
        dsc_reason: Reason for dsc decision if not running.
        dce_reason: Reason for dce decision if not running.
    """

    run_anat: bool = True
    run_tumor_seg: bool = False
    run_dsc: bool = False
    run_dce: bool = False
    anat_reason: Optional[str] = None
    tumor_seg_reason: Optional[str] = None
    dsc_reason: Optional[str] = None
    dce_reason: Optional[str] = None


def decide_pipelines(
    session_data: SessionData,
    skip_flags: PipelineSkipFlags,
    tumor_seg_config: Optional[TumorSegConfig] = None,
) -> PipelineDecision:
    """Decide which pipelines to run based on data and skip flags.

    This function applies the following rules:
    1. anat always runs (unless explicitly skipped)
    2. tumor_seg runs if all 4 modalities are present (unless skipped)
       and the requested GPU device is available
    3. dsc runs if DSC data is present (unless skipped)
    4. dce runs if DCE data is present (unless skipped)

    Args:
        session_data: Available data from discovery.
        skip_flags: User-specified skip flags.
        tumor_seg_config: Tumor segmentation config for device check.

    Returns:
        PipelineDecision with boolean flags and reasons.
    """
    decision = PipelineDecision()

    # Anatomical pipeline
    if skip_flags.skip_anat:
        decision.run_anat = False
        decision.anat_reason = "Skipped by user (--skip-anat)"
    else:
        decision.run_anat = True

    # Tumor segmentation
    # Tumor segmentation requires anat outputs (can come from running anat or previous run)
    if skip_flags.skip_tumor_seg:
        decision.run_tumor_seg = False
        decision.tumor_seg_reason = "Skipped by user (--skip-tumor-seg)"
    elif not session_data.pipelines.tumor_seg:
        decision.run_tumor_seg = False
        decision.tumor_seg_reason = session_data.pipelines.tumor_seg_reason
    else:
        # Tumor segmentation can run if anat is running OR if previous anat outputs exist
        # (checked at workflow connection time)
        device_available, device_reason = _check_tumor_seg_device(tumor_seg_config)
        if not device_available:
            decision.run_tumor_seg = False
            decision.tumor_seg_reason = device_reason
        else:
            decision.run_tumor_seg = True

    # DSC perfusion
    if skip_flags.skip_dsc:
        decision.run_dsc = False
        decision.dsc_reason = "Skipped by user (--skip-dsc)"
    elif not session_data.pipelines.dsc:
        decision.run_dsc = False
        decision.dsc_reason = session_data.pipelines.dsc_reason
    else:
        decision.run_dsc = True

    # DCE perfusion
    if skip_flags.skip_dce:
        decision.run_dce = False
        decision.dce_reason = "Skipped by user (--skip-dce)"
    elif not session_data.pipelines.dce:
        decision.run_dce = False
        decision.dce_reason = session_data.pipelines.dce_reason
    else:
        decision.run_dce = True

    return decision


def _check_tumor_seg_device(
    config: Optional[TumorSegConfig] = None,
) -> tuple[bool, str]:
    """Check whether the requested tumor-seg device is available.

    When ``conda_env`` is set, the check is skipped because the host
    environment's PyTorch may differ from the conda env where HD-GLIO
    actually runs (e.g. bare-metal MPS with --tumor-seg-conda-env hdglio-mps).

    When config is None (e.g. in tests that don't pass a config), returns
    True — the data-availability check earlier in decide_pipelines already
    gates whether tumor seg can run.

    Args:
        config: Tumor segmentation configuration, or None.

    Returns:
        Tuple of (available, reason). Reason is only meaningful when
        available is False.
    """
    if config is None:
        return True, ""

    # When running in a separate conda env, the host PyTorch is irrelevant
    if config.conda_env:
        return True, ""

    device = config.device

    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                return False, (
                    "No CUDA GPU detected (--tumor-seg-device cuda). "
                    "Tumor segmentation requires GPU acceleration. "
                    "Use --skip-tumor-seg to proceed without tumor segmentation."
                )
        except ImportError:
            return False, (
                "PyTorch not installed; cannot run tumor segmentation. "
                "Use --skip-tumor-seg to proceed without it."
            )
    elif device == "mps":
        try:
            import torch

            if not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                return False, (
                    "MPS not available (--tumor-seg-device mps). "
                    "Tumor segmentation requires GPU acceleration. "
                    "Use --skip-tumor-seg to proceed without tumor segmentation."
                )
        except ImportError:
            return False, (
                "PyTorch not installed; cannot run tumor segmentation. "
                "Use --skip-tumor-seg to proceed without it."
            )
    # device == "cpu": always available (user explicitly chose slow path)

    return True, ""


def _find_previous_anat_outputs(
    work_dir: Path,
    subject_id: str,
    session_id: Optional[str],
) -> dict[str, Optional[str]]:
    """Find previously-computed anat outputs in the work directory.

    When --skip-anat is used, downstream workflows still need anat outputs
    (brain mask, T1w reference, tissue segmentation, ACPC transform).
    This function locates them from a prior run's work directory.

    Args:
        work_dir: Nipype working directory (parent of the workflow directory).
        subject_id: BIDS subject ID (without 'sub-' prefix).
        session_id: BIDS session ID (without 'ses-' prefix), or None.

    Returns:
        Dict with keys: t1w_preproc, brain_mask, wm_probseg, csf_probseg, t1w_to_mni_xfm, flair_coreg, t2w_coreg.
        Values are file paths (str) or None if not found.
    """
    session_label = f"ses-{session_id}" if session_id else "no-session"
    wf_name = f"pmriprep_sub{subject_id}_{session_label}_wf"
    anat_dir = work_dir / wf_name / "anat_wf"

    outputs: dict[str, Optional[str]] = {
        "t1w_preproc": None,
        "t1w_postcon_coreg": None,
        "brain_mask": None,
        "wm_probseg": None,
        "gm_probseg": None,
        "csf_probseg": None,
        "t1w_to_mni_xfm": None,
        "flair_coreg": None,
        "flair_postcon_coreg": None,
        "t2w_coreg": None,
    }

    if not anat_dir.exists():
        return outputs

    # T1w preprocessed (bias-corrected)
    bias_dir = anat_dir / "bias_correct"
    if bias_dir.exists():
        candidates = sorted(bias_dir.glob("*_biascorr.nii.gz"))
        if candidates:
            outputs["t1w_preproc"] = str(candidates[0])

    # Brain mask
    brain_dir = anat_dir / "brain_extract"
    if brain_dir.exists():
        candidates = sorted(brain_dir.glob("*_brain_mask.nii.gz"))
        if candidates:
            outputs["brain_mask"] = str(candidates[0])

    # White matter probability segmentation (FAST pve_2 = WM)
    seg_dir = anat_dir / "tissue_segment"
    if seg_dir.exists():
        candidates = sorted(seg_dir.glob("*_pve_2.nii.gz"))
        if candidates:
            outputs["wm_probseg"] = str(candidates[0])

    # GM probability segmentation (FAST pve_1 = GM)
    if seg_dir.exists():
        candidates = sorted(seg_dir.glob("*_pve_1.nii.gz"))
        if candidates:
            outputs["gm_probseg"] = str(candidates[0])

    # CSF probability segmentation (FAST pve_0 = CSF)
    if seg_dir.exists():
        candidates = sorted(seg_dir.glob("*_pve_0.nii.gz"))
        if candidates:
            outputs["csf_probseg"] = str(candidates[0])

    # ACPC alignment transform
    acpc_dir = anat_dir / "acpc_align"
    if acpc_dir.exists():
        candidates = sorted(acpc_dir.glob("*.mat"))
        if candidates:
            outputs["t1w_to_mni_xfm"] = str(candidates[0])

    # FLAIR coregistered to T1w
    flair_dir = anat_dir / "coreg_flair"
    if flair_dir.exists():
        candidates = sorted(flair_dir.glob("*_coreg.nii.gz"))
        if candidates:
            outputs["flair_coreg"] = str(candidates[0])

    # T2w coregistered to T1w
    t2w_dir = anat_dir / "mask_t2w"
    if t2w_dir.exists():
        candidates = sorted(t2w_dir.glob("*_masked.nii.gz"))
        if candidates:
            outputs["t2w_coreg"] = str(candidates[0])

    # T1w post-contrast coregistered and masked
    postcon_dir = anat_dir / "mask_postcon"
    if postcon_dir.exists():
        candidates = sorted(postcon_dir.glob("*_masked.nii.gz"))
        if candidates:
            outputs["t1w_postcon_coreg"] = str(candidates[0])

    # FLAIR post-contrast coregistered and masked
    flair_postcon_dir = anat_dir / "mask_flair_postcon"
    if flair_postcon_dir.exists():
        candidates = sorted(flair_postcon_dir.glob("*_masked.nii.gz"))
        if candidates:
            outputs["flair_postcon_coreg"] = str(candidates[0])

    return outputs


def _find_previous_tumor_seg_inputs(
    work_dir: Path,
    subject_id: str,
    session_id: Optional[str],
) -> dict[str, Optional[str]]:
    """Find previously-computed anat outputs needed for tumor segmentation.

    When --skip-anat is used but tumor segmentation should run, this function
    locates the skull-stripped, coregistered images from a prior anat run.

    Args:
        work_dir: Nipype working directory (parent of the workflow directory).
        subject_id: BIDS subject ID (without 'sub-' prefix).
        session_id: BIDS session ID (without 'ses-' prefix), or None.

    Returns:
        Dict with keys for tumor segmentation inputs. Values are file paths or None.
    """
    session_label = f"ses-{session_id}" if session_id else "no-session"
    wf_name = f"pmriprep_sub{subject_id}_{session_label}_wf"
    anat_dir = work_dir / wf_name / "anat_wf"

    outputs: dict[str, Optional[str]] = {
        "tumor_seg_t1w_pre": None,
        "tumor_seg_t1w_postcon": None,
        "tumor_seg_t2w": None,
        "tumor_seg_flair": None,
        "brain_mask": None,
        "wm_probseg": None,
        "t1w_to_mni_xfm": None,
        "t1w_preproc": None,
    }

    if not anat_dir.exists():
        return outputs

    # Tumor seg masked T1w (pre-contrast) - skull-stripped without bias correction
    tumor_seg_t1w_dir = anat_dir / "tumor_seg_mask_t1w"
    if tumor_seg_t1w_dir.exists():
        candidates = sorted(tumor_seg_t1w_dir.glob("*_masked.nii.gz"))
        if candidates:
            outputs["tumor_seg_t1w_pre"] = str(candidates[0])

    # Tumor seg masked post-contrast T1w
    tumor_seg_postcon_dir = anat_dir / "tumor_seg_mask_postcon"
    if tumor_seg_postcon_dir.exists():
        candidates = sorted(tumor_seg_postcon_dir.glob("*_masked.nii.gz"))
        if candidates:
            outputs["tumor_seg_t1w_postcon"] = str(candidates[0])

    # Tumor seg masked T2w
    tumor_seg_t2w_dir = anat_dir / "tumor_seg_mask_t2w"
    if tumor_seg_t2w_dir.exists():
        candidates = sorted(tumor_seg_t2w_dir.glob("*_masked.nii.gz"))
        if candidates:
            outputs["tumor_seg_t2w"] = str(candidates[0])

    # Tumor seg masked FLAIR
    tumor_seg_flair_dir = anat_dir / "tumor_seg_mask_flair"
    if tumor_seg_flair_dir.exists():
        candidates = sorted(tumor_seg_flair_dir.glob("*_masked.nii.gz"))
        if candidates:
            outputs["tumor_seg_flair"] = str(candidates[0])

    # Brain mask
    brain_dir = anat_dir / "brain_extract"
    if brain_dir.exists():
        candidates = sorted(brain_dir.glob("*_brain_mask.nii.gz"))
        if candidates:
            outputs["brain_mask"] = str(candidates[0])

    # White matter probability segmentation
    seg_dir = anat_dir / "tissue_segment"
    if seg_dir.exists():
        candidates = sorted(seg_dir.glob("*_pve_2.nii.gz"))
        if candidates:
            outputs["wm_probseg"] = str(candidates[0])

    # ACPC alignment transform
    acpc_dir = anat_dir / "acpc_align"
    if acpc_dir.exists():
        candidates = sorted(acpc_dir.glob("*.mat"))
        if candidates:
            outputs["t1w_to_mni_xfm"] = str(candidates[0])

    # T1w preprocessed (bias-corrected)
    bias_dir = anat_dir / "bias_correct"
    if bias_dir.exists():
        candidates = sorted(bias_dir.glob("*_biascorr.nii.gz"))
        if candidates:
            outputs["t1w_preproc"] = str(candidates[0])

    return outputs


def _read_bids_sidecar(
    nifti_path: Union[str, Path],
) -> dict[str, Any]:
    """Read acquisition parameters from the BIDS sidecar JSON.

    Looks for the JSON sidecar corresponding to a NIfTI file and extracts
    EchoTime, RepetitionTime, RepetitionTimeExcitation, MagneticFieldStrength,
    and contrast agent information.

    For DCE 3D sequences, there are two different TRs:
    - Volume TR (RepetitionTime): Time between successive 3D volume acquisitions
      (~5-10 seconds). Used for: time vector, model auto-selection, AIF timing.
    - Pulse TR (RepetitionTimeExcitation): Time between RF excitations (~3-10 ms).
      Used for: SPGR signal equation in signal-to-concentration conversion.

    Args:
        nifti_path: Path to a NIfTI file (.nii.gz or .nii).

    Returns:
        Dict with keys: te, tr, volume_tr, pulse_tr, field_strength, flip_angle,
        r1_relaxivity, contrast_agent.
        - tr: Pulse TR for signal equation (defaults to pulse_tr)
        - volume_tr: Volume TR for temporal resolution (model selection, time vector)
        - pulse_tr: Pulse TR for SPGR signal equation
        - r1_relaxivity: r1 from contrast agent lookup, or None
        - contrast_agent: Contrast agent name from DICOM, or None
        Values are floats/str or None if not found in the sidecar.
    """
    import json

    from pmriprep.utils.contrast_agent import get_r1_from_sidecar_dict

    result: dict[str, Any] = {
        "te": None,
        "tr": None,
        "volume_tr": None,
        "pulse_tr": None,
        "field_strength": None,
        "flip_angle": None,
        "r1_relaxivity": None,
        "contrast_agent": None,
        "bids_agent_string": None,  # Original ContrastBolusAgent from sidecar
        "agent_standard_dose": 0.1,  # Default standard dose (mmol/kg)
        "mr_acquisition_type": "",   # BIDS MRAcquisitionType ('3D'/'2D')
    }

    nifti = Path(nifti_path)
    sidecar = nifti.with_name(
        nifti.name.replace(".nii.gz", ".json").replace(".nii", ".json")
    )
    if not sidecar.exists():
        return result

    with open(sidecar) as f:
        meta = json.load(f)

    if "EchoTime" in meta:
        result["te"] = float(meta["EchoTime"])

    # Volume TR: temporal resolution between 3D volume acquisitions
    # Used for: time vector, model auto-selection, AIF timing
    if "RepetitionTime" in meta:
        result["volume_tr"] = float(meta["RepetitionTime"])

    # Pulse TR: time between RF excitations (for SPGR signal equation)
    # Used for: signal-to-concentration conversion
    if "RepetitionTimeExcitation" in meta:
        result["pulse_tr"] = float(meta["RepetitionTimeExcitation"])
    elif "RepetitionTime" in meta:
        # Fallback: if no pulse TR, use volume TR (2D sequences)
        result["pulse_tr"] = float(meta["RepetitionTime"])

    # For backward compatibility, "tr" defaults to pulse TR (signal equation)
    result["tr"] = result.get("pulse_tr")

    if "MagneticFieldStrength" in meta:
        result["field_strength"] = float(meta["MagneticFieldStrength"])

    if "FlipAngle" in meta:
        result["flip_angle"] = float(meta["FlipAngle"])

    # MRAcquisitionType ("3D" or "2D") gates the inferior-side inflow
    # exclusion in DCE AIF candidate search. 3D-slab excitation has
    # entry-slice inflow contamination at the inferior partitions
    # (van der Schaaf MRM 2006); 2D multi-slice does not.
    if "MRAcquisitionType" in meta:
        result["mr_acquisition_type"] = str(meta["MRAcquisitionType"])

    # Extract contrast agent r1 relaxivity from DICOM metadata
    # This enables agent-specific r1 values (e.g., Gadovist r1=5.0 at 3T)
    r1_info = get_r1_from_sidecar_dict(meta, result["field_strength"])
    if r1_info["r1"] is not None:
        result["r1_relaxivity"] = r1_info["r1"]
        result["contrast_agent"] = r1_info["agent_name"]
    if r1_info.get("agent_string"):
        result["bids_agent_string"] = r1_info["agent_string"]

    # Look up agent-specific standard clinical dose
    if r1_info["agent_name"] is not None:
        from pmriprep.utils.contrast_agent import lookup_standard_dose
        result["agent_standard_dose"] = lookup_standard_dose(r1_info["agent_name"])

    return result


def _ensure_tr_ms(
    tr_value: float,
    label: str = "TR",
) -> float:
    """Convert a TR/TE value to milliseconds, auto-detecting the input unit.

    BIDS stores RepetitionTime and EchoTime in seconds, but some non-compliant
    datasets or manual overrides may already be in milliseconds.

    Heuristic: brain MRI excitation TR is typically 2-50 ms (0.002-0.050 s).
    Values < 1.0 are assumed seconds; values >= 1.0 are assumed milliseconds.
    """
    import logging

    _log = logging.getLogger("pmriprep.workflows")

    if tr_value <= 0:
        _log.warning("%s value is non-positive (%.6f), returning as-is", label, tr_value)
        return tr_value

    if tr_value < 1.0:
        # Likely in seconds (BIDS-compliant) — convert to ms
        tr_ms = tr_value * 1000.0
        _log.debug("%s: %.6f s → %.2f ms", label, tr_value, tr_ms)
        return tr_ms

    # Already in milliseconds (or a very long TR like volume TR for 3D dynamic)
    if tr_value > 500.0:
        _log.warning(
            "%s value %.2f ms is unusually large for an excitation TR; "
            "verify this is not a volume TR",
            label,
            tr_value,
        )
    return tr_value


def _validate_flip_angle(fa: float, label: str = "") -> float:
    """Validate a flip angle value is in degrees and within a plausible range.

    Typical VFA flip angles: 2-30 degrees.
    DCE flip angles: 10-35 degrees.
    """
    import logging

    _log = logging.getLogger("pmriprep.workflows")

    if fa <= 0:
        _log.warning("Flip angle %s is non-positive (%.2f), this is likely an error", label, fa)
    elif fa > 90:
        _log.warning(
            "Flip angle %s = %.2f° exceeds 90° — may be in radians or invalid",
            label,
            fa,
        )
    return fa


def _find_previous_tumor_seg_outputs(
    work_dir: Path,
    subject_id: str,
    session_id: Optional[str],
) -> dict[str, Optional[str]]:
    """Find previously-computed tumor segmentation outputs in the work directory.

    When --skip-tumor-seg is used, DSC/DCE workflows may still need
    lesion and NAWM masks from a prior tumor segmentation run.

    Args:
        work_dir: Nipype working directory.
        subject_id: BIDS subject ID (without 'sub-' prefix).
        session_id: BIDS session ID (without 'ses-' prefix), or None.

    Returns:
        Dict with keys: whole_tumor_mask, enhancing_mask,
        segmentation, nawm_mask.
        Values are file paths (str) or None if not found.
    """
    session_label = f"ses-{session_id}" if session_id else "no-session"
    wf_name = f"pmriprep_sub{subject_id}_{session_label}_wf"
    tumor_seg_dir = work_dir / wf_name / "tumor_seg_wf"

    outputs: dict[str, Optional[str]] = {
        "whole_tumor_mask": None,
        "enhancing_mask": None,
        "non_enhancing_mask": None,
        "segmentation": None,
        "nawm_mask": None,
    }

    if not tumor_seg_dir.exists():
        return outputs

    masks_dir = tumor_seg_dir / "extract_masks"
    if masks_dir.exists():
        for f in masks_dir.glob("*_nawm.nii.gz"):
            outputs["nawm_mask"] = str(f)
        for f in masks_dir.glob("*wholetumor_mask.nii.gz"):
            outputs["whole_tumor_mask"] = str(f)
        for f in masks_dir.glob("*enhancing_mask.nii.gz"):
            if "nonenhancing" not in f.name and "non_enhancing" not in f.name:
                outputs["enhancing_mask"] = str(f)
        for f in masks_dir.glob("*nonenhancing_mask.nii.gz"):
            outputs["non_enhancing_mask"] = str(f)
        for f in masks_dir.glob("*non_enhancing_mask.nii.gz"):
            outputs["non_enhancing_mask"] = str(f)

    # Check tumor_seg_segment node for segmentation output
    seg_dir = tumor_seg_dir / "tumor_seg_segment"
    if seg_dir.exists():
        for f in seg_dir.glob("*_seg.nii.gz"):
            outputs["segmentation"] = str(f)
        for f in seg_dir.glob("*segmentation*.nii.gz"):
            outputs["segmentation"] = str(f)

    if outputs["nawm_mask"] is None:
        nawm_dir = tumor_seg_dir / "create_nawm"
        if nawm_dir.exists():
            for f in nawm_dir.glob("*.nii.gz"):
                outputs["nawm_mask"] = str(f)

    # Check select_whole_tumor node for whole tumor mask
    if outputs["whole_tumor_mask"] is None:
        wt_dir = tumor_seg_dir / "select_whole_tumor"
        if wt_dir.exists():
            for f in wt_dir.glob("*.nii.gz"):
                outputs["whole_tumor_mask"] = str(f)

    # Check select_enhancing node for enhancing mask
    if outputs["enhancing_mask"] is None:
        enh_dir = tumor_seg_dir / "select_enhancing"
        if enh_dir.exists():
            for f in enh_dir.glob("*.nii.gz"):
                outputs["enhancing_mask"] = str(f)

    # Check select_non_enhancing node for non-enhancing mask
    if outputs["non_enhancing_mask"] is None:
        ne_dir = tumor_seg_dir / "select_non_enhancing"
        if ne_dir.exists():
            for f in ne_dir.glob("*.nii.gz"):
                outputs["non_enhancing_mask"] = str(f)

    return outputs


def _pick_nawm_mask(tumor_seg_nawm=None, user_nawm=None):  # type: ignore[no-untyped-def]
    """Pick best NAWM mask: prefer tumor-seg, fall back to user-supplied."""
    from pathlib import Path

    for candidate in (tumor_seg_nawm, user_nawm):
        if candidate is not None and candidate != "None":
            p = Path(candidate)
            if p.exists():
                return str(p)
    return ""


# Must match ``ComputeDCEMaps._MAX_VOLUME_TR_2CXM`` in
# ``pmriprep/interfaces/dce_compute.py:1365``. The constant lives on the
# producer class as a private attribute; duplicating with a named source
# reference avoids a new public-import boundary while keeping the gate
# auditable.
_MAX_VOLUME_TR_2CXM_BUILD = 3.0


def _emit_2cxm_uncertainty(
    model: Optional[str], volume_tr: Optional[float],
) -> bool:
    """Return True iff the DCE producer will emit Fp/PS uncertainty maps.

    Mirrors the runtime gate in
    :func:`pmriprep.interfaces.dce_compute.ComputeDCEMaps._build_candidate_set`
    (line 1388): 2CXM is in the auto-mode candidate set only when
    ``volume_tr <= _MAX_VOLUME_TR_2CXM`` (3.0 s). The producer's
    ``save_2cxm`` flag (``dce_compute.py:1822``) then gates whether
    ``out_fp_std`` / ``out_ps_std`` are saved. Build-time gating must
    match — building a sink the producer never feeds raises
    ``NodeExecutionError: _copy_perf_map missing in_file`` at runtime.

    Args:
        model: Resolved DCE model name (``"auto"``, ``"2cxm"``,
            ``"tofts"``, etc.). ``None`` is treated as no-emission.
        volume_tr: DCE volume TR in seconds (BIDS ``RepetitionTime``).
            ``None`` is treated as long TR (no-emission); a malformed
            sidecar is preferred-conservative over a runtime crash.

    Returns:
        True if the build-time ``dce_present`` set should include
        ``fp_uncertainty`` and ``ps_uncertainty``.
    """
    if model == "2cxm":
        return True
    if model == "auto" and volume_tr is not None:
        try:
            return float(volume_tr) <= _MAX_VOLUME_TR_2CXM_BUILD
        except (TypeError, ValueError):
            return False
    return False


def _find_previous_dsc_outputs(
    output_dir: Path,
    subject_id: str,
    session_id: Optional[str],
) -> dict[str, Optional[str]]:
    """Find previously-computed DSC outputs in the derivatives directory.

    When --skip-dsc is used, the DCE workflow may still need DSC CBV and
    the DSC→T1w transform for vp constraint and cross-validation.

    Args:
        output_dir: Path to output derivatives directory.
        subject_id: BIDS subject ID (without 'sub-' prefix).
        session_id: BIDS session ID (without 'ses-' prefix), or None.

    Returns:
        Dict with keys: rel_cbv_map, dsc_to_t1w_xfm, aif_qc, preload_qc, preprocessing_stats.
        Values are file paths (str) or None if not found.
    """
    outputs: dict[str, Optional[str]] = {
        "rel_cbv_map": None,
        "ncbv_map": None,
        "dsc_to_t1w_xfm": None,
        "aif_qc": None,
        "preload_qc": None,
        "preprocessing_stats": None,
    }

    ses_label = f"ses-{session_id}" if session_id else ""
    dsc_dir = output_dir / "pmriprep" / f"sub-{subject_id}"
    if ses_label:
        dsc_dir = dsc_dir / ses_label

    if not dsc_dir.exists():
        return outputs

    # AUC-based CBV map (primary); fall back to deconv CBV for older outputs
    perf_dir = dsc_dir / "perf"
    if perf_dir.exists():
        for f in perf_dir.glob("*desc-dscRelAUCCbv_stat.nii.gz"):
            outputs["rel_cbv_map"] = str(f)
        if outputs["rel_cbv_map"] is None:
            for f in perf_dir.glob("*desc-dscRelCbv_stat.nii.gz"):
                outputs["rel_cbv_map"] = str(f)
        for f in perf_dir.glob("*desc-dscNawmNormCbv_stat.nii.gz"):
            outputs["ncbv_map"] = str(f)
        if outputs["rel_cbv_map"] is not None and outputs["ncbv_map"] is None:
            logging.getLogger("pmriprep.workflows").warning(
                "Found previous DSC outputs for sub-%s without a NAWM-"
                "normalized CBV map (no *desc-dscNawmNormCbv_stat.nii.gz "
                "in %s). DCE vp constraint will SKIP. Re-run DSC to "
                "obtain nCBV.",
                subject_id, perf_dir,
            )

    # DSC→T1w transform (now in anat/ instead of xfm/)
    anat_dir = dsc_dir / "anat"
    if anat_dir.exists():
        for f in anat_dir.glob("*from-dscref_to-T1w*_xfm.mat"):
            outputs["dsc_to_t1w_xfm"] = str(f)

    # AIF / preload / preprocessing QC from the pipeline-internal resume
    # sidecar at perf/. Carries verbatim producer-shape dicts so the
    # downstream consumers (PK setup + dsc_cbv_constraint) receive the
    # same dict structure they would on the in-process DSC→DCE path.
    prefix = f"sub-{subject_id}"
    if ses_label:
        prefix += f"_{ses_label}"
    if perf_dir.exists():
        resume_json = perf_dir / f"{prefix}_desc-dscResume_pmriprep.json"
        if resume_json.exists():
            try:
                import json
                with open(resume_json) as fh:
                    data = json.load(fh)
            except Exception:
                data = {}
            if isinstance(data, dict):
                if isinstance(data.get("aif_qc"), dict):
                    outputs["aif_qc"] = data["aif_qc"]
                if isinstance(data.get("preload_qc"), dict):
                    outputs["preload_qc"] = data["preload_qc"]
                if isinstance(data.get("preprocessing_stats"), dict):
                    outputs["preprocessing_stats"] = data["preprocessing_stats"]

    return outputs


def init_pmriprep_wf(
    bids_dir: Union[str, Path],
    output_dir: Union[str, Path],
    subject_id: str,
    session_id: Optional[str] = None,
    skip_flags: Optional[PipelineSkipFlags] = None,
    anat_config: Optional[AnatConfig] = None,
    tumor_seg_config: Optional[TumorSegConfig] = None,
    dsc_config: Optional[DSCConfig] = None,
    dce_config: Optional[DCEConfig] = None,
    work_dir: Optional[Union[str, Path]] = None,
    random_seed: Optional[int] = None,
    mem_limit_gb: Optional[float] = None,
    command_string: str = "",
    user_specified: Optional[dict[str, bool]] = None,
    name: str = "pmriprep_wf",
    layout: Optional["BIDSLayout"] = None,
) -> pe.Workflow:
    """Initialize the master PMRIPrep workflow.

    This workflow orchestrates all pipeline components based on available
    data and user-specified skip flags. It ensures proper ordering of
    pipelines (anatomical first, then tumor segmentation, then perfusion)
    and connects outputs between dependent pipelines.

    Parameters
    ----------
    bids_dir : Path or str
        Path to BIDS dataset root directory.
    output_dir : Path or str
        Path to output derivatives directory.
    subject_id : str
        BIDS subject identifier (without 'sub-' prefix).
    session_id : str, optional
        BIDS session identifier (without 'ses-' prefix).
    skip_flags : PipelineSkipFlags, optional
        Flags to explicitly skip pipeline components.
    anat_config : AnatConfig, optional
        Configuration for anatomical preprocessing.
    tumor_seg_config : TumorSegConfig, optional
        Configuration for tumor segmentation.
    dsc_config : DSCConfig, optional
        Configuration for DSC perfusion processing.
    dce_config : DCEConfig, optional
        Configuration for DCE perfusion processing.
    work_dir : Path or str, optional
        Nipype working directory. Required when --skip-anat or --skip-tumor-seg
        is used to locate previously-computed outputs.
    random_seed : int, optional
        Random seed for reproducibility. Passed to ProcessPoolExecutor
        workers for deterministic Bayesian fitting.
    user_specified : dict, optional
        Tracks which parameters were explicitly provided by the user
        (via CLI or config file) vs falling back to pipeline defaults.
        Keys: dce_dose, dsc_dose, dsc_agent_standard_dose, dsc_preload.
    name : str
        Workflow name (default: "pmriprep_wf").

    Returns
    -------
    pe.Workflow
        Nipype workflow for PMRIPrep processing.

    Notes
    -----
    The workflow follows this execution order:
    1. Anatomical preprocessing (always first)
    2. Tumor segmentation (requires anat outputs)
    3. DSC perfusion processing (requires anat outputs, optional tumor seg)
    4. DCE perfusion processing (requires anat outputs, uses DSC CBV if available)

    When both DSC and DCE are present, DCE runs after DSC to use DSC's CBV
    for vp constraint (via US-063 QC check). This is handled by connecting
    DSC outputs to DCE inputs.
    """
    from pmriprep.workflows.anat.base import init_anat_preproc_wf
    from pmriprep.workflows.dce.base import init_dce_preproc_wf
    from pmriprep.workflows.dsc.base import init_dsc_preproc_wf
    from pmriprep.workflows.tumor_seg.base import init_tumor_seg_wf

    # Normalize paths (resolve to absolute for DataSink and FSL commands)
    bids_dir = Path(bids_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if work_dir is not None:
        work_dir = Path(work_dir).resolve()

    # Use default skip flags if not provided
    if skip_flags is None:
        skip_flags = PipelineSkipFlags()

    # Use default configs if not provided
    if anat_config is None:
        anat_config = AnatConfig()
    if tumor_seg_config is None:
        tumor_seg_config = TumorSegConfig()
    if dsc_config is None:
        dsc_config = DSCConfig()
    if dce_config is None:
        dce_config = DCEConfig()

    # Discover available data
    session_data = discover_session_data(
        bids_dir, subject_id, session_id, layout=layout
    )

    # Modality-presence flags used by every per-pipeline DerivativesDataSink
    # factory (anat, tumor_seg, dsc, dce). Computed once at the top so they
    # are in scope under --skip-anat where the anat block is skipped but
    # tumor_seg/dsc/dce factories still run.
    has_t1w_pre = bool(session_data.modalities.t1w_pre)
    has_t1w_postcon = bool(session_data.modalities.t1w_ce_gad)
    is_postcon_ref = has_t1w_postcon and not has_t1w_pre

    # Estimate memory requirements from image dimensions
    from pmriprep.utils.mem import cap_mem_gb, estimate_4d_mem_gb

    if session_data.modalities.dsc:
        dsc_mem_gb, dsc_mem_raw = estimate_4d_mem_gb(
            session_data.modalities.dsc, n_buffers=4, mem_limit_gb=mem_limit_gb,
        )
    else:
        dsc_mem_gb, dsc_mem_raw = cap_mem_gb(4.0, mem_limit_gb), 4.0

    if session_data.modalities.dce:
        dce_mem_gb, dce_mem_raw = estimate_4d_mem_gb(
            session_data.modalities.dce, n_buffers=5, mem_limit_gb=mem_limit_gb,
        )
    else:
        dce_mem_gb, dce_mem_raw = cap_mem_gb(6.0, mem_limit_gb), 6.0

    # Warn on console if memory may be insufficient
    from pmriprep.utils.mem import _system_ram_gb

    _cli_log = logging.getLogger("cli")
    if not _cli_log.handlers:
        _cli_log = logging.getLogger("pmriprep.workflows")
    _sys_ram = _system_ram_gb()
    for label, raw, capped in [("DSC", dsc_mem_raw, dsc_mem_gb), ("DCE", dce_mem_raw, dce_mem_gb)]:
        if mem_limit_gb is not None and capped >= mem_limit_gb * 0.9:
            _cli_log.warning(
                "%s processing needs ~%.1f GB but --mem-gb is %.1f GB. "
                "Consider setting --mem-gb to at least %.0f.",
                label, raw, mem_limit_gb, raw + 1,
            )
        elif raw >= _sys_ram * 0.8:
            _cli_log.warning(
                "%s processing needs ~%.1f GB but system has %.1f GB RAM. "
                "This dataset may require a machine with more memory.",
                label, raw, _sys_ram,
            )

    # Decide which pipelines to run
    decision = decide_pipelines(session_data, skip_flags, tumor_seg_config)

    # Create master workflow
    workflow = pe.Workflow(name=name)

    # Store metadata on workflow object
    workflow.bids_dir = bids_dir  # type: ignore[attr-defined]
    workflow.output_dir = output_dir  # type: ignore[attr-defined]
    workflow.subject_id = subject_id  # type: ignore[attr-defined]
    workflow.session_id = session_id  # type: ignore[attr-defined]
    workflow.session_data = session_data  # type: ignore[attr-defined]
    workflow.pipeline_decision = decision  # type: ignore[attr-defined]

    # Define master input node
    # This node receives BIDS paths from the discovery module
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # Anatomical inputs
                "t1w_pre",
                "t1w_postcon",
                "t2w",
                "flair",
                "flair_postcon",
                # Perfusion inputs
                "dsc_file",
                "dce_file",
                # Mapping inputs
                "vfa_files",
                "afi_files",
                "dwi_file",
                # Lesion mask (from external source or previous run)
                "lesion_mask",
            ]
        ),
        name="inputnode",
    )

    # Set input values from discovered paths
    if session_data.modalities.t1w_pre:
        inputnode.inputs.t1w_pre = str(session_data.modalities.t1w_pre)
    if session_data.modalities.t1w_ce_gad:
        inputnode.inputs.t1w_postcon = str(session_data.modalities.t1w_ce_gad)
    if session_data.modalities.t2w:
        inputnode.inputs.t2w = str(session_data.modalities.t2w)
    if session_data.modalities.flair:
        inputnode.inputs.flair = str(session_data.modalities.flair)
    if session_data.modalities.flair_postcon:
        inputnode.inputs.flair_postcon = str(session_data.modalities.flair_postcon)
    if session_data.modalities.dsc:
        inputnode.inputs.dsc_file = str(session_data.modalities.dsc)
    if session_data.modalities.dce:
        inputnode.inputs.dce_file = str(session_data.modalities.dce)
    if session_data.modalities.vfa:
        inputnode.inputs.vfa_files = [str(p) for p in session_data.modalities.vfa]
    if session_data.modalities.afi:
        inputnode.inputs.afi_files = [str(p) for p in session_data.modalities.afi]
    if session_data.modalities.dwi:
        inputnode.inputs.dwi_file = str(session_data.modalities.dwi)

    # -------------------------------------------------------------------
    # Structural reference selection (confidence-scored)
    # Determine which anat output to use for DSC and DCE registration,
    # plus BBR eligibility and provenance metadata.
    # -------------------------------------------------------------------
    from pmriprep.utils.structural_reference import (
        ImageType,
        build_candidates_from_session,
        select_structural_reference_v2,
        _map_selection_to_anat_output,
    )

    _ref_logger = logging.getLogger("pmriprep.workflows")

    # Determine if a lesion mask will be available at perfusion registration
    has_lesion_mask = (
        (dsc_config is not None and dsc_config.lesion_mask is not None)
        or (dce_config is not None and dce_config.lesion_mask is not None)
        or session_data.pipelines.tumor_seg
    )

    # Build candidates from discovered modalities
    ref_candidates = build_candidates_from_session(
        session_data.modalities, has_lesion_mask=has_lesion_mask,
    )

    # Select for DSC registration
    dsc_ref_selection = None
    if decision.run_dsc:
        dsc_ref_selection = select_structural_reference_v2(
            ref_candidates,
            application="dsc_registration",
            perfusion_path=(
                str(session_data.modalities.dsc) if session_data.modalities.dsc else None
            ),
        )
        if dsc_ref_selection:
            _ref_logger.info(
                "DSC structural reference: %s (confidence=%.2f, bbr=%s)",
                dsc_ref_selection.candidate.image_type.value,
                dsc_ref_selection.confidence,
                dsc_ref_selection.bbr_eligible,
            )

    # Select for DCE registration
    dce_ref_selection = None
    if decision.run_dce:
        dce_ref_selection = select_structural_reference_v2(
            ref_candidates,
            application="dce_registration",
            perfusion_path=(
                str(session_data.modalities.dce) if session_data.modalities.dce else None
            ),
        )
        if dce_ref_selection:
            _ref_logger.info(
                "DCE structural reference: %s (confidence=%.2f, bbr=%s)",
                dce_ref_selection.candidate.image_type.value,
                dce_ref_selection.confidence,
                dce_ref_selection.bbr_eligible,
            )

    # -------------------------------------------------------------------
    # Pre-flight validation: read sidecars and validate metadata
    # before any workflow initialization to catch errors early.
    # -------------------------------------------------------------------
    from pmriprep.utils.preflight import run_preflight

    dsc_sidecar = None
    dce_sidecar = None
    preflight_vfa_flip_angles: Optional[list[float]] = None
    preflight_vfa_trs: Optional[list[Optional[float]]] = None

    _dsc_agent_dose_source: Optional[str] = None
    _dsc_agent_source: Optional[str] = None
    _gd_multiplicity: int = 1

    if decision.run_dsc and session_data.modalities.dsc:
        dsc_sidecar = _read_bids_sidecar(session_data.modalities.dsc)

        from pmriprep.utils.contrast_agent import ParameterSource
        _user_spec_dsc = user_specified or {}
        if _user_spec_dsc.get("dsc_agent_standard_dose"):
            _dsc_agent_dose_source = ParameterSource.USER_SPECIFIED
        elif dsc_sidecar.get("contrast_agent") is not None:
            dsc_config.agent_standard_dose = dsc_sidecar["agent_standard_dose"]
            _dsc_agent_dose_source = ParameterSource.BIDS_SIDECAR
        else:
            _dsc_agent_dose_source = ParameterSource.DEFAULT_FALLBACK

        if dsc_sidecar.get("contrast_agent") is not None:
            _dsc_agent_source = ParameterSource.BIDS_SIDECAR
        else:
            _dsc_agent_source = ParameterSource.DEFAULT_FALLBACK

        from pmriprep.utils.contrast_agent import lookup_gd_multiplicity
        _gd_multiplicity = (
            lookup_gd_multiplicity(dsc_sidecar["contrast_agent"])
            if dsc_sidecar.get("contrast_agent") is not None
            else 1
        )

    if decision.run_dce and session_data.modalities.dce:
        dce_sidecar = _read_bids_sidecar(session_data.modalities.dce)

        # Read VFA metadata for preflight (if VFA data present)
        if session_data.modalities.vfa:
            preflight_vfa_flip_angles = []
            preflight_vfa_trs = []
            for vfa_path in session_data.modalities.vfa:
                vfa_sc = _read_bids_sidecar(vfa_path)
                if vfa_sc["flip_angle"] is not None:
                    preflight_vfa_flip_angles.append(vfa_sc["flip_angle"])
                preflight_vfa_trs.append(vfa_sc.get("tr"))

    preflight_report = run_preflight(
        session_data=session_data,
        decision=decision,
        dsc_sidecar=dsc_sidecar,
        dce_sidecar=dce_sidecar,
        dce_config=dce_config,
        vfa_flip_angles=preflight_vfa_flip_angles,
        vfa_trs=preflight_vfa_trs,
        dsc_config=dsc_config,
    )
    preflight_report.log_all()
    preflight_report.raise_if_errors()

    dsc_preflight_warnings = preflight_report.warning_strings(prefix="DSC_")
    dce_preflight_warnings = preflight_report.warning_strings(prefix="DCE_")
    dce_preflight_warnings.extend(preflight_report.warning_strings(prefix="VFA_"))
    dce_preflight_warnings.extend(preflight_report.warning_strings(prefix="AFI_"))

    # Define master output node
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # Anatomical outputs
                "t1w_preproc",
                "brain_mask",
                "wm_probseg",
                "gm_probseg",
                "csf_probseg",
                # Tumor segmentation outputs
                "lesion_mask",
                "enhancing_mask",
                "non_enhancing_mask",
                "whole_tumor_mask",
                "nawm_mask",
                # DSC outputs
                "dsc_cbf",
                "dsc_cbv",
                "dsc_mtt",
                "dsc_tmax",
                "dsc_aif_qc",
                # DCE outputs
                "dce_ktrans",
                "dce_ve",
                "dce_vp",
                "dce_kep",
                # QC outputs
                "anat_qc",
                "tumor_seg_qc",
                "dsc_qc",
                "dce_qc",
            ]
        ),
        name="outputnode",
    )

    # ===================================================================
    # Initialize sub-workflows based on decisions
    # ===================================================================

    anat_wf = None
    tumor_seg_wf = None
    dsc_wf = None
    dce_wf = None

    # -------------------------------------------------------------------
    # Unified report barrier + node (collects completion from all writers)
    # -------------------------------------------------------------------
    from pmriprep.interfaces.reports import GenerateUnifiedReport

    report_barrier = pe.Node(
        niu.IdentityInterface(
            fields=["anat_done", "tumor_seg_done", "dsc_done", "dce_done"],
        ),
        name="report_barrier",
    )
    # Defaults for pipelines that don't run
    report_barrier.inputs.anat_done = ""
    report_barrier.inputs.tumor_seg_done = ""
    report_barrier.inputs.dsc_done = ""
    report_barrier.inputs.dce_done = ""

    unified_report = pe.Node(
        GenerateUnifiedReport(
            subject_id=subject_id,
            session_id=session_id or "",
            output_dir=str(output_dir),
            ran_anat=decision.run_anat,
            ran_tumor_seg=decision.run_tumor_seg,
            ran_dsc=decision.run_dsc,
            ran_dce=decision.run_dce,
            command_string=command_string,
        ),
        name="unified_report",
    )

    workflow.connect([
        (report_barrier, unified_report, [
            ("anat_done", "anat_done"),
            ("tumor_seg_done", "tumor_seg_done"),
            ("dsc_done", "dsc_done"),
            ("dce_done", "dce_done"),
        ]),
    ])

    # -------------------------------------------------------------------
    # Anatomical preprocessing workflow (runs first)
    # -------------------------------------------------------------------
    if decision.run_anat:
        anat_wf = init_anat_preproc_wf(
            bids_dir=bids_dir,
            subject_id=subject_id,
            session_id=session_id,
            config=anat_config,
            has_t1w_postcon=(
                session_data.modalities.t1w_pre is not None
                and session_data.modalities.t1w_ce_gad is not None
            ),
            has_t2w=session_data.modalities.t2w is not None,
            has_flair=session_data.modalities.flair is not None,
            has_flair_postcon=session_data.modalities.flair_postcon is not None,
            mem_limit_gb=mem_limit_gb,
            name="anat_wf",
        )

        # Connect inputs to anat workflow
        # Route each image to its semantic field; determine_mode picks the reference.
        if session_data.modalities.t1w_pre:
            workflow.connect([
                (inputnode, anat_wf, [("t1w_pre", "inputnode.t1w_precon")]),
            ])
        if session_data.modalities.t1w_ce_gad:
            workflow.connect([
                (inputnode, anat_wf, [("t1w_postcon", "inputnode.t1w_postcon")]),
            ])

        # Connect additional modalities if available
        if session_data.modalities.t2w:
            workflow.connect([
                (inputnode, anat_wf, [("t2w", "inputnode.t2w")]),
            ])
        if session_data.modalities.flair:
            workflow.connect([
                (inputnode, anat_wf, [("flair", "inputnode.flair")]),
            ])
        if session_data.modalities.flair_postcon:
            workflow.connect([
                (inputnode, anat_wf, [("flair_postcon", "inputnode.flair_postcon")]),
            ])
        # Connect anat outputs to master outputnode
        workflow.connect([
            (anat_wf, outputnode, [
                ("outputnode.t1w_preproc", "t1w_preproc"),
                ("outputnode.brain_mask", "brain_mask"),
                ("outputnode.qc_metrics", "anat_qc"),
            ]),
        ])
        if session_data.modalities.t1w_pre:
            workflow.connect([
                (anat_wf, outputnode, [
                    ("outputnode.wm_probseg", "wm_probseg"),
                    ("outputnode.gm_probseg", "gm_probseg"),
                    ("outputnode.csf_probseg", "csf_probseg"),
                ]),
            ])

        # -------------------------------------------------------------------
        # Anat derivatives: niworkflows DerivativesDataSink chain
        # -------------------------------------------------------------------
        from pmriprep.workflows.anat.datasinks import init_anat_datasinks_wf

        # has_t1w_pre, has_t1w_postcon, is_postcon_ref are computed at the top
        # of init_pmriprep_wf so they're in scope under --skip-anat too.
        anat_datasinks_wf = init_anat_datasinks_wf(
            output_dir=str(output_dir),
            bids_dir=str(bids_dir),
            subject_id=subject_id,
            session_id=session_id,
            has_t1w_pre=has_t1w_pre,
            has_t1w_postcon=has_t1w_postcon,
            has_t2w=bool(session_data.modalities.t2w),
            has_flair=bool(session_data.modalities.flair),
            has_flair_postcon=bool(session_data.modalities.flair_postcon),
            has_tissue_seg=has_t1w_pre,
            has_subtraction=has_t1w_pre and has_t1w_postcon,
            is_postcon_ref=is_postcon_ref,
        )

        workflow.connect([
            (anat_wf, anat_datasinks_wf, [
                ("outputnode.t1w_preproc", "inputnode.t1w_preproc"),
                ("outputnode.brain_mask", "inputnode.brain_mask"),
                ("outputnode.t1w_to_mni_xfm", "inputnode.acpc_xfm"),
                ("outputnode.qc_metrics", "inputnode.qc_metrics"),
                ("outputnode.brain_mask_filmstrip", "inputnode.brain_mask_filmstrip"),
                ("outputnode.acpc_filmstrip", "inputnode.acpc_filmstrip"),
            ]),
        ])

        if has_t1w_pre:
            workflow.connect([
                (anat_wf, anat_datasinks_wf, [
                    ("outputnode.wm_probseg", "inputnode.wm_probseg"),
                    ("outputnode.gm_probseg", "inputnode.gm_probseg"),
                    ("outputnode.csf_probseg", "inputnode.csf_probseg"),
                    ("outputnode.tissue_seg_filmstrip", "inputnode.tissue_seg_filmstrip"),
                ]),
                (inputnode, anat_datasinks_wf, [
                    ("t1w_pre", "inputnode.source_t1w"),
                ]),
            ])

        if has_t1w_postcon and has_t1w_pre:
            workflow.connect([
                (anat_wf, anat_datasinks_wf, [
                    ("outputnode.t1w_postcon_coreg", "inputnode.t1w_postcon_coreg"),
                    ("outputnode.t1w_postcon_to_t1w_xfm",
                     "inputnode.t1w_postcon_to_t1w_xfm"),
                    ("outputnode.t1w_subtraction", "inputnode.t1w_subtraction"),
                    ("outputnode.coreg_filmstrip_postcon",
                     "inputnode.coreg_filmstrip_postcon"),
                ]),
            ])
        if has_t1w_postcon:
            workflow.connect([
                (inputnode, anat_datasinks_wf, [
                    ("t1w_postcon", "inputnode.source_t1w_postcon"),
                ]),
            ])
        if session_data.modalities.t2w:
            workflow.connect([
                (anat_wf, anat_datasinks_wf, [
                    ("outputnode.t2w_coreg", "inputnode.t2w_coreg"),
                    ("outputnode.t2w_to_t1w_xfm", "inputnode.t2w_to_t1w_xfm"),
                    ("outputnode.coreg_filmstrip_t2w", "inputnode.coreg_filmstrip_t2w"),
                ]),
                (inputnode, anat_datasinks_wf, [
                    ("t2w", "inputnode.source_t2w"),
                ]),
            ])
        if session_data.modalities.flair:
            workflow.connect([
                (anat_wf, anat_datasinks_wf, [
                    ("outputnode.flair_coreg", "inputnode.flair_coreg"),
                    ("outputnode.flair_to_t1w_xfm", "inputnode.flair_to_t1w_xfm"),
                    ("outputnode.coreg_filmstrip_flair",
                     "inputnode.coreg_filmstrip_flair"),
                ]),
                (inputnode, anat_datasinks_wf, [
                    ("flair", "inputnode.source_flair"),
                ]),
            ])
        if session_data.modalities.flair_postcon:
            workflow.connect([
                (anat_wf, anat_datasinks_wf, [
                    ("outputnode.flair_postcon_coreg", "inputnode.flair_postcon_coreg"),
                    ("outputnode.flair_postcon_to_t1w_xfm",
                     "inputnode.flair_postcon_to_t1w_xfm"),
                    ("outputnode.coreg_filmstrip_flair_postcon",
                     "inputnode.coreg_filmstrip_flair_postcon"),
                ]),
                (inputnode, anat_datasinks_wf, [
                    ("flair_postcon", "inputnode.source_flair_postcon"),
                ]),
            ])

        workflow.connect([
            (anat_datasinks_wf, report_barrier, [
                ("outputnode.done", "anat_done"),
            ]),
        ])
    # -------------------------------------------------------------------
    # Tumor segmentation workflow (requires anat outputs)
    # -------------------------------------------------------------------
    if decision.run_tumor_seg:
        tumor_seg_wf = init_tumor_seg_wf(
            bids_dir=bids_dir,
            subject_id=subject_id,
            session_id=session_id,
            config=tumor_seg_config,
            mem_limit_gb=mem_limit_gb,
            name="tumor_seg_wf",
        )

        # Connect prepared inputs from anat workflow
        # Tumor segmentation receives coregistered, skull-stripped,
        # non-bias-corrected images from the anat pipeline's tumor seg prep track
        if anat_wf is not None:
            workflow.connect([
                (anat_wf, tumor_seg_wf, [
                    ("outputnode.tumor_seg_t1w_pre", "inputnode.t1w_pre"),
                    ("outputnode.tumor_seg_t1w_postcon", "inputnode.t1w_postcon"),
                    ("outputnode.tumor_seg_t2w", "inputnode.t2w"),
                    ("outputnode.tumor_seg_flair", "inputnode.flair"),
                    ("outputnode.brain_mask", "inputnode.brain_mask"),
                    ("outputnode.wm_probseg", "inputnode.wm_probseg"),
                    ("outputnode.t1w_to_mni_xfm", "inputnode.t1w_to_mni_xfm"),
                    ("outputnode.t1w_preproc", "inputnode.t1w_preproc"),
                ]),
            ])
        elif work_dir is not None:
            # --skip-anat: load previously-computed anat outputs for tumor segmentation
            import logging as _logging
            _log = _logging.getLogger("pmriprep.workflows")
            prev_tumor_seg = _find_previous_tumor_seg_inputs(
                Path(work_dir), subject_id, session_id,
            )
            # Check if required inputs are available
            if prev_tumor_seg["tumor_seg_t1w_pre"]:
                tumor_seg_wf.inputs.inputnode.t1w_pre = prev_tumor_seg["tumor_seg_t1w_pre"]
                _log.info("skip-anat (tumor segmentation): loaded t1w_pre from %s", prev_tumor_seg["tumor_seg_t1w_pre"])
            if prev_tumor_seg["tumor_seg_t1w_postcon"]:
                tumor_seg_wf.inputs.inputnode.t1w_postcon = prev_tumor_seg["tumor_seg_t1w_postcon"]
                _log.info("skip-anat (tumor segmentation): loaded t1w_postcon from %s", prev_tumor_seg["tumor_seg_t1w_postcon"])
            if prev_tumor_seg["tumor_seg_t2w"]:
                tumor_seg_wf.inputs.inputnode.t2w = prev_tumor_seg["tumor_seg_t2w"]
                _log.info("skip-anat (tumor segmentation): loaded t2w from %s", prev_tumor_seg["tumor_seg_t2w"])
            if prev_tumor_seg["tumor_seg_flair"]:
                tumor_seg_wf.inputs.inputnode.flair = prev_tumor_seg["tumor_seg_flair"]
                _log.info("skip-anat (tumor segmentation): loaded flair from %s", prev_tumor_seg["tumor_seg_flair"])
            if prev_tumor_seg["brain_mask"]:
                tumor_seg_wf.inputs.inputnode.brain_mask = prev_tumor_seg["brain_mask"]
                _log.info("skip-anat (tumor segmentation): loaded brain_mask from %s", prev_tumor_seg["brain_mask"])
            if prev_tumor_seg["wm_probseg"]:
                tumor_seg_wf.inputs.inputnode.wm_probseg = prev_tumor_seg["wm_probseg"]
                _log.info("skip-anat (tumor segmentation): loaded wm_probseg from %s", prev_tumor_seg["wm_probseg"])
            if prev_tumor_seg["t1w_to_mni_xfm"]:
                tumor_seg_wf.inputs.inputnode.t1w_to_mni_xfm = prev_tumor_seg["t1w_to_mni_xfm"]
                _log.info("skip-anat (tumor segmentation): loaded t1w_to_mni_xfm from %s", prev_tumor_seg["t1w_to_mni_xfm"])
            if prev_tumor_seg["t1w_preproc"]:
                tumor_seg_wf.inputs.inputnode.t1w_preproc = prev_tumor_seg["t1w_preproc"]
                _log.info("skip-anat (tumor segmentation): loaded t1w_preproc from %s", prev_tumor_seg["t1w_preproc"])

        # Connect tumor segmentation outputs to master outputnode
        workflow.connect([
            (tumor_seg_wf, outputnode, [
                ("outputnode.whole_tumor_mask", "lesion_mask"),
                ("outputnode.enhancing_mask", "enhancing_mask"),
                ("outputnode.non_enhancing_mask", "non_enhancing_mask"),
                ("outputnode.whole_tumor_mask", "whole_tumor_mask"),
                ("outputnode.nawm_mask", "nawm_mask"),
                ("outputnode.qc_metrics", "tumor_seg_qc"),
            ]),
        ])

        # -------------------------------------------------------------------
        # Tumor-seg derivatives: niworkflows DerivativesDataSink chain
        # -------------------------------------------------------------------
        from pmriprep.workflows.tumor_seg.datasinks import init_tumor_seg_datasinks_wf

        tumor_seg_datasinks_wf = init_tumor_seg_datasinks_wf(
            output_dir=str(output_dir),
            bids_dir=str(bids_dir),
            subject_id=subject_id,
            session_id=session_id,
        )

        workflow.connect([
            (tumor_seg_wf, tumor_seg_datasinks_wf, [
                ("outputnode.segmentation", "inputnode.segmentation"),
                ("outputnode.enhancing_mask", "inputnode.enhancing_mask"),
                ("outputnode.non_enhancing_mask", "inputnode.non_enhancing_mask"),
                ("outputnode.whole_tumor_mask", "inputnode.whole_tumor_mask"),
                ("outputnode.nawm_mask", "inputnode.nawm_mask"),
                ("outputnode.qc_metrics", "inputnode.qc_metrics"),
                ("outputnode.seg_filmstrip_postcon",
                 "inputnode.seg_filmstrip_postcon"),
                ("outputnode.seg_filmstrip_flair",
                 "inputnode.seg_filmstrip_flair"),
                ("outputnode.nawm_filmstrip_flair",
                 "inputnode.nawm_filmstrip_flair"),
            ]),
            # HD-GLIO source paths for the per-output Sources field.
            (inputnode, tumor_seg_datasinks_wf, [
                ("t1w_postcon", "inputnode.source_t1w_postcon"),
                ("flair", "inputnode.source_flair"),
            ]),
            (tumor_seg_datasinks_wf, report_barrier, [
                ("outputnode.done", "tumor_seg_done"),
            ]),
        ])
        if has_t1w_pre:
            workflow.connect([
                (inputnode, tumor_seg_datasinks_wf, [
                    ("t1w_pre", "inputnode.source_t1w"),
                ]),
            ])
        if session_data.modalities.t2w:
            workflow.connect([
                (inputnode, tumor_seg_datasinks_wf, [
                    ("t2w", "inputnode.source_t2w"),
                ]),
            ])

    # -------------------------------------------------------------------
    # DSC perfusion workflow (requires anat, optional tumor seg)
    # -------------------------------------------------------------------
    if decision.run_dsc:
        dsc_wf = init_dsc_preproc_wf(
            bids_dir=bids_dir,
            output_dir=output_dir,
            subject_id=subject_id,
            session_id=session_id,
            config=dsc_config,
            random_seed=random_seed,
            mem_gb=dsc_mem_gb,
            mem_limit_gb=mem_limit_gb,
            gd_multiplicity=_gd_multiplicity,
            name="dsc_wf",
        )

        # Read TE/TR/field_strength from BIDS sidecar JSON.
        # All three are required for DSC quantification: TE feeds the
        # signal-to-ΔR2* equation, TR sets temporal sampling, B0 selects
        # field-dependent constants (preload thresholds, contrast r1).
        # Silently defaulting any of these to 3T-Siemens values would
        # mis-quantify cohorts at 1.5T/7T or with sparse sidecars.
        if session_data.modalities.dsc:
            sidecar = _read_bids_sidecar(session_data.modalities.dsc)
            missing = [
                k for k in ("te", "tr", "field_strength") if sidecar[k] is None
            ]
            if missing:
                bids_keys = {
                    "te": "EchoTime",
                    "tr": "RepetitionTime/RepetitionTimeExcitation",
                    "field_strength": "MagneticFieldStrength",
                }
                raise ValueError(
                    f"DSC BIDS sidecar missing required fields: "
                    f"{[bids_keys[k] for k in missing]}. "
                    f"Source: {session_data.modalities.dsc}"
                )
            dsc_wf.inputs.inputnode.te = sidecar["te"]
            dsc_wf.inputs.inputnode.tr = sidecar["tr"]
            dsc_wf.inputs.inputnode.field_strength = sidecar["field_strength"]

        # Connect DSC file from inputnode
        workflow.connect([
            (inputnode, dsc_wf, [("dsc_file", "inputnode.dsc_file")]),
        ])

        # Set DSC registration reference_type from confidence-scored selection
        if dsc_ref_selection:
            dsc_reg_ref_type = {
                ImageType.T1W_POSTCON: "postcon",
                ImageType.T1W_PRECON: "precon",
            }.get(dsc_ref_selection.candidate.image_type, "any")
            dsc_wf.inputs.register.reference_type = dsc_reg_ref_type
            _ref_logger.info(
                "DSC registration: reference_type=%s (from selection)",
                dsc_reg_ref_type,
            )

        # Connect anat outputs to DSC workflow
        if anat_wf is not None:
            # t1w_reference: always precon (for registration coordinate frame)
            workflow.connect([
                (anat_wf, dsc_wf, [
                    ("outputnode.t1w_preproc", "inputnode.t1w_reference"),
                    ("outputnode.brain_mask", "inputnode.brain_mask"),
                    ("outputnode.t1w_to_mni_xfm", "inputnode.t1w_to_mni_xfm"),
                ]),
            ])
            # WM probseg for BBR registration (unavailable in postcon-only mode)
            if session_data.modalities.t1w_pre:
                workflow.connect([
                    (anat_wf, dsc_wf, [
                        ("outputnode.wm_probseg", "inputnode.wm_probseg"),
                    ]),
                ])
            # T1w underlays — independent precon and postcon paths so users
            # who configure both get both outputs (precon → t1w_underlay,
            # postcon → t1w_postcon_underlay).
            if "t1w_precon" in dsc_config.underlays:
                if session_data.modalities.t1w_pre:
                    workflow.connect([
                        (anat_wf, dsc_wf, [("outputnode.t1w_preproc", "inputnode.t1w_underlay")]),
                    ])
                else:
                    logging.getLogger("pmriprep.workflows").warning(
                        "DSC: 't1w_precon' underlay requested but no pre-contrast T1w available. Skipping."
                    )
            if "t1w_postcon" in dsc_config.underlays:
                if session_data.modalities.t1w_ce_gad and session_data.modalities.t1w_pre:
                    workflow.connect([
                        (anat_wf, dsc_wf, [("outputnode.t1w_postcon_coreg", "inputnode.t1w_postcon_underlay")]),
                    ])
                elif session_data.modalities.t1w_ce_gad:
                    # Postcon-only mode: t1w_preproc IS the postcon
                    workflow.connect([
                        (anat_wf, dsc_wf, [("outputnode.t1w_preproc", "inputnode.t1w_postcon_underlay")]),
                    ])
                else:
                    logging.getLogger("pmriprep.workflows").warning(
                        "DSC: 't1w_postcon' underlay requested but no post-contrast T1w available. Skipping."
                    )
            # FLAIR underlay (prefer precon, fall back to postcon)
            if "flair" in dsc_config.underlays:
                if session_data.modalities.flair:
                    workflow.connect([
                        (anat_wf, dsc_wf, [("outputnode.flair_coreg", "inputnode.flair_in_t1w")]),
                    ])
                elif session_data.modalities.flair_postcon:
                    workflow.connect([
                        (anat_wf, dsc_wf, [("outputnode.flair_postcon_coreg", "inputnode.flair_in_t1w")]),
                    ])
                else:
                    logging.getLogger("pmriprep.workflows").warning(
                        "DSC: 'flair' underlay requested but no FLAIR available. Skipping."
                    )
            # T2w underlay
            if "t2w" in dsc_config.underlays:
                if session_data.modalities.t2w:
                    workflow.connect([
                        (anat_wf, dsc_wf, [("outputnode.t2w_coreg", "inputnode.t2w_in_t1w")]),
                    ])
                else:
                    logging.getLogger("pmriprep.workflows").warning(
                        "DSC: 't2w' underlay requested but no T2w available. Skipping."
                    )
        elif work_dir is not None:
            # --skip-anat: load previously-computed anat outputs from work dir
            import logging as _logging
            _log = _logging.getLogger("pmriprep.workflows")
            prev_anat = _find_previous_anat_outputs(
                Path(work_dir), subject_id, session_id,
            )
            if prev_anat["t1w_preproc"]:
                dsc_wf.inputs.inputnode.t1w_reference = prev_anat["t1w_preproc"]
                _log.info("skip-anat: loaded t1w_reference from %s", prev_anat["t1w_preproc"])
            # T1w underlays — independent precon and postcon paths.
            if "t1w_precon" in dsc_config.underlays:
                if prev_anat["t1w_preproc"] and session_data.modalities.t1w_pre:
                    dsc_wf.inputs.inputnode.t1w_underlay = prev_anat["t1w_preproc"]
                    _log.info("skip-anat (DSC): loaded t1w_underlay (precon) from %s", prev_anat["t1w_preproc"])
                else:
                    _log.warning("DSC: 't1w_precon' underlay requested but unavailable in skip-anat. Skipping.")
            if "t1w_postcon" in dsc_config.underlays:
                if prev_anat["t1w_postcon_coreg"]:
                    dsc_wf.inputs.inputnode.t1w_postcon_underlay = prev_anat["t1w_postcon_coreg"]
                    _log.info("skip-anat (DSC): loaded t1w_postcon_underlay (postcon) from %s", prev_anat["t1w_postcon_coreg"])
                elif prev_anat["t1w_preproc"] and not session_data.modalities.t1w_pre:
                    # Postcon-only mode: t1w_preproc IS the postcon
                    dsc_wf.inputs.inputnode.t1w_postcon_underlay = prev_anat["t1w_preproc"]
                    _log.info("skip-anat (DSC): loaded t1w_postcon_underlay (postcon-only) from %s", prev_anat["t1w_preproc"])
                else:
                    _log.warning("DSC: 't1w_postcon' underlay requested but unavailable in skip-anat. Skipping.")
            if prev_anat["brain_mask"]:
                dsc_wf.inputs.inputnode.brain_mask = prev_anat["brain_mask"]
                _log.info("skip-anat: loaded brain_mask from %s", prev_anat["brain_mask"])
            if prev_anat["wm_probseg"]:
                dsc_wf.inputs.inputnode.wm_probseg = prev_anat["wm_probseg"]
                _log.info("skip-anat: loaded wm_probseg from %s", prev_anat["wm_probseg"])
            if prev_anat["t1w_to_mni_xfm"]:
                dsc_wf.inputs.inputnode.t1w_to_mni_xfm = prev_anat["t1w_to_mni_xfm"]
                _log.info("skip-anat: loaded t1w_to_mni_xfm from %s", prev_anat["t1w_to_mni_xfm"])
            if "flair" in dsc_config.underlays:
                if prev_anat["flair_coreg"]:
                    dsc_wf.inputs.inputnode.flair_in_t1w = prev_anat["flair_coreg"]
                    _log.info("skip-anat: loaded flair_in_t1w from %s", prev_anat["flair_coreg"])
                elif prev_anat.get("flair_postcon_coreg"):
                    dsc_wf.inputs.inputnode.flair_in_t1w = prev_anat["flair_postcon_coreg"]
                    _log.info("skip-anat: loaded flair_in_t1w (postcon) from %s", prev_anat["flair_postcon_coreg"])
            if "t2w" in dsc_config.underlays and prev_anat["t2w_coreg"]:
                dsc_wf.inputs.inputnode.t2w_in_t1w = prev_anat["t2w_coreg"]
                _log.info("skip-anat: loaded t2w_in_t1w from %s", prev_anat["t2w_coreg"])

            if not any(prev_anat.values()):
                _log.warning(
                    "skip-anat: no previous anat outputs found in %s. "
                    "DSC nodes requiring anat inputs will fail.",
                    work_dir,
                )

        # Connect lesion mask and tumor segmentation from tumor seg if available
        # Also wire user-supplied masks from config (lesion_mask, nawm_mask, roi_mask)
        _dsc_log = logging.getLogger("pmriprep.workflows")
        _user_lesion = dsc_config.lesion_mask
        _user_nawm = dsc_config.nawm_mask
        _user_roi = dsc_config.roi_mask

        # Reorient user-supplied masks to RAS (they may have different
        # orientation than pipeline-processed images, causing registration
        # failures or left-right flips)
        from pmriprep.interfaces.common import Reorient

        if _user_roi:
            dsc_roi_reorient = pe.Node(Reorient(), name="dsc_roi_reorient")
            dsc_roi_reorient.inputs.in_file = str(_user_roi)
            workflow.connect([(dsc_roi_reorient, dsc_wf, [("out_file", "inputnode.roi_mask")])])
            _dsc_log.info("DSC: user roi_mask=%s (reoriented to RAS)", _user_roi)

        if tumor_seg_wf is not None:
            # NAWM picker: prefer tumor-seg NAWM, fall back to user-supplied
            dsc_pick_nawm = pe.Node(
                niu.Function(
                    input_names=["tumor_seg_nawm", "user_nawm"],
                    output_names=["nawm_mask"],
                    function=_pick_nawm_mask,
                ),
                name="dsc_pick_nawm",
            )
            if _user_nawm:
                _nawm_reorient = pe.Node(Reorient(), name="dsc_nawm_reorient")
                _nawm_reorient.inputs.in_file = str(_user_nawm)
                workflow.connect([(_nawm_reorient, dsc_pick_nawm, [("out_file", "user_nawm")])])

            if _user_lesion:
                # Merge tumor seg whole_tumor + user lesion masks
                from pmriprep.utils.mask_utils import MergeMasks

                dsc_lesion_reorient = pe.Node(Reorient(), name="dsc_lesion_reorient")
                dsc_lesion_reorient.inputs.in_file = str(_user_lesion)

                dsc_merge_masks = pe.Node(
                    MergeMasks(), name="dsc_merge_masks",
                )

                workflow.connect([
                    (dsc_lesion_reorient, dsc_merge_masks, [
                        ("out_file", "mask_b"),
                    ]),
                    (tumor_seg_wf, dsc_merge_masks, [
                        ("outputnode.whole_tumor_mask", "mask_a"),
                    ]),
                    (dsc_merge_masks, dsc_wf, [
                        ("merged_mask", "inputnode.whole_tumor_mask"),
                    ]),
                    (tumor_seg_wf, dsc_wf, [
                        ("outputnode.enhancing_mask", "inputnode.enhancing_mask"),
                        ("outputnode.segmentation", "inputnode.tumor_segmentation"),
                    ]),
                    (tumor_seg_wf, dsc_pick_nawm, [
                        ("outputnode.nawm_mask", "tumor_seg_nawm"),
                    ]),
                    (dsc_pick_nawm, dsc_wf, [
                        ("nawm_mask", "inputnode.nawm_mask"),
                    ]),
                ])
                _dsc_log.info(
                    "DSC: merging user lesion_mask=%s with tumor segmentation whole_tumor_mask",
                    _user_lesion,
                )
            else:
                # Tumor segmentation only
                workflow.connect([
                    (tumor_seg_wf, dsc_wf, [
                        ("outputnode.whole_tumor_mask", "inputnode.whole_tumor_mask"),
                        ("outputnode.enhancing_mask", "inputnode.enhancing_mask"),
                        ("outputnode.segmentation", "inputnode.tumor_segmentation"),
                    ]),
                    (tumor_seg_wf, dsc_pick_nawm, [
                        ("outputnode.nawm_mask", "tumor_seg_nawm"),
                    ]),
                    (dsc_pick_nawm, dsc_wf, [
                        ("nawm_mask", "inputnode.nawm_mask"),
                    ]),
                ])
        elif work_dir is not None and not decision.run_tumor_seg:
            # --skip-tumor-seg: load previously-computed tumor seg outputs
            prev_tumor_seg = _find_previous_tumor_seg_outputs(
                Path(work_dir), subject_id, session_id,
            )
            if _user_lesion and prev_tumor_seg["whole_tumor_mask"]:
                # Merge user lesion with loaded whole_tumor_mask
                from pmriprep.utils.mask_utils import MergeMasks

                dsc_lesion_reorient2 = pe.Node(Reorient(), name="dsc_lesion_reorient_skip")
                dsc_lesion_reorient2.inputs.in_file = str(_user_lesion)

                dsc_merge_masks = pe.Node(
                    MergeMasks(), name="dsc_merge_masks",
                )
                dsc_merge_masks.inputs.mask_a = prev_tumor_seg["whole_tumor_mask"]

                workflow.connect([
                    (dsc_lesion_reorient2, dsc_merge_masks, [
                        ("out_file", "mask_b"),
                    ]),
                    (dsc_merge_masks, dsc_wf, [
                        ("merged_mask", "inputnode.whole_tumor_mask"),
                    ]),
                ])
                _dsc_log.info(
                    "DSC skip-tumor-seg: merging user lesion_mask=%s (reoriented) with loaded whole_tumor_mask",
                    _user_lesion,
                )
            else:
                if prev_tumor_seg["whole_tumor_mask"]:
                    dsc_wf.inputs.inputnode.whole_tumor_mask = prev_tumor_seg["whole_tumor_mask"]
                    _dsc_log.info("skip-tumor-seg: loaded whole_tumor_mask from %s", prev_tumor_seg["whole_tumor_mask"])

            if prev_tumor_seg["enhancing_mask"]:
                dsc_wf.inputs.inputnode.enhancing_mask = prev_tumor_seg["enhancing_mask"]
                _dsc_log.info("skip-tumor-seg: loaded enhancing_mask from %s", prev_tumor_seg["enhancing_mask"])
            if prev_tumor_seg["segmentation"]:
                dsc_wf.inputs.inputnode.tumor_segmentation = prev_tumor_seg["segmentation"]
                _dsc_log.info("skip-tumor-seg: loaded tumor_segmentation from %s", prev_tumor_seg["segmentation"])
            if prev_tumor_seg["nawm_mask"]:
                dsc_wf.inputs.inputnode.nawm_mask = prev_tumor_seg["nawm_mask"]
                _dsc_log.info("skip-tumor-seg: loaded nawm_mask from %s", prev_tumor_seg["nawm_mask"])
            elif _user_nawm:
                _nawm_reorient2 = pe.Node(Reorient(), name="dsc_nawm_reorient_skip")
                _nawm_reorient2.inputs.in_file = str(_user_nawm)
                workflow.connect([(_nawm_reorient2, dsc_wf, [("out_file", "inputnode.nawm_mask")])])
                _dsc_log.info("skip-tumor-seg: using user nawm_mask=%s (reoriented, no cached nawm)", _user_nawm)
        else:
            # No tumor segmentation — user-supplied masks only
            if _user_lesion:
                _lesion_reorient3 = pe.Node(Reorient(), name="dsc_lesion_reorient_only")
                _lesion_reorient3.inputs.in_file = str(_user_lesion)
                workflow.connect([(_lesion_reorient3, dsc_wf, [("out_file", "inputnode.whole_tumor_mask")])])
                _dsc_log.info("DSC: user lesion_mask=%s → whole_tumor_mask (reoriented)", _user_lesion)

            if _user_nawm:
                _nawm_reorient3 = pe.Node(Reorient(), name="dsc_nawm_reorient_only")
                _nawm_reorient3.inputs.in_file = str(_user_nawm)
                workflow.connect([(_nawm_reorient3, dsc_wf, [("out_file", "inputnode.nawm_mask")])])
                _dsc_log.info("DSC: user nawm_mask=%s (reoriented)", _user_nawm)

        # Connect DSC outputs to master outputnode
        workflow.connect([
            (dsc_wf, outputnode, [
                ("outputnode.rel_cbf_map", "dsc_cbf"),
                ("outputnode.rel_cbv_auc_map", "dsc_cbv"),
                ("outputnode.mtt_map", "dsc_mtt"),
                ("outputnode.tmax_map", "dsc_tmax"),
                ("outputnode.aif_qc", "dsc_aif_qc"),
                ("outputnode.qc_metrics", "dsc_qc"),
            ]),
        ])

        # -------------------------------------------------------------------
        # Build DSC provenance dict for output sidecars
        from pmriprep.utils.contrast_agent import (
            ParameterSource,
            build_dsc_provenance,
        )
        dsc_provenance = build_dsc_provenance(
            contrast_agent=dsc_sidecar.get("contrast_agent") if dsc_sidecar else None,
            agent_source=_dsc_agent_source or ParameterSource.DEFAULT_FALLBACK,
            bids_agent_string=dsc_sidecar.get("bids_agent_string") if dsc_sidecar else None,
            dose_multiplier=dsc_config.dose,
            dose_source=(
                ParameterSource.USER_SPECIFIED
                if _user_spec_dsc.get("dsc_dose")
                else ParameterSource.DEFAULT_FALLBACK
            ),
            agent_standard_dose=dsc_config.agent_standard_dose,
            agent_standard_dose_source=_dsc_agent_dose_source or ParameterSource.DEFAULT_FALLBACK,
            gd_multiplicity=_gd_multiplicity,
            preload_source=(
                ParameterSource.USER_SPECIFIED
                if _user_spec_dsc.get("dsc_preload")
                else ParameterSource.AUTO_DETECTED
            ),
            deconv_method=dsc_config.method,
            inference=dsc_config.inference,
            leakage_method=dsc_config.leakage_method,
            scanner_metadata=dsc_sidecar,
        )

        # DSC fallback logging warnings (parity with DCE warnings)
        _dsc_logger = logging.getLogger("pmriprep.workflows")
        if not _user_spec_dsc.get("dsc_dose"):
            _dsc_logger.warning(
                "DSC contrast dose not specified. Using default=%.1f. "
                "Use --dsc-dose to specify.",
                dsc_config.dose,
            )
        if _dsc_agent_dose_source == ParameterSource.DEFAULT_FALLBACK:
            _dsc_logger.info(
                "DSC agent standard dose not specified. Using default=%.3f mmol/kg. "
                "Use --dsc-agent-standard-dose for non-standard agents "
                "(e.g., 0.05 for Vueway).",
                dsc_config.agent_standard_dose,
            )
        elif _dsc_agent_dose_source == ParameterSource.BIDS_SIDECAR:
            _dsc_logger.info(
                "DSC agent standard dose resolved from BIDS sidecar: %s -> %.3f mmol/kg.",
                dsc_sidecar.get("contrast_agent", "unknown") if dsc_sidecar else "unknown",
                dsc_config.agent_standard_dose,
            )
        if _dsc_agent_source == ParameterSource.BIDS_SIDECAR:
            _dsc_logger.info(
                "DSC contrast agent identified from BIDS sidecar: %s.",
                dsc_sidecar.get("contrast_agent", "unknown") if dsc_sidecar else "unknown",
            )

        # DSC derivatives: niworkflows DerivativesDataSink chain
        # -------------------------------------------------------------------
        from pmriprep.workflows.dsc.datasinks import init_dsc_datasinks_wf

        dsc_underlays = list(dsc_config.underlays)
        # Build the set of inputnode fields that this session will wire,
        # gated to match what dsc_wf.outputnode actually populates. Including
        # an unpopulated key here triggers nipype mandatory-input crashes
        # because the sink's `in_file` becomes Undefined.
        dsc_method = (dsc_config.method or "").lower()
        dsc_inference = (dsc_config.inference or "").lower()
        # Gate on actual NAWM source, not t1w_pre presence. t1w_pre alone
        # doesn't produce NAWM — that needs tumor_seg or a user mask.
        # Without a real source, _transform_nawm_to_dsc returns "" and
        # niworkflows File(exists=True) rejects it at runtime.
        dsc_has_nawm = bool(
            dsc_config.nawm_mask
            or tumor_seg_wf is not None
            or (
                work_dir is not None
                and not decision.run_tumor_seg
                and _find_previous_tumor_seg_outputs(
                    Path(work_dir), subject_id, session_id,
                ).get("nawm_mask")
            )
        )

        dsc_present: set = {
            # Always-present (mandatory in dsc_wf.outputnode)
            "ttp_map", "cnr_map", "auc_diff_map", "snr_map", "tsnr_map",
            "rel_cbf_map", "rel_cbv_map", "rel_cbv_auc_map",
            "rel_auc_raw_map", "rel_auc_corr_map",
            "ncbv_map", "ncbf_map",
            "k2_map",
            "dsc_moco", "dsc_mean", "motion_params",
            "aif_mask",
            "brain_mask_dsc",
            "dsc_to_t1w_xfm",
        }
        # AIF-detection-derived TTP variants
        dsc_present.add("ttp_corrected_map")
        if dsc_has_nawm:
            dsc_present.add("rttp_map")
        # Parametric methods only (oSVD/cSVD/sSVD/tikhonov produce no
        # vascular params, no Bayesian uncertainty, no parametric R²)
        if dsc_method in ("cpi", "vascular"):
            dsc_present |= {"parametric_r2_map", "fallback_mask"}
            dsc_present |= {
                "cbf_std_map", "cbv_std_map", "cbf_cv_map", "cbv_cv_map",
            }
            # CI95 maps only for posterior-sample inference (VI/MCMC/SVB)
            if dsc_inference in ("vi", "svb", "mcmc"):
                dsc_present |= {
                    "cbf_ci95_lower_map", "cbf_ci95_upper_map",
                    "cbv_ci95_lower_map", "cbv_ci95_upper_map",
                }
        # Vascular method only (alpha/beta/cth params)
        if dsc_method == "vascular":
            dsc_present |= {"alpha_map", "beta_map", "cth_map"}

        if dsc_has_nawm:
            dsc_present.add("nawm_mask_dsc")
        if dsc_config.lesion_mask is not None or decision.run_tumor_seg:
            dsc_present.add("lesion_mask_dsc")
        if dsc_config.roi_mask is not None:
            dsc_present.add("roi_mask_dsc")
        # Underlays: only add to present if the modality is actually
        # available in the session. dsc_config.underlays is user intent;
        # session_data.modalities is what's actually wired. Mismatch makes
        # the upstream _transform_structural_to_dsc emit "" and the sink
        # crash on niworkflows File(exists=True).
        _has_t1w_precon_underlay = (
            "t1w_precon" in dsc_underlays
            and session_data.modalities.t1w_pre is not None
        )
        _has_t1w_postcon_underlay = (
            "t1w_postcon" in dsc_underlays
            and session_data.modalities.t1w_ce_gad is not None
        )
        _has_flair_underlay = (
            "flair" in dsc_underlays
            and session_data.modalities.flair is not None
        )
        _has_t2w_underlay = (
            "t2w" in dsc_underlays
            and session_data.modalities.t2w is not None
        )
        if _has_t1w_precon_underlay:
            dsc_present.add("t1w_in_dsc")
        if _has_t1w_postcon_underlay:
            dsc_present.add("t1w_postcon_in_dsc")
        if _has_flair_underlay:
            dsc_present.add("flair_in_dsc")
        if _has_t2w_underlay:
            dsc_present.add("t2w_in_dsc")
        if (
            _has_t1w_precon_underlay or _has_t1w_postcon_underlay
            or _has_flair_underlay or _has_t2w_underlay
        ):
            dsc_present.add("t1w_to_dsc_xfm")

        dsc_datasinks_wf = init_dsc_datasinks_wf(
            output_dir=str(output_dir),
            bids_dir=str(bids_dir),
            subject_id=subject_id,
            session_id=session_id,
            present_inputs=dsc_present,
            has_smoothing=dsc_config.smooth_fwhm > 0,
        )
        if dsc_provenance:
            dsc_datasinks_wf.inputs.inputnode.provenance_dict = dsc_provenance
        if dsc_preflight_warnings:
            dsc_datasinks_wf.inputs.inputnode.preflight_warnings = list(
                dsc_preflight_warnings
            )

        workflow.connect([
            (dsc_wf, dsc_datasinks_wf, [
                # Quantification-independent maps
                ("outputnode.mtt_map", "inputnode.mtt_map"),
                ("outputnode.tmax_map", "inputnode.tmax_map"),
                ("outputnode.psr_map", "inputnode.psr_map"),
                ("outputnode.ph_map", "inputnode.ph_map"),
                ("outputnode.ttp_map", "inputnode.ttp_map"),
                ("outputnode.ttp_corrected_map", "inputnode.ttp_corrected_map"),
                ("outputnode.rttp_map", "inputnode.rttp_map"),
                ("outputnode.cnr_map", "inputnode.cnr_map"),
                ("outputnode.auc_diff_map", "inputnode.auc_diff_map"),
                ("outputnode.snr_map", "inputnode.snr_map"),
                ("outputnode.tsnr_map", "inputnode.tsnr_map"),
                # Relative maps
                ("outputnode.rel_cbf_map", "inputnode.rel_cbf_map"),
                ("outputnode.rel_cbv_map", "inputnode.rel_cbv_map"),
                ("outputnode.rel_cbv_auc_map", "inputnode.rel_cbv_auc_map"),
                ("outputnode.rel_auc_raw_map", "inputnode.rel_auc_raw_map"),
                ("outputnode.rel_auc_corr_map", "inputnode.rel_auc_corr_map"),
                # Normalized maps
                ("outputnode.ncbv_map", "inputnode.ncbv_map"),
                ("outputnode.ncbf_map", "inputnode.ncbf_map"),
                # Vascular model parameters
                ("outputnode.alpha_map", "inputnode.alpha_map"),
                ("outputnode.beta_map", "inputnode.beta_map"),
                ("outputnode.cth_map", "inputnode.cth_map"),
                # Bayesian uncertainty maps
                ("outputnode.cbf_std_map", "inputnode.cbf_std_map"),
                ("outputnode.cbv_std_map", "inputnode.cbv_std_map"),
                ("outputnode.cbf_cv_map", "inputnode.cbf_cv_map"),
                ("outputnode.cbv_cv_map", "inputnode.cbv_cv_map"),
                ("outputnode.cbf_ci95_lower_map", "inputnode.cbf_ci95_lower_map"),
                ("outputnode.cbf_ci95_upper_map", "inputnode.cbf_ci95_upper_map"),
                ("outputnode.cbv_ci95_lower_map", "inputnode.cbv_ci95_lower_map"),
                ("outputnode.cbv_ci95_upper_map", "inputnode.cbv_ci95_upper_map"),
                # Parametric-fit diagnostics (CPI/Vascular only)
                ("outputnode.parametric_r2_map", "inputnode.parametric_r2_map"),
                ("outputnode.fallback_mask", "inputnode.fallback_mask"),
                # Leakage correction
                ("outputnode.k2_map", "inputnode.k2_map"),
                ("outputnode.preload_qc", "inputnode.preload_qc"),
                # Phase 2/3 BIDS cleanup: in-memory dicts feeding the
                # preproc-4D cosmetic enrichment + verbatim resume sidecar.
                ("outputnode.aif_qc", "inputnode.aif_qc"),
                ("outputnode.preprocessing_stats", "inputnode.preprocessing_stats"),
                ("outputnode.motion_qc", "inputnode.motion_qc"),
                ("outputnode.aif_metrics", "inputnode.aif_metrics"),
                # Motion-corrected data
                ("outputnode.dsc_moco", "inputnode.dsc_moco"),
                ("outputnode.dsc_mean", "inputnode.dsc_mean"),
                ("outputnode.motion_params", "inputnode.motion_params"),
                # AIF outputs
                ("outputnode.aif_curve", "inputnode.aif_curve"),
                ("outputnode.aif_mask", "inputnode.aif_mask"),
                # Registration
                ("outputnode.dsc_to_t1w_xfm", "inputnode.dsc_to_t1w_xfm"),
                ("outputnode.registration_filmstrip",
                 "inputnode.registration_filmstrip"),
                # Structural underlays
                ("outputnode.t1w_in_dsc", "inputnode.t1w_in_dsc"),
                ("outputnode.t1w_postcon_in_dsc", "inputnode.t1w_postcon_in_dsc"),
                ("outputnode.flair_in_dsc", "inputnode.flair_in_dsc"),
                ("outputnode.t2w_in_dsc", "inputnode.t2w_in_dsc"),
                ("outputnode.t1w_to_dsc_xfm", "inputnode.t1w_to_dsc_xfm"),
                # AIF QC plots
                ("outputnode.aif_curve_plot", "inputnode.aif_curve_plot"),
                ("outputnode.aif_location_plot", "inputnode.aif_location_plot"),
                ("outputnode.aif_candidates_plot",
                 "inputnode.aif_candidates_plot"),
                # Motion QC plot
                ("outputnode.motion_qc_plot", "inputnode.motion_qc_plot"),
                # QC metrics
                ("outputnode.qc_metrics", "inputnode.qc_metrics"),
                # DSC-space masks
                ("outputnode.brain_mask_dsc", "inputnode.brain_mask_dsc"),
                ("outputnode.nawm_mask_dsc", "inputnode.nawm_mask_dsc"),
                # Lesion statistics
                ("outputnode.lesion_stats_json", "inputnode.lesion_stats_json"),
                ("outputnode.lesion_mask_dsc", "inputnode.lesion_mask_dsc"),
                # ROI statistics
                ("outputnode.roi_stats_json", "inputnode.roi_stats_json"),
                ("outputnode.roi_mask_dsc", "inputnode.roi_mask_dsc"),
                # Brain-mask + DSC QC filmstrips
                ("outputnode.brain_mask_filmstrip",
                 "inputnode.brain_mask_filmstrip"),
                ("outputnode.dsc_preprocessing_qc_plot",
                 "inputnode.dsc_preprocessing_qc_plot"),
                ("outputnode.dsc_param_maps_qc_plot",
                 "inputnode.dsc_param_maps_qc_plot"),
                ("outputnode.dsc_fit_qc_plot", "inputnode.dsc_fit_qc_plot"),
                ("outputnode.dsc_leakage_qc_plot",
                 "inputnode.dsc_leakage_qc_plot"),
            ]),
            (inputnode, dsc_datasinks_wf, [("dsc_file", "inputnode.source_dsc")]),
        ])

        if dsc_config.smooth_fwhm > 0:
            workflow.connect([
                (dsc_wf, dsc_datasinks_wf, [
                    ("outputnode.smooth_hemodynamic",
                     "inputnode.smooth_hemodynamic"),
                    ("outputnode.smooth_auc", "inputnode.smooth_auc"),
                ]),
            ])
            dsc_datasinks_wf.inputs.inputnode.smooth_fwhm = dsc_config.smooth_fwhm

        workflow.connect([
            (dsc_datasinks_wf, report_barrier, [
                ("outputnode.done", "dsc_done"),
            ]),
        ])

    # -------------------------------------------------------------------
    # DCE perfusion workflow (requires anat, uses DSC CBV if available)
    # -------------------------------------------------------------------
    if decision.run_dce:
        import logging as _logging

        _log = _logging.getLogger("pmriprep.workflows")

        # Auto-skip T1 mapping when no VFA data is available
        if not session_data.modalities.vfa and not dce_config.no_t1:
            _log.warning(
                "No VFA data found for sub-%s ses-%s; "
                "disabling T1 mapping (using field-strength default T1)",
                subject_id, session_id,
            )
            dce_config.no_t1 = True
            # Without VFA, AFI-based B1+ is unnecessary (no flip angles to correct).
            # tfl_b1map still works — it directly corrects the DCE flip angle.
            if not session_data.modalities.b1_precomputed_file:
                dce_config.b1_method = "none"

        # Resolve b1_method="auto" to a concrete method based on available data
        b1_method = dce_config.b1_method
        if b1_method == "auto":
            detected = session_data.modalities.b1_method_detected
            if detected in ("afi", "tfl_b1map"):
                b1_method = detected  # type: ignore[assignment]
            elif (
                session_data.modalities.afi
                and session_data.modalities.afi_mode == "4d"
            ):
                b1_method = "afi"
            elif session_data.modalities.afi:
                _log.warning(
                    "AFI files found for sub-%s ses-%s but no 4D TB1AFI "
                    "volume; skipping B1+ correction.",
                    subject_id, session_id,
                )
                b1_method = "none"
            else:
                _log.warning(
                    "No B1+ data (AFI or tfl_b1map) found for sub-%s ses-%s; "
                    "skipping B1+ correction (using nominal flip angles)",
                    subject_id, session_id,
                )
                b1_method = "none"
            dce_config.b1_method = b1_method

        dce_wf = init_dce_preproc_wf(
            bids_dir=bids_dir,
            output_dir=output_dir,
            subject_id=subject_id,
            session_id=session_id,
            config=dce_config,
            has_tissue_seg=bool(session_data.modalities.t1w_pre),
            random_seed=random_seed,
            mem_gb=dce_mem_gb,
            mem_limit_gb=mem_limit_gb,
            name="dce_wf",
        )

        # Connect DCE file from inputnode
        workflow.connect([
            (inputnode, dce_wf, [("dce_file", "inputnode.dce_file")]),
        ])

        # Read acquisition parameters from DCE BIDS sidecar JSON.
        # All five fields are required: TE/pulse TR/FA feed the SPGR
        # signal equation, volume TR sets the time vector for PK fitting,
        # B0 selects field-dependent r1 and T1 references. Silent defaults
        # would mis-quantify Ktrans/ve by orders of magnitude (volume_tr
        # → pulse_tr fallback compresses the time axis ~1000×) or by
        # 5-30% (B0 mismatch on relaxivity).
        dce_provenance: dict[str, Any] = {}
        if session_data.modalities.dce:
            dce_sidecar = _read_bids_sidecar(session_data.modalities.dce)
            missing = [
                k for k in ("te", "tr", "volume_tr", "flip_angle", "field_strength")
                if dce_sidecar[k] is None
            ]
            if missing:
                bids_keys = {
                    "te": "EchoTime",
                    "tr": "RepetitionTimeExcitation (or RepetitionTime for 2D)",
                    "volume_tr": "RepetitionTime",
                    "flip_angle": "FlipAngle",
                    "field_strength": "MagneticFieldStrength",
                }
                raise ValueError(
                    f"DCE BIDS sidecar missing required fields: "
                    f"{[bids_keys[k] for k in missing]}. "
                    f"Source: {session_data.modalities.dce}"
                )
            dce_wf.inputs.inputnode.te = dce_sidecar["te"]
            dce_wf.inputs.inputnode.tr = dce_sidecar["tr"]
            dce_wf.inputs.inputnode.volume_tr = dce_sidecar["volume_tr"]
            dce_wf.inputs.inputnode.flip_angle = dce_sidecar["flip_angle"]
            dce_wf.inputs.inputnode.field_strength = dce_sidecar["field_strength"]
            dce_wf.inputs.inputnode.mr_acquisition_type = (
                dce_sidecar.get("mr_acquisition_type", "")
            )

            # Set r1 relaxivity — priority chain:
            # 1. User-supplied override (dce_config.r1_relaxivity)
            # 2. Agent-specific lookup from BIDS sidecar
            # 3. Generic Gd-DTPA for known field strength
            # 4. Hardcoded Gd-DTPA at 3T fallback (3.8 s^-1/mM)
            from pmriprep.utils.contrast_agent import (
                ParameterSource,
                build_dce_provenance,
            )

            logger = logging.getLogger("pmriprep.workflows")
            _resolved_r1: float = 3.8
            _resolved_fs: float = (
                dce_sidecar["field_strength"]
                if dce_sidecar["field_strength"] is not None
                else 3.0
            )
            _agent_source = ParameterSource.DEFAULT_FALLBACK
            _agent_name: Optional[str] = None
            _bids_agent_string: Optional[str] = None

            if dce_config.r1_relaxivity is not None:
                _resolved_r1 = dce_config.r1_relaxivity
                _agent_source = ParameterSource.USER_OVERRIDE
                dce_wf.inputs.inputnode.r1_relaxivity = dce_config.r1_relaxivity
                dce_wf.inputs.inputnode.agent_r1_known = True
                logger.info(
                    f"Using user-supplied r1={dce_config.r1_relaxivity:.1f} "
                    f"s^-1/mM (overrides sidecar lookup)"
                )
            elif dce_sidecar["r1_relaxivity"] is not None:
                _resolved_r1 = dce_sidecar["r1_relaxivity"]
                _agent_source = ParameterSource.BIDS_SIDECAR
                _agent_name = dce_sidecar.get("contrast_agent")
                _bids_agent_string = dce_sidecar.get("bids_agent_string")
                dce_wf.inputs.inputnode.r1_relaxivity = dce_sidecar["r1_relaxivity"]
                dce_wf.inputs.inputnode.agent_r1_known = True
                logger.info(
                    f"Using r1={dce_sidecar['r1_relaxivity']:.1f} s^-1/mM "
                    f"for contrast agent: {dce_sidecar['contrast_agent']}"
                )
            elif dce_sidecar["field_strength"] is not None:
                # Fallback to generic Gd-DTPA r1 for field strength
                from pmriprep.utils.dce_constants import get_r1_relaxivity
                fallback_r1 = get_r1_relaxivity(dce_sidecar["field_strength"])
                _resolved_r1 = fallback_r1
                dce_wf.inputs.inputnode.r1_relaxivity = fallback_r1
                logger.warning(
                    "ContrastBolusAgent not found in BIDS JSON or unknown agent. "
                    f"Using default Gd-DTPA r1={fallback_r1:.1f} s^-1/mM "
                    f"at {dce_sidecar['field_strength']:.1f}T. "
                    "Provide ContrastBolusAgent in JSON sidecar for "
                    "agent-specific accuracy."
                )
            else:
                # Default to 3T value
                from pmriprep.utils.dce_constants import get_r1_relaxivity
                _resolved_r1 = get_r1_relaxivity(3.0)
                dce_wf.inputs.inputnode.r1_relaxivity = _resolved_r1
                logger.warning(
                    "ContrastBolusAgent and field strength not found in BIDS JSON. "
                    "Using default Gd-DTPA r1=3.8 s^-1/mM at 3.0T. "
                    "Provide ContrastBolusAgent in JSON sidecar for "
                    "agent-specific accuracy."
                )

            # Set agent-specific standard dose (e.g. 0.05 for Vueway)
            dce_wf.inputs.inputnode.agent_standard_dose = dce_sidecar["agent_standard_dose"]

            # Build DCE provenance dict for output sidecars
            _user_spec_dce = user_specified or {}
            _dce_dose_source = (
                ParameterSource.USER_SPECIFIED
                if _user_spec_dce.get("dce_dose")
                else ParameterSource.DEFAULT_FALLBACK
            )
            dce_provenance = build_dce_provenance(
                contrast_agent=_agent_name,
                agent_source=_agent_source,
                r1_relaxivity=_resolved_r1,
                field_strength=_resolved_fs,
                dose_multiplier=dce_config.dose,
                dose_source=_dce_dose_source,
                agent_standard_dose=dce_sidecar["agent_standard_dose"],
                bids_agent_string=_bids_agent_string,
                scanner_metadata=dce_sidecar,
            )

        # Determine DCE structural reference from confidence-scored selection
        dce_ref_output = "outputnode.t1w_preproc"  # default
        dce_ref_is_postcon = False
        if dce_ref_selection:
            dce_ref_output = _map_selection_to_anat_output(
                dce_ref_selection.candidate.image_type,
            )
            dce_ref_is_postcon = (
                dce_ref_selection.candidate.image_type == ImageType.T1W_POSTCON
            )
            # Postcon-only override: t1w_postcon_coreg requires both pre and postcon
            if dce_ref_is_postcon and not session_data.modalities.t1w_pre:
                dce_ref_output = "outputnode.t1w_preproc"

            # Override DCE registration reference type from selection
            ref_type_map = {
                ImageType.T1W_POSTCON: "postcon",
                ImageType.T1W_PRECON: "precon",
            }
            dce_reg_ref_type = ref_type_map.get(
                dce_ref_selection.candidate.image_type, "any",
            )
            dce_wf.inputs.dce_registration.reference_type = dce_reg_ref_type
            _ref_logger.info(
                "DCE registration: reference_type=%s (from selection)",
                dce_reg_ref_type,
            )

        # Connect anat outputs to DCE workflow
        if anat_wf is not None:
            # Use confidence-scored reference for DCE registration
            # IMPORTANT: Always use anat pipeline's preprocessed outputs (RAS-reoriented,
            # bias-corrected) rather than raw BIDS inputs for proper orientation alignment
            workflow.connect([
                (anat_wf, dce_wf, [(dce_ref_output, "inputnode.t1w_reference")]),
            ])
            workflow.connect([
                (anat_wf, dce_wf, [
                    ("outputnode.brain_mask", "inputnode.brain_mask"),
                    # Pass T1w→MNI transform for atlas-guided AIF (Gap 6)
                    ("outputnode.t1w_to_mni_xfm", "inputnode.t1w_to_mni_xfm"),
                ]),
            ])
            # Tissue probseg maps (unavailable in postcon-only mode)
            if session_data.modalities.t1w_pre:
                workflow.connect([
                    (anat_wf, dce_wf, [
                        ("outputnode.wm_probseg", "inputnode.wm_probseg"),
                        ("outputnode.gm_probseg", "inputnode.gm_probseg"),
                        ("outputnode.csf_probseg", "inputnode.csf_probseg"),
                    ]),
                ])
            # T1w underlays — independent precon and postcon paths so users
            # who configure both get both outputs (precon → t1w_underlay,
            # postcon → t1w_postcon_underlay).
            if "t1w_precon" in dce_config.underlays:
                if session_data.modalities.t1w_pre:
                    workflow.connect([
                        (anat_wf, dce_wf, [("outputnode.t1w_preproc", "inputnode.t1w_underlay")]),
                    ])
                else:
                    logging.getLogger("pmriprep.workflows").warning(
                        "DCE: 't1w_precon' underlay requested but no pre-contrast T1w available. Skipping."
                    )
            if "t1w_postcon" in dce_config.underlays:
                if session_data.modalities.t1w_ce_gad and session_data.modalities.t1w_pre:
                    workflow.connect([
                        (anat_wf, dce_wf, [("outputnode.t1w_postcon_coreg", "inputnode.t1w_postcon_underlay")]),
                    ])
                elif session_data.modalities.t1w_ce_gad:
                    # Postcon-only mode: t1w_preproc IS the postcon
                    workflow.connect([
                        (anat_wf, dce_wf, [("outputnode.t1w_preproc", "inputnode.t1w_postcon_underlay")]),
                    ])
                else:
                    logging.getLogger("pmriprep.workflows").warning(
                        "DCE: 't1w_postcon' underlay requested but no post-contrast T1w available. Skipping."
                    )
            # FLAIR underlay (prefer precon, fall back to postcon)
            if "flair" in dce_config.underlays:
                if session_data.modalities.flair:
                    workflow.connect([
                        (anat_wf, dce_wf, [("outputnode.flair_coreg", "inputnode.flair_in_t1w")]),
                    ])
                elif session_data.modalities.flair_postcon:
                    workflow.connect([
                        (anat_wf, dce_wf, [("outputnode.flair_postcon_coreg", "inputnode.flair_in_t1w")]),
                    ])
                else:
                    logging.getLogger("pmriprep.workflows").warning(
                        "DCE: 'flair' underlay requested but no FLAIR available. Skipping."
                    )
            # T2w underlay
            if "t2w" in dce_config.underlays:
                if session_data.modalities.t2w:
                    workflow.connect([
                        (anat_wf, dce_wf, [("outputnode.t2w_coreg", "inputnode.t2w_in_t1w")]),
                    ])
                else:
                    logging.getLogger("pmriprep.workflows").warning(
                        "DCE: 't2w' underlay requested but no T2w available. Skipping."
                    )
        elif work_dir is not None:
            # --skip-anat: load previously-computed anat outputs from work dir
            import logging as _logging
            _log = _logging.getLogger("pmriprep.workflows")
            prev_anat = _find_previous_anat_outputs(
                Path(work_dir), subject_id, session_id,
            )
            # Use confidence-scored selection when skip-anat
            if dce_ref_selection and dce_ref_is_postcon and prev_anat["t1w_postcon_coreg"]:
                dce_wf.inputs.inputnode.t1w_reference = prev_anat["t1w_postcon_coreg"]
                _log.info("skip-anat (DCE): loaded t1w_reference from %s", prev_anat["t1w_postcon_coreg"])
            elif session_data.modalities.t1w_ce_gad and prev_anat["t1w_postcon_coreg"]:
                # Fallback: use postcon if available
                dce_wf.inputs.inputnode.t1w_reference = prev_anat["t1w_postcon_coreg"]
                _log.info("skip-anat (DCE): loaded t1w_reference from %s", prev_anat["t1w_postcon_coreg"])
            elif prev_anat["t1w_preproc"]:
                dce_wf.inputs.inputnode.t1w_reference = prev_anat["t1w_preproc"]
                _log.info("skip-anat (DCE): loaded t1w_reference from %s", prev_anat["t1w_preproc"])
            if prev_anat["brain_mask"]:
                dce_wf.inputs.inputnode.brain_mask = prev_anat["brain_mask"]
            if prev_anat["wm_probseg"]:
                dce_wf.inputs.inputnode.wm_probseg = prev_anat["wm_probseg"]
            if prev_anat["gm_probseg"]:
                dce_wf.inputs.inputnode.gm_probseg = prev_anat["gm_probseg"]
            if prev_anat["csf_probseg"]:
                dce_wf.inputs.inputnode.csf_probseg = prev_anat["csf_probseg"]
            if prev_anat["t1w_to_mni_xfm"]:
                dce_wf.inputs.inputnode.t1w_to_mni_xfm = prev_anat["t1w_to_mni_xfm"]
                _log.info("skip-anat (DCE): loaded t1w_to_mni_xfm from %s", prev_anat["t1w_to_mni_xfm"])
            # T1w underlays — independent precon and postcon paths.
            if "t1w_precon" in dce_config.underlays:
                if prev_anat["t1w_preproc"] and session_data.modalities.t1w_pre:
                    dce_wf.inputs.inputnode.t1w_underlay = prev_anat["t1w_preproc"]
                    _log.info("skip-anat (DCE): loaded t1w_underlay (precon) from %s", prev_anat["t1w_preproc"])
                else:
                    _log.warning("DCE: 't1w_precon' underlay requested but unavailable in skip-anat. Skipping.")
            if "t1w_postcon" in dce_config.underlays:
                if prev_anat["t1w_postcon_coreg"]:
                    dce_wf.inputs.inputnode.t1w_postcon_underlay = prev_anat["t1w_postcon_coreg"]
                    _log.info("skip-anat (DCE): loaded t1w_postcon_underlay (postcon) from %s", prev_anat["t1w_postcon_coreg"])
                elif prev_anat["t1w_preproc"] and not session_data.modalities.t1w_pre:
                    dce_wf.inputs.inputnode.t1w_postcon_underlay = prev_anat["t1w_preproc"]
                    _log.info("skip-anat (DCE): loaded t1w_postcon_underlay (postcon-only) from %s", prev_anat["t1w_preproc"])
                else:
                    _log.warning("DCE: 't1w_postcon' underlay requested but unavailable in skip-anat. Skipping.")
            if "flair" in dce_config.underlays:
                if prev_anat["flair_coreg"]:
                    dce_wf.inputs.inputnode.flair_in_t1w = prev_anat["flair_coreg"]
                    _log.info("skip-anat (DCE): loaded flair_in_t1w from %s", prev_anat["flair_coreg"])
                elif prev_anat.get("flair_postcon_coreg"):
                    dce_wf.inputs.inputnode.flair_in_t1w = prev_anat["flair_postcon_coreg"]
                    _log.info("skip-anat (DCE): loaded flair_in_t1w (postcon) from %s", prev_anat["flair_postcon_coreg"])
            if "t2w" in dce_config.underlays and prev_anat["t2w_coreg"]:
                dce_wf.inputs.inputnode.t2w_in_t1w = prev_anat["t2w_coreg"]
                _log.info("skip-anat (DCE): loaded t2w_in_t1w from %s", prev_anat["t2w_coreg"])

        # Connect lesion mask, tumor segmentation, and NAWM mask from tumor seg if available
        # Also wire user-supplied masks from config (lesion_mask, nawm_mask, roi_mask)
        _dce_log = logging.getLogger("pmriprep.workflows")
        _dce_user_lesion = dce_config.lesion_mask
        _dce_user_nawm = dce_config.nawm_mask
        _dce_user_roi = dce_config.roi_mask

        if _dce_user_roi:
            dce_roi_reorient = pe.Node(Reorient(), name="dce_roi_reorient")
            dce_roi_reorient.inputs.in_file = str(_dce_user_roi)
            workflow.connect([(dce_roi_reorient, dce_wf, [("out_file", "inputnode.roi_mask")])])
            _dce_log.info("DCE: user roi_mask=%s (reoriented to RAS)", _dce_user_roi)

        if tumor_seg_wf is not None:
            # NAWM picker: prefer tumor-seg NAWM, fall back to user-supplied
            dce_pick_nawm = pe.Node(
                niu.Function(
                    input_names=["tumor_seg_nawm", "user_nawm"],
                    output_names=["nawm_mask"],
                    function=_pick_nawm_mask,
                ),
                name="dce_pick_nawm",
            )
            if _dce_user_nawm:
                _dce_nawm_reorient = pe.Node(Reorient(), name="dce_nawm_reorient")
                _dce_nawm_reorient.inputs.in_file = str(_dce_user_nawm)
                workflow.connect([(_dce_nawm_reorient, dce_pick_nawm, [("out_file", "user_nawm")])])

            if _dce_user_lesion:
                # Merge tumor seg whole_tumor + user lesion masks
                from pmriprep.utils.mask_utils import MergeMasks

                dce_lesion_reorient = pe.Node(Reorient(), name="dce_lesion_reorient")
                dce_lesion_reorient.inputs.in_file = str(_dce_user_lesion)

                dce_merge_masks = pe.Node(
                    MergeMasks(), name="dce_merge_masks",
                )

                workflow.connect([
                    (dce_lesion_reorient, dce_merge_masks, [
                        ("out_file", "mask_b"),
                    ]),
                    (tumor_seg_wf, dce_merge_masks, [
                        ("outputnode.whole_tumor_mask", "mask_a"),
                    ]),
                    (dce_merge_masks, dce_wf, [
                        ("merged_mask", "inputnode.whole_tumor_mask"),
                    ]),
                    (tumor_seg_wf, dce_wf, [
                        ("outputnode.enhancing_mask", "inputnode.enhancing_mask"),
                        ("outputnode.non_enhancing_mask", "inputnode.non_enhancing_mask"),
                        ("outputnode.segmentation", "inputnode.tumor_segmentation"),
                    ]),
                    (tumor_seg_wf, dce_pick_nawm, [
                        ("outputnode.nawm_mask", "tumor_seg_nawm"),
                    ]),
                    (dce_pick_nawm, dce_wf, [
                        ("nawm_mask", "inputnode.nawm_mask"),
                    ]),
                    # Wire user lesion separately for QC bars (distinct from tumor seg masks)
                    (dce_lesion_reorient, dce_wf, [
                        ("out_file", "inputnode.user_lesion_mask"),
                    ]),
                ])
                _dce_log.info(
                    "DCE: merging user lesion_mask=%s (reoriented) with tumor segmentation whole_tumor_mask",
                    _dce_user_lesion,
                )
            else:
                # Tumor segmentation only
                workflow.connect([
                    (tumor_seg_wf, dce_wf, [
                        ("outputnode.whole_tumor_mask", "inputnode.whole_tumor_mask"),
                        ("outputnode.enhancing_mask", "inputnode.enhancing_mask"),
                        ("outputnode.non_enhancing_mask", "inputnode.non_enhancing_mask"),
                        ("outputnode.segmentation", "inputnode.tumor_segmentation"),
                    ]),
                    (tumor_seg_wf, dce_pick_nawm, [
                        ("outputnode.nawm_mask", "tumor_seg_nawm"),
                    ]),
                    (dce_pick_nawm, dce_wf, [
                        ("nawm_mask", "inputnode.nawm_mask"),
                    ]),
                ])
        elif work_dir is not None and not decision.run_tumor_seg:
            # --skip-tumor-seg: load previously-computed tumor seg outputs
            prev_tumor_seg = _find_previous_tumor_seg_outputs(
                Path(work_dir), subject_id, session_id,
            )
            if _dce_user_lesion:
                # Reorient user lesion mask and wire for QC bars
                dce_lesion_reorient2 = pe.Node(Reorient(), name="dce_lesion_reorient_skip")
                dce_lesion_reorient2.inputs.in_file = str(_dce_user_lesion)
                workflow.connect([
                    (dce_lesion_reorient2, dce_wf, [
                        ("out_file", "inputnode.user_lesion_mask"),
                    ]),
                ])

                if prev_tumor_seg["whole_tumor_mask"]:
                    # Merge user lesion with loaded whole_tumor_mask
                    from pmriprep.utils.mask_utils import MergeMasks

                    dce_merge_masks = pe.Node(
                        MergeMasks(), name="dce_merge_masks",
                    )
                    dce_merge_masks.inputs.mask_a = prev_tumor_seg["whole_tumor_mask"]

                    workflow.connect([
                        (dce_lesion_reorient2, dce_merge_masks, [
                            ("out_file", "mask_b"),
                        ]),
                        (dce_merge_masks, dce_wf, [
                            ("merged_mask", "inputnode.whole_tumor_mask"),
                        ]),
                    ])
                    _dce_log.info(
                        "DCE skip-tumor-seg: merging user lesion_mask=%s (reoriented) with loaded whole_tumor_mask",
                        _dce_user_lesion,
                    )
                else:
                    # No cached whole_tumor_mask — user lesion IS the whole_tumor_mask
                    workflow.connect([
                        (dce_lesion_reorient2, dce_wf, [
                            ("out_file", "inputnode.whole_tumor_mask"),
                        ]),
                    ])
                    _dce_log.info(
                        "DCE skip-tumor-seg: user lesion_mask=%s → whole_tumor_mask (no cached mask)",
                        _dce_user_lesion,
                    )
            else:
                if prev_tumor_seg["whole_tumor_mask"]:
                    dce_wf.inputs.inputnode.whole_tumor_mask = prev_tumor_seg["whole_tumor_mask"]
                    _dce_log.info("skip-tumor-seg (DCE): loaded whole_tumor_mask from %s", prev_tumor_seg["whole_tumor_mask"])

            if prev_tumor_seg.get("enhancing_mask"):
                dce_wf.inputs.inputnode.enhancing_mask = prev_tumor_seg["enhancing_mask"]
                _dce_log.info("skip-tumor-seg (DCE): loaded enhancing_mask from %s", prev_tumor_seg["enhancing_mask"])
            if prev_tumor_seg.get("non_enhancing_mask"):
                dce_wf.inputs.inputnode.non_enhancing_mask = prev_tumor_seg["non_enhancing_mask"]
                _dce_log.info("skip-tumor-seg (DCE): loaded non_enhancing_mask from %s", prev_tumor_seg["non_enhancing_mask"])
            if prev_tumor_seg["segmentation"]:
                dce_wf.inputs.inputnode.tumor_segmentation = prev_tumor_seg["segmentation"]
                _dce_log.info("skip-tumor-seg (DCE): loaded tumor_segmentation from %s", prev_tumor_seg["segmentation"])
            if prev_tumor_seg["nawm_mask"]:
                dce_wf.inputs.inputnode.nawm_mask = prev_tumor_seg["nawm_mask"]
                _dce_log.info("skip-tumor-seg (DCE): loaded nawm_mask from %s", prev_tumor_seg["nawm_mask"])
            elif _dce_user_nawm:
                _dce_nawm_reorient2 = pe.Node(Reorient(), name="dce_nawm_reorient_skip")
                _dce_nawm_reorient2.inputs.in_file = str(_dce_user_nawm)
                workflow.connect([(_dce_nawm_reorient2, dce_wf, [("out_file", "inputnode.nawm_mask")])])
                _dce_log.info("skip-tumor-seg (DCE): using user nawm_mask=%s (reoriented, no cached nawm)", _dce_user_nawm)
        else:
            # No tumor segmentation — user-supplied masks only
            if _dce_user_lesion:
                _dce_lesion_reorient3 = pe.Node(Reorient(), name="dce_lesion_reorient_only")
                _dce_lesion_reorient3.inputs.in_file = str(_dce_user_lesion)
                workflow.connect([
                    (_dce_lesion_reorient3, dce_wf, [
                        ("out_file", "inputnode.whole_tumor_mask"),
                        ("out_file", "inputnode.user_lesion_mask"),
                    ]),
                ])
                _dce_log.info("DCE: user lesion_mask=%s → whole_tumor_mask + user_lesion_mask (reoriented)", _dce_user_lesion)

            if _dce_user_nawm:
                _dce_nawm_reorient3 = pe.Node(Reorient(), name="dce_nawm_reorient_only")
                _dce_nawm_reorient3.inputs.in_file = str(_dce_user_nawm)
                workflow.connect([(_dce_nawm_reorient3, dce_wf, [("out_file", "inputnode.nawm_mask")])])
                _dce_log.info("DCE: user nawm_mask=%s (reoriented)", _dce_user_nawm)

        # Connect VFA and AFI inputs if available
        if session_data.modalities.vfa:
            import logging as _logging

            _log = _logging.getLogger("pmriprep.workflows")
            vfa_list = session_data.modalities.vfa

            workflow.connect([
                (inputnode, dce_wf, [("vfa_files", "inputnode.vfa_files")]),
            ])

            # Read flip angles and TR from VFA BIDS sidecar JSONs
            vfa_flip_angles: list[float] = []
            vfa_tr_ms: Optional[float] = None
            for vfa_path in vfa_list:
                vfa_sidecar = _read_bids_sidecar(vfa_path)
                if vfa_sidecar["flip_angle"] is not None:
                    fa = _validate_flip_angle(
                        vfa_sidecar["flip_angle"],
                        label=str(Path(str(vfa_path)).name),
                    )
                    vfa_flip_angles.append(fa)
                # VFA TR should be identical across all flip angles; take from first
                if vfa_tr_ms is None and vfa_sidecar["tr"] is not None:
                    vfa_tr_ms = _ensure_tr_ms(vfa_sidecar["tr"], label="VFA TR")

            if vfa_flip_angles:
                if len(vfa_flip_angles) != len(vfa_list):
                    _log.warning(
                        "Found FlipAngle in %d of %d VFA sidecars; "
                        "T1 mapping may fail",
                        len(vfa_flip_angles),
                        len(vfa_list),
                    )
                dce_wf.inputs.inputnode.vfa_flip_angles = vfa_flip_angles
            else:
                _log.warning(
                    "No FlipAngle found in VFA sidecar JSONs; "
                    "T1 mapping will fail unless flip angles are configured manually"
                )

            if vfa_tr_ms is not None:
                dce_wf.inputs.inputnode.vfa_tr = vfa_tr_ms
            else:
                _log.warning(
                    "No RepetitionTime found in VFA sidecar JSONs; "
                    "T1 mapping will fail unless TR is configured manually"
                )
        # Wire B1+ data based on resolved method
        b1_method = dce_config.b1_method
        if b1_method == "tfl_b1map" and session_data.modalities.b1_precomputed_file:
            dce_wf.inputs.inputnode.precomputed_b1_file = str(
                session_data.modalities.b1_precomputed_file
            )
            if session_data.modalities.b1_nominal_prep_fa is not None:
                dce_wf.inputs.inputnode.b1_nominal_prep_fa = (
                    session_data.modalities.b1_nominal_prep_fa
                )
        elif b1_method == "afi" and session_data.modalities.afi:
            # Only 4D AFI is supported.
            if (
                session_data.modalities.afi_mode == "4d"
                and session_data.modalities.afi_4d_file
            ):
                dce_wf.inputs.inputnode.afi_4d_file = str(
                    session_data.modalities.afi_4d_file
                )
                # Pass TR overrides from discovery if available
                if session_data.modalities.afi_tr1_ms is not None:
                    dce_wf.inputs.inputnode.afi_tr1 = session_data.modalities.afi_tr1_ms
                if session_data.modalities.afi_tr2_ms is not None:
                    dce_wf.inputs.inputnode.afi_tr2 = session_data.modalities.afi_tr2_ms

            # Wire AFI nominal flip angle from BIDS sidecar (default 60 in B1Mapping)
            if session_data.modalities.afi_flip_angle is not None:
                dce_wf.inputs.inputnode.afi_flip_angle = session_data.modalities.afi_flip_angle

        # Connect DWI for tortuosity if available
        if session_data.modalities.dwi:
            workflow.connect([
                (inputnode, dce_wf, [("dwi_file", "inputnode.dwi_file")]),
            ])
            # Pass full b-value array from BIDS .bval sidecar for tortuosity
            import logging as _logging
            import numpy as _np
            _log = _logging.getLogger("pmriprep.workflows")
            dwi_path = session_data.modalities.dwi
            bval_path = str(dwi_path).replace(".nii.gz", ".bval").replace(".nii", ".bval")
            if Path(bval_path).exists():
                try:
                    bvals = _np.atleast_1d(_np.loadtxt(bval_path))
                    dce_wf.inputs.inputnode.dwi_bvals = bvals.tolist()
                    _log.info(
                        "DWI: loaded %d b-values from %s (range %.0f-%.0f s/mm²)",
                        len(bvals), bval_path, bvals.min(), bvals.max(),
                    )
                except Exception as exc:
                    _log.warning("DWI: failed to read b-values from %s: %s", bval_path, exc)

            # Read b-vectors for DTI direction averaging
            bvec_path = str(dwi_path).replace(".nii.gz", ".bvec").replace(".nii", ".bvec")
            if Path(bvec_path).exists():
                try:
                    bvecs = _np.loadtxt(bvec_path)
                    # Normalise (3, N) -> (N, 3) for consistent passing
                    if bvecs.ndim == 2 and bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
                        bvecs = bvecs.T
                    dce_wf.inputs.inputnode.dwi_bvecs = bvecs.tolist()
                    _log.info(
                        "DWI: loaded %d b-vectors from %s",
                        bvecs.shape[0], bvec_path,
                    )
                except Exception as exc:
                    _log.warning("DWI: failed to read b-vectors from %s: %s", bvec_path, exc)

        # DSC dose (bolus multiplier + agent standard dose) flows into the
        # DCE constraint quality gate so CNR coverage thresholds scale with
        # administered Gd fraction. Read from dsc_config; defaults
        # (1.0, 0.1) apply when DSC is absent.
        if dsc_config is not None:
            dce_wf.inputs.inputnode.dsc_dose = float(dsc_config.dose)
            dce_wf.inputs.inputnode.dsc_agent_standard_dose = float(
                dsc_config.agent_standard_dose
            )

        # Connect DSC CBV for vp constraint in DCE PK modeling.
        # Models with vp (extended_tofts, 2cxm) use DSC CBV as a fitting constraint.
        # Other models with vp (patlak) fit freely, cross-validate post-hoc.
        if dsc_wf is not None:
            # Fitting constraint: models with vp parameter (or auto, which may resolve to one)
            if dce_config.model in ("extended_tofts", "2cxm", "auto"):
                workflow.connect([
                    (dsc_wf, dce_wf, [
                        # Relative/absolute CBV → cross-validation + QC
                        # resampling path (DSC 128x128 → DCE 256x256).
                        ("outputnode.rel_cbv_auc_map", "inputnode.dsc_cbv_file"),
                        # NAWM-normalized CBV → vp prior path. None propagates
                        # cleanly when no NAWM mask exists (missing-NAWM
                        # handling lives in prepare_vp_constraint_from_data).
                        ("outputnode.ncbv_map", "inputnode.dsc_ncbv_file"),
                        ("outputnode.dsc_to_t1w_xfm", "inputnode.dsc_to_t1w_xfm"),
                        # AIF QC is a JSON file, no spatial resampling needed
                        ("outputnode.aif_qc", "pk_modeling.dsc_aif_qc"),
                        ("outputnode.preload_qc", "pk_modeling.dsc_preload_qc"),
                        # CNR coverage for CBV quality assessment
                        ("outputnode.preprocessing_stats", "pk_modeling.dsc_preprocessing_stats"),
                    ]),
                ])

            # Cross-validation: Patlak fits vp freely and cross-validates post-hoc.
            # CBV is resampled inside DCE workflow via resample_dsc_cbv node.
            # Only need to connect if not already connected by the constraint block above.
            if dce_config.model == "patlak":
                workflow.connect([
                    (dsc_wf, dce_wf, [
                        ("outputnode.rel_cbv_auc_map", "inputnode.dsc_cbv_file"),
                        ("outputnode.ncbv_map", "inputnode.dsc_ncbv_file"),
                        ("outputnode.dsc_to_t1w_xfm", "inputnode.dsc_to_t1w_xfm"),
                    ]),
                ])

            # Pass DSC preload config to validation node.
            # True when user explicitly overrode auto-detection with a
            # numeric preload dose (any float fraction including 0.0).
            # `dsc_config.preload` is "auto" or a float; `!= "auto"` matches
            # every float value, so this comparison is the correct
            # "user-specified?" check under the new type.
            dce_wf.inputs.dsc_dce_validation.dsc_preload_configured = (
                dsc_config.preload != "auto"
            )

        elif dce_config.model in ("extended_tofts", "2cxm", "auto", "patlak"):
            # DSC skipped but DCE needs CBV: look for previous DSC outputs
            prev_dsc = _find_previous_dsc_outputs(
                Path(output_dir), subject_id, session_id,
            )
            if prev_dsc["rel_cbv_map"] and prev_dsc["dsc_to_t1w_xfm"]:
                dce_wf.inputs.inputnode.dsc_cbv_file = prev_dsc["rel_cbv_map"]
                dce_wf.inputs.inputnode.dsc_to_t1w_xfm = prev_dsc["dsc_to_t1w_xfm"]
                if prev_dsc.get("ncbv_map"):
                    dce_wf.inputs.inputnode.dsc_ncbv_file = prev_dsc["ncbv_map"]
                    _log.info(
                        "skip-dsc: loaded DSC nCBV from %s", prev_dsc["ncbv_map"]
                    )
                else:
                    _log.warning(
                        "skip-dsc: nCBV not found in previous DSC outputs "
                        "(older run or no NAWM mask); vp constraint will "
                        "cleanly SKIP. Use --no-dsc-validation to avoid "
                        "cross-validation noise if this is unexpected."
                    )
                _log.info(
                    "skip-dsc: loaded DSC CBV from %s", prev_dsc["rel_cbv_map"]
                )
                if prev_dsc["aif_qc"]:
                    dce_wf.inputs.pk_modeling.dsc_aif_qc = prev_dsc["aif_qc"]
                if prev_dsc.get("preload_qc"):
                    dce_wf.inputs.pk_modeling.dsc_preload_qc = prev_dsc["preload_qc"]
                if prev_dsc.get("preprocessing_stats"):
                    dce_wf.inputs.pk_modeling.dsc_preprocessing_stats = prev_dsc["preprocessing_stats"]
                if not prev_dsc["aif_qc"]:
                    _log.warning(
                        "skip-dsc: AIF QC not found in previous DSC outputs; "
                        "constraint quality assessment will use defaults"
                    )
            else:
                _log.warning(
                    "skip-dsc: no previous DSC CBV/transform found in %s; "
                    "DCE vp constraint will be skipped",
                    output_dir,
                )

        # Connect DCE outputs to master outputnode
        workflow.connect([
            (dce_wf, outputnode, [
                ("outputnode.ktrans_map", "dce_ktrans"),
                ("outputnode.ve_map", "dce_ve"),
                ("outputnode.vp_map", "dce_vp"),
                ("outputnode.kep_map", "dce_kep"),
                ("outputnode.qc_metrics", "dce_qc"),
            ]),
        ])

        # -------------------------------------------------------------------
        # DCE derivatives: niworkflows DerivativesDataSink chain
        # -------------------------------------------------------------------
        from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

        dce_underlays = list(dce_config.underlays)
        dce_inference = (dce_config.inference or "").lower()
        dce_uses_bayes = dce_inference != "nlls"
        dce_model = dce_config.model or "auto"
        dce_has_nawm = bool(
            dce_config.nawm_mask
            or tumor_seg_wf is not None
            or (
                work_dir is not None
                and not decision.run_tumor_seg
                and _find_previous_tumor_seg_outputs(
                    Path(work_dir), subject_id, session_id,
                ).get("nawm_mask")
            )
        )
        dce_has_lesion = (
            dce_config.lesion_mask is not None or decision.run_tumor_seg
        )

        # Inputs that dce_wf.outputnode is expected to populate. Gating
        # mirrors what compute_pk_maps and friends actually emit per config.
        dce_present: set = {
            # Mandatory always
            "ktrans_map",
            # Always-present model-free maps + QC
            "iauc60_map", "iauc90_map", "iauc120_map",
            "peak_enhancement_map", "time_to_peak_map", "ser_map",
            "wash_in_slope_map", "wash_out_slope_map",
            "bat_map", "curve_type_map",
            "tsnr_map", "qc_mask",
            # Always-present registration + AIF
            "dce_moco", "dce_mean", "motion_params",
            "aif_curve", "aif_mask",
            "dce_to_t1w_xfm",
            # Brain mask in DCE always when DCE runs
            "brain_mask",
        }
        # Per-model PK parameter gating (mirrors dce_compute._has_ve / _has_vp):
        #   tofts          → ve, kep
        #   extended_tofts → ve, kep, vp
        #   patlak         → vp     (no ve/kep — would crash _copy_perf_map)
        #   2cxm           → ve, kep, vp + Fp/PS
        #   auto           → all (per-voxel BIC selection)
        _has_ve_models = {"tofts", "extended_tofts", "2cxm", "auto"}
        _has_vp_models = {"extended_tofts", "patlak", "2cxm", "auto"}
        # 2CXM uncertainty (Fp/PS std) sinks are gated on whether the
        # producer will actually emit the files — see
        # :func:`_emit_2cxm_uncertainty` which mirrors the runtime
        # candidate-set TR gate.
        _dce_volume_tr = (
            dce_sidecar.get("volume_tr")
            if isinstance(dce_sidecar, dict) else None
        )
        _emit_fp_ps_uncertainty = _emit_2cxm_uncertainty(
            dce_model, _dce_volume_tr,
        )
        if dce_model in _has_ve_models:
            dce_present |= {"ve_map", "kep_map"}
        if dce_model in _has_vp_models:
            dce_present.add("vp_map")
        # Auto-BIC produces model_selection + delta_bic
        if dce_model == "auto":
            dce_present |= {"model_selection_map", "delta_bic_map"}
        # R² + quality flag are emitted by every PK fit
        dce_present |= {"r2_map", "quality_flag_map"}
        # Bayesian uncertainty/CV — gated by model AND inference != "nlls".
        # CI95 is written by ANY non-NLLS inference (incl. MAP); the legacy
        # gate was just `inference != "nlls"`.
        if dce_uses_bayes:
            dce_present |= {"ktrans_uncertainty", "ktrans_cv"}
            if dce_model in _has_ve_models:
                dce_present |= {"ve_uncertainty", "ve_cv"}
            if dce_model in _has_vp_models:
                dce_present |= {"vp_uncertainty", "vp_cv"}
            if _emit_fp_ps_uncertainty:
                dce_present |= {"fp_uncertainty", "ps_uncertainty"}
            dce_present |= {"ktrans_ci95_lower", "ktrans_ci95_upper"}
            if dce_model in _has_ve_models:
                dce_present |= {"ve_ci95_lower", "ve_ci95_upper"}
            if dce_model in _has_vp_models:
                dce_present |= {"vp_ci95_lower", "vp_ci95_upper"}
        # Tortuosity correction (DWI-derived ADC)
        if not dce_config.no_tortuosity and session_data.modalities.dwi:
            dce_present.add("ktrans_tortuosity_corrected")
        # NAWM normalization + NAWM mask
        if dce_has_nawm:
            dce_present.add("nktrans_map")
            dce_present.add("niauc60_map")
            dce_present.add("nawm_mask")
        # T1map + B1+map (anat/)
        if not dce_config.no_t1:
            dce_present.add("t1_map")
        if dce_config.b1_method != "none":
            dce_present.add("b1_map")
        # Lesion / ROI
        if dce_has_lesion:
            dce_present.add("lesion_mask_dce")
        if dce_config.roi_mask is not None:
            dce_present.add("roi_mask_dce")
        # Structural underlays — gate on session_data, not just config.
        # See dsc_present underlay block for rationale.
        _dce_has_t1w_precon_underlay = (
            "t1w_precon" in dce_underlays
            and session_data.modalities.t1w_pre is not None
        )
        _dce_has_t1w_postcon_underlay = (
            "t1w_postcon" in dce_underlays
            and session_data.modalities.t1w_ce_gad is not None
        )
        _dce_has_flair_underlay = (
            "flair" in dce_underlays
            and session_data.modalities.flair is not None
        )
        _dce_has_t2w_underlay = (
            "t2w" in dce_underlays
            and session_data.modalities.t2w is not None
        )
        if _dce_has_t1w_precon_underlay:
            dce_present.add("t1w_in_dce")
        if _dce_has_t1w_postcon_underlay:
            dce_present.add("t1w_postcon_in_dce")
        if _dce_has_flair_underlay:
            dce_present.add("flair_in_dce")
        if _dce_has_t2w_underlay:
            dce_present.add("t2w_in_dce")

        dce_datasinks_wf = init_dce_datasinks_wf(
            output_dir=str(output_dir),
            bids_dir=str(bids_dir),
            subject_id=subject_id,
            session_id=session_id,
            pk_model=dce_model,
            present_inputs=dce_present,
            has_smoothing=dce_config.smooth_fwhm > 0,
        )
        if dce_provenance:
            dce_datasinks_wf.inputs.inputnode.provenance_dict = dce_provenance
        if dce_preflight_warnings:
            dce_datasinks_wf.inputs.inputnode.preflight_warnings = list(
                dce_preflight_warnings
            )

        # Connect DCE outputs to factory
        workflow.connect([
            (inputnode, dce_datasinks_wf, [
                ("dce_file", "inputnode.source_dce"),
            ]),
            (dce_wf, dce_datasinks_wf, [
                # Core PK parameter maps
                ("outputnode.ktrans_map", "inputnode.ktrans_map"),
                ("outputnode.ve_map", "inputnode.ve_map"),
                ("outputnode.kep_map", "inputnode.kep_map"),
                ("outputnode.vp_map", "inputnode.vp_map"),
                # Model-free maps
                ("outputnode.iauc60_map", "inputnode.iauc60_map"),
                ("outputnode.iauc90_map", "inputnode.iauc90_map"),
                ("outputnode.iauc120_map", "inputnode.iauc120_map"),
                ("outputnode.peak_enhancement_map", "inputnode.peak_enhancement_map"),
                ("outputnode.time_to_peak_map", "inputnode.time_to_peak_map"),
                ("outputnode.ser_map", "inputnode.ser_map"),
                ("outputnode.wash_in_slope_map", "inputnode.wash_in_slope_map"),
                ("outputnode.wash_out_slope_map", "inputnode.wash_out_slope_map"),
                ("outputnode.bat_map", "inputnode.bat_map"),
                ("outputnode.curve_type_map", "inputnode.curve_type_map"),
                # QC maps
                ("outputnode.tsnr_map", "inputnode.tsnr_map"),
                ("outputnode.qc_mask", "inputnode.qc_mask"),
                ("outputnode.r2_map", "inputnode.r2_map"),
                ("outputnode.quality_flag_map", "inputnode.quality_flag_map"),
                ("outputnode.model_selection_map", "inputnode.model_selection_map"),
                ("outputnode.delta_bic_map", "inputnode.delta_bic_map"),
                # T1/B1 mapping
                ("outputnode.t1_map", "inputnode.t1_map"),
                ("outputnode.b1_map", "inputnode.b1_map"),
                # Motion-corrected data
                ("outputnode.dce_moco", "inputnode.dce_moco"),
                ("outputnode.dce_mean", "inputnode.dce_mean"),
                ("outputnode.motion_params", "inputnode.motion_params"),
                # AIF outputs
                ("outputnode.aif_curve", "inputnode.aif_curve"),
                ("outputnode.aif_mask", "inputnode.aif_mask"),
                # Registration
                ("outputnode.dce_to_t1w_xfm", "inputnode.dce_to_t1w_xfm"),
                # Uncertainty maps (Bayesian)
                ("outputnode.ktrans_uncertainty", "inputnode.ktrans_uncertainty"),
                ("outputnode.ve_uncertainty", "inputnode.ve_uncertainty"),
                ("outputnode.vp_uncertainty", "inputnode.vp_uncertainty"),
                ("outputnode.fp_uncertainty", "inputnode.fp_uncertainty"),
                ("outputnode.ps_uncertainty", "inputnode.ps_uncertainty"),
                # Bayesian CV maps
                ("outputnode.ktrans_cv", "inputnode.ktrans_cv"),
                ("outputnode.ve_cv", "inputnode.ve_cv"),
                ("outputnode.vp_cv", "inputnode.vp_cv"),
                # Bayesian CI maps
                ("outputnode.ktrans_ci95_lower", "inputnode.ktrans_ci95_lower"),
                ("outputnode.ktrans_ci95_upper", "inputnode.ktrans_ci95_upper"),
                ("outputnode.ve_ci95_lower", "inputnode.ve_ci95_lower"),
                ("outputnode.ve_ci95_upper", "inputnode.ve_ci95_upper"),
                ("outputnode.vp_ci95_lower", "inputnode.vp_ci95_lower"),
                ("outputnode.vp_ci95_upper", "inputnode.vp_ci95_upper"),
                # Tortuosity-corrected Ktrans
                ("outputnode.ktrans_tortuosity_corrected", "inputnode.ktrans_tortuosity_corrected"),
                # Normalized maps (NAWM-normalized)
                ("outputnode.nktrans_map", "inputnode.nktrans_map"),
                ("outputnode.niauc60_map", "inputnode.niauc60_map"),
                # Smoothed maps
                ("outputnode.smooth_pk_maps", "inputnode.smooth_pk_maps"),
                ("outputnode.smooth_model_free_maps", "inputnode.smooth_model_free_maps"),
                ("outputnode.smooth_uncertainty_maps", "inputnode.smooth_uncertainty_maps"),
                ("outputnode.smooth_tortuosity_maps", "inputnode.smooth_tortuosity_maps"),
                # QC metrics
                ("outputnode.qc_metrics", "inputnode.qc_metrics"),
                # QC plots
                ("outputnode.aif_curve_qc_plot", "inputnode.aif_curve_plot"),
                ("outputnode.aif_location_qc_plot", "inputnode.aif_location_plot"),
                ("outputnode.aif_candidates_qc_plot", "inputnode.aif_candidates_plot"),
                ("outputnode.registration_qc_filmstrip", "inputnode.registration_filmstrip"),
                ("outputnode.motion_qc_plot", "inputnode.motion_qc_plot"),
                # Brain mask filmstrip
                ("outputnode.brain_mask_filmstrip", "inputnode.brain_mask_filmstrip"),
                # DCE QC plots
                ("outputnode.preprocessing_qc_plot", "inputnode.preprocessing_qc_plot"),
                ("outputnode.pk_modeling_qc_plot", "inputnode.pk_modeling_qc_plot"),
                ("outputnode.pk_param_maps_qc_plot", "inputnode.pk_param_maps_qc_plot"),
                ("outputnode.model_selection_qc_plot", "inputnode.model_selection_qc_plot"),
                # B1/T1 QC plots
                ("outputnode.b1_qc_plot", "inputnode.b1_qc_plot"),
                ("outputnode.t1_qc_plot", "inputnode.t1_qc_plot"),
                # DSC-DCE cross-validation plot
                ("outputnode.dsc_dce_xval_plot", "inputnode.dsc_dce_xval_plot"),
                # Structural underlays
                ("outputnode.t1w_in_dce", "inputnode.t1w_in_dce"),
                ("outputnode.t1w_postcon_in_dce", "inputnode.t1w_postcon_in_dce"),
                ("outputnode.flair_in_dce", "inputnode.flair_in_dce"),
                ("outputnode.t2w_in_dce", "inputnode.t2w_in_dce"),
                # Lesion statistics
                ("outputnode.lesion_stats_json", "inputnode.lesion_stats_json"),
                ("outputnode.lesion_mask_dce", "inputnode.lesion_mask_dce"),
                # ROI statistics
                ("outputnode.roi_stats_json", "inputnode.roi_stats_json"),
                ("outputnode.roi_mask_dce", "inputnode.roi_mask_dce"),
                # Anatomical masks in DCE space
                ("outputnode.brain_mask", "inputnode.brain_mask"),
                ("outputnode.nawm_mask", "inputnode.nawm_mask"),
            ]),
        ])

        workflow.connect([
            (dce_datasinks_wf, report_barrier, [
                ("outputnode.done", "dce_done"),
            ]),
        ])

    return workflow


def get_pipeline_summary(workflow: pe.Workflow) -> str:
    """Generate a summary of the pipeline decision.

    Args:
        workflow: The initialized pmriprep workflow.

    Returns:
        Human-readable summary of what will run and why.
    """
    decision: PipelineDecision = workflow.pipeline_decision  # type: ignore[attr-defined]

    lines = ["PMRIPrep Pipeline Summary", "=" * 40]

    # Anatomical
    if decision.run_anat:
        lines.append("Anatomical preprocessing: WILL RUN")
    else:
        lines.append(f"Anatomical preprocessing: SKIPPED ({decision.anat_reason})")

    # Tumor segmentation
    if decision.run_tumor_seg:
        lines.append("Tumor segmentation: WILL RUN")
    else:
        lines.append(f"Tumor segmentation: SKIPPED ({decision.tumor_seg_reason})")

    # DSC
    if decision.run_dsc:
        lines.append("DSC perfusion: WILL RUN")
    else:
        lines.append(f"DSC perfusion: SKIPPED ({decision.dsc_reason})")

    # DCE
    if decision.run_dce:
        lines.append("DCE perfusion: WILL RUN")
        if decision.run_dsc:
            lines.append("  (DCE will use DSC CBV for vp constraint)")
    else:
        lines.append(f"DCE perfusion: SKIPPED ({decision.dce_reason})")

    return "\n".join(lines)
