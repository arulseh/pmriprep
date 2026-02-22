"""Master workflow for PMRIPrep BIDS-App.

This module implements the master workflow that orchestrates all pipeline
components (anatomical, HD-GLIO, DSC, DCE) based on available data and
user-specified skip flags.
"""

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Any, Optional, Union

import nipype.interfaces.utility as niu  # type: ignore[import-untyped]
import nipype.pipeline.engine as pe  # type: ignore[import-untyped]

from pmriprep.config.anat import AnatConfig
from pmriprep.config.dce import DCEConfig
from pmriprep.config.dsc import DSCConfig
from pmriprep.config.hdglio import HDGLIOConfig
from pmriprep.workflows.discovery import discover_session_data, SessionData


@dataclass
class PipelineSkipFlags:
    """Flags to control which pipeline components to skip.

    These flags allow users to explicitly disable pipeline components
    even when the required data is available.

    Attributes:
        skip_anat: Skip anatomical preprocessing pipeline.
        skip_tumor_seg: Skip HD-GLIO tumor segmentation pipeline.
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
        run_hdglio: Whether to run HD-GLIO tumor segmentation.
        run_dsc: Whether to run DSC perfusion processing.
        run_dce: Whether to run DCE perfusion processing.
        anat_reason: Reason for anat decision if not running.
        hdglio_reason: Reason for hdglio decision if not running.
        dsc_reason: Reason for dsc decision if not running.
        dce_reason: Reason for dce decision if not running.
    """

    run_anat: bool = True
    run_hdglio: bool = False
    run_dsc: bool = False
    run_dce: bool = False
    anat_reason: Optional[str] = None
    hdglio_reason: Optional[str] = None
    dsc_reason: Optional[str] = None
    dce_reason: Optional[str] = None


def decide_pipelines(
    session_data: SessionData,
    skip_flags: PipelineSkipFlags,
) -> PipelineDecision:
    """Decide which pipelines to run based on data and skip flags.

    This function applies the following rules:
    1. anat always runs (unless explicitly skipped)
    2. hdglio runs if all 4 modalities are present (unless skipped)
    3. dsc runs if DSC data is present (unless skipped)
    4. dce runs if DCE data is present (unless skipped)

    Args:
        session_data: Available data from discovery.
        skip_flags: User-specified skip flags.

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

    # HD-GLIO tumor segmentation
    # HD-GLIO requires anat outputs (can come from running anat or previous run)
    if skip_flags.skip_tumor_seg:
        decision.run_hdglio = False
        decision.hdglio_reason = "Skipped by user (--skip-tumor-seg)"
    elif not session_data.pipelines.hdglio:
        decision.run_hdglio = False
        decision.hdglio_reason = session_data.pipelines.hdglio_reason
    else:
        # HD-GLIO can run if anat is running OR if previous anat outputs exist
        # (checked at workflow connection time)
        decision.run_hdglio = True

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
        Dict with keys: t1w_preproc, brain_mask, wm_probseg, csf_probseg, t1w_to_mni_xfm, flair_coreg.
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
        "csf_probseg": None,
        "t1w_to_mni_xfm": None,
        "flair_coreg": None,
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

    # T1w post-contrast coregistered and masked
    postcon_dir = anat_dir / "mask_postcon"
    if postcon_dir.exists():
        candidates = sorted(postcon_dir.glob("*_masked.nii.gz"))
        if candidates:
            outputs["t1w_postcon_coreg"] = str(candidates[0])

    return outputs


def _find_previous_hdglio_inputs(
    work_dir: Path,
    subject_id: str,
    session_id: Optional[str],
) -> dict[str, Optional[str]]:
    """Find previously-computed anat outputs needed for HD-GLIO.

    When --skip-anat is used but HD-GLIO should run, this function locates
    the skull-stripped, coregistered images from a prior anat run.

    Args:
        work_dir: Nipype working directory (parent of the workflow directory).
        subject_id: BIDS subject ID (without 'sub-' prefix).
        session_id: BIDS session ID (without 'ses-' prefix), or None.

    Returns:
        Dict with keys for HD-GLIO inputs. Values are file paths or None.
    """
    session_label = f"ses-{session_id}" if session_id else "no-session"
    wf_name = f"pmriprep_sub{subject_id}_{session_label}_wf"
    anat_dir = work_dir / wf_name / "anat_wf"

    outputs: dict[str, Optional[str]] = {
        "hdglio_t1w_pre": None,
        "hdglio_t1w_postcon": None,
        "hdglio_t2w": None,
        "hdglio_flair": None,
        "brain_mask": None,
        "wm_probseg": None,
        "t1w_to_mni_xfm": None,
        "t1w_preproc": None,
    }

    if not anat_dir.exists():
        return outputs

    # HD-GLIO masked T1w (pre-contrast) - skull-stripped without bias correction
    hdglio_t1w_dir = anat_dir / "hdglio_mask_t1w"
    if hdglio_t1w_dir.exists():
        candidates = sorted(hdglio_t1w_dir.glob("*_masked.nii.gz"))
        if candidates:
            outputs["hdglio_t1w_pre"] = str(candidates[0])

    # HD-GLIO masked post-contrast T1w
    hdglio_postcon_dir = anat_dir / "hdglio_mask_postcon"
    if hdglio_postcon_dir.exists():
        candidates = sorted(hdglio_postcon_dir.glob("*_masked.nii.gz"))
        if candidates:
            outputs["hdglio_t1w_postcon"] = str(candidates[0])

    # HD-GLIO masked T2w
    hdglio_t2w_dir = anat_dir / "hdglio_mask_t2w"
    if hdglio_t2w_dir.exists():
        candidates = sorted(hdglio_t2w_dir.glob("*_masked.nii.gz"))
        if candidates:
            outputs["hdglio_t2w"] = str(candidates[0])

    # HD-GLIO masked FLAIR
    hdglio_flair_dir = anat_dir / "hdglio_mask_flair"
    if hdglio_flair_dir.exists():
        candidates = sorted(hdglio_flair_dir.glob("*_masked.nii.gz"))
        if candidates:
            outputs["hdglio_flair"] = str(candidates[0])

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
        "agent_standard_dose": 0.1,  # Default standard dose (mmol/kg)
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

    # Extract contrast agent r1 relaxivity from DICOM metadata
    # This enables agent-specific r1 values (e.g., Gadovist r1=5.0 at 3T)
    r1_info = get_r1_from_sidecar_dict(meta, result["field_strength"])
    if r1_info["r1"] is not None:
        result["r1_relaxivity"] = r1_info["r1"]
        result["contrast_agent"] = r1_info["agent_name"]

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


def _find_previous_hdglio_outputs(
    work_dir: Path,
    subject_id: str,
    session_id: Optional[str],
) -> dict[str, Optional[str]]:
    """Find previously-computed hdglio outputs in the work directory.

    When --skip-tumor-seg is used, DSC/DCE workflows may still need
    lesion and NAWM masks from a prior HD-GLIO run.

    Args:
        work_dir: Nipype working directory.
        subject_id: BIDS subject ID (without 'sub-' prefix).
        session_id: BIDS session ID (without 'ses-' prefix), or None.

    Returns:
        Dict with keys: dilated_lesion_mask, whole_tumor_mask,
        enhancing_mask, segmentation, nawm_mask.
        Values are file paths (str) or None if not found.
    """
    session_label = f"ses-{session_id}" if session_id else "no-session"
    wf_name = f"pmriprep_sub{subject_id}_{session_label}_wf"
    hdglio_dir = work_dir / wf_name / "hdglio_wf"

    outputs: dict[str, Optional[str]] = {
        "dilated_lesion_mask": None,
        "whole_tumor_mask": None,
        "enhancing_mask": None,
        "segmentation": None,
        "nawm_mask": None,
    }

    if not hdglio_dir.exists():
        return outputs

    masks_dir = hdglio_dir / "extract_masks"
    if masks_dir.exists():
        for f in masks_dir.glob("*_dilated.nii.gz"):
            outputs["dilated_lesion_mask"] = str(f)
        for f in masks_dir.glob("*_nawm.nii.gz"):
            outputs["nawm_mask"] = str(f)
        for f in masks_dir.glob("*wholetumor_mask.nii.gz"):
            outputs["whole_tumor_mask"] = str(f)
        for f in masks_dir.glob("*enhancing_mask.nii.gz"):
            if "nonenhancing" not in f.name and "non_enhancing" not in f.name:
                outputs["enhancing_mask"] = str(f)

    # Check hdglio_segment node for segmentation output
    seg_dir = hdglio_dir / "hdglio_segment"
    if seg_dir.exists():
        for f in seg_dir.glob("*_seg.nii.gz"):
            outputs["segmentation"] = str(f)
        for f in seg_dir.glob("*segmentation*.nii.gz"):
            outputs["segmentation"] = str(f)

    # Also check dilate_lesion and create_nawm nodes
    if outputs["dilated_lesion_mask"] is None:
        dl_dir = hdglio_dir / "dilate_lesion"
        if dl_dir.exists():
            for f in dl_dir.glob("*.nii.gz"):
                outputs["dilated_lesion_mask"] = str(f)

    if outputs["nawm_mask"] is None:
        nawm_dir = hdglio_dir / "create_nawm"
        if nawm_dir.exists():
            for f in nawm_dir.glob("*.nii.gz"):
                outputs["nawm_mask"] = str(f)

    # Check select_whole_tumor node for whole tumor mask
    if outputs["whole_tumor_mask"] is None:
        wt_dir = hdglio_dir / "select_whole_tumor"
        if wt_dir.exists():
            for f in wt_dir.glob("*.nii.gz"):
                outputs["whole_tumor_mask"] = str(f)

    # Check select_enhancing node for enhancing mask
    if outputs["enhancing_mask"] is None:
        enh_dir = hdglio_dir / "select_enhancing"
        if enh_dir.exists():
            for f in enh_dir.glob("*.nii.gz"):
                outputs["enhancing_mask"] = str(f)

    return outputs


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
        Dict with keys: rel_cbv_map, dsc_to_t1w_xfm, aif_qc.
        Values are file paths (str) or None if not found.
    """
    outputs: dict[str, Optional[str]] = {
        "rel_cbv_map": None,
        "dsc_to_t1w_xfm": None,
        "aif_qc": None,
    }

    ses_label = f"ses-{session_id}" if session_id else ""
    dsc_dir = output_dir / "pmriprep" / f"sub-{subject_id}"
    if ses_label:
        dsc_dir = dsc_dir / ses_label

    if not dsc_dir.exists():
        return outputs

    # Relative CBV map
    perf_dir = dsc_dir / "perf"
    if perf_dir.exists():
        for f in perf_dir.glob("*desc-dscRel_cbv.nii.gz"):
            outputs["rel_cbv_map"] = str(f)

    # DSC→T1w transform (now in anat/ instead of xfm/)
    anat_dir = dsc_dir / "anat"
    if anat_dir.exists():
        for f in anat_dir.glob("*from-DSC_to-anat*_xfm.mat"):
            outputs["dsc_to_t1w_xfm"] = str(f)

    # AIF QC from BIDS-named QC JSON in session root
    prefix = f"sub-{subject_id}"
    if ses_label:
        prefix += f"_{ses_label}"
    qc_json = dsc_dir / f"{prefix}_desc-dscQC_quality.json"
    if qc_json.exists():
        try:
            import json
            with open(qc_json) as fh:
                qc_data = json.load(fh)
            if "aif" in qc_data:
                outputs["aif_qc"] = qc_data["aif"]
        except Exception:
            pass

    return outputs


def init_pmriprep_wf(
    bids_dir: Union[str, Path],
    output_dir: Union[str, Path],
    subject_id: str,
    session_id: Optional[str] = None,
    skip_flags: Optional[PipelineSkipFlags] = None,
    anat_config: Optional[AnatConfig] = None,
    hdglio_config: Optional[HDGLIOConfig] = None,
    dsc_config: Optional[DSCConfig] = None,
    dce_config: Optional[DCEConfig] = None,
    work_dir: Optional[Union[str, Path]] = None,
    name: str = "pmriprep_wf",
) -> pe.Workflow:
    """Initialize the master PMRIPrep workflow.

    This workflow orchestrates all pipeline components based on available
    data and user-specified skip flags. It ensures proper ordering of
    pipelines (anatomical first, then HD-GLIO, then perfusion) and
    connects outputs between dependent pipelines.

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
    hdglio_config : HDGLIOConfig, optional
        Configuration for HD-GLIO tumor segmentation.
    dsc_config : DSCConfig, optional
        Configuration for DSC perfusion processing.
    dce_config : DCEConfig, optional
        Configuration for DCE perfusion processing.
    work_dir : Path or str, optional
        Nipype working directory. Required when --skip-anat or --skip-tumor-seg
        is used to locate previously-computed outputs.
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
    2. HD-GLIO tumor segmentation (requires anat outputs)
    3. DSC perfusion processing (requires anat outputs, optional hdglio)
    4. DCE perfusion processing (requires anat outputs, uses DSC CBV if available)

    When both DSC and DCE are present, DCE runs after DSC to use DSC's CBV
    for vp constraint (via US-063 QC check). This is handled by connecting
    DSC outputs to DCE inputs.
    """
    from pmriprep.workflows.anat.base import init_anat_preproc_wf
    from pmriprep.workflows.dce.base import init_dce_preproc_wf
    from pmriprep.workflows.dsc.base import init_dsc_preproc_wf
    from pmriprep.workflows.hdglio.base import init_hdglio_wf

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
    if hdglio_config is None:
        hdglio_config = HDGLIOConfig()
    if dsc_config is None:
        dsc_config = DSCConfig()
    if dce_config is None:
        dce_config = DCEConfig()

    # Discover available data
    session_data = discover_session_data(bids_dir, subject_id, session_id)

    # Decide which pipelines to run
    decision = decide_pipelines(session_data, skip_flags)

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
                "pdw",
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
    if session_data.modalities.pdw:
        inputnode.inputs.pdw = str(session_data.modalities.pdw)
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
        or session_data.pipelines.hdglio
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
                # HD-GLIO outputs
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
                "hdglio_qc",
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
    hdglio_wf = None
    dsc_wf = None
    dce_wf = None

    # -------------------------------------------------------------------
    # Unified report barrier + node (collects completion from all writers)
    # -------------------------------------------------------------------
    from pmriprep.interfaces.reports import GenerateUnifiedReport

    report_barrier = pe.Node(
        niu.IdentityInterface(
            fields=["anat_done", "hdglio_done", "dsc_done", "dce_done"],
        ),
        name="report_barrier",
    )
    # Defaults for pipelines that don't run
    report_barrier.inputs.anat_done = ""
    report_barrier.inputs.hdglio_done = ""
    report_barrier.inputs.dsc_done = ""
    report_barrier.inputs.dce_done = ""

    unified_report = pe.Node(
        GenerateUnifiedReport(
            subject_id=subject_id,
            session_id=session_id or "",
            output_dir=str(output_dir),
            ran_anat=decision.run_anat,
            ran_hdglio=decision.run_hdglio,
            ran_dsc=decision.run_dsc,
            ran_dce=decision.run_dce,
        ),
        name="unified_report",
    )

    workflow.connect([
        (report_barrier, unified_report, [
            ("anat_done", "anat_done"),
            ("hdglio_done", "hdglio_done"),
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
            has_pdw=session_data.modalities.pdw is not None,
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
        if session_data.modalities.pdw:
            workflow.connect([
                (inputnode, anat_wf, [("pdw", "inputnode.pdw")]),
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
        # AnatDerivativeWriter: Write BIDS-compliant anat outputs
        # -------------------------------------------------------------------
        from pmriprep.interfaces.anat_outputs import AnatDerivativeWriter

        anat_writer = pe.Node(
            AnatDerivativeWriter(
                output_dir=str(output_dir),
                subject_id=subject_id,
                session_id=session_id or "",
                bids_dir=str(bids_dir),
            ),
            name="anat_writer",
        )

        # Connect core anat outputs
        workflow.connect([
            (anat_wf, anat_writer, [
                ("outputnode.t1w_preproc", "t1w_preproc"),
                ("outputnode.brain_mask", "brain_mask"),
                ("outputnode.t1w_to_mni_xfm", "acpc_xfm"),
                ("outputnode.qc_metrics", "qc_metrics"),
            ]),
        ])

        # Tissue probability maps: only available when precon T1w exists
        # (FAST segmentation is skipped in postcon-only mode because
        # gadolinium disrupts tissue boundaries)
        if session_data.modalities.t1w_pre:
            workflow.connect([
                (anat_wf, anat_writer, [
                    ("outputnode.wm_probseg", "wm_probseg"),
                    ("outputnode.gm_probseg", "gm_probseg"),
                    ("outputnode.csf_probseg", "csf_probseg"),
                ]),
            ])

        # Connect source paths for JSON sidecar Sources field
        if session_data.modalities.t1w_pre:
            workflow.connect([
                (inputnode, anat_writer, [("t1w_pre", "source_t1w")]),
            ])

        # Connect optional modalities
        # Postcon coreg/xfm outputs only exist when both pre and postcon are available
        if session_data.modalities.t1w_ce_gad and session_data.modalities.t1w_pre:
            writer_connections = [
                ("outputnode.t1w_postcon_coreg", "t1w_postcon_coreg"),
                ("outputnode.t1w_postcon_to_t1w_xfm", "t1w_postcon_to_t1w_xfm"),
            ]
            # Subtraction requires both pre and post-contrast
            if session_data.modalities.t1w_pre:
                writer_connections.append(
                    ("outputnode.t1w_subtraction", "t1w_subtraction")
                )
            workflow.connect([
                (anat_wf, anat_writer, writer_connections),
            ])
        if session_data.modalities.t1w_ce_gad:
            workflow.connect([
                (inputnode, anat_writer, [("t1w_postcon", "source_t1w_postcon")]),
            ])
        if session_data.modalities.t2w:
            workflow.connect([
                (anat_wf, anat_writer, [
                    ("outputnode.t2w_coreg", "t2w_coreg"),
                    ("outputnode.t2w_to_t1w_xfm", "t2w_to_t1w_xfm"),
                ]),
                (inputnode, anat_writer, [("t2w", "source_t2w")]),
            ])
        if session_data.modalities.flair:
            workflow.connect([
                (anat_wf, anat_writer, [
                    ("outputnode.flair_coreg", "flair_coreg"),
                    ("outputnode.flair_to_t1w_xfm", "flair_to_t1w_xfm"),
                ]),
                (inputnode, anat_writer, [("flair", "source_flair")]),
            ])
        if session_data.modalities.pdw:
            workflow.connect([
                (anat_wf, anat_writer, [
                    ("outputnode.pdw_coreg", "pdw_coreg"),
                    ("outputnode.pdw_to_t1w_xfm", "pdw_to_t1w_xfm"),
                ]),
                (inputnode, anat_writer, [("pdw", "source_pdw")]),
            ])

        # -------------------------------------------------------------------
        # Anat: connect writer to unified report barrier
        # -------------------------------------------------------------------
        workflow.connect([
            (anat_writer, report_barrier, [("qc_json", "anat_done")]),
        ])

        # Connect filmstrips to writer (writer saves to figures/ dir)
        workflow.connect([
            (anat_wf, anat_writer, [
                ("outputnode.brain_mask_filmstrip", "brain_mask_filmstrip"),
            ]),
        ])
        # Tissue segmentation filmstrip: unavailable in postcon-only mode
        # (FAST skipped because gadolinium disrupts tissue boundaries)
        if session_data.modalities.t1w_pre:
            workflow.connect([
                (anat_wf, anat_writer, [
                    ("outputnode.tissue_seg_filmstrip", "tissue_seg_filmstrip"),
                ]),
            ])

        workflow.connect([
            (anat_wf, anat_writer, [
                ("outputnode.acpc_filmstrip", "acpc_filmstrip"),
            ]),
        ])

        if session_data.modalities.t1w_ce_gad:
            workflow.connect([
                (anat_wf, anat_writer, [
                    ("outputnode.coreg_filmstrip_postcon", "coreg_filmstrip_postcon"),
                ]),
            ])
        if session_data.modalities.t2w:
            workflow.connect([
                (anat_wf, anat_writer, [
                    ("outputnode.coreg_filmstrip_t2w", "coreg_filmstrip_t2w"),
                ]),
            ])
        if session_data.modalities.flair:
            workflow.connect([
                (anat_wf, anat_writer, [
                    ("outputnode.coreg_filmstrip_flair", "coreg_filmstrip_flair"),
                ]),
            ])
        if session_data.modalities.pdw:
            workflow.connect([
                (anat_wf, anat_writer, [
                    ("outputnode.coreg_filmstrip_pdw", "coreg_filmstrip_pdw"),
                ]),
            ])

    # -------------------------------------------------------------------
    # HD-GLIO tumor segmentation workflow (requires anat outputs)
    # -------------------------------------------------------------------
    if decision.run_hdglio:
        hdglio_wf = init_hdglio_wf(
            bids_dir=bids_dir,
            subject_id=subject_id,
            session_id=session_id,
            config=hdglio_config,
            name="hdglio_wf",
        )

        # Connect prepared inputs from anat workflow
        # HD-GLIO receives coregistered, skull-stripped, non-bias-corrected
        # images from the anat pipeline's HD-GLIO prep track
        if anat_wf is not None:
            workflow.connect([
                (anat_wf, hdglio_wf, [
                    ("outputnode.hdglio_t1w_pre", "inputnode.t1w_pre"),
                    ("outputnode.hdglio_t1w_postcon", "inputnode.t1w_postcon"),
                    ("outputnode.hdglio_t2w", "inputnode.t2w"),
                    ("outputnode.hdglio_flair", "inputnode.flair"),
                    ("outputnode.brain_mask", "inputnode.brain_mask"),
                    ("outputnode.wm_probseg", "inputnode.wm_probseg"),
                    ("outputnode.t1w_to_mni_xfm", "inputnode.t1w_to_mni_xfm"),
                    ("outputnode.t1w_preproc", "inputnode.t1w_preproc"),
                ]),
            ])
        elif work_dir is not None:
            # --skip-anat: load previously-computed anat outputs for HD-GLIO
            import logging as _logging
            _log = _logging.getLogger("pmriprep.workflows")
            prev_hdglio = _find_previous_hdglio_inputs(
                Path(work_dir), subject_id, session_id,
            )
            # Check if required inputs are available
            if prev_hdglio["hdglio_t1w_pre"]:
                hdglio_wf.inputs.inputnode.t1w_pre = prev_hdglio["hdglio_t1w_pre"]
                _log.info("skip-anat (HD-GLIO): loaded t1w_pre from %s", prev_hdglio["hdglio_t1w_pre"])
            if prev_hdglio["hdglio_t1w_postcon"]:
                hdglio_wf.inputs.inputnode.t1w_postcon = prev_hdglio["hdglio_t1w_postcon"]
                _log.info("skip-anat (HD-GLIO): loaded t1w_postcon from %s", prev_hdglio["hdglio_t1w_postcon"])
            if prev_hdglio["hdglio_t2w"]:
                hdglio_wf.inputs.inputnode.t2w = prev_hdglio["hdglio_t2w"]
                _log.info("skip-anat (HD-GLIO): loaded t2w from %s", prev_hdglio["hdglio_t2w"])
            if prev_hdglio["hdglio_flair"]:
                hdglio_wf.inputs.inputnode.flair = prev_hdglio["hdglio_flair"]
                _log.info("skip-anat (HD-GLIO): loaded flair from %s", prev_hdglio["hdglio_flair"])
            if prev_hdglio["brain_mask"]:
                hdglio_wf.inputs.inputnode.brain_mask = prev_hdglio["brain_mask"]
                _log.info("skip-anat (HD-GLIO): loaded brain_mask from %s", prev_hdglio["brain_mask"])
            if prev_hdglio["wm_probseg"]:
                hdglio_wf.inputs.inputnode.wm_probseg = prev_hdglio["wm_probseg"]
                _log.info("skip-anat (HD-GLIO): loaded wm_probseg from %s", prev_hdglio["wm_probseg"])
            if prev_hdglio["t1w_to_mni_xfm"]:
                hdglio_wf.inputs.inputnode.t1w_to_mni_xfm = prev_hdglio["t1w_to_mni_xfm"]
                _log.info("skip-anat (HD-GLIO): loaded t1w_to_mni_xfm from %s", prev_hdglio["t1w_to_mni_xfm"])
            if prev_hdglio["t1w_preproc"]:
                hdglio_wf.inputs.inputnode.t1w_preproc = prev_hdglio["t1w_preproc"]
                _log.info("skip-anat (HD-GLIO): loaded t1w_preproc from %s", prev_hdglio["t1w_preproc"])

        # Connect hdglio outputs to master outputnode
        workflow.connect([
            (hdglio_wf, outputnode, [
                ("outputnode.dilated_lesion_mask", "lesion_mask"),
                ("outputnode.enhancing_mask", "enhancing_mask"),
                ("outputnode.non_enhancing_mask", "non_enhancing_mask"),
                ("outputnode.whole_tumor_mask", "whole_tumor_mask"),
                ("outputnode.nawm_mask", "nawm_mask"),
                ("outputnode.qc_metrics", "hdglio_qc"),
            ]),
        ])

        # -------------------------------------------------------------------
        # HDGLIODerivativeWriter: Write BIDS-compliant HD-GLIO outputs
        # -------------------------------------------------------------------
        from pmriprep.interfaces.hdglio_outputs import HDGLIODerivativeWriter

        hdglio_writer = pe.Node(
            HDGLIODerivativeWriter(
                output_dir=str(output_dir),
                subject_id=subject_id,
                session_id=session_id or "",
                bids_dir=str(bids_dir),
            ),
            name="hdglio_writer",
        )

        workflow.connect([
            (hdglio_wf, hdglio_writer, [
                ("outputnode.segmentation", "segmentation"),
                ("outputnode.enhancing_mask", "enhancing_mask"),
                ("outputnode.non_enhancing_mask", "non_enhancing_mask"),
                ("outputnode.whole_tumor_mask", "whole_tumor_mask"),
                ("outputnode.dilated_lesion_mask", "dilated_lesion_mask"),
                ("outputnode.nawm_mask", "nawm_mask"),
                ("outputnode.qc_metrics", "qc_metrics"),
            ]),
        ])

        # -------------------------------------------------------------------
        # HD-GLIO: connect writer to unified report barrier
        # -------------------------------------------------------------------
        workflow.connect([
            (hdglio_writer, report_barrier, [("qc_json", "hdglio_done")]),
        ])

        # Connect filmstrips to writer (writer saves to figures/ dir)
        workflow.connect([
            (hdglio_wf, hdglio_writer, [
                ("outputnode.seg_filmstrip_postcon", "seg_filmstrip_postcon"),
                ("outputnode.seg_filmstrip_flair", "seg_filmstrip_flair"),
                ("outputnode.nawm_filmstrip_flair", "nawm_filmstrip_flair"),
            ]),
        ])

    # -------------------------------------------------------------------
    # DSC perfusion workflow (requires anat, optional hdglio)
    # -------------------------------------------------------------------
    if decision.run_dsc:
        dsc_wf = init_dsc_preproc_wf(
            bids_dir=bids_dir,
            subject_id=subject_id,
            session_id=session_id,
            config=dsc_config,
            name="dsc_wf",
        )

        # Read TE/TR/field_strength from BIDS sidecar JSON
        if session_data.modalities.dsc:
            sidecar = _read_bids_sidecar(session_data.modalities.dsc)
            if sidecar["te"] is not None:
                dsc_wf.inputs.inputnode.te = sidecar["te"]
            if sidecar["tr"] is not None:
                dsc_wf.inputs.inputnode.tr = sidecar["tr"]
            if sidecar["field_strength"] is not None:
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

        # Determine DSC structural reference from confidence-scored selection
        dsc_underlay_is_postcon = False
        dsc_ref_output = "outputnode.t1w_preproc"  # default
        if dsc_ref_selection:
            dsc_ref_output = _map_selection_to_anat_output(
                dsc_ref_selection.candidate.image_type,
            )
            dsc_underlay_is_postcon = (
                dsc_ref_selection.candidate.image_type == ImageType.T1W_POSTCON
            )
            # Postcon-only override: t1w_postcon_coreg requires both pre and postcon
            if dsc_underlay_is_postcon and not session_data.modalities.t1w_pre:
                dsc_ref_output = "outputnode.t1w_preproc"

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
            # T1w underlay: use confidence-scored selection
            if dsc_ref_selection and dsc_ref_output != "outputnode.t1w_preproc":
                workflow.connect([
                    (anat_wf, dsc_wf, [(dsc_ref_output, "inputnode.t1w_underlay")]),
                ])
            else:
                # Fallback: postcon if available (and precon exists for coreg), else precon
                if (session_data.modalities.t1w_ce_gad
                        and session_data.modalities.t1w_pre):
                    workflow.connect([
                        (anat_wf, dsc_wf, [("outputnode.t1w_postcon_coreg", "inputnode.t1w_underlay")]),
                    ])
                    dsc_underlay_is_postcon = True
                else:
                    # In postcon-only mode, t1w_preproc IS the postcon
                    if session_data.modalities.t1w_ce_gad:
                        dsc_underlay_is_postcon = True
                    workflow.connect([
                        (anat_wf, dsc_wf, [("outputnode.t1w_preproc", "inputnode.t1w_underlay")]),
                    ])
            # Connect FLAIR underlay if available
            if session_data.modalities.flair:
                workflow.connect([
                    (anat_wf, dsc_wf, [("outputnode.flair_coreg", "inputnode.flair_in_t1w")]),
                ])
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
            # Set t1w_underlay: use confidence-scored selection when possible
            if dsc_ref_selection and dsc_ref_selection.candidate.image_type == ImageType.T1W_POSTCON:
                if prev_anat["t1w_postcon_coreg"]:
                    dsc_wf.inputs.inputnode.t1w_underlay = prev_anat["t1w_postcon_coreg"]
                    dsc_underlay_is_postcon = True
                    _log.info("skip-anat (DSC): loaded t1w_underlay from %s", prev_anat["t1w_postcon_coreg"])
                elif prev_anat["t1w_preproc"]:
                    dsc_wf.inputs.inputnode.t1w_underlay = prev_anat["t1w_preproc"]
                    _log.info("skip-anat (DSC): loaded t1w_underlay from %s", prev_anat["t1w_preproc"])
            elif session_data.modalities.t1w_ce_gad and prev_anat["t1w_postcon_coreg"]:
                dsc_wf.inputs.inputnode.t1w_underlay = prev_anat["t1w_postcon_coreg"]
                dsc_underlay_is_postcon = True
                _log.info("skip-anat (DSC): loaded t1w_underlay from %s", prev_anat["t1w_postcon_coreg"])
            elif prev_anat["t1w_preproc"]:
                dsc_wf.inputs.inputnode.t1w_underlay = prev_anat["t1w_preproc"]
                _log.info("skip-anat (DSC): loaded t1w_underlay from %s", prev_anat["t1w_preproc"])
            if prev_anat["brain_mask"]:
                dsc_wf.inputs.inputnode.brain_mask = prev_anat["brain_mask"]
                _log.info("skip-anat: loaded brain_mask from %s", prev_anat["brain_mask"])
            if prev_anat["wm_probseg"]:
                dsc_wf.inputs.inputnode.wm_probseg = prev_anat["wm_probseg"]
                _log.info("skip-anat: loaded wm_probseg from %s", prev_anat["wm_probseg"])
            if prev_anat["t1w_to_mni_xfm"]:
                dsc_wf.inputs.inputnode.t1w_to_mni_xfm = prev_anat["t1w_to_mni_xfm"]
                _log.info("skip-anat: loaded t1w_to_mni_xfm from %s", prev_anat["t1w_to_mni_xfm"])
            if prev_anat["flair_coreg"]:
                dsc_wf.inputs.inputnode.flair_in_t1w = prev_anat["flair_coreg"]
                _log.info("skip-anat: loaded flair_in_t1w from %s", prev_anat["flair_coreg"])

            if not any(prev_anat.values()):
                _log.warning(
                    "skip-anat: no previous anat outputs found in %s. "
                    "DSC nodes requiring anat inputs will fail.",
                    work_dir,
                )

        # Connect lesion mask and tumor segmentation from hdglio if available
        # Also wire user-supplied masks from config (lesion_mask, nawm_mask, roi_mask)
        _dsc_log = logging.getLogger("pmriprep.workflows")
        _user_lesion = dsc_config.lesion_mask
        _user_nawm = dsc_config.nawm_mask
        _user_roi = dsc_config.roi_mask

        if _user_roi:
            dsc_wf.inputs.inputnode.roi_mask = str(_user_roi)
            _dsc_log.info("DSC: user roi_mask=%s", _user_roi)

        if hdglio_wf is not None:
            if _user_lesion:
                # Merge HD-GLIO + user lesion masks
                from pmriprep.utils.mask_utils import DilateMask, MergeMasks

                dsc_dilate_user_lesion = pe.Node(
                    DilateMask(radius_mm=15.0),
                    name="dsc_dilate_user_lesion",
                )
                dsc_dilate_user_lesion.inputs.mask_file = str(_user_lesion)

                dsc_merge_dilated = pe.Node(
                    MergeMasks(), name="dsc_merge_dilated",
                )
                dsc_merge_undilated = pe.Node(
                    MergeMasks(), name="dsc_merge_undilated",
                )
                dsc_merge_undilated.inputs.mask_b = str(_user_lesion)

                workflow.connect([
                    (dsc_dilate_user_lesion, dsc_merge_dilated, [
                        ("dilated_mask", "mask_b"),
                    ]),
                    (hdglio_wf, dsc_merge_dilated, [
                        ("outputnode.dilated_lesion_mask", "mask_a"),
                    ]),
                    (hdglio_wf, dsc_merge_undilated, [
                        ("outputnode.whole_tumor_mask", "mask_a"),
                    ]),
                    (dsc_merge_dilated, dsc_wf, [
                        ("merged_mask", "inputnode.lesion_mask"),
                    ]),
                    (dsc_merge_undilated, dsc_wf, [
                        ("merged_mask", "inputnode.whole_tumor_mask"),
                    ]),
                    (hdglio_wf, dsc_wf, [
                        ("outputnode.enhancing_mask", "inputnode.enhancing_mask"),
                        ("outputnode.segmentation", "inputnode.hdglio_segmentation"),
                        ("outputnode.nawm_mask", "inputnode.nawm_mask"),
                    ]),
                ])
                _dsc_log.info(
                    "DSC: merging user lesion_mask=%s with HD-GLIO masks",
                    _user_lesion,
                )
            else:
                # HD-GLIO only — current behavior
                workflow.connect([
                    (hdglio_wf, dsc_wf, [
                        ("outputnode.dilated_lesion_mask", "inputnode.lesion_mask"),
                        ("outputnode.whole_tumor_mask", "inputnode.whole_tumor_mask"),
                        ("outputnode.enhancing_mask", "inputnode.enhancing_mask"),
                        ("outputnode.segmentation", "inputnode.hdglio_segmentation"),
                        ("outputnode.nawm_mask", "inputnode.nawm_mask"),
                    ]),
                ])
        elif work_dir is not None and not decision.run_hdglio:
            # --skip-tumor-seg: load previously-computed hdglio outputs
            prev_hdglio = _find_previous_hdglio_outputs(
                Path(work_dir), subject_id, session_id,
            )
            if _user_lesion and prev_hdglio["dilated_lesion_mask"]:
                # Merge user lesion with loaded HD-GLIO masks
                from pmriprep.utils.mask_utils import DilateMask, MergeMasks

                dsc_dilate_user_lesion = pe.Node(
                    DilateMask(radius_mm=15.0),
                    name="dsc_dilate_user_lesion",
                )
                dsc_dilate_user_lesion.inputs.mask_file = str(_user_lesion)

                dsc_merge_dilated = pe.Node(
                    MergeMasks(), name="dsc_merge_dilated",
                )
                dsc_merge_dilated.inputs.mask_a = prev_hdglio["dilated_lesion_mask"]

                dsc_merge_undilated = pe.Node(
                    MergeMasks(), name="dsc_merge_undilated",
                )
                dsc_merge_undilated.inputs.mask_b = str(_user_lesion)
                if prev_hdglio["whole_tumor_mask"]:
                    dsc_merge_undilated.inputs.mask_a = prev_hdglio["whole_tumor_mask"]

                workflow.connect([
                    (dsc_dilate_user_lesion, dsc_merge_dilated, [
                        ("dilated_mask", "mask_b"),
                    ]),
                    (dsc_merge_dilated, dsc_wf, [
                        ("merged_mask", "inputnode.lesion_mask"),
                    ]),
                    (dsc_merge_undilated, dsc_wf, [
                        ("merged_mask", "inputnode.whole_tumor_mask"),
                    ]),
                ])
                _dsc_log.info(
                    "DSC skip-tumor-seg: merging user lesion_mask=%s with loaded HD-GLIO masks",
                    _user_lesion,
                )
            else:
                if prev_hdglio["dilated_lesion_mask"]:
                    dsc_wf.inputs.inputnode.lesion_mask = prev_hdglio["dilated_lesion_mask"]
                    _dsc_log.info("skip-tumor-seg: loaded lesion_mask from %s", prev_hdglio["dilated_lesion_mask"])
                if prev_hdglio["whole_tumor_mask"]:
                    dsc_wf.inputs.inputnode.whole_tumor_mask = prev_hdglio["whole_tumor_mask"]
                    _dsc_log.info("skip-tumor-seg: loaded whole_tumor_mask from %s", prev_hdglio["whole_tumor_mask"])

            if prev_hdglio["enhancing_mask"]:
                dsc_wf.inputs.inputnode.enhancing_mask = prev_hdglio["enhancing_mask"]
                _dsc_log.info("skip-tumor-seg: loaded enhancing_mask from %s", prev_hdglio["enhancing_mask"])
            if prev_hdglio["segmentation"]:
                dsc_wf.inputs.inputnode.hdglio_segmentation = prev_hdglio["segmentation"]
                _dsc_log.info("skip-tumor-seg: loaded hdglio_segmentation from %s", prev_hdglio["segmentation"])
            if prev_hdglio["nawm_mask"]:
                dsc_wf.inputs.inputnode.nawm_mask = prev_hdglio["nawm_mask"]
                _dsc_log.info("skip-tumor-seg: loaded nawm_mask from %s", prev_hdglio["nawm_mask"])
        else:
            # No HD-GLIO — user-supplied masks only
            if _user_lesion:
                from pmriprep.utils.mask_utils import DilateMask

                dsc_dilate_user_lesion = pe.Node(
                    DilateMask(radius_mm=15.0),
                    name="dsc_dilate_user_lesion",
                )
                dsc_dilate_user_lesion.inputs.mask_file = str(_user_lesion)
                workflow.connect([
                    (dsc_dilate_user_lesion, dsc_wf, [
                        ("dilated_mask", "inputnode.lesion_mask"),
                    ]),
                ])
                dsc_wf.inputs.inputnode.whole_tumor_mask = str(_user_lesion)
                _dsc_log.info("DSC: user lesion_mask=%s (dilated→lesion_mask, original→whole_tumor_mask)", _user_lesion)

            if _user_nawm:
                dsc_wf.inputs.inputnode.nawm_mask = str(_user_nawm)
                _dsc_log.info("DSC: user nawm_mask=%s", _user_nawm)

        # Connect DSC outputs to master outputnode
        workflow.connect([
            (dsc_wf, outputnode, [
                ("outputnode.rel_cbf_map", "dsc_cbf"),
                ("outputnode.rel_cbv_map", "dsc_cbv"),
                ("outputnode.mtt_map", "dsc_mtt"),
                ("outputnode.tmax_map", "dsc_tmax"),
                ("outputnode.aif_qc", "dsc_aif_qc"),
                ("outputnode.qc_metrics", "dsc_qc"),
            ]),
        ])

        # -------------------------------------------------------------------
        # DSCDerivativeWriter: Write BIDS-compliant DSC outputs
        # -------------------------------------------------------------------
        from pmriprep.interfaces.dsc_outputs import DSCDerivativeWriter
        import dataclasses

        dsc_config_dict = dataclasses.asdict(dsc_config)
        # Convert Path objects to strings for JSON serialization
        for k, v in dsc_config_dict.items():
            if isinstance(v, Path):
                dsc_config_dict[k] = str(v)

        dsc_writer = pe.Node(
            DSCDerivativeWriter(
                output_dir=str(output_dir),
                subject_id=subject_id,
                session_id=session_id or "",
                bids_dir=str(bids_dir),
                config_dict=dsc_config_dict,
                smooth_fwhm=dsc_config.smooth_fwhm,
                has_postcon_t1w=dsc_underlay_is_postcon,
            ),
            name="dsc_writer",
        )

        # Connect perfusion maps to writer
        workflow.connect([
            (dsc_wf, dsc_writer, [
                # Quantification-independent maps
                ("outputnode.mtt_map", "mtt_map"),
                ("outputnode.tmax_map", "tmax_map"),
                ("outputnode.psr_map", "psr_map"),
                ("outputnode.ph_map", "ph_map"),
                ("outputnode.ttp_map", "ttp_map"),
                ("outputnode.ttp_corrected_map", "ttp_corrected_map"),
                ("outputnode.rttp_map", "rttp_map"),
                ("outputnode.cnr_map", "cnr_map"),
                ("outputnode.auc_diff_map", "auc_diff_map"),
                ("outputnode.snr_map", "snr_map"),
                ("outputnode.tsnr_map", "tsnr_map"),
                # Relative maps
                ("outputnode.rel_cbf_map", "rel_cbf_map"),
                ("outputnode.rel_cbv_map", "rel_cbv_map"),
                ("outputnode.rel_auc_raw_map", "rel_auc_raw_map"),
                ("outputnode.rel_auc_corr_map", "rel_auc_corr_map"),
                # Normalized maps
                ("outputnode.ncbv_map", "ncbv_map"),
                ("outputnode.ncbf_map", "ncbf_map"),
                # Vascular model parameters
                ("outputnode.alpha_map", "alpha_map"),
                ("outputnode.beta_map", "beta_map"),
                ("outputnode.cth_map", "cth_map"),
                # Bayesian uncertainty maps
                ("outputnode.cbf_std_map", "cbf_std_map"),
                ("outputnode.cbv_std_map", "cbv_std_map"),
                ("outputnode.cbf_cv_map", "cbf_cv_map"),
                ("outputnode.cbv_cv_map", "cbv_cv_map"),
                ("outputnode.cbf_ci95_lower_map", "cbf_ci95_lower_map"),
                ("outputnode.cbf_ci95_upper_map", "cbf_ci95_upper_map"),
                ("outputnode.cbv_ci95_lower_map", "cbv_ci95_lower_map"),
                ("outputnode.cbv_ci95_upper_map", "cbv_ci95_upper_map"),
                # Leakage correction
                ("outputnode.k2_map", "k2_map"),
                # Motion-corrected data
                ("outputnode.dsc_moco", "dsc_moco"),
                ("outputnode.dsc_mean", "dsc_mean"),
                ("outputnode.motion_params", "motion_params"),
                # AIF outputs
                ("outputnode.aif_curve", "aif_curve"),
                ("outputnode.aif_mask", "aif_mask"),
                # Registration
                ("outputnode.dsc_to_t1w_xfm", "dsc_to_t1w_xfm"),
                ("outputnode.registration_filmstrip", "registration_filmstrip"),
                # Structural underlays
                ("outputnode.t1w_in_dsc", "t1w_in_dsc"),
                ("outputnode.flair_in_dsc", "flair_in_dsc"),
                ("outputnode.t1w_to_dsc_xfm", "t1w_to_dsc_xfm"),
                # AIF QC plots
                ("outputnode.aif_curve_plot", "aif_curve_plot"),
                ("outputnode.aif_location_plot", "aif_location_plot"),
                # Motion QC plot
                ("outputnode.motion_qc_plot", "motion_qc_plot"),
                # QC metrics
                ("outputnode.qc_metrics", "qc_metrics"),
                # DSC-space masks
                ("outputnode.brain_mask_dsc", "brain_mask_dsc"),
                ("outputnode.nawm_mask_dsc", "nawm_mask_dsc"),
                # Lesion statistics
                ("outputnode.lesion_stats_json", "lesion_stats_json"),
                ("outputnode.lesion_stats_txt", "lesion_stats_txt"),
                ("outputnode.lesion_mask_dsc", "lesion_mask_dsc"),
                # ROI statistics
                ("outputnode.roi_stats_json", "roi_stats_json"),
                ("outputnode.roi_stats_txt", "roi_stats_txt"),
                ("outputnode.roi_mask_dsc", "roi_mask_dsc"),
            ]),
            (inputnode, dsc_writer, [("dsc_file", "source_dsc")]),
        ])

        # Connect smoothed maps to writer (if smoothing enabled)
        if dsc_config.smooth_fwhm > 0:
            workflow.connect([
                (dsc_wf, dsc_writer, [
                    ("outputnode.smooth_hemodynamic", "smooth_hemodynamic"),
                    ("outputnode.smooth_auc", "smooth_auc"),
                ]),
            ])

        # -------------------------------------------------------------------
        # DSC: connect writer to unified report barrier
        # -------------------------------------------------------------------
        workflow.connect([
            (dsc_writer, report_barrier, [("qc_json", "dsc_done")]),
        ])

        # Connect DSC QC plots to writer (writer saves to figures/ dir)
        workflow.connect([
            (dsc_wf, dsc_writer, [
                ("outputnode.brain_mask_filmstrip", "brain_mask_filmstrip"),
                ("outputnode.dsc_preprocessing_qc_plot", "dsc_preprocessing_qc_plot"),
                ("outputnode.dsc_param_maps_qc_plot", "dsc_param_maps_qc_plot"),
                ("outputnode.dsc_fit_qc_plot", "dsc_fit_qc_plot"),
                ("outputnode.dsc_leakage_qc_plot", "dsc_leakage_qc_plot"),
            ]),
        ])

    # -------------------------------------------------------------------
    # DCE perfusion workflow (requires anat, uses DSC CBV if available)
    # -------------------------------------------------------------------
    if decision.run_dce:
        import logging as _logging

        _log = _logging.getLogger("pmriprep.workflows")

        # Auto-skip T1 mapping when no VFA data is available
        # (B1 mapping is also unnecessary without VFA)
        if not session_data.modalities.vfa and not dce_config.no_t1:
            _log.warning(
                "No VFA data found for sub-%s ses-%s; "
                "disabling T1 mapping (using field-strength default T1)",
                subject_id, session_id,
            )
            dce_config.no_t1 = True
            dce_config.no_b1 = True

        # Auto-skip B1 when no AFI data (VFA may still exist)
        if not session_data.modalities.afi and not dce_config.no_b1:
            _log.warning(
                "No AFI/B1+ data found for sub-%s ses-%s; "
                "disabling B1+ correction (using nominal flip angles)",
                subject_id, session_id,
            )
            dce_config.no_b1 = True

        dce_wf = init_dce_preproc_wf(
            bids_dir=bids_dir,
            subject_id=subject_id,
            session_id=session_id,
            config=dce_config,
            has_tissue_seg=bool(session_data.modalities.t1w_pre),
            name="dce_wf",
        )

        # Connect DCE file from inputnode
        workflow.connect([
            (inputnode, dce_wf, [("dce_file", "inputnode.dce_file")]),
        ])

        # Read acquisition parameters from DCE BIDS sidecar JSON
        if session_data.modalities.dce:
            dce_sidecar = _read_bids_sidecar(session_data.modalities.dce)
            if dce_sidecar["te"] is not None:
                dce_wf.inputs.inputnode.te = dce_sidecar["te"]
            if dce_sidecar["tr"] is not None:
                dce_wf.inputs.inputnode.tr = dce_sidecar["tr"]
            # Volume TR for model auto-selection (temporal resolution)
            if dce_sidecar["volume_tr"] is not None:
                dce_wf.inputs.inputnode.volume_tr = dce_sidecar["volume_tr"]
            if dce_sidecar["flip_angle"] is not None:
                dce_wf.inputs.inputnode.flip_angle = dce_sidecar["flip_angle"]
            if dce_sidecar["field_strength"] is not None:
                dce_wf.inputs.inputnode.field_strength = dce_sidecar["field_strength"]

            # Set r1 relaxivity from contrast agent lookup or default
            # Agent-specific r1 (e.g., Gadovist=5.0 at 3T) improves AIF accuracy
            logger = logging.getLogger("pmriprep.workflows")
            if dce_sidecar["r1_relaxivity"] is not None:
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
                dce_wf.inputs.inputnode.r1_relaxivity = fallback_r1
                logger.warning(
                    "ContrastBolusAgent not found in BIDS JSON or unknown agent. "
                    f"Using default Gd-DTPA r1={fallback_r1:.1f} s^-1/mM "
                    f"at {dce_sidecar['field_strength']:.1f}T. "
                    "Provide ContrastBolusAgent in JSON sidecar for "
                    "agent-specific accuracy."
                )
            else:
                # Default to 3T value (3.8 s^-1/mM)
                dce_wf.inputs.inputnode.r1_relaxivity = 3.8
                logger.warning(
                    "ContrastBolusAgent and field strength not found in BIDS JSON. "
                    "Using default Gd-DTPA r1=3.8 s^-1/mM at 3.0T. "
                    "Provide ContrastBolusAgent in JSON sidecar for "
                    "agent-specific accuracy."
                )

            # Set agent-specific standard dose (e.g. 0.05 for Vueway)
            dce_wf.inputs.inputnode.agent_standard_dose = dce_sidecar["agent_standard_dose"]

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
                        ("outputnode.csf_probseg", "inputnode.csf_probseg"),
                    ]),
                ])
            # Connect FLAIR underlay if available
            if session_data.modalities.flair:
                workflow.connect([
                    (anat_wf, dce_wf, [("outputnode.flair_coreg", "inputnode.flair_in_t1w")]),
                ])
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
            if prev_anat["csf_probseg"]:
                dce_wf.inputs.inputnode.csf_probseg = prev_anat["csf_probseg"]
            if prev_anat["t1w_to_mni_xfm"]:
                dce_wf.inputs.inputnode.t1w_to_mni_xfm = prev_anat["t1w_to_mni_xfm"]
                _log.info("skip-anat (DCE): loaded t1w_to_mni_xfm from %s", prev_anat["t1w_to_mni_xfm"])
            if prev_anat["flair_coreg"]:
                dce_wf.inputs.inputnode.flair_in_t1w = prev_anat["flair_coreg"]
                _log.info("skip-anat (DCE): loaded flair_in_t1w from %s", prev_anat["flair_coreg"])

        # Connect lesion mask, tumor segmentation, and NAWM mask from hdglio if available
        # Also wire user-supplied masks from config (lesion_mask, nawm_mask, roi_mask)
        _dce_log = logging.getLogger("pmriprep.workflows")
        _dce_user_lesion = dce_config.lesion_mask
        _dce_user_nawm = dce_config.nawm_mask
        _dce_user_roi = dce_config.roi_mask

        if _dce_user_roi:
            dce_wf.inputs.inputnode.roi_mask = str(_dce_user_roi)
            _dce_log.info("DCE: user roi_mask=%s", _dce_user_roi)

        if hdglio_wf is not None:
            if _dce_user_lesion:
                # Merge HD-GLIO + user lesion masks
                from pmriprep.utils.mask_utils import DilateMask, MergeMasks

                dce_dilate_user_lesion = pe.Node(
                    DilateMask(radius_mm=15.0),
                    name="dce_dilate_user_lesion",
                )
                dce_dilate_user_lesion.inputs.mask_file = str(_dce_user_lesion)

                dce_merge_dilated = pe.Node(
                    MergeMasks(), name="dce_merge_dilated",
                )
                dce_merge_undilated = pe.Node(
                    MergeMasks(), name="dce_merge_undilated",
                )
                dce_merge_undilated.inputs.mask_b = str(_dce_user_lesion)

                workflow.connect([
                    (dce_dilate_user_lesion, dce_merge_dilated, [
                        ("dilated_mask", "mask_b"),
                    ]),
                    (hdglio_wf, dce_merge_dilated, [
                        ("outputnode.dilated_lesion_mask", "mask_a"),
                    ]),
                    (hdglio_wf, dce_merge_undilated, [
                        ("outputnode.whole_tumor_mask", "mask_a"),
                    ]),
                    (dce_merge_dilated, dce_wf, [
                        ("merged_mask", "inputnode.lesion_mask"),
                    ]),
                    (dce_merge_undilated, dce_wf, [
                        ("merged_mask", "inputnode.whole_tumor_mask"),
                    ]),
                    (hdglio_wf, dce_wf, [
                        ("outputnode.enhancing_mask", "inputnode.enhancing_mask"),
                        ("outputnode.segmentation", "inputnode.hdglio_segmentation"),
                        ("outputnode.nawm_mask", "inputnode.nawm_mask"),
                    ]),
                ])
                _dce_log.info(
                    "DCE: merging user lesion_mask=%s with HD-GLIO masks",
                    _dce_user_lesion,
                )
            else:
                # HD-GLIO only — current behavior
                workflow.connect([
                    (hdglio_wf, dce_wf, [
                        ("outputnode.dilated_lesion_mask", "inputnode.lesion_mask"),
                        ("outputnode.whole_tumor_mask", "inputnode.whole_tumor_mask"),
                        ("outputnode.enhancing_mask", "inputnode.enhancing_mask"),
                        ("outputnode.segmentation", "inputnode.hdglio_segmentation"),
                        ("outputnode.nawm_mask", "inputnode.nawm_mask"),
                    ]),
                ])
        elif work_dir is not None and not decision.run_hdglio:
            # --skip-tumor-seg: load previously-computed hdglio outputs
            prev_hdglio = _find_previous_hdglio_outputs(
                Path(work_dir), subject_id, session_id,
            )
            if _dce_user_lesion and prev_hdglio["dilated_lesion_mask"]:
                # Merge user lesion with loaded HD-GLIO masks
                from pmriprep.utils.mask_utils import DilateMask, MergeMasks

                dce_dilate_user_lesion = pe.Node(
                    DilateMask(radius_mm=15.0),
                    name="dce_dilate_user_lesion",
                )
                dce_dilate_user_lesion.inputs.mask_file = str(_dce_user_lesion)

                dce_merge_dilated = pe.Node(
                    MergeMasks(), name="dce_merge_dilated",
                )
                dce_merge_dilated.inputs.mask_a = prev_hdglio["dilated_lesion_mask"]

                dce_merge_undilated = pe.Node(
                    MergeMasks(), name="dce_merge_undilated",
                )
                dce_merge_undilated.inputs.mask_b = str(_dce_user_lesion)
                if prev_hdglio["whole_tumor_mask"]:
                    dce_merge_undilated.inputs.mask_a = prev_hdglio["whole_tumor_mask"]

                workflow.connect([
                    (dce_dilate_user_lesion, dce_merge_dilated, [
                        ("dilated_mask", "mask_b"),
                    ]),
                    (dce_merge_dilated, dce_wf, [
                        ("merged_mask", "inputnode.lesion_mask"),
                    ]),
                    (dce_merge_undilated, dce_wf, [
                        ("merged_mask", "inputnode.whole_tumor_mask"),
                    ]),
                ])
                _dce_log.info(
                    "DCE skip-tumor-seg: merging user lesion_mask=%s with loaded HD-GLIO masks",
                    _dce_user_lesion,
                )
            else:
                if prev_hdglio["dilated_lesion_mask"]:
                    dce_wf.inputs.inputnode.lesion_mask = prev_hdglio["dilated_lesion_mask"]
                    _dce_log.info("skip-tumor-seg (DCE): loaded lesion_mask from %s", prev_hdglio["dilated_lesion_mask"])
                if prev_hdglio["whole_tumor_mask"]:
                    dce_wf.inputs.inputnode.whole_tumor_mask = prev_hdglio["whole_tumor_mask"]
                    _dce_log.info("skip-tumor-seg (DCE): loaded whole_tumor_mask from %s", prev_hdglio["whole_tumor_mask"])

            if prev_hdglio.get("enhancing_mask"):
                dce_wf.inputs.inputnode.enhancing_mask = prev_hdglio["enhancing_mask"]
                _dce_log.info("skip-tumor-seg (DCE): loaded enhancing_mask from %s", prev_hdglio["enhancing_mask"])
            if prev_hdglio["segmentation"]:
                dce_wf.inputs.inputnode.hdglio_segmentation = prev_hdglio["segmentation"]
                _dce_log.info("skip-tumor-seg (DCE): loaded hdglio_segmentation from %s", prev_hdglio["segmentation"])
            if prev_hdglio["nawm_mask"]:
                dce_wf.inputs.inputnode.nawm_mask = prev_hdglio["nawm_mask"]
                _dce_log.info("skip-tumor-seg (DCE): loaded nawm_mask from %s", prev_hdglio["nawm_mask"])
        else:
            # No HD-GLIO — user-supplied masks only
            if _dce_user_lesion:
                from pmriprep.utils.mask_utils import DilateMask

                dce_dilate_user_lesion = pe.Node(
                    DilateMask(radius_mm=15.0),
                    name="dce_dilate_user_lesion",
                )
                dce_dilate_user_lesion.inputs.mask_file = str(_dce_user_lesion)
                workflow.connect([
                    (dce_dilate_user_lesion, dce_wf, [
                        ("dilated_mask", "inputnode.lesion_mask"),
                    ]),
                ])
                dce_wf.inputs.inputnode.whole_tumor_mask = str(_dce_user_lesion)
                _dce_log.info("DCE: user lesion_mask=%s (dilated→lesion_mask, original→whole_tumor_mask)", _dce_user_lesion)

            if _dce_user_nawm:
                dce_wf.inputs.inputnode.nawm_mask = str(_dce_user_nawm)
                _dce_log.info("DCE: user nawm_mask=%s", _dce_user_nawm)

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
        if session_data.modalities.afi:
            afi_mode = session_data.modalities.afi_mode

            if afi_mode == "4d" and session_data.modalities.afi_4d_file:
                # 4D mode: single file with 2 volumes
                dce_wf.inputs.inputnode.afi_4d_file = str(
                    session_data.modalities.afi_4d_file
                )
                # Pass TR overrides from discovery if available
                if session_data.modalities.afi_tr1_ms is not None:
                    dce_wf.inputs.inputnode.afi_tr1 = session_data.modalities.afi_tr1_ms
                if session_data.modalities.afi_tr2_ms is not None:
                    dce_wf.inputs.inputnode.afi_tr2 = session_data.modalities.afi_tr2_ms
            else:
                # 3D mode: use TR-classified file paths from discovery
                mods = session_data.modalities
                tr1_file = mods.afi_tr1_file or (mods.afi[0] if len(mods.afi) >= 2 else None)
                tr2_file = mods.afi_tr2_file or (mods.afi[1] if len(mods.afi) >= 2 else None)
                if tr1_file and tr2_file:
                    dce_wf.inputs.inputnode.afi_tr1_file = str(tr1_file)
                    dce_wf.inputs.inputnode.afi_tr2_file = str(tr2_file)
                    # Use pre-read TR values from discovery; fall back to sidecar read
                    if mods.afi_tr1_ms is not None and mods.afi_tr2_ms is not None:
                        dce_wf.inputs.inputnode.afi_tr1 = mods.afi_tr1_ms
                        dce_wf.inputs.inputnode.afi_tr2 = mods.afi_tr2_ms
                    else:
                        afi_sidecar_1 = _read_bids_sidecar(tr1_file)
                        afi_sidecar_2 = _read_bids_sidecar(tr2_file)
                        if afi_sidecar_1["tr"] is not None and afi_sidecar_2["tr"] is not None:
                            dce_wf.inputs.inputnode.afi_tr1 = afi_sidecar_1["tr"] * 1000
                            dce_wf.inputs.inputnode.afi_tr2 = afi_sidecar_2["tr"] * 1000

            # Wire AFI nominal flip angle from BIDS sidecar (default 60 in B1Mapping)
            if session_data.modalities.afi_flip_angle is not None:
                dce_wf.inputs.inputnode.afi_flip_angle = session_data.modalities.afi_flip_angle

        # Connect DWI for tortuosity if available
        if session_data.modalities.dwi:
            workflow.connect([
                (inputnode, dce_wf, [("dwi_file", "inputnode.dwi_file")]),
            ])
            # Gap 4: Extract b-value from BIDS .bval sidecar for tortuosity correction
            import logging as _logging
            _log = _logging.getLogger("pmriprep.workflows")
            dwi_path = session_data.modalities.dwi
            bval_path = str(dwi_path).replace(".nii.gz", ".bval").replace(".nii", ".bval")
            if Path(bval_path).exists():
                try:
                    import numpy as _np
                    bvals = _np.loadtxt(bval_path)
                    # Select high-b value in range 800-1200 s/mm²
                    high_b = bvals[(bvals >= 800) & (bvals <= 1200)]
                    if len(high_b) > 0:
                        dce_wf.inputs.inputnode.dwi_bval = float(high_b[0])
                        _log.info(
                            "DWI: extracted b-value %.0f s/mm² from %s",
                            high_b[0], bval_path
                        )
                    else:
                        _log.warning(
                            "DWI: no b-value in range 800-1200 s/mm² found in %s",
                            bval_path
                        )
                except Exception as exc:
                    _log.warning("DWI: failed to read b-value from %s: %s", bval_path, exc)

        # Connect DSC CBV for vp constraint in DCE PK modeling.
        # Models with vp (extended_tofts, 2cxm) use DSC CBV as a fitting constraint.
        # Other models with vp (patlak) fit freely, cross-validate post-hoc.
        if dsc_wf is not None:
            # Fitting constraint: models with vp parameter (or auto, which may resolve to one)
            if dce_config.model in ("extended_tofts", "2cxm", "auto"):
                workflow.connect([
                    (dsc_wf, dce_wf, [
                        # Route CBV through DCE inputnode → resample_dsc_cbv node
                        # (DSC native 128x128 must be resampled to DCE native 256x256)
                        ("outputnode.rel_cbv_map", "inputnode.dsc_cbv_file"),
                        ("outputnode.dsc_to_t1w_xfm", "inputnode.dsc_to_t1w_xfm"),
                        # AIF QC is a JSON file, no spatial resampling needed
                        ("outputnode.aif_qc", "pk_modeling.dsc_aif_qc"),
                    ]),
                ])

            # Cross-validation: Patlak fits vp freely and cross-validates post-hoc.
            # CBV is resampled inside DCE workflow via resample_dsc_cbv node.
            # Only need to connect if not already connected by the constraint block above.
            if dce_config.model == "patlak":
                workflow.connect([
                    (dsc_wf, dce_wf, [
                        ("outputnode.rel_cbv_map", "inputnode.dsc_cbv_file"),
                        ("outputnode.dsc_to_t1w_xfm", "inputnode.dsc_to_t1w_xfm"),
                    ]),
                ])

        elif dce_config.model in ("extended_tofts", "2cxm", "auto", "patlak"):
            # DSC skipped but DCE needs CBV: look for previous DSC outputs
            prev_dsc = _find_previous_dsc_outputs(
                Path(output_dir), subject_id, session_id,
            )
            if prev_dsc["rel_cbv_map"] and prev_dsc["dsc_to_t1w_xfm"]:
                dce_wf.inputs.inputnode.dsc_cbv_file = prev_dsc["rel_cbv_map"]
                dce_wf.inputs.inputnode.dsc_to_t1w_xfm = prev_dsc["dsc_to_t1w_xfm"]
                _log.info(
                    "skip-dsc: loaded DSC CBV from %s", prev_dsc["rel_cbv_map"]
                )
                if prev_dsc["aif_qc"]:
                    dce_wf.inputs.pk_modeling.dsc_aif_qc = prev_dsc["aif_qc"]
                else:
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
        # DCEDerivativeWriter: Write BIDS-compliant DCE outputs
        # -------------------------------------------------------------------
        from pmriprep.interfaces.dce_outputs import DCEDerivativeWriter
        import dataclasses

        dce_config_dict = dataclasses.asdict(dce_config)
        # Convert Path objects to strings for JSON serialization
        for k, v in dce_config_dict.items():
            if isinstance(v, Path):
                dce_config_dict[k] = str(v)

        dce_writer = pe.Node(
            DCEDerivativeWriter(
                output_dir=str(output_dir),
                subject_id=subject_id,
                session_id=session_id or "",
                bids_dir=str(bids_dir),
                pk_model=dce_config.model,
                config_dict=dce_config_dict,
            ),
            name="dce_writer",
        )

        # Connect DCE outputs to writer
        workflow.connect([
            (inputnode, dce_writer, [("dce_file", "source_dce")]),
            (dce_wf, dce_writer, [
                # Core PK parameter maps
                ("outputnode.ktrans_map", "ktrans_map"),
                ("outputnode.ve_map", "ve_map"),
                ("outputnode.kep_map", "kep_map"),
                ("outputnode.vp_map", "vp_map"),
                # Model-free maps
                ("outputnode.iauc60_map", "iauc60_map"),
                ("outputnode.iauc90_map", "iauc90_map"),
                ("outputnode.iauc120_map", "iauc120_map"),
                ("outputnode.peak_enhancement_map", "peak_enhancement_map"),
                ("outputnode.time_to_peak_map", "time_to_peak_map"),
                ("outputnode.ser_map", "ser_map"),
                # QC maps
                ("outputnode.tsnr_map", "tsnr_map"),
                ("outputnode.qc_mask", "qc_mask"),
                ("outputnode.r2_map", "r2_map"),
                ("outputnode.quality_flag_map", "quality_flag_map"),
                # T1/B1 mapping
                ("outputnode.t1_map", "t1_map"),
                ("outputnode.b1_map", "b1_map"),
                # Motion-corrected data
                ("outputnode.dce_moco", "dce_moco"),
                ("outputnode.dce_mean", "dce_mean"),
                ("outputnode.motion_params", "motion_params"),
                # AIF outputs
                ("outputnode.aif_curve", "aif_curve"),
                ("outputnode.aif_mask", "aif_mask"),
                # Registration
                ("outputnode.dce_to_t1w_xfm", "dce_to_t1w_xfm"),
                # Uncertainty maps (Bayesian)
                ("outputnode.ktrans_uncertainty", "ktrans_uncertainty"),
                ("outputnode.ve_uncertainty", "ve_uncertainty"),
                ("outputnode.vp_uncertainty", "vp_uncertainty"),
                ("outputnode.fp_uncertainty", "fp_uncertainty"),
                ("outputnode.ps_uncertainty", "ps_uncertainty"),
                # Bayesian CV maps
                ("outputnode.ktrans_cv", "ktrans_cv"),
                ("outputnode.ve_cv", "ve_cv"),
                ("outputnode.vp_cv", "vp_cv"),
                # Bayesian CI maps
                ("outputnode.ktrans_ci95_lower", "ktrans_ci95_lower"),
                ("outputnode.ktrans_ci95_upper", "ktrans_ci95_upper"),
                ("outputnode.ve_ci95_lower", "ve_ci95_lower"),
                ("outputnode.ve_ci95_upper", "ve_ci95_upper"),
                ("outputnode.vp_ci95_lower", "vp_ci95_lower"),
                ("outputnode.vp_ci95_upper", "vp_ci95_upper"),
                # Tortuosity-corrected Ktrans
                ("outputnode.ktrans_tortuosity_corrected", "ktrans_tortuosity_corrected"),
                # Normalized maps (NAWM-normalized)
                ("outputnode.nktrans_map", "nktrans_map"),
                ("outputnode.niauc60_map", "niauc60_map"),
                # Smoothed maps
                ("outputnode.smooth_pk_maps", "smooth_pk_maps"),
                ("outputnode.smooth_model_free_maps", "smooth_model_free_maps"),
                ("outputnode.smooth_uncertainty_maps", "smooth_uncertainty_maps"),
                ("outputnode.smooth_tortuosity_maps", "smooth_tortuosity_maps"),
                # QC metrics
                ("outputnode.qc_metrics", "qc_metrics"),
                # QC plots
                ("outputnode.aif_curve_qc_plot", "aif_curve_plot"),
                ("outputnode.aif_location_qc_plot", "aif_location_plot"),
                ("outputnode.registration_qc_filmstrip", "registration_filmstrip"),
                ("outputnode.motion_qc_plot", "motion_qc_plot"),
                # Brain mask filmstrip
                ("outputnode.brain_mask_filmstrip", "brain_mask_filmstrip"),
                # DCE QC plots
                ("outputnode.preprocessing_qc_plot", "preprocessing_qc_plot"),
                ("outputnode.pk_modeling_qc_plot", "pk_modeling_qc_plot"),
                ("outputnode.pk_param_maps_qc_plot", "pk_param_maps_qc_plot"),
                # B1/T1 QC plots
                ("outputnode.b1_qc_plot", "b1_qc_plot"),
                ("outputnode.t1_qc_plot", "t1_qc_plot"),
                # Structural underlays
                ("outputnode.t1w_in_dce", "t1w_in_dce"),
                ("outputnode.flair_in_dce", "flair_in_dce"),
                # Lesion statistics
                ("outputnode.lesion_stats_json", "lesion_stats_json"),
                ("outputnode.lesion_stats_txt", "lesion_stats_txt"),
                ("outputnode.lesion_mask_dce", "lesion_mask_dce"),
                # ROI statistics
                ("outputnode.roi_stats_json", "roi_stats_json"),
                ("outputnode.roi_stats_txt", "roi_stats_txt"),
                ("outputnode.roi_mask_dce", "roi_mask_dce"),
                # Anatomical masks in DCE space
                ("outputnode.brain_mask", "brain_mask"),
                ("outputnode.nawm_mask", "nawm_mask"),
            ]),
        ])

        # -------------------------------------------------------------------
        # DCE: connect writer to unified report barrier
        # -------------------------------------------------------------------
        workflow.connect([
            (dce_writer, report_barrier, [("qc_json", "dce_done")]),
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

    # HD-GLIO
    if decision.run_hdglio:
        lines.append("HD-GLIO tumor segmentation: WILL RUN")
    else:
        lines.append(f"HD-GLIO tumor segmentation: SKIPPED ({decision.hdglio_reason})")

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
