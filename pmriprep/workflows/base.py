"""Master workflow for PMRIPrep BIDS-App.

This module implements the master workflow that orchestrates all pipeline
components (anatomical, HD-GLIO, DSC, DCE) based on available data and
user-specified skip flags.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

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
        Dict with keys: t1w_preproc, brain_mask, wm_probseg, t1w_to_mni_xfm, flair_coreg.
        Values are file paths (str) or None if not found.
    """
    session_label = f"ses-{session_id}" if session_id else "no-session"
    wf_name = f"pmriprep_sub{subject_id}_{session_label}_wf"
    anat_dir = work_dir / wf_name / "anat_wf"

    outputs: dict[str, Optional[str]] = {
        "t1w_preproc": None,
        "brain_mask": None,
        "wm_probseg": None,
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
) -> dict[str, Optional[float]]:
    """Read acquisition parameters from the BIDS sidecar JSON.

    Looks for the JSON sidecar corresponding to a NIfTI file and extracts
    EchoTime, RepetitionTime, RepetitionTimeExcitation, and MagneticFieldStrength.

    For DCE 3D sequences, there are two different TRs:
    - Volume TR (RepetitionTime): Time between successive 3D volume acquisitions
      (~5-10 seconds). Used for: time vector, model auto-selection, AIF timing.
    - Pulse TR (RepetitionTimeExcitation): Time between RF excitations (~3-10 ms).
      Used for: SPGR signal equation in signal-to-concentration conversion.

    Args:
        nifti_path: Path to a NIfTI file (.nii.gz or .nii).

    Returns:
        Dict with keys: te, tr, volume_tr, pulse_tr, field_strength, flip_angle.
        - tr: Pulse TR for signal equation (defaults to pulse_tr)
        - volume_tr: Volume TR for temporal resolution (model selection, time vector)
        - pulse_tr: Pulse TR for SPGR signal equation
        Values are floats or None if not found in the sidecar.
    """
    import json

    result: dict[str, Optional[float]] = {
        "te": None,
        "tr": None,
        "volume_tr": None,
        "pulse_tr": None,
        "field_strength": None,
        "flip_angle": None,
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
        enhancing_mask, nawm_mask.
        Values are file paths (str) or None if not found.
    """
    session_label = f"ses-{session_id}" if session_id else "no-session"
    wf_name = f"pmriprep_sub{subject_id}_{session_label}_wf"
    hdglio_dir = work_dir / wf_name / "hdglio_wf"

    outputs: dict[str, Optional[str]] = {
        "dilated_lesion_mask": None,
        "whole_tumor_mask": None,
        "enhancing_mask": None,
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
    # Anatomical preprocessing workflow (runs first)
    # -------------------------------------------------------------------
    if decision.run_anat:
        anat_wf = init_anat_preproc_wf(
            bids_dir=bids_dir,
            subject_id=subject_id,
            session_id=session_id,
            config=anat_config,
            has_t1w_postcon=session_data.modalities.t1w_ce_gad is not None,
            has_t2w=session_data.modalities.t2w is not None,
            has_flair=session_data.modalities.flair is not None,
            has_pdw=session_data.modalities.pdw is not None,
            name="anat_wf",
        )

        # Connect inputs to anat workflow
        # Use t1w_pre if available, otherwise t1w_postcon (will be handled by anat workflow)
        if session_data.modalities.t1w_pre:
            workflow.connect([
                (inputnode, anat_wf, [("t1w_pre", "inputnode.t1w")]),
            ])
            if session_data.modalities.t1w_ce_gad:
                workflow.connect([
                    (inputnode, anat_wf, [("t1w_postcon", "inputnode.t1w_postcon")]),
                ])
        elif session_data.modalities.t1w_ce_gad:
            # Only post-contrast T1w available
            workflow.connect([
                (inputnode, anat_wf, [("t1w_postcon", "inputnode.t1w")]),
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
                ("outputnode.wm_probseg", "wm_probseg"),
                ("outputnode.gm_probseg", "gm_probseg"),
                ("outputnode.csf_probseg", "csf_probseg"),
                ("outputnode.qc_metrics", "anat_qc"),
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
                ("outputnode.wm_probseg", "wm_probseg"),
                ("outputnode.gm_probseg", "gm_probseg"),
                ("outputnode.csf_probseg", "csf_probseg"),
                ("outputnode.t1w_to_mni_xfm", "acpc_xfm"),
                ("outputnode.qc_metrics", "qc_metrics"),
            ]),
        ])

        # Connect source paths for JSON sidecar Sources field
        if session_data.modalities.t1w_pre:
            workflow.connect([
                (inputnode, anat_writer, [("t1w_pre", "source_t1w")]),
            ])

        # Connect optional modalities
        if session_data.modalities.t1w_ce_gad:
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

        # Connect anat outputs to DSC workflow
        if anat_wf is not None:
            workflow.connect([
                (anat_wf, dsc_wf, [
                    ("outputnode.t1w_preproc", "inputnode.t1w_reference"),
                    ("outputnode.brain_mask", "inputnode.brain_mask"),
                    ("outputnode.wm_probseg", "inputnode.wm_probseg"),
                    ("outputnode.t1w_to_mni_xfm", "inputnode.t1w_to_mni_xfm"),
                ]),
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

        # Connect lesion mask and enhancing mask from hdglio if available
        if hdglio_wf is not None:
            workflow.connect([
                (hdglio_wf, dsc_wf, [
                    ("outputnode.dilated_lesion_mask", "inputnode.lesion_mask"),
                    ("outputnode.whole_tumor_mask", "inputnode.whole_tumor_mask"),
                    ("outputnode.enhancing_mask", "inputnode.enhancing_mask"),
                    ("outputnode.nawm_mask", "inputnode.nawm_mask"),
                ]),
            ])
        elif work_dir is not None and not decision.run_hdglio:
            # --skip-tumor-seg: load previously-computed hdglio outputs
            import logging as _logging
            _log = _logging.getLogger("pmriprep.workflows")
            prev_hdglio = _find_previous_hdglio_outputs(
                Path(work_dir), subject_id, session_id,
            )
            if prev_hdglio["dilated_lesion_mask"]:
                dsc_wf.inputs.inputnode.lesion_mask = prev_hdglio["dilated_lesion_mask"]
                _log.info("skip-tumor-seg: loaded lesion_mask from %s", prev_hdglio["dilated_lesion_mask"])
            if prev_hdglio["whole_tumor_mask"]:
                dsc_wf.inputs.inputnode.whole_tumor_mask = prev_hdglio["whole_tumor_mask"]
                _log.info("skip-tumor-seg: loaded whole_tumor_mask from %s", prev_hdglio["whole_tumor_mask"])
            if prev_hdglio["enhancing_mask"]:
                dsc_wf.inputs.inputnode.enhancing_mask = prev_hdglio["enhancing_mask"]
                _log.info("skip-tumor-seg: loaded enhancing_mask from %s", prev_hdglio["enhancing_mask"])
            if prev_hdglio["nawm_mask"]:
                dsc_wf.inputs.inputnode.nawm_mask = prev_hdglio["nawm_mask"]
                _log.info("skip-tumor-seg: loaded nawm_mask from %s", prev_hdglio["nawm_mask"])

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
                ("outputnode.cnr_map", "cnr_map"),
                ("outputnode.auc_diff_map", "auc_diff_map"),
                ("outputnode.snr_map", "snr_map"),
                ("outputnode.tsnr_map", "tsnr_map"),
                # Relative maps
                ("outputnode.rel_cbf_map", "rel_cbf_map"),
                ("outputnode.rel_cbv_map", "rel_cbv_map"),
                ("outputnode.rel_auc_raw_map", "rel_auc_raw_map"),
                ("outputnode.rel_auc_corr_map", "rel_auc_corr_map"),
                # Absolute maps (may be undefined if gating failed)
                ("outputnode.abs_cbf_map", "abs_cbf_map"),
                ("outputnode.abs_cbv_map", "abs_cbv_map"),
                ("outputnode.abs_auc_raw_map", "abs_auc_raw_map"),
                ("outputnode.abs_auc_corr_map", "abs_auc_corr_map"),
                # Normalized maps
                ("outputnode.ncbv_map", "ncbv_map"),
                ("outputnode.ncbf_map", "ncbf_map"),
                # Leakage correction
                ("outputnode.k2_map", "k2_map"),
                # Motion-corrected data
                ("outputnode.dsc_moco", "dsc_moco"),
                ("outputnode.dsc_mean", "dsc_mean"),
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
                # QC metrics
                ("outputnode.qc_metrics", "qc_metrics"),
                # Lesion statistics
                ("outputnode.lesion_stats_json", "lesion_stats_json"),
                ("outputnode.lesion_stats_txt", "lesion_stats_txt"),
                ("outputnode.lesion_mask_dsc", "lesion_mask_dsc"),
            ]),
            (inputnode, dsc_writer, [("dsc_file", "source_dsc")]),
        ])

        # Connect smoothed maps to writer (if smoothing enabled)
        if dsc_config.smooth_fwhm > 0:
            workflow.connect([
                (dsc_wf, dsc_writer, [
                    ("outputnode.smooth_quant_indep", "smooth_quant_indep"),
                    ("outputnode.smooth_rel", "smooth_rel"),
                    ("outputnode.smooth_abs", "smooth_abs"),
                ]),
            ])

    # -------------------------------------------------------------------
    # DCE perfusion workflow (requires anat, uses DSC CBV if available)
    # -------------------------------------------------------------------
    if decision.run_dce:
        dce_wf = init_dce_preproc_wf(
            bids_dir=bids_dir,
            subject_id=subject_id,
            session_id=session_id,
            config=dce_config,
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
                # Derive r1_relaxivity from field_strength (standard Gadolinium contrast)
                from pmriprep.utils.dce_constants import get_r1_relaxivity
                dce_wf.inputs.inputnode.r1_relaxivity = get_r1_relaxivity(
                    dce_sidecar["field_strength"]
                )
            else:
                # Default to 3T value (3.8 s^-1/mM)
                dce_wf.inputs.inputnode.r1_relaxivity = 3.8

        # Connect anat outputs to DCE workflow
        if anat_wf is not None:
            # Use post-contrast T1w as reference if available, otherwise preprocessed T1w
            if session_data.modalities.t1w_ce_gad:
                workflow.connect([
                    (inputnode, dce_wf, [("t1w_postcon", "inputnode.t1w_reference")]),
                ])
            else:
                workflow.connect([
                    (anat_wf, dce_wf, [("outputnode.t1w_preproc", "inputnode.t1w_reference")]),
                ])
            workflow.connect([
                (anat_wf, dce_wf, [
                    ("outputnode.brain_mask", "inputnode.brain_mask"),
                    ("outputnode.wm_probseg", "inputnode.wm_probseg"),
                    ("outputnode.csf_probseg", "inputnode.csf_probseg"),
                    # Pass T1w→MNI transform for atlas-guided AIF (Gap 6)
                    ("outputnode.t1w_to_mni_xfm", "inputnode.t1w_to_mni_xfm"),
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
            if session_data.modalities.t1w_ce_gad:
                # Post-contrast T1w is set from inputnode directly
                workflow.connect([
                    (inputnode, dce_wf, [("t1w_postcon", "inputnode.t1w_reference")]),
                ])
            elif prev_anat["t1w_preproc"]:
                dce_wf.inputs.inputnode.t1w_reference = prev_anat["t1w_preproc"]
                _log.info("skip-anat (DCE): loaded t1w_reference from %s", prev_anat["t1w_preproc"])
            if prev_anat["brain_mask"]:
                dce_wf.inputs.inputnode.brain_mask = prev_anat["brain_mask"]
            if prev_anat["wm_probseg"]:
                dce_wf.inputs.inputnode.wm_probseg = prev_anat["wm_probseg"]
            if prev_anat["t1w_to_mni_xfm"]:
                dce_wf.inputs.inputnode.t1w_to_mni_xfm = prev_anat["t1w_to_mni_xfm"]
                _log.info("skip-anat (DCE): loaded t1w_to_mni_xfm from %s", prev_anat["t1w_to_mni_xfm"])
            if prev_anat["flair_coreg"]:
                dce_wf.inputs.inputnode.flair_in_t1w = prev_anat["flair_coreg"]
                _log.info("skip-anat (DCE): loaded flair_in_t1w from %s", prev_anat["flair_coreg"])

        # Connect lesion mask from hdglio if available
        if hdglio_wf is not None:
            workflow.connect([
                (hdglio_wf, dce_wf, [
                    ("outputnode.dilated_lesion_mask", "inputnode.lesion_mask"),
                ]),
            ])
        elif work_dir is not None and not decision.run_hdglio:
            # --skip-tumor-seg: load previously-computed hdglio outputs
            import logging as _logging
            _log = _logging.getLogger("pmriprep.workflows")
            prev_hdglio = _find_previous_hdglio_outputs(
                Path(work_dir), subject_id, session_id,
            )
            if prev_hdglio["dilated_lesion_mask"]:
                dce_wf.inputs.inputnode.lesion_mask = prev_hdglio["dilated_lesion_mask"]
                _log.info("skip-tumor-seg (DCE): loaded lesion_mask from %s", prev_hdglio["dilated_lesion_mask"])

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
                # 3D mode: separate files (existing behavior)
                afi_list = session_data.modalities.afi
                if len(afi_list) >= 2:
                    dce_wf.inputs.inputnode.afi_tr1_file = str(afi_list[0])
                    dce_wf.inputs.inputnode.afi_tr2_file = str(afi_list[1])
                    # Read AFI TR values from BIDS sidecar JSONs
                    # B1Mapping expects TR in ms; BIDS stores RepetitionTime in s
                    afi_sidecar_1 = _read_bids_sidecar(afi_list[0])
                    afi_sidecar_2 = _read_bids_sidecar(afi_list[1])
                    if afi_sidecar_1["tr"] is not None and afi_sidecar_2["tr"] is not None:
                        dce_wf.inputs.inputnode.afi_tr1 = afi_sidecar_1["tr"] * 1000
                        dce_wf.inputs.inputnode.afi_tr2 = afi_sidecar_2["tr"] * 1000

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
        # Connect directly to internal nodes since these are optional inputs.
        if dsc_wf is not None:
            workflow.connect([
                (dsc_wf, dce_wf, [
                    ("outputnode.rel_cbv_map", "pk_modeling.dsc_cbv_file"),
                    ("outputnode.aif_qc", "pk_modeling.dsc_aif_qc_file"),
                    ("outputnode.rel_cbv_map", "dsc_dce_validation.dsc_cbv_file"),
                ]),
            ])

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
            (dce_wf, dce_writer, [
                # Core PK parameter maps
                ("outputnode.ktrans_map", "ktrans_map"),
                ("outputnode.ve_map", "ve_map"),
                ("outputnode.kep_map", "kep_map"),
                ("outputnode.vp_map", "vp_map"),
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
                # QC metrics
                ("outputnode.qc_metrics", "qc_metrics"),
                # Structural underlays
                ("outputnode.t1w_in_dce", "t1w_in_dce"),
                ("outputnode.flair_in_dce", "flair_in_dce"),
                # Lesion statistics
                ("outputnode.lesion_stats_json", "lesion_stats_json"),
                ("outputnode.lesion_stats_txt", "lesion_stats_txt"),
                ("outputnode.lesion_mask_dce", "lesion_mask_dce"),
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
