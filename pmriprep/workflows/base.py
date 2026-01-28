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
    # HD-GLIO requires anat pipeline for input preparation (coregistration,
    # skull-stripping without bias correction)
    if skip_flags.skip_tumor_seg:
        decision.run_hdglio = False
        decision.hdglio_reason = "Skipped by user (--skip-tumor-seg)"
    elif skip_flags.skip_anat:
        decision.run_hdglio = False
        decision.hdglio_reason = "Requires anatomical pipeline (--skip-anat disables it)"
    elif not session_data.pipelines.hdglio:
        decision.run_hdglio = False
        decision.hdglio_reason = session_data.pipelines.hdglio_reason
    else:
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

    # Normalize paths (resolve to absolute for DataSink)
    bids_dir = Path(bids_dir).resolve()
    output_dir = Path(output_dir).resolve()

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
                ]),
            ])

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

        # Connect lesion mask from hdglio if available
        if hdglio_wf is not None:
            workflow.connect([
                (hdglio_wf, dsc_wf, [
                    ("outputnode.dilated_lesion_mask", "inputnode.lesion_mask"),
                    ("outputnode.nawm_mask", "inputnode.nawm_mask"),
                ]),
            ])

        # Connect DSC outputs to master outputnode
        workflow.connect([
            (dsc_wf, outputnode, [
                ("outputnode.cbf_map", "dsc_cbf"),
                ("outputnode.cbv_map", "dsc_cbv"),
                ("outputnode.mtt_map", "dsc_mtt"),
                ("outputnode.tmax_map", "dsc_tmax"),
                ("outputnode.aif_qc", "dsc_aif_qc"),
                ("outputnode.qc_metrics", "dsc_qc"),
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
                ]),
            ])

        # Connect lesion mask from hdglio if available
        if hdglio_wf is not None:
            workflow.connect([
                (hdglio_wf, dce_wf, [
                    ("outputnode.dilated_lesion_mask", "inputnode.lesion_mask"),
                ]),
            ])

        # Connect VFA and AFI inputs if available
        if session_data.modalities.vfa:
            workflow.connect([
                (inputnode, dce_wf, [("vfa_files", "inputnode.vfa_files")]),
            ])
        if session_data.modalities.afi:
            # AFI typically has 2 files - connect first two if available
            afi_list = session_data.modalities.afi
            if len(afi_list) >= 2:
                workflow.connect([
                    (inputnode, dce_wf, [
                        (("afi_files", lambda x: x[0] if x else None), "inputnode.afi_tr1_file"),
                        (("afi_files", lambda x: x[1] if len(x) > 1 else None), "inputnode.afi_tr2_file"),
                    ]),
                ])

        # Connect DWI for tortuosity if available
        if session_data.modalities.dwi:
            workflow.connect([
                (inputnode, dce_wf, [("dwi_file", "inputnode.dwi_file")]),
            ])

        # Connect DSC CBV for vp constraint (US-063 QC)
        # This is why DCE runs after DSC when both are present
        if dsc_wf is not None:
            workflow.connect([
                (dsc_wf, dce_wf, [
                    ("outputnode.cbv_map", "inputnode.dsc_cbv_file"),
                    ("outputnode.aif_qc", "inputnode.dsc_aif_qc_file"),
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
