"""Smoke + sidecar-correctness tests for ``init_dce_datasinks_wf``."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _write_nii(path: Path, value: float = 0.0) -> str:
    nib.save(nib.Nifti1Image(np.full((4, 4, 4), value, dtype=np.float32), np.eye(4)), path)
    return str(path)


@pytest.fixture
def minimal_bids(tmp_path: Path) -> tuple[Path, str]:
    bids = tmp_path / "bids"
    perf = bids / "sub-01" / "ses-0001" / "perf"
    perf.mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        json.dumps({"Name": "T", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    src = perf / "sub-01_ses-0001_dce.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), src)
    return bids, str(src)


def test_dce_factory_smoke(tmp_path: Path, minimal_bids: tuple[Path, str]) -> None:
    """Mandatory PK output (Ktrans) writes correctly with required sidecar fields."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"

    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="auto", inference="vi",
        present_inputs=set(),  # only mandatory ktrans_map
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "ktrans.nii.gz")
    inp.qc_metrics = {}
    wf.run()

    perf_out = out / "pmriprep" / "sub-01" / "ses-0001" / "perf"
    ktrans = perf_out / "sub-01_ses-0001_desc-dceKtrans_stat.nii.gz"
    assert ktrans.exists(), f"Missing: {ktrans}"
    sidecar = json.loads(
        (perf_out / "sub-01_ses-0001_desc-dceKtrans_stat.json").read_text()
    )
    assert sidecar["Description"].startswith("Volume transfer constant")
    assert sidecar["Units"] == "1/min"
    assert sidecar["SpatialReference"].startswith("bids::sub-")
    assert "Resolution" not in sidecar
    assert isinstance(sidecar["VoxelSize"], list)
    assert sidecar["InputFunctionCompartment"] == "plasma"


def test_registered_vfa_t1map_writes_registration_provenance(
    tmp_path: Path, minimal_bids: tuple[Path, str],
) -> None:
    """The public T1map identifies its VFA sources and accepted registration."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    anat = bids / "sub-01" / "ses-0001" / "anat"
    anat.mkdir(parents=True)
    vfa = anat / "sub-01_ses-0001_flip-05_VFA.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.float32), np.eye(4)), vfa)
    out = tmp_path / "out"
    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="tofts", inference="nlls",
        present_inputs={"t1_map"}, t1_source_files=[str(vfa)],
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "ktrans.nii.gz")
    inp.t1_map = _write_nii(tmp_path / "t1_map.nii.gz")
    inp.qc_metrics = {
            "t1_mapping": {"t1_method": "wlls"},
            "signal_conversion": {
                "t1_source": "map",
                "vfa_to_dce_registration": {
                "qc_status": "PASS", "correlation": 0.8,
            },
        },
    }
    wf.run()

    anat_out = out / "pmriprep" / "sub-01" / "ses-0001" / "anat"
    sidecar = json.loads(
        (anat_out / "sub-01_ses-0001_space-dceref_T1map.json").read_text()
    )
    assert sidecar["Units"] == "ms"
    assert sidecar["RegistrationSource"].startswith("automatically_selected_VFA")
    assert sidecar["RegistrationQC"]["qc_status"] == "PASS"
    assert any("flip-05_VFA.nii.gz" in source for source in sidecar["Sources"])


def test_registered_vfa_t1map_skips_absent_registration_output(
    tmp_path: Path, minimal_bids: tuple[Path, str],
) -> None:
    """A rejected VFA registration omits T1map without failing the run."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"
    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="tofts", inference="nlls",
        present_inputs={"t1_map"},
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "ktrans.nii.gz")
    inp.t1_map = None
    inp.qc_metrics = {
        "signal_conversion": {
            "vfa_to_dce_registration": {"qc_status": "POOR"},
        },
    }

    wf.run()

    assert not list((out / "pmriprep").rglob("*T1map.nii.gz"))


def test_registered_vfa_t1map_skips_population_fallback(
    tmp_path: Path, minimal_bids: tuple[Path, str],
) -> None:
    """A uniform fallback used internally is not published as measured VFA T1."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"
    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="tofts", inference="nlls",
        present_inputs={"t1_map"},
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "ktrans.nii.gz")
    inp.t1_map = _write_nii(tmp_path / "population_t1.nii.gz")
    inp.qc_metrics = {
        "t1_mapping": {
            "t1_method": "nlls",
            "t1_correction": {"correction_applied": "fallback"},
        },
        "signal_conversion": {
            "vfa_to_dce_registration": {"qc_status": "PASS"},
        },
    }

    wf.run()

    assert not list((out / "pmriprep").rglob("*T1map.nii.gz"))


def test_dce_writer_recreates_deleted_cached_derivative(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """NLLS parameter and standard-error derivatives are rewritten despite cached nodes."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"
    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="tofts", inference="nlls",
        present_inputs={"ktrans_uncertainty"},
    )
    wf.base_dir = str(tmp_path / "work")
    wf.inputs.inputnode.source_dce = src
    wf.inputs.inputnode.ktrans_map = _write_nii(tmp_path / "ktrans.nii.gz", 0.25)
    wf.inputs.inputnode.ktrans_uncertainty = _write_nii(tmp_path / "se.nii.gz", 0.03125)
    wf.inputs.inputnode.config_dict = {"inference": "nlls"}
    wf.inputs.inputnode.qc_metrics = {}

    perf = out / "pmriprep" / "sub-01" / "ses-0001" / "perf"
    expected = {"dceKtrans": 0.25, "dceStandardErrorKtrans": 0.03125}
    for run in range(2):
        wf.run()
        for desc, value in expected.items():
            nii = perf / f"sub-01_ses-0001_desc-{desc}_stat.nii.gz"
            sidecar = nii.with_suffix("").with_suffix(".json")
            np.testing.assert_array_equal(nib.load(nii).get_fdata(), value)
            assert json.loads(sidecar.read_text())["Units"] == "1/min"
            if run == 0:
                nii.unlink()
                sidecar.unlink()


def test_dce_factory_with_bayesian_uncertainty(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Bayesian inference emits absolute uncertainty per PK parameter."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"

    outputs = {
        "ve_map": ("dceVe", 0.25, "1"),
        "vp_map": ("dceVp", 0.125, "1"),
        "ktrans_uncertainty": ("dcePosteriorSDKtrans", 0.03125, "1/min"),
        "ve_uncertainty": ("dcePosteriorSDVe", 0.0625, "1"),
        "vp_uncertainty": ("dcePosteriorSDVp", 0.015625, "1"),
        "r2_map": ("dceRsquared", 0.875, "1"),
    }
    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="extended_tofts", inference="vi",
        present_inputs=set(outputs),
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "k.nii.gz", 0.5)
    for attr, (_, value, _) in outputs.items():
        setattr(inp, attr, _write_nii(tmp_path / f"{attr}.nii.gz", value))
    inp.qc_metrics = {}
    inp.config_dict = {"inference": "vi"}
    wf.run()

    perf_out = out / "pmriprep" / "sub-01" / "ses-0001" / "perf"
    for desc, value, units in [("dceKtrans", 0.5, "1/min"), *outputs.values()]:
        nii = perf_out / f"sub-01_ses-0001_desc-{desc}_stat.nii.gz"
        np.testing.assert_array_equal(nib.load(nii).get_fdata(), value)
        sidecar = nii.with_suffix("").with_suffix(".json")
        assert json.loads(sidecar.read_text())["Units"] == units

    uncertainty_meta = json.loads(
        next(perf_out.glob("*desc-dcePosteriorSDKtrans_stat.json")).read_text()
    )
    assert uncertainty_meta["EstimationMethod"] == "vi"
    assert uncertainty_meta["InferenceMethod"] == "variational_bayes"
    assert (
        uncertainty_meta["UncertaintyEstimationMethod"]
        == "variational_posterior_covariance"
    )
    assert uncertainty_meta["ConditionalOnConvertedConcentration"] is True
    assert uncertainty_meta["ConditionalOnSuppliedAIF"] is True
    assert uncertainty_meta["UncertaintyStatistic"] == "posterior_standard_deviation"


@pytest.mark.parametrize("inference", ["nlls", "vi", "map"])
def test_uncertainty_sidecar_records_collected_qc(
    tmp_path: Path, minimal_bids: tuple[Path, str], inference: str,
) -> None:
    """Collected fitting QC reaches uncertainty sidecars through the writer."""
    from pmriprep.workflows.dce._helpers import _collect_qc_metrics
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf
    from pmriprep.models.dce.identification import identification_policy

    bids, src = minimal_bids
    out = tmp_path / "out"
    wf = init_dce_datasinks_wf(
        output_dir=str(out),
        bids_dir=str(bids),
        subject_id="01",
        session_id="0001",
        pk_model="tofts", inference=inference,
        present_inputs={"ktrans_uncertainty", "qc_mask"},
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "k.nii.gz")
    inp.ktrans_uncertainty = _write_nii(tmp_path / "k_std.nii.gz")
    inp.qc_mask = _write_nii(tmp_path / "qc.nii.gz", 1)
    inp.provenance_dict = {"PerfusionProcessing": {"PKModeling": {"model": "tofts"}}}
    inp.config_dict = {"inference": inference}
    fit_qc = {
        "qc_acceptance_policy": {"identification": identification_policy()},
        "identification_qc": {"basis": "selected_nlls", "by_model": {"tofts": {"evaluated": 10}}},
        "parameter_uncertainty": {
            "aif_inflation_input_factor": 2.0,
            "aif_parameter_scale_factors": {"Ktrans": 1.5, "ve": 1.0},
        },
        "map_uncertainty": {
            "n_retained_voxels": 10,
            "n_evaluated": 9,
            "n_laplace_hessian": 7,
            "n_hessian_unavailable": 2,
            "n_not_computed": 1,
            "hessian_unavailable_fraction": 2 / 9,
        }
    }
    inp.qc_metrics = _collect_qc_metrics(
        b1_qc=None, t1_qc=None, signal_conversion_qc=None,
        preprocess_qc=None, brain_extraction_qc=None, registration_qc=None,
        aif_qc=None, pk_qc=fit_qc, validation_result=None,
    )
    wf.run()

    qc_sidecar = next((out / "pmriprep").rglob("*desc-dceQc_mask.json"))
    qc_meta = json.loads(qc_sidecar.read_text())
    assert identification_policy()["description"] in qc_meta["Description"]
    assert qc_meta["PerfusionProcessing"]["PKModeling"]["QCAcceptancePolicy"] == fit_qc["qc_acceptance_policy"]
    assert qc_meta["PerfusionProcessing"]["PKModeling"]["IdentificationQC"] == fit_qc["identification_qc"]

    desc = "dceStandardErrorKtrans" if inference == "nlls" else "dcePosteriorSDKtrans"
    sidecar = next(
        (out / "pmriprep" / "sub-01" / "ses-0001" / "perf").glob(
            f"*desc-{desc}_stat.json"
        )
    )
    metadata = json.loads(sidecar.read_text())
    if inference == "map":
        accounting = metadata["MAPUncertaintyAccounting"]
        assert accounting["n_laplace_hessian"] == 7
        assert accounting["n_hessian_unavailable"] == 2
        assert accounting["n_not_computed"] == 1
    inflation = metadata["AIFUncertaintyInflation"]
    assert inflation["Applied"] is True
    assert inflation["InputFactor"] == 2.0
    assert inflation["ParameterScaleFactors"]["Ktrans"] == 1.5


@pytest.mark.parametrize("ser_complete", [False, True])
def test_dce_factory_model_free_no_model_entity(
    tmp_path: Path, minimal_bids: tuple[Path, str], ser_complete: bool,
) -> None:
    """Model-free sidecars retain collected timing QC and omit the model entity."""
    from pmriprep.workflows.dce._helpers import _collect_qc_metrics
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"

    present = {"iauc60_map", "peak_enhancement_map", "time_to_peak_map", "ser_map"}
    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="auto", inference="vi",
        present_inputs=present,
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "k.nii.gz")
    for attr in present:
        value = np.nan if attr == "ser_map" and not ser_complete else 1.0
        setattr(inp, attr, _write_nii(tmp_path / f"{attr}.nii.gz", value))
    available_duration = 330.0 if ser_complete else 180.0
    inp.config_dict = {"ser_early_time_s": 90.0, "ser_late_time_s": 300.0}
    fit_qc = {
        "iauc": {
            "reference_frame": 5,
            "reference_time_s": 25.0,
            "reference_selection": "earlier_of_baseline_count_and_detected_arrival",
            "configured_baseline_frames": 8,
            "effective_baseline_frames": 5,
            "detected_bolus_arrival_frame": 6,
            "available_post_reference_duration_s": available_duration,
            "windows": {
                "60": {
                    "requested_duration_s": 60.0,
                    "complete": True,
                    "cutoff_interpolated": False,
                }
            },
        },
        "ser": {
            "reference_frame": 5,
            "reference_time_s": 25.0,
            "reference_selection": "earlier_of_baseline_count_and_detected_arrival",
            "available_post_reference_duration_s": available_duration,
            "complete": ser_complete,
        },
    }
    inp.qc_metrics = _collect_qc_metrics(
        b1_qc=None, t1_qc=None, signal_conversion_qc=None,
        preprocess_qc=None, brain_extraction_qc=None, registration_qc=None,
        aif_qc=None, pk_qc=fit_qc, validation_result=None,
    )
    wf.run()

    perf_out = out / "pmriprep" / "sub-01" / "ses-0001" / "perf"
    # No model entity in filename
    assert (perf_out / "sub-01_ses-0001_desc-dceIAUC60_stat.nii.gz").exists()
    iauc_sidecar = json.loads(
        (perf_out / "sub-01_ses-0001_desc-dceIAUC60_stat.json").read_text()
    )
    assert iauc_sidecar["ReferencePoint"] == "selected global IAUC reference frame"
    assert iauc_sidecar["CutoffInterpolation"] == "linear"
    assert iauc_sidecar["IncompleteWindowValue"] == "NaN"
    assert iauc_sidecar["ValueDomain"] == "signed finite concentration integral, or NaN"
    assert iauc_sidecar["NegativeValueHandling"] == "retained"
    assert iauc_sidecar["ReferenceFrame"] == 5
    assert iauc_sidecar["ConfiguredBaselineFrames"] == 8
    assert iauc_sidecar["DetectedBolusArrivalFrame"] == 6
    assert iauc_sidecar["ResolvedWindow"]["complete"] is True
    assert (perf_out / "sub-01_ses-0001_desc-dcePeakEnhancement_stat.nii.gz").exists()
    peak_sidecar = json.loads(
        (perf_out / "sub-01_ses-0001_desc-dcePeakEnhancement_stat.json").read_text()
    )
    assert peak_sidecar["InputCurve"] == "motion_corrected_signal"
    assert peak_sidecar["PeakSearchStart"] == "first post-baseline frame"
    assert peak_sidecar["TemporalEstimator"] == (
        "raw maximum without smoothing or clipping"
    )
    assert "Residual motion or noise spikes" in peak_sidecar["Limitation"]
    ttp_sidecar = json.loads(
        (perf_out / "sub-01_ses-0001_desc-dceTtp_stat.json").read_text()
    )
    assert ttp_sidecar["ReferencePoint"] == "selected global contrast reference frame"
    assert ttp_sidecar["ReferenceFrame"] == 5
    assert ttp_sidecar["ReferenceTimeSeconds"] == 25.0
    assert ttp_sidecar["MinimumPeakEnhancementNoiseSD"] == 5.0
    assert "brain-median" in ttp_sidecar["NoiseEstimate"]
    assert "biased" in ttp_sidecar["Limitation"]
    ser_sidecar = json.loads(
        (perf_out / "sub-01_ses-0001_desc-dceSER_stat.json").read_text()
    )
    assert ser_sidecar["ReferenceFrame"] == 5
    assert ser_sidecar["ReferenceTimeSeconds"] == 25.0
    assert ser_sidecar["EarlySampleTimeSecondsAfterReference"] == 90.0
    assert ser_sidecar["LateSampleTimeSecondsAfterReference"] == 300.0
    assert ser_sidecar["AvailablePostReferenceDurationSeconds"] == available_duration
    assert ser_sidecar["CompleteTiming"] is ser_complete


def test_patlak_only_no_ve_kep(tmp_path: Path, minimal_bids: tuple[Path, str]) -> None:
    """Patlak runs do not populate ve/kep — factory must not crash on Undefined."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"

    # Patlak only emits Ktrans + vp; ve/kep are NOT in present_inputs.
    present = {"vp_map"}
    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="patlak", inference="vi",
        present_inputs=present,
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "k.nii.gz")
    inp.vp_map = _write_nii(tmp_path / "vp.nii.gz")
    inp.qc_metrics = {}
    wf.run()  # Must not crash

    perf_out = out / "pmriprep" / "sub-01" / "ses-0001" / "perf"
    assert (perf_out / "sub-01_ses-0001_desc-dceKtrans_stat.nii.gz").exists()
    assert (perf_out / "sub-01_ses-0001_desc-dceVp_stat.nii.gz").exists()
    assert not list(perf_out.glob("*Ve_stat.nii.gz"))
    assert not list(perf_out.glob("*Kep_stat.nii.gz"))


def test_patlak_absent_model_selection_plot_reaches_figures_writer(
    tmp_path: Path,
) -> None:
    """Fixed Patlak propagates ``None``, not an invalid empty file path."""
    import nipype.pipeline.engine as pe
    from nipype.interfaces import utility as niu

    from pmriprep.interfaces.qc_writers import DCEFiguresWriter
    from pmriprep.workflows.dce._helpers import _generate_dce_pk_maps_qc

    ktrans = tmp_path / "ktrans.nii.gz"
    vp = tmp_path / "vp.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(np.full((4, 4, 4), 0.1), affine), ktrans)
    nib.save(nib.Nifti1Image(np.full((4, 4, 4), 0.05), affine), vp)
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4)), affine), mask)
    aif = tmp_path / "aif.txt"
    motion = tmp_path / "motion.par"
    np.savetxt(aif, np.linspace(0.0, 1.0, 8))
    np.savetxt(motion, np.zeros((8, 6)))

    producer = pe.Node(
        niu.Function(
            input_names=[
                "ktrans_file", "vp_file", "brain_mask_file", "model_name",
                "subject_id", "session_id",
            ],
            output_names=[
                "pk_param_maps_qc_plot", "model_selection_qc_plot",
                "model_selection_regions",
            ],
            function=_generate_dce_pk_maps_qc,
        ),
        name="patlak_qc",
    )
    producer.inputs.ktrans_file = str(ktrans)
    producer.inputs.vp_file = str(vp)
    producer.inputs.brain_mask_file = str(mask)
    producer.inputs.model_name = "patlak"
    producer.inputs.subject_id = "01"
    producer.inputs.session_id = "0001"

    writer = pe.Node(DCEFiguresWriter(), name="figures_writer")
    writer.inputs.output_dir = str(tmp_path / "out")
    writer.inputs.subject_id = "01"
    writer.inputs.session_id = "0001"
    writer.inputs.aif_curve = str(aif)
    writer.inputs.motion_params = str(motion)
    writer.inputs.qc_metrics = {}

    workflow = pe.Workflow(name="patlak_writer", base_dir=str(tmp_path / "work"))
    workflow.connect([
        (producer, writer, [
            ("pk_param_maps_qc_plot", "pk_param_maps_qc_plot"),
            ("model_selection_qc_plot", "model_selection_qc_plot"),
        ]),
    ])
    workflow.run(plugin="Linear")

    session = tmp_path / "out" / "pmriprep" / "sub-01" / "ses-0001"
    assert (session / "perf" / "sub-01_ses-0001_desc-dceAif_timeseries.tsv").exists()
    assert (session / "figures" / "sub-01_ses-0001_desc-pkparammaps_dce.png").exists()
    assert not list((session / "figures").glob("*modelselection*"))


def test_pk_parameter_montage_receives_qc_mask_and_preserves_nan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parameter QC uses accepted voxels without making NaN look like zero."""
    from pmriprep.utils import dce_pk_qc_plots
    from pmriprep.workflows.dce._helpers import _generate_dce_pk_maps_qc

    affine = np.eye(4)
    ktrans = np.full((4, 4, 4), 0.1, dtype=np.float32)
    ve = np.full((4, 4, 4), 0.2, dtype=np.float32)
    ve[1, 1, 1] = np.nan
    brain = np.ones((4, 4, 4), dtype=np.uint8)
    qc_mask = np.zeros((4, 4, 4), dtype=np.uint8)
    qc_mask[1:3, 1:3, 1:3] = 1

    def _save(name: str, data: np.ndarray) -> str:
        path = tmp_path / name
        nib.save(nib.Nifti1Image(data, affine), path)
        return str(path)

    captured: dict[str, object] = {}

    def _capture_plot(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(tmp_path / "pk_parameter_maps_qc.png")

    monkeypatch.setattr(
        dce_pk_qc_plots,
        "generate_pk_parameter_maps_qc_plot",
        _capture_plot,
    )

    result = _generate_dce_pk_maps_qc(
        ktrans_file=_save("ktrans.nii.gz", ktrans),
        ve_file=_save("ve.nii.gz", ve),
        brain_mask_file=_save("brain.nii.gz", brain),
        qc_mask_file=_save("qc.nii.gz", qc_mask),
        model_name="auto",
    )

    assert result[0] == str(tmp_path / "pk_parameter_maps_qc.png")
    np.testing.assert_array_equal(captured["qc_mask"], qc_mask)
    param_maps = captured["param_maps"]
    assert isinstance(param_maps, dict)
    assert np.isnan(param_maps["ve"][1, 1, 1])


def test_2cxm_extraction_fraction_and_bayesian_uncertainty(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """2CXM emits dimensionless extraction fraction and Fp/PS posterior SDs."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"

    outputs = {
        "e_map": ("dceE", 0.25, "1"),
        "fp_uncertainty": ("dcePosteriorSDFp", 2.0, "mL/min/100mL"),
        "ps_uncertainty": ("dcePosteriorSDPS", 4.0, "mL/min/100mL"),
    }
    present = {"ve_map", "kep_map", "vp_map", *outputs}
    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="2cxm", inference="vi",
        present_inputs=present,
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "k.nii.gz")
    for attr in present:
        value = outputs[attr][1] if attr in outputs else 0.0
        setattr(inp, attr, _write_nii(tmp_path / f"{attr}.nii.gz", value))
    inp.qc_metrics = {}
    wf.run()

    perf_out = out / "pmriprep" / "sub-01" / "ses-0001" / "perf"
    assert not (
        perf_out / "sub-01_ses-0001_desc-dcePosteriorSDKtrans_stat.nii.gz"
    ).exists()
    for desc, value, units in outputs.values():
        nii = perf_out / f"sub-01_ses-0001_desc-{desc}_stat.nii.gz"
        np.testing.assert_array_equal(nib.load(nii).get_fdata(), value)
        sidecar = nii.with_suffix("").with_suffix(".json")
        assert json.loads(sidecar.read_text())["Units"] == units
    e_meta = json.loads((perf_out / "sub-01_ses-0001_desc-dceE_stat.json").read_text())
    assert "Extraction fraction" in e_meta["Description"]


def test_dce_factory_warnings_in_anchor_sidecar(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """``QCWarnings`` lands on ``desc-dcePreproc_dce.json`` from has_* flags."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"

    # Include dce_moco so the anchor preproc sidecar is emitted; everything
    # else absent so all six presence warnings fire.
    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="tofts", inference="vi",
        present_inputs={"dce_moco"},
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "k.nii.gz")
    inp.dce_moco = _write_nii(tmp_path / "moco.nii.gz")
    inp.qc_metrics = {"ok": True}
    wf.run()

    anchor = (
        out / "pmriprep" / "sub-01" / "ses-0001" / "perf"
        / "sub-01_ses-0001_desc-dcePreproc_dce.json"
    )
    assert anchor.exists(), f"Anchor sidecar missing: {anchor}"
    sidecar = json.loads(anchor.read_text())
    warnings = sidecar.get("QCWarnings", [])
    assert any("Brain mask" in w for w in warnings)
    assert any("registration transform" in w for w in warnings)
    assert "InputFunctionCompartment" not in sidecar

# ---------------------------------------------------------------------------
# Dual-T1w-underlay support: a session configured with both
# t1w_precon and t1w_postcon must produce TWO anat-side cross-modality
# outputs differing only by the BIDS ce-GAD entity.
# ---------------------------------------------------------------------------


def _build_dce_underlay_wf(
    bids: Path, src: str, out: Path, tmp_path: Path,
    present_inputs: set, work_subdir: str,
    *, t1w_in_dce: str | None = None,
    t1w_postcon_in_dce: str | None = None,
    t1w_subtraction_in_dce: str | None = None,
    flair_in_dce: str | None = None,
    flair_postcon_in_dce: str | None = None,
):
    """Helper: build the DCE datasinks workflow with the given underlays."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="tofts", inference="vi",
        present_inputs=present_inputs,
    )
    wf.base_dir = str(tmp_path / work_subdir)
    inp = wf.inputs.inputnode
    inp.source_dce = src
    anat = bids / "sub-01" / "ses-0001" / "anat"
    anat.mkdir(parents=True, exist_ok=True)
    for attr, name in (
        ("source_t1w", "sub-01_ses-0001_acq-mprage_run-1_T1w.nii.gz"),
        ("source_t1w_postcon", "sub-01_ses-0001_acq-mprage_ce-GAD_run-2_T1w.nii.gz"),
        ("source_flair", "sub-01_ses-0001_acq-3d_run-1_FLAIR.nii.gz"),
        ("source_flair_postcon", "sub-01_ses-0001_acq-3d_ce-GAD_run-2_FLAIR.nii.gz"),
        ("source_t2w", "sub-01_ses-0001_acq-tse_run-1_T2w.nii.gz"),
    ):
        source = anat / name
        if not source.exists():
            _write_nii(source)
        setattr(inp, attr, str(source))
    inp.ktrans_map = _write_nii(tmp_path / f"{work_subdir}-k.nii.gz")
    inp.qc_metrics = {}
    if t1w_in_dce is not None:
        inp.t1w_in_dce = t1w_in_dce
    if t1w_postcon_in_dce is not None:
        inp.t1w_postcon_in_dce = t1w_postcon_in_dce
    if t1w_subtraction_in_dce is not None:
        inp.t1w_subtraction_in_dce = t1w_subtraction_in_dce
    if flair_in_dce is not None:
        inp.flair_in_dce = flair_in_dce
    if flair_postcon_in_dce is not None:
        inp.flair_postcon_in_dce = flair_postcon_in_dce
    wf.run()


def test_dce_t1w_subtraction_underlay_lands_in_anat(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Subtraction underlay: anat/ with desc-subtraction, no ce-."""
    bids, src = minimal_bids
    out = tmp_path / "out"
    underlay = _write_nii(tmp_path / "t1w_subtraction_in_dce.nii.gz")

    _build_dce_underlay_wf(
        bids, src, out, tmp_path,
        present_inputs={"t1w_subtraction_in_dce"},
        work_subdir="work-subtraction",
        t1w_subtraction_in_dce=underlay,
    )

    anat_dir = out / "pmriprep" / "sub-01" / "ses-0001" / "anat"
    expected = anat_dir / "sub-01_ses-0001_acq-mprage_run-1_space-dceref_desc-subtraction_T1w.nii.gz"
    assert expected.exists(), (
        f"subtraction underlay missing at {expected}; "
        f"anat/ contains: {sorted(p.name for p in anat_dir.iterdir())}"
    )
    # desc distinguishes it from the plain precon underlay in the same space.
    assert not (anat_dir / "sub-01_ses-0001_space-dceref_T1w.nii.gz").exists()
    assert not (anat_dir / "sub-01_ses-0001_ce-GAD_space-dceref_desc-subtraction_T1w.nii.gz").exists()
    sidecar = json.loads(expected.with_suffix("").with_suffix(".json").read_text())
    assert "subtraction" in sidecar["Description"].lower()
    assert "SpatialReference" in sidecar
    assert sidecar["IntensityNormalizationApplied"] is False
    assert sidecar["Quantitative"] is False
    assert sidecar["Units"] == "arbitrary"
    assert len(sidecar["Sources"]) == 3


def test_dce_t1w_subtraction_absent_skips_cleanly(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Present-set without the subtraction key => no sink built, no error."""
    bids, src = minimal_bids
    out = tmp_path / "out"
    # present_inputs excludes t1w_subtraction_in_dce -> mandatory ktrans only.
    _build_dce_underlay_wf(
        bids, src, out, tmp_path,
        present_inputs=set(),
        work_subdir="work-sub-absent",
    )
    anat_dir = out / "pmriprep" / "sub-01" / "ses-0001" / "anat"
    subtraction = list(anat_dir.glob("*desc-subtraction*")) if anat_dir.exists() else []
    assert not subtraction, f"subtraction must not be emitted; got {subtraction}"


def test_dce_both_t1w_underlays(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Both underlays configured: two anat/ outputs differing by ce- entity."""
    bids, src = minimal_bids
    out = tmp_path / "out"
    pre = _write_nii(tmp_path / "t1w_in_dce.nii.gz")
    post = _write_nii(tmp_path / "t1w_postcon_in_dce.nii.gz")

    _build_dce_underlay_wf(
        bids, src, out, tmp_path,
        present_inputs={"t1w_in_dce", "t1w_postcon_in_dce"},
        work_subdir="work-both",
        t1w_in_dce=pre,
        t1w_postcon_in_dce=post,
    )

    anat_dir = out / "pmriprep" / "sub-01" / "ses-0001" / "anat"
    precon = anat_dir / "sub-01_ses-0001_acq-mprage_run-1_space-dceref_T1w.nii.gz"
    postcon = anat_dir / "sub-01_ses-0001_acq-mprage_ce-GAD_run-2_space-dceref_T1w.nii.gz"
    assert precon.exists() and postcon.exists(), (
        f"both underlays expected; anat/ contains: "
        f"{sorted(p.name for p in anat_dir.iterdir())}"
    )
    pre_meta = json.loads(precon.with_suffix("").with_suffix(".json").read_text())
    post_meta = json.loads(postcon.with_suffix("").with_suffix(".json").read_text())
    assert "Pre-contrast" in pre_meta["Description"]
    assert "Post-contrast" in post_meta["Description"]
    assert pre_meta["Sources"] == [
        "bids:raw:sub-01/ses-0001/anat/"
        "sub-01_ses-0001_acq-mprage_run-1_T1w.nii.gz",
        "bids:raw:sub-01/ses-0001/perf/sub-01_ses-0001_dce.nii.gz",
    ]


def test_dce_both_flair_underlays(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Both FLAIR underlays configured: two anat/ outputs differing by ce- entity."""
    bids, src = minimal_bids
    out = tmp_path / "out"
    pre = _write_nii(tmp_path / "flair_in_dce.nii.gz")
    post = _write_nii(tmp_path / "flair_postcon_in_dce.nii.gz")

    _build_dce_underlay_wf(
        bids, src, out, tmp_path,
        present_inputs={"flair_in_dce", "flair_postcon_in_dce"},
        work_subdir="work-flair-both",
        flair_in_dce=pre,
        flair_postcon_in_dce=post,
    )

    anat_dir = out / "pmriprep" / "sub-01" / "ses-0001" / "anat"
    precon = anat_dir / "sub-01_ses-0001_acq-3d_run-1_space-dceref_FLAIR.nii.gz"
    postcon = anat_dir / "sub-01_ses-0001_acq-3d_ce-GAD_run-2_space-dceref_FLAIR.nii.gz"
    assert precon.exists() and postcon.exists(), (
        f"both FLAIR underlays expected; anat/ contains: "
        f"{sorted(p.name for p in anat_dir.iterdir())}"
    )
    post_meta = json.loads(postcon.with_suffix("").with_suffix(".json").read_text())
    assert "Post-contrast" in post_meta["Description"]


# ---------------------------------------------------------------------------
# Fp/PS uncertainty TR gate: the build-time decision to add
# fp_uncertainty/ps_uncertainty to dce_present must mirror the runtime
# decision in dce_compute._build_candidate_set so the sink isn't built
# without an upstream feed.
# ---------------------------------------------------------------------------


def test_2cxm_uncertainty_tr_gate_auto_short_tr() -> None:
    """auto-mode + TR ≤ 2.0 s → 2CXM in candidates → emit Fp/PS std."""
    from pmriprep.workflows.base import _emit_2cxm_uncertainty

    assert _emit_2cxm_uncertainty("auto", 1.99) is True
    assert _emit_2cxm_uncertainty("auto", 2.0) is True
    assert _emit_2cxm_uncertainty("auto", 2.0000000000000004) is True


def test_2cxm_uncertainty_tr_gate_auto_long_tr() -> None:
    """Auto mode excludes 2CXM uncertainty outputs above the 2 s interval limit."""
    from pmriprep.workflows.base import _emit_2cxm_uncertainty

    assert _emit_2cxm_uncertainty("auto", 2.00001) is False
    assert _emit_2cxm_uncertainty("auto", 2.04) is False
    assert _emit_2cxm_uncertainty("auto", 4.06) is False
    assert _emit_2cxm_uncertainty("auto", 7.6) is False


def test_2cxm_uncertainty_tr_gate_explicit_2cxm() -> None:
    """Explicit ``model="2cxm"`` always emits — bypasses the TR check."""
    from pmriprep.workflows.base import _emit_2cxm_uncertainty

    assert _emit_2cxm_uncertainty("2cxm", 4.06) is True
    assert _emit_2cxm_uncertainty("2cxm", None) is True


def test_2cxm_uncertainty_tr_gate_other_models() -> None:
    """Non-2CXM single-model paths never emit 2CXM uncertainty."""
    from pmriprep.workflows.base import _emit_2cxm_uncertainty

    for model in ("tofts", "extended_tofts", "patlak"):
        assert _emit_2cxm_uncertainty(model, 2.5) is False, model


def test_2cxm_uncertainty_tr_gate_malformed_inputs() -> None:
    """Defensive: missing/malformed values fall through to no-emission."""
    from pmriprep.workflows.base import _emit_2cxm_uncertainty

    assert _emit_2cxm_uncertainty(None, 2.5) is False
    assert _emit_2cxm_uncertainty("auto", None) is False
    assert _emit_2cxm_uncertainty("auto", "not-a-number") is False  # type: ignore[arg-type]


def _build_dce_wf_for_tr_gate(
    tmp_path: Path, *, volume_tr: float, model: str, inference: str,
):
    """Build init_pmriprep_wf with a minimal DCE-only fixture at the given TR."""
    import math
    import json as _json

    from pmriprep.config.tumor_seg import TumorSegConfig
    from pmriprep.config.anat import AnatConfig
    from pmriprep.config.dsc import DSCConfig
    from pmriprep.config.dce import DCEConfig
    from pmriprep.workflows.base import PipelineSkipFlags, init_pmriprep_wf

    n_volumes = max(20, math.ceil(121.0 / volume_tr))

    bids = tmp_path / "bids"
    raw = bids / "sub-01" / "ses-0001"
    (raw / "anat").mkdir(parents=True)
    (raw / "perf").mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        _json.dumps({"Name": "T", "BIDSVersion": "1.9.0", "DatasetType": "raw"})
    )
    nib.save(
        nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)),
        raw / "anat" / "sub-01_ses-0001_T1w.nii.gz",
    )
    (raw / "anat" / "sub-01_ses-0001_T1w.json").write_text(
        _json.dumps({"RepetitionTime": 2.0})
    )
    nib.save(
        nib.Nifti1Image(np.zeros((4, 4, 4, n_volumes), dtype=np.float32), np.eye(4)),
        raw / "perf" / "sub-01_ses-0001_dce.nii.gz",
    )
    (raw / "perf" / "sub-01_ses-0001_dce.json").write_text(
        _json.dumps({
            "ContrastBolusAgent": "Magnevist",
            "RepetitionTime": volume_tr,
            "FlipAngle": 12.0,
            "MagneticFieldStrength": 3.0,
            "EchoTime": 0.0014,
        })
    )

    dce_cfg = DCEConfig()
    dce_cfg.model = model
    dce_cfg.inference = inference
    dce_cfg.no_t1 = True
    dce_cfg.b1_method = "none"

    return init_pmriprep_wf(
        bids_dir=str(bids),
        output_dir=str(tmp_path / "out"),
        subject_id="01",
        session_id="0001",
        skip_flags=PipelineSkipFlags(
            skip_anat=True, skip_dsc=True, skip_dce=False,
        ),
        anat_config=AnatConfig(),
        tumor_seg_config=TumorSegConfig(),
        dsc_config=DSCConfig(),
        dce_config=dce_cfg,
        work_dir=str(tmp_path / "work"),
    )


def _has_fp_uncertainty_sink(wf) -> bool:
    """Return whether the workflow emits Fp absolute uncertainty."""
    return any(name.endswith("fp_uncertainty_sink") for name in wf.list_node_names())


def test_init_pmriprep_wf_auto_long_tr_no_fp_sink(tmp_path: Path) -> None:
    """Auto mode with VI and a 4.06 s interval omits the Fp uncertainty sink."""
    wf = _build_dce_wf_for_tr_gate(
        tmp_path, volume_tr=4.06, model="auto", inference="vi",
    )
    assert not _has_fp_uncertainty_sink(wf), (
        "auto + Bayesian + TR>2.0 must NOT build fp_uncertainty_sink "
        "(producer doesn't include 2CXM in candidate set at this TR; "
        "sink without feed → runtime crash)"
    )


def test_init_pmriprep_wf_auto_short_tr_builds_fp_sink(tmp_path: Path) -> None:
    """Auto Bayesian short-TR runs emit Fp absolute uncertainty."""
    wf = _build_dce_wf_for_tr_gate(
        tmp_path, volume_tr=2.0, model="auto", inference="vi",
    )
    assert _has_fp_uncertainty_sink(wf), (
        "auto + Bayesian + TR≤2.0 must build fp_uncertainty_sink "
        "(producer includes 2CXM in candidate set; sink should consume it)"
    )


# ---------------------------------------------------------------------------
# Optional figure inputs accept Python None from upstream Function nodes
# without triggering Nipype File validation errors.
# ---------------------------------------------------------------------------


_ANAT_OPTIONAL_FIGURES = (
    "brain_mask_filmstrip",
    "tissue_seg_filmstrip",
    "acpc_filmstrip",
    "coreg_filmstrip_postcon",
    "coreg_filmstrip_t2w",
    "coreg_filmstrip_flair",
    "coreg_filmstrip_flair_postcon",
)

_TUMOR_SEG_OPTIONAL_FIGURES = (
    "seg_filmstrip_postcon",
    "seg_filmstrip_flair",
    "nawm_filmstrip_flair",
)

_DSC_OPTIONAL_FIGURES = (
    "brain_mask_filmstrip",
    "motion_qc_plot",
    "aif_curve_plot",
    "aif_location_plot",
    "aif_candidates_plot",
    "registration_filmstrip",
    "dsc_preprocessing_qc_plot",
    "dsc_param_maps_qc_plot",
    "dsc_fit_qc_plot",
    "dsc_leakage_qc_plot",
)

_DCE_OPTIONAL_FIGURES = (
    "brain_mask_filmstrip",
    "motion_qc_plot",
    "aif_curve_plot",
    "aif_location_plot",
    "aif_candidates_plot",
    "registration_filmstrip",
    "b1_qc_plot",
    "t1_qc_plot",
    "preprocessing_qc_plot",
    "pk_modeling_qc_plot",
    "pk_param_maps_qc_plot",
    "model_selection_qc_plot",
    "dsc_dce_xval_plot",
)


def test_dsc_absent_qc_plot_producers_use_none() -> None:
    """Absent DSC QC figures satisfy the optional writer-file contract."""
    from pmriprep.interfaces.qc_writers import DSCFiguresWriter
    from pmriprep.workflows.dsc._helpers import (
        _generate_dsc_fit_qc,
        _generate_dsc_leakage_qc,
    )

    leakage = _generate_dsc_leakage_qc("", "", "", "", "")
    fit = _generate_dsc_fit_qc("", "", "", "01", "0001")
    assert leakage is None
    assert fit is None

    writer = DSCFiguresWriter()
    writer.inputs.dsc_leakage_qc_plot = leakage
    writer.inputs.dsc_fit_qc_plot = fit


def test_dsc_leakage_qc_skips_all_nan_k2(tmp_path: Path) -> None:
    """Unavailable signed K2 must not create a meaningless montage."""
    from pmriprep.workflows.dsc._helpers import _generate_dsc_leakage_qc

    shape = (4, 4, 4)
    paths: list[str] = []
    for name, data in (
        ("k2", np.full(shape, np.nan)),
        ("auc_raw", np.ones(shape)),
        ("auc_corr", np.ones(shape)),
        ("auc_diff", np.zeros(shape)),
        ("mask", np.ones(shape)),
    ):
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)
        paths.append(str(path))

    assert _generate_dsc_leakage_qc(*paths) is None


def test_dsc_fit_qc_node_accepts_undefined_optional_inputs(tmp_path: Path) -> None:
    """Nipype may omit unavailable fit-QC inputs without failing the workflow."""
    import nipype.interfaces.utility as niu
    import nipype.pipeline.engine as pe

    from pmriprep.workflows.dsc._helpers import _generate_dsc_fit_qc

    node = pe.Node(
        niu.Function(
            input_names=[
                "qc_fit_data_file",
                "brain_mask_file",
                "cbf_file",
                "subject_id",
                "session_id",
            ],
            output_names=["plot_file"],
            function=_generate_dsc_fit_qc,
        ),
        name="dsc_fit_qc",
        base_dir=str(tmp_path),
    )
    node.inputs.subject_id = "01"
    node.inputs.session_id = "0001"

    result = node.run()

    assert not result.outputs.trait_get()["plot_file"]


def _figures_dir(tmp_path: Path, subject: str = "test") -> Path:
    return tmp_path / "pmriprep" / f"sub-{subject}" / "figures"


def test_anat_figures_writer_run_with_all_none_figures(tmp_path: Path) -> None:
    from pmriprep.interfaces.qc_writers import AnatFiguresWriter

    writer = AnatFiguresWriter()
    writer.inputs.output_dir = str(tmp_path)
    writer.inputs.subject_id = "test"
    for field in _ANAT_OPTIONAL_FIGURES:
        setattr(writer.inputs, field, None)
    writer.run()

    assert not _figures_dir(tmp_path).exists(), (
        "all-None figure inputs must not create the figures/ directory"
    )


def test_tumor_seg_figures_writer_run_with_all_none_figures(tmp_path: Path) -> None:
    from pmriprep.interfaces.qc_writers import TumorSegFiguresWriter

    writer = TumorSegFiguresWriter()
    writer.inputs.output_dir = str(tmp_path)
    writer.inputs.subject_id = "test"
    for field in _TUMOR_SEG_OPTIONAL_FIGURES:
        setattr(writer.inputs, field, None)
    writer.run()

    assert not _figures_dir(tmp_path).exists(), (
        "all-None figure inputs must not create the figures/ directory"
    )


def test_dsc_figures_writer_run_with_all_none_figures(tmp_path: Path) -> None:
    from pmriprep.interfaces.qc_writers import DSCFiguresWriter

    writer = DSCFiguresWriter()
    writer.inputs.output_dir = str(tmp_path)
    writer.inputs.subject_id = "test"
    for field in _DSC_OPTIONAL_FIGURES:
        setattr(writer.inputs, field, None)
    writer.run()

    assert not _figures_dir(tmp_path).exists(), (
        "all-None figure inputs must not create the figures/ directory"
    )


def test_dce_figures_writer_run_with_all_none_figures(tmp_path: Path) -> None:
    from pmriprep.interfaces.qc_writers import DCEFiguresWriter

    writer = DCEFiguresWriter()
    writer.inputs.output_dir = str(tmp_path)
    writer.inputs.subject_id = "test"
    for field in _DCE_OPTIONAL_FIGURES:
        setattr(writer.inputs, field, None)
    writer.run()

    assert not _figures_dir(tmp_path).exists(), (
        "all-None figure inputs must not create the figures/ directory"
    )
