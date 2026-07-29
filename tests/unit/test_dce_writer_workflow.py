"""Smoke + sidecar-correctness tests for ``init_dce_datasinks_wf``."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _write_nii(path: Path) -> str:
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), path)
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
    """Mandatory PK output (Ktrans) writes correctly with M3 sidecar fields."""
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
    assert sidecar["Resolution"] == "native"
    assert isinstance(sidecar["VoxelSize"], list)
    assert sidecar["InputFunctionCompartment"] == "plasma"
    assert "KtransConvention" not in sidecar


def test_unvalidated_model_free_maps_have_no_derivative_sinks(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Wash slopes and SER-derived curve classes remain internal outputs."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, _ = minimal_bids
    wf = init_dce_datasinks_wf(
        output_dir=str(tmp_path / "out"), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="auto", inference="nlls",
        present_inputs={"wash_in_slope_map", "wash_out_slope_map", "curve_type_map"},
    )
    node_names = set(wf.list_node_names())
    assert not any("wash_in_slope_map" in name for name in node_names)
    assert not any("wash_out_slope_map" in name for name in node_names)
    assert not any("curve_type_map" in name for name in node_names)


def test_dce_writer_recreates_deleted_cached_derivative(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """External derivatives are rewritten even when the node cache survives."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"
    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="tofts", inference="nlls",
    )
    wf.base_dir = str(tmp_path / "work")
    wf.inputs.inputnode.source_dce = src
    wf.inputs.inputnode.ktrans_map = _write_nii(tmp_path / "ktrans.nii.gz")
    wf.inputs.inputnode.qc_metrics = {}

    wf.run()
    perf = out / "pmriprep" / "sub-01" / "ses-0001" / "perf"
    nii = perf / "sub-01_ses-0001_desc-dceKtrans_stat.nii.gz"
    sidecar = nii.with_suffix("").with_suffix(".json")
    assert nii.exists() and sidecar.exists()

    nii.unlink()
    sidecar.unlink()
    wf.run()

    assert nii.exists() and sidecar.exists()


def test_dce_factory_with_bayesian_uncertainty(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Bayesian inference emits absolute uncertainty per PK parameter."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"

    present = {
        "ve_map", "vp_map",
        "ktrans_uncertainty", "ve_uncertainty", "vp_uncertainty",
        "r2_map",
    }
    wf = init_dce_datasinks_wf(
        output_dir=str(out), bids_dir=str(bids),
        subject_id="01", session_id="0001",
        pk_model="extended_tofts", inference="vi",
        present_inputs=present,
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "k.nii.gz")
    for attr in present:
        setattr(inp, attr, _write_nii(tmp_path / f"{attr}.nii.gz"))
    inp.qc_metrics = {}
    inp.config_dict = {"inference": "vi"}
    wf.run()

    perf_out = out / "pmriprep" / "sub-01" / "ses-0001" / "perf"
    expected = [
        "sub-01_ses-0001_desc-dceKtrans_stat.nii.gz",
        "sub-01_ses-0001_desc-dceVe_stat.nii.gz",
        "sub-01_ses-0001_desc-dceVp_stat.nii.gz",
        "sub-01_ses-0001_desc-dcePosteriorSDKtrans_stat.nii.gz",
    ]
    for fname in expected:
        assert (perf_out / fname).exists(), f"Missing: {fname}"

    expected_units = {
        "desc-dcePosteriorSDKtrans_stat.json": "1/min",
        "desc-dceRsquared_stat.json": "1",
    }
    for ending, units in expected_units.items():
        sidecar = next(perf_out.glob(f"*{ending}"))
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
    posterior_sd = json.loads(
        next(perf_out.glob("*desc-dcePosteriorSDKtrans_stat.json")).read_text()
    )
    assert posterior_sd["UncertaintyStatistic"] == "posterior_standard_deviation"


def test_map_uncertainty_sidecar_records_run_accounting(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """MAP uncertainty sidecars include actual Hessian/fallback counts."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"
    wf = init_dce_datasinks_wf(
        output_dir=str(out),
        bids_dir=str(bids),
        subject_id="01",
        session_id="0001",
        pk_model="tofts", inference="map",
        present_inputs={"ktrans_uncertainty"},
    )
    wf.base_dir = str(tmp_path / "work")
    inp = wf.inputs.inputnode
    inp.source_dce = src
    inp.ktrans_map = _write_nii(tmp_path / "k.nii.gz")
    inp.ktrans_uncertainty = _write_nii(tmp_path / "k_std.nii.gz")
    inp.config_dict = {"inference": "map"}
    inp.qc_metrics = {
        "parameter_uncertainty": {
            "aif_inflation_input_factor": 2.0,
            "aif_parameter_scale_factors": {"Ktrans": 1.5, "ve": 1.0},
        },
        "map_uncertainty": {
            "n_retained_voxels": 10,
            "n_evaluated": 9,
            "n_laplace_hessian": 7,
            "n_prior_fallback": 2,
            "n_not_computed": 1,
            "prior_fallback_fraction": 2 / 9,
        }
    }
    wf.run()

    sidecar = next(
        (out / "pmriprep" / "sub-01" / "ses-0001" / "perf").glob(
            "*desc-dcePosteriorSDKtrans_stat.json"
        )
    )
    metadata = json.loads(sidecar.read_text())
    accounting = metadata["MAPUncertaintyAccounting"]
    assert accounting["n_laplace_hessian"] == 7
    assert accounting["n_prior_fallback"] == 2
    assert accounting["n_not_computed"] == 1
    inflation = metadata["AIFUncertaintyInflation"]
    assert inflation["Applied"] is True
    assert inflation["ParameterScaleFactors"]["Ktrans"] == 1.5


def test_nlls_absolute_uncertainty_uses_frequentist_derivative_name(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """NLLS absolute uncertainty uses the standard-error name."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, _ = minimal_bids
    workflow = init_dce_datasinks_wf(
        output_dir=str(tmp_path / "out"),
        bids_dir=str(bids),
        subject_id="01",
        session_id="0001",
        pk_model="tofts",
        inference="nlls",
        present_inputs={"ktrans_uncertainty"},
    )
    assert workflow.get_node("ktrans_uncertainty_sink").inputs.desc == "dceStandardError"


def test_dce_factory_model_free_no_model_entity(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Model-free maps (IAUC60, peakEnhancement, etc.) drop the model entity."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"

    present = {"iauc60_map", "peak_enhancement_map", "time_to_peak_map"}
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
        setattr(inp, attr, _write_nii(tmp_path / f"{attr}.nii.gz"))
    inp.qc_metrics = {
        "iauc": {
            "reference_frame": 5,
            "reference_time_s": 25.0,
            "reference_selection": "earlier_of_baseline_count_and_detected_arrival",
            "configured_baseline_frames": 8,
            "effective_baseline_frames": 5,
            "detected_bolus_arrival_frame": 6,
            "available_post_reference_duration_s": 180.0,
            "windows": {
                "60": {
                    "requested_duration_s": 60.0,
                    "complete": True,
                    "cutoff_interpolated": False,
                }
            },
        }
    }
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
    assert "MaximumValueCap" not in peak_sidecar
    ttp_sidecar = json.loads(
        (perf_out / "sub-01_ses-0001_desc-dceTtp_stat.json").read_text()
    )
    assert ttp_sidecar["ReferencePoint"] == "selected global contrast reference frame"
    assert ttp_sidecar["ReferenceFrame"] == 5
    assert ttp_sidecar["ReferenceTimeSeconds"] == 25.0
    assert ttp_sidecar["MinimumPeakEnhancementNoiseSD"] == 5.0
    assert "brain-median" in ttp_sidecar["NoiseEstimate"]
    assert "biased" in ttp_sidecar["Limitation"]


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
                "model_selection_regions", "vp_at_bound_fractions",
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


def test_2cxm_bayesian_fp_ps_absolute_uncertainty(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """2CXM Bayesian inference emits Fp/PS posterior-SD maps."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"

    present = {
        "ve_map", "kep_map", "vp_map",
        "fp_uncertainty", "ps_uncertainty",
    }
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
        setattr(inp, attr, _write_nii(tmp_path / f"{attr}.nii.gz"))
    inp.qc_metrics = {}
    wf.run()

    perf_out = out / "pmriprep" / "sub-01" / "ses-0001" / "perf"
    assert (perf_out / "sub-01_ses-0001_desc-dcePosteriorSDFp_stat.nii.gz").exists()
    assert (perf_out / "sub-01_ses-0001_desc-dcePosteriorSDPS_stat.nii.gz").exists()
    assert not (
        perf_out / "sub-01_ses-0001_desc-dcePosteriorSDKtrans_stat.nii.gz"
    ).exists()
    for parameter in ("Fp", "PS"):
        sidecar = perf_out / f"sub-01_ses-0001_desc-dcePosteriorSD{parameter}_stat.json"
        assert json.loads(sidecar.read_text())["Units"] == "mL/min/100mL"


def test_2cxm_extraction_fraction_derivative(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """The existing 2CXM extraction-fraction result reaches derivatives."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, src = minimal_bids
    out = tmp_path / "out"
    wf = init_dce_datasinks_wf(
        output_dir=str(out),
        bids_dir=str(bids),
        subject_id="01",
        session_id="0001",
        pk_model="2cxm",
        inference="nlls",
        present_inputs={"e_map"},
    )
    wf.base_dir = str(tmp_path / "work")
    wf.inputs.inputnode.source_dce = src
    wf.inputs.inputnode.ktrans_map = _write_nii(tmp_path / "k.nii.gz")
    wf.inputs.inputnode.e_map = _write_nii(tmp_path / "e.nii.gz")
    wf.inputs.inputnode.qc_metrics = {}
    wf.run()

    perf = out / "pmriprep" / "sub-01" / "ses-0001" / "perf"
    derivative = perf / "sub-01_ses-0001_desc-dceE_stat.nii.gz"
    sidecar = json.loads(derivative.with_suffix("").with_suffix(".json").read_text())
    assert derivative.exists()
    assert sidecar["Units"] == "1"
    assert "Extraction fraction" in sidecar["Description"]


def test_init_pmriprep_wf_dce_patlak_doesnt_crash(tmp_path: Path) -> None:
    """Build init_pmriprep_wf with --dce-model patlak and verify no NameError /
    crash on factory construction.
    """
    import json as _json

    from pmriprep.config.tumor_seg import TumorSegConfig
    from pmriprep.config.anat import AnatConfig
    from pmriprep.config.dsc import DSCConfig
    from pmriprep.config.dce import DCEConfig
    from pmriprep.workflows.base import PipelineSkipFlags, init_pmriprep_wf

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
        nib.Nifti1Image(np.zeros((4, 4, 4, 20), dtype=np.float32), np.eye(4)),
        raw / "perf" / "sub-01_ses-0001_dce.nii.gz",
    )
    (raw / "perf" / "sub-01_ses-0001_dce.json").write_text(
        _json.dumps({
            "RepetitionTime": 4.0,
            "FlipAngle": 12.0,
            "MagneticFieldStrength": 3.0,
            "EchoTime": 0.0014,
        })
    )

    dce_cfg = DCEConfig()
    dce_cfg.model = "patlak"
    dce_cfg.inference = "nlls"
    dce_cfg.no_t1 = True
    dce_cfg.b1_method = "none"

    wf = init_pmriprep_wf(
        bids_dir=str(bids),
        output_dir=str(tmp_path / "out"),
        subject_id="01",
        session_id="0001",
        skip_flags=PipelineSkipFlags(skip_anat=True, skip_dsc=True, skip_dce=False),
        anat_config=AnatConfig(),
        tumor_seg_config=TumorSegConfig(),
        dsc_config=DSCConfig(),
        dce_config=dce_cfg,
        work_dir=str(tmp_path / "work"),
    )
    assert wf is not None


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
    assert any("B1+" in w for w in warnings)
    assert any("T1 relaxation" in w for w in warnings)
    assert any("Brain mask" in w for w in warnings)
    assert any("registration transform" in w for w in warnings)
    assert "InputFunctionCompartment" not in sidecar

    # Session-root QC JSON orphan must not exist
    legacy = (
        out / "pmriprep" / "sub-01" / "ses-0001"
        / "sub-01_ses-0001_desc-dceQC_quality.json"
    )
    assert not legacy.exists()


def test_dce_intentional_t1_b1_skips_do_not_emit_missing_warnings(
    tmp_path: Path, minimal_bids: tuple[Path, str],
) -> None:
    """Configured T1/B1 opt-outs are not reported as missing derivatives."""
    from pmriprep.workflows.dce.datasinks import init_dce_datasinks_wf

    bids, _ = minimal_bids
    workflow = init_dce_datasinks_wf(
        output_dir=str(tmp_path / "out"),
        bids_dir=str(bids),
        subject_id="01",
        session_id="0001",
        pk_model="tofts",
        inference="nlls",
        present_inputs={"dce_moco"},
        t1_map_expected=False,
        b1_map_expected=False,
    )

    warnings = workflow.get_node("dce_moco_meta").inputs.presence_warnings
    assert not any("T1 relaxation" in warning for warning in warnings)
    assert not any("B1+" in warning for warning in warnings)


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


def test_dce_t1w_precon_underlay_lands_in_anat(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Precon-only underlay: lands in anat/ with no ce- entity."""
    bids, src = minimal_bids
    out = tmp_path / "out"
    underlay = _write_nii(tmp_path / "t1w_in_dce.nii.gz")

    _build_dce_underlay_wf(
        bids, src, out, tmp_path,
        present_inputs={"t1w_in_dce"},
        work_subdir="work-precon",
        t1w_in_dce=underlay,
    )

    anat_dir = out / "pmriprep" / "sub-01" / "ses-0001" / "anat"
    expected = anat_dir / "sub-01_ses-0001_acq-mprage_run-1_space-dceref_T1w.nii.gz"
    assert expected.exists(), (
        f"precon underlay missing at {expected}; "
        f"anat/ contains: {sorted(p.name for p in anat_dir.iterdir())}"
    )
    assert "ce-GAD" not in expected.name
    sidecar = json.loads(expected.with_suffix("").with_suffix(".json").read_text())
    assert "Pre-contrast" in sidecar["Description"]
    assert sidecar["Sources"] == [
        "bids:raw:sub-01/ses-0001/anat/"
        "sub-01_ses-0001_acq-mprage_run-1_T1w.nii.gz",
        "bids:raw:sub-01/ses-0001/perf/sub-01_ses-0001_dce.nii.gz",
    ]


def test_dce_t1w_postcon_underlay_lands_in_anat(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Postcon-only underlay: lands in anat/ with ce-GAD entity."""
    bids, src = minimal_bids
    out = tmp_path / "out"
    underlay = _write_nii(tmp_path / "t1w_postcon_in_dce.nii.gz")

    _build_dce_underlay_wf(
        bids, src, out, tmp_path,
        present_inputs={"t1w_postcon_in_dce"},
        work_subdir="work-postcon",
        t1w_postcon_in_dce=underlay,
    )

    anat_dir = out / "pmriprep" / "sub-01" / "ses-0001" / "anat"
    expected = anat_dir / "sub-01_ses-0001_acq-mprage_ce-GAD_run-2_space-dceref_T1w.nii.gz"
    assert expected.exists(), (
        f"postcon underlay missing at {expected}; "
        f"anat/ contains: {sorted(p.name for p in anat_dir.iterdir())}"
    )
    assert not (anat_dir / "sub-01_ses-0001_space-dceref_T1w.nii.gz").exists()
    sidecar = json.loads(expected.with_suffix("").with_suffix(".json").read_text())
    assert "Post-contrast" in sidecar["Description"]


def test_dce_t1w_subtraction_underlay_lands_in_anat(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Subtraction underlay: anat/ with desc-subtraction (C3 desc plumbing), no ce-."""
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


def test_dce_flair_precon_underlay_lands_in_anat(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Precon FLAIR underlay: lands in anat/ with no ce- entity."""
    bids, src = minimal_bids
    out = tmp_path / "out"
    underlay = _write_nii(tmp_path / "flair_in_dce.nii.gz")

    _build_dce_underlay_wf(
        bids, src, out, tmp_path,
        present_inputs={"flair_in_dce"},
        work_subdir="work-flair-precon",
        flair_in_dce=underlay,
    )

    anat_dir = out / "pmriprep" / "sub-01" / "ses-0001" / "anat"
    expected = anat_dir / "sub-01_ses-0001_acq-3d_run-1_space-dceref_FLAIR.nii.gz"
    assert expected.exists(), (
        f"precon FLAIR underlay missing at {expected}; "
        f"anat/ contains: {sorted(p.name for p in anat_dir.iterdir())}"
    )
    assert not (anat_dir / "sub-01_ses-0001_ce-GAD_space-dceref_FLAIR.nii.gz").exists()


def test_dce_flair_postcon_underlay_lands_in_anat(
    tmp_path: Path, minimal_bids: tuple[Path, str]
) -> None:
    """Postcon FLAIR underlay: lands in anat/ with ce-GAD entity."""
    bids, src = minimal_bids
    out = tmp_path / "out"
    underlay = _write_nii(tmp_path / "flair_postcon_in_dce.nii.gz")

    _build_dce_underlay_wf(
        bids, src, out, tmp_path,
        present_inputs={"flair_postcon_in_dce"},
        work_subdir="work-flair-postcon",
        flair_postcon_in_dce=underlay,
    )

    anat_dir = out / "pmriprep" / "sub-01" / "ses-0001" / "anat"
    expected = anat_dir / "sub-01_ses-0001_acq-3d_ce-GAD_run-2_space-dceref_FLAIR.nii.gz"
    assert expected.exists(), (
        f"postcon FLAIR underlay missing at {expected}; "
        f"anat/ contains: {sorted(p.name for p in anat_dir.iterdir())}"
    )
    assert not (anat_dir / "sub-01_ses-0001_space-dceref_FLAIR.nii.gz").exists()
    sidecar = json.loads(expected.with_suffix("").with_suffix(".json").read_text())
    assert "Post-contrast" in sidecar["Description"]


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
    """auto-mode + TR ≤ 3.0 s → 2CXM in candidates → emit Fp/PS std."""
    from pmriprep.workflows.base import _emit_2cxm_uncertainty

    assert _emit_2cxm_uncertainty("auto", 2.5) is True
    assert _emit_2cxm_uncertainty("auto", 3.0) is True


def test_2cxm_uncertainty_tr_gate_auto_long_tr() -> None:
    """auto-mode + TR > 3.0 s → 2CXM excluded → don't emit Fp/PS std.

    Reproduces the original crash scenario: current institutional cohort
    has volume_tr=4.06 s; the prior (auto, _) gate built the sink but the
    producer never fed it → ``NodeExecutionError: _copy_perf_map missing
    in_file``.
    """
    from pmriprep.workflows.base import _emit_2cxm_uncertainty

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
    """Regression for crash.txt: auto + Bayesian + TR>3.0 → no fp/ps sink built.

    Reproduces the original ``NodeExecutionError: _copy_perf_map missing
    in_file`` failure mode: ``model="auto"``, ``inference="vi"``,
    ``RepetitionTime=4.06`` (current institutional cohort).
    """
    wf = _build_dce_wf_for_tr_gate(
        tmp_path, volume_tr=4.06, model="auto", inference="vi",
    )
    assert not _has_fp_uncertainty_sink(wf), (
        "auto + Bayesian + TR>3.0 must NOT build fp_uncertainty_sink "
        "(producer doesn't include 2CXM in candidate set at this TR; "
        "sink without feed → runtime crash)"
    )


def test_init_pmriprep_wf_auto_short_tr_builds_fp_sink(tmp_path: Path) -> None:
    """Auto Bayesian short-TR runs emit Fp absolute uncertainty."""
    wf = _build_dce_wf_for_tr_gate(
        tmp_path, volume_tr=2.5, model="auto", inference="vi",
    )
    assert _has_fp_uncertainty_sink(wf), (
        "auto + Bayesian + TR≤3.0 must build fp_uncertainty_sink "
        "(producer includes 2CXM in candidate set; sink should consume it)"
    )


# ---------------------------------------------------------------------------
# Optional-figure trait hardening (regression for crash where upstream
# Function nodes — e.g. ``_dsc_dce_xval_plot_node`` — return Python ``None``
# and trip ``TraitError`` in nipype's File validator, stalling the DAG.)
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


def test_anat_figures_writer_accepts_none_for_optional_figures(
    tmp_path: Path,
) -> None:
    from pmriprep.interfaces.qc_writers import AnatFiguresWriter

    writer = AnatFiguresWriter()
    writer.inputs.output_dir = str(tmp_path)
    writer.inputs.subject_id = "test"
    for field in _ANAT_OPTIONAL_FIGURES:
        setattr(writer.inputs, field, None)


def test_tumor_seg_figures_writer_accepts_none_for_optional_figures(
    tmp_path: Path,
) -> None:
    from pmriprep.interfaces.qc_writers import TumorSegFiguresWriter

    writer = TumorSegFiguresWriter()
    writer.inputs.output_dir = str(tmp_path)
    writer.inputs.subject_id = "test"
    for field in _TUMOR_SEG_OPTIONAL_FIGURES:
        setattr(writer.inputs, field, None)


def test_dsc_figures_writer_accepts_none_for_optional_figures(
    tmp_path: Path,
) -> None:
    from pmriprep.interfaces.qc_writers import DSCFiguresWriter

    writer = DSCFiguresWriter()
    writer.inputs.output_dir = str(tmp_path)
    writer.inputs.subject_id = "test"
    for field in _DSC_OPTIONAL_FIGURES:
        setattr(writer.inputs, field, None)


def test_dce_figures_writer_accepts_none_for_optional_figures(
    tmp_path: Path,
) -> None:
    """Regression: dsc_dce_xval_plot=None must not raise TraitError or stall the DAG."""
    from pmriprep.interfaces.qc_writers import DCEFiguresWriter

    writer = DCEFiguresWriter()
    writer.inputs.output_dir = str(tmp_path)
    writer.inputs.subject_id = "test"
    for field in _DCE_OPTIONAL_FIGURES:
        setattr(writer.inputs, field, None)


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
