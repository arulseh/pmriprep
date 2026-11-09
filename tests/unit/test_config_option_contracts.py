"""Public configuration-file structure and option-name contracts."""

from pathlib import Path

import pytest

from pmriprep.config.loader import (
    build_anat_config,
    build_dce_config,
    build_dsc_config,
    build_tumor_seg_config,
    load_config,
    validate_config_paths,
)
from pmriprep.config.anat import AnatConfig
from pmriprep.config.dce import DCEConfig
from pmriprep.config.dsc import DSCConfig
from pmriprep.config.tumor_seg import TumorSegConfig
from pmriprep.cli.run import _build_parser
from pmriprep.config.loader import merge_cli_config


def _write_config(path: Path, content: str) -> Path:
    """Write and return a configuration fixture."""
    path.write_text(content)
    return path


@pytest.mark.parametrize("suffix", ["yaml", "json"])
def test_load_config_rejects_non_mapping_root(tmp_path: Path, suffix: str) -> None:
    """A sequence root must fail before workflow construction."""
    content = "- dce\n" if suffix == "yaml" else '["dce"]'
    path = _write_config(tmp_path / f"config.{suffix}", content)

    with pytest.raises(ValueError, match="root must be a mapping"):
        load_config(path)


def test_load_config_rejects_unknown_section(tmp_path: Path) -> None:
    """Misspelled first-level section names must not silently disappear."""
    path = _write_config(tmp_path / "config.yaml", "dec:\n  model: patlak\n")

    with pytest.raises(ValueError, match="Unknown configuration section.*dec"):
        load_config(path)


def test_load_config_rejects_non_mapping_section(tmp_path: Path) -> None:
    """Each public section must contain named options."""
    path = _write_config(tmp_path / "config.yaml", "dce: patlak\n")

    with pytest.raises(ValueError, match="section 'dce' must be a mapping"):
        load_config(path)


def test_load_config_rejects_unknown_option(tmp_path: Path) -> None:
    """Option typos must fail rather than selecting a default silently."""
    path = _write_config(tmp_path / "config.yaml", "dce:\n  modle: patlak\n")

    with pytest.raises(ValueError, match="Unknown dce configuration option.*modle"):
        load_config(path)


def test_load_config_rejects_nested_dce_sections(tmp_path: Path) -> None:
    """DCE options follow the same flat section contract as other pipelines."""
    path = _write_config(
        tmp_path / "config.yaml",
        "dce:\n  unsupported_group:\n    tsnr_threshold: 8\n",
    )

    with pytest.raises(
        ValueError,
        match="Unknown dce configuration option.*unsupported_group",
    ):
        load_config(path)


def test_dce_builder_rejects_nested_sections() -> None:
    """Programmatic construction cannot bypass the flat DCE schema."""
    with pytest.raises(
        ValueError,
        match="Unknown dce configuration option.*unsupported_group",
    ):
        build_dce_config({"dce": {"unsupported_group": {"tsnr_threshold": 8}}})


@pytest.mark.parametrize(
    ("builder", "section", "option"),
    [
        (build_dsc_config, "dsc", "methd"),
        (build_dce_config, "dce", "modle"),
        (build_anat_config, "anat", "denosie"),
        (build_tumor_seg_config, "tumor_seg", "devce"),
    ],
)
def test_direct_builders_reject_unknown_options(builder, section: str, option: str) -> None:
    """Programmatic config construction follows the file-loading contract."""
    with pytest.raises(ValueError, match=f"Unknown {section} configuration option"):
        builder({section: {option: True}})


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: DCEConfig(model="etm"), "Invalid model"),
        (lambda: DCEConfig(inference="svb"), "Invalid inference"),
        (lambda: DCEConfig(aif_method="synthetic"), "Invalid aif_method"),
        (lambda: DCEConfig(t1_method="unsupported"), "Invalid t1_method"),
        (lambda: DSCConfig(method="osvd"), "Invalid method"),
        (lambda: DSCConfig(inference="mcmc"), "Invalid inference"),
        (lambda: DSCConfig(leakage_method="bidirectional"), "Invalid leakage_method"),
        (lambda: AnatConfig(tool="bet"), "Invalid tool"),
        (lambda: TumorSegConfig(device="auto"), "Invalid device"),
    ],
)
def test_invalid_categorical_options_fail_before_workflow(factory, match: str) -> None:
    """Categorical YAML values must fail during config construction."""
    with pytest.raises(ValueError, match=match):
        factory()


def test_dsc_leakage_correction_can_be_disabled() -> None:
    """The supported DSC policies are signed BSW and explicit pass-through."""
    assert DSCConfig(leakage_method="bsw").leakage_method == "bsw"
    assert DSCConfig(leakage_method="none").leakage_method == "none"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DCEConfig(no_t1="false"),
        lambda: DCEConfig(aif_peak_floor=1),
        lambda: DSCConfig(motion_correct="false"),
        lambda: DSCConfig(cpi_infer_delay=0),
        lambda: AnatConfig(create_subtraction="false"),
    ],
)
def test_boolean_options_reject_truthy_strings_and_integers(factory) -> None:
    """Quoted booleans must not silently invert requested behavior."""
    with pytest.raises(TypeError, match="must be a boolean"):
        factory()


def test_path_objects_are_resolved_before_nipype_changes_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI-shaped relative Path objects receive the same normalization as YAML."""
    t1_map = tmp_path / "t1.nii.gz"
    t1_map.write_bytes(b"fixture")
    monkeypatch.chdir(tmp_path)

    config = build_dce_config({"dce": {"t1_map_file": Path("t1.nii.gz")}})

    assert config.t1_map_file == t1_map.resolve()


@pytest.mark.parametrize(
    ("section", "option"),
    [
        ("dsc", "aif_file"),
        ("dsc", "aif_mask_file"),
        ("dsc", "lesion_mask"),
        ("dsc", "nawm_mask"),
        ("dsc", "roi_mask"),
        ("dce", "aif_file"),
        ("dce", "aif_mask_file"),
        ("dce", "lesion_mask"),
        ("dce", "nawm_mask"),
        ("dce", "roi_mask"),
        ("dce", "t1_map_file"),
    ],
)
def test_missing_configured_paths_fail_before_workflow(
    tmp_path: Path, section: str, option: str,
) -> None:
    """Every external config path follows one existence contract."""
    missing = tmp_path / "missing.nii.gz"
    dsc = build_dsc_config({section: {option: missing}}) if section == "dsc" else DSCConfig()
    dce = build_dce_config({section: {option: missing}}) if section == "dce" else DCEConfig()

    with pytest.raises(FileNotFoundError, match=f"{section}.{option}"):
        validate_config_paths(dsc, dce)


def test_configured_aif_extension_is_validated(tmp_path: Path) -> None:
    """YAML AIF paths enforce the same formats as CLI AIF paths."""
    aif = tmp_path / "aif.csv"
    aif.write_text("0,1")
    dce = build_dce_config({"dce": {"aif_file": aif}})

    with pytest.raises(ValueError, match="unsupported extension"):
        validate_config_paths(DSCConfig(), dce)


@pytest.mark.parametrize(
    ("arguments", "section", "field", "expected"),
    [
        (["--dsc-deconv-method", "cpi"], "dsc", "method", "cpi"),
        (["--dsc-inference", "svb"], "dsc", "inference", "svb"),
        (["--dsc-preload", "0.5"], "dsc", "preload", 0.5),
        (["--dsc-dose", "0.5"], "dsc", "dose", 0.5),
        (["--dsc-agent-standard-dose", "0.05"], "dsc", "agent_standard_dose", 0.05),
        (["--dce-model", "patlak"], "dce", "model", "patlak"),
        (["--dce-inference", "vi"], "dce", "inference", "vi"),
        (["--dce-aif-method", "population"], "dce", "aif_method", "population"),
        (["--dce-dose", "2.0"], "dce", "dose", 2.0),
        (["--dce-r1", "5.0"], "dce", "r1_relaxivity", 5.0),
        (["--dce-b1-method", "none"], "dce", "b1_method", "none"),
        (["--dce-b1-map-encoding", "percent"], "dce", "b1_map_encoding", "percent"),
        (["--dce-no-t1"], "dce", "no_t1", True),
        (["--brain-extraction-tool", "ants"], "anat", "tool", "ants"),
        (["--denoise"], "anat", "denoise", True),
        (["--tumor-seg-device", "cpu"], "tumor_seg", "device", "cpu"),
    ],
)
def test_cli_and_file_options_build_identical_effective_values(
    arguments, section: str, field: str, expected,
) -> None:
    """Every exposed scientific CLI control maps to its config-file field."""
    parsed = _build_parser().parse_args(
        ["/tmp/bids", "/tmp/out", "participant", *arguments]
    )
    merged = merge_cli_config(vars(parsed), {})
    builders = {
        "dsc": build_dsc_config,
        "dce": build_dce_config,
        "anat": build_anat_config,
        "tumor_seg": build_tumor_seg_config,
    }

    assert getattr(builders[section](merged), field) == expected


@pytest.mark.parametrize("option", ["--dsc-dose", "--dce-dose"])
def test_cli_dose_rejects_named_presets(option: str) -> None:
    """Dose CLI options accept numeric multipliers only."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["/tmp/bids", "/tmp/out", "participant", option, "half"]
        )


@pytest.mark.parametrize(
    ("builder", "section"),
    [(build_dsc_config, "dsc"), (build_dce_config, "dce")],
)
@pytest.mark.parametrize("value", ["half", "0.5", True])
def test_config_dose_rejects_non_numeric_values(builder, section: str, value) -> None:
    """YAML and JSON dose fields require numeric values, not strings or booleans."""
    with pytest.raises(TypeError, match="dose must be an unquoted numeric multiplier"):
        builder({section: {"dose": value}})


def test_shared_hdbet_device_cli_reaches_all_brain_extraction_configs() -> None:
    """The shared device flag has identical DSC, DCE, and anatomical effect."""
    parsed = _build_parser().parse_args(
        ["/tmp/bids", "/tmp/out", "participant", "--hdbet-device", "cpu"]
    )
    merged = merge_cli_config(vars(parsed), {})

    assert build_dsc_config(merged).hdbet_device == "cpu"
    assert build_dce_config(merged).hdbet_device == "cpu"
    assert build_anat_config(merged).hdbet_device == "cpu"


def test_cli_none_does_not_override_file_option() -> None:
    """Absent CLI values preserve explicitly configured YAML values."""
    parsed = _build_parser().parse_args(["/tmp/bids", "/tmp/out", "participant"])
    merged = merge_cli_config(
        vars(parsed),
        {"dce": {"model": "extended_tofts"}, "dsc": {"method": "tikhonov"}},
    )

    assert build_dce_config(merged).model == "extended_tofts"
    assert build_dsc_config(merged).method == "tikhonov"


def test_cli_merge_does_not_mutate_file_config() -> None:
    """CLI precedence preserves the loaded file values used by provenance."""
    file_config = {
        "dsc": {"method": "tikhonov"},
        "dce": {"model": "extended_tofts"},
    }

    merged = merge_cli_config(
        {"dsc_deconv_method": "cpi", "dce_model": "patlak"},
        file_config,
    )

    assert merged["dsc"]["method"] == "cpi"
    assert merged["dce"]["model"] == "patlak"
    assert file_config == {
        "dsc": {"method": "tikhonov"},
        "dce": {"model": "extended_tofts"},
    }


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--nprocs", "0"),
        ("--nprocs", "-1"),
        ("--omp-nthreads", "0"),
        ("--mem-gb", "0"),
        ("--mem-gb", "nan"),
    ],
)
def test_invalid_resource_limits_fail_at_cli_parse(flag: str, value: str) -> None:
    """Invalid scheduler resources fail before creating output/work trees."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["/tmp/bids", "/tmp/out", "participant", flag, value]
        )


def test_skipping_optional_pipelines_remains_anatomical_only() -> None:
    """Disabling every optional pipeline still leaves useful anatomy processing."""
    parsed = _build_parser().parse_args([
        "/tmp/bids",
        "/tmp/out",
        "participant",
        "--skip-tumor-seg",
        "--skip-dsc",
        "--skip-dce",
    ])

    assert parsed.skip_tumor_seg is True
    assert parsed.skip_dsc is True
    assert parsed.skip_dce is True


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: DCEConfig(baseline_frames=1), r"0 \(auto\) or at least 5"),
        (lambda: DCEConfig(baseline_frames=4), r"0 \(auto\) or at least 5"),
        (lambda: DCEConfig(n_aif_candidates=2), "at least 3"),
        (lambda: DCEConfig(smooth_fwhm=-1), "smooth_fwhm"),
        (lambda: DCEConfig(tsnr_threshold=-1), "tsnr_threshold"),
        (lambda: DCEConfig(target_peak=0), "target_peak"),
        (lambda: DCEConfig(num_processes=-1), "num_processes"),
        (lambda: DSCConfig(pre_bolus=2), "pre_bolus"),
        (lambda: DSCConfig(n_aif_candidates=2), "at least 3"),
        (lambda: DSCConfig(motion_thresh=-1), "motion_thresh"),
        (lambda: DSCConfig(cpi_control_points=-1), "cpi_control_points"),
        (lambda: DSCConfig(num_processes=-1), "num_processes"),
    ],
)
def test_invalid_numeric_domains_fail_before_workflow(factory, match: str) -> None:
    """Impossible public numeric values fail during config construction."""
    with pytest.raises(ValueError, match=match):
        factory()


@pytest.mark.parametrize("field,option,value", [
    ("method", "--dsc-deconv-method", "unknown"),
    ("inference", "--dsc-inference", "nlls"),
    ("inference", "--dsc-inference", "map"),
    ("inference", "--dsc-inference", "mcmc"),
])
def test_invalid_dsc_choice_fails_before_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, option: str, value: str,
) -> None:
    """Both CLI and merged configuration reject unsupported choices early."""
    from pmriprep.cli.run import main

    bids = tmp_path / "bids"
    bids.mkdir()
    output = tmp_path / "out"
    config = _write_config(tmp_path / "config.yaml", f"dsc:\n  {field}: {value}\n")
    monkeypatch.setattr("pmriprep.utils.bids.validate_bids_input",
                        lambda *args, **kwargs: pytest.fail("input processing started"))
    monkeypatch.setattr("pmriprep.cli.run.run_pmriprep_wf",
                        lambda *args, **kwargs: pytest.fail("workflow started"))
    for selection in ([option, value], ["--config-file", str(config)]):
        with pytest.raises(SystemExit) as error:
            main([str(bids), str(output), "participant", *selection])
        assert error.value.code != 0
        assert not output.exists()


@pytest.mark.parametrize("method", ["sSVD", "oSVD", "tikhonov", "cpi"])
def test_supported_dsc_override_reaches_effective_config(method: str) -> None:
    """Explicit supported CLI choices supersede file values without coercion."""
    parsed = _build_parser().parse_args(
        ["/tmp/bids", "/tmp/out", "participant", "--dsc-deconv-method", method])
    merged = merge_cli_config(vars(parsed), {"dsc": {"method": "invalid"}})
    assert build_dsc_config(merged).method == method


def test_dsc_interface_rejects_unsupported_method() -> None:
    """The interface and datasinks reject an unknown deconvolution method."""
    from traits.trait_errors import TraitError
    from pmriprep.interfaces.dsc_compute import ComputeDSCMaps
    from pmriprep.workflows.dsc.datasinks import init_dsc_datasinks_wf

    with pytest.raises(TraitError):
        ComputeDSCMaps(method="unknown")
    with pytest.raises(ValueError, match="method"):
        init_dsc_datasinks_wf(output_dir="/tmp/out", bids_dir="/tmp/bids",
            subject_id="01", session_id="01", deconvolution_method="unknown")


@pytest.mark.parametrize("inference", ["nlls", "map", "mcmc"])
def test_dsc_interface_rejects_unsupported_inference(inference: str) -> None:
    """Direct interfaces cannot retain an ineffective inference selection."""
    from traits.trait_errors import TraitError
    from pmriprep.interfaces.dsc_compute import ComputeDSCMaps

    with pytest.raises(TraitError):
        ComputeDSCMaps(inference=inference)


@pytest.mark.parametrize("inference", ["bls", "vi"])
def test_dce_inference_selection_is_independent(inference: str) -> None:
    """DSC's inference choices do not alter DCE's existing dispatch controls."""
    from pmriprep.interfaces.dce_compute import ComputeDCEMaps

    parsed = _build_parser().parse_args([
        "/tmp/bids", "/tmp/out", "participant", "--dce-inference", inference])
    config = build_dce_config(merge_cli_config(vars(parsed), {}))
    assert config.inference == inference
    assert ComputeDCEMaps(inference=config.inference).inputs.inference == inference


@pytest.mark.parametrize("inference", ["nlls", "map", "mcmc"])
def test_retired_dce_pipeline_modes_are_rejected(inference: str) -> None:
    from traits.trait_errors import TraitError
    from pmriprep.interfaces.dce_compute import ComputeDCEMaps

    with pytest.raises(ValueError, match="Invalid inference"):
        build_dce_config({"dce": {"inference": inference}})
    with pytest.raises(TraitError):
        ComputeDCEMaps(inference=inference)
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["/tmp/bids", "/tmp/out", "participant", "--dce-inference", inference])


def test_delay_correction_boolean_and_retired_settings() -> None:
    assert DCEConfig().delay_correction is True
    assert build_dce_config({"dce": {"delay_correction": False}}).delay_correction is False
    for value in ("off", "auto", 1):
        with pytest.raises(TypeError, match="delay_correction must be a boolean"):
            build_dce_config({"dce": {"delay_correction": value}})
    for option in ("mcmc_samples", "aif_bat_alignment"):
        with pytest.raises(ValueError, match="Unknown"):
            build_dce_config({"dce": {option: 1}})


def test_sloppy_is_run_wide_and_cli_only(tmp_path: Path) -> None:
    """File configuration cannot create mixed normal/sloppy pipelines."""
    for section in ("anat", "dsc", "dce"):
        for value in ("true", "false"):
            path = _write_config(tmp_path / "config.yaml", f"{section}:\n  sloppy: {value}\n")
            with pytest.raises(ValueError, match="CLI-only; use --sloppy"):
                load_config(path)
        with pytest.raises(ValueError, match="CLI-only; use --sloppy"):
            merge_cli_config({}, {section: {"sloppy": True}})
    path = _write_config(tmp_path / "config.yaml", "dce:\n  model: 2cxm\n  delay_correction: false\n")
    config = load_config(path)
    for arguments, expected_global in (([], False), (["--sloppy"], True)):
        parsed = _build_parser().parse_args(["/tmp/bids", "/tmp/out", "participant", *arguments])
        merged = merge_cli_config(vars(parsed), config)
        dce = build_dce_config(merged)
        assert dce.sloppy is expected_global
        assert dce.model == "2cxm" and dce.delay_correction is False
        assert build_dsc_config(merged).sloppy is expected_global
        assert build_anat_config(merged).sloppy is expected_global
