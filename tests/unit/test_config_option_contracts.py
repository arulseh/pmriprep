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


def test_load_config_accepts_supported_dce_nested_sections(tmp_path: Path) -> None:
    """Legacy supported DCE grouping remains a valid input structure."""
    path = _write_config(
        tmp_path / "config.yaml",
        "dce:\n  processing:\n    tsnr_threshold: 8\n  aif:\n    aif_method: population\n",
    )

    config = load_config(path)

    assert config["dce"]["processing"]["tsnr_threshold"] == 8


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
        (lambda: DCEConfig(aif_bat_alignment="on"), "Invalid aif_bat_alignment"),
        (lambda: DCEConfig(t1_method="linear"), "Invalid t1_method"),
        (lambda: DCEConfig(t1_correction_mode="uniform"), "Invalid t1_correction_mode"),
        (lambda: DSCConfig(method="osvd"), "Invalid method"),
        (lambda: DSCConfig(inference="mcmc"), "Invalid inference"),
        (lambda: DSCConfig(aif_method="population"), "Invalid aif_method"),
        (lambda: AnatConfig(tool="bet"), "Invalid tool"),
        (lambda: TumorSegConfig(device="auto"), "Invalid device"),
    ],
)
def test_invalid_categorical_options_fail_before_workflow(factory, match: str) -> None:
    """Categorical YAML values must fail during config construction."""
    with pytest.raises(ValueError, match=match):
        factory()


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
        (["--dsc-dose", "half"], "dsc", "dose", 0.5),
        (["--dsc-agent-standard-dose", "0.05"], "dsc", "agent_standard_dose", 0.05),
        (["--dsc-aif-method", "synthetic"], "dsc", "aif_method", "synthetic"),
        (["--dsc-registration", "flirt_bbr"], "dsc", "registration", "flirt_bbr"),
        (["--dce-model", "patlak"], "dce", "model", "patlak"),
        (["--dce-inference", "vi"], "dce", "inference", "vi"),
        (["--dce-aif-method", "population"], "dce", "aif_method", "population"),
        (["--dce-dose", "double"], "dce", "dose", 2.0),
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
        {"dce": {"model": "extended_tofts"}, "dsc": {"method": "vascular"}},
    )

    assert build_dce_config(merged).model == "extended_tofts"
    assert build_dsc_config(merged).method == "vascular"


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
