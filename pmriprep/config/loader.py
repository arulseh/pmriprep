"""Configuration loading utilities for PMRIPrep."""

from copy import deepcopy
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml  # type: ignore[import-untyped]

from .anat import AnatConfig
from .dce import DCEConfig
from .dsc import DSCConfig
from .tumor_seg import TumorSegConfig


_CONFIG_CLASSES = {
    "dsc": DSCConfig,
    "dce": DCEConfig,
    "anat": AnatConfig,
    "tumor_seg": TumorSegConfig,
}


def _validate_config_mapping(config: Any) -> Dict[str, Any]:
    """Validate the public configuration-file structure.

    Parameters
    ----------
    config : Any
        Parsed YAML or JSON object.

    Returns
    -------
    dict
        Validated configuration mapping.

    Raises
    ------
    ValueError
        If the root, section names, section values, or option names are invalid.
    """
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")

    unknown_sections = set(config) - set(_CONFIG_CLASSES)
    if unknown_sections:
        raise ValueError(
            "Unknown configuration section(s): "
            f"{sorted(unknown_sections)}. Valid sections: {sorted(_CONFIG_CLASSES)}"
        )

    result = dict(config)
    for section, config_class in _CONFIG_CLASSES.items():
        section_value = result.setdefault(section, {})
        if not isinstance(section_value, dict):
            raise ValueError(f"Configuration section {section!r} must be a mapping.")
        valid_options = {item.name for item in fields(config_class)}
        unknown_options = set(section_value) - valid_options
        if unknown_options:
            raise ValueError(
                f"Unknown {section} configuration option(s): "
                f"{sorted(unknown_options)}. Valid options: {sorted(valid_options)}"
            )
    return result


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML or JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the configuration file. File format is determined
        by extension (.yaml, .yml, or .json).

    Returns
    -------
    dict
        Configuration dictionary with sections: dsc, dce, anat, tumor_seg.
        Missing sections are returned as empty dicts.

    Raises
    ------
    ValueError
        If the file extension is not .yaml, .yml, or .json.
    FileNotFoundError
        If the configuration file does not exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    elif suffix == ".json":
        with open(path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError(
            f"Unsupported configuration file format: {suffix}. "
            "Use .yaml, .yml, or .json."
        )

    return _validate_config_mapping(config)


def _merge_nested_dict(
    base: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively merge override dict into base dict.

    Parameters
    ----------
    base : dict
        Base dictionary to merge into.
    override : dict
        Override dictionary with values to apply.

    Returns
    -------
    dict
        Merged dictionary where override values take precedence.
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _merge_nested_dict(result[key], value)
        else:
            result[key] = value
    return dict(result)


def merge_cli_config(
    cli_args: Dict[str, Any], file_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge CLI arguments with file configuration.

    CLI arguments take precedence over file configuration values.
    Only non-None CLI arguments override file values.

    Parameters
    ----------
    cli_args : dict
        Dictionary of CLI argument values.
    file_config : dict
        Dictionary loaded from configuration file.

    Returns
    -------
    dict
        Merged configuration dictionary with CLI values taking precedence.
    """
    # Copy nested sections as well as the root: CLI precedence must not mutate
    # the loaded file configuration used for explicit-value provenance.
    result = deepcopy(file_config)

    # Map CLI args to config sections and keys
    cli_to_config = {
        # DSC options
        "dsc_deconv_method": ("dsc", "method"),
        "dsc_inference": ("dsc", "inference"),
        "dsc_preload": ("dsc", "preload"),
        "dsc_dose": ("dsc", "dose"),
        "dsc_agent_standard_dose": ("dsc", "agent_standard_dose"),
        "dsc_aif_method": ("dsc", "aif_method"),
        "dsc_aif_file": ("dsc", "aif_file"),
        # DCE options
        "dce_model": ("dce", "model"),
        "dce_inference": ("dce", "inference"),
        "dce_aif_method": ("dce", "aif_method"),
        "dce_aif_file": ("dce", "aif_file"),
        "dce_b1_method": ("dce", "b1_method"),
        "dce_b1_map_encoding": ("dce", "b1_map_encoding"),
        "dce_no_t1": ("dce", "no_t1"),
        "dce_t1_map": ("dce", "t1_map_file"),
        "dce_dose": ("dce", "dose"),
        "dce_r1": ("dce", "r1_relaxivity"),
        # Anatomical options
        "brain_extraction_tool": ("anat", "tool"),
        "denoise": ("anat", "denoise"),
        "hdbet_conda_env": ("anat", "hdbet_conda_env"),
        # Tumor segmentation options
        "tumor_seg_conda_env": ("tumor_seg", "conda_env"),
        "tumor_seg_device": ("tumor_seg", "device"),
    }

    for cli_key, (section, config_key) in cli_to_config.items():
        if cli_key in cli_args and cli_args[cli_key] is not None:
            if section not in result:
                result[section] = {}
            result[section][config_key] = cli_args[cli_key]

    # --hdbet-device sets hdbet_device on DSC, DCE, and anat configs
    if cli_args.get("hdbet_device") is not None:
        result.setdefault("dsc", {})["hdbet_device"] = cli_args["hdbet_device"]
        result.setdefault("dce", {})["hdbet_device"] = cli_args["hdbet_device"]
        result.setdefault("anat", {})["hdbet_device"] = cli_args["hdbet_device"]

    # --hdbet-conda-env propagates to DSC and DCE configs
    if cli_args.get("hdbet_conda_env") is not None:
        result.setdefault("dsc", {})["hdbet_conda_env"] = cli_args["hdbet_conda_env"]
        result.setdefault("dce", {})["hdbet_conda_env"] = cli_args["hdbet_conda_env"]

    # --sloppy sets sloppy on DSC, DCE, and anat configs
    if cli_args.get("sloppy"):
        result.setdefault("dsc", {})["sloppy"] = True
        result.setdefault("dce", {})["sloppy"] = True
        result.setdefault("anat", {})["sloppy"] = True

    return result


def _dict_to_path(d: Dict[str, Any], key: str) -> None:
    """Convert a string value to an absolute Path in-place if key exists.

    Relative paths are resolved against the current working directory
    so that nipype nodes can locate them regardless of execution directory.
    """
    if key in d and d[key] is not None and isinstance(d[key], (str, Path)):
        d[key] = Path(d[key]).expanduser().resolve()


def validate_config_paths(
    dsc_config: DSCConfig,
    dce_config: DCEConfig,
) -> None:
    """Validate configured external files before workflow construction.

    Parameters
    ----------
    dsc_config : DSCConfig
        Resolved DSC configuration.
    dce_config : DCEConfig
        Resolved DCE configuration.

    Raises
    ------
    FileNotFoundError
        If a configured external file does not exist.
    ValueError
        If an external AIF uses an unsupported file extension.
    """
    path_options = {
        "dsc": (
            dsc_config,
            ("aif_file", "aif_mask_file", "lesion_mask", "nawm_mask", "roi_mask"),
        ),
        "dce": (
            dce_config,
            (
                "aif_file",
                "aif_mask_file",
                "lesion_mask",
                "nawm_mask",
                "roi_mask",
                "t1_map_file",
            ),
        ),
    }
    for section, (config, options) in path_options.items():
        for option in options:
            path = getattr(config, option)
            if path is not None and not path.is_file():
                raise FileNotFoundError(
                    f"Configured {section}.{option} does not exist: {path}"
                )

    valid_aif_extensions = {".txt", ".tsv", ".npz", ".npy"}
    for section, config in (("dsc", dsc_config), ("dce", dce_config)):
        if (
            config.aif_file is not None
            and config.aif_file.suffix.lower() not in valid_aif_extensions
        ):
            raise ValueError(
                f"Configured {section}.aif_file has unsupported extension "
                f"{config.aif_file.suffix!r}. Supported: "
                f"{sorted(valid_aif_extensions)}"
            )


def _coerce_int(d: Dict[str, Any], key: str) -> None:
    """Convert a value to int in-place if key exists and is not None."""
    if key in d and d[key] is not None:
        try:
            d[key] = int(d[key])
        except (ValueError, TypeError):
            pass  # Leave as-is if conversion fails


def _coerce_float(d: Dict[str, Any], key: str) -> None:
    """Convert a value to float in-place if key exists and is not None."""
    if key in d and d[key] is not None:
        try:
            d[key] = float(d[key])
        except (ValueError, TypeError):
            pass  # Leave as-is if conversion fails


def build_dsc_config(config_dict: Dict[str, Any]) -> DSCConfig:
    """Build DSCConfig from a configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary with DSC parameters.

    Returns
    -------
    DSCConfig
        Configured DSCConfig instance.
    """
    dsc_dict = config_dict.get("dsc", {}).copy()
    valid_fields = {f.name for f in DSCConfig.__dataclass_fields__.values()}
    unknown_fields = set(dsc_dict) - valid_fields
    if unknown_fields:
        raise ValueError(
            f"Unknown dsc configuration option(s): {sorted(unknown_fields)}"
        )

    # Convert path strings to Path objects
    for path_key in ("aif_file", "aif_mask_file", "lesion_mask", "nawm_mask", "roi_mask"):
        _dict_to_path(dsc_dict, path_key)

    # Type coerce numeric fields
    _coerce_int(dsc_dict, "pre_bolus")
    if isinstance(dsc_dict.get("cpi_control_points"), str):
        _coerce_int(dsc_dict, "cpi_control_points")
    _coerce_int(dsc_dict, "n_aif_candidates")
    _coerce_int(dsc_dict, "num_processes")
    _coerce_float(dsc_dict, "smooth_fwhm")
    _coerce_float(dsc_dict, "lesion_distance")
    _coerce_float(dsc_dict, "motion_thresh")

    # Filter to only valid DSCConfig fields
    return DSCConfig(**dsc_dict)


def build_dce_config(config_dict: Dict[str, Any]) -> DCEConfig:
    """Build DCEConfig from a configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary with DCE parameters.

    Returns
    -------
    DCEConfig
        Configured DCEConfig instance.
    """
    dce_dict = config_dict.get("dce", {}).copy()

    # Convert path strings to Path objects
    for path_key in ("lesion_mask", "nawm_mask", "aif_mask_file", "aif_file", "roi_mask", "t1_map_file"):
        _dict_to_path(dce_dict, path_key)

    # Type coerce numeric fields
    _coerce_int(dce_dict, "baseline_frames")
    _coerce_int(dce_dict, "mcmc_samples")
    _coerce_int(dce_dict, "n_aif_candidates")
    _coerce_int(dce_dict, "num_processes")
    _coerce_float(dce_dict, "smooth_fwhm")
    _coerce_float(dce_dict, "lesion_distance")
    _coerce_float(dce_dict, "tsnr_threshold")
    _coerce_float(dce_dict, "motion_thresh")
    _coerce_float(dce_dict, "hematocrit")
    _coerce_float(dce_dict, "target_peak")
    _coerce_float(dce_dict, "r1_relaxivity")
    _coerce_float(dce_dict, "ser_early_time_s")
    _coerce_float(dce_dict, "ser_late_time_s")

    # Reject unknown keys after supported nested sections are flattened.
    valid_fields = {f.name for f in DCEConfig.__dataclass_fields__.values()}
    unknown_fields = set(dce_dict) - valid_fields
    if unknown_fields:
        raise ValueError(
            f"Unknown dce configuration option(s): {sorted(unknown_fields)}"
        )

    return DCEConfig(**dce_dict)


def build_anat_config(config_dict: Dict[str, Any]) -> AnatConfig:
    """Build AnatConfig from a configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary with anatomical parameters.

    Returns
    -------
    AnatConfig
        Configured AnatConfig instance.
    """
    anat_dict = config_dict.get("anat", {}).copy()

    # Reject unknown options instead of silently reverting to defaults.
    valid_fields = {f.name for f in AnatConfig.__dataclass_fields__.values()}
    unknown_fields = set(anat_dict) - valid_fields
    if unknown_fields:
        raise ValueError(
            f"Unknown anat configuration option(s): {sorted(unknown_fields)}"
        )

    return AnatConfig(**anat_dict)


def build_tumor_seg_config(config_dict: Dict[str, Any]) -> TumorSegConfig:
    """Build TumorSegConfig from a configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary with tumor segmentation parameters.

    Returns
    -------
    TumorSegConfig
        Configured TumorSegConfig instance.
    """
    tumor_seg_dict = config_dict.get("tumor_seg", {}).copy()

    # Reject unknown options instead of silently reverting to defaults.
    valid_fields = {f.name for f in TumorSegConfig.__dataclass_fields__.values()}
    unknown_fields = set(tumor_seg_dict) - valid_fields
    if unknown_fields:
        raise ValueError(
            "Unknown tumor_seg configuration option(s): "
            f"{sorted(unknown_fields)}"
        )

    return TumorSegConfig(**tumor_seg_dict)


def build_all_configs(
    cli_args: Optional[Dict[str, Any]] = None,
    config_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Build all configuration objects from CLI args and config file.

    Parameters
    ----------
    cli_args : dict, optional
        Dictionary of CLI argument values.
    config_file : str or Path, optional
        Path to configuration file.

    Returns
    -------
    dict
        Dictionary with keys 'dsc', 'dce', 'anat', 'tumor_seg' containing
        the respective config objects.
    """
    # Load file config if provided
    if config_file is not None:
        file_config = load_config(config_file)
    else:
        file_config = {"dsc": {}, "dce": {}, "anat": {}, "tumor_seg": {}}

    # Merge CLI args if provided
    if cli_args is not None:
        merged_config = merge_cli_config(cli_args, file_config)
    else:
        merged_config = file_config

    return {
        "dsc": build_dsc_config(merged_config),
        "dce": build_dce_config(merged_config),
        "anat": build_anat_config(merged_config),
        "tumor_seg": build_tumor_seg_config(merged_config),
    }
