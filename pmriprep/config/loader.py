"""Configuration loading utilities for PMRIPrep."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml  # type: ignore[import-untyped]

from .anat import AnatConfig
from .dce import DCEConfig
from .dsc import DSCConfig
from .hdglio import HDGLIOConfig


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
        Configuration dictionary with sections: dsc, dce, anat, hdglio.
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

    # Ensure config is a dict (handle empty files)
    if config is None:
        config = {}

    # Ensure all sections exist
    for section in ("dsc", "dce", "anat", "hdglio"):
        if section not in config:
            config[section] = {}

    return dict(config)


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
    result = file_config.copy()

    # Map CLI args to config sections and keys
    cli_to_config = {
        # DSC options
        "dsc_deconv_method": ("dsc", "method"),
        "dsc_bayesian": ("dsc", "bayesian"),
        "dsc_preload": ("dsc", "preload_present"),
        "dsc_dose": ("dsc", "dose"),
        "dsc_aif_method": ("dsc", "aif_method"),
        # DCE options
        "dce_model": ("dce", "model"),
        "dce_bayesian": ("dce", "bayesian"),
        "dce_aif_method": ("dce", "aif_method"),
        "dce_no_b1": ("dce", "no_b1"),
        "dce_no_t1": ("dce", "no_t1"),
        "dce_dose": ("dce", "dose"),
        # Anatomical options
        "brain_extraction_tool": ("anat", "tool"),
        "denoise": ("anat", "denoise"),
        # HD-GLIO options
        "hdglio_conda_env": ("hdglio", "conda_env"),
        "hdglio_device": ("hdglio", "device"),
    }

    for cli_key, (section, config_key) in cli_to_config.items():
        if cli_key in cli_args and cli_args[cli_key] is not None:
            if section not in result:
                result[section] = {}
            result[section][config_key] = cli_args[cli_key]

    # --use-gpu sets use_gpu on both DSC and DCE configs
    if cli_args.get("use_gpu"):
        result.setdefault("dsc", {})["use_gpu"] = True
        result.setdefault("dce", {})["use_gpu"] = True

    # --hdbet-device sets hdbet_device on both DSC and DCE configs
    if cli_args.get("hdbet_device") is not None:
        result.setdefault("dsc", {})["hdbet_device"] = cli_args["hdbet_device"]
        result.setdefault("dce", {})["hdbet_device"] = cli_args["hdbet_device"]

    return result


def _dict_to_path(d: Dict[str, Any], key: str) -> None:
    """Convert a string value to Path in-place if key exists."""
    if key in d and d[key] is not None and isinstance(d[key], str):
        d[key] = Path(d[key])


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

    # Convert path strings to Path objects
    for path_key in ("aif_mask_file", "lesion_mask", "nawm_mask"):
        _dict_to_path(dsc_dict, path_key)

    # Type coerce numeric fields
    _coerce_int(dsc_dict, "pre_bolus")
    _coerce_int(dsc_dict, "mcmc_samples")
    _coerce_int(dsc_dict, "cpi_control_points")
    _coerce_int(dsc_dict, "n_aif_candidates")
    _coerce_int(dsc_dict, "num_processes")
    _coerce_float(dsc_dict, "smooth_fwhm")
    _coerce_float(dsc_dict, "lesion_distance")
    _coerce_float(dsc_dict, "motion_thresh")

    # Filter to only valid DSCConfig fields
    valid_fields = {f.name for f in DSCConfig.__dataclass_fields__.values()}
    filtered_dict = {k: v for k, v in dsc_dict.items() if k in valid_fields}

    return DSCConfig(**filtered_dict)


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

    # Flatten nested sections (e.g., processing.tsnr_threshold -> tsnr_threshold)
    for nested_key in ("processing", "registration", "aif"):
        nested = dce_dict.pop(nested_key, None)
        if isinstance(nested, dict):
            for k, v in nested.items():
                if k not in dce_dict:  # Top-level keys take priority
                    dce_dict[k] = v

    # Convert path strings to Path objects
    for path_key in ("lesion_mask", "aif_mask_file"):
        _dict_to_path(dce_dict, path_key)

    # Type coerce numeric fields
    _coerce_int(dce_dict, "baseline_frames")
    _coerce_int(dce_dict, "mcmc_samples")
    _coerce_int(dce_dict, "n_aif_candidates")
    _coerce_int(dce_dict, "num_processes")
    _coerce_float(dce_dict, "smooth_fwhm")
    _coerce_float(dce_dict, "lesion_distance")
    _coerce_float(dce_dict, "tsnr_threshold")
    _coerce_float(dce_dict, "hematocrit")
    _coerce_float(dce_dict, "target_peak")

    # Filter to only valid DCEConfig fields
    valid_fields = {f.name for f in DCEConfig.__dataclass_fields__.values()}
    filtered_dict = {k: v for k, v in dce_dict.items() if k in valid_fields}

    return DCEConfig(**filtered_dict)


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

    # Filter to only valid AnatConfig fields
    valid_fields = {f.name for f in AnatConfig.__dataclass_fields__.values()}
    filtered_dict = {k: v for k, v in anat_dict.items() if k in valid_fields}

    return AnatConfig(**filtered_dict)


def build_hdglio_config(config_dict: Dict[str, Any]) -> HDGLIOConfig:
    """Build HDGLIOConfig from a configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary with HD-GLIO parameters.

    Returns
    -------
    HDGLIOConfig
        Configured HDGLIOConfig instance.
    """
    hdglio_dict = config_dict.get("hdglio", {}).copy()

    # Filter to only valid HDGLIOConfig fields
    valid_fields = {f.name for f in HDGLIOConfig.__dataclass_fields__.values()}
    filtered_dict = {k: v for k, v in hdglio_dict.items() if k in valid_fields}

    return HDGLIOConfig(**filtered_dict)


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
        Dictionary with keys 'dsc', 'dce', 'anat', 'hdglio' containing
        the respective config objects.
    """
    # Load file config if provided
    if config_file is not None:
        file_config = load_config(config_file)
    else:
        file_config = {"dsc": {}, "dce": {}, "anat": {}, "hdglio": {}}

    # Merge CLI args if provided
    if cli_args is not None:
        merged_config = merge_cli_config(cli_args, file_config)
    else:
        merged_config = file_config

    return {
        "dsc": build_dsc_config(merged_config),
        "dce": build_dce_config(merged_config),
        "anat": build_anat_config(merged_config),
        "hdglio": build_hdglio_config(merged_config),
    }
