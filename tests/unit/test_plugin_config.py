"""Tests for the --use-plugin loader.

Catches typos and shape errors at CLI parse time rather than at workflow
run-time where they'd silently degrade parallelism or surface as opaque
nipype errors.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pmriprep.cli.run import _load_plugin_config


def test_yaml_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "plugin.yaml"
    p.write_text(
        "plugin: SLURM\n"
        "plugin_args:\n"
        '  template: "#SBATCH --mem=8G"\n'
        '  sbatch_args: "--time=2:00:00"\n'
    )
    plugin, args = _load_plugin_config(p)
    assert plugin == "SLURM"
    assert args == {
        "template": "#SBATCH --mem=8G",
        "sbatch_args": "--time=2:00:00",
    }


def test_json_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "plugin.json"
    p.write_text(json.dumps({"plugin": "MultiProc", "plugin_args": {"n_procs": 4}}))
    plugin, args = _load_plugin_config(p)
    assert plugin == "MultiProc"
    assert args == {"n_procs": 4}


def test_no_plugin_args_is_empty_dict(tmp_path: Path) -> None:
    p = tmp_path / "plugin.yaml"
    p.write_text("plugin: Linear\n")
    plugin, args = _load_plugin_config(p)
    assert plugin == "Linear"
    assert args == {}


def test_rejects_unknown_extension(tmp_path: Path) -> None:
    p = tmp_path / "plugin.txt"
    p.write_text("plugin: foo")
    with pytest.raises(ValueError, match="must be .yaml/.yml/.json"):
        _load_plugin_config(p)


def test_rejects_missing_plugin_key(tmp_path: Path) -> None:
    p = tmp_path / "plugin.yaml"
    p.write_text("foo: bar\n")
    with pytest.raises(ValueError, match="must contain a top-level 'plugin' key"):
        _load_plugin_config(p)


def test_rejects_non_string_plugin(tmp_path: Path) -> None:
    p = tmp_path / "plugin.yaml"
    p.write_text("plugin:\n  - SLURM\n  - MultiProc\n")
    with pytest.raises(ValueError, match="'plugin' must be a string"):
        _load_plugin_config(p)


def test_rejects_non_mapping_plugin_args(tmp_path: Path) -> None:
    p = tmp_path / "plugin.yaml"
    p.write_text("plugin: SLURM\nplugin_args: not_a_mapping\n")
    with pytest.raises(ValueError, match="'plugin_args' must be a mapping"):
        _load_plugin_config(p)
