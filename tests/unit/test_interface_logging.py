"""Regression tests for user-visible Nipype interface logging."""

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "pmriprep.interfaces.dce_compute",
        "pmriprep.interfaces.dsc_compute",
        "pmriprep.models.dsc.spatial_vb",
    ],
)
def test_execution_messages_use_nipype_interface_logger(module_name: str) -> None:
    """Long-running computation progress reaches the Nipype execution log."""
    import importlib

    module = importlib.import_module(module_name)
    assert module.logger.name == "nipype.interface"
