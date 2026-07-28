"""Tests for DSC parameter-map QC display semantics."""

from pathlib import Path

import matplotlib.figure
import numpy as np
import pytest

from pmriprep.utils.dsc_qc_plots import generate_dsc_parameter_maps_qc_plot


def test_dsc_map_labels_are_horizontal_and_nonfinite_support_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The montage reports finite support and renders non-finite voxels separately."""
    captured: dict[str, matplotlib.figure.Figure] = {}

    def capture_figure(
        figure: matplotlib.figure.Figure,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        captured["figure"] = figure

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_figure)
    brain = np.ones((5, 5, 5), dtype=bool)
    rcbf = np.full(brain.shape, 1.0)
    rcbf[0, 0, 0] = np.nan
    rcbf[0, 0, 1] = np.inf

    generate_dsc_parameter_maps_qc_plot(
        {"rCBF": rcbf},
        brain,
        subject_id="01",
        output_path=str(tmp_path / "dsc.png"),
    )

    figure = captured["figure"]
    row_labels = [
        text
        for axis in figure.axes
        for text in axis.texts
        if text.get_text().startswith("rCBF")
    ]
    assert len(row_labels) == 1
    assert "support=98.4%" in row_labels[0].get_text()
    assert row_labels[0].get_rotation() == 0.0

    image_axes = [axis for axis in figure.axes if len(axis.images) == 2]
    assert len(image_axes) == 3
    masked_voxels = 0
    for axis in image_axes:
        plotted = axis.images[1].get_array()
        assert np.ma.isMaskedArray(plotted)
        masked_voxels += int(np.count_nonzero(np.ma.getmaskarray(plotted)))
    assert masked_voxels > 0
