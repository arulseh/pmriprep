"""Tests for DCE PK parameter-map QC display semantics."""

from pathlib import Path

import matplotlib.figure
import numpy as np
import pytest

from pmriprep.utils.dce_pk_qc_plots import (
    _finite_parameter_support,
    generate_pk_parameter_maps_qc_plot,
)


def test_finite_parameter_support_is_independent_of_quantitative_qc() -> None:
    """The montage retains finite estimates rather than applying the QC mask."""
    brain = np.ones((2, 2, 1), dtype=bool)
    parameter = np.array([[[0.1], [np.nan]], [[0.2], [0.0]]])

    support = _finite_parameter_support(parameter, brain)

    assert np.array_equal(
        support,
        np.array([[[True], [False]], [[True], [True]]]),
    )


def test_finite_parameter_support_rejects_shape_mismatch() -> None:
    """Parameter and brain-mask geometry must agree."""
    with pytest.raises(ValueError, match="Parameter map shape"):
        _finite_parameter_support(
            np.ones((2, 2, 2)),
            np.ones((2, 2, 1), dtype=bool),
        )


def test_pk_map_labels_are_horizontal_and_report_separate_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sparse quantitative QC does not collapse the displayed map support."""
    captured: dict[str, matplotlib.figure.Figure] = {}

    def capture_figure(
        figure: matplotlib.figure.Figure,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        captured["figure"] = figure

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_figure)
    brain = np.ones((5, 5, 5), dtype=bool)
    ktrans = np.full(brain.shape, 0.1)
    qc_mask = np.zeros(brain.shape, dtype=np.uint8)
    qc_mask[0, 0, 0] = 1

    generate_pk_parameter_maps_qc_plot(
        {"Ktrans": ktrans},
        brain,
        qc_mask=qc_mask,
        subject_id="01",
        output_path=str(tmp_path / "pk.png"),
    )

    figure = captured["figure"]
    assert figure._suptitle.get_text().endswith("PK Parameter Maps")
    assert "QC" not in figure._suptitle.get_text()
    assert all("finite retained estimate" not in text.get_text() for text in figure.texts)
    row_labels = [
        text
        for axis in figure.axes
        for text in axis.texts
        if text.get_text().startswith("Ktrans")
    ]
    assert len(row_labels) == 1
    assert "support=100.0%" in row_labels[0].get_text()
    assert row_labels[0].get_rotation() == 0.0
