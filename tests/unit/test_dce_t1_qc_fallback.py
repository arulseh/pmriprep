"""Regression coverage for VFA T1 fallback QC figures."""

from __future__ import annotations

import inspect
from pathlib import Path

import matplotlib.axes
import matplotlib.figure
import numpy as np

from pmriprep.interfaces.dce import T1Mapping
from pmriprep.utils.dce_qc_plots import generate_vfa_t1_qc_plot


def test_t1_mapping_does_not_suppress_fallback_qc() -> None:
    """Rejected VFA fits remain connected to an explanatory QC figure."""
    source = inspect.getsource(T1Mapping._run_interface)

    assert "show_t1_qc" not in source
    assert "measured_t1_map=measured_t1_map" in source
    assert "fallback_brain_mask = brain_mask.copy()" in source
    assert "fallback_brain_mask," in source


def test_fallback_qc_distinguishes_measured_and_final_maps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Fallback QC labels the rejected measurement and final map separately."""
    shape = (8, 8, 5)
    grid = np.indices(shape, dtype=np.float64).sum(axis=0)
    vfa_data = [100.0 + grid, 180.0 + 2.0 * grid]
    measured = 650.0 + 80.0 * grid
    final = np.full(shape, 1090.0)
    mask = np.ones(shape, dtype=bool)
    titles: list[str] = []
    figure_text: list[str] = []
    original_set_title = matplotlib.axes.Axes.set_title
    original_figure_text = matplotlib.figure.Figure.text

    def _capture_title(self, label, *args, **kwargs):
        titles.append(str(label))
        return original_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", _capture_title)

    def _capture_figure_text(self, x, y, s, *args, **kwargs):
        figure_text.append(str(s))
        return original_figure_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "text", _capture_figure_text)
    out_file = tmp_path / "fallback_qc.png"

    result = generate_vfa_t1_qc_plot(
        vfa_data=vfa_data,
        flip_angles=[5.0, 15.0],
        t1_map=final,
        measured_t1_map=measured,
        brain_mask=mask,
        correction_info={
            "correction_applied": "fallback",
            "median_t1_original": 1800.0,
            "median_t1_corrected": 1090.0,
        },
        output_path=str(out_file),
    )

    assert Path(result).is_file()
    assert "Rejected Measured VFA T1" in titles
    assert "Final Uniform Fallback T1" in titles
    assert not any(text.startswith("Correction:") for text in figure_text)
