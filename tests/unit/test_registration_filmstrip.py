"""Regression coverage for registration-filmstrip boundary overlays."""

from __future__ import annotations

from pathlib import Path

import matplotlib.axes
import numpy as np
import pytest

from pmriprep.utils.qc_standards import COLORBLIND_PALETTE
from pmriprep.utils.qc_utils import plot_registration_qc_filmstrip


def test_registration_filmstrip_compares_fixed_and_moving_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both explicit masks are drawn without clipping the moving image."""
    fixed = np.ones((7, 7, 7), dtype=float)
    moving = np.full((7, 7, 7), 10.0, dtype=float)
    moving[0, :, :] = 20.0
    fixed_mask = np.zeros_like(fixed)
    fixed_mask[1:6, 1:6, 1:6] = 1
    moving_mask = np.zeros_like(fixed)
    moving_mask[2:7, 1:6, 1:6] = 1

    calls: list[tuple[np.ndarray, dict[str, object]]] = []
    original_contour = matplotlib.axes.Axes.contour

    def _record_contour(self, data, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((np.asarray(data).copy(), dict(kwargs)))
        return original_contour(self, data, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "contour", _record_contour)
    output = plot_registration_qc_filmstrip(
        fixed,
        moving,
        tmp_path / "registration.png",
        mask=fixed_mask,
        moving_mask=moving_mask,
        n_slices=1,
    )

    assert output.is_file()
    colors = [call[1].get("colors") for call in calls]
    assert colors.count([COLORBLIND_PALETTE.secondary]) == 3
    assert colors.count([COLORBLIND_PALETTE.warning]) == 3
    assert colors.count([COLORBLIND_PALETTE.success]) == 3
    moving_calls = [call for call in calls if call[1].get("colors") == [COLORBLIND_PALETTE.secondary]]
    assert any(np.max(data) == 20.0 for data, _kwargs in moving_calls)


def test_registration_filmstrip_rejects_moving_mask_grid_mismatch(
    tmp_path: Path,
) -> None:
    """A moving mask must already be registered to the displayed grid."""
    image = np.ones((5, 5, 5), dtype=float)
    with pytest.raises(ValueError, match="Moving mask shape"):
        plot_registration_qc_filmstrip(
            image,
            image,
            tmp_path / "registration.png",
            moving_mask=np.ones((4, 5, 5), dtype=float),
        )
