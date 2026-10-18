"""AIF failure figures pair diagnostic curves with their spatial support."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


@pytest.mark.parametrize("modality", ["DSC", "DCE"])
def test_rejected_aif_has_curve_and_location(tmp_path, monkeypatch, modality) -> None:
    from matplotlib.figure import Figure
    from pmriprep.interfaces.dce_aif_qc import DCEAIFQCPlot
    from pmriprep.interfaces.dsc_aif_qc import AIFQCPlot

    time = np.arange(40, dtype=float)
    curve = np.exp(-0.5 * ((time - 12) / 3) ** 2)
    background = 800 + 10 * np.indices((10, 10, 10)).sum(axis=0)
    signal = background[..., None] - 100 * curve
    mask = np.ones((10, 10, 10), dtype=np.uint8)
    rejected_mask = np.zeros_like(mask)
    rejected_mask[4:6, 4:6, 4:6] = 1
    for name, data in (("signal", signal), ("brain", mask), ("rejected", rejected_mask)):
        nib.save(nib.Nifti1Image(data, np.eye(4)), tmp_path / f"{name}.nii.gz")
    np.savetxt(tmp_path / "curve.txt", curve)
    captured = []
    save = Figure.savefig

    def capture(fig, *args, **kwargs):
        captured.append([(ax.get_title(), [line.get_label() for line in ax.lines]) for ax in fig.axes])
        for ax in fig.axes:
            if len(ax.images) == 2:
                overlay = ax.images[1]
                assert overlay.norm(1) > overlay.norm(0)
        return save(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    common = dict(
        brain_mask_file=str(tmp_path / "brain.nii.gz"),
        rejected_aif_curve_file=str(tmp_path / "curve.txt"),
        rejected_aif_mask_file=str(tmp_path / "rejected.nii.gz"),
        tr=1.5,
        qc_metrics={"detection_method": "unavailable" if modality == "DSC" else "population",
                    "detection_status": "failed", "baseline_frames": 5,
                    "rejected_candidate": {"reason": "quality_checks_failed",
                                           "candidate_coords": [[4, 4, 4]],
                                           "candidate_scores": [0.4]}},
    )
    if modality == "DSC":
        interface = AIFQCPlot(dsc_file=str(tmp_path / "signal.nii.gz"), **common)
    else:
        np.savetxt(tmp_path / "population.txt", curve * 2)
        interface = DCEAIFQCPlot(
            dce_file=str(tmp_path / "signal.nii.gz"),
            aif_curve_file=str(tmp_path / "population.txt"), **common,
        )
    outputs = interface.run(cwd=str(tmp_path)).outputs
    for path in (outputs.out_aif_plot, outputs.out_aif_location):
        assert Path(path).stat().st_size > 1000
    curve_title, labels = captured[0][0]
    assert curve_title.strip().endswith(f"{modality} AIF")
    assert "Rejected candidate" in labels
    if modality == "DCE":
        assert "Population AIF (used)" in labels
