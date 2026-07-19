"""Characterization tests for DCE fit-input and domain preparation."""

from pathlib import Path

import nibabel as nib
import numpy as np

from pmriprep.interfaces.dce_compute import ComputeDCEMaps


def test_prepare_fit_inputs_preserves_model_time_and_domain_gates(
    tmp_path: Path,
) -> None:
    """Raw-signal QC and enhancement gates define the pre-BAT fitting domain."""
    shape = (2, 2, 1)
    concentration = np.zeros((*shape, 4), dtype=np.float32)
    concentration[0, 0, 0] = [0.0, 0.0, 0.2, 0.5]
    concentration[1, 0, 0] = [0.0, 0.0, 0.3, 0.7]
    concentration[1, 1, 0] = [0.0, 0.0, 0.4, 0.8]
    mask = np.ones(shape, dtype=np.uint8)
    mask[1, 0, 0] = 0

    signal = np.full((*shape, 4), 100.0, dtype=np.float32)
    signal[0, 0, 0] = [100.0, 102.0, 110.0, 125.0]
    signal[1, 0, 0] = [100.0, 102.0, 110.0, 125.0]

    concentration_file = tmp_path / "concentration.nii.gz"
    mask_file = tmp_path / "mask.nii.gz"
    signal_file = tmp_path / "signal.nii.gz"
    aif_file = tmp_path / "aif.txt"
    nib.save(nib.Nifti1Image(concentration, np.eye(4)), concentration_file)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_file)
    nib.save(nib.Nifti1Image(signal, np.eye(4)), signal_file)
    np.savetxt(aif_file, [0.0, 0.0, 1.0, 0.5])

    interface = ComputeDCEMaps(
        concentration_file=str(concentration_file),
        brain_mask_file=str(mask_file),
        dce_signal_file=str(signal_file),
        aif_file=str(aif_file),
        volume_tr=2.0,
        model="auto",
        baseline_frames=2,
        tsnr_threshold=10.0,
        inference="nlls",
        mcmc_samples=123,
        save_fitted_curves=True,
    )

    prepared = interface._prepare_fit_inputs()

    np.testing.assert_allclose(prepared.time_points, [0.0, 1 / 30, 2 / 30, 0.1])
    assert prepared.selection_tr_seconds == 2.0
    assert prepared.is_auto is True
    assert prepared.candidate_names == ["patlak", "extended_tofts", "2cxm"]
    assert prepared.model_name == "2cxm"
    assert prepared.baseline_frames == 2
    assert prepared.inference == "nlls"
    assert prepared.mcmc_samples == 123
    assert prepared.save_fitted_curves is True
    expected_domain = np.zeros(shape, dtype=bool)
    expected_domain[0, 0, 0] = True
    np.testing.assert_array_equal(prepared.fitting_mask, expected_domain)
    assert prepared.tsnr_map[0, 0, 0] > 10.0
    assert prepared.peak_enhancement is not None
    assert prepared.peak_enhancement[0, 0, 0] > 0.0
