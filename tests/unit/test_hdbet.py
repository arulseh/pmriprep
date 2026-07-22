"""Tests for shared HD-BET utility and perfusion brain extraction interfaces."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
import pytest


# =============================================================================
# Helpers
# =============================================================================


def _make_nifti(path: str, data: np.ndarray) -> None:
    """Save a simple NIfTI image."""
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    nib.save(img, path)


# =============================================================================
# Tests: run_hdbet()
# =============================================================================


class TestRunHdbet:
    """Tests for the shared run_hdbet() utility function."""

    def test_success(self, tmp_path: Path) -> None:
        """Successful HD-BET run produces brain and mask."""
        from pmriprep.utils.hdbet import run_hdbet

        in_file = str(tmp_path / "input.nii.gz")
        out_brain = str(tmp_path / "brain.nii.gz")
        _make_nifti(in_file, np.random.rand(10, 10, 10))

        # Mock subprocess to simulate HD-BET writing output
        def fake_run(cmd, **kwargs):
            brain_data = np.random.rand(10, 10, 10).astype(np.float32)
            _make_nifti(out_brain, brain_data)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch("pmriprep.utils.hdbet.run_isolated_command", side_effect=fake_run):
            succeeded, mask_path = run_hdbet(
                in_file=in_file, out_brain=out_brain, device="cpu",
            )

        assert succeeded is True
        assert mask_path is not None
        assert Path(mask_path).exists()

        # Mask should be binary uint8
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        assert set(np.unique(mask_data)).issubset({0, 1})

    def test_called_process_error_logs_stderr(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """CalledProcessError should log stderr and return False."""
        from pmriprep.utils.hdbet import run_hdbet

        in_file = str(tmp_path / "input.nii.gz")
        out_brain = str(tmp_path / "brain.nii.gz")
        _make_nifti(in_file, np.random.rand(10, 10, 10))

        result = subprocess.CompletedProcess(
            ["hd-bet"], 1, "", "CUDA out of memory",
        )
        with patch("pmriprep.utils.hdbet.run_isolated_command", return_value=result):
            succeeded, mask_path = run_hdbet(
                in_file=in_file, out_brain=out_brain, device="cpu",
            )

        assert succeeded is False
        assert mask_path is None
        assert "CUDA out of memory" in caplog.text

    def test_timeout_returns_false(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """TimeoutExpired should return False."""
        from pmriprep.utils.hdbet import run_hdbet

        in_file = str(tmp_path / "input.nii.gz")
        out_brain = str(tmp_path / "brain.nii.gz")
        _make_nifti(in_file, np.random.rand(10, 10, 10))

        with patch(
            "pmriprep.utils.hdbet.run_isolated_command",
            side_effect=subprocess.TimeoutExpired("hd-bet", 300),
        ):
            succeeded, mask_path = run_hdbet(
                in_file=in_file, out_brain=out_brain, device="cpu",
            )

        assert succeeded is False
        assert "timed out" in caplog.text

    def test_file_not_found_returns_false(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """FileNotFoundError should return False."""
        from pmriprep.utils.hdbet import run_hdbet

        in_file = str(tmp_path / "input.nii.gz")
        out_brain = str(tmp_path / "brain.nii.gz")
        _make_nifti(in_file, np.random.rand(10, 10, 10))

        with patch(
            "pmriprep.utils.hdbet.run_isolated_command",
            side_effect=FileNotFoundError("hd-bet"),
        ):
            succeeded, mask_path = run_hdbet(
                in_file=in_file, out_brain=out_brain, device="cpu",
            )

        assert succeeded is False

    def test_os_error_returns_false(
        self, tmp_path: Path,
    ) -> None:
        """OSError (Apptainer) should return False."""
        from pmriprep.utils.hdbet import run_hdbet

        in_file = str(tmp_path / "input.nii.gz")
        out_brain = str(tmp_path / "brain.nii.gz")
        _make_nifti(in_file, np.random.rand(10, 10, 10))

        with patch(
            "pmriprep.utils.hdbet.run_isolated_command",
            side_effect=OSError("[Errno 11] Resource temporarily unavailable"),
        ):
            succeeded, _ = run_hdbet(
                in_file=in_file, out_brain=out_brain, device="cpu",
            )

        assert succeeded is False

    def test_empty_brain_guard(self, tmp_path: Path) -> None:
        """All-zero brain output should return False."""
        from pmriprep.utils.hdbet import run_hdbet

        in_file = str(tmp_path / "input.nii.gz")
        out_brain = str(tmp_path / "brain.nii.gz")
        _make_nifti(in_file, np.random.rand(10, 10, 10))

        # Mock subprocess to write an all-zero brain
        def fake_run(cmd, **kwargs):
            _make_nifti(out_brain, np.zeros((10, 10, 10)))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch("pmriprep.utils.hdbet.run_isolated_command", side_effect=fake_run):
            succeeded, mask_path = run_hdbet(
                in_file=in_file, out_brain=out_brain, device="cpu",
            )

        assert succeeded is False
        assert mask_path is None


# =============================================================================
# Tests: DSCBrainExtract
# =============================================================================


class TestDSCBrainExtract:
    """Tests for the DSCBrainExtract SimpleInterface."""

    def test_gpu_node_detection(self) -> None:
        """Node with use_gpu=True should be detected as GPU node."""
        import nipype.pipeline.engine as pe

        from pmriprep.interfaces.dsc_brain import DSCBrainExtract

        node = pe.Node(DSCBrainExtract(), name="test_hdbet")
        node.inputs.use_gpu = True
        assert node.is_gpu_node() is True

    def test_gpu_node_false_by_default(self) -> None:
        """Node with default use_gpu should NOT be GPU node."""
        import nipype.pipeline.engine as pe

        from pmriprep.interfaces.dsc_brain import DSCBrainExtract

        node = pe.Node(DSCBrainExtract(), name="test_hdbet")
        assert node.is_gpu_node() is False

    def test_fallback_chain_hdbet_fail_bet_succeed(
        self, tmp_path: Path,
    ) -> None:
        """When HD-BET fails, should fall back to BET."""
        from pmriprep.interfaces.dsc_brain import DSCBrainExtract

        in_file = str(tmp_path / "dsc_mean.nii.gz")
        _make_nifti(in_file, np.random.rand(10, 10, 10) * 1000)

        iface = DSCBrainExtract()
        iface.inputs.dsc_mean_file = in_file
        iface.inputs.hdbet_device = "cpu"

        # Mock _run_hdbet to fail, _run_bet to succeed
        with (
            patch.object(iface, "_run_hdbet", return_value=False),
            patch.object(iface, "_run_bet", return_value=True) as mock_bet,
            patch("os.getcwd", return_value=str(tmp_path)),
        ):
            # Create expected output files (as _run_bet would)
            _make_nifti(
                str(tmp_path / "dsc_brain.nii.gz"),
                np.random.rand(10, 10, 10),
            )
            _make_nifti(
                str(tmp_path / "dsc_brain_mask.nii.gz"),
                np.ones((10, 10, 10)),
            )
            iface._run_interface(MagicMock())

        mock_bet.assert_called_once()
        assert iface._results["method_used"] == "bet"

    def test_fallback_chain_all_fail_to_threshold(
        self, tmp_path: Path,
    ) -> None:
        """When HD-BET and BET both fail, should use threshold."""
        from pmriprep.interfaces.dsc_brain import DSCBrainExtract

        in_file = str(tmp_path / "dsc_mean.nii.gz")
        _make_nifti(in_file, np.random.rand(10, 10, 10) * 1000)

        iface = DSCBrainExtract()
        iface.inputs.dsc_mean_file = in_file
        iface.inputs.hdbet_device = "cpu"

        with (
            patch.object(iface, "_run_hdbet", return_value=False),
            patch.object(iface, "_run_bet", return_value=False),
            patch("os.getcwd", return_value=str(tmp_path)),
        ):
            iface._run_interface(MagicMock())

        assert iface._results["method_used"] == "threshold"
        assert Path(iface._results["dsc_brain_file"]).exists()
        assert Path(iface._results["dsc_brain_mask_file"]).exists()

    def test_output_trait_names(self) -> None:
        """Output trait names must match workflow expectations."""
        from pmriprep.interfaces.dsc_brain import DSCBrainExtractOutputSpec

        spec = DSCBrainExtractOutputSpec()
        assert hasattr(spec, "dsc_brain_file")
        assert hasattr(spec, "dsc_brain_mask_file")
        assert hasattr(spec, "method_used")

    def test_input_trait_names(self) -> None:
        """Input trait names must match workflow connections."""
        from pmriprep.interfaces.dsc_brain import DSCBrainExtractInputSpec

        spec = DSCBrainExtractInputSpec()
        assert hasattr(spec, "dsc_mean_file")
        assert hasattr(spec, "hdbet_device")
        assert hasattr(spec, "hdbet_conda_env")
        assert hasattr(spec, "use_gpu")


class TestDCEBrainExtract:
    """Tests for the DCE brain extraction interface contract."""

    def test_input_trait_names(self) -> None:
        """Inputs describe the native DCE extraction chain."""
        from pmriprep.interfaces.dce import DCEBrainExtractInputSpec

        spec = DCEBrainExtractInputSpec()
        assert hasattr(spec, "dce_mean")
        assert hasattr(spec, "hdbet_device")
        assert hasattr(spec, "hdbet_conda_env")
        assert hasattr(spec, "use_gpu")
        assert not hasattr(spec, "t1w_brain_mask")


class TestAnatomicalBrainExtraction:
    """Tests for the anatomical brain extraction interface contract."""

    def test_contract_contains_only_computed_qc(self) -> None:
        """The interface omits registration-dependent MNI overlap QC."""
        from pmriprep.interfaces.anat import (
            BrainExtractionInputSpec,
            BrainExtractionOutputSpec,
        )

        inputs = BrainExtractionInputSpec().traits()
        outputs = BrainExtractionOutputSpec().traits()
        assert "mni_template" not in inputs
        assert "mni_mask" not in inputs
        assert "dice_vs_mni" not in outputs
