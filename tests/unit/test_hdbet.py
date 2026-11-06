"""Tests for shared HD-BET utility and perfusion brain extraction interfaces."""

import subprocess
import os
import shlex
import sys
from types import SimpleNamespace
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


@pytest.mark.parametrize("conda_env", ["", "hdbet-runtime"])
def test_bounded_command_shared_by_anatomy_and_perfusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conda_env: str,
) -> None:
    """Both launch paths preserve the tool environment, threads and path quoting."""
    from pmriprep.interfaces.anat import BrainExtraction
    from pmriprep.utils.hdbet import build_hdbet_command

    executable = tmp_path / "hd-bet"
    executable.write_text(f"#!{sys.executable}\n")
    monkeypatch.setattr("pmriprep.utils.hdbet.shutil.which", lambda _: str(executable))
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    image = tmp_path / "input image.nii.gz"
    _make_nifti(str(image), np.ones((3, 3, 3)))
    interface = BrainExtraction(in_file=str(image), device="cpu", conda_env=conda_env)
    command = build_hdbet_command(str(image), interface._gen_outfilename(), "cpu", False, conda_env)
    if conda_env:
        assert command[:6] == ["conda", "run", "-n", conda_env, "--no-capture-output", "python"]
    else:
        assert command[0] == sys.executable
    assert command[command.index("--threads") + 1] == "3"
    shell_command = shlex.split(interface.cmdline)[2]
    assert shlex.split(shell_command.split(" && ", 1)[0]) == command


def test_bounded_runner_preserves_predictor_and_masking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reset HD-BET's host-thread override and bound the two auxiliary workers."""
    from pmriprep.utils import hdbet_runner

    threads = []
    prediction = {}
    initialization = {}
    source = tmp_path / "input.nii.gz"
    output = tmp_path / "brain.nii.gz"
    _make_nifti(str(source), np.ones((3, 3, 3)))

    def predict(inputs, masks, **kwargs):
        assert inputs == [[str(source)]]
        assert threads[-1] == 3
        assert os.environ["OMP_NUM_THREADS"] == "1"
        prediction.update(kwargs)
        _make_nifti(masks[0], np.ones((3, 3, 3)))

    def get_predictor(**kwargs):
        initialization.update(kwargs)
        threads.append(128)
        return SimpleNamespace(predict_from_files=predict)

    def apply_bet(image, mask, out):
        data = nib.load(image).get_fdata() * (nib.load(mask).get_fdata() > 0)
        _make_nifti(out, data)

    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(sys, "argv", ["runner", "-i", str(source), "-o", str(output),
                                      "-device", "cpu", "--disable_tta", "--threads", "3"])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=threads.append, device=str))
    monkeypatch.setitem(sys.modules, "SimpleITK", SimpleNamespace(ProcessObject=SimpleNamespace(
        SetGlobalDefaultNumberOfThreads=lambda n: None,
    )))
    monkeypatch.setitem(sys.modules, "HD_BET.checkpoint_download", SimpleNamespace(maybe_download_parameters=lambda: None))
    monkeypatch.setitem(sys.modules, "HD_BET.hd_bet_prediction", SimpleNamespace(
        apply_bet=apply_bet, get_hdbet_predictor=get_predictor,
    ))
    hdbet_runner.main()
    assert threads == [3, 128, 3]
    assert initialization == {"use_tta": False, "device": "cpu", "verbose": False}
    assert prediction == dict(save_probabilities=False, overwrite=True,
                              num_processes_preprocessing=1, num_processes_segmentation_export=1,
                              folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    np.testing.assert_array_equal(nib.load(output).get_fdata(), nib.load(source).get_fdata())
    assert not (tmp_path / "brain_bet.nii.gz").exists()


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

    @pytest.mark.parametrize(
        ("device", "sloppy", "expects_disabled"),
        [
            ("cpu", False, True),
            ("cuda", False, False),
            ("mps", False, False),
            ("cuda", True, True),
            ("mps", True, True),
        ],
    )
    def test_tta_command_policy(
        self,
        tmp_path: Path,
        device: str,
        sloppy: bool,
        expects_disabled: bool,
    ) -> None:
        """Normal GPU execution retains TTA; CPU and sloppy disable it."""
        from pmriprep.utils.hdbet import run_hdbet

        in_file = str(tmp_path / f"input_{device}_{sloppy}.nii.gz")
        out_brain = str(tmp_path / f"brain_{device}_{sloppy}.nii.gz")
        _make_nifti(in_file, np.ones((3, 3, 3)))
        captured: list[str] = []
        captured_timeout: list[int] = []

        def fake_run(cmd, **kwargs):
            captured.extend(cmd)
            captured_timeout.append(kwargs["timeout"])
            _make_nifti(out_brain, np.ones((3, 3, 3)))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch("pmriprep.utils.hdbet.run_isolated_command", side_effect=fake_run):
            succeeded, _ = run_hdbet(
                in_file=in_file,
                out_brain=out_brain,
                device=device,
                sloppy=sloppy,
            )

        assert succeeded
        assert ("--disable_tta" in captured) is expects_disabled
        assert captured_timeout == [300 if expects_disabled else 2400]

    def test_auto_device_controls_tta(self) -> None:
        """Auto resolution applies the policy to the concrete device."""
        from pmriprep.utils.hdbet import resolve_hdbet_execution

        with patch("pmriprep.utils.device.detect_torch_device", return_value="cuda"):
            assert resolve_hdbet_execution("auto") == ("cuda", True)

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
        assert iface._results["qc_metrics"]["brain_voxels"] == 1000
        assert iface._results["qc_metrics"]["brain_volume_ml"] == pytest.approx(1.0)

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


    def test_hdbet_qc_records_execution_policy(self, tmp_path: Path) -> None:
        """Successful DSC HD-BET records its concrete TTA policy."""
        from pmriprep.interfaces.dsc_brain import DSCBrainExtract

        in_file = str(tmp_path / "dsc_mean.nii.gz")
        _make_nifti(in_file, np.ones((3, 3, 3)))
        interface = DSCBrainExtract(
            dsc_mean_file=in_file,
            hdbet_device="mps",
        )
        def write_mask(*args: object) -> bool:
            _make_nifti(str(args[2]), np.ones((3, 3, 3)))
            return True

        with (
            patch.object(interface, "_run_hdbet", side_effect=write_mask),
            patch("os.getcwd", return_value=str(tmp_path)),
        ):
            interface._run_interface(MagicMock())

        qc = interface._results["qc_metrics"]
        assert qc == {
            "method_used": "hdbet",
            "brain_voxels": 27,
            "brain_volume_ml": 0.027,
            "hdbet_device": "mps",
            "hdbet_tta_used": True,
        }


class TestDCEBrainExtract:
    """Tests for the DCE brain extraction interface contract."""


    def test_hdbet_qc_records_execution_policy(self, tmp_path: Path) -> None:
        """Successful DCE HD-BET records its concrete TTA policy."""
        from pmriprep.interfaces.dce import DCEBrainExtract

        in_file = str(tmp_path / "dce_mean.nii.gz")
        _make_nifti(in_file, np.ones((3, 3, 3)))
        interface = DCEBrainExtract(
            dce_mean=in_file,
            hdbet_device="cuda",
            sloppy=True,
            fill_holes=False,
        )
        with (
            patch.object(interface, "_run_hdbet", return_value=True),
            patch.object(interface, "_compute_qc_metrics", return_value={}),
        ):
            interface._run_interface(MagicMock())

        qc = interface._results["qc_metrics"]
        assert qc == {
            "method_used": "hdbet",
            "hdbet_device": "cuda",
            "hdbet_tta_used": False,
        }


def test_perfusion_workflows_propagate_sloppy(tmp_path: Path) -> None:
    """DCE and DSC HD-BET nodes receive the public sloppy setting."""
    from pmriprep.config.dce import DCEConfig
    from pmriprep.config.dsc import DSCConfig
    from pmriprep.workflows.dce.base import init_dce_brain_extraction_wf
    from pmriprep.workflows.dsc.base import init_dsc_preproc_wf

    dce_workflow = init_dce_brain_extraction_wf(
        config=DCEConfig(sloppy=True),
        mem_gb=1.0,
    )
    dce_nodes = {node.name: node for node in dce_workflow._get_all_nodes()}
    assert dce_nodes["dce_brain_extract"].inputs.sloppy is True

    bids_dir = tmp_path / "bids"
    bids_dir.mkdir()
    dsc_workflow = init_dsc_preproc_wf(
        bids_dir=bids_dir,
        output_dir=tmp_path / "derivatives",
        subject_id="01",
        session_id="01",
        config=DSCConfig(sloppy=True),
    )
    dsc_nodes = {node.name: node for node in dsc_workflow._get_all_nodes()}
    assert dsc_nodes["hdbet_dsc"].inputs.sloppy is True
