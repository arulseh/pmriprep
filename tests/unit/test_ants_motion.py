"""Physical-transform, execution and common-QC contracts for ANTs motion."""

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pmriprep.utils.ants_motion import (
    motion_parameters, rigid_field_transforms, run_motion_correction,
)
from pmriprep.utils.motion import summarize_motion


def reference(shape: tuple[int, ...], affine: np.ndarray) -> nib.Nifti1Image:
    """Construct a nonuniform finite reference."""
    return nib.Nifti1Image(np.arange(np.prod(shape), dtype=np.float32).reshape(shape), affine)


def write_field(path: Path, image: Any, matrices: np.ndarray) -> None:
    """Encode known physical pull transforms as ANTs vector displacements."""
    coords = np.moveaxis(np.indices(image.shape), 0, -1)
    physical = nib.affines.apply_affine(np.diag([-1., -1., 1., 1.]) @ image.affine, coords)
    field = np.zeros((*image.shape, len(matrices), 4), dtype=np.float32)
    for i, matrix in enumerate(matrices):
        field[..., i, :3] = nib.affines.apply_affine(matrix, physical) - physical
    nib.save(nib.Nifti1Image(field, image.affine), path)


@pytest.mark.parametrize("shape", [(9, 10, 11), (9, 10, 1)])
@pytest.mark.parametrize("handedness", [-1, 1])
def test_rigid_field_geometry(tmp_path: Path, shape: tuple[int, ...], handedness: int) -> None:
    """Oblique, anisotropic, flipped and single-slice grids recover physical motion."""
    affine = np.eye(4)
    affine[:3, :3] = Rotation.from_euler("xyz", [12, -7, 21], degrees=True).as_matrix() @ np.diag([handedness * 1.4, 2., 4.])
    affine[:3, 3] = [31, -42, 10]
    image = reference(shape, affine)
    matrices = np.repeat(np.eye(4)[None], 3, axis=0)
    matrices[1, :3, :3] = Rotation.from_euler("xyz", [0.02, -0.03, 0.01]).as_matrix()
    center = np.array([10., -40., 22.])
    matrices[1, :3, 3] = center - matrices[1, :3, :3] @ center + [1., -2., .5]
    matrices[2, :3, 3] = [-.4, .2, 1.1]
    path = tmp_path / "field.nii.gz"
    write_field(path, image, matrices)
    actual = rigid_field_transforms(path, image, 3)
    np.testing.assert_allclose(actual, matrices, rtol=0, atol=3e-6)
    assert not list(tmp_path.glob("motion_field_*"))


def test_common_parameters_preserve_rotation_center_and_direction() -> None:
    """An independently constructed FSL transform yields the prescribed parameters."""
    image = reference((9, 10, 11), np.diag([-2., 3., 4., 1.]))
    weights = image.get_fdata()
    center_vox = np.array([(weights * np.indices(image.shape)[i]).sum() / weights.sum() for i in range(3)])
    center = center_vox * [2, 3, 4]
    angles = np.array([.02, -.03, .01])
    rotation = Rotation.from_euler("XYZ", -angles).as_matrix()
    translation = np.array([1.2, -.5, .7])
    fsl = np.eye(4)
    fsl[:3, :3] = rotation
    fsl[:3, 3] = center + translation - rotation @ center
    # This grid has scaled axes (+x,+y,+z) = LPS (+x,-y,+z).
    basis = np.diag([1., -1., 1., 1.])
    pull = basis @ np.linalg.inv(fsl) @ basis
    actual = motion_parameters(np.array([np.eye(4), pull]), image)
    np.testing.assert_allclose(actual[0], 0, atol=1e-12)
    np.testing.assert_allclose(actual[1], np.r_[angles, translation], atol=1e-12)
    assert summarize_motion(actual)["fd_mean"] == pytest.approx(np.abs(translation).sum() + 50 * np.abs(angles).sum())


def test_nonrigid_field_rejected(tmp_path: Path) -> None:
    """Affine scaling must not masquerade as rigid motion."""
    image = reference((8, 9, 10), np.eye(4))
    matrices = np.array([np.diag([1.01, 1., 1., 1.])])
    path = tmp_path / "field.nii.gz"
    write_field(path, image, matrices)
    with pytest.raises(ValueError, match="not rigid"):
        rigid_field_transforms(path, image, 1)


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize("threads", [1, 4])
def test_runner_contract(tmp_path: Path, monkeypatch: Any, apply: bool, threads: int) -> None:
    """The runner preserves settings, frame order, geometry and skipped input bytes."""
    image = reference((8, 9, 10), np.diag([2., 2., 3., 1.]))
    ref_file, in_file, out_file = [tmp_path / s for s in ("reference.nii.gz", "input.nii.gz", "output.nii.gz")]
    nib.save(image, ref_file)
    data = np.stack([image.get_fdata() + i * 100 for i in range(3)], axis=-1)
    nib.save(nib.Nifti1Image(data, image.affine), in_file)
    original_bytes = in_file.read_bytes()
    captured: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> None:
        captured.update(command=command, env=kwargs["env"])
        kwargs["stdout"].write("\n".join(f"done resampling timepoint : {i}" for i in range(3)))
        matrices = np.repeat(np.eye(4)[None], 3, axis=0)
        matrices[:, 0, 3] = [0., 1., 2.]
        prefix = tmp_path / "ants_motion_"
        write_field(Path(str(prefix) + "Warp.nii.gz"), image, matrices)
        write_field(Path(str(prefix) + "InverseWarp.nii.gz"), image, np.linalg.inv(matrices))
        if apply:
            nib.save(nib.Nifti1Image(data + 10, image.affine), out_file)

    monkeypatch.setattr("pmriprep.utils.ants_motion.shutil.which", lambda name: "/bin/antsMotionCorr")
    monkeypatch.setattr("pmriprep.utils.ants_motion.subprocess.run", run)
    result = run_motion_correction(str(in_file), str(ref_file), str(out_file),
                                   apply=apply, num_threads=threads)
    command = captured["command"]
    for flag, value in [("-t", "Rigid[0.1]"), ("-i", "100x50x20"), ("-f", "4x2x1"), ("-s", "2x1x0mm"), ("--random-seed", "1"), ("-p", "Linear"), ("-u", "1"), ("-e", "1")]:
        assert command[command.index(flag) + 1] == value
    assert captured["env"]["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] == str(threads)
    assert captured["env"]["OMP_NUM_THREADS"] == "1"
    assert captured["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert result.provenance["motion_itk_threads"] == threads
    assert result.provenance["motion_applied"] is apply
    assert np.loadtxt(result.params_file).shape == (3, 6)
    if apply:
        np.testing.assert_array_equal(nib.load(out_file).get_fdata(), data + 10)
    else:
        assert out_file.read_bytes() == original_bytes
        assert command[command.index("-o") + 1] == f"[{tmp_path}/ants_motion_]"
    assert in_file.read_bytes() == original_bytes


def test_summary_matches_plot_nondefault_threshold(tmp_path: Path) -> None:
    """Plots and interface summaries share strict flags and the transition mean."""
    from pmriprep.utils.qc_utils import plot_motion_qc

    params = np.zeros((4, 6))
    params[:, 3] = [0., .25, .75, 1.5]
    summary = summarize_motion(params, .5)
    assert summary["fd_mean"] == .5
    assert summary["outlier_frames"] == [3]
    assert summary["percent_outliers"] == 25.
    _, plotted = plot_motion_qc(params, tmp_path / "qc.png", fd_threshold=.25)
    assert plotted == summarize_motion(params, .25)
    single = summarize_motion(np.zeros((1, 6)))
    assert single["fd_mean"] == 0 and single["fd_max"] == 0


def test_public_motion_qc_excludes_local_paths() -> None:
    """Public QC keeps method provenance without exposing execution paths."""
    from pmriprep.utils.sidecar_builders import filter_pipeline_qc

    private = {
        "motion_command": ["/local/bin/antsMotionCorr", "MI[/local/ref,/local/input]"],
        "motion_executable": "/local/bin/antsMotionCorr",
        "motion_reference": "/local/work/reference.nii.gz",
        "motion_tool": "ANTs", "motion_version": "2.6.5",
        "motion_applied": False, "motion_interpolation": "none",
        "motion_itk_threads": 1, "fd_mean": 0.2,
    }
    public = filter_pipeline_qc({"preprocessing": private})["preprocessing"]
    assert public == {key: value for key, value in private.items() if key not in {
        "motion_command", "motion_executable", "motion_reference",
    }}
    assert "motion_command" in private


def test_command_failure_propagates(tmp_path: Path, monkeypatch: Any) -> None:
    """A failed registration cannot produce successful zero-motion QC."""
    import subprocess

    image = reference((8, 9, 10), np.eye(4))
    fixed, moving = tmp_path / "ref.nii.gz", tmp_path / "input.nii.gz"
    nib.save(image, fixed)
    nib.save(nib.Nifti1Image(np.stack([image.get_fdata()] * 2, axis=-1), image.affine), moving)

    def fail(command: list[str], **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr("pmriprep.utils.ants_motion.shutil.which", lambda name: "/bin/antsMotionCorr")
    monkeypatch.setattr("pmriprep.utils.ants_motion.subprocess.run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        run_motion_correction(str(moving), str(fixed), str(tmp_path / "output.nii.gz"))
    assert not (tmp_path / "motion.par").exists()


@pytest.mark.parametrize("apply", [False, True])
def test_dsc_interface_selected_series(tmp_path: Path, monkeypatch: Any, apply: bool) -> None:
    """DSC derives its mean and QC from the selected series and common parameters."""
    from pmriprep.interfaces.dsc import MotionCorrection
    from pmriprep.utils.ants_motion import MotionResult
    import shutil

    monkeypatch.chdir(tmp_path)
    image = reference((8, 9, 10), np.eye(4))
    data = np.stack([image.get_fdata() + i for i in range(6)], axis=-1)
    source = tmp_path / "input.nii.gz"
    nib.save(nib.Nifti1Image(data, image.affine), source)

    def run(in_file: str, reference_file: str, out_file: str, *, apply: bool,
            num_threads: int) -> MotionResult:
        assert num_threads == 3
        np.testing.assert_array_equal(nib.load(reference_file).get_fdata(), data[..., :3].mean(axis=-1))
        if apply:
            nib.save(nib.Nifti1Image(data + 100, image.affine), out_file)
        else:
            shutil.copyfile(in_file, out_file)
        params = np.zeros((6, 6))
        params[:, 3] = np.arange(6) * .4
        path = tmp_path / "motion.par"
        np.savetxt(path, params)
        return MotionResult(out_file, str(path), {"motion_applied": apply, "motion_tool": "ANTs"})

    monkeypatch.setattr("pmriprep.utils.ants_motion.run_motion_correction", run)
    result = MotionCorrection(in_file=str(source), baseline_frames=3,
                              motion_thresh=.25, num_threads=3,
                              apply_motion_correction=apply).run().outputs
    expected = data + 100 if apply else data
    np.testing.assert_array_equal(nib.load(result.out_file).get_fdata(), expected)
    np.testing.assert_allclose(nib.load(result.mean_img).get_fdata(), expected.mean(axis=-1))
    assert result.fd_mean == pytest.approx(.4)
    assert result.n_outliers == 5
    assert result.qc_metrics["motion_applied"] is apply
    assert result.qc_metrics["motion_reference_frames"] == 3


@pytest.mark.parametrize("budget,requested,expected", [
    (1, 8, 1), (2, 8, 2), (8, 8, 8),
    (8, 1, 1), (54, 8, 8), (54, 16, 16),
])
@pytest.mark.parametrize("apply", [False, True])
def test_motion_threads_match_nested_node_reservations(
    budget: int, requested: int, expected: int, apply: bool,
) -> None:
    """Both modalities reserve their actual threads within the pipeline budget."""
    import nipype.pipeline.engine as pe
    from nipype.interfaces.utility import IdentityInterface

    from pmriprep.cli.run import _configure_motion_threads
    from pmriprep.interfaces.dce import DCEPreprocess
    from pmriprep.interfaces.dsc import MotionCorrection

    parent = pe.Workflow(name="parent")
    child = pe.Workflow(name="child")
    dsc = pe.Node(MotionCorrection(apply_motion_correction=apply), name="dsc")
    dce = pe.Node(DCEPreprocess(apply_motion_correction=apply), name="dce")
    other = pe.Node(IdentityInterface(fields=["value"]), name="other", n_procs=2)
    child.add_nodes([dsc, dce, other])
    parent.add_nodes([child])
    assert _configure_motion_threads(parent, budget, requested) == expected
    for node in (dsc, dce):
        assert node.inputs.num_threads == node.n_procs == expected
        assert node.n_procs <= budget
    assert other.n_procs == 2
    assert _configure_motion_threads(parent, 1, 8) == 1
    assert dsc.n_procs == dce.n_procs == 1


@pytest.mark.parametrize("threads", [0, -1, 1.5, True])
def test_runner_rejects_invalid_threads(threads: Any) -> None:
    """Reject invalid allocations before reading images or launching ANTs."""
    with pytest.raises(ValueError, match="positive integer"):
        run_motion_correction("missing", "missing", "missing", num_threads=threads)
