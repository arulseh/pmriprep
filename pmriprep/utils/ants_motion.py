"""Rigid ANTs motion estimation and common motion-parameter conversion."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


@dataclass(frozen=True)
class MotionResult:
    """Selected series, common parameters and registration provenance."""

    out_file: str
    params_file: str
    provenance: dict[str, Any]


def scaled_voxels(image: Any) -> Array:
    """Return the FSL voxel-to-scaled-mm mapping, including handedness.

    Parameters
    ----------
    image : nibabel spatial image
        Image defining voxel dimensions, affine and shape.

    Returns
    -------
    ndarray
        Homogeneous voxel-to-scaled-mm matrix.
    """
    result = np.diag([*image.header.get_zooms()[:3], 1.0]).astype(float)
    if np.linalg.det(image.affine[:3, :3]) > 0:
        result[0, 0] *= -1
        result[0, 3] = (image.shape[0] - 1) * image.header.get_zooms()[0]
    return result


def reference_center(image: Any) -> Array:
    """Calculate the minimum-subtracted intensity centre in scaled mm.

    Parameters
    ----------
    image : nibabel spatial image
        Finite, nonconstant three-dimensional reference.

    Returns
    -------
    ndarray
        Three coordinates of the intensity centre.
    """
    data = image.get_fdata(dtype=np.float32)
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError("Motion reference must be finite and three-dimensional")
    weights = (data - data.min()).astype(float)
    total = weights.sum()
    if total <= 0:
        raise ValueError("Motion reference has no intensity contrast")
    center = [
        np.dot(weights.sum(axis=tuple(j for j in range(3) if j != i)),
               np.arange(image.shape[i])) / total for i in range(3)
    ]
    return np.asarray((scaled_voxels(image) @ [*center, 1.0])[:3], dtype=np.float64)


def motion_parameters(pull_lps: Array, reference: Any) -> Array:
    """Convert physical reference-to-input transforms to common parameters.

    Parameters
    ----------
    pull_lps : ndarray
        Rigid transforms, shape (frames, 4, 4), in physical LPS coordinates.
    reference : nibabel spatial image
        Reference on the same spatial grid as the input series.

    Returns
    -------
    ndarray
        FSL-convention moving-to-reference rotations (radians) followed by
        translations (mm), about the reference intensity centre.
    """
    basis = scaled_voxels(reference)
    world_to_scaled = basis @ np.linalg.inv(reference.affine) @ _LPS
    scaled_to_world = np.linalg.inv(world_to_scaled)
    center = reference_center(reference)
    parameters = []
    for pull in pull_lps:
        matrix = world_to_scaled @ np.linalg.inv(pull) @ scaled_to_world
        rotation, triangular = np.linalg.qr(matrix[:3, :3])
        rotation = rotation @ np.diag(np.sign(np.diag(triangular)))
        cy = np.hypot(rotation[0, 0], rotation[0, 1])
        if cy < 1e-4:
            angles = [np.arctan2(-rotation[2, 1], rotation[1, 1]),
                      np.arctan2(-rotation[0, 2], 0.0), 0.0]
        else:
            angles = [np.arctan2(rotation[1, 2], rotation[2, 2]),
                      np.arctan2(-rotation[0, 2], cy),
                      np.arctan2(rotation[0, 1], rotation[0, 0])]
        parameters.append(np.r_[angles, matrix[:3, :3] @ center + matrix[:3, 3] - center])
    return np.asarray(parameters)


def same_spatial_grid(first: Any, second: Any) -> bool:
    """Compare spatial grids within one micrometre at every volume corner.

    Parameters
    ----------
    first, second : nibabel spatial image
        Images whose spatial shape and voxel-to-world geometry are compared.

    Returns
    -------
    bool
        Whether the shapes agree and corner displacements are at most 0.001 mm.
    """
    if first.shape[:3] != second.shape[:3]:
        return False
    corners = np.array(np.meshgrid(*[
        (0, size - 1) for size in first.shape[:3]
    ], indexing="ij")).reshape(3, -1).T
    displacement = (nib.affines.apply_affine(first.affine, corners)
                    - nib.affines.apply_affine(second.affine, corners))
    return bool(np.all(np.linalg.norm(displacement, axis=1) <= 1e-3))


def rigid_field_transforms(path: Path, reference: Any, n_frames: int) -> Array:
    """Recover rigid transforms from ANTs fields without loading the full field.

    Parameters
    ----------
    path : Path
        ANTs four-dimensional displacement field with four components.
    reference : nibabel spatial image
        Reference spatial grid.
    n_frames : int
        Expected number of frames in acquisition order.

    Returns
    -------
    ndarray
        Verified physical LPS transforms of shape (frames, 4, 4).

    Notes
    -----
    A temporary uncompressed file allows memory-mapped sparse reads. A
    held-out lattice checks the recovered affine and its rigidity. Single-slice
    fields determine the third rotation direction by the right-hand rule.
    """
    with tempfile.TemporaryDirectory(prefix="motion_field_", dir=path.parent) as temp:
        unpacked = Path(temp) / "field.nii"
        with gzip.open(path, "rb") as source, unpacked.open("wb") as target:
            shutil.copyfileobj(source, target)
        field: Any = nib.load(unpacked, mmap=True)
        if field.shape != (*reference.shape, n_frames, 4):
            raise ValueError(f"Unexpected ANTs field shape: {field.shape}")
        if not same_spatial_grid(field, reference):
            raise ValueError("ANTs field and reference grids differ")
        shape = reference.shape
        axes = [i for i in range(3) if shape[i] > 1]
        if len(axes) < 2:
            raise ValueError("Motion estimation needs at least two spatial dimensions")
        train = np.array(np.meshgrid(*[
            np.unique(np.linspace(0, n - 1, min(n, 5), dtype=int)) for n in shape
        ], indexing="ij")).reshape(3, -1).T
        check = np.array(np.meshgrid(*[
            np.unique(np.linspace(0, n - 1, min(n, 7), dtype=int)) for n in shape
        ], indexing="ij")).reshape(3, -1).T
        data = np.asanyarray(field.dataobj)
        displacements = np.asarray(data[tuple(train.T)], dtype=float)
        heldout = np.asarray(data[tuple(check.T)], dtype=float)
        if not np.isfinite(displacements).all() or not np.isfinite(heldout).all():
            raise ValueError("Nonfinite ANTs displacement field")
        if np.any(displacements[..., 3]) or np.any(heldout[..., 3]):
            raise ValueError("ANTs changed temporal coordinates")
        affine = _LPS @ field.affine
        coordinates = nib.affines.apply_affine(affine, train)
        check_coordinates = nib.affines.apply_affine(affine, check)
        design = np.c_[train[:, axes], np.ones(len(train))]
        matrices = []
        for frame in range(n_frames):
            mapped = coordinates + displacements[:, frame, :3]
            coefficients = np.linalg.lstsq(design, mapped, rcond=None)[0]
            mapped_basis = np.zeros((3, 3))
            mapped_basis[:, axes] = coefficients[:-1].T
            if len(axes) == 2:
                missing = next(i for i in range(3) if i not in axes)
                a, b = (missing + 1) % 3, (missing + 2) % 3
                vector = np.cross(mapped_basis[:, a], mapped_basis[:, b])
                sign = np.sign(np.linalg.det(affine[:3, :3]))
                mapped_basis[:, missing] = sign * vector / np.linalg.norm(vector) * np.linalg.norm(affine[:3, missing])
            rotation = mapped_basis @ np.linalg.inv(affine[:3, :3])
            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = coefficients[-1] - rotation @ affine[:3, 3]
            prediction = nib.affines.apply_affine(matrix, check_coordinates) - check_coordinates
            if not np.allclose(prediction, heldout[:, frame, :3], rtol=0, atol=2e-5):
                raise ValueError(f"ANTs displacement is not affine at frame {frame}")
            if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=2e-6)
                    or abs(np.linalg.det(rotation) - 1) > 2e-6):
                raise ValueError(f"ANTs displacement is not rigid at frame {frame}")
            matrices.append(matrix)
        del data
    return np.asarray(matrices)


def run_motion_correction(
    in_file: str, reference_file: str, out_file: str, *, apply: bool = True,
    num_threads: int = 1,
) -> MotionResult:
    """Estimate rigid motion with ANTs and optionally resample the series.

    Parameters
    ----------
    in_file : str
        Four-dimensional input on the reference spatial grid.
    reference_file : str
        Three-dimensional fixed reference.
    out_file : str
        Selected output series filename in the node working directory.
    apply : bool, optional
        Resample when true; otherwise copy the original series unchanged.
    num_threads : int, optional
        Positive ITK thread count reserved for this node by the scheduler.

    Returns
    -------
    MotionResult
        Series, common motion parameters and execution provenance.
    """
    from pmriprep.utils.sidecar_builders import get_ants_version

    if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
        raise ValueError("Motion thread count must be a positive integer")
    image: Any = nib.load(in_file)
    reference: Any = nib.load(reference_file)
    if len(image.shape) != 4 or image.shape[3] < 1:
        raise ValueError("Motion input must be a nonempty 4D series")
    if len(reference.shape) != 3 or not same_spatial_grid(image, reference):
        raise ValueError("Motion input and reference grids must match")
    reference_center(reference)
    output = Path(out_file).absolute()
    if output == Path(in_file).absolute():
        raise ValueError("Motion output must not overwrite its input")
    prefix = str(output.parent / "ants_motion_")
    executable = shutil.which("antsMotionCorr")
    if executable is None:
        raise RuntimeError("antsMotionCorr is required for motion estimation")
    version = get_ants_version()
    command = [executable, "-d", "3", "-u", "1", "-e", "1", "-t", "Rigid[0.1]",
               "-m", f"MI[{reference_file},{in_file},1,32,Regular,0.25]",
               "-i", "100x50x20", "-f", "4x2x1", "-s", "2x1x0mm",
               "--use-histogram-matching", "1", "--random-seed", "1", "-p", "Linear",
               "-w", "1", "-o", f"[{prefix},{output}]" if apply else f"[{prefix}]", "-v", "1"]
    environment = dict(os.environ, ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(num_threads),
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    log_path = output.parent / "ants_motion.log"
    with log_path.open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)
    log_text = log_path.read_text()
    completed = [int(value) for value in re.findall(r"done resampling timepoint\s*:\s*(\d+)", log_text)]
    if completed != list(range(image.shape[3])) or "Exception caught" in log_text:
        raise RuntimeError(f"ANTs did not complete every input frame; inspect {log_path}")
    pull = rigid_field_transforms(Path(prefix + "Warp.nii.gz"), reference, image.shape[3])
    push = rigid_field_transforms(Path(prefix + "InverseWarp.nii.gz"), reference, image.shape[3])
    if not np.allclose(push @ pull, np.eye(4), rtol=0, atol=2e-5):
        raise ValueError("ANTs forward/inverse motion transforms disagree")
    parameters = motion_parameters(pull, reference)
    params_file = str(output.parent / "motion.par")
    np.savetxt(params_file, parameters, fmt="%.12g")
    np.savez_compressed(output.parent / "motion_transforms.npz", pull_lps=pull, push_lps=push)
    if not apply:
        shutil.copyfile(in_file, output)
    result: Any = nib.load(output)
    if result.shape != image.shape or not same_spatial_grid(result, image):
        raise ValueError("ANTs output changed the acquisition grid or frame count")
    provenance = {
        "motion_tool": "ANTs", "motion_version": version,
        "motion_executable": executable,
        "motion_command": command, "motion_reference": reference_file,
        "motion_interpolation": "Linear" if apply else "none",
        "motion_estimated": True, "motion_applied": bool(apply),
        "motion_itk_threads": num_threads, "motion_random_seed": 1,
        "motion_parameter_convention": "FSL scaled-voxel moving-to-reference; reference intensity centre",
        "motion_transform_direction": "reference-to-input physical LPS (pull)",
        "motion_center_scaled_mm": reference_center(reference).tolist(),
    }
    (output.parent / "motion_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return MotionResult(str(output), params_file, provenance)
