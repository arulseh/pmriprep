"""Lossless, non-pickle transport of BSW guidance between workflow nodes."""

from pathlib import Path

import numpy as np

from .bsw_guidance import BSWGuidance


def save_bsw_guidance(
    guidance: BSWGuidance, path: str | Path, affine: np.ndarray
) -> str:
    """Save exact coefficient/reference arrays and their native spatial grid.

    Parameters
    ----------
    guidance : BSWGuidance
        Unmodified output of the upstream correction.
    path : str or Path
        Destination NPZ file in the current workflow node's working directory.
    affine : ndarray
        Spatial affine of the source concentration image.

    Returns
    -------
    str
        Absolute archive path. Missing arrays remain absent, not zero-filled.
    """
    arrays = {
        name: getattr(guidance, name)
        for name in (
            "reference_curve",
            "reference_integral",
            "reference_basis",
            "reference_mask",
            "k2",
            "k2_sd",
            "status",
        )
        if getattr(guidance, name) is not None
    }
    arrays.update(
        schema=np.array(1),
        affine=np.asarray(affine),
        reference_count=np.array(guidance.reference_count),
        reference_reason=np.array(guidance.reference_reason or ""),
        baseline_frames=np.array(guidance.baseline_frames),
        tr=np.array(guidance.tr),
        degrees_of_freedom=np.array(guidance.degrees_of_freedom),
    )
    destination = Path(path).absolute()
    with destination.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return str(destination)


def load_bsw_guidance(
    path: str | Path, *, affine: np.ndarray, shape: tuple[int, ...], tr: float
) -> BSWGuidance:
    """Read guidance only when the archive belongs to the requested image grid.

    Parameters
    ----------
    path : str or Path
        Archive produced by save_bsw_guidance.
    affine : ndarray
        Expected source voxel-to-world affine, shape (4, 4).
    shape : tuple of int
        Expected full four-dimensional image shape.
    tr : float
        Expected interval between volumes in seconds.

    Returns
    -------
    BSWGuidance
        Exact stored arrays; malformed or mismatched archives raise explicitly.

    Raises
    ------
    ValueError
        If the archive schema is unsupported or its spatial shape, affine,
        frame count or sampling interval does not match the requested acquisition.
    """
    with np.load(path, allow_pickle=False) as data:
        if int(data["schema"]) != 1:
            raise ValueError("unsupported BSW guidance archive schema")
        if (
            len(shape) != 4
            or data["affine"].shape != (4, 4)
            or not np.allclose(data["affine"], affine, rtol=0, atol=1e-5)
            or data["k2"].shape != shape[:3]
            or data["k2_sd"].shape != shape[:3]
            or data["status"].shape != shape[:3]
            or int(data["degrees_of_freedom"]) != shape[-1] - 2
            or not np.isclose(float(data["tr"]), tr, rtol=1e-10, atol=1e-12)
        ):
            raise ValueError("BSW guidance does not match the signal acquisition grid")
        return BSWGuidance(
            reference_curve=data["reference_curve"]
            if "reference_curve" in data
            else None,
            reference_integral=data["reference_integral"]
            if "reference_integral" in data
            else None,
            reference_basis=data["reference_basis"]
            if "reference_basis" in data
            else None,
            reference_mask=data["reference_mask"] if "reference_mask" in data else None,
            reference_count=int(data["reference_count"]),
            reference_reason=str(data["reference_reason"]) or None,
            k2=data["k2"],
            k2_sd=data["k2_sd"],
            status=data["status"],
            baseline_frames=int(data["baseline_frames"]),
            tr=float(data["tr"]),
            degrees_of_freedom=int(data["degrees_of_freedom"]),
        )
