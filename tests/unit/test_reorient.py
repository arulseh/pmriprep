"""Execution tests for the nibabel-based reorient interfaces/helpers.

These paths (``Reorient``, ``reorient_list_to_ras``, ``ReorientAndCrop`` reorient
step) are pure nibabel and run without FSL; only the ``robustfov`` crop needs FSL.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pmriprep.interfaces.anat import ReorientAndCrop
from pmriprep.interfaces.common import Reorient, reorient_list_to_ras
from pmriprep.utils.nifti import reorient_img_to_ras

# A deliberately non-RAS affine (axcodes L, S, A): axis permutation + flip.
_NON_RAS_AFF = np.array(
    [[-2.0, 0.0, 0.0, 10.0],
     [0.0, 0.0, 3.0, -5.0],
     [0.0, 2.0, 0.0, 7.0],
     [0.0, 0.0, 0.0, 1.0]]
)
_RAS_AFF = np.diag([2.0, 2.0, 2.0, 1.0])


def _save(data: np.ndarray, affine: np.ndarray, path: Path) -> str:
    """Save an array+affine as a NIfTI and return its path string."""
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return str(path)


def _world_values_preserved(src_data, src_aff, out_img) -> bool:
    """Every source voxel's physical point holds the same value post-reorient."""
    out_data = out_img.get_fdata()
    inv = np.linalg.inv(out_img.affine)
    for i in (0, src_data.shape[0] - 1):
        for j in (0, src_data.shape[1] - 1):
            for k in (0, src_data.shape[2] - 1):
                world = src_aff @ [i, j, k, 1]
                ri, rj, rk = np.round(inv @ world)[:3].astype(int)
                if not np.allclose(src_data[i, j, k], out_data[ri, rj, rk]):
                    return False
    return True


class TestReorientInterface:
    """The nibabel-only ``Reorient`` SimpleInterface."""

    def test_3d_las_to_ras(self, tmp_path):
        """A 3D LAS image is reoriented to RAS, values preserved."""
        assert nib.aff2axcodes(_NON_RAS_AFF) == ("L", "S", "A")
        d = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
        src = _save(d, _NON_RAS_AFF, tmp_path / "in.nii.gz")
        r = Reorient(in_file=src, out_file=str(tmp_path / "out.nii.gz"))
        out = nib.load(r.run().outputs.out_file)
        assert nib.aff2axcodes(out.affine) == ("R", "A", "S")
        assert _world_values_preserved(d, _NON_RAS_AFF, out)

    def test_4d_reorient_leaves_time_axis(self, tmp_path):
        """A 4D image reorients the spatial axes and leaves the time axis intact."""
        d = np.arange(4 * 5 * 6 * 3, dtype=np.float32).reshape(4, 5, 6, 3)
        src = _save(d, _NON_RAS_AFF, tmp_path / "in4d.nii.gz")
        r = Reorient(in_file=src, out_file=str(tmp_path / "out4d.nii.gz"))
        out = nib.load(r.run().outputs.out_file)
        assert nib.aff2axcodes(out.affine) == ("R", "A", "S")
        assert out.shape[3] == 3  # time axis untouched
        # per-volume world-coordinate preservation catches any time-axis scramble
        out_data = out.get_fdata()
        for t in range(3):
            vol = nib.Nifti1Image(out_data[..., t], out.affine)
            assert _world_values_preserved(d[..., t], _NON_RAS_AFF, vol)

    def test_already_ras_still_writes(self, tmp_path):
        """Always-writes: an already-RAS input is copied to a distinct out_file."""
        d = np.arange(3 * 3 * 3, dtype=np.float32).reshape(3, 3, 3)
        src = _save(d, _RAS_AFF, tmp_path / "ras.nii.gz")
        out_path = str(tmp_path / "ras_out.nii.gz")
        r = Reorient(in_file=src, out_file=out_path)
        res = r.run()
        assert res.outputs.out_file == out_path
        assert res.outputs.out_file != src
        out = nib.load(res.outputs.out_file)
        assert nib.aff2axcodes(out.affine) == ("R", "A", "S")
        assert np.array_equal(out.get_fdata(), d)


class TestReorientListToRas:
    """The nibabel-only ``reorient_list_to_ras`` Function-node helper."""

    def test_none_returns_none(self):
        """None input passes through as None."""
        assert reorient_list_to_ras(None) is None

    def test_empty_returns_empty(self):
        """An empty list returns an empty list."""
        assert reorient_list_to_ras([]) == []

    def test_list_reoriented(self, tmp_path, monkeypatch):
        """Each file in the list is reoriented to RAS."""
        monkeypatch.chdir(tmp_path)  # outputs land in cwd
        d = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
        a = _save(d, _NON_RAS_AFF, tmp_path / "a.nii.gz")
        b = _save(d, _NON_RAS_AFF, tmp_path / "b.nii.gz")
        outs = reorient_list_to_ras([a, b])
        assert len(outs) == 2
        for o in outs:
            assert nib.aff2axcodes(nib.load(o).affine) == ("R", "A", "S")


class TestReorientAndCrop:
    """The ``ReorientAndCrop`` SimpleInterface (nibabel reorient + FSL robustfov)."""

    def test_no_crop_is_nibabel_only(self, tmp_path):
        """crop_fov=False reorients with nibabel, no FSL required."""
        d = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
        src = _save(d, _NON_RAS_AFF, tmp_path / "t2w.nii.gz")
        r = ReorientAndCrop(
            in_file=src, out_file=str(tmp_path / "out.nii.gz"), crop_fov=False
        )
        out = nib.load(r.run().outputs.out_file)
        assert nib.aff2axcodes(out.affine) == ("R", "A", "S")
        assert _world_values_preserved(d, _NON_RAS_AFF, out)

    def test_crop_plumbing_with_fake_robustfov(self, tmp_path, monkeypatch):
        """The crop path (subprocess/temp/reload/re-RAS) runs FSL-free via a fake robustfov."""
        bindir = tmp_path / "bin"
        bindir.mkdir()
        fake = bindir / "robustfov"
        fake.write_text(
            "#!/usr/bin/env bash\n"
            'while [ $# -gt 0 ]; do case "$1" in '
            '-i) IN="$2"; shift 2;; -r) OUT="$2"; shift 2;; *) shift;; esac; done\n'
            'cp "$IN" "$OUT"\n'  # identity crop: copy input to output
        )
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
        d = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
        src = _save(d, _NON_RAS_AFF, tmp_path / "t1w.nii.gz")
        r = ReorientAndCrop(
            in_file=src, out_file=str(tmp_path / "out.nii.gz"), crop_fov=True
        )
        out = nib.load(r.run().outputs.out_file)
        assert nib.aff2axcodes(out.affine) == ("R", "A", "S")
        # identity fake-crop → the reorient must preserve world values end-to-end
        assert _world_values_preserved(d, _NON_RAS_AFF, out)

    @pytest.mark.skipif(
        shutil.which("robustfov") is None, reason="FSL robustfov not available"
    )
    def test_crop_output_is_ras(self, tmp_path):
        """crop_fov=True runs real robustfov; the output must still be RAS."""
        d = np.arange(16 * 16 * 24, dtype=np.float32).reshape(16, 16, 24)
        src = _save(d, _NON_RAS_AFF, tmp_path / "t1w.nii.gz")
        r = ReorientAndCrop(
            in_file=src, out_file=str(tmp_path / "out.nii.gz"), crop_fov=True
        )
        out = nib.load(r.run().outputs.out_file)
        assert nib.aff2axcodes(out.affine) == ("R", "A", "S")


class TestHelper:
    """The shared ``reorient_img_to_ras`` primitive."""

    def test_idempotent_on_ras(self):
        """An already-RAS image is returned unchanged (same object)."""
        img = nib.Nifti1Image(np.zeros((2, 2, 2), np.float32), _RAS_AFF)
        assert reorient_img_to_ras(img) is img

    def test_reorients_non_ras(self):
        """A non-RAS image is reoriented to RAS."""
        img = nib.Nifti1Image(np.zeros((2, 3, 4), np.float32), _NON_RAS_AFF)
        assert nib.aff2axcodes(reorient_img_to_ras(img).affine) == ("R", "A", "S")
