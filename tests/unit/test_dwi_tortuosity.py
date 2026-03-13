"""Unit tests for DWI direction averaging and trace classification."""

import numpy as np
import pytest

from pmriprep.utils.dce_tortuosity import (
    DWI_TRACE_MAX_DIRS,
    average_dwi_directions,
    is_trace_dwi,
)


# ---------------------------------------------------------------------------
# is_trace_dwi tests
# ---------------------------------------------------------------------------

class TestIsTraceDwi:
    """Tests for is_trace_dwi()."""

    def test_none_bvecs_returns_true(self):
        """Missing bvecs -> conservative True (assume trace)."""
        bvals = np.array([0, 1000, 1000])
        assert is_trace_dwi(bvals, b_vectors=None) is True

    def test_trace_classification(self):
        """3 directions per shell -> trace."""
        bvals = np.array([0, 0, 0, 1000, 1000, 1000])
        bvecs = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        assert is_trace_dwi(bvals, bvecs) is True

    @pytest.mark.parametrize("n_dirs", [6, 12, 30, 64])
    def test_dti_classification(self, n_dirs):
        """Various DTI direction counts -> not trace."""
        rng = np.random.default_rng(42)
        bvals = np.concatenate([np.zeros(1), np.full(n_dirs, 1000.0)])
        bvecs = np.zeros((n_dirs + 1, 3))
        # Random unit vectors for diffusion-encoding directions
        dirs = rng.standard_normal((n_dirs, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        bvecs[1:] = dirs
        assert is_trace_dwi(bvals, bvecs) is False

    def test_single_direction_is_trace(self):
        """1 direction per shell -> trace."""
        bvals = np.array([0, 1000])
        bvecs = np.array([[0, 0, 0], [1, 0, 0]])
        assert is_trace_dwi(bvals, bvecs) is True

    def test_multi_shell_trace(self):
        """Multi-shell with <=3 dirs each -> trace."""
        bvals = np.array([0, 500, 500, 1000, 1000])
        bvecs = np.array([
            [0, 0, 0],
            [1, 0, 0], [0, 1, 0],
            [1, 0, 0], [0, 0, 1],
        ])
        assert is_trace_dwi(bvals, bvecs) is True

    def test_transposed_bvecs(self):
        """(3, N) bvec format -> handled correctly."""
        bvals = np.array([0, 1000, 1000, 1000, 1000, 1000, 1000])
        rng = np.random.default_rng(0)
        dirs = rng.standard_normal((6, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        bvecs_3xn = np.zeros((3, 7))
        bvecs_3xn[:, 1:] = dirs.T
        assert is_trace_dwi(bvals, bvecs_3xn) is False


# ---------------------------------------------------------------------------
# average_dwi_directions tests
# ---------------------------------------------------------------------------

class TestAverageDwiDirections:
    """Tests for average_dwi_directions()."""

    def test_none_bvecs_passthrough(self):
        """No bvecs -> data returned unchanged."""
        signal = np.random.default_rng(0).random((5, 5, 5, 4))
        bvals = np.array([0, 1000, 1000, 1000])
        out_signal, out_bvals = average_dwi_directions(signal, bvals, b_vectors=None)
        np.testing.assert_array_equal(out_signal, signal)
        np.testing.assert_array_equal(out_bvals, bvals)

    def test_trace_3dir_unchanged(self):
        """Trace DWI (3 directions) -> unchanged."""
        signal = np.random.default_rng(1).random((5, 5, 5, 4))
        bvals = np.array([0, 1000, 1000, 1000])
        bvecs = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        out_signal, out_bvals = average_dwi_directions(signal, bvals, bvecs)
        np.testing.assert_array_equal(out_signal, signal)
        np.testing.assert_array_equal(out_bvals, bvals)

    @pytest.mark.parametrize("n_dirs", [6, 20, 30, 64])
    def test_dti_averaged(self, n_dirs):
        """DTI with various direction counts -> averaged to 2 shells."""
        rng = np.random.default_rng(42)
        n_b0 = 2
        n_total = n_b0 + n_dirs
        signal = rng.random((4, 4, 4, n_total)) * 1000

        bvals = np.concatenate([np.zeros(n_b0), np.full(n_dirs, 1000.0)])
        bvecs = np.zeros((n_total, 3))
        dirs = rng.standard_normal((n_dirs, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        bvecs[n_b0:] = dirs

        out_signal, out_bvals = average_dwi_directions(signal, bvals, bvecs)

        # Should have 2 shells: b=0 and b=1000
        assert out_signal.shape == (4, 4, 4, 2)
        assert len(out_bvals) == 2
        assert out_bvals[0] < 100  # b=0 shell
        assert abs(out_bvals[1] - 1000.0) < 50  # b=1000 shell

    def test_averaging_correctness(self):
        """Verify averaged signal matches np.mean per shell."""
        rng = np.random.default_rng(99)
        n_dirs = 6
        signal = rng.random((3, 3, 3, 1 + n_dirs)) * 500

        bvals = np.concatenate([[0], np.full(n_dirs, 1000.0)])
        bvecs = np.zeros((1 + n_dirs, 3))
        dirs = rng.standard_normal((n_dirs, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        bvecs[1:] = dirs

        out_signal, out_bvals = average_dwi_directions(signal, bvals, bvecs)

        # b=0 shell: just volume 0
        np.testing.assert_allclose(out_signal[..., 0], signal[..., 0])
        # b=1000 shell: mean of volumes 1-6
        expected_b1000 = np.mean(signal[..., 1:], axis=-1)
        np.testing.assert_allclose(out_signal[..., 1], expected_b1000)

    def test_zero_norm_bvec(self):
        """b=0 volumes with [0,0,0] bvec must not produce NaN."""
        signal = np.ones((3, 3, 3, 8)) * 500
        bvals = np.concatenate([[0, 0], np.full(6, 1000.0)])
        bvecs = np.zeros((8, 3))
        rng = np.random.default_rng(0)
        dirs = rng.standard_normal((6, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        bvecs[2:] = dirs

        out_signal, out_bvals = average_dwi_directions(signal, bvals, bvecs)
        assert not np.any(np.isnan(out_signal))

    def test_shape_mismatch_raises(self):
        """Mismatched dimensions -> ValueError."""
        signal = np.ones((3, 3, 3, 5))
        bvals = np.array([0, 1000, 1000])  # 3 != 5
        bvecs = np.zeros((5, 3))
        with pytest.raises(ValueError, match="b-values"):
            average_dwi_directions(signal, bvals, bvecs)

    def test_bvec_count_mismatch_raises(self):
        """Mismatched bvec count -> ValueError."""
        signal = np.ones((3, 3, 3, 5))
        bvals = np.array([0, 1000, 1000, 1000, 1000])
        bvecs = np.zeros((3, 3))  # 3 != 5 (and not (3,5) shape)
        with pytest.raises(ValueError, match="b-vectors"):
            average_dwi_directions(signal, bvals, bvecs)

    def test_multi_shell_dti(self):
        """2-shell DTI -> averaged per shell."""
        rng = np.random.default_rng(7)
        n_b0 = 2
        n_b500 = 6
        n_b1000 = 6
        n_total = n_b0 + n_b500 + n_b1000

        signal = rng.random((3, 3, 3, n_total)) * 800
        bvals = np.concatenate([
            np.zeros(n_b0),
            np.full(n_b500, 500.0),
            np.full(n_b1000, 1000.0),
        ])
        bvecs = np.zeros((n_total, 3))
        dirs = rng.standard_normal((n_b500 + n_b1000, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        bvecs[n_b0:] = dirs

        out_signal, out_bvals = average_dwi_directions(signal, bvals, bvecs)

        # 3 shells: b=0, b=500, b=1000
        assert out_signal.shape == (3, 3, 3, 3)
        assert len(out_bvals) == 3
        assert out_bvals[0] < 100
        assert abs(out_bvals[1] - 500) < 50
        assert abs(out_bvals[2] - 1000) < 50
