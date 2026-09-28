"""Shared product-integration operator and finite-grid convolution accuracy.

Check the explicit stencil, endpoint handling and independent fine-grid
convolution. These guards require no external executable.
"""
import numpy as np
import numpy.typing as npt
import pytest

from pmriprep.utils.dsc_deconvolution import build_volterra_matrix


def _sourbron_eq7_matrix(
    aif: npt.NDArray[np.floating], dt: float
) -> npt.NDArray[np.float64]:
    """Literal Sourbron eq-(7) / Appendix A.4 Volterra matrix (row 0 all-zero)."""
    aif = np.asarray(aif, dtype=np.float64)
    n = len(aif)
    A = np.zeros((n, n), dtype=np.float64)
    a0plus = (2.0 * aif[0] + aif[1]) / 6.0
    for i in range(1, n):
        A[i, i] = a0plus  # diagonal A_0^+
        A[i, 0] = (2.0 * aif[i] + aif[i - 1]) / 6.0  # first column A_i^-
        for j in range(1, i):  # interior A_{i-j}^± centred on the lag
            m = i - j
            A[i, j] = (aif[m - 1] + 4.0 * aif[m] + aif[m + 1]) / 6.0
    return A * dt


def _fabber_voltera_matrix(
    aif: npt.NDArray[np.floating], dt: float
) -> npt.NDArray[np.float64]:
    """Literal 1-indexed port of FABBER ``DoConvolution`` voltera (the off-by-one)."""
    aif = np.asarray(aif, dtype=np.float64)
    nt = len(aif)

    def ae(k: int) -> float:  # 1-indexed aifextend = [0, aif, 0]
        if k == 1 or k == nt + 2:
            return 0.0
        return float(aif[k - 2])

    A = np.zeros((nt, nt), dtype=np.float64)
    for i in range(1, nt + 1):
        for j in range(1, i + 1):
            x, z = i + 1, i - j + 1
            if j == 1:
                v = (2 * ae(x) + ae(x - 1)) / 6.0
            elif j == i:
                v = (2 * ae(2) + ae(3)) / 6.0
            else:
                v = (4 * ae(z) + ae(z - 1) + ae(z + 1)) / 6.0
            A[i - 1, j - 1] = v
    return A * dt


def _rel_l2(got: npt.NDArray[np.floating], ref: npt.NDArray[np.floating]) -> float:
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


_AIF_CASES = [
    ((np.arange(50) / 3.0) * np.exp(-np.arange(50) / 5.0), 1.5),
    (np.r_[0.15, (np.arange(1, 40) / 3.0) * np.exp(-np.arange(1, 40) / 5.0)], 1.0),
    (np.array([0.4, 0.9]), 2.0),
]
_AIF_IDS = ["zero_baseline", "nonzero_baseline", "n2_edge"]


class TestVolterraMatrixSourbron:
    """build_volterra_matrix == literal Sourbron eq-(7) reference port (round-off)."""

    @pytest.mark.parametrize("aif,dt", _AIF_CASES, ids=_AIF_IDS)
    def test_matches_sourbron_eq7(self, aif, dt) -> None:
        aif = np.asarray(aif, dtype=np.float64)
        got = build_volterra_matrix(aif, dt)
        ref = _sourbron_eq7_matrix(aif, dt)
        assert np.max(np.abs(got - ref)) < 1e-12


class TestFabberDeviation:
    """The operator deliberately differs from FABBER's off-by-one voltera."""

    def test_differs_from_fabber(self) -> None:
        # Non-zero-baseline AIF so the corner + first-column rules carry weight.
        aif = np.r_[0.15, (np.arange(1, 40) / 3.0) * np.exp(-np.arange(1, 40) / 5.0)]
        dt = 1.0
        eq7 = build_volterra_matrix(aif, dt)
        fabber = _fabber_voltera_matrix(aif, dt)
        assert not np.allclose(eq7, fabber)
        # Corner: Sourbron row-0 is zero; FABBER puts (2*aif[0])/6 there.
        assert eq7[0, 0] == 0.0
        assert fabber[0, 0] == pytest.approx((2.0 * aif[0]) / 6.0 * dt)


class TestVolterraForwardAccuracy:
    """eq-(7) convolves to within ~1% of a fine-grid continuous convolution."""

    @pytest.mark.parametrize("mtt", [3.0, 6.0, 12.0])
    def test_forward_vs_fine_grid(self, mtt: float) -> None:
        dt, n, over = 1.5, 60, 20
        t = np.arange(n) * dt
        aif = np.zeros(n)
        m = t > 15.0
        aif[m] = ((t[m] - 15.0) ** 3.0) * np.exp(-(t[m] - 15.0) / 1.5)
        aif = aif / aif.max()
        residue = np.exp(-t / mtt)  # R(0) = 1
        disc = build_volterra_matrix(aif, dt) @ residue
        # fine-grid continuous convolution of the same residue
        fine_dt = dt / over
        tf = np.arange(n * over) * fine_dt
        aif_f = np.interp(tf, t, aif)
        res_f = np.exp(-tf / mtt)
        cont = (np.convolve(aif_f, res_f)[: n * over] * fine_dt)[::over]
        assert _rel_l2(disc, cont) < 0.02
