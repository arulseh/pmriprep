"""Can-fail test: the 2CXM bi-exponential impulse response must match a
direct ODE integration of the two-compartment exchange system.

This guards the residue amplitude coefficients (A1, A2). A prior bug omitted
the ``k_pe = PS/vp`` term, which preserves I(0) = Fp but collapses the slow
(extravascular) amplitude toward zero and breaks the Tofts limit; that bug
leaves this test failing by 1-2 orders of magnitude while a correct
implementation matches the ODE to ~1e-8.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import odeint

from pmriprep.models.dce.two_compartment_exchange import (
    TwoCompartmentExchangeModel,
)


def _ode_impulse_response(
    fp: float, ps: float, ve: float, vp: float, t: np.ndarray
) -> np.ndarray:
    """Tissue IRF C_t(t) from direct integration of the 2CXM ODEs.

    State [cp, ce] with dcp/dt = k_out*Ca - (k_out+k_pe)cp + k_pe*ce,
    dce/dt = k_ep*cp - k_ep*ce, driven by Ca = delta(t) so u(0+) = [k_out, 0].
    Measured tissue concentration is C_t = vp*cp + ve*ce.
    """
    fpr, psr = fp / 100.0, ps / 100.0
    k_pe, k_ep, k_out = psr / vp, psr / ve, fpr / vp
    a_mat = np.array([[-(k_out + k_pe), k_pe], [k_ep, -k_ep]])
    u0 = np.array([k_out, 0.0])
    sol = odeint(lambda u, _t: a_mat @ u, u0, t)
    return vp * sol[:, 0] + ve * sol[:, 1]


@pytest.mark.parametrize(
    "fp,ps,ve,vp",
    [
        (80.0, 20.0, 0.4, 0.10),   # moderate extraction
        (60.0, 2.0, 0.2, 0.05),    # low PS (leakage-limited)
        (40.0, 5.0, 0.3, 0.02),    # small vp
        (50.0, 10.0, 0.5, 0.15),   # larger volumes
    ],
)
def test_2cxm_irf_matches_ode(fp: float, ps: float, ve: float, vp: float) -> None:
    model = TwoCompartmentExchangeModel()
    t = np.linspace(0.0, 10.0, 4000)  # minutes, fine grid
    k1, k2, a1, a2 = model._compute_eigenvalues_and_coefficients(fp, ps, ve, vp)
    irf_model = a1 * np.exp(-k1 * t) + a2 * np.exp(-k2 * t)
    irf_true = _ode_impulse_response(fp, ps, ve, vp, t)
    scale = float(np.max(np.abs(irf_true)))
    max_err = float(np.max(np.abs(irf_model - irf_true)))
    assert max_err / scale < 1e-3, (
        f"2CXM IRF deviates from ODE integration by {max_err:.4e} "
        f"(relative {max_err / scale:.2e}); check A1/A2 coefficients"
    )
